import os
import sys
import random
import time
import logging
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from dis_vrnn_model import dis_VRNN

IMG_FOLDER = 'dis_vrnn_images/'
LOG_FILENAME = 'logs/dis_vrnn_{}.log'.format(time.strftime("%Y%m%d-%H%M%S"))
logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

x_dim = 784
h_dim = 100
f_dim = 20
z_dim = 3
n_epochs = 100
clip = 10
learning_rate = 1e-3
batch_size = 8
seed = 128
# print_every = 250
print_every = 1000
save_every = 10


def plot_sample(sample, epoch):

    with torch.no_grad():
        d_sample = model.sample(8)
        d_sample_imgs = d_sample.numpy().reshape(8, 28, 28)
        plt.figure(figsize=(8, 2))
        for i, img in enumerate(d_sample_imgs):
            plt.subplot(1, len(d_sample_imgs), i + 1)
            plt.axis('off')
            plt.imshow(img, cmap="gray")
        figname = 'd_fig_{}.png'.format(epoch)
        dest = os.path.join(IMG_FOLDER, figname)
        plt.savefig(dest)
        plt.show()
        plt.pause(2)
        plt.close()

        c_sample = model.content_sample(8)
        c_sample_imgs = c_sample.numpy().reshape(8, 28, 28)
        plt.figure(figsize=(8, 2))
        for i, img in enumerate(c_sample_imgs):
            plt.subplot(1, len(c_sample_imgs), i + 1)
            plt.axis('off')
            plt.imshow(img, cmap="gray")
        figname = 'c_fig_{}.png'.format(epoch)
        dest = os.path.join(IMG_FOLDER, figname)
        plt.savefig(dest)
        plt.show()
        plt.pause(2)
        plt.close()


def train_iter_log(epoch, batch_idx, data, f_kld_loss, z_kld_loss, nll_loss):
    str = 'Train Epoch: {} KLD Loss (f, z, f + z): ({:.3f},' \
        ' {:.3f}, {:.3f}) NLL Loss: {:.3f}'
    logging.info((str.format(epoch,
                             f_kld_loss.item() / batch_size,
                             z_kld_loss.item() / batch_size,
                             (f_kld_loss.item() + z_kld_loss.item()) / batch_size,
                             nll_loss.item() / batch_size)))


def min_max_norm(x, min, max):
    return (x - min) / (max - min)


def dataset(cls_dataset):
    return cls_dataset[random.randint(0, 9)]


def batch_gen(data, batch_size):
    i = 0
    while (i + 1) * batch_size <= len(data):
        iter_list = list(range(len(data)))
        random.shuffle(iter_list)
        ix = range(i * batch_size, (i + 1) * batch_size)
        i += 1
        yield torch.stack([data[j] for j in ix]).view(batch_size, -1)


def train(epoch, train_loader):
    train_loss = 0
    for batch_idx, (data, _) in enumerate(dem_train_loader):

        data = data.view(batch_size, -1)
        data = min_max_norm(data, data.min().item(), data.max().item())

        optimizer.zero_grad()
        f_kld_loss, z_kld_loss, nll_loss = model(data)
        loss = f_kld_loss + z_kld_loss + nll_loss
        loss.backward()
        optimizer.step()

        nn.utils.clip_grad_norm_(model.parameters(), clip)

        if batch_idx % print_every == 0:
            train_iter_log(epoch, batch_idx, data, f_kld_loss, z_kld_loss,
                           nll_loss)

        train_loss += loss.item()

    plot_sample(data, epoch)
    avg_loss = train_loss / train_dset_size
    logging.info('==> Epoch: {} Average loss: {:.4f}'.format(epoch,
                                                             avg_loss))


def test(epoch, test_loader):

    mean_f_kld_loss, mean_z_kld_loss, mean_nll_loss = 0, 0, 0
    for i, (data, _) in enumerate(dem_test_loader):

        data = data.squeeze().view(-1, x_dim)
        data = min_max_norm(data, data.min().item(), data.max().item())

        f_kld_loss, z_kld_loss, nll_loss = model(data)
        mean_f_kld_loss += f_kld_loss.item()
        mean_z_kld_loss += z_kld_loss.item()
        mean_nll_loss += nll_loss.item()

    mean_f_kld_loss /= test_dset_size
    mean_z_kld_loss /= test_dset_size
    mean_nll_loss /= test_dset_size

    logging.info('==> Test loss: KLD Loss (f, z, f+z) = ({:.3f}, {:.3f}, {:.3f}),'
                 ' NLL Loss = {:.3f} '.format(mean_f_kld_loss,
                                              mean_z_kld_loss,
                                              mean_z_kld_loss + mean_f_kld_loss,
                                              mean_nll_loss))


if __name__ == "__main__":

    torch.manual_seed(seed)
    plt.ion()

    train_dataset = datasets.MNIST('data', train=True, download=True,
                                   transform=transforms.ToTensor())
    test_dataset = datasets.MNIST('data', train=False,
                                  transform=transforms.ToTensor())

    dem_train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size, shuffle=True,
                                  drop_last=True)

    dem_test_loader = DataLoader(test_dataset,
                                 batch_size=batch_size, shuffle=True,
                                 drop_last=True)

    train_cls_data = [[] for i in range(10)]
    test_cls_data = [[] for i in range(10)]

    train_data = train_dataset.data.float()
    test_data = test_dataset.data.float()

    for i in range(len(train_data)):
        train_cls_data[train_dataset.targets[i]].append(train_data[i])

    for i in range(len(test_data)):
        test_cls_data[test_dataset.targets[i]].append(test_data.data[i])

    train_dset_size = len(train_dataset)
    test_dset_size = len(test_dataset)

    seq_len = batch_size
    model = dis_VRNN(seq_len, x_dim, f_dim, z_dim, h_dim)
    # model.load_state_dict(torch.load('saves/dis_vrnn_state_dict_21.pth'))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    total_time = 0
    epoch_start_time = time.time()
    for epoch in range(0, n_epochs + 1):

        train_loader = batch_gen(train_cls_data[random.randint(0, 9)], batch_size)
        test_loader = batch_gen(test_cls_data[random.randint(0, 9)], batch_size)

        train(epoch, train_loader)
        test(epoch, test_loader)

        if epoch % save_every == 1:
            fn = 'saves/dis_vrnn_state_dict_' + str(epoch) + '.pth'
            torch.save(model.state_dict(), fn)
            logging.info('Saved model to ' + fn)

        curr_time = time.time()
        epoch_time = (curr_time - epoch_start_time) / 60
        epoch_start_time = curr_time
        total_time += epoch_time
        remaining_epochs = n_epochs - epoch
        logging.info('Time for current epoch - {:.2f} mins'.format(epoch_time))
        logging.info('Est. remaining training time for {} epochs - {:.2f} mins'.format(
                     remaining_epochs, remaining_epochs * epoch_time))
        logging.info('*****' * 15)
    logging.info('Training done for {} epochs in {} time'.format(n_epochs,
                                                                 total_time))
