import sys
import os
import time
import logging
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, BatchSampler, RandomSampler
from torchvision import datasets, transforms

from model import VRNN

IMG_FOLDER = 'images/'
LOG_FILENAME = 'logs/vrnn_{}.log'.format(time.strftime("%Y%m%d-%H%M%S"))
logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

x_dim = 784
h_dim = 100
z_dim = 16
n_epochs = 100
clip = 10
learning_rate = 1e-3
batch_size = 128
seq_len = 8
seed = 128
print_every = 60000 // (4 * batch_size * seq_len)
save_every = 10


def plot_sample(model, epoch):
    sample = model.sample(8)
    imgs = sample.numpy().reshape(8, 28, 28)
    plt.figure(figsize=(8, 2))
    for i, img in enumerate(imgs):
        plt.subplot(1, len(imgs), i + 1)
        plt.axis('off')
        plt.imshow(img, cmap="gray")
    figname = 'fig_{}.png'.format(epoch)
    dest = os.path.join(IMG_FOLDER, figname)
    plt.savefig(dest)
    plt.show()
    plt.pause(2)
    plt.close()


def train_iter_log(epoch, batch_idx, data, kld_loss, nll_loss):
    str = 'Train Epoch: {} [{}/{} ({:.0f}%)] KLD Loss: {:.3f} NLL Loss: {:.3f}'
    logging.info((str.format(epoch, batch_idx * len(data),
                             len(train_loader.dataset),
                             100. * batch_idx / len(train_loader),
                             kld_loss.item() / batch_size,
                             nll_loss.item() / batch_size)))


def min_max_norm(x, min, max):
    return (x - min) / (max - min)


def mini_batchify(mini_batch_size, data):

    num_m_batch = len(data) // mini_batch_size
    mini_batched_data = [] * num_m_batch
    for i in range(num_m_batch):
        mini_batched_data.append(data[i * mini_batch_size:(i + 1) * mini_batch_size])

    return torch.stack(mini_batched_data)


def train(epoch):
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):

        data = mini_batchify(seq_len, data)
        data = data.view(batch_size, seq_len, x_dim)
        data = min_max_norm(data, data.min().item(), data.max().item())

        optimizer.zero_grad()
        kld_loss, nll_loss = model(data)
        loss = kld_loss + nll_loss
        loss.backward()
        optimizer.step()

        nn.utils.clip_grad_norm_(model.parameters(), clip)

        if batch_idx % print_every == 0:
            train_iter_log(epoch, batch_idx, data, kld_loss, nll_loss)

        train_loss += loss.item()

    plot_sample(model, epoch)
    avg_loss = train_loss / len(train_loader.dataset)
    logging.info('==> Epoch: {} Average loss: {:.4f}'.format(epoch,
                                                             avg_loss))


def test(epoch):

    mean_kld_loss, mean_nll_loss = 0, 0
    for i, (data, _) in enumerate(test_loader):

        data = mini_batchify(seq_len, data)
        data = data.squeeze().view(batch_size, seq_len, x_dim)
        data = min_max_norm(data, data.min().item(), data.max().item())

        kld_loss, nll_loss = model(data)
        mean_kld_loss += kld_loss.item()
        mean_nll_loss += nll_loss.item()

    mean_kld_loss /= len(test_loader.dataset)
    mean_nll_loss /= len(test_loader.dataset)

    logging.info('==> Test loss: KLD Loss = {:.3f}, NLL Loss = {:.3f} '.format(
        mean_kld_loss, mean_nll_loss))


if __name__ == "__main__":

    torch.manual_seed(seed)
    plt.ion()

    train_dataset = datasets.MNIST('data', train=True, download=True,
                                   transform=transforms.ToTensor())
    test_dataset = datasets.MNIST('data', train=False, download=True,
                                  transform=transforms.ToTensor())

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size * seq_len, shuffle=True,
                              drop_last=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size * seq_len, shuffle=True,
                             drop_last=True)

    model = VRNN(seq_len, x_dim, h_dim, z_dim)
    # model.load_state_dict(torch.load('saves/vrnn_state_dict_21.pth'))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    total_time = 0
    epoch_start_time = time.time()
    for epoch in range(1, n_epochs + 1):

        train(epoch)
        test(epoch)

        if epoch % save_every == 1:
            fn = 'saves/vrnn_state_dict_' + str(epoch) + '.pth'
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
