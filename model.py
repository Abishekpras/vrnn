import torch
import torch.nn as nn
from torch.nn import functional as F


class VRNN(nn.Module):
  def __init__(self, seq_len, x_dim, h_dim, z_dim):
    super(VRNN, self).__init__()

    self.x_dim = x_dim
    self.h_dim = h_dim
    self.z_dim = z_dim
    self.seq_len = seq_len

    self.phi_x = nn.Sequential(nn.Linear(x_dim, h_dim),
                               nn.ReLU(),
                               nn.Linear(h_dim, h_dim),
                               nn.ReLU())

    self.phi_z = nn.Sequential(nn.Linear(z_dim, h_dim),
                               nn.ReLU())

    self.prior = nn.Sequential(nn.Linear(h_dim, h_dim),
                               nn.ReLU())
    self.prior_mean = nn.Linear(h_dim, z_dim)
    self.prior_std = nn.Sequential(nn.Linear(h_dim, z_dim),
                                   nn.Softplus())

    self.enc = nn.Sequential(nn.Linear(h_dim + h_dim, h_dim),
                             nn.ReLU(),
                             nn.Linear(h_dim, h_dim),
                             nn.ReLU())
    self.enc_mean = nn.Linear(h_dim, z_dim)
    self.enc_std = nn.Sequential(nn.Linear(h_dim, z_dim),
                                 nn.Softplus())

    self.dec = nn.Sequential(nn.Linear(h_dim + h_dim, h_dim),
                             nn.ReLU(),
                             nn.Linear(h_dim, h_dim),
                             nn.ReLU())
    self.dec_mean = nn.Sequential(nn.Linear(h_dim, x_dim),
                                  nn.Sigmoid())
    self.dec_std = nn.Sequential(nn.Linear(h_dim, x_dim),
                                 nn.Softplus())

    self.rnn = nn.GRUCell(h_dim + h_dim, h_dim)

  def forward(self, x):

    kld_loss = 0
    nll_loss = 0
    h = torch.zeros(x.size(1), self.h_dim)

    for t in range(self.seq_len):

      phi_x_t = self.phi_x(x[t:t + 1].squeeze())

      prior_t = self.prior(h)
      prior_mean_t = self.prior_mean(prior_t)
      prior_std_t = self.prior_std(prior_t)

      enc_t = self.enc(torch.cat([phi_x_t, h], 1))
      enc_mean_t = self.enc_mean(enc_t)
      enc_std_t = self.enc_std(enc_t)

      z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
      phi_z_t = self.phi_z(z_t)

      dec_t = self.dec(torch.cat([phi_z_t, h], 1))
      dec_mean_t = self.dec_mean(dec_t)

      h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1), h)

      kld_loss += self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
      nll_loss += self._nll_bernoulli(dec_mean_t, x[t].squeeze())

    return kld_loss, nll_loss

  def sample(self, seq_len):

    sample = torch.zeros(seq_len, self.x_dim)

    h = torch.zeros(1, self.h_dim)
    for t in range(seq_len):

      prior_t = self.prior(h)
      prior_mean_t = self.prior_mean(prior_t)
      prior_std_t = self.prior_std(prior_t)

      z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)
      phi_z_t = self.phi_z(z_t)

      dec_t = self.dec(torch.cat([phi_z_t, h], 1))
      dec_mean_t = self.dec_mean(dec_t)

      phi_x_t = self.phi_x(dec_mean_t)

      h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1), h)

      sample[t] = dec_mean_t.data

    return sample

  def _reparameterized_sample(self, mean, std):
    eps = torch.FloatTensor(std.size()).normal_()
    return eps.mul(std).add_(mean)

  def _kld_gauss(self, mu_1, sd_1, mu_2, sd_2):

    sd_ratio = sd_1 / sd_2
    kl_unit = ((mu_1 - mu_2) / sd_2).pow(2)
    kl_unit += sd_ratio.pow(2) - 2 * torch.log(sd_ratio) - 1
    return 0.5 * torch.sum(kl_unit)

  def _nll_bernoulli(self, recon_x, x):
    return F.binary_cross_entropy(recon_x.view(-1, 784),
                                  x.view(-1, 784),
                                  reduction='sum')
