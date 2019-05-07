import torch
import torch.nn as nn
from torch.nn import functional as F


class dis_VRNN(nn.Module):
    def __init__(self, seq_len, x_dim, f_dim, z_dim, h_dim, with_attn=False):
        super(dis_VRNN, self).__init__()

        self.x_dim = x_dim
        self.f_dim = f_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.seq_len = seq_len
        self.with_attn = with_attn

        self.phi_x = nn.Sequential(nn.Linear(x_dim, h_dim),
                                   nn.ReLU(),
                                   nn.Linear(h_dim, h_dim),
                                   nn.ReLU())

        self.phi_z = nn.Sequential(nn.Linear(z_dim, h_dim),
                                   nn.ReLU())

        self.phi_f = nn.Sequential(nn.Linear(f_dim, h_dim),
                                   nn.ReLU())

        self.rnn = nn.GRUCell(h_dim + h_dim + h_dim, h_dim)

        if with_attn:
            self.rnn_attn_weights = nn.Linear(h_dim + h_dim, seq_len)

            self.rnn_attended_inputs = nn.Sequential(nn.Linear(h_dim, h_dim + h_dim),
                                                     nn.ReLU())

        # Prior Networks

        self.z_prior = nn.Sequential(nn.Linear(h_dim, h_dim),
                                     nn.ReLU())
        self.z_prior_mean = nn.Linear(h_dim, z_dim)
        self.z_prior_std = nn.Sequential(nn.Linear(h_dim, z_dim),
                                         nn.Softplus())

        # Encoder Networks

        self.f_enc_rnn = nn.GRU(h_dim, h_dim)
        self.f_enc_mean = nn.Linear(h_dim, f_dim)
        self.f_enc_std = nn.Sequential(nn.Linear(h_dim, f_dim),
                                       nn.Softplus())

        self.z_enc = nn.Sequential(nn.Linear(h_dim + h_dim, h_dim),
                                   nn.ReLU(),
                                   nn.Linear(h_dim, h_dim),
                                   nn.ReLU())
        self.z_enc_mean = nn.Linear(h_dim, z_dim)
        self.z_enc_std = nn.Sequential(nn.Linear(h_dim, z_dim),
                                       nn.Softplus())

        # Decoder Networks

        self.dec = nn.Sequential(nn.Linear(h_dim + h_dim + h_dim, h_dim),
                                 nn.ReLU(),
                                 nn.Linear(h_dim, h_dim),
                                 nn.ReLU())
        self.dec_mean = nn.Sequential(nn.Linear(h_dim, x_dim),
                                      nn.Sigmoid())
        self.dec_std = nn.Sequential(nn.Linear(h_dim, x_dim),
                                     nn.Softplus())

    def forward(self, x):

        z_kld_loss = 0
        nll_loss = 0
        h = torch.zeros(x.size(1), self.h_dim)

        phi_x = self.phi_x(x).view(self.seq_len, x.size(1),
                                   self.h_dim)

        f_enc_out, f_enc_hidden = self.f_enc_rnn(phi_x)
        f_enc_mean = self.f_enc_mean(f_enc_hidden)
        f_enc_std = self.f_enc_std(f_enc_hidden)

        f = self._reparameterized_sample(f_enc_mean, f_enc_std).squeeze()
        phi_f = self.phi_f(f)

        f_kld_loss = self._kld_std_gauss(f_enc_mean, f_enc_std)

        for t in range(x.size(0)):

            phi_x_t = phi_x[t]

            z_prior_t = self.z_prior(h)
            z_prior_mean_t = self.z_prior_mean(z_prior_t)
            z_prior_std_t = self.z_prior_std(z_prior_t)

            z_enc_t = self.z_enc(torch.cat([phi_x_t, h], 1))
            z_enc_mean_t = self.z_enc_mean(z_enc_t)
            z_enc_std_t = self.z_enc_std(z_enc_t)

            z_t = self._reparameterized_sample(z_enc_mean_t, z_enc_std_t)
            phi_z_t = self.phi_z(z_t)

            dec_t = self.dec(torch.cat([phi_z_t, phi_f, h], 1))
            dec_mean_t = self.dec_mean(dec_t)

            if self.with_attn is True:

                attn_wts_t = self.rnn_attn_weights(torch.cat([phi_x_t, h], 1))
                attn_wts_t = F.softmax(attn_wts_t, dim=1)
                print(phi_x_t.shape, h.shape, attn_wts_t.shape, f_enc_out.shape)
                h_attn = torch.bmm(attn_wts_t, f_enc_out)

                phi_attn_ip_t = self.rnn_attended_inputs(torch.cat([phi_x_t, h_attn], 1))

                rnn_inp = torch.cat([phi_attn_ip_t, phi_z_t], 1)

            else:

                rnn_inp = torch.cat([phi_x_t, phi_z_t, phi_f], 1)

            h = self.rnn(rnn_inp, h)

            z_kld_loss += self._kld_gauss(z_enc_mean_t, z_enc_std_t,
                                          z_prior_mean_t, z_prior_std_t)
            nll_loss += self._nll_bernoulli(dec_mean_t, x[t])

        return f_kld_loss, z_kld_loss, nll_loss

    def sample(self, seq_len):

        sample = torch.zeros(seq_len, self.x_dim)

        h = torch.zeros(1, self.h_dim)

        f_prior = torch.randn(1, self.f_dim)
        phi_f = self.phi_f(f_prior)

        for t in range(seq_len):

            z_prior_t = self.z_prior(h)
            z_prior_mean_t = self.z_prior_mean(z_prior_t)
            z_prior_std_t = self.z_prior_std(z_prior_t)

            z_t = self._reparameterized_sample(z_prior_mean_t, z_prior_std_t)
            phi_z_t = self.phi_z(z_t)

            dec_t = self.dec(torch.cat([phi_z_t, phi_f, h], 1))
            dec_mean_t = self.dec_mean(dec_t)

            phi_x_t = self.phi_x(dec_mean_t)

            h = self.rnn(torch.cat([phi_x_t, phi_z_t, phi_f], 1), h)

            sample[t] = dec_mean_t.data

        return sample

    def content_sample(self, seq_len):

        sample = torch.zeros(seq_len, self.x_dim)

        h = torch.zeros(1, self.h_dim)

        z_prior_t = self.z_prior(h)
        z_prior_mean_t = self.z_prior_mean(z_prior_t)
        z_prior_std_t = self.z_prior_std(z_prior_t)

        z_t = self._reparameterized_sample(z_prior_mean_t, z_prior_std_t)
        phi_z_t = self.phi_z(z_t)

        for t in range(seq_len):

            phi_f = self.phi_f(torch.randn(1, self.f_dim))

            dec_t = self.dec(torch.cat([phi_z_t, phi_f, h], 1))
            dec_mean_t = self.dec_mean(dec_t)

            phi_x_t = self.phi_x(dec_mean_t)

            h = self.rnn(torch.cat([phi_x_t, phi_z_t, phi_f], 1), h)

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

    def _kld_std_gauss(self, mu, sd):
        return self._kld_gauss(mu, sd, torch.zeros_like(mu), torch.ones_like(sd))

    def _nll_bernoulli(self, recon_x, x):
        return F.binary_cross_entropy(recon_x.view(-1, 784),
                                      x.view(-1, 784),
                                      reduction='sum')
