import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Cont(nn.Module):
    def __init__(self, feature_dim, label_dim, z_dim, latent_dim=64, emb_size=2048, keep_prob=0.5, reg='gmvae', scale_coeff=1.0):
        super(Cont, self).__init__()
        self.input_dim = feature_dim
        self.label_dim = label_dim
        self.latent_dim = latent_dim
        self.emb_size = emb_size
        self.z_dim = z_dim
        self.reg = reg
        self.keep_prob = keep_prob
        self.scale_coeff = scale_coeff # 1.0
        self.fx1 = nn.Linear(self.input_dim, 256)
        self.fx2 = nn.Linear(256, 512)
        self.fx3 = nn.Linear(512, 256)
        self.fx_mu = nn.Linear(256, self.latent_dim)
        self.fx_logvar = nn.Linear(256, self.latent_dim)

        self.emb_size = self.emb_size

        self.fd_x1 = nn.Linear(self.input_dim+self.latent_dim, 512)

        self.fd_x2 = torch.nn.Sequential(
            nn.Linear(512, self.emb_size) # 512 -> 2048
        )
        self.feat_mp_mu = nn.Linear(self.emb_size, self.label_dim) # 2048 -> 38

        # label layers
        self.fe0 = nn.Linear(self.label_dim, self.emb_size) # 38 -> 2048
        self.fe1 = nn.Linear(self.emb_size, 512) # 2048 -> 512
        self.fe2 = nn.Linear(512, 256)
        self.fe_mu = nn.Linear(256, self.latent_dim) # 256 -> 64
        self.fe_logvar = nn.Linear(256, self.latent_dim) # 256 -> 64

        self.fd1 = self.fd_x1
        self.fd2 = self.fd_x2
        self.label_mp_mu = self.feat_mp_mu

        self.bias = nn.Parameter(torch.zeros(self.label_dim))

        assert id(self.fd_x1) == id(self.fd1)
        assert id(self.fd_x2) == id(self.fd2)

        # things they share
        self.dropout = nn.Dropout(p=self.keep_prob)

    def feat_decode(self, z):
        d1 = F.relu(self.fd_x1(z)) # [B, 512]
        d2 = F.leaky_relu(self.fd_x2(d1)) # [B, 2048]
        d3 = F.normalize(d2, dim=1) # [B, 2048]
        return d3

    def label_forward(self, x, feat):
        # if self.reg == "gmvae":
        n_label = x.shape[1] # [B, 38] = 38
        all_labels = torch.eye(n_label).to(x.device) # [38, 38]
        fe_output = self.label_encode(all_labels)
        mu = fe_output['fe_mu'] # [38, 64]
        logvar = fe_output['fe_logvar'] # [38, 64]
        
        if self.reg == "wae" or not self.training:
            if self.reg == "gmvae":
                z = torch.matmul(x, mu) / (x.sum(1, keepdim=True) + 1e-7)
            else:
                z = mu
        else:
            if self.reg == "gmvae":
                z = torch.matmul(x, mu) / (x.sum(1, keepdim=True) + 1e-7) # [B, 64]
            else:
                z = self.label_reparameterize(mu, logvar)
        label_emb = self.label_decode(torch.cat((feat, z), 1)) # [B, 2048]
        single_label_emb = F.normalize(self.label_recon(mu), dim=1) # [38, 2048]

        fe_output['label_emb'] = label_emb
        fe_output['single_label_emb'] = single_label_emb
        return fe_output

    def feat_forward(self, x):
        fx_output = self.feat_encode(x)
        mu = fx_output['fx_mu']
        logvar = fx_output['fx_logvar']

        z = self.feat_reparameterize(mu, logvar)
        z2 = self.feat_reparameterize(mu, logvar)
        feat_emb = self.feat_decode(torch.cat((x, z), 1)) # [B, 2048]
        feat_emb2 = self.feat_decode(torch.cat((x, z2), 1)) # [B, 2048]
        fx_output['feat_emb'] = feat_emb
        fx_output['feat_emb2'] = feat_emb2

        feat_recon = self.recon(z)
        fx_output['feat_recon'] = feat_recon
        return fx_output
