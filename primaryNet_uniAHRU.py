import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import os
import scipy.io as sio
from hypernetwork_modules import HyperNetwork
import numpy as np
from layer import *
import datetime

class uniAHRU(nn.Module):
    def __init__(self, dec_hid_dim, output_dim, gru_layers, gru_drop, dec_drop):
        super().__init__()
        self.rnn = nn.GRU(output_dim, dec_hid_dim, num_layers=gru_layers, dropout=gru_drop) 
        self.fc_out = nn.Sequential(nn.Linear(output_dim, output_dim),nn.ReLU(True)) 
        self.dec_drop = dec_drop
        if self.dec_drop != 0:
            self.dropout = nn.Dropout(dec_drop)
        self.ln_hid = nn.LayerNorm(dec_hid_dim)

    def forward(self, w_input, s):
        embedded = w_input.unsqueeze(0)  
        if self.dec_drop != 0:
            embedded = self.dropout(embedded)
        rnn_input = embedded  
        dec_output, dec_hidden = self.rnn(rnn_input, s.unsqueeze(1))  
        embedded = embedded.squeeze(0)
        dec_output = dec_output.squeeze(0)
        dec_output = self.fc_out(dec_output)
        dec_hidden = self.ln_hid(dec_hidden.squeeze(1))

        return dec_output, dec_hidden  

class ConvDecoder(nn.Module):

    def __init__(self):
        super(ConvDecoder, self).__init__()

    def forward(self, hyper_net, sketch):  
        kernel = hyper_net(sketch)
        return kernel


class PrimaryNetwork(nn.Module):

    def __init__(self, args,device="cuda:2",temp=10., temp_learnable=True):  
        super(PrimaryNetwork, self).__init__()
        self.skt_a = args.skt_a
        self.dec_hid_dim = args.hid_dim  
        self.output_dim = args.out_dim   
        self.device = device
        self.nch_ker = args.nch_ker
        self.nch_in = args.nch_in
        self.nch_out = args.nch_out

        if temp_learnable:
            self.temp = nn.Parameter(torch.tensor(temp))
        else:
            self.temp = temp

        self.denoise = nn.Sequential(nn.Conv2d(3, 16, 5, stride=2, padding=2),  nn.GELU(),nn.Conv2d(16, 32, 5, stride=2, padding=2),  nn.GELU(),nn.Conv2d(32, 32, 5, stride=2, padding=2), nn.GELU()) 
        
        self.hid_times = args.gru_layers
        if args.nx_in == 80:
            lin_in = 3200
        elif args.nx_in == 128:
            lin_in = 8192
        self.denotmp = nn.Sequential(nn.Linear(lin_in,args.hid_dim*self.hid_times), nn.BatchNorm1d(args.hid_dim*self.hid_times), nn.GELU())  

        self.filter_size_enc = [[1 * self.nch_in,  1 * self.nch_ker],[1 * self.nch_ker, 1 * self.nch_ker],[1 * self.nch_ker, 2 * self.nch_ker],[2 * self.nch_ker, 2 * self.nch_ker],[2 * self.nch_ker, 4 * self.nch_ker],[4 * self.nch_ker, 4 * self.nch_ker],[4 * self.nch_ker, 8 * self.nch_ker],[8 * self.nch_ker, 8 * self.nch_ker],[8 * self.nch_ker, 2 * 8 * self.nch_ker]]  
        self.filter_size_dec = [[2 * 8 * self.nch_ker, 8 * self.nch_ker],[2 * 8 * self.nch_ker, 8 * self.nch_ker],[8 * self.nch_ker,     4 * self.nch_ker],[2 * 4 * self.nch_ker, 4 * self.nch_ker],[4 * self.nch_ker,   2 * self.nch_ker],[2 * 2 * self.nch_ker, 2 * self.nch_ker],[2 * self.nch_ker,1 * self.nch_ker],[2 * 1 * self.nch_ker, 1 * self.nch_ker],[1 * self.nch_ker,1 * self.nch_out]]
        
        self.ahru = uniAHRU(dec_hid_dim=self.dec_hid_dim,output_dim=self.output_dim, gru_layers=args.gru_layers,gru_drop=args.gru_drop,dec_drop=args.dec_drop)
        
        self.dec_in_for0 = torch.randn(self.dec_hid_dim, requires_grad=True).unsqueeze(0).to(self.device)
        
        self.dec_in_re0 = torch.randn(self.dec_hid_dim, requires_grad=True).unsqueeze(0).to(self.device)
       
        self.unet_enc = nn.ModuleList()
        self.pool_enc = nn.ModuleList()
        self.unet_dec = nn.ModuleList()
        self.pool_dec = nn.ModuleList()
        self.hyper_unet = nn.ModuleList()  

        for i in range(9):
            self.unet_enc.append(CNR2d(nch_in=self.filter_size_enc[i][0], nch_out=self.filter_size_enc[i][1], kernel_size=3))
            if i in (1,3,5,7): 
                self.pool_enc.append(Pooling2d(pool=2, type='avg'))

            self.hyper_unet.append(
                HyperNetwork(z_dim=self.output_dim, out_size=self.filter_size_enc[i][1], in_size=self.filter_size_enc[i][0], hid= args.hid_hyper, device=self.device))  

        for i in range(9):
            self.unet_dec.append(DECNR2d(nch_in=self.filter_size_dec[i][0], nch_out=self.filter_size_dec[i][1], kernel_size=3))
            if i in (0,2,4,6): 
                self.pool_dec.append(UnPooling2d(pool=2, type='nearest'))

            self.hyper_unet.append(
                HyperNetwork(z_dim=self.output_dim, out_size=self.filter_size_dec[i][0],in_size=self.filter_size_dec[i][1], hid=args.hid_hyper, device=self.device)) 
                                
        self.lns = nn.ModuleList()

        for i in range(18):
            self.lns.append(nn.LayerNorm(self.output_dim))

    def forward(self, x, sketch): 
        sketch = self.denoise(sketch)  
        sketch = sketch.view(sketch.size(0), -1)  
        skt = self.denotmp(sketch)                
        skt = torch.mean(skt,dim=0).unsqueeze(0)  
        skt_hi = skt.view(self.hid_times,int(skt.size(1)/self.hid_times)) 
        
        dec_in_for = self.dec_in_for0
        dec_hi_for = skt_hi
        dec_in_re = self.dec_in_re0
        dec_hi_re = skt_hi
        k = 0
        dec_outlist_for = []  
        
        dec_hilist_re = []
        for i in range(18):
            dec_out_for, dec_hi_for = self.ahru(dec_in_for, dec_hi_for)
            dec_outlist_for.append(dec_out_for)
            dec_in_for = dec_out_for
            
        count_enc = 0
        count_dec = 0
        fea_enc = []
        for layer_id in range(18):
            dec_output = dec_outlist_for[layer_id]
            skt_tmp = skt_hi
            skt_tmp = skt_tmp.mean(0).unsqueeze(0)
            dec_output = dec_output * (1 - self.skt_a) + skt_tmp * self.skt_a
            dec_output = self.lns[layer_id](dec_output)

            w_pred = self.hyper_unet[layer_id](dec_output)  

            if layer_id < 9:
                x = self.unet_enc[layer_id](x, w_pred)
                if layer_id in (1, 3, 5, 7):
                    fea_enc.append(x)
                    x = self.pool_enc[count_enc](x)
                    count_enc += 1
                
            else:
                l_id = layer_id - 9
                x = self.unet_dec[l_id](x, w_pred)
                if l_id in (0, 2, 4, 6):
                    x = self.pool_dec[count_dec](x)
                    x = torch.cat([fea_enc[3-count_dec], x], dim=1)
                    count_dec += 1
                
        out_x = x
        return out_x


    def _normalize(self, p, no_relu=True, is_w=False):
        
        if p.dim() > 2:  

            if no_relu:
                beta = 1.
            else:
                beta = 2.

            # fan-out:
            # p = p * (beta / (sz[0] * p[0, 0].numel())) ** 0.5

            # fan-in:
            p = p * (beta / p[0].numel()) ** 0.5  # p:n*c*w*h   p[0].numel()= c*w*h

        else:

            if is_w:  
                p = 2 * torch.sigmoid(0.5 * p)  
            else:
                p = torch.tanh(0.2 * p)         

        return p


    def classify(self, query_feature, support_feature):
        
        q_nor=F.normalize(query_feature,dim=1)
        s_nor=F.normalize(support_feature,dim=1)
        similarity=torch.matmul(q_nor, s_nor.detach().t())
        return similarity*self.temp, self.temp










