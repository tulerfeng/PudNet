import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class HyperNetwork(nn.Module):

    def __init__(self, z_dim, f_size=3, out_size=64, in_size=64, hid=64, device="cuda:2"):
        super(HyperNetwork, self).__init__()
        self.z_dim = z_dim  
        self.out_size = out_size
        self.in_size = in_size
        self.f_size = f_size
        self.device = device
        self.hid = hid  

        self.fc = nn.Sequential(nn.Linear(self.z_dim,self.hid*self.f_size*self.f_size), nn.GELU())  
        self.conv1 = nn.Sequential(nn.Conv2d(self.hid, self.hid*2, 1, stride=1, padding=0), nn.GELU())  
        self.conv2 = nn.Conv2d(self.hid*2, self.out_size*self.in_size, 1, stride=1, padding=0)
    

    def forward(self, skt):  
        skt_in = skt
        fea_tmp = self.fc(skt_in).view(-1, self.hid, self.f_size, self.f_size)     

        ker_tmp = self.conv1(fea_tmp)  
        ker_tmp = self.conv2(ker_tmp)  
        kernel = ker_tmp.view(self.out_size, self.in_size, self.f_size, self.f_size)
        return kernel  




