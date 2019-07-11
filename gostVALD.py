#codeing=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torch.autograd import Variable
'''
针对人脸问题，针对同一人多张人脸照片问题，多张人脸特征后进行特征融合，修改VLAD，将FC层替换掉卷积层

'''
class netVLAD(nn.Module):
    '''
    参数量:8*128*
    '''
    def __init__(self,num_clusters=8,dim=128,normalize_input=True):
        super(netVLAD, self).__init__()
        self.num_clusters=num_clusters
        self.dim=dim
        self.normalize_input=normalize_input
        self.fc=nn.Linear(dim,num_clusters)
        self.centroids=nn.Parameter(torch.rand(num_clusters,dim))
        self._init_params()
    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight.data)  
        nn.init.constant_(self.fc.bias.data, 0.0) 
        #self.alpha=100.
        #self.fc.weight = nn.Parameter(
        #    (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        #)
        #self.fc.bias = nn.Parameter(
        #    - self.alpha * self.centroids.norm(dim=1)
        #)
    def forward(self,x):
        '''
        x:(10,128)
        '''
        N,C=x.shape[:2]#10,128
        assert C==self.dim ,"feature dim not correct"
        if self.normalize_input:
            x=F.normalize(x,p=2,dim=0)
        soft_assign=self.fc(x).unsqueeze(0).permute(0,2,1)#(10,8)->(1,10,8)->(1,8,10)
        soft_assign=F.softmax(soft_assign,dim=1) #nn.Softmax(dim=1)
        x_flatten=x.view(1,C,-1)
        #print(x_flatten.shape)
        #print(x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3).shape)
        #print(self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0).shape)
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
            self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        residual *= soft_assign.unsqueeze(2)                
        vlad = residual.sum(dim=-1)#(1,8,128)
        vlad = F.normalize(vlad, p=2, dim=2)
        vlad = vlad.view(1, -1)
        vlad = F.normalize(vlad, p=2, dim=1) #(1,8*128)
        return vlad

class netVLAD2(nn.Module):
    '''
    参数量:8*128*
    '''
    def __init__(self,num_clusters=8,dim=128,normalize_input=True):
        super(netVLAD2, self).__init__()
        self.num_clusters=num_clusters
        self.dim=dim
        self.normalize_input=normalize_input
        self.fc=nn.Linear(dim,num_clusters)
        self.batch_norm = nn.BatchNorm1d(num_clusters, eps=1e-3, momentum=0.01)
        self.softmax = nn.Softmax(dim=1)
        self.centroids=nn.Parameter(torch.rand(num_clusters,dim))
        self._init_params()
    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight.data)  
        nn.init.constant_(self.fc.bias.data, 0.0)
    def forward(self,x):
        N,C=x.shape[:2]
        if self.normalize_input:
            x=F.normalize(x,p=2,dim=1)
        soft_assign=self.fc(x)
        soft_assign=self.softmax(soft_assign).unsqueeze(0)#(1,10,8)
        a_sum = soft_assign.sum(-2).unsqueeze(1)#(1,1,8)
        a = torch.mul(a_sum, self.centroids.transpose(1,0).unsqueeze(0))#(1,128,8)
        print(soft_assign.size(),a_sum.size(),a.size())
        soft_assign = soft_assign.permute(0, 2, 1).contiguous()
        x=x.view([-1, N, self.dim])
        vlad = torch.matmul(soft_assign, x).permute(0, 2, 1).contiguous() 
        vlad = vlad.sub(a).view([-1, self.num_clusters * self.dim])
        vlad = F.normalize(vlad, p=2, dim=1)
        return vlad
    def forward2(self,x):
        '''
        x:(10,128)
        '''
        N,C=x.shape[:2]#10,128
        assert C==self.dim ,"feature dim not correct"
        if self.normalize_input:
            x=F.normalize(x,p=2,dim=1)
        soft_assign=self.fc(x).unsqueeze(0).permute(0,2,1)#(10,8)->(1,10,8)->(1,8,10)
        soft_assign=F.softmax(soft_assign,dim=1) #nn.Softmax(dim=1) #(1,8,10)
        x_flatten=x.unsqueeze(0).permute(0,2,1)#(1,128,10)
        #print(x_flatten.shape)
        #print(x_flatten.expand(self.num_clusters, -1, -1, -1).shape)#(8,1,128,40)
        #print(self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0).shape)
        #[(1,128,10)->(8,1,128,10)->(1,8,128,10)]-[(8,128)->(10,8,128)->(8,128,10)->(1,8,128,10)]
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
            self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        #print(residual.size())#(1,8,128,10)
        residual *= soft_assign.unsqueeze(2)  #(1,8,128,10)*(1,8,1,10)->(1,8,128,10)              
        vlad = residual.sum(dim=-1)#(1,8,128)
        vlad = F.normalize(vlad, p=2, dim=2)
        vlad = vlad.view(1, -1)
        vlad = F.normalize(vlad, p=2, dim=1) #(1,8*128)
        return vlad
class gostVLAD(nn.Module):
    def __init__(self,num_clusters=8,gost=1,dim=128,normalize_input=True):
        super(gostVLAD, self).__init__()
        self.num_clusters=num_clusters
        self.dim=dim
        self.gost=gost
        self.normalize_input=normalize_input
        self.fc=nn.Linear(dim,num_clusters+gost)
        self.centroids=nn.Parameter(torch.rand(num_clusters,dim))
        self._init_params()
    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight.data)  
        nn.init.constant_(self.fc.bias.data, 0.0) 
    def forward(self,x):
        '''
        x:NxD 
        '''
        N,C=x.shape[:2]#10,128
        assert C==self.dim ,"feature dim not correct"
        if self.normalize_input:
            x=F.normalize(x,p=2,dim=0)
        soft_assign=self.fc(x).unsqueeze(0).permute(0,2,1)#(10,9)->(1,10,9)->(1,9,10)
        soft_assign=F.softmax(soft_assign,dim=1) 
 
        soft_assign=soft_assign[:,:self.num_clusters,:]#(1,8,10)

        x_flatten=x.view(1,C,-1)
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
            self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        residual *= soft_assign.unsqueeze(2)                
        vlad = residual.sum(dim=-1)#(1,8,128)
        vlad = F.normalize(vlad, p=2, dim=2)
        vlad = vlad.view(1, -1)
        vlad = F.normalize(vlad, p=2, dim=1) #(1,8*128)
        return vlad


class gostVLAD2(nn.Module):
    def __init__(self,num_clusters=8,gost=1,dim=128,normalize_input=True):
        super(gostVLAD2, self).__init__()
        self.num_clusters=num_clusters
        self.dim=dim
        self.gost=gost
        self.normalize_input=normalize_input
        self.fc=nn.Linear(dim,num_clusters+gost)
        self.centroids=nn.Parameter(torch.rand(num_clusters+gost,dim))
        self._init_params()
    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight.data)  
        nn.init.constant_(self.fc.bias.data, 0.0) 
    def forward(self,x):
        '''
        x:NxD 
        '''
        N,C=x.shape[:2]#10,128
        assert C==self.dim ,"feature dim not correct"
        if self.normalize_input:
            x=F.normalize(x,p=2,dim=0)
        soft_assign=self.fc(x).unsqueeze(0).permute(0,2,1)#(10,9)->(1,10,9)->(1,9,10)
        soft_assign=F.softmax(soft_assign,dim=1) 
 
        #soft_assign=soft_assign[:,:self.num_clusters,:]#(1,8,10)

        x_flatten=x.unsqueeze(0).permute(0,2,1)#x.view(1,C,-1)
        residual = x_flatten.expand(self.num_clusters+self.gost, -1, -1, -1).permute(1, 0, 2, 3) - \
            self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        residual *= soft_assign.unsqueeze(2)                
        vlad = residual.sum(dim=-1)#(1,9,128)
        vald=vald[:,:self.num_clusters,:]#(1,8,128)
        vlad = F.normalize(vlad, p=2, dim=2)
        vlad = vlad.view(1, -1)
        vlad = F.normalize(vlad, p=2, dim=1) #(1,8*128)
        return vlad


class EmbedNet(nn.Module):
    def __init__(self, base_model, net_vlad,dim_in=512,dim_out=128):
        super(EmbedNet, self).__init__()
        self.base_model = base_model
        self.net_vlad = net_vlad
        self.conv=nn.Conv2d(dim_in,dim_out,kernel_size=(1,1),bias=True)
        self.avgp=nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        x = self.base_model(x)
        x=self.conv(x) #
        x=self.avgp(x)
        x=x.squeeze() #(N,128)
        embedded_x = self.net_vlad.forward(x)
        emb2=self.net_vlad.forward2(x)
        return embedded_x,emb2


def test():
    encoder = resnet18(pretrained=False)
    base_model = nn.Sequential(
        encoder.conv1,
        encoder.bn1,
        encoder.relu,
        encoder.maxpool,
        encoder.layer1,
        encoder.layer2,
        encoder.layer3,
        encoder.layer4,
    )
    dim_in = list(base_model.parameters())[-1].shape[0]#512
    dim_out=128
    net_vlad=netVLAD2(dim=dim_out)
    #net_vlad=gostVLAD(dim=dim_out)
    model=EmbedNet(base_model,net_vlad,dim_in=dim_in,dim_out=dim_out)
    
    x=torch.rand(10,3,128,128)
    output1,output2=model(x)
    print(output1.shape,output2.shape)#(1,8*128)
    print(output1)
    print(output2.detach().numpy())


test()

     
