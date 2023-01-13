import torch
import torch.nn as nn


def pad2d(input_, target, device = 'cpu'):
    output_ = torch.zeros((target.shape[0],input_.shape[1],target.shape[2],target.shape[3]),device=device)
    start_idx = int((output_.shape[-1] - input_.shape[-1]) / 2)
    try:
        output_[:,:,start_idx:-start_idx,start_idx:-start_idx] = input_
    except:
        try:
            output_[:,:,start_idx:-start_idx-1,start_idx:-start_idx-1] = input_
        except:
            output_[:,:,1:,1:] = input_
    return output_


class Block(nn.Module):
    def __init__(self,in_c,embd_dim,out_c,hid_c=None):
        super(Block, self).__init__()
        
        if hid_c is None:
            self.mlp = nn.Sequential(nn.Linear(embd_dim,out_c),nn.ReLU())
            
            self.layer = nn.Sequential(nn.Conv2d(in_channels = in_c, out_channels = out_c, kernel_size = 3),nn.ReLU(),nn.BatchNorm2d(out_c))
            
            self.out_block = nn.Sequential(nn.Conv2d(in_channels = out_c, out_channels = out_c, kernel_size = 2),nn.ReLU(),nn.BatchNorm2d(out_c))
        else:
            self.mlp = nn.Sequential(nn.Linear(embd_dim,hid_c),nn.ReLU())
            
            self.layer = nn.Sequential(nn.Conv2d(in_channels = in_c, out_channels = hid_c, kernel_size = 3),nn.ReLU(),nn.BatchNorm2d(hid_c))
            
            self.out_block = nn.Sequential(nn.Conv2d(in_channels = hid_c, out_channels = hid_c, kernel_size = 2),nn.ReLU(),nn.BatchNorm2d(hid_c),
                                           nn.ConvTranspose2d(in_channels = hid_c, out_channels = out_c, kernel_size = 2, stride= 2),nn.ReLU(),nn.BatchNorm2d(out_c))
        
    def forward(self,x,t):
        t = self.mlp(t)
        y = self.layer(x)
        t = t[(..., ) + (None, ) * 2]
        y = y + t
        y = self.out_block(y)
        return y


class UNet(nn.Module):
    def __init__(self,CH=3,t_emb=32,n=1):
        super(UNet, self).__init__()
        #layers
        self.time_mlp = nn.Sequential(nn.Linear(t_emb, t_emb),nn.ReLU())
        
        self.layer1 = nn.Conv2d(CH, int(64/n), 3, 1)
        
        self.layer2 = Block(in_c = int(64/n), embd_dim = t_emb, out_c = int(128/n))
        
        self.layer3 = Block(in_c = int(128/n), embd_dim = t_emb, out_c = int(256/n))
        
        self.layer4 = Block(in_c = int(256/n), embd_dim = t_emb, out_c = int(512/n))
        
        self.layer5 = Block(in_c = int(512/n), embd_dim = t_emb, out_c = int(512/n), hid_c = int(1024/n))
        
        self.layer6 = Block(in_c = int(1024/n), embd_dim = t_emb, out_c = int(256/n), hid_c = int(512/n))
        
        self.layer7 = Block(in_c = int(512/n), embd_dim = t_emb, out_c = int(128/n), hid_c = int(256/n))
        
        self.layer8 = Block(in_c = int(256/n), embd_dim = t_emb, out_c = int(64/n))
        
        self.out = nn.Conv2d(in_channels = int(64/n), out_channels = CH, kernel_size = 1)
        
        self.pool2 = nn.Conv2d(in_channels=int(128/n),out_channels=int(128/n),kernel_size=2,stride=2)
        
        self.pool3 = nn.Conv2d(in_channels=int(256/n),out_channels=int(256/n),kernel_size=2,stride=2)
        
        self.pool4 = nn.Conv2d(in_channels=int(512/n),out_channels=int(512/n),kernel_size=2,stride=2)
        
        
    def forward(self,x,t,device = 'cuda:0'):
        t = self.time_mlp(t)        
        y = self.layer1(x)
        
        y2 = self.layer2(y,t)
        y = self.pool2(y2)
        
        y3 = self.layer3(y,t)
        y = self.pool3(y3)
        
        y4 = self.layer4(y,t)
        y = self.pool4(y4)
        
        y = self.layer5(y,t)
        
        y = torch.cat((y4,pad2d(y, y4, device = device)),dim = 1)
        y = self.layer6(y,t)
        
        y = torch.cat((y3,pad2d(y, y3, device = device)),dim = 1)
        y = self.layer7(y,t)
        
        y = torch.cat((y2,pad2d(y, y2, device = device)),dim = 1)
        y = self.layer8(y,t)
        
        y = pad2d(y, x, device = device)
        
        y = self.out(y)
        
        return y
    

def test(device = 'cpu'):
    batch = 1
    a = torch.ones((batch,3,64,64),device=device)
    t = torch.ones((batch,32),device=device)
    
    model = UNet().to(device)
    print(model)
    
    b = model(a,t,device)
    
    print(a.shape,t.shape)
    print(b.shape)
    
    from matplotlib import pyplot as plt
    
    plt.imshow(a[0].T.detach().cpu().numpy())
    plt.show()
    
    plt.imshow(b[0].T.detach().cpu().numpy())
    plt.show()
        
        
if __name__ == '__main__':
    test('cpu')
    
'''
import torch
from matplotlib import pyplot as plt

def getPositionEncoding(seq_len, d=512, n=10000):
    P = torch.zeros((seq_len, d))
    for k in range(seq_len):
        for i in torch.arange(int(d/2)):
            denominator = torch.pow(n, 2*i/d)
            P[k, 2*i] = torch.sin(k/denominator)
            P[k, 2*i+1] = torch.cos(k/denominator)
    return P


p = getPositionEncoding(200,256)

print(p.shape)

plt.imshow(p.cpu().numpy(),cmap='jet')
plt.show()
'''