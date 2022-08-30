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
            self.mlp = nn.Linear(embd_dim,out_c)
            
            self.layer = nn.Sequential(nn.Conv2d(in_channels = in_c, out_channels = out_c, kernel_size = 3),nn.ReLU(),nn.BatchNorm2d(out_c))
            
            self.out_block = nn.Sequential(nn.Conv2d(in_channels = out_c, out_channels = out_c, kernel_size = 3),nn.ReLU(),nn.BatchNorm2d(out_c))
        else:
            self.mlp = nn.Linear(embd_dim,hid_c)
            
            self.layer = nn.Sequential(nn.Conv2d(in_channels = in_c, out_channels = hid_c, kernel_size = 3),nn.ReLU(),nn.BatchNorm2d(hid_c))
            
            self.out_block = nn.Sequential(nn.Conv2d(in_channels = hid_c, out_channels = hid_c, kernel_size = 3),nn.ReLU(),nn.BatchNorm2d(hid_c),
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
        self.layer1 = Block(in_c = CH, embd_dim = t_emb, out_c = int(64/n))
        
        self.layer2 = Block(in_c = int(64/n), embd_dim = t_emb, out_c = int(128/n))
        
        self.layer3 = Block(in_c = int(128/n), embd_dim = t_emb, out_c = int(256/n))
        
        self.layer4 = Block(in_c = int(256/n), embd_dim = t_emb, out_c = int(512/n))
        
        self.layer5 = Block(in_c = int(512/n), embd_dim = t_emb, out_c = int(512/n), hid_c = int(1024/n))
        
        self.layer6 = Block(in_c = int(1024/n), embd_dim = t_emb, out_c = int(256/n), hid_c = int(512/n))
        
        self.layer7 = Block(in_c = int(512/n), embd_dim = t_emb, out_c = int(128/n), hid_c = int(256/n))
        
        self.layer8 = Block(in_c = int(256/n), embd_dim = t_emb, out_c = int(64/n), hid_c = int(128/n))
        
        self.layer9 = Block(in_c = int(128/n), embd_dim = t_emb, out_c = int(64/n))
        
        self.out = nn.Conv2d(in_channels = int(64/n), out_channels = CH, kernel_size = 1)
        
        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        
        
    def forward(self,x,t,device = 'cuda:0'):
        
        y1 = self.layer1(x,t)
        y = self.maxpool(y1)
        
        y2 = self.layer2(y,t)
        y = self.maxpool(y2)
        
        y3 = self.layer3(y,t)
        y = self.maxpool(y3)
        
        y4 = self.layer4(y,t)
        y = self.maxpool(y4)
        
        y = self.layer5(y,t)
        
        y = torch.cat((y4,pad2d(y, y4, device = device)),dim = 1)
        y = self.layer6(y,t)
        
        y = torch.cat((y3,pad2d(y, y3, device = device)),dim = 1)
        y = self.layer7(y,t)
        
        y = torch.cat((y2,pad2d(y, y2, device = device)),dim = 1)
        y = self.layer8(y,t)
        
        y = torch.cat((y1,pad2d(y, y1, device = device)),dim = 1)
        y = self.layer9(y,t)
        
        y = pad2d(y, x, device = device)
        
        y = self.out(y)
        
        return y
    

def test(device = 'cpu'):
    batch = 32
    a = torch.ones((batch,3,140,140),device=device)
    t = torch.ones((batch,32),device=device)
    
    model = UNet().to(device)
    print(model)
    
    b = model(a,t,device)
    
    print(a.shape,t.shape)
    print(b.shape)
        
        
if __name__ == '__main__':
    test('cpu')