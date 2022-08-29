import torch
from tqdm import trange
import random

import diffusion_models as models
import dataset as dst


print('cuda detected:',torch.cuda.is_available())


def getPositionEncoding(seq_len, d=512, n=10000):
    P = torch.zeros((seq_len, d))
    for k in range(seq_len):
        for i in torch.arange(int(d/2)):
            denominator = torch.pow(n, 2*i/d)
            P[k, 2*i] = torch.sin(k/denominator)
            P[k, 2*i+1] = torch.cos(k/denominator)
    return P


def norm(x):
    return (x-torch.min(x))/(torch.max(x)-torch.min(x))


def apply_random_noise(x,a1,a2,t_encodings):
    x_epsilon = torch.rand_like(x)
    list_len = len(a1)
    idx = torch.tensor(random.sample(range(1, list_len), x.shape[0]))
    
    x_t1 = a1[idx-1].view(len(idx),1,1,1) * x + a2[idx-1].view(len(idx),1,1,1) * x_epsilon
    x_t2 = a1[idx].view(len(idx),1,1,1) * x + a2[idx].view(len(idx),1,1,1) * x_epsilon
    t2 = t_encodings[idx]
    
    return x_t1, x_t2, t2


class noise_generator():
    def __init__(self, T = 1000, b_0 = 0.0001, b_t = 0.02, time_encoding_dim = 128):
        alpha_bar = torch.cumprod(1. - torch.linspace(b_0,b_t,T), dim=0)
        self.a1 = torch.cat([torch.ones(1), alpha_bar ** 0.5, torch.zeros(1)])
        self.a2 = torch.cat([torch.zeros(1), (1 - alpha_bar) ** 0.5, torch.ones(1)])
        self.t_encode = getPositionEncoding(T+2,time_encoding_dim) 
        
    def apply(self, x0):
        return apply_random_noise(x0,self.a1,self.a2,self.t_encode)
        

def diff_train(dataset, lr = 1E-4, epochs = 5, batch=32, beta1=0.5, beta2=0.999, T=1000, time_encoding_dim = 32, dmt = 32, load_state = False, state=None):   
    REG = int(batch / dmt)
    batch -=  REG
    
    Glosses = []
    
    CH = dataset.shape[1]
    
    print('loading generator...', end =" ")
    #Generator 
    Gen = models.UNet(CH).cuda()
    
    if load_state:
        print('loading previous run state...', end =" ")
        Gen.load_state_dict(torch.load("E:\ML\Dog-Cat-GANs\Gen-diff-Autosave.pt")) 
        print('done.')
        
    if state is not None:
        Gen.load_state_dict(torch.load(state)) 
    
    optimizerG = torch.optim.Adam(Gen.parameters(),lr,betas=(beta1, beta2))
    
    print('loading noise generator...')
    noise = noise_generator(T=T, time_encoding_dim=time_encoding_dim)
    
    print('loading error function...')
    error = torch.nn.MSELoss()
    
    print('optimizing...')
    
    for eps in trange(epochs):        
        idx_ = torch.randperm(dataset.shape[0])
        
        for b in range(0,dataset.shape[0]-batch,batch):  
                
            optimizerG.zero_grad()       
                
            xt1,xt2,t2 = noise.apply(norm(dataset[idx_[b:b+batch]]))
            xt1,xt2,t2 = xt1.cuda(), xt2.cuda(), t2.cuda()
                
            xt1p = Gen(xt2,t2)
                
            errG = error(xt1,xt1p)
            errG.backward()
            optimizerG.step()
    
            Glosses.append(errG.item())
            
            if b % int(dataset.shape[0]/10) == 0 or b == 0:
                #print(torch.max(xt1),torch.min(xt1),torch.mean(xt1),torch.var(xt1))
                #print(torch.max(xt2),torch.min(xt2),torch.mean(xt2),torch.var(xt2))
                #print(torch.max(xt1p),torch.min(xt1p),torch.mean(xt1p),torch.var(xt1p))
                #dst.visualize(norm(xt1[0]).cpu().detach().numpy(),dark = False, title='X_t1')
                #dst.visualize(norm(xt2[0]).cpu().detach().numpy(),dark = False, title='X_t2')
                #dst.visualize(norm(xt1p[0]).cpu().detach().numpy(),dark = True, title='X_t2 -> X_t1')
                torch.save(Gen.state_dict(), "E:\ML\Dog-Cat-GANs\Gen-diff-Autosave.pt")
                    
        print('[%d/%d]\tAverage Error: %.4f'% (eps, epochs, sum(Glosses[-int(len(idx_)/batch):])/int(len(idx_)/batch)))  
                    
      
    return Gen, Glosses


def train(T = 200, Gsave = "E:\ML\Dog-Cat-GANs\Gen_temp.pt"):
    
    print('loading data...')
    dataset = dst.torch_celeb_dataset()
    print('done.')

    Gen,Gl = diff_train(dataset=dataset,lr = 1E-4, epochs = 100, batch=32, beta1=0.5, beta2=0.999, T=T, dmt = 32, load_state = True)
   
    dst.plt.figure(figsize=(10,5))
    dst.plt.title("Generator Loss During Training")
    dst.plt.plot(Gl, label='G_loss')
    dst.plt.legend()
    dst.plt.xlabel("iterations")
    dst.plt.ylabel("Loss")
    dst.plt.legend()
    dst.plt.show()
    
    torch.save(Gen.state_dict(), Gsave)   
    return Gen


def gen_img(Gen = None, T = 200):
    with torch.no_grad():
        t_encode = getPositionEncoding(T,32)
        if Gen is None:
            Gsave = "E:\ML\Dog-Cat-GANs\Gen-diff-Autosave.pt"
            Gen = models.UNet(3).eval().cuda()
            try:
                Gen.load_state_dict(torch.load(Gsave))
            except:
                print('Warning: Could not load generator parameters at',Gsave)
        
        noise = torch.rand((25,3,140,140)).cuda()
        for t in range(T):
            t_en = t_encode[T-t-1].cuda()
            warped = Gen(noise,t_en)
            noise = warped
            
            if t % 1 == 0:
                wd = norm(warped).cpu().detach().numpy()
                dst.visualize_25(wd,dark=False)
                del wd
        
    
def main():
    x = int(input('Would you like to train model(press \'1\') or generate synthetic images from previous state(press \'2\')?'))
    if x == 1:
        train()
    elif x==2:
        gen_img()
    else:
        print('Value Error: Please enter either 1 or 2')
        

if __name__ == "__main__":
    main()
