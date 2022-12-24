import torch
from tqdm import trange
from numpy.random import choice
from torchvision import transforms as tf

import diffusion_models as models
import dataset as dst


print('cuda detected:',torch.cuda.is_available())


class rand_augment():
    def __init__(self):
        self.aug =tf.Compose([tf.RandomRotation(30),tf.ColorJitter(),tf.RandomVerticalFlip()])
    def __call__(self,x):
        return self.aug(x)


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
    idx = torch.tensor(choice(range(0, list_len), x.shape[0]),dtype=torch.long)
    #idx = torch.tensor(choice(range(1, list_len), x.shape[0]),dtype=torch.long)
    
    x_t1 = a1[idx-1].view(len(idx),1,1,1) * x + a2[idx-1].view(len(idx),1,1,1) * x_epsilon
    x_t2 = a1[idx].view(len(idx),1,1,1) * x + a2[idx].view(len(idx),1,1,1) * x_epsilon
    t2 = t_encodings[idx]
    
    return x_t1, x_t2, t2


class noise_generator():
    def __init__(self, T = 300, b_0 = 0.0001, b_t = 0.02, time_encoding_dim = 128):
        alpha_bar = torch.cumprod(1. - torch.linspace(b_0,b_t,T), dim=0)
        self.a1 = torch.cat([torch.ones(1), alpha_bar ** 0.5, torch.zeros(1)])
        self.a2 = torch.cat([torch.zeros(1), (1 - alpha_bar) ** 0.5, torch.ones(1)])
        self.t_encode = getPositionEncoding(T+2,time_encoding_dim) 
        
    def apply(self, x0):
        return apply_random_noise(x0,self.a1,self.a2,self.t_encode)
        

def diff_train(dataset, lr = 1E-2, epochs = 5, batch=32, T=1000, time_encoding_dim = 32, load_state = False, state=None, loss_type = 'smoothL1'):
    
    if not load_state:
        with open('E:\ML\Dog-Cat-GANs\diffuse_training_log.txt', 'w') as file:
            file.write('lr = %f, epochs = %d, batch = %d, T = %d, time_encoding_dim = %d, loss_type = %s\n'%(lr, epochs, batch, T, time_encoding_dim, loss_type))
            file.write('Average losses per epoch:\n')
            file.close()
    
    aug = rand_augment()
    
    Glosses = []
    
    CH = dataset.shape[1]
    
    print('loading noise generator...')
    noise = noise_generator(T=T, time_encoding_dim=time_encoding_dim)
    
    print('loading error function...')
    if loss_type == 'l1':
        error = torch.nn.L1Loss()
        print('   l1')
    elif loss_type == 'l2':
        error = torch.nn.MSELoss()
        print('   l2')
    else:
        error = torch.nn.SmoothL1Loss()
        print('   smoothL1')
    
    print('loading generator...', end =" ")
    #Generator 
    Gen = models.UNet(CH=CH,t_emb=time_encoding_dim,n=1).cuda()
    
    if load_state:
        print('loading previous run state...', end =" ")
        Gen.load_state_dict(torch.load("E:\ML\Dog-Cat-GANs\Gen-diff-Autosave.pt")) 
        print('done.')
        
    if state is not None:
        Gen.load_state_dict(torch.load(state)) 
    
    optimizerG = torch.optim.Adam(Gen.parameters(),lr)
    
    print('optimizing...')
    
    for eps in range(epochs):  
        print('Epoch: [%d/%d]'%(eps,epochs))
        idx_ = torch.randperm(dataset.shape[0])
        Gen.train()
        
        for b in trange(0,dataset.shape[0]-batch,batch):  
                
            optimizerG.zero_grad()       
                
            xt1,xt2,t2 = noise.apply(norm(aug(dataset[idx_[b:b+batch]])))
            xt1,xt2,t2 = xt1.cuda(), xt2.cuda(), t2.cuda()
                
            xt1p = Gen(xt2,t2)
                
            errG = error(xt1,xt1p)
            errG.backward()
            optimizerG.step()
    
            Glosses.append(errG.item())
            
            if b % int(dataset.shape[0]/10) == 0 or b == 0:
                torch.save(Gen.state_dict(), "E:\ML\Dog-Cat-GANs\Gen-diff-Autosave.pt")
        
        av_ls = sum(Glosses[-int(len(idx_)/batch):])/int(len(idx_)/batch)
        print('\tAverage Error: %.10f'%(av_ls))         
        with open('E:\ML\Dog-Cat-GANs\diffuse_training_log.txt', 'a') as file:
            file.writelines(str(av_ls)+'\n')
            file.close()
        
        if eps % 10 == 0: 
            with torch.no_grad():
                t_ = 75
                y = torch.rand(batch,dataset.shape[1],dataset.shape[2],dataset.shape[3]).cuda()
                Gen.eval()
                dst.plt.figure(figsize=(T,5))
                r = 1
                c = int(T/t_) + 1
                fig = dst.plt.figure(figsize=(c*6,r*6))
                fig.add_subplot(r,c,1)
                dst.plt.imshow(dst.cv2.cvtColor(norm(torch.squeeze(y[0])).cpu().numpy().T, dst.cv2.COLOR_BGR2RGB))
                dst.plt.axis('off')
                ctr = 2
                for t in range(T):               
                    t_en = noise.t_encode[T-t-1].cuda()
                    y = Gen(y,t_en)        
                    if t%t_ == 0:
                        fig.add_subplot(r,c,ctr)
                        ctr+=1
                        dst.plt.imshow(dst.cv2.cvtColor(norm(torch.squeeze(y[0])).cpu().numpy().T, dst.cv2.COLOR_BGR2RGB))
                        dst.plt.axis('off')
                dst.plt.show()
                    
    return Gen, Glosses


def train(T = 300, Gsave = 'E:\ML\Dog-Cat-GANs\Gen-diff-Autosave.pt'):#"E:\ML\Dog-Cat-GANs\Gen_temp.pt"):
    
    print('loading data...')
    dataset = dst.torch_cat_dataset(True)#torch_celeb_dataset()
    print('done.')

    Gen,Gl = diff_train(dataset = dataset, lr = 1E-6, epochs = 40000, batch = 32, T=T,loss_type ='smoothl1', load_state = True, time_encoding_dim = 256)
    
    torch.save(Gen.state_dict(), Gsave)
   
    dst.plt.figure(figsize=(10,5))
    dst.plt.title("Generator Loss During Training")
    dst.plt.plot(Gl, label='G_loss')
    dst.plt.legend()
    dst.plt.xlabel("iterations")
    dst.plt.ylabel("Loss")
    dst.plt.legend()
    dst.plt.show()
       
    return Gen


def gen_img(T = 500):    
    with torch.no_grad():
        Gen = models.UNet(CH=3,t_emb=32,n=1).cuda()
        Gen.load_state_dict(torch.load("E:\ML\Dog-Cat-GANs\Gen-diff-Autosave.pt")) 
        t_encode = getPositionEncoding(T,32)
        y = torch.rand(1,3,140,140).cuda()
        Gen.eval()
        dst.plt.figure(figsize=(T,5))
        r = 1
        c = T+1
        fig = dst.plt.figure(figsize=(T*4,4))
        fig.add_subplot(r,c,1)
        dst.plt.imshow(dst.cv2.cvtColor(norm(torch.squeeze(y)).cpu().numpy().T, dst.cv2.COLOR_BGR2RGB))
        dst.plt.axis('off')
        for t in range(T):
            t_en = t_encode[T-t-1].cuda()
            y = Gen(y,t_en)
            fig.add_subplot(r,c,t+2)
            dst.plt.imshow(dst.cv2.cvtColor(norm(torch.squeeze(y)).cpu().numpy().T, dst.cv2.COLOR_BGR2RGB))
            dst.plt.axis('off')
        dst.plt.show()
        
    
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
