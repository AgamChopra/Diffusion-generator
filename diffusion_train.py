import torch
from tqdm import trange
from numpy.random import choice
from torchvision import transforms as tf

import diffusion_models as models
import dataset as dst

print('cuda detected:',torch.cuda.is_available())

MODE = int(input('train model(press \'1\') generate synthetic images from previous state(press \'2\'): '))
if MODE == 1:
    load_checkpoint = input('Continue training from previous checkpoint? (yes/no)')
    if load_checkpoint == 'yes':
        CT = True
    elif load_checkpoint == 'no':
        CT = False
    else:
        print('Incorrect value, using fresh model')
        CT = False
    DATASET = dst.torch_car_dataset(False,2)#torch.zeros((1,3,128,128))#
else:
    DATASET = torch.zeros((1,int(input('Channel size:')),int(input('Height:')),int(input('Width:'))))
    
T_ENC = 64
T_DIFF = 1000
N_UNET = 0.5
LR = 2E-5
EPS = 1000
BATCH = 32
LOSS_TYPE = 'l1'
CH = DATASET.shape[1]
T_PRINT = 4


class rand_augment():
    def __init__(self):
        self.aug =tf.Compose([tf.RandomRotation(2),tf.ColorJitter(),tf.RandomVerticalFlip()])
    def __call__(self,x):
        return x#self.aug(x)


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
    #print(len(a1), len(a2), len(t_encodings))
    x_epsilon = torch.rand_like(x)
    list_len = len(a1)
    idx = torch.tensor(choice(range(0, list_len - 1), x.shape[0]),dtype=torch.long)
    #idx = torch.tensor(choice(range(1, list_len), x.shape[0]),dtype=torch.long)    
    #x_t1 = a1[idx-1].view(len(idx),1,1,1) * x + a2[idx-1].view(len(idx),1,1,1) * x_epsilon
    x_t2 = a1[idx].view(len(idx),1,1,1) * x + a2[idx].view(len(idx),1,1,1) * x_epsilon
    t2 = t_encodings[idx]
    
    return x_t2, t2, x_epsilon


def get_alphas(T = 300, b_0 = 0.0001, b_t = 0.02):
    alpha_bar = torch.cumprod(1. - torch.linspace(b_0,b_t,T), dim=0)
    a1 = alpha_bar ** 0.5
    a2 = (1 - alpha_bar) ** 0.5
    return a1, a2
    

class noise_generator():
    def __init__(self, T = 300, time_encoding_dim = 128):
        self.a1, self.a2 = get_alphas(T = T)
        self.t_encode = getPositionEncoding(T,time_encoding_dim) #getPositionEncoding(T+2,time_encoding_dim) 
        
    def apply(self, x0):
        return apply_random_noise(x0,self.a1,self.a2,self.t_encode)
    

def remove_noise(X_2, noise, a1_1, a2_1, a1_2, a2_2):
    X_1 = X_2 - noise * (a1_1 * a2_2 - a1_2 * a2_1)
    return X_1
        

def diff_train(dataset, lr = 1E-2, epochs = 5, batch=32, T=1000, time_encoding_dim = 32, load_state = False, state=None, loss_type = 'smoothL1'):
    
    if not load_state:
        with open('/home/agam/Documents/diffusion_logs/diffuse_training_log.txt', 'w') as file:
            file.write('lr = %f, epochs = %d, batch = %d, T = %d, time_encoding_dim = %d, loss_type = %s\n'%(lr, epochs, batch, T, time_encoding_dim, loss_type))
            file.write('Average losses per epoch:\n')
            file.close()
    
    aug = rand_augment()
    
    Glosses = []
    
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
    Gen = models.UNet(CH=CH,t_emb=time_encoding_dim,n=N_UNET).cuda()
    
    if load_state:
        print('loading previous run state...', end =" ")
        Gen.load_state_dict(torch.load("/home/agam/Documents/diffusion_logs/Gen-diff-Autosave.pt")) 
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
                
            xt2,t2,psi = noise.apply(norm(aug(dataset[idx_[b:b+batch]])))
            xt2,t2,psi = xt2.cuda(), t2.cuda(), psi.cuda()
                
            psi_p = Gen(xt2,t2)
                
            errG = error(psi,psi_p)
            errG.backward()
            optimizerG.step()
    
            Glosses.append(errG.item())
            
        if eps % 10 == 0:
            torch.save(Gen.state_dict(), "/home/agam/Documents/diffusion_logs/Gen-diff-Autosave.pt")
        
        av_ls = sum(Glosses[-int(len(idx_)/batch):])/int(len(idx_)/batch)
        print('\tAverage Error: %.10f'%(av_ls))         
        with open('/home/agam/Documents/diffusion_logs/diffuse_training_log.txt', 'a') as file:
            file.writelines(str(av_ls)+'\n')
            file.close()
        
        if eps % 20 == 0:           
            dst.plt.figure(figsize=(10,5))
            dst.plt.title("Loss Training")
            dst.plt.plot(Glosses, label='loss')
            dst.plt.legend()
            dst.plt.xlabel("iterations")
            dst.plt.ylabel("Loss")
            dst.plt.legend()
            dst.plt.show()
            
            with torch.no_grad():
                a1, a2 = get_alphas(T = T)
                t_ = int(T / T_PRINT)
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
                    psi = Gen(y,t_en)                          
                    y = remove_noise(y, psi, a1[T-t-1].cuda(), a2[T-t-1].cuda(), a1[T-t-2].cuda(), a2[T-t-2].cuda())                   
                    if t%t_ == 0:
                        fig.add_subplot(r,c,ctr)
                        ctr+=1
                        dst.plt.imshow(dst.cv2.cvtColor(norm(torch.squeeze(y[0])).cpu().numpy().T, dst.cv2.COLOR_BGR2RGB))
                        dst.plt.axis('off')
                dst.plt.show()
                    
    return Gen, Glosses


def train(T = T_DIFF, Gsave = '/home/agam/Documents/diffusion_logs/Gen-diff-Autosave.pt', continue_trn = False):#"E:\ML\Dog-Cat-GANs\Gen_temp.pt"):
    
    print('loading data...')
    dataset = DATASET#dst.torch_car_dataset(True)#dst.torch_celeb_dataset()#dst.torch_cat_dataset(True)
    print('done.')

    Gen,Gl = diff_train(dataset = dataset, lr = LR, epochs = EPS, batch = BATCH, T=T,loss_type =LOSS_TYPE, load_state = continue_trn, time_encoding_dim = T_ENC)
    
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


def gen_img(T = T_DIFF):    
    with torch.no_grad():
        r = 1
        c = 10
        
        Gen = models.UNet(CH=CH,t_emb=T_ENC,n=N_UNET).cuda()
        Gen.load_state_dict(torch.load("/home/agam/Documents/diffusion_logs/Gen-diff-Autosave.pt")) 
        t_encode = getPositionEncoding(T,T_ENC)
        y = torch.rand(c,CH,DATASET.shape[2],DATASET.shape[3]).cuda()
        Gen.eval()
             
        a1, a2 = get_alphas(T = T)
        
        t_ = int(T / T_PRINT)
        for t in trange(T):               
            t_en = t_encode[T-t-1].cuda()
            
            psi = Gen(y,t_en)        
            y = remove_noise(y, psi, a1[T-t-1].cuda(), a2[T-t-1].cuda(), a1[T-t-2].cuda(), a2[T-t-2].cuda())

            if t%t_ == 0:
                fig = dst.plt.figure(figsize=(15,8),dpi=250)
                for i in range(c):
                    fig.add_subplot(r,c,i+1)
                    dst.plt.imshow(dst.cv2.cvtColor(norm(torch.squeeze(y[i])).cpu().numpy().T, dst.cv2.COLOR_BGR2RGB))
                    dst.plt.axis('off')
                dst.plt.show()
        
    
def main():   
    if MODE == 1:
        train(continue_trn=CT)
    elif MODE==2:
        gen_img()
    else:
        print('Value Error: Please enter either 1 or 2')
        

if __name__ == "__main__":
    main()
