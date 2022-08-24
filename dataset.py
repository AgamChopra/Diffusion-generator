import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt 
import random
from tqdm import trange


def load_human_bw():
    human_list = []
    for i in range(1,5233):
        img = cv2.imread('Dataset\human\humans (%d).jpg'%(i+1))[:,:,0:1]
        human_list.append(cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_CUBIC))  
    print('.human data loaded')
    return human_list


def load_celeb():
    human_list = []
    for i in trange(1,202598):
        idx = str(i).zfill(6)
        img = cv2.imread('E:\ML\Dog-Cat-GANs\Dataset\img_align_celeba\%s.jpg'%(idx))
        human_list.append(cv2.resize(img, dsize=(140, 140), interpolation=cv2.INTER_CUBIC))  
    print('.celeb data loaded')
    return human_list


def load_celeb_sample(N=10):
    human_list = []
    sample = np.random.randint(low=0, high=202598, size=N, dtype=int)
    for i in sample:
        idx = str(i).zfill(6)
        img = cv2.imread('E:\ML\Dog-Cat-GANs\Dataset\img_align_celeba\%s.jpg'%(idx))
        human_list.append(cv2.resize(img, dsize=(140, 140), interpolation=cv2.INTER_CUBIC))  
    print('.celeb data loaded')
    return human_list


def load_cats():
    cat_list = []
    for i in range(5650):
        img = cv2.imread('Dataset\cat_hq\cat (%d).jpg'%(i+1))
        cat_list.append(cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)) 
    print('.cat data loaded')
    return cat_list


def load_not_cats():
    not_cat_list = []
    for i in range(5000):
        img = cv2.imread('Dataset\cats\catnt (%d).jpg'%(i+1))
        not_cat_list.append(cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_CUBIC))
    print('..not cat data loaded')
    return not_cat_list


def load_photos():
    cat_list = []
    for i in range(7036):
        img = cv2.imread('Dataset\photo_jpg\photo (%d).jpg'%(i+1))
        cat_list.append(cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)) 
    print('.photo data loaded')
    return cat_list


def load_art():
    cat_list = []
    for i in range(300):
        img = cv2.imread('Dataset\monet_jpg\photo (%d).jpg'%(i+1))
        cat_list.append(cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)) 
    print('.art data loaded')
    return cat_list


def dataset():
    cat = load_cats()
    catnt = load_not_cats()
    return np.swapaxes(np.asanyarray(catnt), 1, -1), np.swapaxes(np.asanyarray(cat), 1, -1)


def cat_dataset():
    cat = load_cats()
    return np.swapaxes(np.asanyarray(cat), 1, -1)


def dog_dataset():
    dog = load_not_cats()
    return np.swapaxes(np.asanyarray(dog), 1, -1)


def human_dataset_bw():
    human = load_human_bw()
    human = np.expand_dims(np.asanyarray(human), axis=1)
    return np.swapaxes(human, 2, -1)


def photo_dataset():
    cat = load_photos()
    return np.swapaxes(np.asanyarray(cat), 1, -1)


def art_dataset():
    cat = load_art()
    return np.swapaxes(np.asanyarray(cat), 1, -1)


def celeb_dataset():
    cat = load_celeb()
    return np.swapaxes(np.asanyarray(cat), 1, -1)


def celeb_dataset_sample(N=10):
    cat = load_celeb_sample(N)
    return np.swapaxes(np.asanyarray(cat), 1, -1)


def load_dataset():
    random.seed(15)
    cat = load_cats()
    catnt = load_not_cats()
    rand_seed = random.sample(range(10000), 10000)
    x = []
    y = []
    for i in range(10000):
        if rand_seed[i] < 5000:
            x.append(cat[rand_seed[i]].T)
            y.append(1)
        else:
            x.append(catnt[rand_seed[i]-5000].T)
            y.append(0)
    print('...data stitching and randomization finished')
    return x,y


def dataset_():
    x,y = load_dataset()
    print('....train test data loaded')
    return np.stack(x[:9900]),np.stack(y[:9900]),np.stack(x[9900:]),np.stack(y[9900:])


def visualize(x,dark=True,title=None):
    if dark:
        plt.style.use('dark_background')
    else:
        plt.style.use('default')
    plt.imshow(cv2.cvtColor(x.T, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    if title != None:
        plt.title(title)
    plt.show()
    
    
def visualize_25(x,dark=True):
    if dark:
        plt.style.use('dark_background')
    else:
        plt.style.use('default')
    r = 5
    c = 5
    fig = plt.figure(figsize=(20,20))
    for i in range(x.shape[0]):
        fig.add_subplot(r,c,i+1)
        plt.imshow(cv2.cvtColor(x[i].T, cv2.COLOR_BGR2RGB))
        plt.axis('off')
    plt.show()
    
    
def visualize_16(x,dark=True):
    if dark:
        plt.style.use('dark_background')
    else:
        plt.style.use('default')
    r = 4
    c = 4
    fig = plt.figure(figsize=(10,10))
    for i in range(x.shape[0]):
        fig.add_subplot(r,c,i+1)
        plt.imshow(cv2.cvtColor(x[i].T, cv2.COLOR_BGR2RGB))
        plt.axis('off')
    plt.show()
    
    
def img_load(path, show = True):
    img = cv2.imread(path)
    x = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
    if show:
        plt.imshow(cv2.cvtColor(x, cv2.COLOR_BGR2RGB))
        plt.show()  
    print('image loaded!')
    return x


def torch_celeb_dataset():
    data = celeb_dataset()
    data = torch.from_numpy(data).to(dtype = torch.float)
    return data


def torch_celeb_dataset_sample(N=10):
    data = celeb_dataset_sample(N)
    data = torch.from_numpy(data).to(dtype = torch.float)
    return data


def torch_cat_dataset():
    data = cat_dataset()
    data = torch.from_numpy(data).to(dtype = torch.float)
    return data


def torch_photo_dataset():
    data = photo_dataset()
    data = torch.from_numpy(data).to(dtype = torch.float)
    return data


def main():
    data = celeb_dataset()
    print(data.shape)
    visualize_25(data[0:25])
    visualize(data[0])
    #data = torch_celeb_dataset()
    #print(data[0], data.max(),data.min())
    

if __name__ == '__main__':
    main()
    