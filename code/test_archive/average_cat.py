# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 15:57:56 2023

@author: Agam
"""
import numpy as np
import cv2
from tqdm import trange
from matplotlib import pyplot as plt


def load_cats():
    cat_list = []
    for i in trange(5653):
        img = cv2.imread(
            'E:/ML/Dog-Cat-GANs/Dataset/cat_hq/cat (%d).jpg' % (i+1))
        x = cv2.resize(img, dsize=(512, 512),
                       interpolation=cv2.INTER_CUBIC).T
        cat_list.append([x[2], x[1], x[0]])
    print('.cat data loaded')
    return cat_list


def cat_dataset():
    cat = load_cats()
    return np.asanyarray(cat)


def conv2D(img, kernel, padding=0, strides=1):  # input img -> 2d (x,y). Greyscale imgs only!
    kernel = np.flipud(np.fliplr(kernel))

    output = np.zeros((int(((img.shape[0] - kernel.shape[0] + 2 * padding) / strides) + 1), int(
        ((img.shape[1] - kernel.shape[1] + 2 * padding) / strides) + 1)))

    if padding != 0:
        img_ = np.zeros((img.shape[0] + padding*2, img.shape[1] + padding*2))

        img_[int(padding):int(-1 * padding),
             int(padding):int(-1 * padding)] = img
    else:
        img_ = img

    for y in range(img.shape[1]):
        if y > img.shape[1] - kernel.shape[1]:
            break

        if y % strides == 0:
            for x in range(img.shape[0]):
                if x > img.shape[0] - kernel.shape[0]:
                    break
                try:
                    if x % strides == 0:
                        output[x, y] = (
                            kernel * img_[x: x + kernel.shape[0], y: y + kernel.shape[1]]).sum()
                except:
                    break

    return output


data = cat_dataset()
print(data.shape)
data_gr = 0.2126 * data[:, 0] + 0.7152 * data[:, 1] + 0.0722 * data[:, 2]
data_ = data_gr / 255
avg_cat = np.sum(data_, axis=0) / data_.shape[0]
print(avg_cat.shape)
print(data.max(), data.min())

print(avg_cat.max(), avg_cat.min())
avg_cat_ = ((avg_cat - avg_cat.min()) /
            (avg_cat.max() - avg_cat.min())) * 255

plt.figure(figsize=(15, 15), dpi=500)
plt.imshow(avg_cat.T, cmap='gray')
plt.axis('off')
plt.show()

kernel_x = np.asarray([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
kernel_y = kernel_x.T
kernel_Gauss = (1/273) * np.array([[1, 4, 7, 4, 1], [4, 16, 26, 16, 4], [
    7, 26, 41, 26, 7], [4, 16, 26, 16, 4], [1, 4, 7, 4, 1]])

blurred = conv2D(avg_cat_, kernel_Gauss, padding=2)
high_freq = avg_cat_ - blurred

img = avg_cat_ + 2 * high_freq
img = ((img - img.min()) /
       (img.max() - img.min())) * 255

plt.figure(figsize=(15, 15), dpi=500)
plt.imshow(img.T, cmap='gray')
plt.axis('off')
plt.show()

cv2.imwrite("E:/ML/Dog-Cat-GANs/Dataset/cat_hq/cat_average_sharp_2.jpg",
            img.T.astype(dtype=np.int32))

I_x = conv2D(img, kernel_x, padding=1)
I_x = ((I_x - I_x.min()) /
       (I_x.max() - I_x.min()))
I_y = conv2D(avg_cat, kernel_y, padding=1)
I_y = ((I_y - I_y.min()) /
       (I_y.max() - I_y.min()))
I_fin = I_x**2 * I_y**2
I_fin = ((I_fin - I_fin.min()) /
         (I_fin.max() - I_fin.min())) * 255

plt.figure(figsize=(15, 15), dpi=500)
plt.imshow(I_fin.T, cmap='gray')
plt.axis('off')
plt.show()

cv2.imwrite("E:/ML/Dog-Cat-GANs/Dataset/cat_hq/cat_average_grad_map.jpg",
            I_fin.T.astype(dtype=np.int32))
