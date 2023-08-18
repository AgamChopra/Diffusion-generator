#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 16:09:44 2023

@author: agam
"""
from matplotlib import pyplot as plt


with open('/home/agam/Documents/diffusion_logs/diffuse_training_log.txt', 'r') as file:
    lines = file.readlines()
    file.close() 
l = [float(a.strip()) for a in lines[2:]]


plt.figure(figsize=(10,5))
plt.title("Generator Loss During Training")
plt.plot(l, label='G_loss')
plt.legend()
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()