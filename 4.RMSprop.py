# Date: 2018-08-04 18:43
# Author: Enneng Yang
# Abstract：simple linear regression problem:Y=AX+B, optimization is RMSprop

import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import numpy as np
import tensorflow as tf


rate = 0.01 # learning rate
def da(y, y_p,x):
    return (y-y_p)*(-x)

def db(y, y_p):
    return (y-y_p)*(-1)

def calc_loss(a,b,x,y):
    tmp = y - (a * x + b)
    tmp = tmp ** 2  # Square every element in the matrix
    SSE = sum(tmp) / (2*len(x)) # Take the average
    return SSE

#draw all curve point
def draw_hill(x,y):
    a = np.linspace(-20,20,100)
    # print(a)
    b = np.linspace(-20,20,100)
    x = np.array(x)
    y = np.array(y)

    allSSE = np.zeros(shape=(len(a),len(b)))
    for ai in range(0,len(a)):
        for bi in range(0,len(b)):
            a0 = a[ai]
            b0 = b[bi]
            SSE = calc_loss(a=a0, b=b0, x=x, y=y)
            allSSE[ai][bi] = SSE

    a,b = np.meshgrid(a, b)

    return [a,b,allSSE]

def shuffle_data(x,y):
    # random disturb x，y，while keep x_i corresponding to y_i
    seed = random.random()
    random.seed(seed)
    random.shuffle(x)
    random.seed(seed)
    random.shuffle(y)

def get_batch_data(x, y, batch=3):
    shuffle_data(x, y)
    x_new = x[0:batch]
    y_new = y[0:batch]
    return [x_new, y_new]

# create simulated data
# x = [30, 35, 37, 59, 70, 76, 88, 100]
# y = [1100,	1423,	1377,	1800,	2304,	2588,	3495,	4839]
a = np.loadtxt('../Data/LRdata.txt')
x = a[:, 1]
y = a[:, 2]
# Data normalization
x_max = max(x)
x_min = min(x)
y_max = max(y)
y_min = min(y)

for i in range(0, len(x)):
    x[i] = (x[i] - x_min)/(x_max - x_min)
    y[i] = (y[i] - y_min)/(y_max - y_min)


[ha, hb, hallSSE] = draw_hill(x, y)

# init a,b value
a = 10.0
b = -20.0
fig = plt.figure(1, figsize=(12, 8))
# fig.suptitle('Adagrad, learning rate: %.2f'%(rate), fontsize=15)

# draw fig.1 contour line
plt.subplot(1, 2, 1)
plt.contourf(ha, hb, hallSSE, 15, alpha=0.5, cmap=plt.cm.hot)
C = plt.contour(ha, hb, hallSSE, 15, colors='blue')
plt.clabel(C, inline=True)
plt.title('RMSprop')
plt.xlabel('opt param: a')
plt.ylabel('opt param: b')


plt.ion() # iteration on


all_loss = []
all_step = []

last_a = a
last_b = b

theta = np.array([0, 0]).astype(np.float32)
E_grad = np.array([0, 0]).astype(np.float32)
# E_theta = np.array([0, 0]).astype(np.float32)

epsilon = 1e-2
gamma = 0.9

for step in range(1, 100):
    loss = 0
    all_da = 0
    all_db = 0

    shuffle_data(x, y)
    [x_new, y_new] = get_batch_data(x, y, batch=4)
    all_d = np.array([0, 0]).astype(np.float32)
    for i in range(0, len(x_new)):
        y_p = a * x_new[i] + b
        loss += (y_new[i] - y_p) * (y_new[i] - y_p)/2
        all_da += da(y_new[i], y_p, x_new[i])
        all_db += db(y_new[i], y_p)

    loss = loss / len(x_new)
    all_d = np.array([all_da, all_db])

    # draw fig.1 contour line
    plt.subplot(1, 2, 1)
    plt.scatter(a, b, s=5, color='black')
    plt.plot([last_a, a], [last_b, b], color='red', label="RMSprop")

    # draw fig.2 loss line
    all_loss.append(loss)
    all_step.append(step)

    plt.subplot(1, 2, 2)
    plt.plot(all_step, all_loss, color='orange', label='RMSprop')

    plt.title('RMSprop')
    plt.xlabel("step")
    plt.ylabel("loss")

    last_a = a
    last_b = b

    # update param

    E_grad = gamma * E_grad + (1-gamma) * (all_d ** 2)
    rms_grad = np.sqrt(E_grad + epsilon)

    # E_theta = gamma * E_theta + (1-gamma) * (theta ** 2)
    # rms_theta = np.sqrt(E_theta + epsilon)

    theta = -( rate / rms_grad) * all_d
    [a, b] = [a, b] + theta

    if step % 1 == 0:
        print("step: ", step, " loss: ", loss, " a: ", a, " b: ", b)
        plt.show()
        plt.pause(0.01)

plt.show()
plt.pause(1000)
