import numpy as np
import random as rd
import math
import matplotlib
import matplotlib.pyplot as plt


ALG_init_x = []
ALG_init_y = []
ALG_x = []
ALG_y = []
ALG_xx = []
ALG_yy = []
ALG_batch_x = []
ALG_batch_y = []
trainedmodel_x = []
trainedmodel_y = []
SAC_x = []
SAC_y = []
A2C_x = []
A2C_y = []

'''
with open('ALG-init-x.txt', 'r+') as f:
    for line in f.readlines():
        try:
            line = float(line)
        except ValueError:
            print('invalid input %s' %line)
        ALG_init_x.append(line)
with open('ALG-init-y.txt', 'r+') as f:
    for line in f.readlines():
        try:
            line = float(line)
        except ValueError:
            print('invalid input %s' %line)
        ALG_init_y.append(line)

with open('trainedmodel-x.txt', 'r+') as f:
    for line in f.readlines():
        try:
            line = float(line)
        except ValueError:
            print('invalid input %s' %line)
        trainedmodel_x.append(line)
with open('trainedmodel-y.txt', 'r+') as f:
    for line in f.readlines():
        try:
            line = float(line)
        except ValueError:
            print('invalid input %s' %line)
        trainedmodel_y.append(line)

with open('ALG-x.txt', 'r+') as f:
    for line in f.readlines():
        try:
            line = float(line)
        except ValueError:
            print('invalid input %s' %line)
        ALG_x.append(line)
with open('ALG-y.txt', 'r+') as f:
    for line in f.readlines():
        try:
            line = float(line)
        except ValueError:
            print('invalid input %s' %line)
        ALG_y.append(line)
with open('ALG-xx.txt', 'r+') as f:
    for line in f.readlines():
        try:
            line = float(line)
        except ValueError:
            print('invalid input %s' %line)
        ALG_xx.append(line)
with open('ALG-yy.txt', 'r+') as f:
    for line in f.readlines():
        try:
            line = float(line)
        except ValueError:
            print('invalid input %s' %line)
        ALG_yy.append(line)

with open('ALG-batch-x.txt', 'r+') as f:
    for line in f.readlines():
        try:
            line = float(line)
        except ValueError:
            print('invalid input %s' %line)
        ALG_batch_x.append(line)
with open('ALG-batch-y.txt', 'r+') as f:
    for line in f.readlines():
        try:
            line = float(line)
        except ValueError:
            print('invalid input %s' %line)
        ALG_batch_y.append(line)

with open('SAC-x.txt', 'r+') as f:
    for line in f.readlines():
        try:
            line = float(line)
        except ValueError:
            print('invalid input %s' %line)
        SAC_x.append(line)
with open('SAC-y.txt', 'r+') as f:
    for line in f.readlines():
        try:
            line = float(line)
        except ValueError:
            print('invalid input %s' %line)
        SAC_y.append(line)
'''


'''
with open('A2C-batch.txt', 'r+') as f:
    for line in f.readlines():
        try:
            line = float(line)
        except ValueError:
            print('invalid input %s' %line)
        A2C_y.append(line)
A2C_y = A2C_y[:1000]
#plt.title('Simulation 9')
#plt.plot(ALG_init_x, ALG_init_y, color = 'red', label = 'AMIX-ND')
#plt.plot(trainedmodel_x, trainedmodel_y, color = 'grey', label = 'Trained-model')
#plt.plot(ALG_x, ALG_y, color = 'green', label = 'ALG-pretrained')
#plt.plot(ALG_xx, ALG_yy, color = 'blue', label = 'ALG-random')
#plt.plot(ALG_batch_x, ALG_batch_y, color = 'blue', label = 'ALG-batch')
plt.plot(A2C_y, color = 'blue', label = 'A2C-batch')
plt.legend()
plt.xlabel('Steps')
plt.ylabel('Average delivery ratio')
plt.show()


'''
ALG_init_epoch = []
ALG_init_loss = []
with open('ALG-init-epoch.txt', 'r+') as f:
    l = 0
    for line in f.readlines():
        try:
            line = float(line) * 10
        except ValueError:
            print('invalid input %s' %line)
        ALG_init_epoch.append(line)
        l += 1
        if l >= 1000:
            break
with open('ALG-init-loss.txt', 'r+') as f:
    l = 0
    sum = 0
    for line in f.readlines():
        try:
            line = float(line)
        except ValueError:
            print('invalid input %s' %line)
        l += 1
        sum += line
        if l == 10:
            sum /= 10
            ALG_init_loss.append(sum)
            l = 0
            sum = 0

plt.title('Training loss')
plt.plot(ALG_init_epoch, ALG_init_loss, color = 'red', label = 'Trained-model')
plt.legend()
plt.xlabel('Number of epochs')
plt.ylabel('Training loss')
plt.show()
