import numpy as np
import random as rd
import math
import matplotlib
import matplotlib.pyplot as plt

NUM = 1000
ND = []
ND_trained_model = []
A2C_batch = []
A2C_on_policy = []
A2C_batch_multisamples = []

'''
with open('ALG-init-y.txt', 'r+') as f:
    for line in f.readlines():
        try:
            line = float(line)
        except ValueError:
            print('invalid input %s' %line)
        ALG_init_y.append(line)
'''

with open('ND-traj.txt', 'r+') as f:
    for line in f.readlines():
        try:
            line = float(line)
        except ValueError:
            print('invalid input %s' %line)
        ND.append(line)
with open('trainedmodel-traj.txt', 'r+') as f:
    for line in f.readlines():
        try:
            line = float(line)
        except ValueError:
            print('invalid input %s' %line)
        ND_trained_model.append(line)
with open('A2C-batch.txt', 'r+') as f:
    for line in f.readlines():
        try:
            line = float(line)
        except ValueError:
            print('invalid input %s' %line)
        A2C_batch.append(line)
with open('A2C-on-policy.txt', 'r+') as f:
    for line in f.readlines():
        try:
            line = float(line)
        except ValueError:
            print('invalid input %s' %line)
        A2C_on_policy.append(line)
with open('A2C-batch-multisamples.txt', 'r+') as f:
    for line in f.readlines():
        try:
            line = float(line)
        except ValueError:
            print('invalid input %s' %line)
        A2C_batch_multisamples.append(line)
ND = ND[:NUM]
ND_trained_model = ND_trained_model[:NUM]
A2C_batch = A2C_batch[:NUM]
A2C_on_policy = A2C_on_policy[:NUM]
A2C_batch_multisamples = A2C_batch_multisamples[:NUM]
#plt.title('Simulation 9')
#plt.plot(ALG_init_x, ALG_init_y, color = 'red', label = 'AMIX-ND')
#plt.plot(trainedmodel_x, trainedmodel_y, color = 'grey', label = 'Trained-model')
#plt.plot(ALG_x, ALG_y, color = 'green', label = 'ALG-pretrained')
#plt.plot(ALG_xx, ALG_yy, color = 'blue', label = 'ALG-random')
#plt.plot(ALG_batch_x, ALG_batch_y, color = 'blue', label = 'ALG-batch')
plt.plot(ND, color = 'blue', label = 'ND')
plt.plot(ND_trained_model, color = 'red', label = 'ND_trained_model')
plt.plot(A2C_batch, color = 'green', label = 'A2C_batch')
plt.plot(A2C_on_policy, color = 'grey', label = 'A2C_on_policy')
plt.plot(A2C_batch, color = 'orange', label = 'A2C_batch_multisamples')
plt.legend()
plt.xlabel('Steps')
plt.ylabel('Average delivery ratio')
plt.show()


'''
epoch = []
loss = []
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
'''