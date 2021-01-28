import numpy as np
import random as rd
import math
import matplotlib
import matplotlib.pyplot as plt

START = 400
NUM = 10000
ND = []
ND_trained_model = []
A2C_batch_multisamples = []
A2C_on_policy_singlestep = []
A2C_on_policy_multisteps = []
A2C_on_policy_singlestep_16 = []
A2C_on_policy_singlestep_32 = []
A2C_on_policy_singlestep_64 = []
A2C_on_policy_singlestep_128 = []
#A2C_batch = []
#A2C_on_policy = []
#A2C_on_policy_multisamples = []
#A2C_on_policy_multisamples_delayupdate = []

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
with open('A2C-batch-multisamples.txt', 'r+') as f:
    for line in f.readlines():
        try:
            line = float(line)
        except ValueError:
            print('invalid input %s' %line)
        A2C_batch_multisamples.append(line)
with open('A2C-on-policy-singlestep.txt', 'r+') as f:
    for line in f.readlines():
        try:
            line = float(line)
        except ValueError:
            print('invalid input %s' %line)
        A2C_on_policy_singlestep.append(line)
with open('A2C-on-policy-multisteps.txt', 'r+') as f:
    for line in f.readlines():
        try:
            line = float(line)
        except ValueError:
            print('invalid input %s' %line)
        A2C_on_policy_multisteps.append(line)
with open('A2C-on-policy-singlestep-16.txt', 'r+') as f:
    for line in f.readlines():
        try:
            line = float(line)
        except ValueError:
            print('invalid input %s' %line)
        A2C_on_policy_singlestep_16.append(line)
with open('A2C-on-policy-singlestep-32.txt', 'r+') as f:
    for line in f.readlines():
        try:
            line = float(line)
        except ValueError:
            print('invalid input %s' %line)
        A2C_on_policy_singlestep_32.append(line)
with open('A2C-on-policy-singlestep-64.txt', 'r+') as f:
    for line in f.readlines():
        try:
            line = float(line)
        except ValueError:
            print('invalid input %s' %line)
        A2C_on_policy_singlestep_64.append(line)
with open('A2C-on-policy-singlestep-128.txt', 'r+') as f:
    for line in f.readlines():
        try:
            line = float(line)
        except ValueError:
            print('invalid input %s' %line)
        A2C_on_policy_singlestep_128.append(line)
'''
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
with open('A2C-on-policy-multisamples.txt', 'r+') as f:
    for line in f.readlines():
        try:
            line = float(line)
        except ValueError:
            print('invalid input %s' %line)
        A2C_on_policy_multisamples.append(line)
with open('A2C-on-policy-multisamples-delayupdate.txt', 'r+') as f:
    for line in f.readlines():
        try:
            line = float(line)
        except ValueError:
            print('invalid input %s' %line)
        A2C_on_policy_multisamples_delayupdate.append(line)
'''

ND = ND[START:NUM]
ND_trained_model = ND_trained_model[START:NUM]
A2C_batch_multisamples = A2C_batch_multisamples[START:NUM]
A2C_on_policy_singlestep = A2C_on_policy_singlestep[START:NUM]
A2C_on_policy_multisteps = A2C_on_policy_multisteps[START:NUM]
A2C_on_policy_singlestep_16 = A2C_on_policy_singlestep_16[START:NUM]
A2C_on_policy_singlestep_32 = A2C_on_policy_singlestep_32[START:NUM]
A2C_on_policy_singlestep_64 = A2C_on_policy_singlestep_64[START:NUM]
A2C_on_policy_singlestep_128 = A2C_on_policy_singlestep_128[START:NUM]
'''
A2C_batch = A2C_batch[START:NUM]
A2C_on_policy = A2C_on_policy[START:NUM]
A2C_on_policy_multisamples = A2C_on_policy_multisamples[START:NUM]
A2C_on_policy_multisamples_delayupdate = A2C_on_policy_multisamples_delayupdate[START:NUM]
'''

plt.plot(ND, color = 'blue', label = 'ND')
plt.plot(ND_trained_model, color = 'red', label = 'ND_trained_model')
#plt.plot(A2C_batch_multisamples, color = 'orange', label = 'A2C_batch_multisamples')
plt.plot(A2C_on_policy_singlestep, color = 'green', label = 'A2C_on_policy_singlestep')
#plt.plot(A2C_on_policy_multisteps, color = 'grey', label = 'A2C_on_policy_multisteps')
plt.plot(A2C_on_policy_singlestep_16, color = 'orange', label = 'A2C_on_policy_singlestep_16')
plt.plot(A2C_on_policy_singlestep_32, color = 'grey', label = 'A2C_on_policy_singlestep_32')
plt.plot(A2C_on_policy_singlestep_64, color = 'black', label = 'A2C_on_policy_singlestep_64')
plt.plot(A2C_on_policy_singlestep_128, color = 'purple', label = 'A2C_on_policy_singlestep_128')
#plt.plot(A2C_batch, color = 'green', label = 'A2C_batch')
#plt.plot(A2C_on_policy, color = 'grey', label = 'A2C_on_policy')
#plt.plot(A2C_on_policy_multisamples, color = 'black', label = 'A2C_on_policy_multisamples')
#plt.plot(A2C_on_policy_multisamples_delayupdate, color = 'purple', label = 'A2C_on_policy_multisamples_delayupdate')
plt.legend()
plt.xlabel('Steps')
plt.ylabel('Average delivery ratio')
plt.show()


'''
epoch = []
loss = []
with open('ND-init-epoch.txt', 'r+') as f:
    l = 0
    for line in f.readlines():
        try:
            line = float(line) * 10
        except ValueError:
            print('invalid input %s' %line)
        epoch.append(line)
        l += 1
        if l >= 1000:
            break
with open('ND-init-loss.txt', 'r+') as f:
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
            loss.append(sum)
            l = 0
            sum = 0

plt.title('Training loss')
plt.plot(loss, color = 'grey', label = 'Trained-model')
plt.legend()
plt.xlabel('Number of epochs')
plt.ylabel('Training loss')
plt.show()
'''