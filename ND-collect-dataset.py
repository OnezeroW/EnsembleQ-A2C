import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random as rd
import torch.optim as optim
import math
import matplotlib
import matplotlib.pyplot as plt

NUM_OF_LINKS=6
MAX_DEADLINE=10
# LAMBDA = [0.05, 0.07, 0.13]     # arrival rate = 0.5
LAMBDA = [0.01, 0.015, 0.025, 0.01, 0.015, 0.025]
# LAMBDA = [0.008, 0.012, 0.03] # arrival rate = 1
# LAMBDA = [0.45, 0.15, 0.12, 0.1, 0.1, 0.08] # arrival rate = 1, new arrival packets share the same max deadline
# LAMBDA = []
#LAMBDA = 0.01
Arrival = np.zeros((NUM_OF_LINKS, MAX_DEADLINE+1), dtype=np.float)
sumArrival = np.zeros((NUM_OF_LINKS), dtype=np.float)
totalArrival = np.zeros((NUM_OF_LINKS), dtype=np.float)   #TA: total arrival packets from beginning
totalDelivered = np.zeros((NUM_OF_LINKS), dtype=np.float) #TD: total delivered packets
# 1 <= d <= d_max, Buffer[l][0]=0, Buffer[l][d_max+1]=0
Buffer = np.zeros((NUM_OF_LINKS, MAX_DEADLINE+2), dtype=np.float)
totalBuff = np.zeros((NUM_OF_LINKS), dtype=np.float)
Deficit = np.zeros((NUM_OF_LINKS), dtype=np.float)
Action = np.zeros((NUM_OF_LINKS, MAX_DEADLINE+2), dtype=np.int)
sumAction = np.zeros((NUM_OF_LINKS), dtype=np.int)
# e[l]: earliest deadline of link l
e = np.zeros((NUM_OF_LINKS), dtype=np.int)

p_next = np.zeros((NUM_OF_LINKS), dtype=np.float)

N_ACTIONS = NUM_OF_LINKS
N_STATES = NUM_OF_LINKS * (MAX_DEADLINE + 3)    #State s = (Deficit, e, Buffer, p)
INIT_P = [0.8, 0.9, 1.0, 0.8, 0.9, 1.0]
NUM_EPISODE = 500   # the number of episode
LEN_EPISODE = 2000   # the length of each episode
# BATCH_SIZE = 128
# HIDDEN_SIZE = 256    # the dim of hidden layers
# NUM_EPOCH = 30000
# LR = 3e-4                   # learning rate
MEMORY_CAPACITY = NUM_EPISODE * LEN_EPISODE
memory_counter = 0
memory = np.zeros((MEMORY_CAPACITY, N_STATES + N_ACTIONS))
buffer_counter = 0
savedBuffer = np.zeros((MEMORY_CAPACITY, NUM_OF_LINKS, MAX_DEADLINE+2))
savedDeficit = np.zeros((MEMORY_CAPACITY, NUM_OF_LINKS))
savedTotalArrival = np.zeros((MEMORY_CAPACITY, NUM_OF_LINKS))
savedTotalDelivered = np.zeros((MEMORY_CAPACITY, NUM_OF_LINKS))

weight = np.ones((NUM_OF_LINKS), dtype=np.float)

# find the link l which should be active in current time slot
#AMIX_ND algorithm
def AMIX_ND():
    ND_ActiveLink = 0
    # remember: localDeficit = Deficit is wrong!
    localDeficit = Deficit.copy()
    ND = []
    # prob[l]: probability of link l to be active
    prob = np.zeros((NUM_OF_LINKS), dtype=np.float)
    # only links with nonempty buffer should be taken into account
    for l in range(NUM_OF_LINKS):
        #totalBuff = 0
        if totalBuff[l] == 0:
            localDeficit[l] = 0

    while True:
        maxDeficit = 0
        maxDificitLink = 0
        # find the link with maximal deficit
        for l in range(NUM_OF_LINKS):
            if maxDeficit < localDeficit[l]:
                maxDeficit = localDeficit[l]
                maxDificitLink = l
        if maxDeficit > 0:
            # find all the links with the same maximal deficit (nonzero), then choose the one with smallest e
            for l in range(NUM_OF_LINKS):
                if localDeficit[l] == maxDeficit:
                    if e[l] < e[maxDificitLink]:
                        maxDificitLink = l
            ND.append(maxDificitLink)
            for l in range(NUM_OF_LINKS):
                if e[l] >= e[maxDificitLink]:
                    localDeficit[l] = 0 # delete the dominated links
        else:
            break

    k = len(ND)
    # if all deficit=0, then return the link with smallest e
    if k == 0:
        # if all buffers are empty, then no link should be active
        if np.min(e) == MAX_DEADLINE+1: # e[l] initialized as MAX_DEADLINE+1
            ND_ActiveLink = -1
            prob = np.zeros((NUM_OF_LINKS), dtype=np.float)
        else:
            ND_ActiveLink = np.argmin(e)
            prob[ND_ActiveLink] = 1
    # if one link dominates all other links, then active_prob = 1
    elif k == 1:
        ND_ActiveLink = ND[0]
        prob[ND_ActiveLink] = 1
    else:
        r = 1
        for i in range(k-1):
            prob[ND[i]] = min(r, 1 - Deficit[ND[i+1]] / Deficit[ND[i]])
            r = r - prob[ND[i]]
        prob[ND[k-1]] = r
        start = 0
        randnum = rd.randint(1, 1000000)
        for i in range(k):
            start = start + 1000000*prob[ND[i]]
            if randnum <= start:
                ND_ActiveLink = ND[i]
                break
    return ND_ActiveLink, prob

def generateState(Deficit, e, Buffer, p):  #generate 1-D state
    #arr1 = np.array(Buffer)[:, 1:MAX_DEADLINE+1]
    arr1 = np.array(Deficit)
    arr2 = np.array(e)
    arr3 = Buffer.flatten()
    arr4 = np.array(p)
    result = np.concatenate((arr1, arr2, arr3, arr4))
    result = torch.FloatTensor(result)
    return result

def store_transition(s, action_probs):
    global memory
    global memory_counter
    transition = np.hstack((s, action_probs))
    memory[memory_counter, :] = transition
    memory_counter += 1

# i.i.d. with Poisson distribution
def ARR_POISSON(lam):
    Arrival.fill(0)
    for l in range(NUM_OF_LINKS):
        for d in range(1, MAX_DEADLINE+1):
            Arrival[l][d] = np.random.poisson(lam[l])

result = torch.zeros((LEN_EPISODE, NUM_EPISODE))
print('Collecting experience...')
for i_episode in range(NUM_EPISODE):
    # initialization
    if buffer_counter == 0:
        Buffer.fill(0)
        Deficit.fill(0)
        totalArrival.fill(0)
        totalDelivered.fill(0)
    else:
        i_buffer = np.random.randint(0,buffer_counter)
        Buffer = savedBuffer[i_buffer].copy()
        Deficit = savedDeficit[i_buffer].copy()
        totalArrival = savedTotalArrival[i_buffer].copy()
        totalDelivered = savedTotalDelivered[i_buffer].copy()
    # Buffer.fill(0)
    # Deficit.fill(0)
    # totalArrival.fill(0)
    # totalDelivered.fill(0)
    # p_next = np.zeros((NUM_OF_LINKS), dtype=np.float)   #delivery ratio

    Action.fill(0)
    sumAction.fill(0)
    e.fill(MAX_DEADLINE+1)

    for index in range(LEN_EPISODE):
        # new arrival packets
        ARR_POISSON(LAMBDA)
        # total arrival num on link l
        sumArrival.fill(0)
        for l in range(NUM_OF_LINKS):
            for d in range(1, MAX_DEADLINE+1):
                sumArrival[l] = sumArrival[l] + Arrival[l][d]
        # update total arrival packets from beginning
        for l in range(NUM_OF_LINKS):
            totalArrival[l] += sumArrival[l]
            totalDelivered[l] += sumAction[l]
        # update buffer
        for l in range(NUM_OF_LINKS):
            for d in range(1, MAX_DEADLINE+1):
                Buffer[l][d] = max(Buffer[l][d+1] + Arrival[l][d] - Action[l][d+1], 0)
        # update totalBuff
        totalBuff.fill(0)
        for l in range(NUM_OF_LINKS):
            for d in range(1, MAX_DEADLINE+1):
                totalBuff[l] = totalBuff[l] + Buffer[l][d]
        # update deficit
        for l in range(NUM_OF_LINKS):
            # Deficit[l] = max(Deficit[l] + sumArrival[l]*INIT_P - sumAction[l], 0)
            Deficit[l] = max(Deficit[l] + sumArrival[l]*INIT_P[l] - sumAction[l], 0)
        # update the earliest deadline on link l
        e.fill(MAX_DEADLINE+1)      #initial earliest deadline should be MAX_DEADLINE+1
        for l in range(NUM_OF_LINKS):
            for d in range(1, MAX_DEADLINE+1):
                if Buffer[l][d] > 0:
                    e[l] = d
                    break

        # algorithm invoked here
        #currentActiveLink = LDF()
        currentActiveLink, action_probs = AMIX_ND()

        for l in range(NUM_OF_LINKS):
            if totalArrival[l] == 0:
                p_next[l] = 0
            else:
                p_next[l] = totalDelivered[l] / totalArrival[l] #next delivery ratio

        result[index][i_episode] = round(np.min(p_next), 6)

        # current state s
        s = generateState(Deficit, e, Buffer[:,1:MAX_DEADLINE+1], p_next)

        store_transition(s, action_probs)
        savedBuffer[buffer_counter, :] = Buffer
        savedDeficit[buffer_counter, :] = Deficit
        savedTotalArrival[buffer_counter, :] = totalArrival
        savedTotalDelivered[buffer_counter, :] = totalDelivered
        buffer_counter += 1
        # update Action
        Action.fill(0)

        if currentActiveLink != -1:
            elstDeadline = e[currentActiveLink]
            Action[currentActiveLink][elstDeadline] = 1

        # total departure num on link l
        sumAction.fill(0)
        for l in range(NUM_OF_LINKS):
            for d in range(1, MAX_DEADLINE+1):
                sumAction[l] = sumAction[l] + Action[l][d]

np.save('dataset-500-2k.npy', memory)     # save dataset

result = result.sum(-1, keepdim = True)
result = result / NUM_EPISODE
res = result.detach().numpy()

print('Collecting experience complete. Ready to train model...')