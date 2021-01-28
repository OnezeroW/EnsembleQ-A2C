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

NUM_OF_LINKS=10
MAX_DEADLINE=10
#LAMBDA = [0.01, 0.015, 0.02]
LAMBDA = 0.015
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

N_ACTIONS = NUM_OF_LINKS
#N_STATES = NUM_OF_LINKS * (MAX_DEADLINE + 3)    #State s = (Buffer, Deficit, TA, TD)
N_STATES = NUM_OF_LINKS * 5    #State s = (Deficit, e, TB, TA, TD)
INIT_P = 0.75    #initial delivery ratio p0
NUM_EPISODE = 100  # the number of episode
LEN_EPISODE = 10000   # the length of each episodex
BATCH_SIZE = 128
HIDDEN_SIZE = 64    # the dim of hidden layers
NUM_EPOCH = 10000
LR = 3e-4                   # learning rate
MEMORY_CAPACITY = NUM_EPISODE * LEN_EPISODE
memory_counter = 0
memory = np.zeros((MEMORY_CAPACITY, N_STATES + N_ACTIONS))

#weight = np.random.random(NUM_OF_LINKS)   #weight of each link
weight = np.ones((NUM_OF_LINKS), dtype=np.float)
#weight = np.array([0.05, 0.3, 0.4, 0.1, 0.15])


class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(N_STATES, HIDDEN_SIZE)
        self.layer1.weight.data.normal_(0, 0.02)
        self.layer2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.layer2.weight.data.normal_(0, 0.02)
        self.action = nn.Linear(HIDDEN_SIZE, N_ACTIONS)
        self.action.weight.data.normal_(0, 0.02)
        
    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.action(x)
        action_probs = F.softmax(x, dim=-1)
        return action_probs

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
        randnum = rd.randint(1, 10000)
        for i in range(k):
            start = start + 10000*prob[ND[i]]
            if randnum <= start:
                ND_ActiveLink = ND[i]
                break
    return ND_ActiveLink, prob

def generateState(Deficit, e, totalBuff, totalArrival, totalDelivered):  #generate 1-D state
    #arr1 = np.array(Buffer)[:, 1:MAX_DEADLINE+1]
    arr1 = np.array(Deficit)
    arr2 = np.array(e)
    arr3 = np.array(totalBuff)
    arr4 = np.array(totalArrival)
    arr5 = np.array(totalDelivered)
    result = np.concatenate((arr1, arr2, arr3, arr4, arr5))
    result = torch.FloatTensor(result)
    return result

def store_transition(s, action_probs):
    global memory
    global memory_counter
    #print('mc=', memory_counter, ', aps=', action_probs)
    transition = np.hstack((s, action_probs))
    memory[memory_counter, :] = transition
    memory_counter += 1

# i.i.d. with Poisson distribution
def ARR_POISSON(lam):
    Arrival.fill(0)
    for l in range(NUM_OF_LINKS):
        for d in range(1, MAX_DEADLINE+1):
            Arrival[l][d] = np.random.poisson(lam)
            #Arrival[l][d] = np.random.poisson(lam[l%3])

# non-i.i.d., periodic traffic pattern
def ARR_PERIODIC(index):
    global Arrival
    Arrival.fill(0)
    if index % 4 == 0:
        # pattern A
        Arrival[0][1] = 1
        Arrival[1][2] = 1
    elif index % 4 == 2:
        # pattern B
        Arrival[0][2] = 1
        Arrival[1][1] = 1
    else:
        pass
'''
        # pattern A
        Arrival[0][1] = 1
        Arrival[1][2] = 1
        # pattern B
        Arrival[0][2] = 1
        Arrival[1][1] = 1
        # pattern C
        Arrival[0][1] = 1
        Arrival[1][2] = 1
        Arrival[1][3] = 1
'''


xval = []
yval = []
result = torch.zeros((LEN_EPISODE, NUM_EPISODE))
print('Collecting experience...')
for i_episode in range(NUM_EPISODE):
    # initialization
    Buffer.fill(0)
    Deficit.fill(0)
    totalArrival.fill(0)    #mark
    '''
    Deficit = np.random.rand(NUM_OF_LINKS) * 10
    for l in range(NUM_OF_LINKS):
        for d in range(MAX_DEADLINE+2):
            Buffer[l][d] = np.random.randint(10)

    totalArrival.fill(1)
    for l in range(NUM_OF_LINKS):
        for d in range(1, MAX_DEADLINE+1):
            totalArrival[l] += Buffer[l][d]
    '''
    totalDelivered.fill(0)
    Action.fill(0)
    sumAction.fill(0)
    e.fill(MAX_DEADLINE+1)
    p_next = np.zeros((NUM_OF_LINKS), dtype=np.float)   #delivery ratio

    for index in range(LEN_EPISODE):
        # new arrival packets

        #ARR_UNIFROM(low = 0, high = 0.4)
        ARR_POISSON(LAMBDA)
        #ARR_PERIODIC(index)

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
            Deficit[l] = max(Deficit[l] + sumArrival[l]*INIT_P - sumAction[l], 0)
            '''
            randx = rd.randint(1, 10000)
            if randx < 10000 * p:
                Deficit[l] = max(Deficit[l] + sumArrival[l] - sumAction[l], 0)
            else:
                Deficit[l] = max(Deficit[l] - sumAction[l], 0)
            '''
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

        # current state s
        s = generateState(Deficit, e, totalBuff, totalArrival, totalDelivered)
        store_transition(s, action_probs)
        #print(index, '\n', Deficit, '\n', Arrival[:,1:MAX_DEADLINE+1], '\n', Buffer[:,1:MAX_DEADLINE+1], '\n', action_probs)

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

        for l in range(NUM_OF_LINKS):
            if totalArrival[l] == 0:
                p_next[l] = 0
            else:
                p_next[l] = totalDelivered[l] / totalArrival[l] #next delivery ratio
        rr = 0
        sumWeight = 0
        for l in range(NUM_OF_LINKS):
            rr += weight[l] * p_next[l]
            sumWeight += weight[l]
        rr /= sumWeight
        result[index][i_episode] = round(rr, 3)
    xval.append(i_episode)
    yval.append(round(rr, 3))
'''
with open('ALG-init-x.txt', 'a+') as f:
    for x in xval:
        f.write(str(x)+'\n')
with open('ALG-init-y.txt', 'a+') as f:
    for y in yval:
        f.write(str(y)+'\n')
'''
result = result.sum(-1, keepdim = True)
result = result / NUM_EPISODE
res = result.detach().numpy()
#print(res)
with open('ND-traj.txt', 'a+') as f:
    for x in res:
        f.write(str(x.item())+'\n')

epochnum = range(NUM_EPOCH)
zval = []
print('Collecting experience complete. Training model...')
initialNet = Net()
optimizer = optim.Adam(initialNet.parameters(), lr = LR)
criterion = nn.MSELoss()
# sample batch transitions
for epoch in range(NUM_EPOCH):
    sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
    b_memory = memory[sample_index, :]
    b_s = Variable(torch.FloatTensor(b_memory[:, :N_STATES]))
    b_ap = Variable(torch.FloatTensor(b_memory[:, -N_ACTIONS:]))
    b_net = initialNet(b_s)
    loss = criterion(b_net, b_ap)
    print('epoch=', epoch, ', loss=', loss.item())
    zval.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

with open('ND-init-epoch.txt', 'a+') as f:
    for ep in epochnum:
        f.write(str(ep)+'\n')
with open('ND-init-loss.txt', 'a+') as f:
    for z in zval:
        f.write(str(z)+'\n')
torch.save(initialNet.state_dict(), 'params.pkl')
print('Training model complete. Model saved.')