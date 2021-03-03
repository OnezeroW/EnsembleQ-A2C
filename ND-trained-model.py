import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random as rd
import torch.optim as optim

# Hyper Parameters
NUM_OF_LINKS=6
MAX_DEADLINE=10
LAMBDA = [0.05, 0.07, 0.13]     # arrival rate = 0.5
# LAMBDA = [0.01, 0.015, 0.025]
# LAMBDA = [0.005, 0.01, 0.015]
# LAMBDA = 0.01
#EPSILON = 0.05
# 1 <= d <= d_max, Buffer[l][0]=0, Buffer[l][d_max+1]=0
Buffer = np.zeros((NUM_OF_LINKS, MAX_DEADLINE+2), dtype=np.float)
Deficit = np.zeros((NUM_OF_LINKS), dtype=np.float)
Arrival = np.zeros((NUM_OF_LINKS, MAX_DEADLINE+1), dtype=np.float)
sumArrival = np.zeros((NUM_OF_LINKS), dtype=np.float)     #total arrival packets at current time slot
totalArrival = np.zeros((NUM_OF_LINKS), dtype=np.float)   #TA: total arrival packets from beginning
totalDelivered = np.zeros((NUM_OF_LINKS), dtype=np.float) #TD: total delivered packets
Action = np.zeros((NUM_OF_LINKS, MAX_DEADLINE+2), dtype=np.float)

totalBuff = np.zeros((NUM_OF_LINKS), dtype=np.float)
# e[l]: earliest deadline of link l
e = np.zeros((NUM_OF_LINKS), dtype=np.int)

#weight = np.random.random(NUM_OF_LINKS)   #weight of each link
weight = np.ones((NUM_OF_LINKS), dtype=np.float)
# weight = np.array([0.2, 0.15, 0.15, 0.2, 0.15, 0.15])

p_current = np.zeros((NUM_OF_LINKS), dtype=np.float)    #current delivery ratio
p_next = np.zeros((NUM_OF_LINKS), dtype=np.float)   #next delivery ratio

N_ACTIONS = NUM_OF_LINKS
N_STATES = NUM_OF_LINKS * 5    #State s = (Deficit, e, TB, TA, TD)
HIDDEN_SIZE = 64    # the dim of hidden layers
INIT_P = 0.75    #initial delivery ratio p0
NUM_EPISODE = 10  # the number of episode
LEN_EPISODE = 100000   # the length of each episode

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

# i.i.d. with Uniform distribution
def ARR_UNIFROM(low, high):
    global Arrival
    Arrival.fill(0)
    for l in range(NUM_OF_LINKS):
        for d in range(1, MAX_DEADLINE+1):
            Arrival[l][d] = rd.uniform(low, high)

# i.i.d. with Poisson distribution
def ARR_POISSON(lam):
    global Arrival
    Arrival.fill(0)
    for l in range(NUM_OF_LINKS):
        Arrival[l][MAX_DEADLINE] = np.random.poisson(lam[l%3])
        # for d in range(1, MAX_DEADLINE+1):
        #     # Arrival[l][d] = np.random.poisson(lam)
        #     Arrival[l][d] = np.random.poisson(lam[l%3])

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

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(N_STATES, HIDDEN_SIZE)
        # self.layer1.weight.data.normal_(0, 0.02)
        self.layer2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        # self.layer2.weight.data.normal_(0, 0.02)
        self.action = nn.Linear(HIDDEN_SIZE, N_ACTIONS)
        # self.action.weight.data.normal_(0, 0.02)
        
    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.action(x)
        action_probs = F.softmax(x, dim=-1)
        return action_probs

net = Net()
net.load_state_dict(torch.load('params.pkl'))     # load initial NN model trained from base policy
print('Model loaded! Start running...')

# xval = []
# yval = []
result = torch.zeros((LEN_EPISODE, NUM_EPISODE))
#print('\nCollecting experience...')
for i_episode in range(NUM_EPISODE):
    #initialize state s
    Buffer.fill(0)
    Deficit.fill(0)
    totalArrival.fill(0)    #mark
    totalDelivered.fill(0)
    totalBuff.fill(0)
    e.fill(MAX_DEADLINE + 1) 

    # current state s
    s = generateState(Deficit, e, totalBuff, totalArrival, totalDelivered)
    p_current.fill(0)

    ep_r = 0    #total reward of one episode
    for len in range(LEN_EPISODE):  #length of each episode
        dist = net(s)
        '''
        eps = rd.random()   # epsilon-greedy
        if eps < EPSILON:
            a = rd.randint(0, NUM_OF_LINKS-1)
        else:
            a = torch.multinomial(dist, 1).item()
        '''
        a = torch.multinomial(dist, 1).item()

        # take action
        
        #consider arrival packets here
        #ARR_UNIFROM(0,1)
        ARR_POISSON(LAMBDA)
        #ARR_PERIODIC(len)
        # update total arrival packets at current time slot
        sumArrival.fill(0)
        for l in range(NUM_OF_LINKS):
            for d in range(1, MAX_DEADLINE+1):
                sumArrival[l] += Arrival[l][d]
        # update total arrival packets from beginning
        for l in range(NUM_OF_LINKS):
            totalArrival[l] += sumArrival[l]
        # update buffer
        Action.fill(0)
        for d in range(1, MAX_DEADLINE+1):
            if Buffer[a][d] > 0:
                Action[a][d] = 1
                # update total delivered packets from beginning
                totalDelivered[a] += 1
                break

        for l in range(NUM_OF_LINKS):
            for d in range(1, MAX_DEADLINE+1):
                Buffer[l][d] = max(Buffer[l][d+1] + Arrival[l][d] - Action[l][d+1], 0)
        
        # update totalBuff
        totalBuff.fill(0)
        for l in range(NUM_OF_LINKS):
            for d in range(1, MAX_DEADLINE+1):
                totalBuff[l] = totalBuff[l] + Buffer[l][d]
        # update the earliest deadline on link l
        e.fill(MAX_DEADLINE + 1)      #initial earliest deadline should be MAX_DEADLINE+1
        for l in range(NUM_OF_LINKS):
            for d in range(1, MAX_DEADLINE+1):
                if Buffer[l][d] > 0:
                    e[l] = d
                    break
        
        # update deficit
        for l in range(NUM_OF_LINKS):
            if l == a:
                Deficit[l] = max(Deficit[l] + sumArrival[l] * INIT_P - 1, 0)
            else:
                Deficit[l] = max(Deficit[l] + sumArrival[l] * INIT_P, 0)

        s_ = generateState(Deficit, e, totalBuff, totalArrival, totalDelivered)   # next state s_

        for l in range(NUM_OF_LINKS):
            if totalArrival[l] == 0:
                p_next[l] = 0
            else:
                p_next[l] = totalDelivered[l] / totalArrival[l] #next delivery ratio
        # r = 0
        # sumWeight = 0
        # for l in range(NUM_OF_LINKS):
        #     r += weight[l] * (p_next[l] - p_current[l]) #reward calculation, R(t+1)=\sum c_l*[p_l(t+1)-p_l(t)]
        #     sumWeight += weight[l]
        # r /= sumWeight   # reward r
        # p_current = p_next.copy()   #current delivery ratio

        r = (p_next[a] - p_current[a]) / NUM_OF_LINKS

        print('step: ', len, ', action: ', a, ', reward: ', r, p_current[a], p_next[a])

        p_current[a] = p_next[a]

        ep_r += r
        s = s_
        result[len][i_episode] = round(ep_r, 3)

        # if len == LEN_EPISODE-1:
        #     ###print('Iteration number: ', i_episode,
        #             ###'| Total reward: ', round(ep_r, 3))
        #     xval.append(i_episode)
        #     yval.append(round(ep_r, 3))

result = result.sum(-1, keepdim = True)
result = result / NUM_EPISODE
res = result.detach().numpy()
#print(res)

'''
with open('trainedmodel-x.txt', 'a+') as f:
    for x in xval:
        f.write(str(x)+'\n')
with open('trainedmodel-y.txt', 'a+') as f:
    for y in yval:
        f.write(str(y)+'\n')
'''
with open('trainedmodel-traj.txt', 'a+') as f:
    for x in res:
        f.write(str(x.item())+'\n')