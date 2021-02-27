import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random as rd
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Hyper Parameters
NUM_OF_LINKS=10
MAX_DEADLINE=10
NUM_OF_AGENT=32
LAMBDA = [0.005, 0.01, 0.015]
# LAMBDA = 0.01
# 1 <= d <= d_max, Buffer[l][0]=0, Buffer[l][d_max+1]=0
Buffer = np.zeros((NUM_OF_AGENT, NUM_OF_LINKS, MAX_DEADLINE+2), dtype=np.float)
Deficit = np.zeros((NUM_OF_AGENT, NUM_OF_LINKS), dtype=np.float)
Arrival = np.zeros((NUM_OF_AGENT, NUM_OF_LINKS, MAX_DEADLINE+1), dtype=np.float)
sumArrival = np.zeros((NUM_OF_AGENT, NUM_OF_LINKS), dtype=np.float)     #total arrival packets at current time slot
totalArrival = np.zeros((NUM_OF_AGENT, NUM_OF_LINKS), dtype=np.float)   #TA: total arrival packets from beginning
totalDelivered = np.zeros((NUM_OF_AGENT, NUM_OF_LINKS), dtype=np.float) #TD: total delivered packets
Action = np.zeros((NUM_OF_AGENT, NUM_OF_LINKS, MAX_DEADLINE+2), dtype=np.float)

totalBuff = np.zeros((NUM_OF_AGENT, NUM_OF_LINKS), dtype=np.float)
# e[l]: earliest deadline of link l
e = np.zeros((NUM_OF_AGENT, NUM_OF_LINKS), dtype=np.int)


#weight = np.random.random(NUM_OF_LINKS)   #weight of each link
weight = np.ones((NUM_OF_LINKS), dtype=np.float)
#weight = np.array([0.05, 0.3, 0.4, 0.1, 0.15])
p_current = np.zeros((NUM_OF_AGENT, NUM_OF_LINKS), dtype=np.float)    #current delivery ratio
p_next = np.zeros((NUM_OF_AGENT, NUM_OF_LINKS), dtype=np.float)   #next delivery ratio


N_ACTIONS = NUM_OF_LINKS
N_STATES = NUM_OF_LINKS * 5
INIT_P = 0.75    #initial delivery ratio p0
NUM_EPISODE = 10  # the number of episode
LEN_EPISODE = 10000   # the length of each episode
BATCH_SIZE = 128
HIDDEN_SIZE = 64    # the dim of hidden layers
LR = 3e-4                   # learning rate
MEMORY_CAPACITY = NUM_OF_AGENT
memory_counter = 0
memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 3))

s = torch.zeros(NUM_OF_AGENT, N_STATES)
s_ = torch.zeros(NUM_OF_AGENT, N_STATES)

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

# i.i.d. with Poisson distribution
def ARR_POISSON(lam, agent):
    global Arrival
    Arrival[agent].fill(0)
    for l in range(NUM_OF_LINKS):
        for d in range(1, MAX_DEADLINE+1):
            # Arrival[agent][l][d] = np.random.poisson(lam)
            Arrival[agent][l][d] = np.random.poisson(lam[l%3])

class Actor(nn.Module):
    def __init__(self, ):
        super(Actor, self).__init__()
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

class Critic(nn.Module):
    def __init__(self, ):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(N_STATES, HIDDEN_SIZE)
        self.layer1.weight.data.normal_(0, 0.02)
        self.layer2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.layer2.weight.data.normal_(0, 0.02)
        self.value = nn.Linear(HIDDEN_SIZE, 1)
        self.value.weight.data.normal_(0, 0.02)
        
    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        state_value = self.value(x)
        return state_value

def store_transition(s, a, r, s_, log_prob):
    global memory
    global memory_counter
    transition = np.hstack((s, [a, r], s_, [log_prob]))
    # replace the old memory with new memory
    index = memory_counter % MEMORY_CAPACITY
    memory[index, :] = transition
    memory_counter += 1

actor, critic = Actor(), Critic()
actor.load_state_dict(torch.load('params.pkl'))     # load initial NN model trained from base policy
optimizerA = optim.Adam(actor.parameters(), lr=LR)
optimizerC = optim.Adam(critic.parameters(), lr=LR)

critic_criterion = nn.MSELoss()

result = torch.zeros((LEN_EPISODE, NUM_OF_AGENT))        # store the reward of each trajectory
print('\nCollecting experience...')
#for i_episode in range(NUM_EPISODE):
#initialize state s
Buffer.fill(0)
Deficit.fill(0)
totalArrival.fill(0)    #mark
totalDelivered.fill(0)
totalBuff.fill(0)
e.fill(MAX_DEADLINE+1)
p_current.fill(0)

# current state s
for agent in range(NUM_OF_AGENT):
    s[agent] = generateState(Deficit[agent], e[agent], totalBuff[agent], totalArrival[agent], totalDelivered[agent])

#ep_r = 0    #total reward of one episode
ep_r = np.zeros((NUM_OF_AGENT), dtype=np.float)

for len in range(LEN_EPISODE):  #length of each episode
    for agent in range(NUM_OF_AGENT):
    #for num in range(MEMORY_CAPACITY):   #generate multiple samples during each time slot
        dist, value = actor(s[agent]), critic(s[agent])
        a = torch.multinomial(dist, 1).item()
        log_prob = torch.log(dist[a])

        ARR_POISSON(LAMBDA, agent)
        # update total arrival packets at current time slot
        sumArrival.fill(0)
        for l in range(NUM_OF_LINKS):
            for d in range(1, MAX_DEADLINE+1):
                sumArrival[agent][l] += Arrival[agent][l][d]
        # update total arrival packets from beginning
        for l in range(NUM_OF_LINKS):
            totalArrival[agent][l] += sumArrival[agent][l]
        # update buffer
        Action.fill(0)
        for d in range(1, MAX_DEADLINE+1):
            if Buffer[agent][a][d] > 0:
                Action[agent][a][d] = 1
                # update total delivered packets from beginning
                totalDelivered[agent][a] += 1
                break

        for l in range(NUM_OF_LINKS):
            for d in range(1, MAX_DEADLINE+1):
                Buffer[agent][l][d] = max(Buffer[agent][l][d+1] + Arrival[agent][l][d] - Action[agent][l][d+1], 0)

        # update totalBuff
        totalBuff.fill(0)
        for l in range(NUM_OF_LINKS):
            for d in range(1, MAX_DEADLINE+1):
                totalBuff[agent][l] = totalBuff[agent][l] + Buffer[agent][l][d]
        # update the earliest deadline on link l
        e.fill(MAX_DEADLINE+1)      #initial earliest deadline should be MAX_DEADLINE+1
        for l in range(NUM_OF_LINKS):
            for d in range(1, MAX_DEADLINE+1):
                if Buffer[agent][l][d] > 0:
                    e[agent][l] = d
                    break

        # update deficit
        for l in range(NUM_OF_LINKS):
            if l == a:
                Deficit[agent][l] = max(Deficit[agent][l] + sumArrival[agent][l] * INIT_P - 1, 0)
            else:
                Deficit[agent][l] = max(Deficit[agent][l] + sumArrival[agent][l] * INIT_P, 0)

        s_[agent] = generateState(Deficit[agent], e[agent], totalBuff[agent], totalArrival[agent], totalDelivered[agent])   # next state s_

        for l in range(NUM_OF_LINKS):
            if totalArrival[agent][l] == 0:
                p_next[agent][l] = 0
            else:
                p_next[agent][l] = totalDelivered[agent][l] / totalArrival[agent][l] #next delivery ratio
        r = 0
        sumWeight = 0
        for l in range(NUM_OF_LINKS):
            r += weight[l] * (p_next[agent][l] - p_current[agent][l]) #reward calculation, R(t+1)=\sum c_l*[p_l(t+1)-p_l(t)]
            sumWeight += weight[l]
        r /= sumWeight   # reward r
        store_transition(s[agent], a, r, s_[agent], log_prob)

        p_current[agent] = p_next[agent].copy()   #current delivery ratio
        ep_r[agent] += r
        s[agent] = s_[agent]
        result[len][[agent]] = round(ep_r[agent], 3)     # current total reward

    # sample batch transitions
    #sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
    b_memory = memory[:, :]
    b_s = Variable(torch.FloatTensor(b_memory[:, :N_STATES]))
    b_a = Variable(torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int)))
    b_r = Variable(torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2]))
    b_s_ = Variable(torch.FloatTensor(b_memory[:, N_STATES+2:2*N_STATES+2]))
    b_log = Variable(torch.FloatTensor(b_memory[:, 2*N_STATES+2:2*N_STATES+3]), requires_grad=True)

    b_value = critic(b_s)
    b_next_value = critic(b_s_)
    b_advantage = b_r + b_next_value - b_value
    critic_loss = critic_criterion(b_value, b_r + b_next_value)
    actor_loss = (-b_log * b_advantage.detach()).mean()
    optimizerA.zero_grad()
    optimizerC.zero_grad()
    actor_loss.backward()
    critic_loss.backward()
    optimizerA.step()
    optimizerC.step()


result = result.sum(-1, keepdim = True)
result = result / NUM_OF_AGENT
res = result.detach().numpy()

with open('A2C-on-policy-parallel.txt', 'a+') as f:
    for x in res:
        f.write(str(x.item())+'\n')
