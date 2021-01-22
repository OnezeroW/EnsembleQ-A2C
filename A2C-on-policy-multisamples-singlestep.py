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
LAMBDA = [0.005, 0.01, 0.015]
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
#weight = np.array([0.05, 0.3, 0.4, 0.1, 0.15])
p_current = np.zeros((NUM_OF_LINKS), dtype=np.float)    #current delivery ratio
p_next = np.zeros((NUM_OF_LINKS), dtype=np.float)   #next delivery ratio

N_ACTIONS = NUM_OF_LINKS
N_STATES = NUM_OF_LINKS * 5
INIT_P = 0.75    #initial delivery ratio p0
NUM_EPISODE = 10  # the number of episode
LEN_EPISODE = 10000   # the length of each episode
BATCH_SIZE = 128
HIDDEN_SIZE = 64    # the dim of hidden layers
LR = 3e-4                   # learning rate
MEMORY_CAPACITY = 10
memory_counter = 0
memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 3))

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
        for d in range(1, MAX_DEADLINE+1):
            Arrival[l][d] = np.random.poisson(lam[l%3])

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

actor_criterion = nn.MSELoss()

xval = []
yval = []
result = torch.zeros((LEN_EPISODE, NUM_EPISODE))        # store the reward of each trajectory
print('\nCollecting experience...')
for i_episode in range(NUM_EPISODE + 1):      # the first 10000 samples are stored in memory
    #initialize state s
    Buffer.fill(0)
    Deficit.fill(0)
    totalArrival.fill(0)    #mark
    totalDelivered.fill(0)

    totalBuff.fill(0)
    e.fill(MAX_DEADLINE+1) 

    # current state s
    s = generateState(Deficit, e, totalBuff, totalArrival, totalDelivered)
    p_current.fill(0)

    ep_r = 0    #total reward of one episode
    for len in range(LEN_EPISODE):  #length of each episode
        for num in range(MEMORY_CAPACITY):   #generate multiple samples during each time slot
            dist, value = actor(s), critic(s)
            if num == MEMORY_CAPACITY - 1:
                a = torch.multinomial(dist, 1).item()
                log_prob = torch.log(dist[a])

                ARR_POISSON(LAMBDA)
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
                e.fill(MAX_DEADLINE+1)      #initial earliest deadline should be MAX_DEADLINE+1
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
                r = 0
                sumWeight = 0
                for l in range(NUM_OF_LINKS):
                    r += weight[l] * (p_next[l] - p_current[l]) #reward calculation, R(t+1)=\sum c_l*[p_l(t+1)-p_l(t)]
                    sumWeight += weight[l]
                r /= sumWeight   # reward r

                store_transition(s, a, r, s_, log_prob)
        
                p_current = p_next.copy()   #current delivery ratio
                ep_r += r
                s = s_

                if i_episode > 0:
                    result[len][i_episode-1] = round(ep_r, 3)     # current total reward

            else:
                Buffer_temp = Buffer.copy()
                Deficit_temp = Deficit.copy()
                #Arrival_temp = Arrival.copy()
                sumArrival_temp = sumArrival.copy()
                totalArrival_temp = totalArrival.copy()
                totalDelivered_temp = totalDelivered.copy()
                #Action_temp = Action.copy()
                totalBuff_temp = totalBuff.copy()
                e_temp = e.copy()
                p_current_temp = p_current.copy()
                p_next_temp = p_next.copy()

                #dist, value = actor(s), critic(s)
                a = torch.multinomial(dist, 1).item()
                log_prob = torch.log(dist[a])

                ARR_POISSON(LAMBDA)
                # update total arrival packets at current time slot
                sumArrival_temp.fill(0)
                for l in range(NUM_OF_LINKS):
                    for d in range(1, MAX_DEADLINE+1):
                        sumArrival_temp[l] += Arrival[l][d]
                # update total arrival packets from beginning
                for l in range(NUM_OF_LINKS):
                    totalArrival_temp[l] += sumArrival_temp[l]
                # update buffer
                Action.fill(0)
                for d in range(1, MAX_DEADLINE+1):
                    if Buffer_temp[a][d] > 0:
                        Action[a][d] = 1
                        # update total delivered packets from beginning
                        totalDelivered_temp[a] += 1
                        break

                for l in range(NUM_OF_LINKS):
                    for d in range(1, MAX_DEADLINE+1):
                        Buffer_temp[l][d] = max(Buffer_temp[l][d+1] + Arrival[l][d] - Action[l][d+1], 0)
        
                # update totalBuff
                totalBuff_temp.fill(0)
                for l in range(NUM_OF_LINKS):
                    for d in range(1, MAX_DEADLINE+1):
                        totalBuff_temp[l] = totalBuff_temp[l] + Buffer_temp[l][d]
                # update the earliest deadline on link l
                e_temp.fill(MAX_DEADLINE+1)      #initial earliest deadline should be MAX_DEADLINE+1
                for l in range(NUM_OF_LINKS):
                    for d in range(1, MAX_DEADLINE+1):
                        if Buffer_temp[l][d] > 0:
                            e_temp[l] = d
                            break

                # update deficit
                for l in range(NUM_OF_LINKS):
                    if l == a:
                        Deficit_temp[l] = max(Deficit_temp[l] + sumArrival_temp[l] * INIT_P - 1, 0)
                    else:
                        Deficit_temp[l] = max(Deficit_temp[l] + sumArrival_temp[l] * INIT_P, 0)

                s_ = generateState(Deficit_temp, e_temp, totalBuff_temp, totalArrival_temp, totalDelivered_temp)   # next state s_

                for l in range(NUM_OF_LINKS):
                    if totalArrival_temp[l] == 0:
                        p_next_temp[l] = 0
                    else:
                        p_next_temp[l] = totalDelivered_temp[l] / totalArrival_temp[l] #next delivery ratio
                r = 0
                sumWeight = 0
                for l in range(NUM_OF_LINKS):
                    r += weight[l] * (p_next_temp[l] - p_current_temp[l]) #reward calculation, R(t+1)=\sum c_l*[p_l(t+1)-p_l(t)]
                    sumWeight += weight[l]
                r /= sumWeight   # reward r

                store_transition(s, a, r, s_, log_prob)


        # if samples are enough, then start training
        if memory_counter > MEMORY_CAPACITY:
            #print(i_episode, len, round(ep_r, 3))
            if memory_counter == MEMORY_CAPACITY + MEMORY_CAPACITY:
                print('\nCollecting experience completed! Start training...')
            '''
            if i_episode > 0:
                result[len][i_episode-1] = round(ep_r, 3)     # current total reward
            '''

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
            critic_loss = actor_criterion(b_value, b_r + b_next_value)
            actor_loss = (-b_log * b_advantage.detach()).mean() # minus ?

            #print('actor_loss=', actor_loss, 'critic_loss=', critic_loss)

            optimizerA.zero_grad()
            optimizerC.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            optimizerA.step()
            optimizerC.step()
'''
            for parameters in actor.parameters():
                print(parameters)
'''
'''
            if len == LEN_EPISODE-1:
                print('Episode number: ', i_episode,
                    '| Total reward: ', round(ep_r + INIT_P, 3))
                xval.append(i_episode)
                yval.append(round(ep_r + INIT_P, 3))
'''
result = result.sum(-1, keepdim = True)
result = result / NUM_EPISODE
res = result.detach().numpy()
#print(res)

with open('A2C-on-policy-multisamples-singlestep.txt', 'a+') as f:
    for x in res:
        f.write(str(x.item())+'\n')
