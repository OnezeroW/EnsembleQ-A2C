import torch
from torch._C import dtype
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

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if use_cuda else "cpu")

clip = 10.0     # gradient clipping
# Hyper Parameters
NUM_Q = 10  # N: number of Q networks
NUM_MIN = 2 # M: number of selected Q networks to update target Q value
POLICY_UPDATE_DELAY = 10      # G: policy update delay
NUM_OF_AGENT=64
polyak = 0.995   # update target networks
alpha = 1.0     # SAC entropy hyperparameter
LEN_EPISODE = 30000   # the length of each episode
# NUM_EPISODE = 10  # the number of episode
# BATCH_SIZE = 128
HIDDEN_SIZE = 256    # the dim of hidden layers
LR = 3e-4                   # learning rate
MEMORY_CAPACITY = NUM_OF_AGENT  # H*K ------------------0710 updated--------------------

NUM_OF_LINKS=6
MAX_DEADLINE=10
LAMBDA = [0.01, 0.015, 0.025, 0.01, 0.015, 0.025]
INIT_P = [0.8, 0.9, 1.0, 0.8, 0.9, 1.0]

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
weight = np.ones((NUM_OF_LINKS), dtype=np.float)
p_current = np.zeros((NUM_OF_AGENT, NUM_OF_LINKS), dtype=np.float)    #current delivery ratio
p_next = np.zeros((NUM_OF_AGENT, NUM_OF_LINKS), dtype=np.float)   #next delivery ratio
N_ACTIONS = NUM_OF_LINKS
N_STATES = NUM_OF_LINKS * (MAX_DEADLINE + 3)    #State s = (Deficit, e, Buffer, p)
memory_counter = 0
memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 3))

s = torch.zeros(NUM_OF_AGENT, N_STATES)
s_ = torch.zeros(NUM_OF_AGENT, N_STATES)

def generateState(Deficit, e, Buffer, p):  #generate 1-D state
    arr1 = np.array(Deficit)
    arr2 = np.array(e)
    arr3 = Buffer.flatten()
    arr4 = np.array(p)
    result = np.concatenate((arr1, arr2, arr3, arr4))
    result = torch.FloatTensor(result)
    return result

def store_transition(s, a, r, s_, log_prob):
    global memory
    global memory_counter
    transition = np.hstack((s, [a, r], s_, [log_prob]))
    # replace the old memory with new memory
    index = memory_counter % MEMORY_CAPACITY
    memory[index, :] = transition
    memory_counter += 1

# i.i.d. with Poisson distribution
def ARR_POISSON(lam, agent):
    global Arrival
    Arrival[agent].fill(0)
    for l in range(NUM_OF_LINKS):
        # Arrival[agent][l][MAX_DEADLINE] = np.random.poisson(lam[l%6])
        for d in range(1, MAX_DEADLINE+1):
            # Arrival[agent][l][d] = np.random.poisson(lam)
            Arrival[agent][l][d] = np.random.poisson(lam[l])

# wi = 0.02
# wi = 1 / HIDDEN_SIZE
# wi = 2 / HIDDEN_SIZE
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
        self.layer1 = nn.Linear(N_STATES + 1, HIDDEN_SIZE)    # input: state, action
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

actor = Actor().to(device)
actor.load_state_dict(torch.load('params-500-2k.pkl'))     # load initial NN model trained from base policy

# q1, q2 = Critic().to(device), Critic().to(device)
# q1_target, q2_target = Critic().to(device), Critic().to(device)
# for target_param, param in zip(q1_target.parameters(), q1.parameters()):
#     target_param.data.copy_(param.data)
# for target_param, param in zip(q2_target.parameters(), q2.parameters()):
#     target_param.data.copy_(param.data)
# optimizerA = optim.Adam(actor.parameters(), lr=LR)
# optimizerQ1 = optim.Adam(q1.parameters(), lr=LR)
# optimizerQ2 = optim.Adam(q2.parameters(), lr=LR)

q_net_list, q_target_net_list = [], []
for q_i in range(NUM_Q):    # N: number of Q networks
    new_q_net = Critic().to(device)
    q_net_list.append(new_q_net)
    new_q_target_net = Critic().to(device)
    new_q_target_net.load_state_dict(new_q_net.state_dict())
    q_target_net_list.append(new_q_target_net)

optimizerA = optim.Adam(actor.parameters(), lr=LR)
optimizerQ_list = []
for q_i in range(NUM_Q):
    optimizerQ_list.append(optim.Adam(q_net_list[q_i].parameters(), lr=LR))

mse_criterion = nn.MSELoss()

result = torch.zeros((LEN_EPISODE, NUM_OF_AGENT))        # store the reward of each trajectory

#initialize state 
Buffer.fill(0)
Deficit.fill(0)
totalArrival.fill(0)    #mark
totalDelivered.fill(0)
totalBuff.fill(0)
e.fill(MAX_DEADLINE+1)
p_current.fill(0)
p_next.fill(0)

# current state s
for agent in range(NUM_OF_AGENT):
    s[agent] = generateState(Deficit[agent], e[agent], Buffer[agent][:,1:MAX_DEADLINE+1], p_next[agent])

#total reward of one episode
ep_r = np.zeros((NUM_OF_AGENT), dtype=np.float)

for len in range(LEN_EPISODE):  #length of each episode
    for agent in range(NUM_OF_AGENT):
    #for num in range(MEMORY_CAPACITY):   #generate multiple samples during each time slot
        # s[agent] = s[agent].to(device)
        # dist, value = actor(s[agent]), critic(s[agent])
        dist = actor(s[agent].to(device)).detach().cpu()
        try:
            a = torch.multinomial(dist, 1).item()
        except:
            print("ERROR! ", dist)
        log_prob = torch.log(dist[a])   # no use

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
                Deficit[agent][l] = max(Deficit[agent][l] + sumArrival[agent][l] * INIT_P[l] - 1, 0)
            else:
                Deficit[agent][l] = max(Deficit[agent][l] + sumArrival[agent][l] * INIT_P[l], 0)
        for l in range(NUM_OF_LINKS):
            if totalArrival[agent][l] == 0:
                p_next[agent][l] = 0
            else:
                p_next[agent][l] = totalDelivered[agent][l] / totalArrival[agent][l] #next delivery ratio

        s_[agent] = generateState(Deficit[agent], e[agent], Buffer[agent][:,1:MAX_DEADLINE+1], p_next[agent])
        r = np.min(p_next[agent])

        store_transition(s[agent], a, r, s_[agent], log_prob)
        p_current[agent] = p_next[agent].copy()   #current delivery ratio
        ep_r[agent] += r
        s[agent] = s_[agent].clone().detach()
        result[len][agent] = round(r, 6)     # current total reward

    # sample batch transitions
    #sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
    b_memory = memory[:, :]
    b_s = Variable(torch.FloatTensor(b_memory[:, :N_STATES]))
    b_s = b_s.to(device)
    b_a = Variable(torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int)))
    b_a = b_a.to(device)
    b_r = Variable(torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2]))
    b_r = b_r.to(device)
    b_s_ = Variable(torch.FloatTensor(b_memory[:, N_STATES+2:2*N_STATES+2]))
    b_s_ = b_s_.to(device)
    b_log = Variable(torch.FloatTensor(b_memory[:, 2*N_STATES+2:2*N_STATES+3]), requires_grad=True) # no use
    b_log = b_log.to(device)

    update_a = torch.zeros(NUM_OF_AGENT, dtype=torch.int)
    update_log_prob = torch.zeros(NUM_OF_AGENT)
    for i in range(POLICY_UPDATE_DELAY):    # G: policy update delay
        update_dist = actor(b_s_).cpu()     # next state -> next action -> log prob
        for j in range(NUM_OF_AGENT):
            try:
                update_a[j] = torch.multinomial(update_dist[j], 1).item()     # use cpu for sampling
            except:
                print("ERROR! ", len, i, j, update_dist[j])
            update_log_prob[j] = torch.log(update_dist[j,int(update_a[j])])
        update_a = update_a.reshape([NUM_OF_AGENT,1])
        update_a = update_a.to(device)
        update_log_prob = update_log_prob.reshape([NUM_OF_AGENT,1])
        update_log_prob = update_log_prob.to(device)

        # select M q_nets from N q_nets
        sample_idxs = np.random.choice(NUM_Q, NUM_MIN, replace=False)
        q_prediction_next_list = []
        for sample_idx in sample_idxs:
            q_prediction_next = q_target_net_list[sample_idx](torch.cat((b_s_, update_a), 1))
            q_prediction_next_list.append(q_prediction_next)
        q_prediction_next_cat = torch.cat(q_prediction_next_list, 1)
        min_q, min_indices = torch.min(q_prediction_next_cat, dim=1, keepdim=True)
        y_q = b_r + min_q - update_log_prob * alpha     # alpha is a SAC entropy hyperparameter

        q_prediction_list = []
        for q_i in range(NUM_Q):
            q_prediction = q_net_list[q_i](torch.cat((b_s, b_a), 1))
            q_prediction_list.append(q_prediction)
        q_prediction_cat = torch.cat(q_prediction_list, dim=1)
        y_q = y_q.expand((-1, NUM_Q)) if y_q.shape[1] == 1 else y_q
        q_loss_all = mse_criterion(q_prediction_cat, y_q.detach()) * NUM_Q   # loss function of N q_nets
        
        for q_i in range(NUM_Q):
            optimizerQ_list[q_i].zero_grad()
        q_loss_all.backward()
        for q_i in range(NUM_Q):
            torch.nn.utils.clip_grad_norm_(q_net_list[q_i].parameters(), clip)  # gradient clipping
        for q_i in range(NUM_Q):
            optimizerQ_list[q_i].step()

        for q_i in range(NUM_Q):
            for target_param, param in zip(q_target_net_list[q_i].parameters(), q_net_list[q_i].parameters()):
                target_param.data.copy_(
                    target_param.data * polyak + param.data * (1 - polyak)
                )

        dist_tilda = actor(b_s).cpu()     # current state -> action -> log prob
        a_tilda = torch.zeros(NUM_OF_AGENT, dtype=torch.int)
        log_prob_tilda = torch.zeros(NUM_OF_AGENT)
        for j in range(NUM_OF_AGENT):
            a_tilda[j] = torch.multinomial(dist_tilda[j], 1).item()
            log_prob_tilda[j] = torch.log(dist_tilda[j,int(a_tilda[j])])
        a_tilda = a_tilda.reshape([NUM_OF_AGENT,1])
        a_tilda = a_tilda.to(device)
        log_prob_tilda = log_prob_tilda.reshape([NUM_OF_AGENT,1])
        log_prob_tilda = log_prob_tilda.to(device)

        q_a_tilda_list = []
        for sample_idx in range(NUM_Q):
            q_a_tilda = q_net_list[sample_idx](torch.cat((b_s, a_tilda), 1))    # ***
            q_a_tilda_list.append(q_a_tilda)
        q_a_tilda_cat = torch.cat(q_a_tilda_list, 1)
        ave_q = torch.mean(q_a_tilda_cat, dim=1, keepdim=True)
        actor_loss = (log_prob_tilda * alpha - ave_q).mean()    # alpha is a SAC entropy hyperparameter
        
        optimizerA.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), clip)  # gradient clipping
        optimizerA.step()


    #     q1_value = q1_target(torch.cat((b_s_, update_a), 1))
    #     q2_value = q2_target(torch.cat((b_s_, update_a), 1))
    #     y = b_r + torch.min(q1_value, q2_value) - update_log_prob

    #     q1_loss = critic_criterion(q1(torch.cat((b_s, b_a), 1)), y.detach())
    #     optimizerQ1.zero_grad()
    #     q1_loss.backward()
    #     optimizerQ1.step()

    #     q2_loss = critic_criterion(q2(torch.cat((b_s, b_a), 1)), y.detach())
    #     optimizerQ2.zero_grad()
    #     q2_loss.backward()
    #     optimizerQ2.step()

    #     for target_param, param in zip(q1_target.parameters(), q1.parameters()):
    #         target_param.data.copy_(
    #             target_param.data * polyak + param.data * (1 - polyak)
    #         )
    #     for target_param, param in zip(q2_target.parameters(), q2.parameters()):
    #         target_param.data.copy_(
    #             target_param.data * polyak + param.data * (1 - polyak)
    #         )

    # actor_loss = (0.5 * q1(torch.cat((b_s, b_a), 1)) + 0.5 * q2(torch.cat((b_s, b_a), 1)) - b_log).mean()
    # optimizerA.zero_grad()
    # actor_loss.backward()
    # optimizerA.step()

result = result.sum(-1, keepdim = True)
result = result / NUM_OF_AGENT
res = result.detach().numpy()

with open('EnsembleQ-N'+str(NUM_Q)+'M'+str(NUM_MIN)+'.txt', 'a+') as f:
    for x in res:
        f.write(str(x.item())+'\n')

# torch.save(actor.state_dict(), 'actor.pkl')
# for q_i in range(NUM_Q):
#     torch.save(q_net_list[q_i].state_dict(), 'critic_'+q_i+'.pkl')
