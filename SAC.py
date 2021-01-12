import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random as rd
import torch.optim as optim

# Hyper Parameters
INIT_P = 0.5    #initial delivery ratio p0
NUM_OF_LINKS=5
MAX_DEADLINE=10
EPSILON = 0.05
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
p_current = np.zeros((NUM_OF_LINKS), dtype=np.float)    #current delivery ratio
p_next = np.zeros((NUM_OF_LINKS), dtype=np.float)   #next delivery ratio

N_ACTIONS = NUM_OF_LINKS
#N_STATES = NUM_OF_LINKS * (MAX_DEADLINE + 3)    #State s = (Buffer, Deficit, TA, TD)
N_STATES = NUM_OF_LINKS * 5    #State s = (Deficit, e, TB, TA, TD)

NUM_EPISODE = 1000  # the number of episode
LEN_EPISODE = 1000   # the length of each episode
BATCH_SIZE = 64
LR = 0.01                   # learning rate
MEMORY_CAPACITY = 10000
memory_counter = 0
memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))

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
            Arrival[l][d] = np.random.poisson(lam)

# non-i.i.d., periodic traffic pattern
def ARR_PERIODIC(index):
    global Arrival
    Arrival.fill(0)
    if index % 5 == 0:
        # pattern C
        Arrival[0][1] = 1
        Arrival[1][2] = 1
        Arrival[1][3] = 1
    elif index % 5 == 3:
        # pattern B
        Arrival[0][2] = 1
        Arrival[1][1] = 1
    else:
        Arrival[0][0] = 0

class ValueNet(nn.Module):      # input: state; output: state value
    def __init__(self, ):
        super(ValueNet, self).__init__()
        self.layer1 = nn.Linear(N_STATES, 20)
        self.layer1.weight.data.normal_(0, 0.1)
        self.layer2 = nn.Linear(20, 10)
        self.layer2.weight.data.normal_(0, 0.1)
        self.value = nn.Linear(10, 1)
        self.value.weight.data.normal_(0, 0.1)
        
    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        state_value = self.value(x)
        return state_value

class QNet(nn.Module):      # input: state, action; output: Q value
    def __init__(self, ):
        super(QNet, self).__init__()
        self.layer1 = nn.Linear(N_STATES + 1, 20)
        self.layer1.weight.data.normal_(0, 0.1)
        self.layer2 = nn.Linear(20, 10)
        self.layer2.weight.data.normal_(0, 0.1)
        self.value = nn.Linear(10, 1)
        self.value.weight.data.normal_(0, 0.1)
        
    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        q_value = self.value(x)
        return q_value

class PolicyNet(nn.Module):      # input: state; output: actions' probability vector
    def __init__(self, ):
        super(PolicyNet, self).__init__()
        self.layer1 = nn.Linear(N_STATES, 20)
        self.layer1.weight.data.normal_(0, 0.1)
        self.layer2 = nn.Linear(20, 10)
        self.layer2.weight.data.normal_(0, 0.1)
        self.action = nn.Linear(10, N_ACTIONS)
        self.action.weight.data.normal_(0, 0.1)
        
    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.action(x)
        action_probs = F.softmax(x, dim=-1)
        return action_probs

def store_transition(s, a, r, s_):
    global memory
    global memory_counter
    transition = np.hstack((s, [a, r], s_))
    # replace the old memory with new memory
    index = memory_counter % MEMORY_CAPACITY
    memory[index, :] = transition
    memory_counter += 1

v_net, q_net, p_net = ValueNet(), QNet(), PolicyNet()
#actor.load_state_dict(torch.load('params.pkl'))     # load initial NN model trained from base policy
optimizerV = optim.Adam(v_net.parameters())
optimizerQ = optim.Adam(q_net.parameters())
optimizerP = optim.Adam(p_net.parameters())

v_criterion = nn.MSELoss()
q_criterion = nn.MSELoss()

xval = []
yval = []
print('\nCollecting experience...')
for i_episode in range(NUM_EPISODE):
    #initialize state s
    Buffer.fill(0)
    Deficit.fill(0)
    totalArrival.fill(1)    #mark
    totalDelivered.fill(0)

    totalBuff.fill(0)
    #e.fill(MAX_DEADLINE + 1) 
    e.fill(0) 

    # current state s
    #s = generateState(Buffer, Deficit, totalArrival, totalDelivered)
    s = generateState(Deficit, e, totalBuff, totalArrival, totalDelivered)
    p_current.fill(INIT_P)

    ep_r = 0    #total reward of one episode
    for len in range(LEN_EPISODE):  #length of each episode
        action_prob = p_net(s)
        a = torch.multinomial(action_prob, 1).item()
        print(i_episode, len, a, action_prob)

        # take action
        
        #consider arrival packets here
        #ARR_UNIFROM(0,1)
        ARR_POISSON(0.02)
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
        #e.fill(MAX_DEADLINE + 1)      #initial earliest deadline should be MAX_DEADLINE+1
        e.fill(0) 
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

        #s_ = generateState(Buffer, Deficit, totalArrival, totalDelivered)   # next state s_
        s_ = generateState(Deficit, e, totalBuff, totalArrival, totalDelivered)   # next state s_

        for l in range(NUM_OF_LINKS):
            p_next[l] = totalDelivered[l] / totalArrival[l] #next delivery ratio
            #print('EP=',i_episode, ', len=', len, ', l=', l, ', TD=', totalDelivered[l], ', TA=', totalArrival[l], ', Deficit=', Deficit[l])
        r = 0
        sumWeight = 0
        for l in range(NUM_OF_LINKS):
            r += weight[l] * (p_next[l] - p_current[l]) #reward calculation, R(t+1)=\sum c_l*[p_l(t+1)-p_l(t)]
            sumWeight += weight[l]
        r /= sumWeight   # reward r

        store_transition(s, a, r, s_)

        p_current = p_next.copy()   #current delivery ratio
        ep_r += r
        s = s_

        if memory_counter > MEMORY_CAPACITY:
            # sample batch transitions
            sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
            b_memory = memory[sample_index, :]
            b_s = torch.FloatTensor(b_memory[:, :N_STATES])
            b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
            b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
            b_s_ = torch.FloatTensor(b_memory[:, N_STATES+2:2*N_STATES+2])
            #b_log = Variable(torch.FloatTensor(b_memory[:, 2*N_STATES+2:2*N_STATES+3]), requires_grad=True)

            temp = torch.cat((b_s, b_a), 1) #input of q_net
            expected_q_value = q_net(temp)
            expected_value = v_net(b_s)
            action_probs = p_net(b_s)
            #print(action_probs)
            
            #q_loss
            next_q_value = b_r + v_net(b_s_)
            q_loss = q_criterion(expected_q_value, next_q_value.detach())
            
            #choose new_action
            new_action = torch.multinomial(action_probs, 1)
            log_prob = torch.zeros((BATCH_SIZE,1))
            for i in range(BATCH_SIZE):
                log_prob[i] = action_probs[i][new_action[i]]
            log_prob = torch.log(log_prob)
            #sum_log_prob = torch.unsqueeze(torch.sum(log_prob, 1), dim=1)

            #value_loss
            temp_1 = torch.cat((b_s, new_action), 1)
            expected_new_q_value = q_net(temp_1)
            next_value = expected_new_q_value - log_prob
            value_loss = v_criterion(expected_value, next_value.detach())

            #policy_loss
            log_prob_target = expected_new_q_value - expected_value
            policy_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()

            optimizerV.zero_grad()
            value_loss.backward()
            optimizerV.step()

            optimizerQ.zero_grad()        
            q_loss.backward()
            optimizerQ.step()

            optimizerP.zero_grad()        
            policy_loss.backward()
            optimizerP.step()

            #if len == LEN_EPISODE-1 and i_episode % 100 == 0:
            if len == LEN_EPISODE-1:
                print('Iteration number: ', i_episode,
                        '| Total reward: ', round(ep_r + INIT_P, 3))
                xval.append(i_episode)
                yval.append(round(ep_r + INIT_P, 3))

plt.title('Figure 1')
plt.plot(xval, yval, color = 'red', label = 'SAC')
plt.legend()
plt.xlabel('Episode number')
plt.ylabel('Average delivery ratio')
plt.show()
'''
with open('SAC-x.txt', 'a+') as f:
    for x in xval:
        f.write(str(x)+'\n')
with open('SAC-y.txt', 'a+') as f:
    for y in yval:
        f.write(str(y)+'\n')
'''