import random
import math
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def push(self, obs, action, next_obs, reward,done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = (obs, action, next_obs, reward, done)
        #print('self.memory[self.position]',self.memory[self.position])
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Samples batch_size transitions from the replay memory and returns a tuple
            (obs, action, next_obs, reward)
        """
        sample = random.sample(self.memory, batch_size)
        #print('sample',sample)
        return tuple(zip(*sample))


class DQN(nn.Module):
    def __init__(self, env_config):
        super(DQN, self).__init__()

        # Save hyperparameters needed in the DQN class.
        self.batch_size = env_config["batch_size"]
        self.gamma = env_config["gamma"]
        self.eps_start = env_config["eps_start"]
        self.eps_end = env_config["eps_end"]
        self.anneal_length = env_config["anneal_length"]
        self.n_actions = env_config["n_actions"]

        

        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0) 
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0) 
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, self.n_actions)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.eps = self.eps_start
        self.steps = 0

        self.features1 = nn.Sequential(
            self.conv1,
            self.relu
            )
        self.features2 = nn.Sequential(
            self.conv2,
            self.relu,
            self.conv3,
            self.relu,
            )
        self.fc = nn.Sequential(
            self.fc1,
            self.relu,
            self.fc2
        )
    def forward(self, x):
        """Runs the forward pass of the NN depending on architecture."""
        #print(x.shape)
        x = self.features1(x)
        x = self.features2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def act(self, observation, exploit=False):
        """Selects an action with an epsilon-greedy exploration strategy."""
        # TODO: Implement action selection using the Deep Q-network. This function
        #       takes an observation tensor and should return a tensor of actions.
        #       For example, if the state dimension is 4 and the batch size is 32,
        #       the input would be a [32, 4] tensor and the output a [32, 1] tensor.
        # TODO: Implement epsilon-greedy exploration.
        self.steps = self.steps + 1

        #self.eps = self.eps_end + (self.eps_start - self.eps_end) * \
        #math.exp(-1. * self.steps / 200)
        self.eps = 0.1

        

        ran_num = random.random()
        #print('self.eps', self.eps)
        #print('ran_num', ran_num)
        if ran_num < self.eps:
            actions = [random.randint(0,self.n_actions-1)]
            actions =  torch.tensor(actions)
            #print('random actions',actions)
            return actions
            #batch_actions = random.sample(range(0, self.n_actions), observation.shape[0])
            #return torch.tensor(batch_actions)
        else :
            self.eval()
            with torch.no_grad():
                #print('observation.shape',observation.shape)
                q_values = self.forward(observation)
                
                
            self.train()
            #print('q_values',q_values)
            actions = torch.tensor([torch.argmax(q_values)])
            #print('select actions',actions)
            return actions
            #batch_actions = forward(observation)
            #return torch.tensor(batch_actions)

def optimize(dqn, target_dqn, memory, optimizer):
    """This function samples a batch from the replay buffer and optimizes the Q-network."""
    # If we don't have enough transitions stored yet, we don't train.
    if len(memory) < dqn.batch_size:
        return

    # TODO: Sample a batch from the replay memory and concatenate so that there are
    #       four tensors in total: observations, actions, next observations and rewards.
    #       Remember to move them to GPU if it is available, e.g., by using Tensor.to(device).
    #       Note that special care is needed for terminal transitions!
    #optimizer.zero_grad()
    
    observations, actions, next_observations , rewards ,dones = memory.sample(dqn.batch_size)
    #print('observations', observations)

    #print('actions**',actions)
    observations = torch.stack(list(observations), dim=0)
    #print('observations>>', observations)
    #print('observations.shape', observations.shape)
    actions = torch.stack(list(actions), dim=0)
    rewards = torch.stack(list(rewards), dim=0)
    next_observations =  torch.stack(list(next_observations), dim=0)
    dones =  torch.stack(list(dones), dim=0) 
    #print('dones',dones)     
    # TODO: Compute the current estimates of the Q-values for each state-action
    #       pair (s,a). Here, torch.gather() is useful for selecting the Q-values
    #       corresponding to the chosen actions.
    
    q_values = dqn(observations)
    params = list(dqn.parameters())
    #print(params[0])
    #print('q_values',q_values)

    #print('actions.unsqueeze(-1)',actions.unsqueeze(-1).long())

    
    #print((actions).long())
    #print(q_values)
    q_values =  torch.gather(q_values, 1, (actions.unsqueeze(-1)).long())
    #q_values =  torch.gather(q_values, 1, (actions).long())
    #q_values =  torch.gather(q_values, 1, (actions.unsqueeze(-1)).long())
    #print(q_values)
    #print('q_values       ',q_values)
    # TODO: Compute the Q-value targets. Only do this for non-terminal transitions!

    #print('torch.max(target_dqn.forward(next_observations))', torch.max(target_dqn.forward(next_observations),axis=0))
    next_q_value_targets = target_dqn(next_observations).detach().max(1)[0]
    #print('next_q_value_targets*************', next_q_value_targets)
    
    #print('next_q_value_targets', next_q_value_targets)
    #print('torch.max(next_q_value_targets,axis=0)',torch.max(next_q_value_targets,axis=-1))
    #print('dones', dones)
    #print('next_q_value_targets          ',  next_q_value_targets)
    #print('next_q_value_targets.max(1)[0]',  next_q_value_targets.max(1)[0])
    q_value_targets = rewards + (target_dqn.gamma * next_q_value_targets * (1-dones)) # torch.max(next_q_value_targets,axis=1)[0]
    # Compute loss.
    #print('q_values          ',q_values)

    #print('q_values.squeeze()',q_values.squeeze())
    #print('q_value_targets   ',q_value_targets)
    #print('---------')
    #loss = F.smooth_l1_loss(q_values, q_value_targets.unsqueeze(-1))
    loss = F.mse_loss(q_values.squeeze(), q_value_targets)

    # Perform gradient descent.
    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    return loss.item()
