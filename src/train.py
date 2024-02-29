import math
import os
import random
from collections import deque, namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from decimal import Decimal

from env_hiv import HIVPatient
from evaluate import evaluate_HIV
from gymnasium.wrappers import TimeLimit

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
) 


Transition = namedtuple(
    'Transition',
    ('state', 'action', 'next_state', 'reward')
    )


class ReplayMemory:

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    

class LinearBlock(nn.Module):

    def __init__(self, in_features, out_features):
        super(LinearBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)
    

class DQN(nn.Module):

    def __init__(self, state_dim=6, num_actions=4, num_neurons=256, num_linear_blocks=4):
        super(DQN, self).__init__()
        # first layer
        layers = [nn.Linear(state_dim, num_neurons), nn.ReLU()]
        # linear blocks 
        for _ in range(num_linear_blocks):
            layers.append(LinearBlock(num_neurons, num_neurons))
        # last linear layer
        layers.append(nn.Linear(num_neurons, num_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

        
class ProjectAgent:

    def __init__(self):
        self.policy_net = DQN()

    def act(self, observation, use_random=False):
        with torch.no_grad():
            out = self.policy_net(torch.tensor(observation, dtype=torch.float32).unsqueeze(0))
            return out.argmax().item()

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self):
        path = os.path.join(CURRENT_DIR, 'best_policy.pt')
        self.policy_net.load_state_dict(torch.load(path))


if __name__ == "__main__":

    # ===== CONF ===== #
    with open(os.path.join(CURRENT_DIR, 'train_config.yml'), 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    BATCH_SIZE = config['batch_size']
    GAMMA = config['gamma']
    EPS_START = config['eps_start']
    EPS_END = config['eps_end']
    EPS_DECAY = config['eps_decay']
    TAU = config['tau']
    LR = float(config['lr'])
    print(type(LR))
    print(LR)
    # =============== #

    # ===== ENV ===== #
    env = TimeLimit(
    env = HIVPatient(domain_randomization=True), max_episode_steps=200)
    agent = ProjectAgent()
    # ============== # 
    
    # ===== TRAIN ===== #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target_net = DQN()
    target_net.load_state_dict(agent.policy_net.state_dict())
    optimizer = optim.AdamW(agent.policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)

    steps_done = 0

    def select_action(state, agent):
        global steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return agent.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[env.action_space.sample()]], dtype=torch.long)
        
    def optimize_model(agent):
        if len(memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = agent.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(agent.policy_net.parameters(), 100)
        optimizer.step()

    num_episodes = 600
    best_val = 0

    for i_episode in range(num_episodes):
        # Initialize the environment and get its state
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            if t % 50 == 0:
                print(t)
            action = select_action(state, agent)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device, dtype=torch.float32)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model(agent)

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = agent.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)
            
            if len(memory) < BATCH_SIZE:
                continue
                
            validation_score = evaluate_HIV(agent=agent, nb_episode=1)
            if validation_score > best_val:
                print(f"better model, val score: {Decimal(str(validation_score)):.2E}")
                best_val = validation_score
                torch.save(agent.policy_net.state_dict(), os.path.join(CURRENT_DIR, f'./best_policy.pt'))

        







        

        
    
    



    
    