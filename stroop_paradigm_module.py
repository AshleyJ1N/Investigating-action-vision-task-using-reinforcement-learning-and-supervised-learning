import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple, deque
import random
import os
import json
from datetime import datetime

class StroopEnvironment:
    def __init__(self):
        self.colors = ['RED', 'BLUE', 'GREEN', 'YELLOW']
        self.color_to_idx = {color: idx for idx, color in enumerate(self.colors)}
        self.reset()
        
    def reset(self):
        self.word = random.choice(self.colors)
        self.ink_color = random.choice(self.colors)
        word_state = np.zeros(len(self.colors))
        color_state = np.zeros(len(self.colors))
        word_state[self.color_to_idx[self.word]] = 1
        color_state[self.color_to_idx[self.ink_color]] = 1
        self.state = np.concatenate([word_state, color_state])
        return self.state
        
    def step(self, action):
        reward = 1 if action == self.color_to_idx[self.ink_color] else -1
        rt = self._simulate_reaction_time(action)
        done = True
        return self.state, reward, done, {'rt': rt}
    
    def _simulate_reaction_time(self, action):
        base_rt = 0.4  
        congruent = self.word == self.ink_color
        
        if congruent:
            rt = base_rt + np.random.normal(0, 0.1)
        else:
            rt = base_rt + 0.2 + np.random.normal(0, 0.15)  # 不一致条件下反应时间更长
            
        # 错误反应通常更快
        if action != self.color_to_idx[self.ink_color]:
            rt *= 0.9
            
        return max(0.2, rt)  # 确保反应时间为正
    

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    

class StroopAgent:
    def __init__(self, state_size, action_size, base_dir='stroop_saved_models'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.base_dir = base_dir
        
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.memory = deque(maxlen=10000)
        
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 0.9
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
    def act(self, state):
        if random.random() > self.epsilon:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state)
                return q_values.argmax().item()
        return random.randrange(4)
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def train(self):
        if len(self.memory) < self.batch_size:
            return
            
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, training_params=None, metrics=None, model_name='stroop_model.pt', 
             training_params_name='training_params.json', metrics_name='metrics.json'):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = os.path.join(self.base_dir, f'model_{timestamp}')
        os.makedirs(save_dir)
        
        model_dict = {
            'state_size': self.state_size,
            'action_size': self.action_size,
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'hyperparameters': {
                'batch_size': self.batch_size,
                'gamma': self.gamma,
                'epsilon_decay': self.epsilon_decay,
                'epsilon_min': self.epsilon_min
            }
        }
        
        model_path = os.path.join(save_dir, model_name)
        torch.save(model_dict, model_path)
        
        if training_params:
            params_path = os.path.join(save_dir, training_params_name)
            with open(params_path, 'w') as f:
                json.dump(training_params, f, indent=4)
        
        if metrics:
            metrics_path = os.path.join(save_dir, metrics_name)
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
                
        print(f"模型已保存到: {save_dir}")
        return save_dir
    
    def load(cls, base_dir, model_name='stroop_model.pt',
             training_params_name='training_params.json', metrics_name='metrics.json'):
        model_path = os.path.join(base_dir, model_name)
        model_dict = torch.load(model_path)
        
        agent = cls(
            state_size=model_dict['state_size'],
            action_size=model_dict['action_size']
        )
        
        agent.policy_net.load_state_dict(model_dict['policy_net'])
        agent.target_net.load_state_dict(model_dict['target_net'])
        agent.optimizer.load_state_dict(model_dict['optimizer'])
        agent.epsilon = model_dict['epsilon']
        
        for key, value in model_dict['hyperparameters'].items():
            setattr(agent, key, value)
        
        params_path = os.path.join(base_dir, training_params_name)
        metrics_path = os.path.join(base_dir, metrics_name)
        
        training_params = None
        metrics = None
        
        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                training_params = json.load(f)
                
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                
        print(f"模型已从{base_dir} 加载")
        return agent, training_params, metrics

