import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from environment import SnakeGame

class ReplayMemory(object):
    """
    Implements an experience replay memory
    """
    def __init__(self, capacity):
        """
        :param capacity: capacity of the experience replay memory
        """
        self.memory = deque(maxlen=capacity)

    def __len__(self):
        """
        :returns: size of the experience replay memory
        """
        return len(self.memory)

    def push(self, state, action, reward, next_state, done):
        """
        Stores a tuple (s, a, r, s', done)
        :param state: current state s
        :param action: action taken a
        :param reward: reward r obtained taking action a being in state s
        :param next_state: state s' reached taking action a being in state s
        :param done: True if a final state is reached
        """
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        :returns: a sample of size batch_size
        """
        batch_size = min(batch_size, len(self.memory))
        return random.sample(self.memory, batch_size)


class DQN(nn.Module):
    """
    Simple feed-forward neural network with two fully connected layers
    """
    def __init__(self, input_dim, output_dim, activation):
        """
        :param input_dim: size of the state vector
        :param output_dim: number of possible actions
        :param activation: activation function for the hidden layers
        """
        super().__init__()
        self.fc1 = nn.Linear(in_features=input_dim, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=512)
        self.out = nn.Linear(in_features=512, out_features=output_dim)
        self.act = activation

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.out(x)
        return x

class DeepQAgent:
    def __init__(self, n_features=8, n_actions=4, discount=0.95, batch_size=32):
        self.replay_memory = ReplayMemory(10000)
        self.discount = discount
        self.policy_net = DQN(n_features, n_actions, nn.ReLU())
        self.target_net = DQN(n_features, n_actions, nn.ReLU())
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.training_iterations = 0
        self.replace_every = 10
        self.batch_size = batch_size

    def get_state(self, observation):
        head = observation[0][0]
        nodes = observation[0]
        food = observation[2]
        direction = observation[3]
        size = observation[4]
        x_head, y_head = head[0], head[1]
        x_food, y_food = food[0], food[1]

        dx = x_food - x_head
        dy = y_food - y_head
        
        state = np.zeros((9,9))
        for i in range(-5,4):
            for j in range(-5,4):
                x = x_head + i 
                y = y_head + j 
                for node in nodes:
                    if x == node[0] and y == node[1]:
                        state[j+4][i+4] = 1
                if (x == 0 or x == size[0] -1) or (y == 0 or y == size[1] - 1):
                    state[j+4][i+4] = 1

        state = state.flatten()
        state = list(state)
        du, dd, dl, dr =  int(direction == 'UP'), int(direction == 'DOWN'), int(direction == 'LEFT'), int(direction == 'RIGHT')
        state = state + [dx, dy, du, dd, dl, dr]
        return state

    def choose_action(self, state, epsilon):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        if np.random.uniform(0,1) < epsilon:
            action = np.random.choice([0,1,2,3])
        else:
            self.policy_net.eval()
            with torch.no_grad():
                output = self.policy_net(state)
            action = int(output.argmax())
        return action

    def learn(self, s, a, r, s_, done):
        self.replay_memory.push(s, a, r, s_, done)

        if len(self.replay_memory) < self.batch_size:
            return
        
        self.training_iterations += 1 
        batch = self.replay_memory.sample(self.batch_size)

        states = torch.tensor([x[0] for x in batch], dtype=torch.float32)
        actions = torch.tensor([x[1] for x in batch], dtype=torch.int64)
        rewards = torch.tensor([x[2] for x in batch], dtype=torch.float32)
        next_states = torch.tensor([x[3] for x in batch], dtype=torch.float32)

        self.policy_net.train()                                     
        q_values = self.policy_net(states)                          # computes Q(s, a_1), Q(s, a_2), ... , Q(s, a_n)
        q_state_action = q_values.gather(1, actions.unsqueeze(1))   # gets the right Q(s, a)

        with torch.no_grad():
            self.target_net.eval()                                  
            target_q_values = self.target_net(next_states)          # computes Q'(s', a_1), Q'(s', a_2), ..., Q'(s', a_n)
        next_state_max_q = target_q_values.max(dim=1)[0]            # gets max_a {Q(s', a)}

        target = rewards + self.discount * next_state_max_q         # r + discount * Q_max(s)
        target = target.unsqueeze(1)
        
        loss = self.loss_fn(q_state_action, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu()
        
    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


def test():
    episodes = 10000
    max_steps = 2000
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.99
    discount = 0.99
    agent = DeepQAgent(n_features=87, n_actions=4, batch_size=512)
    env = SnakeGame(board_size=(32,32),block_size=15)
    for ep in range(1, episodes+1):
        steps = 0
        total_reward = 0
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        env.reset()

        a = random.choice([0,1,2,3]) 
        action = env.actions[a]
        observation, r, done = env.step(action)
        # Initial state
        s = agent.get_state(observation)
        env.render()
        while not env.game_over and steps < max_steps:
            a = agent.choose_action(s, epsilon)
            action = env.actions[a]
            if action == env.opposite[env.snake.direction]:
                action = env.snake.direction
            
            # Now I have s, a
            observation, r, done = env.step(action)
            steps += 1
            # Now I have s, a, r
            s_ = agent.get_state(observation)

            # Now I have s, a, r, s'
            loss = agent.learn(s, a, r, s_, done)
            if ep % 1 == 0:
                env.render()
                env.clock.tick(24)
            s = s_
            total_reward += r
        if loss == None:
            loss = 0
        print(f"EPISODE {ep}    SCORE {env.score}   EPSILON {epsilon:.4f}   TOTAL REWARD {total_reward}    LOSS {loss:.4f}")

        if ep % agent.replace_every == 0:
            print(f"UPDATING TARGET NETWORK")
            agent.update_target_net()


if __name__ == "__main__":
    test()
