import random
import numpy as np
import itertools
from environment import SnakeGame

class QAgent:
    def __init__(self, n_features, n_actions):
        states = itertools.product([0,1], repeat=n_features)
        self.Q = {state : np.zeros(n_actions) for state in states}
        self.actions = list(range(n_actions))

    def choose_action(self, state, epsilon=0.0):
        if random.uniform(0, 1) < epsilon:
            action = random.choice(self.actions)
        else:
            action = np.argmax(self.Q[state])
        return action

    def get_state(self, observation):
        head = observation[0][0]
        food = observation[2]
        direction = observation[3]
        size = observation[4]
        x_head, y_head = head[0], head[1]
        x_food, y_food = food[0], food[1]

        dx = x_food - x_head
        dy = y_food - y_head
        u, d, l, r = 0, 0, 0, 0


        if dy > 0:
            d = 1
        else:
            u = 1
        if dx > 0:
            r = 1
        else:
            l = 1
        ub, db, lb, rb = 0, 0, 0, 0
        # check if there are any walls
        if x_head == size[0] - 1:
            rb = 1
        if y_head == size[1] - 1:
            db = 1
        if x_head == 0:
            lb = 1
        if y_head == 0:
            ub = 1
        
        above, below, left, right = y_head - 1, y_head + 1, x_head - 1, x_head + 1
        for nodes in observation[0][1:]:
            x, y = nodes[0], [1]
            if y == above:
                ub = 1
            if y == below:
                db = 1
            if x == left:
                lb = 1
            if x == right:
                rb = 1
        du, dd, dl, dr =  int(direction == 'UP'), int(direction == 'DOWN'), int(direction == 'LEFT'), int(direction == 'RIGHT')
        state = (u, d, l, r, ub, db, lb, rb, du, dd, dl, dr)
        return state

save_epochs = [1, 100, 200, 300]
import pickle

def test():
    episodes = 300
    max_steps = 2000
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.95
    discount = 0.95
    alpha = 0.1
    agent = QAgent(n_features=12, n_actions=4)
    env = SnakeGame()
    for ep in range(1, episodes+1):
        if ep in save_epochs:
            with open(f"./results/Q{ep}.pkl", "wb") as f:
                pickle.dump(agent.Q, f)
        steps = 0
        print(f"EPISODE {ep}", end="\t")
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        env.reset()

        a = random.choice([0,1,2,3]) 
        action = env.actions[a]
        observation, R, done = env.step(action)
        # Initial state
        s = agent.get_state(observation)
        while not env.game_over and steps < max_steps:
            a = agent.choose_action(s, epsilon)
            action = env.actions[a]
            if action == env.opposite[env.snake.direction]:
                action = env.snake.direction
            
            # Now I have s, a
            observation, R, done = env.step(action)
            steps += 1
            # Now I have s, a, r
            s_ = agent.get_state(observation)
            # Now I have s, a, r, s'
            agent.Q[s][a] = agent.Q[s][a] + alpha * (R + discount * np.max(agent.Q[s_]) - agent.Q[s][a])
            if ep % 50 == 0:
                env.render()
                env.clock.tick(24)
            s = s_    
        print(f"SCORE {env.score}   EPSILON {epsilon:.4f}")

if __name__ == "__main__":
    test()
