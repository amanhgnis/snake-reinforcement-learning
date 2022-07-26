import random
import numpy as np
import itertools
from environment import SnakeGame

class QAgent:
    def __init__(self, n_features, n_actions):
        self.name = "QAgent"
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

