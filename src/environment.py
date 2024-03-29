from snake import Snake
import random
import pygame
from consts import *
import numpy as np
import pickle

class SnakeGame():
    def __init__(self, board_size=(32,32), level=[], block_size=15):
        self.WIDTH, self.HEIGHT = board_size[0], board_size[1]
        self.SCREEN = None
        self.BLOCK_SIZE = block_size
        self.level = level
        self.SCORE_X = (self.WIDTH - 8) * self.BLOCK_SIZE
        self.SCORE_Y = 1 * self.BLOCK_SIZE
        self.reset()
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.opposite = {'UP':'DOWN', 'DOWN':'UP', 'LEFT':'RIGHT', 'RIGHT':'LEFT','STOP':'', None : None}

    def reset(self):
        if self.SCREEN != None:
            pygame.quit()
            self.SCREEN = None
        self.game_over = False
        self.snake = Snake(0,0)
        self.snake.add_node(0,0)
        self.get_starting_position()
        self.food = False
        self.score = 0
        self.reward = None

    def step(self, a):
        action = self.actions[a]
        if action == self.opposite[self.snake.direction]:
            action = self.snake.direction
        self.snake.direction = action

        self.reward = 0
        if not self.food:
            self.spawn_food()
            self.food = True
        prev_x, prev_y = self.snake.head.x, self.snake.head.y
        self.snake.previous_direction = self.snake.direction

        self.check_snake_position()
        collision = self.snake.move(self.level)
        if collision:
            self.game_over =True
        if self.game_over:
            self.reward = -100
            #self.observation = [self.snake.nodes_positions, self.reward, (self.food_x, self.food_y), self.snake.direction, (self.WIDTH, self.HEIGHT)]
            return self.observation, self.reward, self.game_over
        

        if self.snake.head.x == self.food_x and self.snake.head.y == self.food_y:
            self.score += 1
            self.reward += 100
            self.snake.add_node(self.food_x, self.food_y)
            self.spawn_food()

        if (abs(prev_x - self.food_x) > abs(self.snake.head.x - self.food_x)) or (abs(prev_y - self.food_y) > abs(self.snake.head.y - self.food_y)):
           self.reward += 1
        else:
            self.reward -= 1
        
        self.observation = [self.snake.nodes_positions, self.reward, (self.food_x, self.food_y), self.snake.direction, (self.WIDTH, self.HEIGHT)]
        return self.observation, self.reward, self.game_over

    def get_starting_position(self):
        found_snake_location = False
        while not found_snake_location:
            x = random.randrange(0, self.WIDTH - 1, 1)
            y = random.randrange(0, self.HEIGHT - 1, 1)
            if (x,y) not in self.level:
                self.snake.head.x, self.snake.head.y = x, y
                found_snake_location = True 
  
    def check_snake_position(self):
        if self.snake.direction == 'STOP':
            self.game_over = True
        if self.snake.head.x == (self.WIDTH - 1) and self.snake.direction == 'RIGHT':
            #print("HIT RIGHT")
            self.snake.direction = 'STOP'
            self.game_over = True
        if self.snake.head.x == 0 and self.snake.direction == 'LEFT':
            #print("HIT LEFT")
            self.snake.direction = 'STOP'
            self.game_over = True
        if self.snake.head.y == (self.HEIGHT - 1) and self.snake.direction == 'DOWN':
            #print("HIT BOTTOM")
            self.snake.direction = 'STOP'
            self.game_over = True
        if self.snake.head.y == 0 and self.snake.direction == 'UP':
            #print("HIT UP")
            self.snake.direction = 'STOP'
            self.game_over = True
    
    def spawn_food(self):
        found_food_location = False
        while not found_food_location:
            x = random.randrange(0, self.WIDTH - 1, 1)
            y = random.randrange(0, self.HEIGHT - 1, 1)
            if (x,y) not in self.snake.nodes_positions and (x,y) not in self.level:
                self.food_x, self.food_y = x, y
                found_food_location = True

    def render(self):
            if self.SCREEN == None:
                pygame.init()
                pygame.display.set_caption("Snake")
                self.SCREEN = pygame.display.set_mode((self.WIDTH * self.BLOCK_SIZE, self.HEIGHT * self.BLOCK_SIZE))
                self.clock = pygame.time.Clock()
                self.font = pygame.font.Font(None, FONT_SIZE)
            self.SCREEN.fill(BACKGROUD_COLOUR)

            # draw the grid
            for x in range(0, self.WIDTH * self.BLOCK_SIZE, self.BLOCK_SIZE):
                for y in range(0, self.HEIGHT * self.BLOCK_SIZE, self.BLOCK_SIZE):
                    pygame.draw.rect(self.SCREEN, GRID_COLOUR, pygame.Rect(x, y, self.BLOCK_SIZE, self.BLOCK_SIZE), 1)
            # draw the snake
            self.snake.draw(self.SCREEN, self.BLOCK_SIZE)
            # draw the food
            pygame.draw.rect(self.SCREEN, FOOD_COLOUR, pygame.Rect(self.food_x * self.BLOCK_SIZE, self.food_y * self.BLOCK_SIZE, self.BLOCK_SIZE, self.BLOCK_SIZE))
            # draw the obstacles
            for obstacle in self.level:
                x, y = obstacle[0] * self.BLOCK_SIZE, obstacle[1] * self.BLOCK_SIZE
                pygame.draw.rect(self.SCREEN, OBSTACLE_COLOUR, pygame.Rect(x, y, self.BLOCK_SIZE, self.BLOCK_SIZE))
            # draw the score
            img = self.font.render(f"Score: {self.score}", True, FONT_COLOUR)
            self.SCREEN.blit(img, (self.SCORE_X, self.SCORE_Y))

            pygame.display.update()
