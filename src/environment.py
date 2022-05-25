import random
import pygame
from snake import Snake
from consts import *

class SnakeGame:
    def __init__(self, size=(21,21), BLOCK_SIZE=25, level=[], FPS=10):
        self.actions = ["UP", "DOWN", "LEFT", "RIGHT"]
        self.level = level
        self.WIDTH = size[0]
        self.HEIGHT = size[1]
        self.BLOCK_SIZE = BLOCK_SIZE
        self.FPS = FPS
        pygame.init()
        self.SCREEN = pygame.display.set_mode((self.WIDTH * self.BLOCK_SIZE, self.HEIGHT * self.BLOCK_SIZE))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, FONT_SIZE)
        self.SCORE_X = (self.WIDTH  -  4)* self.BLOCK_SIZE
        self.SCORE_Y = 5
        self.reset()
 
    def reset(self):
        self.done = False
        self.observation = None
        self.action = None
        self.reward = None
        self.score = 0
        self.snake = Snake(0, 0)
        self.get_starting_position()
        self.food = False

    def step(self, action):
        self.reward = self.score
        self.check_snake_position()

        if not self.food:
            self.spawn_food()
            self.food = True

        self.snake.direction = action
        self.snake.move(self.level)

        if self.snake.head.x == self.food_x and self.snake.head.y == self.food_y:
            self.score += 1
            self.food = False
            self.snake.add_node(self.food_x, self.food_y)
        
        if self.snake.direction == "STOP":
            self.done = True

        self.reward -= self.score
        self.observation = [self.snake.nodes_positions, self.level, (self.food_x, self.food_y)]
        return self.observation, self.reward, self.done

    def get_starting_position(self):
        found_snake_location = False
        while not found_snake_location:
            x = random.randrange(0, self.WIDTH - 1, 1)
            y = random.randrange(0, self.HEIGHT - 1, 1)
            if (x,y) not in self.level:
                self.snake.head.x, self.snake.head.y = x, y
                found_snake_location = True      
    
    def spawn_food(self):
        found_food_location = False
        while not found_food_location:
            x = random.randrange(0, self.WIDTH - 1, 1)
            y = random.randrange(0, self.HEIGHT - 1, 1)
            if (x,y) not in self.snake.nodes_positions and (x,y) not in self.level:
                self.food_x, self.food_y = x, y
                found_food_location = True

    def check_snake_position(self):
        if self.snake.head.x >= self.WIDTH:
            self.snake.head.x = 0
        if self.snake.head.x < 0:
            self.snake.head.x = self.WIDTH
        if self.snake.head.y >=self.HEIGHT:
            self.snake.head.y = 0
        if self.snake.head.y < 0:
            self.snake.head.y = self.HEIGHT

    def render(self):
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


def test():
    env = SnakeGame()
    for _ in range(100):
        env.clock.tick(env.FPS)
        action = random.choice(env.actions)
        observation, reward, done = env.step(action)
        env.render()

if __name__ == "__main__":
    test()