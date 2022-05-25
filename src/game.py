import random
import pygame
from snake import Snake
from consts import *

class Game:
    def __init__(self, size=(21, 21), block_size=25, level=[], FPS = 10):
        """
        :param size: size of the grid
        :param block_size: size of each block of the grid
        :param level: list containing the coordinates of the obstacles in the grid
        :param food: boolean, 0 if there is no food on the grid, 1 otherwise
        """
        pygame.init()
        pygame.display.set_caption("SNAKE")
        self.WIDTH = size[0]
        self.HEIGHT = size[1]
        self.block_size = block_size
        self.level = level
        self.snake = Snake(0, 0)
        self.get_starting_position()
        self.SCREEN = pygame.display.set_mode((self.WIDTH * self.block_size, self.HEIGHT * self.block_size))
        self.clock = pygame.time.Clock()
        self.FPS = FPS
        self.font = pygame.font.Font(None, FONT_SIZE)
        self.SCORE_X = (self.WIDTH  -  4)* self.block_size
        self.SCORE_Y = 5
        self.food = False
        self.running = True
        self.finished = False
        self.score = 0

    def play(self):

        while self.running:
            self.clock.tick(self.FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            self.check_snake_position()
            if not self.food:
                self.spawn_food()
                self.food = True
            if not self.finished:
                self.snake.get_direction()
            self.snake.move(self.level)
            self.draw()

            if self.snake.head.x == self.food_x and self.snake.head.y == self.food_y:
                self.score += 1
                self.food = False
                self.snake.add_node(self.food_x, self.food_y)
            
            if self.snake.direction == "STOP":
                self.finished = True

    def check_snake_position(self):
        if self.snake.head.x >= self.WIDTH:
            self.snake.head.x = 0
        if self.snake.head.x < 0:
            self.snake.head.x = self.WIDTH
        if self.snake.head.y >=self.HEIGHT:
            self.snake.head.y = 0
        if self.snake.head.y < 0:
            self.snake.head.y = self.HEIGHT

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

    def draw(self):
        self.SCREEN.fill(BACKGROUD_COLOUR)

        # draw the grid
        for x in range(0, self.WIDTH * self.block_size, self.block_size):
            for y in range(0, self.HEIGHT * self.block_size, self.block_size):
                pygame.draw.rect(self.SCREEN, GRID_COLOUR, pygame.Rect(x, y, self.block_size, self.block_size), 1)
        # draw the snake
        self.snake.draw(self.SCREEN, self.block_size)
        # draw the food
        pygame.draw.rect(self.SCREEN, FOOD_COLOUR, pygame.Rect(self.food_x * self.block_size, self.food_y* self.block_size, self.block_size, self.block_size))
        # draw the obstacles
        for obstacle in self.level:
            x, y = obstacle[0] * self.block_size, obstacle[1] * self.block_size
            pygame.draw.rect(self.SCREEN, OBSTACLE_COLOUR, pygame.Rect(x, y, self.block_size, self.block_size))
        # draw the score
        img = self.font.render(f"Score: {self.score}", True, FONT_COLOUR)
        self.SCREEN.blit(img, (self.SCORE_X, self.SCORE_Y))

        pygame.display.update()


def main():
    level = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5,0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
        (20, 20), (19, 20), (18, 20), (17, 20), (16, 20), (15, 20), (20,19), (20, 18), (20, 17), (20, 16), (20, 15), 
        (0, 20), (0, 19), (0, 18), (0, 17), (0, 16), (0, 15), (1,20), (2, 20), (3, 20), (4, 20), (5, 20),
        (20, 0), (20, 1), (20, 2), (20, 3), (20, 4), (20,5), (19, 0), (18, 0), (17, 0), (16, 0), (15, 0),
    ]
    game = Game(size=(21, 21), level=level)
    game.play()

if __name__ == "__main__":
    main()