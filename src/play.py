import pygame
import random
from snake import *
from consts import *

def main():
    pygame.init()
    WIN = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("SNAKE")
    clock = pygame.time.Clock()
    font = pygame.font.Font(FONT, FONT_SIZE)

    snake = Snake(100, 100)
    score = 0
    food = False
    
    obstacles = [
        
    ]

    running = True

    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        check_position(snake)
        if food == False:
            x_food, y_food = spawn_food()
            food =  True

        WIN.fill(BACKGROUD_COLOUR)
        draw_grid(WIN)
        snake.get_direction()
        snake.move(obstacles)
        snake.draw(WIN)
        pygame.draw.rect(WIN, FOOD_COLOUR, pygame.Rect(x_food, y_food, SIZE, SIZE))
        for obstacle in obstacles:
            x, y = obstacle[0] * SIZE, obstacle[1] * SIZE
            rect = pygame.Rect(x, y, SIZE, SIZE)
            pygame.draw.rect(WIN, OBSTACLE_COLOUR, rect)
        img = font.render(f"Score: {score}", True, FONT_COLOUR)
        WIN.blit(img, (SCORE_X, SCORE_Y))

        if snake.head.x == x_food and snake.head.y == y_food:
            score += 1
            food = False
            snake.add_node(x_food, y_food)

        pygame.display.update()
    
def draw_grid(WIN):
    for x in range(0, WIDTH, SIZE):
        for y in range(0, HEIGHT, SIZE):
            pygame.draw.rect(WIN, GRID_COLOUR, pygame.Rect(x, y, SIZE, SIZE), width=1)

def check_position(snake):
    if snake.head.x >= WIDTH:
        snake.head.x = 0
    if snake.head.x < 0:
        snake.head.x = WIDTH
    if snake.head.y >= HEIGHT:
        snake.head.y = 0
    if snake.head.y < 0:
        snake.head.y = HEIGHT

def spawn_food():
    x_food, y_food = random.randrange(0, WIDTH-SIZE, SIZE), random.randrange(0, HEIGHT-SIZE, SIZE)
    return x_food, y_food

if __name__ == "__main__":
    main()