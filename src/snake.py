import pygame
from consts import *

class Node:
    def __init__(self, x, y, next_node=None):
        self.next_node = next_node
        self.x = x
        self.y = y   


class Snake:
    """"""
    def __init__(self, x=0, y=0):
        """
        Initialize the snake with one node. Head and tail coincides at this point.
        """
        self.head = Node(x, y)
        self.tail = self.head
        self.direction = None
        self.nodes_positions = []

    def add_node(self, x, y):
        """
        """
        temp = Node(x, y)
        self.tail.next_node =  temp
        self.tail = temp

    def get_direction(self):
        key = pygame.key.get_pressed()
        if key[pygame.K_LEFT]:
            self.direction = "LEFT"
        if key[pygame.K_RIGHT]:
            self.direction = "RIGHT"
        if key[pygame.K_UP]:
            self.direction = "UP"
        if key[pygame.K_DOWN]:
            self.direction = "DOWN"
        
    def move(self, obstacles_list=[]):
        """
        """
        # Get the new coordinates of the head
        new_x, new_y = self.head.x, self.head.y
        if self.direction == "RIGHT":
            new_x += SIZE
        if self.direction == "LEFT":
            new_x -= SIZE
        if self.direction == "UP":
            new_y -= SIZE
        if self.direction == "DOWN":
            new_y += SIZE
        if self.direction == "STOP":
           return 

        for obstacle in obstacles_list:
            x, y = obstacle[0] * SIZE, obstacle[1] * SIZE
            if new_x == x and new_y == y:
                self.direction = "STOP"
                return

        for node_position in self.nodes_positions:
            x, y = node_position[0], node_position[1]
            if new_x == x and new_y == y:
                self.direction = "STOP"
                return

        self.nodes_positions = []
        # now move each node
        current_node = self.head
        while current_node.next_node != None:
            current_x, current_y = current_node.x, current_node.y           # get current node position
            current_node.x, current_node.y =  new_x, new_y                  # update current node position
            self.nodes_positions.append((current_node.x, current_node.y))   
            new_x, new_y = current_x, current_y                             # current node position will become next node position
            current_node = current_node.next_node
        # now move the tail
        current_x, current_y = current_node.x, current_node.y
        current_node.x, current_node.y =  new_x, new_y
        self.nodes_positions.append((current_node.x, current_node.y))

    def draw(self, WIN):
        current_node = self.head
        while current_node.next_node != None:   
            rect = pygame.Rect(current_node.x, current_node.y, SIZE, SIZE)
            pygame.draw.rect(WIN, SNAKE_COLOUR, rect)
            current_node = current_node.next_node
        # draw tail
        rect = pygame.Rect(current_node.x, current_node.y, SIZE, SIZE)
        pygame.draw.rect(WIN, SNAKE_COLOUR, rect)
        
        
