

import numpy as np
import pygame as pg
'''
NAME            : Environment
PURPOSE         : To create enviornment where the model or agent plays the game
INVARIANTS      : Step by step we add different layers to empty neural network
'''
class Environment():
    
    def __init__(self, waitTime):
        
        self.width = 500
        self.height = 500
        self.nRows = 10
        self.nColumns = 10
        self.initSnakeLen = 2
        self.defReward = -0.03
        self.negReward = -1.
        self.posReward = 2.
        self.waitTime = waitTime
        
        
        if self.initSnakeLen > self.nRows / 2:
            self.initSnakeLen = int(self.nRows / 2)
        
        self.screen = pg.display.set_mode((self.width, self.height))
        
        self.snakePos = list()
        self.screenMap = np.zeros((self.nRows, self.nColumns))
        
        for i in range(self.initSnakeLen):
            self.snakePos.append((int(self.nRows / 2) + i, int(self.nColumns / 2)))
            self.screenMap[int(self.nRows / 2) + i][int(self.nColumns / 2)] = 0.5
            
        self.applePos = self.placeApple()
        
        self.drawScreen()
        self.collected = False
        
        self.lastMove = 0

    '''
        NAME   : placeApple
        PARAMETERS: None
        PURPOSE: Randomly place a food for snake in the grid and set that cell to 1
        PRECONDITION: Once the snake ate the food this function get called
        POSTCONDITION: it returns the position where next food is to be placed
    '''
    def placeApple(self):
        posx = np.random.randint(0, self.nColumns)
        posy = np.random.randint(0, self.nRows)
        while self.screenMap[posy][posx] == 0.5:
            posx = np.random.randint(0, self.nColumns)
            posy = np.random.randint(0, self.nRows)
        
        self.screenMap[posy][posx] = 1
        
        return (posy, posx)

    '''
        NAME: drawScreen
        PARAMETERS: None
        PURPOSE: To create the grid with snake position (green color rectangle) and food (apple) 
                 and rest as black color
        PRECONDITION: No pre condition
        POSTCONDITION: No post condition
    '''
    def drawScreen(self):

        self.screen.fill((0, 0, 0))

        cellWidth = self.width / self.nColumns
        cellHeight = self.height / self.nRows

        for i in range(self.nRows):
            for j in range(self.nColumns):
                if self.screenMap[i][j] == 0.5:
                    image = pg.image.load("Untitled design2.png").convert()
                    image = pg.transform.scale(image, (50, 50))
                    rect = pg.draw.rect(self.screen, (3, 168, 20),
                                        (j * cellWidth + 1, i * cellHeight + 1, cellWidth - 2, cellHeight - 2),
                                        border_radius=20)
                    self.screen.blit(image, rect)
                elif self.screenMap[i][j] == 1:
                    image = pg.image.load("RedApple.png").convert()
                    image = pg.transform.scale(image, (55, 55))
                    rect = pg.draw.rect(self.screen, (255, 0, 0),
                                        (j * cellWidth + 1, i * cellHeight + 1, cellWidth - 2, cellHeight - 2),
                                        border_radius=80)
                    self.screen.blit(image, rect)
        # update the screen with changes
        pg.display.flip()

    '''
        NAME: moveSnake
        PARAMETERS: nextPos (Nextpos is the position of the first cell in the next stage) , col (Specifies  whether we have collected an apple by performing this action.)
        PURPOSE: To change snake position in the grid based on the present condition
        PRECONDITION: col is true if snake collected food else it is set false
        POSTCONDITION: The function updates the collect value if not 
    '''
    def moveSnake(self, nextPos, col):
        
        self.snakePos.insert(0, nextPos)
        
        if not col:
            self.snakePos.pop(len(self.snakePos) - 1)
        
        self.screenMap = np.zeros((self.nRows, self.nColumns))
        
        for i in range(len(self.snakePos)):
            self.screenMap[self.snakePos[i][0]][self.snakePos[i][1]] = 0.5
        
        if col:
            self.applePos = self.placeApple()
            self.collected = True
            
        self.screenMap[self.applePos[0]][self.applePos[1]] = 1

    '''
        NAME: step
        PARAMETERS: action
        PURPOSE: Based on the action performed it defines the rewards and game condition(game over or not)
        PRECONDITION: it takes action (index) as input
        POSTCONDITION: it returns screenMap, reward, gameOver
    '''
    def step(self, action):
        # action = 0 -> up
        # action = 1 -> down
        # action = 2 -> right
        # action = 3 -> left
        gameOver = False
        reward = self.defReward
        self.collected = False
        
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return
        
        snakeX = self.snakePos[0][1]
        snakeY = self.snakePos[0][0]
        
        if action == 1 and self.lastMove == 0:
            action = 0
        if action == 0 and self.lastMove == 1:
            action = 1
        if action == 3 and self.lastMove == 2:
            action = 2
        if action == 2 and self.lastMove == 3:
            action = 3
        '''
            we check  what lies in front of snake
            if there is snake body (0.5) in front of snake than game is over and we give negative reward
            if there is a food(1) in front of the snake than we gave positive reward
            if there is nothing (0) we gave the little negative reward which is def reward
            else if its a wall we give negative reward and make game end
        '''
        if action == 0:
            if snakeY > 0:
                if self.screenMap[snakeY - 1][snakeX] == 0.5:
                    gameOver = True
                    reward = self.negReward
                elif self.screenMap[snakeY - 1][snakeX] == 1:
                    reward = self.posReward
                    self.moveSnake((snakeY - 1, snakeX), True)
                elif self.screenMap[snakeY - 1][snakeX] == 0:
                    self.moveSnake((snakeY - 1, snakeX), False)
            else:
                gameOver = True
                reward = self.negReward
                
        elif action == 1:
            if snakeY < self.nRows - 1:
                if self.screenMap[snakeY + 1][snakeX] == 0.5:
                    gameOver = True
                    reward = self.negReward
                elif self.screenMap[snakeY + 1][snakeX] == 1:
                    reward = self.posReward
                    self.moveSnake((snakeY + 1, snakeX), True)
                elif self.screenMap[snakeY + 1][snakeX] == 0:
                    self.moveSnake((snakeY + 1, snakeX), False)
            else:
                gameOver = True
                reward = self.negReward
                
        elif action == 2:
            if snakeX < self.nColumns - 1:
                if self.screenMap[snakeY][snakeX + 1] == 0.5:
                    gameOver = True
                    reward = self.negReward
                elif self.screenMap[snakeY][snakeX + 1] == 1:
                    reward = self.posReward
                    self.moveSnake((snakeY, snakeX + 1), True)
                elif self.screenMap[snakeY][snakeX + 1] == 0:
                    self.moveSnake((snakeY, snakeX + 1), False)
            else:
                gameOver = True
                reward = self.negReward 
        
        elif action == 3:
            if snakeX > 0:
                if self.screenMap[snakeY][snakeX - 1] == 0.5:
                    gameOver = True
                    reward = self.negReward
                elif self.screenMap[snakeY][snakeX - 1] == 1:
                    reward = self.posReward
                    self.moveSnake((snakeY, snakeX - 1), True)
                elif self.screenMap[snakeY][snakeX - 1] == 0:
                    self.moveSnake((snakeY, snakeX - 1), False)
            else:
                gameOver = True
                reward = self.negReward
                
        self.drawScreen()
        
        self.lastMove = action
        
        pg.time.wait(self.waitTime)
        
        return self.screenMap, reward, gameOver

    '''
        NAME: reset
        PARAMETERS: none
        PURPOSE: To reset the environment frame
        PRECONDITION: when game is over.
        POSTCONDITION: It resets the screen.
    '''
    def reset(self):
        self.screenMap  = np.zeros((self.nRows, self.nColumns))
        self.snakePos = list()
        
        for i in range(self.initSnakeLen):
            self.snakePos.append((int(self.nRows / 2) + i, int(self.nColumns / 2)))
            self.screenMap[int(self.nRows / 2) + i][int(self.nColumns / 2)] = 0.5
        
        self.screenMap[self.applePos[0]][self.applePos[1]] = 1
        
        self.lastMove = 0
        
        self.drawScreen()
'''
    AUTHOR: Pavan Kumar Ganguru
    FILENAME: environment.py
    SPECIFICATION: environment setup and game rule functions
    FOR: CS 5392 Reinforcement Learning Section 001
'''
if __name__ == '__main__':
     env = Environment(500)
     gameOver = False
     start = False
     action = 0
     while True:
          for event in pg.event.get():
               if event.type == pg.KEYDOWN:
                    if event.key == pg.K_SPACE and not start:
                         start = True
                    elif event.key == pg.K_SPACE and start:
                         start = False
                    if event.key == pg.K_UP:
                         action = 0
                    elif event.key == pg.K_DOWN:
                         action = 1
                    elif event.key == pg.K_RIGHT:
                         action = 2
                    elif event.key == pg.K_LEFT:
                         action = 3
          
          if start:
               _, _, gameOver = env.step(action)
               
          if gameOver:
               start = False
               gameOver = False
               env.reset()
               action = 0
               
              
