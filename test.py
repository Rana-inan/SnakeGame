
#Importing the libraries
from environment import Environment
from brain import Brain
import numpy as np

#Defining the parameters
# wait time specify the time we wait after every move here 75 means 75milli seconds
# nLastStates no.of frame we stack on each other to give input to the neural network
# filepathToSave defines the place to store model episodes that performed well.
waitTime = 75
nLastStates = 4
filepathToOpen = 'model.h5'

#Initializing the Environment and the Brain
env = Environment(waitTime)
brain = Brain((env.nColumns, env.nRows, nLastStates))
model = brain.loadModel(filepathToOpen)

'''
NAME: resetStates
PARAMETERS: none
PURPOSE: we know that current state and next state contains the state of game before the action and next state is state after 
         the action. if we are resetting the environment it contains the states of the last game we played
         so its necessary to erase the states when we reset the environment
PRECONDITION: when game is over.
POSTCONDITION: It resets the current state and next states.
'''
def resetStates():
     
     currentState = np.zeros((1, env.nColumns, env.nRows, nLastStates))
     
     for i in range(nLastStates):
          currentState[0, :, :, i] = env.screenMap
     
     return currentState, currentState

while True:
     
     #Resetting the game and starting to play the game
     env.reset()
     currentState, nextState = resetStates()
     gameOver = False
     while not gameOver:
          
          #Selecting an action to play
          qvalues = model.predict(currentState)[0]
          action = np.argmax(qvalues)
          
          #Updating the environment and the current state
          frame, _, gameOver = env.step(action)

          frame = np.reshape(frame, (1, env.nColumns, env.nRows, 1))
          nextState = np.append(nextState, frame, axis = 3)
          nextState = np.delete(nextState, 0, axis = 3)
          
          currentState = nextState

