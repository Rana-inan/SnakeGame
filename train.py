
#Importing the libraries
from environment import Environment
from brain import Brain
from dqn import Dqn
import numpy as np
import matplotlib.pyplot as plt

# Setting the parameters
# learning rate is a parameter that controls how much to change the model in response to the estimated error each time
# maxMemory defined the maximum number of records/experiences our model can store
# gamma can be used to update Q values based on this model will learn
# batchSize-it defines how many training samples from the data set the algorithm takes and train the network at
# epsilon defines  to take an action, we have Epsilon chance of it happening
# epsilonDecayRate(<1) is the rate epsilon should decrease so that model learns well
# nLastStates defines how many frames we stack on each other so our AI can see IN which direction it is  going
# minEpsilon defines what is the minimum value of epsilon needed to be, it should not be lesser than the specified minEpsilon
# filepathToSave defines the place to store model episodes that performed well.

learningRate = 0.00001
maxMemory = 100000
gamma = 0.9
batchSize = 16
nLastStates = 4

epsilon = 1.
epsilonDecayRate = 0.0002
minEpsilon = 0.05

filepathToSave = 'model3.h5'

# Initializing the Environment,Brain(inputs(Snake game window columns ,Snake game rows , states),learningRate) and DQN
# calling model and the Experience Replay Memory
env = Environment(0)
brain = Brain((env.nColumns, env.nRows, nLastStates), learningRate)
model = brain.model
DQN = Dqn(maxMemory, gamma)

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

#Starting the main loop
epoch = 0
nCollected = 0
maxNCollected = 0
totNCollected = 0
scores = list()
while True:
     epoch += 1
     
     #Resetting the Evironment and starting to play the game
     env.reset()
     currentState, nextState = resetStates()
     gameOver = False
     while not gameOver:
          
          #Selecting an action to play
          if np.random.rand() <= epsilon:
               action = np.random.randint(0, 4)
          else:
               qvalues = model.predict(currentState)[0]
               action = np.argmax(qvalues)

          #Updating the Environment
          frame, reward, gameOver = env.step(action)
          
          frame = np.reshape(frame, (1, env.nColumns, env.nRows, 1))
          nextState = np.append(nextState, frame, axis = 3)
          nextState = np.delete(nextState, 0, axis = 3)
          
          #Remembering new experience and training the AI
          DQN.remember([currentState, action, reward, nextState], gameOver)
          inputs, targets = DQN.getBatch(model, batchSize)
          model.train_on_batch(inputs, targets)
          
          #Updating the score and current state
          if env.collected:
               nCollected += 1
          
          currentState = nextState

     #Updating the epsilon and saving the model
     epsilon -= epsilonDecayRate
     epsilon = max(epsilon, minEpsilon)
     
     if nCollected > maxNCollected and nCollected > 2:
          model.save(filepathToSave)
          maxNCollected = nCollected
          
     #Displaying the results
     totNCollected += nCollected
     nCollected = 0

     if epoch % 10 == 0 and epoch != 0:
          scores.append(totNCollected / 10)
          totNCollected = 0
          plt.plot(scores)
          plt.xlabel('Epoch / 10')
          plt.ylabel('Average Collected')
          plt.show()
     
     print('Epoch: ' + str(epoch) + ' Current Best: ' + str(maxNCollected) + ' Epsilon: {:.5f}'.format(epsilon))
     
     
     
     
     


