
import numpy as np
'''
NAME            : Dqn
PURPOSE         : It creates the memory
INVARIANTS      : stores the memories based on the availble memroy size
'''
class Dqn():
     '''
          NAME: __init__
          PARAMETERS: maxMemory, discountfactor
          PURPOSE: To create and Intialise  variables maxMemory and discount
          PRECONDITION: it should given input maxMemory and discount
          POSTCONDITION: it returns the empty memory buffer is created to store the experiences.
     '''
     def __init__(self, maxMemory, discount):
          self.maxMemory = maxMemory
          self.discount = discount
          self.memory = list()

     '''
          NAME: remember
          PARAMETERS: transition[ current state, action, rewards and next state], gameOver
          PURPOSE: The function sets the memory buffer's size to the maximum permitted size and stores the transition and game-over flags there. 
          PRECONDITION: we should know transition state and whether game is over or not
          POSTCONDITION: if memory is full than it deletes the intial memories 
     '''
     def remember(self, transition, gameOver):
          self.memory.append([transition, gameOver])
          if len(self.memory) > self.maxMemory:
               del self.memory[0]

     '''
         NAME: getBatch
         PARAMETERS: model, batchSize
         PURPOSE: Getting batches of inputs and targets  and updates the targets based on the current state, action, reward, and next state.
         PRECONDITION: The memory should contain enough experiences to extract the desired batch size,The model should be compiled and trained before calling this function.
         POSTCONDITION:it returns the inputs and targets (corresponding Q values )
     '''
     def getBatch(self, model, batchSize):
          lenMemory = len(self.memory)
          numOutputs = model.output_shape[-1]
          
          #Initializing the inputs and targets
          inputs = np.zeros((min(batchSize, lenMemory), self.memory[0][0][0].shape[1], self.memory[0][0][0].shape[2], self.memory[0][0][0].shape[3]))
          targets = np.zeros((min(batchSize, lenMemory), numOutputs))
          
          #Extracting transitions from random experiences 
          for i, inx in enumerate(np.random.randint(0, lenMemory, size = min(batchSize, lenMemory))):
               currentState, action, reward, nextState = self.memory[inx][0]
               gameOver = self.memory[inx][1]
               
               #Updating inputs and targets
               inputs[i] = currentState
               targets[i] = model.predict(currentState)[0]
               if gameOver:
                    targets[i][action] = reward
               else:
                    targets[i][action] = reward + self.discount * np.max(model.predict(nextState)[0])
          
          return inputs, targets

          
