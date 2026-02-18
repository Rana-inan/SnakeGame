
#Importing the libraries
import keras
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam

'''
NAME            : Brain
PURPOSE         : To create neural network completely
INVARIANTS      : Step by step we add different layers to empty neural network
'''
class Brain():
     '''
          NAME: __init__
          PARAMETERS: inputShape,learning rate
          PURPOSE: To create complete convolutional neural network by adding layer by layer
          PRECONDITION: There is no pre condition.
          POSTCONDITION: it returns the neural network
     '''
     def __init__(self, inputShape, lr = 0.005):
          self.inputShape = inputShape
          self.learningRate = lr
          self.numOutputs = 4
          
          #Creating the neural network
          self.model = Sequential()

          # we are creating 32 filter(To extarct 32 features) each of size 3 * 3 
          self.model.add(Conv2D(32, (3,3), activation = 'relu', input_shape = self.inputShape))
          
          self.model.add(MaxPooling2D((2, 2)))
          
          self.model.add(Conv2D(64, (2,2), activation = 'relu'))
          
          self.model.add(Flatten())
          #Full connection layers 
          self.model.add(Dense(256, activation = 'relu'))
          
          self.model.add(Dense(self.numOutputs))
          
          self.model.compile(optimizer = Adam(lr = self.learningRate), loss = 'mean_squared_error')
          
     #Building a method that will load a model
     '''
          NAME: loadModel
          PARAMETERS: filepath
          PURPOSE: Load a pre trained model from the specified path.
          PRECONDITION: It should have file model saved
          POSTCONDITION: it returns the model saved.
     '''
     def loadModel(self, filepath):
          self.model = load_model(filepath)
          return self.model

