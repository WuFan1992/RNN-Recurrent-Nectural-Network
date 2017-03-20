import numpy as np
import RNN_layer
import CNN_forward
import Relu_function

def RNN_forward(self,input_array):

    self.times +=1

    state = (np.dot(self.U,input_array) + np.dot(self.W,self.state_list[-1]))

    treat_element(state,self.activator.forward)

    self.state_list.append(state)

    

    

    
