import numpy as np

# define the class of RNN

class RNN_layer(object):

    def _init_(self,input_width,state_width,learning_rate,activators):

        self.input_width = input_width # the dimension of  x

        self.state_width = state_width  # the dimension of each state

        self.activators = activators

        self.learning_rate = learning_rate

        self.U = np.randoms.uniform(-1e-4,1e-4,(state_width,input_width))

        self.W = np.zeros(-1e-4,1e-4,(state_width,state_width))

        # we also need to save each state

        self.state_list = []
        self.state_list.append(np.zeros(state_width,1))

        self.times = 0
