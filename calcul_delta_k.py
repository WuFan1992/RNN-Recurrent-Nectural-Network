import numpy as np
import RNN_layer
import CNN_forward


def calcul_delat_k(self,k,activator):

    state = self.state_list[k+1]

    treat_element(state,activator.backward)

    deltat_k = np.dot((np.dot(self.state_list[k+1].T,self.W)),np.diag(state[:])).T

    

    

    
