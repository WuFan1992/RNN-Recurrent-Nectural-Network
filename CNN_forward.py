import numpy as np
import class_convolayer
import class_filter
import zero_padding
import Relu_function

# elements function
# this function is used to treat the elements with a function we want to use

def treat_element(input_array,funcion_treate):
    for i in np.nditer(input_array,op_flags=['readwrite']):
        i[...]= function_treate(i)
    



# forward of convolution layer

def feedforward(self, input_array):
    
    self.input_array = input_array
    expand_input = zero_padding(input_array,self.zero_padding)

    for f in range(self.filter_number):

        filter_forward = self.filter_total[f] 

        conv_2d(expand_input,filter_forward.get_weight(),self.output_array[f],self.stride,self.zero_padding,self.filter_forward.get_bias())

    treate_element(self.output_array,self.activator.Relu_forward)

        

        



    
