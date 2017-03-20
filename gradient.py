import numpy as np



def gradient_t(self,t):

    gradient_t = np.dot(self.deltat_list[t].T,self.state_list[t-1])
    self.gradient_list[t] = gradient_t


def gradient(self):

    self.gradient_list=[]
    for i in range(self.times):
        self.gradient_list.append(np.zeros((self.state_width,1))
                                  
    for j in range(self):
        self.gradient_t(j)
        self.gradient_list = reduce(lambda a,b :a+b,self.gradient_list,self.gradient_list[0])
                                  
        
                                  
                                
    
