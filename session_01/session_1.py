# Session 1 Exercise Answers
# Group No. 5

import numpy as np
from keras import backend as K

# Ex 1 
# your code here

a = K.placeholder(shape=(5,))
b = K.placeholder(shape=(5,))
c = K.placeholder(shape=(5,))

inputs_computed = a*a + b*b + c*c + 2*b*c
function = K.function(inputs=(a,b,c), outputs(inputs_computed,))


# Ex 2
# your code here

x = K.placeholder(shape=())
inputs_tanh = K.exp(x)+K.exp(-x)/(K.exp(x)-K.exp(-x))) 

# Ex 3
# your code here

# Ex 4
# your code here

