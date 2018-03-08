####################################################################################################
#                                           session_1.py                                           #
####################################################################################################
# Author: Group Nr. 5                                                                              #
#                                                                                                  #
# Purpose: Session 1/ Learning...                                                                  #
#                                                                                                  #
# Version: 1.1                                                                                     #
#                                                                                                  #
#!/usr/bin/env python                                                                              #
####################################################################################################

import numpy as np
from keras import backend as K


# Ex 1

a = K.placeholder(shape = (5,))
b = K.placeholder(shape = (5,))
c = K.placeholder(shape = (5,))

result_f_abc = a * a + b * b + c * c + 2 * b * c

function_abc = K.function(inputs = (a, b, c), outputs = (result_f_abc,))

print( "a = b = c = [1, 1, 1, 1, 1]:\na^2 + b^2 + c2 + 2bc =" ),
print( function_abc( (np.array([1, 1, 1, 1, 1]),
                      np.array([1, 1, 1, 1, 1]),
                      np.array([1, 1, 1, 1, 1])) ) )
print( '' )   # newline (somehow print(... and ...) doesnt work properly)


# Ex 2

x = K.placeholder(shape = (), dtype = 'float32')
tanh = (K.exp(x) - K.exp(-x))/(K.exp(x) + K.exp(-x))
grad_tanh = K.gradients(loss = tanh, variables = [x])

function_tanh = K.function(inputs = (x,), outputs = (tanh,))
function_grad_tanh = K.function(inputs = (x,), outputs = (grad_tanh))

print( "tanh at (-100 -1 0 1 100):" )
print( function_tanh((-100,)) ),
print( function_tanh((-1,)) ),
print( function_tanh((0,)) ),
print( function_tanh((1,)) ),
print( function_tanh((100,)) )

print( "\ngradient of tanh at (-100 -1 0 1 100):" )
print( function_grad_tanh((-100,)) ),
print( function_grad_tanh((-1,)) ),
print( function_grad_tanh((0,)) ),
print( function_grad_tanh((1,)) ),
print( function_grad_tanh((100,)) )


# Ex 3

b = K.ones(shape = (1,))
w = K.ones(shape = (2,))

x = K.placeholder(shape = (2,))

result_f_bwx = 1.0/(1 + K.exp(-1 * (w[0] * x[0] + w[1] * x[1] + b)))
grad_f_bwx = K.gradients(loss = result_f_bwx, variables = [w])

function_bwx = K.function(inputs = (b, w, x), outputs = (result_f_bwx,))
function_grad_bwx = K.function(inputs = (b, w, x), outputs = (grad_f_bwx))

print( "\nb = 1, w = x = [1, 1]:\n1/(1 + exp(- (w1 * x1 + w2 * x2 + b))) =" ),
print( function_bwx((np.array([1]), np.array([1, 1]), np.array([1, 1]))) )
print( "b = 2, w = x = [1, 1]:\n1/(1 + exp(- (w1 * x1 + w2 * x2 + b))) =" ),
print( function_bwx((np.array([2]), np.array([1, 1]), np.array([1, 1]))) )
print( "b = 1, w = x = [2, 2]:\n1/(1 + exp(- (w1 * x1 + w2 * x2 + b))) =" ),
print( function_bwx((np.array([2]), np.array([1, 1]), np.array([1, 1]))) )

print( "\nb = 1, w = x = [1, 1]:\ngrad ( 1/(1 + exp(- (w1 * x1 + w2 * x2 + b))) ) =" ),
print( function_grad_bwx((np.array([1]), np.array([1, 1]), np.array([1, 1]))) )
print( "b = 2, w = x = [1, 1]:\ngrad ( 1/(1 + exp(- (w1 * x1 + w2 * x2 + b))) ) =" ),
print( function_grad_bwx((np.array([2]), np.array([1, 1]), np.array([1, 1]))) )
print( "b = 1, w = x = [2, 2]:\ngrad ( 1/(1 + exp(- (w1 * x1 + w2 * x2 + b))) ) =" ),
print( function_grad_bwx((np.array([1]), np.array([2, 2]), np.array([1, 1]))) )


# Ex 4

# not sure what to do...
# how can one have: "... input scalar variable x with (n+1) variables ..."
