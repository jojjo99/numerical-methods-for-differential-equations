# Defines differential equation on the form u'=f(t,u)
import numpy as np
def f(t, u):
    return -3*u

# Real solution to differential equation
def f_real(u):
    return np.exp(-3*u)
