## Generalized RK method. Input matrix A, b, c
import numpy as np
def RKnstep(f, told, uold, h, c, b, A):
  unew=uold
  Y_prime = []
  for j in range(len(b)):
    Y_prime.append(getYj(c, h, f, uold, told, j, A, Y_prime))
    unew+=b[j]*Y_prime[-1]*h
  return unew

# Method to compute next Yj_prime based on a list of previous Y_prime and matrices A, b, c.
def getYj(c, h, f, uold, told, j, A, Y_prime):
  local_u = uold
  for i in range(len(Y_prime)):
    local_u+=Y_prime[i]*A[j][i]*h
  return local_u+c[j]*h+f(told + c[j]*h, local_u)

def RK3step(f, told, uold, h):
  A=np.array([[0, 0, 0],
              [1/2, 0, 0],
              [-1, 2, 0]])
  c=np.array([0, 1/2, 1])
  b=np.array([1/6, 2/3, 1/6])
  return RKnstep(f, told, uold, h, c, b, A)

def RK4step(f, told, uold, h):
  c=np.array([0, 1/2, 1/2, 1])
  b=np.array([1/6, 1/3, 1/3, 1/6])
  A=np.array([
    [0, 0, 0, 0 ],
    [1/2, 0, 0, 0 ],
    [0, 1/2, 0, 0 ],
    [0, 0, 1, 0] ])
  return RKnstep(f, told, uold, h, c, b, A)
  

def RK34step(f, told, uold, h):
  RK4=RK4step(f, told, uold, h)
  RK3=RK3step(f, told, uold, h)
  return [RK4, RK4-RK3] 

# Computes an approximation for y_n with RKn.
def RKn(f, initial_value, start_time, end_time, steps, f_real, RKstep_function):
    time_grid = np.linspace(start_time, end_time, steps)
    exact_values = [f_real(t) for t in time_grid]
    stepsize = (end_time-start_time)/steps
    approximations = [initial_value]
    for i in range(1, steps): # Här gjorde vi ett steg för mycket
        approximations.append(RKstep_function(f, time_grid[i], approximations[-1], stepsize))
    errors = [approximations[i]-exact_values[i] for i in range(steps)]
    return time_grid, approximations, errors

def newstep(tol, err, errold, hold, k):
  return ((tol/err)**(2/(3*k))*(tol/errold)**(-1/(3*k)))*hold