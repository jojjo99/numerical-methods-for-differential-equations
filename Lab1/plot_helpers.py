import matplotlib.pyplot as plt
import runge_kutta as rk
import numpy as np

# Function plots error vs stepsize for RK4
def errVShRK4(f, initial_value, start_time, end_time, f_real):
    steps_list = [2**k for k in range(3, 12)]
    y=[]
    x=[]
    for steps in steps_list:
        _, _, error_grid = rk.RKn(f, initial_value, start_time, end_time, steps, f_real, rk.RK4step)
        stepsize = (end_time-start_time)/steps
        plt.loglog(stepsize, np.linalg.norm(error_grid[-1]), "o", label=f"N={steps}")
        y.append(np.linalg.norm(error_grid[-1]))
        x.append(stepsize)
    plt.xlabel("Stepsize")
    plt.ylabel("Error")
    plt.legend()
    plt.show()
    print(x, y)