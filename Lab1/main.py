import differential_equation as d_eq
import runge_kutta as rk
import plot_helpers as ph
# Valid initial conditions for the differential equation
told=0
uold=1
ph.errVShRK4(d_eq.f, uold, told, 2, d_eq.f_real)
