import numpy as np
from scipy import signal


# Rechteck Signal
def square(t,f,Vp=1,D=0.5):
    return ((signal.square(2 * np.pi * f * t , duty=D) + 1) * Vp)/2


# Sprung
def step(t,Vp=1,Toff=0):
    return np.heaviside(t-Toff, 0) * Vp
