# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 16:16:31 2022

@author: Justin Leighton
"""

import matplotlib.pyplot as plt
import numpy as np
R = np.square(np.linspace(1.5, 2, 100000))
#R = np.linspace(2.5, 4, 100000)

X = [] # X aaxis - r- control parameters
Y = [] # Y axis - x values of logistic map

# Generate x for each value of r
for r in R:
    X.append(r)
    
    x = np.random.random() # initialize x for each r value
    for n in range(100): # ignore the transient effect
        x=r*x*(1-x)
        
    for n in range(100):
        x=r*x*(1-x)

    Y.append(x)
    
plt.plot(X, Y, color='#99AAB5', ls='', marker=',')
plt.xlabel("X Label")
plt.ylabel("Y Label")
ax = plt.gca()
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)
ax.set_facecolor((35 / 256, 39 / 256, 42 / 256))
plt.grid(True)
plt.savefig('bifurcation_diagram.png', dpi=500)