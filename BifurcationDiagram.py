# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 21:54:08 2022

@author: Justin Leighton
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
import gc
import os


#%% Plot large bifurcation diagram

# Function parameters
n = 100000
r = np.linspace(2.5, 4.0, n)
iterations = 1000
last = 100
x = 1e-5 * np.ones(n)

# Define logistic map function
def logistic(r, x):
    return r * x * (1 - x)

# Initialize custom scaling class
class CustomScale(mscale.ScaleBase):
    name = 'custom'

    def __init__(self, axis, **kwargs):
        mscale.ScaleBase.__init__(self)
        self.thresh = None

    def get_transform(self):
        return self.InvertedCustomTransform(self.thresh)

    def set_default_locators_and_formatters(self, axis):
        pass

    class CustomTransform(mtransforms.Transform):
        def transform_non_affine(self, a):
            return np.log(1+a)
    
    class InvertedCustomTransform(mtransforms.Transform):
        def transform_non_affine(self, a):
            return np.exp(a)-1

# Initialize plot surface
mscale.register_scale(CustomScale)
fig, ax = plt.subplots(1, 1, figsize=(177,126))
ax.set_facecolor((49/255, 52/255, 58/255))

# Plot points
for i in range(iterations):
    x = logistic(r, x)
    if i >= (iterations - last):
        ax.plot(r, x, ',k', alpha=.25, color="white", marker='.', markersize=2)

# Customize plot for clean image
ax.set_xlim(2.5, 4)
ax.set_xscale('custom')
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
[spine.set_visible(False) for spine in ax.spines.values()]
#r.set_clip_on(False)

# Save image
plt.savefig('bifurcation_diagram_e2.png', bbox_inches='tight')

# Cleanup
del i, iterations, last, n, r, x
gc.collect()

