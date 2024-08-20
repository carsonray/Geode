from core import scale_range

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def basic_object(label, shape):
    img = np.zeros(shape + (3,))

    # Adds background noise
    #img = add_noise(img, 0.1)
    #img = add_line_noise(img, 0.1)

    if label == 1:
        # Creates object
        center = rand_point(shape, 0.2)
        radius = rand_size(shape, 0.1, 0.5)

        img = cv.circle(
              img,
              center = center,
              radius = radius,
              color = (0,255,0),
              thickness = 1
            )

    # Adds foreground noise
    #img = add_noise(img, 0.1)
    #img = add_line_noise(img, 0.1)

    return img

# Random functions
def rand_point(shape, padding):
    """Creates a random point within a shape (with padding)"""
    shape = np.array(shape)
    start = shape*padding
    end = shape - shape*padding
    
    point = scale_range(np.random.rand(2), [start, end])

    # Rounds and formats
    return tuple(np.round(point).astype(int))

def rand_size(shape, min, max, ax=None):
    """Creates random sizes within both axes of a shape"""
    shape = np.array(shape)

    lim = shape.min() if ax is None else shape[ax]

    num = scale_range(np.random.rand(1), [lim*min, lim*max])[0]
    return int(num)

# Noise functions
def add_noise(ax, amount):
    pass

def add_line_noise(ax, amount):
    pass

    
img = basic_object(1, shape=(100, 100))

plt.imshow(img)
plt.show()




