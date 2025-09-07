#!/usr/bin/env python3

"""Atividade 4
Grupo:
Augusto Carvalho
Igor Gonçalves
João Eloy
"""

"""Máscaras de convolução
Detectores de Bordas
-------------------------
borda horizontal (Sobel)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from skimage import io
from skimage import filters
from skimage.data import camera
from skimage.util import compare_images

#image = camera()
image = io.imread('./Lenna.png', as_gray=True)

edge_sobel = filters.sobel(image)
edge_roberts = filters.roberts(image)

fig, axes = plt.subplots(ncols=3, sharex=True, sharey=True, figsize=(12, 5))
axes[0].imshow(image, cmap=plt.cm.gray)
axes[0].set_title('Original Greyscale image')

axes[1].imshow(edge_sobel, cmap=plt.cm.gray)
axes[1].set_title('Sobel Edge Detection')

axes[2].imshow(edge_roberts, cmap=plt.cm.gray)
axes[2].set_title('Roberts Edge Detection')

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.show()


sys.exit(0)
