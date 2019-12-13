#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 11:13:38 2019

@author: gonzalosaravia
"""

from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import glob
import shutil
import re
import string

PREDICTOR_PATH = "/Users/gonzalosaravia/Desktop/Udemy/Neoland/Proyecto Final/shape_predictor_68_face_landmarks.dat"

# Inicializa el detector de cara de dlib (HOG-based) y luego crea
# El factor predictivo de la marca facial
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

#Cargo el dataset de imagenes con import glob 
#imagelabel = np.load('/Users/gonzalosaravia/Desktop/Udemy/Neoland/Proyecto Final/Dataset/face_images.npz')['face_images']
imagelabel = glob.glob("/Users/gonzalosaravia/Desktop/Udemy/Neoland/Proyecto Final/LAGdataset_200/*.png")


from skimage.color import rgb2gray
import cv2
import matplotlib.pyplot as plt
%matplotlib inline
from scipy import ndimage


#Imprimo por consola la imagen
image = plt.imread(imagelabel[500])
image = imutils.resize(image, width=500)
image.shape
plt.imshow(image)

# La paso a escala de grises
gray = rgb2gray(image)
plt.imshow(gray, cmap='gray')


gray.shape

# =============================================================================
# (500, 500)
# =============================================================================

# le pido que me de la imagen en la posicion 0
image = cv2.imread(imagelabel[500])
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Detectar caras en la imagen en escala de grises
rects = detector(gray, 1)

# ciclo sobre las detecciones de la cara
for (i, rect) in enumerate(rects):
# Determinar las marcas faciales para la región de la cara, luego
# Convertir el punto de referencia (x, y) - coordina a un array NumPy
  shape = predictor(gray, rect)
  shape = face_utils.shape_to_np(shape)

 # ciclo sobre las partes de la cara individualmente
  for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
  # Clonar la imagen original para que podamos dibujar en ella, entonces
  # Mostrar el nombre de la parte de la cara en la imagen
    clone = image.copy()
#    cv2.putText(clone, name, (10, 90), cv2.FONT_HERSHEY_SIMPLEX,0.9, (0, 0, 255), 2)

# Sobre el subconjunto de marcas faciales, dibuja la
# Parte de la cara específica
    for (x, y) in shape[i:j]:
      cv2.circle(clone, (x, y), 4, (53, 104, 45), -1)
    
import matplotlib.pyplot as plt

plt.imshow(clone)


# Detecto los objetos por pixeles con otra escala que me marca los bordes
gray_r = gray.reshape(gray.shape[0]*gray.shape[1])
for i in range(gray_r.shape[0]):
    if gray_r[i] > gray_r.mean():
        gray_r[i] = 1
    else:
        gray_r[i] = 0
gray = gray_r.reshape(gray.shape[0],gray.shape[1])
plt.imshow(gray, cmap='gray')
plt.show()

# Marco mas los bordes del objeto
gray = rgb2gray(image)
gray_r = gray.reshape(gray.shape[0]*gray.shape[1])
for i in range(gray_r.shape[0]):
    if gray_r[i] > gray_r.mean():
        gray_r[i] = 3
    elif gray_r[i] > 0.5:
        gray_r[i] = 2
    elif gray_r[i] > 0.25:
        gray_r[i] = 1
    else:
        gray_r[i] = 0
gray = gray_r.reshape(gray.shape[0],gray.shape[1])
plt.imshow(gray, cmap='gray')
plt.show()


gray = rgb2gray(image)

# defino los filtros sobel (El filtro Sobel detecta los bordes horizontales y verticales separadamente)
sobel_horizontal = np.array([np.array([1, 2, 1]), np.array([0, 0, 0]), np.array([-1, -2, -1])])
print(sobel_horizontal, 'is a kernel for detecting horizontal edges')
 
sobel_vertical = np.array([np.array([-1, 0, 1]), np.array([-2, 0, 2]), np.array([-1, 0, 1])])
print(sobel_vertical, 'is a kernel for detecting vertical edges')

# =============================================================================
# [[ 1  2  1]
#  [ 0  0  0]
#  [-1 -2 -1]] is a kernel for detecting horizontal edges
# [[-1  0  1]
#  [-2  0  2]
#  [-1  0  1]] is a kernel for detecting vertical edges
# =============================================================================


out_h = ndimage.convolve(gray, sobel_horizontal, mode='reflect')
out_v = ndimage.convolve(gray, sobel_vertical, mode='reflect')

#lineas horizontales
plt.imshow(out_h, cmap='gray')

#lineas verticales
plt.imshow(out_v, cmap='gray')

kernel_laplace = np.array([np.array([1, 1, 1]), np.array([1, -8, 1]), np.array([1, 1, 1])])
print(kernel_laplace, 'is a laplacian kernel')

#lINEAS COMPLETAS
out_l = ndimage.convolve(gray, kernel_laplace, mode='reflect')
plt.imshow(out_l, cmap='gray')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 3)

t, dst = cv2.threshold(out_h, 0.4, 255, cv2.THRESH_BINARY_INV)
plt.imshow(dst)

def image_show(image, nrows=1, ncols=1, cmap='gray'):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 14))
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    return fig, ax

def circle_points(resolution, center, radius):
    """
    Generate points which define a circle on an image.Centre refers to the centre of the circle
    """   
    radians = np.linspace(0, 2*np.pi, resolution)
    c = center[1] + radius*np.cos(radians)#polar co-ordinates
    r = center[0] + radius*np.sin(radians)
    
    return np.array([c, r]).T

points = circle_points(190, [210, 250], 212)[:-1]

fig, ax = image_show(out_h)
ax.plot(points[:, 0], points[:, 1], '--r', lw=3)


import matplotlib.pyplot as plt
import skimage.data as data
import skimage.segmentation as seg
import skimage.filters as filters
import skimage.draw as draw
import skimage.color as color
import imutils

snake = seg.active_contour(out_h, points)
fig, ax = image_show(out_h)
ax.plot(points[:, 0], points[:, 1], '--r', lw=3)
ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3);

snake = seg.active_contour(out_h, points)
fig, ax = image_show(out_h)
ax.plot(points[:, 0], points[:, 1], '--r', lw=3)
ax.plot(shape[:, 0], shape[:, 1], '-b', lw=3);
ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3);



# Unir los puntos de la mandibula con los de la parte superior
import matplotlib.pyplot as plt
import pandas as pd
df=pd.DataFrame(shape).iloc[0:17]
maximo=max(500-df[1])

df2=pd.DataFrame(snake)
df2=df2[(500-df2[1])>maximo]
plt.plot(500-df[0],500-df[1], marker= 'o')
plt.plot(500-df2[0],500-df2[1], marker= 'o')


contorno= pd.concat([df,df2])
plt.plot(500-contorno[0],500-contorno[1],'bo')

fig, ax = image_show(out_l)
ax.plot(points[:, 0], points[:, 1], '--r', lw=3)
ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3);





























