#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 18:27:13 2019

@author: gonzalosaravia
"""

#librerias y funciones
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import glob
import shutil
import re
import string


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

def coord(x):
    punto=0
    dist=np.inf     
    for i in range(snake_df.shape[0]):
        dist_i=abs(snake_df.iloc[i,0]-x)
        if (dist_i<=dist):
            punto=i
            dist=dist_i
    coord=snake_df.iloc[punto]
    return coord

def distancia_puntos(point1,point2):
    distancia=np.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)
    return distancia



PREDICTOR_PATH = "/Users/gonzalosaravia/Desktop/Udemy/Neoland/Proyecto Final/shape_predictor_68_face_landmarks.dat"


# Inicializa el detector de cara de dlib (HOG-based) y luego crea
# El factor predictivo de la marca facial
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

#Cargo el dataset de imagenes con import glob 
imagelabel = glob.glob("/Users/gonzalosaravia/Desktop/Udemy/Neoland/Proyecto Final/LAGdataset_200/*.png")

from skimage.color import rgb2gray
import cv2
import matplotlib.pyplot as plt
#%matplotlib inline
from scipy import ndimage

import matplotlib.pyplot as plt
import skimage.data as data
import skimage.segmentation as seg
import skimage.filters as filters
import skimage.draw as draw
import skimage.color as color
import imutils
import pandas as pd





images=[]
for i_image in range(7):
    print(imagelabel[i_image])
    image = cv2.imread(imagelabel[i_image])
    image = imutils.resize(image, width=500)
    gray = rgb2gray(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detectar caras en la imagen en escala de grises
    rects = detector(gray, 1) #rects es el rectangulo del detector de caras

    # ciclo sobre las detecciones de la cara
    for (i, rect) in enumerate(rects):
    # Determinar las marcas faciales para la región de la cara, luego
    # Convertir el punto de referencia (x, y) - coordina a un array NumPy
      shape = predictor(gray, rect)
      shape = face_utils.shape_to_np(shape) #array de todos los puntos de la cara

    # ciclo sobre las partes de la cara individualmente
    for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
    # Clonar la imagen original para que podamos dibujar en ella, entonces
    # Mostrar el nombre de la parte de la cara en la imagen
      clone = image.copy()

    # Sobre el subconjunto de marcas faciales, dibuja la
    # Parte de la cara específica
    for (x, y) in shape[i:j]:
      cv2.circle(clone, (x, y), 4, (53, 104, 45), -1)
      
    # Blanco y negro para marcar los bordes  
    gray_r = gray.reshape(gray.shape[0]*gray.shape[1])
    for i in range(gray_r.shape[0]):
        if gray_r[i] > gray_r.mean():
            gray_r[i] = 1
        else:
            gray_r[i] = 0
    gray = gray_r.reshape(gray.shape[0],gray.shape[1])
    
    #Blanco y negro para remarcar los bordes del anterior filtro
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
    
    gray = rgb2gray(image)
    
    # defino los filtros sobel (El filtro Sobel detecta los bordes horizontales y verticales separadamente)
    sobel_horizontal = np.array([np.array([1, 2, 1]), np.array([0, 0, 0]), np.array([-1, -2, -1])])
    print(sobel_horizontal, 'is a kernel for detecting horizontal edges')
     
    sobel_vertical = np.array([np.array([-1, 0, 1]), np.array([-2, 0, 2]), np.array([-1, 0, 1])])
    print(sobel_vertical, 'is a kernel for detecting vertical edges')

    out_h = ndimage.convolve(gray, sobel_horizontal, mode='reflect')
    out_v = ndimage.convolve(gray, sobel_vertical, mode='reflect')
    
    kernel_laplace = np.array([np.array([1, 1, 1]), np.array([1, -8, 1]), np.array([1, 1, 1])])
    print(kernel_laplace, 'is a laplacian kernel')
    
    #lINEAS COMPLETAS
    out_l = ndimage.convolve(gray, kernel_laplace, mode='reflect')
    
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    t, dst = cv2.threshold(out_h, 0.4, 255, cv2.THRESH_BINARY_INV)
    
    
    points = circle_points(190, [210, 250], 212)[:-1]
    
#    fig, ax = image_show(out_h)
#    ax.plot(points[:, 0], points[:, 1], '--r', lw=3)

    snake = seg.active_contour(out_l, points,alpha=0.06,beta=0.3)
    
    snake_df=pd.DataFrame(snake)
    snake_df=snake_df[snake_df[1]<=shape[0,1]]
    
    coords=[]
    
    # los pongo todos en una misma lista y los saca como espejo
    for i in [1,5,6,8,10,11]:
        x=shape[i,0]
        coords.append(coord(x))
    
    coords= np.array(coords)    
    puntos= np.concatenate([shape[0:17],coords])
    
    distancias = {}
    for i in range(len(puntos)):
      for j in range(len(puntos)):
        if i!=j:
          distancias[str(i)+'_'+str(j)] = distancia_puntos(puntos[i], puntos[j])

    list(distancias.values())
    distancias1 = np.array(list(distancias.values()))

    fig, ax = image_show(image) 
    ax.plot(shape[:17, 0], shape[:17, 1], 'ob');
    ax.plot(coords[:,0],coords[:,1],'bo')
    plt.imshow(image)
    #falta que lo saque para cada una, y no se vayan pisando 
    
   
# Función para elegir cuales caras formaran parte de mi dataset a entrenar    
    
images1= [images]
for i in range(2):
    print(imagelabel[i])
    image1 = plt.imread(imagelabel[i])
    image1 = imutils.resize(image1, width=500)
    images.append(image1)



def crear_etiquetas(lista_arrays):
    etiquetas=[]
    ids=[]
    imagenes=[]
    for img_id,i in enumerate(lista_arrays):
#        hacer el tratamiento de filtros etc para sacar los puntos y plotearlos a continuación
#        con el objetivo de decidir si nos quedamos con esa imagen o no
#        hasta que se haga eso imprimo la imagen directamente
        print(imagelabel[img_id])
        plt.imshow(i)
        plt.show()
        etiqueta=input('Qué tipo de cara es? \n 1-Ovalada \n 2-Rectangular \n 3-Diamante \n 4-Redonda \n 5-Largas \n 6-Tringaular \n 0-Eliminar')

        if (etiqueta!='0'):
            etiquetas.append(etiqueta)
            ids.append(img_id)
            imagenes.append(i)
    dicc={'ids':ids, 'imagenes':imagenes, 'etiquetas': etiquetas}
    return dicc

DF=crear_etiquetas(image)


# Load dataframe from pickled pandas object
df= pd.read_pickle(file_name)

































