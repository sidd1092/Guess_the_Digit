from tokenize import Number
from numpy import testing
from numpy.lib.type_check import imag
import pygame, sys
from pygame.locals import *
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import cv2
from tensorflow.python.keras.backend import constant

WINDOWSIZEX = 640
WINDOWSIZEY = 480

BOUNDRYINC = 5
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

IMAGESAVE = False

model = load_model('bestmodel.h5')

LABELS= {0:"Zero", 1:"One",
        2:"Two", 3:"Three", 
        4:"Four", 5:"Five", 
        6:"Six", 7:"Seven", 
        8:"Eight", 9:"Nine"}

# Initialize Pygame
pygame.init()

# font = pygame.font.font("freeSansBold.ttf", 18)
DISPLAYSURF = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))

pygame.display.set_caption('Handwritten Digit Recognition')

isDrawing = False

number_xcord = []
number_ycord = []

image_count = 1

while True:

    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == MOUSEMOTION and isDrawing:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAYSURF, WHITE, (xcord, ycord), 4, 0)

            number_xcord.append(xcord)
            number_ycord.append(ycord)

        if event.type == MOUSEBUTTONDOWN:
            isDrawing = True

        if event.type == MOUSEBUTTONUP:
            isDrawing = False
            number_xcord = sorted(number_xcord)
            number_ycord = sorted(number_ycord)

            rect_min_x , rect_max_x = max(number_xcord[0] - BOUNDRYINC, 0), min(WINDOWSIZEX, number_xcord[-1] + BOUNDRYINC)
            rect_min_y , rect_max_y = max(number_ycord[0] - BOUNDRYINC, 0), min(number_ycord[-1] + BOUNDRYINC, WINDOWSIZEX)

            number_xcord = []
            number_ycord = []

            ing_arr = np.array(pygame.PixelArray(DISPLAYSURF)[rect_min_x:rect_max_x, rect_min_y:rect_max_y]).T.astype(np.float32)

            if IMAGESAVE:
                cv2.imwrite("image.png")
                image_count += 1
            
            if PREDICT:

                image = cv2.resize(img_arr, (28, 28))
                image = np.pad(image,(10,10), 'constant', constant_values=0)
                image = cv2.resize(image, (28, 28))/255

                label= str(np.argmax(model.predict(image.reshape(1, 28, 28, 1))))
                
