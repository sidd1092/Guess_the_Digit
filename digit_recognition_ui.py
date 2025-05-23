import pygame, sys
from pygame.locals import *
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import load_model
import cv2

WINDOWSIZEX = 640
WINDOWSIZEY = 480

BOUNDRYINC = 5
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

IMAGESAVE = False

model = load_model("C:/Users/ssidd/OneDrive/Desktop/Projects/Guess_the_Digit/model.h5")

LABELS= {0:"Zero",1:"One",2:"Two",3:"Three",4:"Four",5:"Five",6:"Six",7:"Seven",8:"Eight",9:"Nine"}

# Initialize Pygame
pygame.init()

FONT = pygame.font.Font(None, 18)
DISPLAYSURF = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))

pygame.display.set_caption('Handwritten Digit Recognition')

iswriting = False

number_xcord = []
number_ycord = []

image_count = 1
PREDICT = True

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAYSURF, WHITE, (xcord, ycord), 4, 0)

            number_xcord.append(xcord)
            number_ycord.append(ycord)

        if event.type == MOUSEBUTTONDOWN:
            iswriting = True

        if event.type == MOUSEBUTTONUP:
            iswriting = False
            number_xcord = sorted(number_xcord)
            number_ycord = sorted(number_ycord)
                
            # Before accessing elements in number_xcord, check if it is not empty
            if number_xcord:
                rect_min_x, rect_max_x = max(number_xcord[0] - BOUNDRYINC, 0), min(WINDOWSIZEX, number_xcord[-1] + BOUNDRYINC)
            else:
                rect_min_x, rect_max_x = 0, WINDOWSIZEX

            if number_ycord:
                rect_min_y, rect_max_y = max(number_ycord[0] - BOUNDRYINC, 0), min(WINDOWSIZEY, number_ycord[-1] + BOUNDRYINC)
            else:
                rect_min_y, rect_max_y = 0, WINDOWSIZEY

            number_xcord = []
            number_ycord = []

            img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x, rect_min_y:rect_max_y].T.astype(np.float32)

            if IMAGESAVE:
                cv2.imwrite("image.png")
                image_count += 1
            
            if PREDICT:
                image = cv2.resize(img_arr, (28, 28))
                image = np.pad(image,(10,10), 'constant', constant_values=0)
                image = cv2.resize(image, (28, 28))/255

                image = np.expand_dims(image, axis=-1)  # Expand dims to match model input

                label= str(LABELS[np.argmax(model.predict(image.reshape(1, 28, 28, 1)))])
                textSurface = FONT.render(label, True, RED, WHITE)
                textRecObj = textSurface.get_rect()
                textRecObj.left, textRecObj.bottom = rect_min_x, rect_max_y

                DISPLAYSURF.blit(textSurface, textRecObj)

        if event.type == KEYDOWN:
            if event.unicode == "n":
                DISPLAYSURF.fill(BLACK)

    pygame.display.update()