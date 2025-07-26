import pygame, sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2

# Configs
WINDOWSIZEX = 640
WINDOWSIZEY = 480
BOUNDARYINC = 5
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

IMAGESAVE = False
PREDICT = True
MODEL = load_model("bestmodel.h5")

LABELS = {
    0: "ZERO", 1: "ONE", 2: "TWO", 3: "THREE", 4: "FOUR",
    5: "FIVE", 6: "SIX", 7: "SEVEN", 8: "EIGHT", 9: "NINE"
}

# Initialize pygame
pygame.init()
FONT = pygame.font.SysFont("Arial", 18)
DISPLAYSURF = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))
pygame.display.set_caption("Digit Board")

iswriting = False
number_xcord = []
number_ycord = []
image_cnt = 1

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == MOUSEMOTION and iswriting:
            x, y = event.pos
            pygame.draw.circle(DISPLAYSURF, WHITE, (x, y), 4, 0)
            number_xcord.append(x)
            number_ycord.append(y)

        if event.type == MOUSEBUTTONDOWN:
            iswriting = True

        if event.type == MOUSEBUTTONUP:
            iswriting = False
            if number_xcord and number_ycord:
                number_xcord = sorted(number_xcord)
                number_ycord = sorted(number_ycord)

                min_x = max(number_xcord[0] - BOUNDARYINC, 0)
                max_x = min(WINDOWSIZEX, number_xcord[-1] + BOUNDARYINC)
                min_y = max(number_ycord[0] - BOUNDARYINC, 0)
                max_y = min(WINDOWSIZEY, number_ycord[-1] + BOUNDARYINC)

                number_xcord = []
                number_ycord = []

                # Get RGB array from surface
                image_surface = pygame.surfarray.array3d(DISPLAYSURF)
                image_crop = image_surface[min_x:max_x, min_y:max_y]
                image_crop = np.transpose(image_crop, (1, 0, 2))  # transpose for height x width x channels

                # Convert to grayscale uint8
                gray = cv2.cvtColor(image_crop.astype(np.uint8), cv2.COLOR_RGB2GRAY)

                if IMAGESAVE:
                    cv2.imwrite(f"image_{image_cnt}.png", gray)
                    image_cnt += 1

                # Resize and predict
                image = cv2.resize(gray, (28, 28))
                image = image / 255.0
                image = image.reshape(1, 28, 28, 1)

                if PREDICT:
                    prediction = MODEL.predict(image)
                    label = LABELS[np.argmax(prediction)]

                    text_surface = FONT.render(label, True, RED, WHITE)
                    text_rect = text_surface.get_rect()
                    text_rect.left, text_rect.bottom = min_x, max_y

                    DISPLAYSURF.blit(text_surface, text_rect)

        if event.type == KEYDOWN:
            if event.unicode == "n":
                DISPLAYSURF.fill(BLACK)

    pygame.display.update()
