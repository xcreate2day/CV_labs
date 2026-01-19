# %%
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tools import *


image = "C:/Users/xcrea/Documents/YandexPython/pythonProject/img/Karelia2_text.jpg"


I = cv2.cvtColor(cv2.imread(image), cv2.COLOR_RGB2GRAY)

# showGray(I)


_, A = cv2.threshold(I, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# showGray(A)

#B = np.ones((5, 5), dtype=np.ubyte)
B = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))


# Дилатация
R = cv2.dilate(A, B)
# showGray(R)


# Эрозия
R = cv2.erode(A, B)
# showGray(R)

# Замыкание
R = cv2.morphologyEx(A, cv2.MORPH_CLOSE, B)
# showGray(R)

# Размыкание
R = cv2.morphologyEx(A, cv2.MORPH_OPEN, B)
# showGray(R)

# Условная дилатация

B = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
C = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
conditional_dilation(A, B, C)



_, A = cv2.threshold(cv2.cvtColor(cv2.imread("C:/Users/xcrea/Documents/YandexPython/pythonProject/img/Karelia2_text.jpg"),
                                  cv2.COLOR_RGB2GRAY), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# showGray(A)
B = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
C = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
conditional_dilation(A, B, C)

morphological_skeleton_show()
