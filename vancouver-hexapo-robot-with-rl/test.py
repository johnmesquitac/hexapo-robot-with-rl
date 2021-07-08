import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

list = []
img_path = r'C:\Users\mesqu\Downloads\vancouver3.png'
img = cv2.imread(img_path, 0) 
img_reverted= cv2.bitwise_not(img)
h, w = img_reverted.shape
print('width:  ', w)
print('height: ', h)
state_matrix = np.ascontiguousarray(np.arange(h*w).reshape(w,h), dtype=int)
new_img = np.argwhere(img_reverted>150)
for i in new_img:
    list.append(state_matrix[i[0]][i[1]])

with open('states.txt','w') as output:
    output.write(str(list))

plt.imshow(img_reverted)
plt.show()

