import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

list = []
imgg = []
img_path = r'C:\Users\mesqu\Downloads\barcelonanov.jpeg'
img = cv2.imread(img_path, 0) 

ret, thresh = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)

h, w = thresh.shape
print('width:  ', w)
print('height: ', h)
state_matrix = np.ascontiguousarray(np.arange(h*w).reshape(w,h), dtype=int)
new_img = np.argwhere(thresh==0)

for i in new_img:
    list.append(state_matrix[i[0]][i[1]])

with open('states.txt','w') as output:
       output.write(str(list))

plt.imshow(thresh)
plt.title('Segmentação Binária do Ambiente')
plt.xticks(np.arange(0, w, 1.0))
plt.yticks(np.arange(0, h, 1.0))
plt.show()

print('Obstáculos identificados nos estados:', list)