import cv2
import numpy as np
from matplotlib import pyplot as plt
 
img = cv2.imread('1.jpg', 0)

edges = cv2.Canny(img,50,50)
'''plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()
'''

im = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5, 5)), iterations=2)
im2, contours, hierarchy = cv2.findContours(im,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#im = cv2.drawContours(im, contours, -1, (120,120,120), 6)
i=0
for cnt in contours:
	if cv2.contourArea(cnt)> 8000:
		i+=1
		string = "roi"+str(i)+".jpg"
		x,y,w,h = cv2.boundingRect(cnt)
		roi = im[y-int(w/10):y+h+int(w/10), x-int(w/10):x+w+int(w/10)]
		
		cv2.imwrite(string, roi)
		
		im = cv2.rectangle(im,(x-int(w/10),y-int(w/10)),(x+w+int(w/10),y+h+int(w/10)),(120, 120, 120),2)
		cv2.imshow("Name", im)
	
cv2.waitKey(0)
'''im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))
cv2.drawContours(edges, contours, 1, (0,255,0), 6)
cv2.namedWindow("output", cv2.WINDOW_NORMAL)
cv2.imshow("output", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
