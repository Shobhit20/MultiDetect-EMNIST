import cv2
import numpy as np
from matplotlib import pyplot as plt
import test as test
import copy
import sys
img = cv2.imread(str(sys.argv[1]), 0)


edges = cv2.Canny(img,50,50)
#cv2.imshow("Nae", edges)


'''plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()
'''

im = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5, 5)), iterations=2)

new_img = copy.deepcopy(im)
contours, hierarchy = cv2.findContours(new_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#im = cv2.drawContours(im, contours, -1, (120,120,120), 6)
cv2.imshow("Nae", im)

cv2.waitKey()
arr = []
i=0
for cnt in contours:
	#if cv2.contourArea(cnt)> 10000:	
		x,y,w,h = cv2.boundingRect(cnt)
		arr.append((x,y,w,h))
model = test.model()
arr = sorted(arr, key=lambda x: x[0])


arrnew = []
for i in range(len(arr)):
	k=0
	arrcpy = copy.deepcopy(arr)
	del arrcpy[i]
	print arrcpy
	for j in range(len(arrcpy)):
		
		if arr[i][0] > arrcpy[j][0] and arr[i][1] > arrcpy[j][1] and (arr[i][2] + arr[i][0]) < (arrcpy[j][2]+ arrcpy[j][0]) and (arr[i][3] + arr[i][0]) < (arrcpy[j][3] + arr[j][0]):
			k+=1
			print "sfhbdshfbs"
	if k == 0:
		arrnew.append(arr[i])
print arrnew



for i in range(len(arrnew)):
	
					x, y, w, h = arrnew[i]	
					k+=1
					string = "roi"+str(k)+".jpg"

					roi = im[y-int(w/10):y+h+int(w/10), x-int(w/10):x+w+int(w/10)]
					cv2.imshow("Nae", im)
					cv2.waitKey()
					
					print i
					cv2.imwrite(string, roi)
					im = cv2.rectangle(im,(x-int(w/10),y-int(w/10)),(x+w+int(w/10),y+h+int(w/10)),(120, 120, 120),2)
					#cv2.imshow("Name", im)
					cv2.imwrite("roi.jpg", roi)
					test.testing(model)
cv2.waitKey(0)
	


