import cv2

#Read the original image
img = cv2.imread("for_watson.png")
#cv2.imshow("original",img)

#add the image to itslef (to make thw whole message visible at once)
img2 = cv2.addWeighted(img,10,img,0,0)
#cv2.imshow("Weighted",img2)

#convert the image to hsv for a uniform background and the gray scale image for accurate thresholding
imghsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
#cv2.imshow("hsv",imghsv)
imggray = cv2.cvtColor(imghsv, cv2.COLOR_BGR2GRAY)
#cv2.imshow("gray",imggray)

#thresholding the image to decode the message
_,finalimg = cv2.threshold(imggray,230,255,cv2.THRESH_BINARY)
cv2.imshow("final",finalimg)
cv2.waitKey(0)
cv2.destroyAllWindows()