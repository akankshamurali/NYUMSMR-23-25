import cv2

def nothing(x): #placeholder for a callback function executes everytime the trackbar value changes
    pass

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

# Create track bar to filter
cv2.namedWindow('mask')
cv2.createTrackbar('Filter','mask',0,255,nothing)

while(1):
    # Get track bar value
    filter = cv2.getTrackbarPos('Filter','mask')
    #thresholding the image to decode the message
    mask = cv2.inRange(imggray, filter-150, filter+100)
    
    #Display mask output
    cv2.imshow("mask",mask)
    
    #Exit Sequence on pressing 'q'
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()