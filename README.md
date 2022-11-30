import matplotlib
import matplotlib.pyplot as plt
import cv2
 
img = cv2.imread('D:/Jobs/SIFT/img03.jpg', cv2.IMREAD_UNCHANGED)
 
print('Original Dimensions : ',img.shape)
 
width = 600
height = 480
dim = (width, height)
 
# resize image
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
 
print('Resized Dimensions : ',resized.shape)
 
cv2.imwrite("Resized_image.jpg", resized)

gray= cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT_create()
kp, des = sift.detectAndCompute(gray,None)
extract_img=cv2.drawKeypoints(gray,kp,resized,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('Extract_kp.jpg',extract_img)

print(len(kp))


Titles =["Original", "Rescale 480-600"]
images =[img, resized]
count = 2

for i in range(count):
    plt.subplot(2, 2, i + 1)
    plt.title(Titles[i])
    plt.imshow(images[i])
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
