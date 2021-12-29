
import cv2
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename

filename = askopenfilename(filetypes=[("allfiles", "*")])
img=cv2.imread(filename)


# displaying the original image
plt.title("Orignal Image:")
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.show()


# detecting edges through LOG to apply hough transform
def edgeDetection():   
    imgBuffer = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sigma=2
    size = int(2*(np.ceil(3*sigma))+1)

    x,y = np.meshgrid(np.arange(-size/2+1, size/2+1),
                       np.arange(-size/2+1, size/2+1))

    normal = 1 / (2.0 * np.pi * sigma**2)

    kernel = ((x**2 + y**2 - (2.0*sigma**2)) / sigma**4) * np.exp(-(x**2+y**2) / (2.0*sigma**2)) / normal  # LoG filter

    kernelSize = kernel.shape[0]
    logArray = np.zeros_like(imgBuffer, dtype=float)

    # applying filter
    for i in range(imgBuffer.shape[0]-(kernelSize-1)):
        for j in range(imgBuffer.shape[1]-(kernelSize-1)):
            window = imgBuffer[i:i+kernelSize, j:j+kernelSize] * kernel
            logArray[i, j] = np.sum(window)

    logArray = logArray.astype(np.int64, copy=False)

    zeroCrossing = np.zeros_like(logArray)

    # computing zero crossing
    for i in range(logArray.shape[0]-(kernelSize-1)):
        for j in range(logArray.shape[1]-(kernelSize-1)):
            if logArray[i][j] == 0:
                if (logArray[i][j-1] < 0 and logArray[i][j+1] > 0) or (logArray[i][j-1] < 0 and logArray[i][j+1] < 0) or (logArray[i-1][j] < 0 and logArray[i+1][j] > 0) or (logArray[i-1][j] > 0 and logArray[i+1][j] < 0):
                    zeroCrossing[i][j] = 255
            if logArray[i][j] < 0:
                if (logArray[i][j-1] > 0) or (logArray[i][j+1] > 0) or (logArray[i-1][j] > 0) or (logArray[i+1][j] > 0):
                    zeroCrossing[i][j] = 255

    return zeroCrossing

binary_image=np.array(edgeDetection())

# displaying binary image image
plt.title("Edge Detection:")
plt.imshow(binary_image, cmap='gray', vmin=0, vmax=255)
plt.show()


edge_image = cv2.GaussianBlur(img, (3, 3), 1)
binary_image=cv2.Canny(img, 100,200)

outputImage=np.array(img)
# applying line detection to the image
def detectLines(binary_image):
  h,w=binary_image.shape
  thetaValues=360
  threshold=90
  r_max=round(np.sqrt(np.square(h) + np.square(w)))
  accumulator=np.zeros((r_max, thetaValues))
  for i in range(0,h):
    for j in range(0,w):
      if binary_image[i][j] == 255:
        for k in range(0,thetaValues):
          r=int(round(j*np.cos(np.deg2rad(k)) + i*np.sin(np.deg2rad(k))))
          accumulator[r][k] += 1
  outputLines=[]
  for i in range(0, accumulator.shape[0]):
    for j in range(0, accumulator.shape[1]):
      if accumulator[i][j] >= threshold:
        outputLines.append((i,j))

  postProcess=[]
  postProcess.append(outputLines[0])
  for i in range(0,len(outputLines)):
    if abs(postProcess[-1][0] - outputLines[i][0]) > 5 or abs(postProcess[-1][1] - outputLines[i][1]) > 5:
      postProcess.append(outputLines[i]) 
  return outputLines

  
co_ordinates=detectLines(binary_image)
for i in range(0, len(co_ordinates)):
        a = np.cos(np.deg2rad(co_ordinates[i][1]))
        b = np.sin(np.deg2rad(co_ordinates[i][1]))
        x0 = a*co_ordinates[i][0]
        y0 = b*co_ordinates[i][0]
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(outputImage,(x1,y1),(x2,y2),(255,0,0),1)


# applying circle detection to the image
def detectCircle(binary_image):
  imgHeight, imgWidth=binary_image.shape
  accumulator=defaultdict(int)
  minimumRadius=10
  maximumRadius=200
  numberOfThetas=100
  deltaRadius=1


  differenceTheta=int(360/numberOfThetas)
  totalRadius=np.arange(minimumRadius, maximumRadius, step=deltaRadius)
  thetaValues=np.arange(0,360, step=differenceTheta)

  cosThetas=np.cos(np.deg2rad(thetaValues))
  sinThetas=np.sin(np.deg2rad(thetaValues))

  circleCenters=[]
  for r in totalRadius:
      for t in range(numberOfThetas):
        circleCenters.append((r, int(r * cosThetas[t]), int(r * sinThetas[t])))

  for i in range(0,imgHeight):
    for j in range(0, imgWidth):
      if binary_image[i][j] != 0:
        for r,x,y in circleCenters:
            xCenter = i - x
            yCenter = j - y
            accumulator[(xCenter, yCenter, r)] += 1
  outputCircles=[]
  for center, count in accumulator.items():
    if count/numberOfThetas >= 0.5:
      outputCircles.append(center)
  outputCircles.sort()
  postProcess=[]
  postProcess.append(outputCircles[0])
  for i in range(0,len(outputCircles)):
    if abs(postProcess[-1][0] - outputCircles[i][0]) > 5 or abs(postProcess[-1][1] - outputCircles[i][1]) > 5  or abs(postProcess[-1][2] - outputCircles[i][2]) > 5 :
      postProcess.append(outputCircles[i]) 
  return postProcess

outputCircles=detectCircle(binary_image)

for co_ordinates in outputCircles:
  cv2.circle(outputImage, (co_ordinates[1], co_ordinates[0]), co_ordinates[2], (0,0,255), 2)

# Displaying original and hough image simultaneously through GUI
fig = plt.figure(figsize=(10, 7))
rows = 2
columns = 2
fig.add_subplot(rows, columns, 1)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.axis('off')
plt.title("Orignal:")
fig.add_subplot(rows, columns, 2)
plt.imshow(outputImage, cmap='gray', vmin=0, vmax=255)
plt.axis('off')
plt.title("Hough Image:")
plt.show()