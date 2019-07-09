from data.voc_scut import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

dataset = VOCDetection("/Users/seungyoun/Desktop/mAyI/RFBNet/data/SCUT_HEAD")

ex = dataset.pull_item( random.randint(1,3403))
print("img shape : ",ex[0].shape)


fig,ax = plt.subplots(1)

# Display the image
ax.imshow(ex[0])

# Create a Rectangle patch
for i in ex[1]:
    startX = i[0]*512
    startY = i[1]*512
    dx = i[2]*512 - i[0]*512
    dy = i[3]*512 - i[1]*512
    print(startX,startY,dx,dy)
    rect = patches.Rectangle((startX,startY),dx,dy,linewidth=0.5,edgecolor='r',facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)
plt.show()
