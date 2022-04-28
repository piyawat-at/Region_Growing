import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Region glowing function
# Perform region glowing in image input by specified seed_point(s).
# The Predicate is the difference between seed_point and exploring point value is less than threshold_value.
# Mark the point that passed predicate to the region.
# Mark the point that not passed predicate and didn't explore it again.
# Explore in 8-connected neighborhood of border point of in the region.



# Region_glowing function 
def region_glowing(seed_point,threshold_value):
    seed_point_mean = (np.sum(img_data[seed_point[1],seed_point[0],:3]))/3 # Calucate mean of seed point
    print('seed_point', seed_point[0], seed_point[1],': ',img_data[seed_point[1],seed_point[0]])

    tmp = np.zeros((weight,height))                         # Create array for marked that point expolred or not
                                                            # 0 = not explore, -1 = explored

    img_data[seed_point[1],seed_point[0]] = [255,0,0,255]   # Set seed point to region
    tmp[seed_point[0],seed_point[1]] = -1                   # Mark seed point that is explored 
    broaders = [(seed_point[0],seed_point[1])]              # Create arrray for keep broaders pixel of region

    # loop untill can't find the point to explore
    loop_count = 0 
    while(len(broaders)>=1):
        broaders_tmp = broaders
        for broader in broaders_tmp:
            x,y  = broader[0],broader[1]
            tmp[x,y] = 1
            broaders.remove((x,y)) # remove explored from broaders
            # 8-connected neighborhood of border point
            neighbor = [(x-1,y-1),(x,y-1),(x+1,y-1),
                        (x-1,y),           (x+1,y),
                        (x-1,y+1),(x,y+1),(x+1,y+1)
            ]
            # loop check predicate of neighborhood
            for pixel in neighbor:
                if (tmp[pixel[0],pixel[1]] == 0):
                    dist = (np.sum(img_data[pixel[1],pixel[0],:3]))/3
                    tmp[pixel[0],pixel[1]] = -1
                    if abs(dist- seed_point_mean) <= threshold_value:
                        img_data[pixel[1],pixel[0]] = [255,0,0,255]
                        broaders.append((pixel[0],pixel[1]))
                        
        loop_count += 1

# Load image to numpy array
img_data = np.array(Image.open('brain.png'))
plt.subplot(1, 2, 1)
plt.imshow(img_data)
plt.title('Test Image')

# Set seed point
seed_point_1 = (188, 226)
seed_point_2 = (301, 207)

# Get height and weight
height, weight, shape = img_data.shape

region_glowing(seed_point_1,100)
region_glowing(seed_point_2,50)

# Showing output image 
plt.subplot(1, 2, 2)
plt.imshow(img_data)
plt.title('Output Image')
plt.show()


