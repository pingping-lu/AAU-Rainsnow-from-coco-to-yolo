#!/usr/bin/env python
# coding: utf-8

# In[3]:


from pycocotools import coco
import numpy as np
import skimage.io as io
import matplotlib
import matplotlib.pyplot as plt
import pylab
import random
import cv2
pylab.rcParams['figure.figsize'] = (8.0, 6.0)

rgbAnnFile='../aauRainSnow-rgb.json'
thermalAnnFile = '../aauRainSnow-thermal.json'

rainSnowRgbGt = coco.COCO(rgbAnnFile)
rainSnowThermalGt = coco.COCO(thermalAnnFile)


# ## Display a random image with overlayed annotations in both the RGB and thermal domain 
# 

# In[4]:


for i in range(0, 2197):
    chosenImgId = i
    annIds = rainSnowRgbGt.getAnnIds(imgIds=[chosenImgId])
    anns = rainSnowRgbGt.loadAnns(annIds)

    rgbImg = rainSnowRgbGt.loadImgs([chosenImgId])[0]
    thermalImg = rainSnowThermalGt.loadImgs([chosenImgId])[0]
    thermalAnns = rainSnowThermalGt.loadAnns(annIds)

    print('Found ' + str(len(anns)) + ' annotations at image ID ' + str(chosenImgId) + '. Image file: ' + rgbImg['file_name'])

    for ann in anns:
        print('Annotation #' + str(ann['id']) + ': ' + rainSnowRgbGt.loadCats(ann['category_id'])[0]['name'])

    matplotlib.rcParams['interactive'] == False
    print("\nRGB Image")
    I = io.imread('../' + rgbImg['file_name'])
    plt.gcf().clear()
    plt.axis('off')
    plt.imshow(I);
    rainSnowRgbGt.showAnns(anns)
    plt.show()

    # For some reason, the image won't show in some Windows/Anaconda configurations. If this is the case, print the image instead
    # plt.savefig("Samples/rgb-" + str(chosenImgId).zfill(5) + ".png")

    print("\nThermal Image")
    # Load thermal annotations
    I = io.imread('../' + thermalImg['file_name'])
    plt.gcf().clear()
    plt.axis('off')
    plt.imshow(I);
    rainSnowThermalGt.showAnns(thermalAnns)
    plt.show()

    # plt.savefig("Samples/thermal-" + str(chosenImgId).zfill(5) + ".png")


# ## Register an annotation in RGB to the thermal domain

# In[12]:


import aauRainSnowUtility

chosenImgId = random.randint(0, 2197)
annIds = rainSnowRgbGt.getAnnIds(imgIds=[chosenImgId])
anns = rainSnowRgbGt.loadAnns(annIds)
rgbImg = rainSnowRgbGt.loadImgs([chosenImgId])[0]


if len(anns) > 0:
    chosenAnnId = random.randint(0, len(anns)-1)
    rgbAnn = anns[chosenAnnId]
    
    thermalSegmentation = []
    for segmentation in rgbAnn['segmentation']:
        thermalCoords = aauRainSnowUtility.registerRgbPointsToThermal(segmentation, rgbImg['file_name'])
        
        
        thermalSegmentation.append(thermalCoords)
        
        print('RGB coordinates for annotation ID ' + str(rgbAnn['id']) +':\n' + str(np.reshape(segmentation, (-1, 2))))
        print('Thermal coordinates:\n' + str(thermalCoords.reshape([-1, 2])))    
    
else:
    print("No annotations found for image ID " + str(chosenImgId) + ", try again")


# ## Register an annotation in thermal to the RGB domain

# In[14]:


import aauRainSnowUtility

chosenImgId = random.randint(0, 2197)
annIds = rainSnowThermalGt.getAnnIds(imgIds=[chosenImgId])
anns = rainSnowThermalGt.loadAnns(annIds)
thermalImg = rainSnowThermalGt.loadImgs([chosenImgId])[0]


if len(anns) > 0:
    chosenAnnId = random.randint(0, len(anns)-1)
    thermalAnn = anns[chosenAnnId]
    
    rgbSegmentation = []
    for segmentation in thermalAnn['segmentation']:
        rgbCoords = aauRainSnowUtility.registerThermalPointsToRgb(segmentation, thermalImg['file_name'])
        
        
        rgbSegmentation.append(rgbCoords)
        
        print('Thermal coordinates for annotation ID ' + str(thermalAnn['id']) +':\n' + str(np.reshape(segmentation, (-1, 2))))
        print('RGB coordinates:\n' + str(rgbCoords.reshape([-1, 2])))    
    
else:
    print("No annotations found for image ID " + str(chosenImgId) + ", try again")

