from pycocotools import coco
import numpy as np
import skimage.io as io
import matplotlib
import matplotlib.pyplot as plt
import pylab
import shutil 
import random
import cv2
import pdb
import json
import random


pylab.rcParams['figure.figsize'] = (8.0, 6.0)

rgbAnnFile='../aauRainSnow-rgb.json'
thermalAnnFile = '../aauRainSnow-thermal.json'

rainSnowRgbGt = coco.COCO(rgbAnnFile)
rainSnowThermalGt = coco.COCO(thermalAnnFile)

annotations = open(rgbAnnFile)
coco = json.load(annotations)
images = coco['images']
random.shuffle(images)
for item in images:
    chosenImgId = item['id']#np.round(random.uniform(0, 2197))
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

    fig = plt.figure(1)
    grid = plt.GridSpec(2,2)
    plt.clf()

    ax1 = plt.subplot(grid[0,0])
    ax1.axis('off')
    plt.imshow(I)

    ax1 = plt.subplot(grid[1,0])
    ax1.axis('off')
    plt.imshow(I)
    rainSnowRgbGt.showAnns(anns)



    print("\nThermal Image")
    # Load thermal annotations
    I = io.imread('../' + thermalImg['file_name'])
    ax2 = plt.subplot(grid[0,1])
    ax2.axis('off')
    plt.imshow(I)

    ax2 = plt.subplot(grid[1,1])
    ax2.axis('off')
    plt.imshow(I)
    rainSnowThermalGt.showAnns(thermalAnns)


    plt.show()





