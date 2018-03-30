#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 09:18:28 2018

@author: gholbrow
"""
import os
import cv2
import numpy as np



imsetDir = '/Users/gholbrow/Dropbox (GoPro)/GOPRO/Stereo Rig/TESTING/Test 4/ImageSet_15'
overwrite = 1
#im = imread('/Users/gholbrow/Dropbox (GoPro)/GOPRO/Stereo Rig/TESTING/Test 4/ImageSet_15/2_pos.jpg')

def main(imsetDir):
    
    #TODO file manager takes input folder and returns base image plus images to be aligned
    midImage,jpegs = fileManager(imsetDir)
    
    #get crop rectangel from midimage
    x, y, width, height = getROI(midImage)
    
    # function that crops all images. Returns list where each sublist is 0- abs path of jpeg 2-full image 3- cropped img
    ready2Align = cropImgs(jpegs,x, y, width, height)
    align2Me    = cropImgs([midImage],x, y, width, height)
    
    aligned = imgAlign(align2Me, ready2Align)
    aligned.sort() #list where each sublist is 0- fname of original jpeg, 1- img post alignment
    
    fileOutput(aligned,overwrite)
            
    #print(midImage, '\n', jpegs)
    
    return(aligned)


def fileManager(inFolder):
    
    if not os.path.isdir(inFolder):
        print('Input is not a folder')
    
    jpegs = []
    for file in os.listdir(inFolder):
        if file.endswith(".jpg") and not file.startswith('.'):    
            jpegs.append(os.path.join(inFolder,file))
    
    if not jpegs:
        print('no Jpeg images found in the input directory')
        
    midImage = os.path.join(inFolder,jpegs[int(len(jpegs)/2)])    
    jpegs.remove(midImage)    
    return(midImage,jpegs)
    

def getROI(imgPath):
    img = cv2.imread(imgPath)
    cv2.namedWindow("Select ROI by clicking and dragging, then press enter",cv2.WINDOW_NORMAL)
    coords = cv2.selectROI("Select ROI by clicking and dragging, then press enter",img)
    cv2.destroyWindow("Select ROI by clicking and dragging, then press enter")
    return(coords)

#imgList is list of abs paths of several jpegs, x, y,width, height is croping params
def cropImgs(imgList,x, y, width, height):
    
    toReturn = []
    for img in imgList:
        subList = []
        fullImg = cv2.imread(img)
        cropImg = fullImg[y:y+height, x:x+width]
        
        subList.append(img)
        subList.append(fullImg)        
        subList.append(cropImg)
        
        toReturn.append(subList)
    
    return(toReturn)

# Read the images to be aligned
def imgAlign(align2, fullList):
    
    align2Name = align2[0][0]
    align2Full = align2[0][1]
    align2Crop = align2[0][2]
    
    alignedImgList = []
    
    
    for subList in fullList:
        
        toAlignFull = subList[1] #full img to be aligned 
        toAlignCrop = subList[2] #cropped image to be aligned
        
        # Convert images to grayscale
        im1_gray = cv2.cvtColor(align2Crop,cv2.COLOR_BGR2GRAY)
        im2_gray = cv2.cvtColor(toAlignCrop,cv2.COLOR_BGR2GRAY)

        
        # Find size of image1
        sz = align2Full.shape
        
        # Define the motion model
        #warp_mode = cv2.MOTION_TRANSLATION
        warp_mode = cv2.MOTION_HOMOGRAPHY
        
        # Define 2x3 or 3x3 matrices and initialize the matrix to identity
        if warp_mode == cv2.MOTION_HOMOGRAPHY :
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else :
            warp_matrix = np.eye(2, 3, dtype=np.float32)
        
        
        # Specify the number of iterations.
        number_of_iterations = 5000;
        
        # Specify the threshold of the increment
        # in the correlation coefficient between two iterations
        termination_eps = 1e-10;
        
        # Define termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
        
        # Run the ECC algorithm. The results are stored in warp_matrix.
        (cc, warp_matrix)  = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria)
        
        if warp_mode == cv2.MOTION_HOMOGRAPHY :
            # Use warpPerspective for Homography
            im2_aligned = cv2.warpPerspective (toAlignFull, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        else :
            # Use warpAffine for Translation, Euclidean and Affine
            im2_aligned = cv2.warpAffine(toAlignFull, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
            
        alignedImgList.append([subList[0],im2_aligned])
        
#        cv2.imwrite('/Users/gholbrow/Downloads/derp1.png',toAlignFull)
#        cv2.imwrite('/Users/gholbrow/Downloads/derp2.png',im2_aligned)
    
    alignedImgList.append( [align2Name, align2Full] ) #adds back in the image that the others are being aligned to
    return(alignedImgList)


def fileOutput(allData, overwrite):
    path,basename = os.path.split(allData[1][0])
    newFolder = os.path.join(path,'Aligned')
    
    if not os.path.isdir(newFolder): 
        os.mkdir(newFolder)
    elif overwrite == 0:
        print('Aligned folder already exists. Use overwrite argument to proceeed')
        return
    
    for data in allData:
        fname = os.path.split(data[0])[1]
        fullName = os.path.join(newFolder,fname)
        #print(data[1])
        cv2.imwrite(fullName,data[1])




    
derp = main(imsetDir)       