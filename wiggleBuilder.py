#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 09:18:28 2018

@author: gholbrow
"""
import os
import cv2
import numpy as np
import sys


imsetDir = r'C:\Users\giles\Pictures\WIGGLEGRAMS\Round2 Skiing\Wigglz\Julien'

overwrite = 1



def main(imsetDir,overwrite):
    
    #file manager takes input folder and returns base image plus images to be aligned
    midImage,jpegs = fileManager(imsetDir)
    
    #get crop rectangel from midimage
    x, y, width, height = getROI(midImage) #pass img filename, return 
    xPoint,yPoint = int(width/2+x), int(height/2+y)
    
    padded = enlargAboutCenter(midImage,[xPoint,yPoint])
    
    small = cv2.resize(padded, (0,0), fx=0.25, fy=0.25)
    cv2.imshow('d',small)
    cv2.waitKey(0)

    
    sys.exit()
    # function that crops all images. Returns list where each sublist is 0- abs path of jpeg 2-full image 3- cropped img
    ready2Align = cropImgs(jpegs,x, y, width, height) #pass list of jpeg filenames, returns with list where sublist is list [fname,numpyArrayCropped]
    align2Me    = cropImgs([midImage],x, y, width, height)
#    aligned = imgAlign(align2Me, ready2Align)
    
    
    aligned,alignedCropped = imgAlign(align2Me, ready2Align) #pass align2Me ()
    
    aligned.sort()
    alignedCropped.sort()
    
    beforeImsCropped = [x[2] for x in ready2Align] + [y[2] for y in align2Me]
    afterImsCropped  = [x[1] for x in alignedCropped]
    
    afterafterImsCropped = cropImgs( [x[0] for x in aligned] ,x, y, width, height)
    
    #imDip(beforeIms,AfterIms):
    #imDisp(beforeImsCropped,afterImsCropped,[x[2] for x in afterafterImsCropped])
    #list where each sublist is 0- fname of original jpeg, 1- img post alignment
    fileOutput(aligned,overwrite)
            
    #print(midImage, '\n', jpegs)
    
    return(aligned)

def enlargAboutCenter(image,newCenter):
    im = cv2.imread(image)
    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    
    rows, cols =  imgray.shape
    offsetX = int(abs(cols/2-(newCenter[0])))
    offsetY = int(abs(rows/2-(newCenter[1])))
    
    imgray = np.pad(imgray,((100,100),(100,100)),'constant', constant_values=(0,0))

    
    if newCenter[0] <= cols/2:
        padSize = 'left'
        imgray = np.pad(imgray,((0,0),(offsetX,1)),'constant', constant_values=(0))
    else:
        padSideX = 'right'
        imgray = np.pad(imgray,((0,0),(1,offsetX)),'constant', constant_values=(0))
    
    if newCenter[1] <= rows/2:
        padSideY = 'top'
        imgray = np.pad(imgray,((offsetY,0),(0,0)),'constant', constant_values=(0))
    else:
        padSideY = 'bottom'
        imgray = np.pad(imgray,((0,offsetY),(0,0)),'constant', constant_values=(0))
    
    paddedImage = imgray
    return(paddedImage)
    
    

def fileManager(inFolder):
    
    if not os.path.isdir(inFolder):
        print('Input is not a folder')
    
    jpegs = []
    for file in os.listdir(inFolder):
        if file.endswith(".jpg") or file.endswith(".JPG") and not file.startswith('.'):    
            jpegs.append(os.path.join(inFolder,file))
    
#    print(jpegs)
    if len(jpegs) < 2:
        print('\nError: You must have at least two jpeg images in the input directory.\n')
        sys.exit()
        return
    
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

#takes two lists of images of the same size and returns the stacked on top ofeachother in a window    
def imDisp(beforeIms,afterIms,afterafterImsCropped):
    before = np.concatenate( beforeIms, axis = 0 )
    after =  np.concatenate( afterIms,  axis = 0 )
    afterafter = np.concatenate( afterafterImsCropped,  axis = 0 )
    beforeAfter = np.concatenate([before, after,afterafter], axis = 1  )#axis = 0)
    
    cv2.imshow('top',beforeAfter)
    cv2.waitKey(0)    
    
    

# Read the images to be aligned
def imgAlign(align2, fullList):
    
    align2Name = align2[0][0] #fname
    align2Full = align2[0][1] #aligning other images to this image, full res
    align2Crop = align2[0][2] #crop section to do actual alignment 
    
    
    alignedCroppedList = [[align2Name, align2Crop]]
    alignedImgList     = [[align2Name, align2Full]]
    
    for subList in fullList:
        
        toAlignFull = subList[1] #full img to be aligned 
        toAlignCrop = subList[2] #cropped image to be aligned
        
        # =============================================================================
        # XY Translation Correction
        # =============================================================================
        # Convert images to grayscale
        align2CropGray = cv2.cvtColor(align2Crop,cv2.COLOR_BGR2GRAY)
        toAlignCropGray = cv2.cvtColor(toAlignCrop,cv2.COLOR_BGR2GRAY)

        
        # Find size of image1
        sz = align2Full.shape
        szCrop = align2Crop.shape
        # Define the motion model
        warp_mode = cv2.MOTION_TRANSLATION
        #warp_mode = cv2.MOTION_EUCLIDEAN
        #warp_mode = cv2.MOTION_HOMOGRAPHY
        
        # Define 2x3 or 3x3 matrices and initialize the matrix to identity
        warp_matrix = np.eye(2, 3, dtype=np.float32)

        # Specify the number of iterations.
        number_of_iterations = 1000;
        
        # Specify the threshold of the increment
        # in the correlation coefficient between two iterations
        termination_eps = 1e-10;
        
        # Define termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
        
        # Run the ECC algorithm. The results are stored in warp_matrix.
        (cc, warp_matrix)  = cv2.findTransformECC (align2CropGray,toAlignCropGray,warp_matrix, warp_mode, criteria)
        print(warp_matrix,'\n')
        
        # Use warpAffine for Translation, Euclidean and Affine
        im_alignedXY         = cv2.warpAffine(toAlignFull, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
        im_alignedXY_cropped = cv2.warpAffine(toAlignCrop, warp_matrix, (szCrop[1],szCrop[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
        

#        cv2.imwrite('/Users/gholbrow/Downloads/derp1.png',toAlignFull)
#        cv2.imwrite('/Users/gholbrow/Downloads/derp2.png',im2_aligned)
        
        # =============================================================================
        #         ROTATION CORRECTION   
        # =============================================================================
        rotCropROI = 500
        rows,cols,layers = align2Full.shape
        rotCropBase = im_alignedXY[int(rows/2 -rotCropROI):int(rows/2+rotCropROI), int(cols/2-rotCropROI):int(cols/2+rotCropROI)]
        rotCropn    = align2Full[int(rows/2 -rotCropROI):int(rows/2+rotCropROI), int(cols/2-rotCropROI):int(cols/2+rotCropROI)]
    
        
        szRot = rotCropBase.shape
        
        rotCropBaseGray = cv2.cvtColor(rotCropBase,cv2.COLOR_BGR2GRAY)
        rotCropnBase    = cv2.cvtColor(rotCropn,cv2.COLOR_BGR2GRAY)
        
        
        cv2.imshow('at',rotCropBase)
        cv2.waitKey(0)
        cv2.imshow('atat',rotCropn)
        
        cv2.waitKey(0) 
        
        warp_mode2 = cv2.MOTION_EUCLIDEAN
        warp_matrix2 = np.eye(2, 3, dtype=np.float32)
        
        (cc, warp_matrix2)  = cv2.findTransformECC (rotCropBaseGray,rotCropnBase,warp_matrix2, warp_mode2, criteria)
        
        im_alignedRot = cv2.warpAffine(im_alignedXY, warp_matrix2, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
        im_alignedRotCropped = cv2.warpAffine(im_alignedXY_cropped, warp_matrix2, (szRot[1],szRot[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
        


        alignedImgList.append(      [subList[0],im_alignedRot])
        alignedCroppedList.append(  [subList[0],im_alignedRotCropped])        
        
        
       

    
    return(alignedImgList,alignedCroppedList)


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
    print('wrote', len(data), 'aligned images to:\n', newFolder)




    
derp = main(imsetDir,overwrite)       