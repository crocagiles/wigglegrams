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


imsetDir = r'C:\Users\giles\Pictures\WIGGLEGRAMS\Round1\ImageSet_11'

overwrite = 1



def main(imsetDir,overwrite):
    
    #file manager takes input folder and returns base image plus images to be aligned
    midImage,jpegs = fileManager(imsetDir)
    midImageData = cv2.imread(midImage)
    
    origY,origX, layers = midImageData.shape
    #get crop rectangel from midimage
    x, y, width, height = getROI(midImage) #pass img filename, return 
    
    #Get center point of ROI returned from the getROI function
    xPoint,yPoint = int(width/2+x), int(height/2+y)
#    xPoint,yPoint = 0,0
    #Pad the middle image (which other images will be aligned to)
    midImagePadded,xPad,yPad  = pad2NewCenter(midImage,[xPoint,yPoint])
    
    #find the center location of the padded image
    midImageRows, midImageCols , midImageLayers  = midImagePadded.shape
    centerX, centerY =  int(midImageRows/2), int(midImageCols/2 )
    
    ROIsize = 250
    #midImageCropped = midImagePadded[ centerX-ROIsize : centerX+ROIsize , centerY-ROIsize : centerY+ROIsize]
    
    
#    cv2.imshow('d',midImageCropped)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    
    #This is bundle format. Each other jpeg will have a list like this one
    
    toAlignBundles = []
    #Pad the "to be aligned" jpegs with the exact padding as midImagePadded, please in correct sublist format.
    for jpeg in (jpegs):
        fname = jpeg
        paddedjpeg, xPad2 ,yPad2 = pad2NewCenter(jpeg,[xPoint,yPoint])
        croppedJpeg = paddedjpeg[ centerX-ROIsize : centerX+ROIsize , centerY-ROIsize : centerY+ROIsize]
        toAlignBundles.append([fname,paddedjpeg,croppedJpeg])
    
    #alignedBundles = [[midImageBundle[0], midImageBundle[1]]]
    
    #alignedBundlesCrop = [[midImageBundle[0],midImageBundle[2]]]
    
    alignedBundles = []
    midImageBundlePad = [midImage, midImagePadded  , midImagePadded[centerX-ROIsize:centerX+ROIsize , centerY-ROIsize : centerY+ROIsize] ]
    for bundle in toAlignBundles:
        #bundle.append(bundle[1][ centerX-ROIsize : centerX+ROIsize , centerY-ROIsize : centerY+ROIsize]) #add crop section
        cv2.imshow('d', midImagePadded[centerX-ROIsize:centerX+ROIsize , centerY-ROIsize : centerY+ROIsize])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        fname,fullImg,cropSect   = bundle[0], bundle[1], bundle[2]
        fullAlign, alignCrop = imgAlign(midImageBundlePad,bundle,cv2.MOTION_TRANSLATION) #cv2.MOTION_EUCLIDEAN, cv2.MOTION_TRANSLATION
        fullAlignTrim = removePadding(fullAlign,xPad,yPad,origY,origX )
        alignedBundles.append([fname,fullAlignTrim,alignCrop])
     
    #alignedBundles.append(midImageBundle)
#        alignedBundlesCrop.append([fname,alignCrop])
    #alignedBundles[0][1] =  removePadding(alignedBundles[0][1],xPad,yPad,origY,origX) 
#    aligned,alignedCropped = imgAlign(midImageBundle, toAlignBundles) #pass align2Me ()
#   
        
    #alignedBundles.sort()
    #alignedBundlesCrop.sort()

    #
    fullyAligned = []
    midImageCropped = midImageData[ centerX-ROIsize : centerX+ROIsize , centerY-ROIsize : centerY+ROIsize]
    midImageBundle = [midImage, midImageData,midImageCropped ] 
    for aligned in alignedBundles:
        
        cv2.imshow('d',midImageCropped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
        cropped = aligned[1][ centerX-ROIsize : centerX+ROIsize , centerY-ROIsize : centerY+ROIsize]
        
        cv2.imshow('d',cropped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        aligned.append(cropped) # add crop center section for xy translation calc
        fname,fullImg,cropSect   = aligned[0], aligned[1], aligned[2]
        
        fullAlignXY, alignCropXY = imgAlign(midImageBundle,aligned,cv2.MOTION_TRANSLATION)
        fullyAligned.append([fname,fullAlignXY])
    fullyAligned.append(midImageBundle) 
    #fullyAligned.sort()
#    beforeImsCropped = [toAlignBundles[0][2], midImageBundle [2], toAlignBundles[1][2]]
#    afterImsCropped  = [x[1] for x in alignedBundlesCrop]
#    afterafterImsCropped = [x[1][ (centerX-ROIsize) : (centerX+ROIsize) , (centerY-ROIsize) : (centerY+ROIsize)] for x in alignedBundles]
#    
    
    
    
    imDisp(beforeImsCropped,afterImsCropped,afterafterImsCropped)
    
    #list where each sublist is 0- fname of original jpeg, 1- img post alignment
    fileOutput(fullyAligned,overwrite)
            
    #print(midImage, '\n', jpegs)
 

def removePadding(paddedImg,xPaddingData,yPaddingData,origDimY,origDimX):

    
    if xPaddingData[0] == 'right':
        depadImgx= paddedImg[ : , :origDimX  ]
    else:
        depadImgx= paddedImg[ : ,  xPaddingData[1]: ]
        
    if yPaddingData[0] == 'bottom':
        depadImg = depadImgx[ :origDimY ,  :]
    else:
        depadImg = depadImgx[ xPaddingData[1]:  ,  :]
        
    return(depadImg)


# =============================================================================
# pad2NewCenter(image,newCenter)
#    input:
#        image - abs path to rgb img
#        newCenter - List with length of two`
#            newCenter[0] - X/Column location of new center for the image
#            newCenter[1] - Y/Row location of new center for the image
#    output:
#        padded - padded RGB image with image center at location: newCenter
# =============================================================================
def pad2NewCenter(image,newCenter):
    imArray = cv2.imread(image)
    rows, cols, layers =  imArray.shape

    offsetX = abs(int(2 * (cols/2 - abs(newCenter[0] - cols))))
    offsetY = abs(int(2 * (rows/2 - abs(newCenter[1] - rows))))
    #imgray = np.pad(imgray,((100,100),(100,100)),'constant', constant_values=(0,0))

    #X Padding
    if newCenter[0] <= cols/2:
        paddingX = 'left'
        padTupX = (offsetX,0)
    else:
        paddingX = 'right'
        padTupX = (0,offsetX)
        
    #Y Padding    
    if newCenter[1] <= rows/2:
        paddingY = 'top'
        padTupY = (offsetY,0)
    else:
        paddingY = 'bottom'
        padTupY = (0,offsetY)
    
    paddedXY = np.pad(imArray,(padTupY,padTupX,(0,0)),'constant', constant_values=(50))
    
    return(paddedXY, [paddingX,offsetX], [paddingY,offsetY]) 

# =============================================================================
# fileManager(inFolder)
#    input:
#       inFolder - abs path to folder containing at least 2 stereoscopic rgb images
#    output:
#       midImage - abs path to rgb image that other images will be aligned to. 
#       jpegs - list where jpegs[n] is abs path to jpeg image that will be aligned to midImage
# =============================================================================
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
    

# =============================================================================
# getROI(imgPath)
#    Input:
#       imgPath - abs path jpeg image
#    Output:
#       x       - lower x bound of user selected area (pixel location)
#       y       - lower y bound of user selected area (pixel location)
#       width   - width of user selected area (pixels)  
#       height  - height of the user selected area (pixels)      
# =============================================================================
def getROI(imgPath):
    img = cv2.imread(imgPath)
    cv2.namedWindow("Select ROI by clicking and dragging, then press enter",cv2.WINDOW_NORMAL)
    coords = cv2.selectROI("Select ROI by clicking and dragging, then press enter",img)
    cv2.destroyWindow("Select ROI by clicking and dragging, then press enter")
    
    x = coords[0]
    y = coords[1]
    width = coords[2]
    height = coords[3]
    
    return(x,y,width,height)



# =============================================================================
# cropImgs(img,x, y, width, height)
#    Input:
#       img     - numpy array image
#       x       - lower x bound of user selected area (pixel location)
#       y       - lower y bound of user selected area (pixel location)
#       width   - width of user selected area (pixels)  
#       height  - height of the user selected area (pixels)
#    output:
#       cropImg - Cropped subsection of input image          
# =============================================================================
def cropImg(img,x, y, width, height):
    
    cropImg = img[y:y+height, x:x+width]
    return(cropImg)

#takes  lists of images of the same size and returns the stacked on top ofeachother in a window    
def imDisp(beforeIms,afterIms,afterafterImsCropped):
    before = np.concatenate( beforeIms, axis = 0 )
    after =  np.concatenate( afterIms,  axis = 0 )
    afterafter = np.concatenate( afterafterImsCropped,  axis = 0 )
    beforeAfter = np.concatenate([before, after,afterafter], axis = 1  )#axis = 0)
    
    cv2.imwrite('/Users/gholbrow/Dropbox (GoPro)/GOPRO/Stereo Rig/TESTING/Test 4/ImageSet_15/Aligned/derpydoo.jpg',beforeAfter)
#    cv2.imshow('top',beforeAfter)
#    cv2.waitKey(0)    
    
    

# Read the images to be aligned
def imgAlign(align2, toAlign,warpMode):
    
    align2Name = align2[0] #fname
    align2Full = align2[1] #aligning other images to this image, full res
    align2Crop = align2[2] #crop section to do actual alignment 
    
    toAlignName = toAlign[0]
    toAlignFull = toAlign[1]
    toAlignCrop = toAlign[2]
        
    
    align2CropGray  = cv2.cvtColor(align2Crop,cv2.COLOR_BGR2GRAY)
    toAlignCropGray = cv2.cvtColor(toAlignCrop,cv2.COLOR_BGR2GRAY)

    
    # Find size of image1
    sz = align2Full.shape
    szCrop = align2Crop.shape
    # Define the motion model
    #warp_mode = cv2.MOTION_TRANSLATION
    warp_mode = warpMode
    #warp_mode = cv2.MOTION_HOMOGRAPHY
    
    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 5000;
    
    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10;
    
    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
    
    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix)  = cv2.findTransformECC (align2CropGray,toAlignCropGray,warp_matrix, warp_mode, criteria)
       
    # Use warpAffine for Translation, Euclidean and Affine
    im_aligned         = cv2.warpAffine(toAlignFull, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
    im_aligned_cropped = cv2.warpAffine(toAlignCrop, warp_matrix, (szCrop[1],szCrop[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
             
    


    
    
    
    
    
    
    return(im_aligned,im_aligned_cropped)


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




    
main(imsetDir,overwrite)
