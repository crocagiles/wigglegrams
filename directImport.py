# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 13:59:34 2018

@author: Giles
"""
import os
import time
import sys
import shutil

cam1 = 'F:/'
cam2 = 'G:/'
cam3 = 'H:/'
tol  = 2
outputDir = 'C:/Users/giles/Pictures/WIGGLEGRAMS/round3 import'
overwrite = 0


rootDirList = [cam1,cam2,cam3]


cam1List = []
cam2List = []
cam3List = []

for rootDir in rootDirList:
    for dirName, subdirList, fileList in os.walk(rootDir):
        for fname in fileList:
            if 'GOPRO' in dirName:
                absPath = os.path.join(dirName,fname)
                timestamp = os.path.getctime(absPath)
                #print('\t%s' % fname, "created: %s" % time.ctime(timestamp)) 
                if dirName.startswith(cam1):
                    cam1List.append([absPath,timestamp])
                elif dirName.startswith(cam2):
                    cam2List.append([absPath,timestamp])
                elif dirName.startswith(cam3):
                    cam3List.append([absPath,timestamp])                    

fullSets = []
                
for image1 in cam1List:
    photo1Path = image1[0]
    cam1Time   = image1[1]
    matchList = []
    matchList.append(image1)
    
    for image2 in cam2List:
        photo2Path = image2[0]
        cam2Time   = image2[1]
        if (cam1Time - tol) < cam2Time < (cam1Time + tol):
            matchList.append(image2)
             
    for image3 in cam3List:
        photo3Path = image3[0]
        cam3Time   = image3[1]
        if (cam1Time - tol) < cam3Time < (cam1Time + tol):
            matchList.append(image3)
            
    if len(matchList) == 3:
        fullSets.append(matchList)

if not os.path.isdir(outputDir):
    try:
        os.mkdir(outputDir)        
    except:
        print('error making directory')

for imSet in enumerate(fullSets):
    
    setDir = os.path.join(outputDir,'ImageSet_'+str(imSet[0]))
    
    if not os.path.isdir(setDir): 
        try:
            os.mkdir(setDir)
        except:
            print('error creating output directory')
            sys.exit()

            
#        print(setDir)
    for ims in enumerate(imSet[1]):
#        print(ims[1])
        oldPath = ims[1][0]
        newPath = os.path.join(setDir,'Image_' + str(ims[0]) + '.jpg')
        if os.path.isfile(newPath):
            if overwrite != 1:
                print('An output file already exists. Please use overwrite keyword to allow overwrite.')
                sys.exit()
        else:
            shutil.copy(oldPath,newPath)
            
        print(oldPath, 'Moved to', newPath)
        
        

        
        
    print('\n')
    
    
    
#    im1 = imSet[1][0]
#    im2 = imSet[1][1]
#    im3 = imSet[1][2]
#    print(im1,im2,im3)
#    
    #os.mkdir(setDir)
    
#    for subSet[1] in imSet:
#        print(subSet,'\n')
    
    

    