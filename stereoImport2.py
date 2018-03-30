#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 17:00:52 2018

@author: gholbrow
"""
import os
import shutil
import argparse

#Path to SD containing images, output path, L/M/R specification. 

sdPath = '/Volumes/NO NAME/DCIM/100GOPRO'
outPath = '/Users/gholbrow/Dropbox (GoPro)/GOPRO/Stereo Rig/TESTING/Test 4'

#position from leftmost camera
position = 2
overWrite = 0 #True

def multiCamImport(outPath,sdPath,position, overWrite):
    
    #TODO validate sdPath
    if not os.path.isdir(sdPath):
        print('Input dir is not a real path.')
        return
        
    #TODO check outpath. Is it dir? if not make it
    if not os.path.isdir(outPath):
        try:
            os.mkdir(outPath)
            print('Created ' + outPath)
        except:
            print('Error making output directory, please check structure')
            
    
    #TODO check folders, make them if they don't exist. 
    #Get all jpegs fnames from sdPath
    sdContents = []
    for file in os.listdir(sdPath):
        if file.endswith(".JPG") and not file.startswith('.'):    
            sdContents.append(file)
    sdContents.sort()
    
    #Generate folder name for each file
    numFolders = len(sdContents)
    folders = []
    for item in enumerate(sdContents):
        padding = len(str(numFolders))
        paddedI = str(item[0]+1).zfill(padding)
        newFolder = os.path.join(outPath,'ImageSet_' + paddedI)
        folders.append(newFolder)
    
    #Make folders if folders don't exist. If folder does exist, use subdirs of sdPath
    for folder in folders:
        if os.path.isdir(folder):
            folders = [x[0] for x in os.walk(outPath)][1:]
            break
        if not os.path.isdir(folder):
            os.mkdir(folder)

    if len(folders) != len(sdContents):
        print('Length of jpeg list is not equal to number of folders. returning.')
        return
            

    #try to move jpegs
    for jpg in enumerate(sdContents):
        sdLoc   = os.path.join(sdPath,jpg[1])
        newName = str(position) + '_pos.jpg'
        newLoc  = os.path.join(folders[jpg[0]],newName)
        
        if overWrite:
            shutil.copyfile(sdLoc,newLoc)
            print('Copied ' + sdLoc + ' to ' + newLoc)
        else:
            if os.path.isfile(newLoc):
                print('Looks like this position has already been written (position ' + str(position) + ')')
                print('please use overwrite argument to overwrite (-o)')
                break
            else:
                shutil.copyfile(sdLoc,newLoc)
                print('Copied ' + sdLoc + ' to ' + newLoc)
def Main():
    
    parser = argparse.ArgumentParser(description= "Import stereo jpegs."  )
    
   
    parser.add_argument("outPath",
                        help="Where jpegs will be transferred. Subfolders will be created in this directory.", type=str)
    parser.add_argument('sdPath', help="Path to folder on SD card where jpegs are located" , type=str)
    parser.add_argument("Position", 
                        help="1 for leftmost, 2 second from left, 3 for third, etc", type=int)
    parser.add_argument("-o","--overwrite", 
                        help="This argument will allow you to overwrite previously transferred jpegs. Otherwise, you will see an error if you import from the same position twice.", action="store_true")

    args = parser.parse_args()
    multiCamImport(args.outPath,args.sdPath,args.Position,args.overwrite)
            
    
    #TODO copy images to new folders, with Position as first char. if images exist, warn and allow overwrite

if __name__ == '__main__':
    Main()
    