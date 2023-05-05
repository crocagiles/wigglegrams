#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 09:18:28 2018

@author: crocagiles
"""
import os
import cv2
import numpy as np
import sys
import argparse
from moviepy.editor import ImageSequenceClip
# from moviepy.video.fx import resize


# imsetDir = r'/Users/gholbrow/Dropbox (GoPro)/GOPRO/Stereo Rig/TESTING/Test 4/ImageSet_15/'

# overwrite = 1
# percentCrop = 3
# ROIsize = 250

def main(imsetDir, overwrite, ROIsize, percentCrop):
    """
    :param imsetDir: Abs directory containing jpegs to be aligned.
    :param overwrite: If True, will overwrite data from previously aligned images
    :param ROIsize: Size of ROI used for alignment. User selects center of ROI.
    :param percentCrop: Use to avoid images non overlap at image periphery. Default is 3%.
    :return: True. Aligned images, output vid, and gif, are placed in subdir of imsetDir
    """
    if ROIsize == None:
        ROIsize = 500
    if percentCrop == None:
        percentCrop = 3

    ROIsize = int(ROIsize / 2)

    # file manager takes input folder and returns base image plus images to be aligned
    midImage, jpegs = find_mid_image(imsetDir)
    midImageData = cv2.imread(midImage)

    origY, origX, layers = midImageData.shape
    # get crop rectangel from midimage
    x, y, width, height = getROI(midImage)  # pass img filename, return

    # Get center point of ROI returned from the getROI function
    xPoint, yPoint = int(width / 2 + x), int(height / 2 + y)

    # Pad the middle image (which other images will be aligned to)
    midImagePadded, xPad, yPad = pad_to_new_center(midImage, [xPoint, yPoint])

    # find the center location of the padded image
    midImageRows, midImageCols, midImageLayers = midImagePadded.shape
    centerX, centerY = int(midImageRows / 2), int(midImageCols / 2)

    toAlignBundles = []
    # Pad the "to be aligned" jpegs with the exact padding as midImagePadded, please in correct sublist format.
    for jpeg in (jpegs):
        fname = jpeg
        paddedjpeg, xPad2, yPad2 = pad_to_new_center(jpeg, [xPoint, yPoint])
        croppedJpeg = paddedjpeg[centerX - ROIsize: centerX + ROIsize, centerY - ROIsize: centerY + ROIsize]
        toAlignBundles.append([fname, paddedjpeg, croppedJpeg])
    toAlignBundles.sort()

    # Rotational Alignment
    print('\n...Rotational alignment in progress...')
    alignedBundles = []
    midImagePadCrop = midImagePadded[centerX - ROIsize:centerX + ROIsize, centerY - ROIsize: centerY + ROIsize]
    midImageBundlePad = [midImage, midImagePadded, midImagePadCrop]
    for bundle in enumerate(toAlignBundles):
        print('Aligning Image', str(bundle[0] + 1), 'of', str(len(toAlignBundles)))
        fname = bundle[1][0]
        fullAlign, alignCrop = imgAlign(midImageBundlePad, bundle[1],
                                        cv2.MOTION_EUCLIDEAN)  # cv2.MOTION_EUCLIDEAN, cv2.MOTION_TRANSLATION
        fullAlignTrim = removePadding(fullAlign, yPad, xPad, origY, origX, midImageRows, midImageCols)
        alignedBundles.append([fname, fullAlignTrim, alignCrop])
    alignedBundles.sort()

    # XY Translation Alignment
    fullyAligned = []
    midImageCropped = midImageData[yPoint - ROIsize: yPoint + ROIsize, xPoint - ROIsize: xPoint + ROIsize]
    midImageBundle = [midImage, midImageData, midImageCropped]

    print('\n...XY translation alignment in progress...')
    for aligned in enumerate(alignedBundles):
        print('Aligning Image', str(aligned[0] + 1), 'of', str(len(aligned)))
        alignedNoEnum = aligned[1]
        cropped = alignedNoEnum[1][yPoint - ROIsize: yPoint + ROIsize, xPoint - ROIsize: xPoint + ROIsize]
        alignedNoEnum[2] = cropped  # add crop center section for xy translation calc
        fname = alignedNoEnum[0]
        fullAlignXY, alignCropXY = imgAlign(midImageBundle, alignedNoEnum, cv2.MOTION_TRANSLATION)
        ready2WriteOut = cropByPercent(fullAlignXY, percentCrop)
        fullyAligned.append([fname, ready2WriteOut, alignCropXY])

    beforeAfterComp(midImageCropped, toAlignBundles, fullyAligned)

    # cropMidImage
    midImageBundle[1] = cropByPercent(midImageData, percentCrop)
    fullyAligned.append(midImageBundle)
    fullyAligned.sort()

    # duplicate some images to create loop effect

    loopBack = list(reversed(fullyAligned[1:-1]))
    fullyAlignedLoop = fullyAligned + loopBack  # images  1 2 3 4 become 1 2 3 4 3 2 to make loop

    toOutput = []
    # change filenames for output
    for bundle in enumerate(fullyAlignedLoop):
        iterate = bundle[0]
        contents = bundle[1]

        origPath, origName = os.path.split(contents[0])
        newName = 'aligned_' + str(iterate) + '.jpg'
        fullnew = os.path.join(origPath, newName)
        fullyAlignedLoop[iterate][0] = fullnew

        toOutput.append([fullnew, contents[1]])

    fileOutput(toOutput, overwrite)

    return True

def removePadding(paddedImg, yPaddingData, xPaddingData, origDimY, origDimX, paddedY, paddedX):
    if xPaddingData == 'right':
        depadImgx = paddedImg[:, :origDimX]
    else:
        depadImgx = paddedImg[:, paddedX - origDimX:]

    if yPaddingData == 'bottom':
        depadImg = depadImgx[:origDimY, :]
    else:
        depadImg = depadImgx[paddedY - origDimY:, :]

    return (depadImg)

def pad_to_new_center(image, newCenter):
    """
    :param image: abs path to rgb img
    :param newCenter: List with length of two
#            newCenter[0] - X/Column location of new center for the image
#            newCenter[1] - Y/Row location of new center for the image
    :return: padded RGB image with image center at location newCenter, padding amount in px
    """
    imArray = cv2.imread(image)
    rows, cols, layers = imArray.shape

    offsetX = abs(int(2 * (cols / 2 - abs(newCenter[0] - cols))))
    offsetY = abs(int(2 * (rows / 2 - abs(newCenter[1] - rows))))
    # imgray = np.pad(imgray,((100,100),(100,100)),'constant', constant_values=(0,0))

    # X Padding
    if newCenter[0] <= cols / 2:
        paddingX = 'left'
        padTupX = (offsetX, 0)
    else:
        paddingX = 'right'
        padTupX = (0, offsetX)

    # Y Padding
    if newCenter[1] <= rows / 2:
        paddingY = 'top'
        padTupY = (offsetY, 0)
    else:
        paddingY = 'bottom'
        padTupY = (0, offsetY)

    paddedXY = np.pad(imArray, (padTupY, padTupX, (0, 0)), 'constant', constant_values=(50))

    return paddedXY, paddingX, paddingY

def find_mid_image(inFolder):
    """
    :param inFolder: abs path to folder containing at least 2 stereoscopic rgb images
    :return: midimage that other imgs will be aligned to. list of jpegs that will be aligned to midimage.
    """
    if not os.path.isdir(inFolder):
        print('Input is not a folder')

    jpegs = []
    for file in os.listdir(inFolder):
        if file.endswith(".jpg") or file.endswith(".JPG") and not file.startswith('.'):
            jpegs.append(os.path.join(inFolder, file))

    #    print(jpegs)
    if len(jpegs) < 2:
        print('\nError: You must have at least two jpeg images in the input directory.\n')
        sys.exit()
        return 0

    if not jpegs:
        print('no Jpeg images found in the input directory')

    midImage = os.path.join(inFolder, jpegs[int(len(jpegs) / 2)])
    jpegs.remove(midImage)
    return midImage, jpegs

def getROI(imgPath):
    """
    :param imgPath: abs path jpeg image
    :return:
        x       - lower x bound of user selected area (pixel location)
        y       - lower y bound of user selected area (pixel location)
        width   - width of user selected area (pixels)
        height  - height of the user selected area (pixels)
    """
    img = cv2.imread(imgPath)
    print('\nPlease select a point on the image and press enter to confirm.')
    cv2.namedWindow("Select ROI by clicking and dragging, then press enter", cv2.WINDOW_NORMAL)
    coords = cv2.selectROI("Select ROI by clicking and dragging, then press enter", img)
    cv2.destroyWindow("Select ROI by clicking and dragging, then press enter")

    x = coords[0]
    y = coords[1]
    width = coords[2]
    height = coords[3]

    return (x, y, width, height)

def cropImg(img, x, y, width, height):
    """
    :param img: numpy array image
    :param x: lower x bound of user selected area (pixel location)
    :param y: lower y bound of user selected area (pixel location)
    :param width: width of user selected area (pixels)
    :param height: height of the user selected area (pixels)
    :return: cropImg - Cropped subsection of input image, numpy array
    """
    cropImg = img[y:y + height, x:x + width]
    return (cropImg)

def beforeAfterComp(compare2, beforeList, afterList):
    assert len(beforeList) == len(afterList)
    preVsPost = []
    # view image alignment
    for i in range(len(beforeList)):
        assert beforeList[i][0] == afterList[i][0]  # fnames should match
        assert beforeList[i][0] != compare2
        assert afterList[i][0] != compare2
        imgName = os.path.split(beforeList[i][0])[1]
        croppedPre = cv2.cvtColor(beforeList[i][2], cv2.COLOR_BGR2GRAY)
        croppedPost = cv2.cvtColor(afterList[i][2], cv2.COLOR_BGR2GRAY)
        compare2Me = cv2.cvtColor(compare2, cv2.COLOR_BGR2GRAY)
        preComp = cv2.subtract(compare2Me, croppedPre) * 2
        postComp = cv2.subtract(compare2Me, croppedPost) * 2

        preVsPost.append([imgName, preComp, postComp])

    #    imsNames = [x[0] for x in preVsPost]
    preIms = [x[1] for x in preVsPost]
    postIms = [x[2] for x in preVsPost]

    preSet = np.concatenate(preIms, axis=0)
    postSet = np.concatenate(postIms, axis=0)

    beforeAfter = np.concatenate([preSet, postSet], axis=1)  # axis = 0)
    beforeAfter = cv2.resize(beforeAfter, (0, 0), fx=0.75, fy=0.75)
    cv2.imshow('before Vs after', beforeAfter)
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    return True

def imgAlign(align2, toAlign, warpMode):
    #    align2Name = align2[0] #fname
    align2Full = align2[1]  # aligning other images to this image, full res
    align2Crop = align2[2]  # crop section to do actual alignment

    #    toAlignName = toAlign[0]
    toAlignFull = toAlign[1]
    toAlignCrop = toAlign[2]

    align2CropGray = cv2.cvtColor(align2Crop, cv2.COLOR_BGR2GRAY)
    toAlignCropGray = cv2.cvtColor(toAlignCrop, cv2.COLOR_BGR2GRAY)

    # Find size of image1
    sz = align2Full.shape
    szCrop = align2Crop.shape
    # Define the motion model
    # warp_mode = cv2.MOTION_TRANSLATION
    warp_mode = warpMode
    # warp_mode = cv2.MOTION_HOMOGRAPHY

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 5000;

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10;

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC(align2CropGray, toAlignCropGray, warp_matrix, warp_mode, criteria, None, 1)

    # Use warpAffine for Translation, Euclidean and Affine
    im_aligned = cv2.warpAffine(toAlignFull, warp_matrix, (sz[1], sz[0]),
                                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
    im_aligned_cropped = cv2.warpAffine(toAlignCrop, warp_matrix, (szCrop[1], szCrop[0]),
                                        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);

    return im_aligned, im_aligned_cropped

def cropByPercent(imgToCrop, percentCrop):
    if percentCrop == 0:
        return (imgToCrop)

    imgRows, imgCols, layers = imgToCrop.shape
    removeRowsPerSide = int((imgRows - (imgRows * (1 - percentCrop / 100))) / 2)
    removeColsPerSide = int((imgCols - (imgCols * (1 - percentCrop / 100))) / 2)

    croppedImg = imgToCrop[removeRowsPerSide:imgRows - removeRowsPerSide, removeColsPerSide:imgCols - removeColsPerSide]

    return croppedImg

def fileOutput(allData, overwrite):
    path, basename = os.path.split(allData[1][0])
    newFolder = os.path.join(path, 'Aligned')

    if not os.path.isdir(newFolder):
        os.mkdir(newFolder)
    elif overwrite == 0 or overwrite == None:
        print('\nAligned folder already exists. Use overwrite argument to proceeed')
        return 0

    toMakeGif = []
    for data in allData:
        fname = os.path.split(data[0])[1]
        fullName = os.path.join(newFolder, fname)
        # print(data[1])
        cv2.imwrite(fullName, data[1])
        toMakeGif.append(fullName)
    print('\nWrote', len(allData), 'aligned images to:\n', newFolder)

    gifOutput(toMakeGif, os.path.join(newFolder, 'gifAligned.gif'))
    print('\nWrote gifAligned.gif to:\n', newFolder)

    videoOutput(toMakeGif, os.path.join(newFolder, 'loopVid.mp4'), 10)
    print('\nWrote loopVid.mp4 to:\n', newFolder)

    return True

def videoOutput(imgList, outLocation, numLoops):
    # writing video out
    frameList = []

    for i in range(numLoops):
        for img in imgList:
            frameList.append(img)

    vid_clip = ImageSequenceClip(frameList, fps=10)
    vid_clip.write_videofile(outLocation)
    return True

def gifOutput(gifImgList, outLocation):
    # imgDataList = []
    # for img in gifImgList:
    #     imgDataList.append(imageio.imread(img))
    # imageio.mimsave(outLocation, imgDataList)

    clip = ImageSequenceClip(gifImgList, fps=10.0, load_images=True)
    clip2 = clip.resize(.2)  # resized = resize(clip)

    clip2.write_gif(outLocation, program='imageio', opt='nq', fuzz=1, )
    return True

def argParser():
    parser = argparse.ArgumentParser(
        description="Pass a folder containing n images taken at the same time from slightly parallax perspectives, receive aligned images.")

    parser.add_argument('alignedImageFolder', help="Absolute path to folder where input images are located", type=str)
    parser.add_argument("-o", "--overwrite",
                        help="Program will not overwrite previous data by default, unless this argument is present",
                        action="store_true")
    parser.add_argument("-r", "--roiSize",
                        help="Size of the ROI used for alignement. Default is 100. Change depending on the precision needed and alignmenet between images.",
                        type=int)
    parser.add_argument("-c", "--cropPercent",
                        help="With no cropping, aligned images will have black bars. Default for this param is 2 (for 2% crop), use larger if big offset between images.",
                        type=int)

    args = parser.parse_args()

    main(args.alignedImageFolder, args.overwrite, args.roiSize, args.cropPercent)

    return True

if __name__ == '__main__':
    argParser()

    # main(r'C:\Users\giles\Pictures\WIGGLEGRAMS\20200818_test\20200819_181111',True,None,6)
