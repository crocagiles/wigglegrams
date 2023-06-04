
# find and select neutral / white patch in photoshop.
# Use histogram tool to find RGB values for the patch in both images
# calculate gain for R and B channels to correct color disparity (may not work in all light sources)

# R G B values for neutral patch in each image
left_img = [187.82, 194.88, 199.58]
right_img = [199.13, 202.9, 210.72]

# alter right image to match left

# Step 1 - apply gain to make exposures match based on green channel
gain_right_module =  left_img[1] / right_img[1]
right_img_gain = [channel * gain_right_module for channel in right_img]
print(f'Gain value for right module green channel: {gain_right_module}')

#step 2 - multiply r and b values to make wb match
wb = [left_img[i] / right_img_gain[i] for i in range(3)]
print(f'wb scales: {wb}')
