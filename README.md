# wigglegrams
All code related to creating wigglegrams


Your environment should be python 3.X, and you'll need the 3rd party modules listed in requirements.txt. (Run "pip install -r requirements.txt" in your env).

The function is called on a directory containing each of input jpg images. I've tested with 2, 3, and 4 jpegs in my wigglegrams and it will run any number. It's pretty slow though, so I wouldn't use it for more than 4.

The main functions for generating a wigglegram are in wiggleBuilder.py. You can run it directly from command line.

usage: wiggleBuilder.py [-h] [-o] [-r ROISIZE] [-c CROPPERCENT] alignedImageFolder

[-r ROISIZE] is the size of the square section in the center of the image that will be used for the alignent. Because of the parallax effect, you can't align one entire image on another. I would usually use 50-200, depending on the size of the image.

[-c CROPPERCENT] accounts for the parts of the aligned images that will be trimmed off. Default is 3 but use more if you're seeing black edges in your wigglesgrams.

alignedImageFolder is just the path to the folder with the input images.

EXAMPLE

python wiggleBuilder.py C:\Users\giles\Pictures\WIGGLEGRAMS\subfolder_w_3_imgs

When the dialog box with the image pops up, you can click and drag an ROI on the image that will be used for alignment. For example, select a square around the face and body in a portrait that you captured. 

you will see output data in the following directory. C:\Users\giles\Pictures\WIGGLEGRAMS\subfolder_w_3_imgs\Aligned

Note that this code is a little slow and not very robust. Sometimes if you see an error after selecting the ROI, go back and try again and make a slightly different selection. It will struggle if there's a lot of fine detail or high interocular distance between the lenses.