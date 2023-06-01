import tensorflow as tf
import tensorflow_hub as hub
`from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.widgets as widgets

import time
import requests
import numpy as np
import moviepy.editor as mpy
import mpo_split
import cv2

from typing import Generator, Iterable, List, Optional

# Todo add environment details anaconda, tensorflow etc to readnme

#https://www.tensorflow.org/hub/tutorials/tf_hub_film_example?utm_source=pocket_saves

_UINT8_MAX_F = float(np.iinfo(np.uint8).max)
# model = hub.load("https://tfhub.dev/google/film/1")

def load_image(img_url: str):
  """Returns an image with shape [height, width, num_channels], with pixels in [0..1] range, and type np.float32."""

  if (img_url.startswith("https")):
    user_agent = {'User-agent': 'Colab Sample (https://tensorflow.org)'}
    response = requests.get(img_url, headers=user_agent)
    image_data = response.content
  else:
    image_data = tf.io.read_file(img_url)

  image = tf.io.decode_image(image_data, channels=3)
  image_numpy = tf.cast(image, dtype=tf.float32).numpy()
  return image_numpy / _UINT8_MAX_F


"""A wrapper class for running a frame interpolation based on the FILM model on TFHub

Usage:
  interpolator = Interpolator()
  result_batch = interpolator(image_batch_0, image_batch_1, batch_dt)
  Where image_batch_1 and image_batch_2 are numpy tensors with TF standard
  (B,H,W,C) layout, batch_dt is the sub-frame time in range [0..1], (B,) layout.
"""


def _pad_to_align(x, align):
  """Pads image batch x so width and height divide by align.

  Args:
    x: Image batch to align.
    align: Number to align to.

  Returns:
    1) An image padded so width % align == 0 and height % align == 0.
    2) A bounding box that can be fed readily to tf.image.crop_to_bounding_box
      to undo the padding.
  """
  # Input checking.
  assert np.ndim(x) == 4
  assert align > 0, 'align must be a positive number.'

  height, width = x.shape[-3:-1]
  height_to_pad = (align - height % align) if height % align != 0 else 0
  width_to_pad = (align - width % align) if width % align != 0 else 0

  bbox_to_pad = {
      'offset_height': height_to_pad // 2,
      'offset_width': width_to_pad // 2,
      'target_height': height + height_to_pad,
      'target_width': width + width_to_pad
  }
  padded_x = tf.image.pad_to_bounding_box(x, **bbox_to_pad)
  bbox_to_crop = {
      'offset_height': height_to_pad // 2,
      'offset_width': width_to_pad // 2,
      'target_height': height,
      'target_width': width
  }
  return padded_x, bbox_to_crop


class Interpolator:
  """A class for generating interpolated frames between two input frames.

  Uses the Film model from TFHub
  """

  def __init__(self, align: int = 64) -> None:
    """Loads a saved model.

    Args:
      align: 'If >1, pad the input size so it divides with this before
        inference.'
    """
    self._model = hub.load("https://tfhub.dev/google/film/1")
    self._align = align

  def __call__(self, x0: np.ndarray, x1: np.ndarray,
               dt: np.ndarray) -> np.ndarray:
    """Generates an interpolated frame between given two batches of frames.

    All inputs should be np.float32 datatype.

    Args:
      x0: First image batch. Dimensions: (batch_size, height, width, channels)
      x1: Second image batch. Dimensions: (batch_size, height, width, channels)
      dt: Sub-frame time. Range [0,1]. Dimensions: (batch_size,)

    Returns:
      The result with dimensions (batch_size, height, width, channels).
    """
    if self._align is not None:
      x0, bbox_to_crop = _pad_to_align(x0, self._align)
      x1, _ = _pad_to_align(x1, self._align)

    inputs = {'x0': x0, 'x1': x1, 'time': dt[..., np.newaxis]}
    result = self._model(inputs, training=False)
    image = result['image']

    if self._align is not None:
      image = tf.image.crop_to_bounding_box(image, **bbox_to_crop)
    return image.numpy()


def _recursive_generator(
    frame1: np.ndarray, frame2: np.ndarray, num_recursions: int,
    interpolator: Interpolator) -> Generator[np.ndarray, None, None]:
  """Splits halfway to repeatedly generate more frames.

  Args:
    frame1: Input image 1.
    frame2: Input image 2.
    num_recursions: How many times to interpolate the consecutive image pairs.
    interpolator: The frame interpolator instance.

  Yields:
    The interpolated frames, including the first frame (frame1), but excluding
    the final frame2.
  """
  if num_recursions == 0:
    yield frame1
  else:
    # Adds the batch dimension to all inputs before calling the interpolator,
    # and remove it afterwards.
    time = np.full(shape=(1,), fill_value=0.5, dtype=np.float32)
    mid_frame = interpolator(
        np.expand_dims(frame1, axis=0), np.expand_dims(frame2, axis=0), time)[0]
    yield from _recursive_generator(frame1, mid_frame, num_recursions - 1,
                                    interpolator)
    yield from _recursive_generator(mid_frame, frame2, num_recursions - 1,
                                    interpolator)


def interpolate_recursively(
    frames: List[np.ndarray], num_recursions: int,
    interpolator: Interpolator) -> Iterable[np.ndarray]:
  """Generates interpolated frames by repeatedly interpolating the midpoint.

  Args:
    frames: List of input frames. Expected shape (H, W, 3). The colors should be
      in the range[0, 1] and in gamma space.
    num_recursions: Number of times to do recursive midpoint
      interpolation.
    interpolator: The frame interpolation model to use.

  Yields:
    The interpolated frames (including the inputs).
  """
  n = len(frames)
  for i in range(1, n):
    # yield from _recursive_generator(frames[i - 1], frames[i], times_to_interpolate, interpolator)
    yield from _recursive_generator(frames[i - 1], frames[i], 6, interpolator)
  # Separately yield the final frame.
  yield frames[-1]

def create_video_from_images(image_list, output_file, fps=30):

    # Convert list of NumPy arrays to a list of moviepy VideoClips
    clips = [mpy.ImageClip(img*255, duration=1/60) for img in image_list]

    # Create a moviepy concatenate_videoclips object
    fps = 60
    video = mpy.concatenate_videoclips(clips)
    video.set_fps(fps)

    # Short by one frame, so get rid on the last frame:
    final_clip = video.subclip(t_end=(video.duration - 1.0 / fps))
    f = final_clip.fx(mpy.vfx.time_symmetrize)
    speedup = f.fx(mpy.vfx.speedx, 2.3)
    #
    # # Set the output video file's FPS (frames per second)
    # video2 = video.set_fps(fps)

    # Write the video to the output file
    speedup.write_videofile(output_file, codec="libx264", fps=fps)

# Example usage:
# Assume `image_list` is a list of NumPy arrays representing images
# and `output_file` is the desired output file name/path

def clip_frames(frames):
    clipped_frames = []
    for frame in frames:
        clipped_frame = np.clip(frame, 0, 1)
        clipped_frames.append(clipped_frame)
    return clipped_frames

def mpo_2_vid_gif(mpo_file):

    # img1, im2

    #split mpo (makes new dir with two jpeg images, in directory where mpo is located
    dir_new, filename_left, filename_right = mpo_split.main([mpo_file])
    output_file = dir_new.parent / (dir_new.name + '_wiggle.mp4')
    # if output_file.exists():
    #     print(f'Skipping {output_file.name}')
    #     return

    # load and downsample, resolution is too high otherwise and memory runs out
    image1 = load_image(str(filename_left))[::4, ::4, :]
    image2 = load_image(str(filename_right))[::4, ::4, :]


    #align images
    image1_ref, image2_warped = img_align(image1, image2)

    # image2_warped will have at least one black edge, from alignment. Let's trim it down.
    smart_crop(image2_warped)

    # image1_ref, image2_warped = image1_ref/255, image2_warped/255

    # image1_ref, image2_warped = image1, image2
    # generation of interpolated frames

    times_to_interpolate = 6
    interpolator = Interpolator()
    input_frames = [image1_ref, image2_warped]
    frames = list(interpolate_recursively(input_frames, times_to_interpolate,interpolator))

    # interpolated frames have range outside of 0-1. Normalizing gives weird results, we must clip instead.
    # I discovered this by looking at the histogram and noticing that the bulk of normalized pixel values are 0-1
    clipped = clip_frames(frames)


    create_video_from_images(clipped, str(output_file), fps=30)
    print(f'video with {len(frames)} frames')
    # media.show_video(frames, fps=30, title='FILM interpolated video')

    return

def smart_crop(img):

    return

def img_align(img_left, image_right):

    # Load the first image (grayscale)
    # img_left =  r"C:\Users\giles\Pictures\mpo_backup\DSCF1356\DSCF1356_left.jpg"
    # image_right= r"C:\Users\giles\Pictures\mpo_backup\DSCF1356\DSCF1356_right.jpg"


    quart_y, quart_x, _ = [i // 4 for i in img_left.shape]
    quart_y, quart_x, _ = [1,1,1]

    img_left_g = np.uint8((img_left[quart_y:-quart_y,quart_x:-quart_x,1]) * 255)
    image_right_g =  np.uint8((image_right[quart_y:-quart_y,quart_x:-quart_x,1]) * 255) # go from normalized to 8 bit


    while True:
        print('\nPlease select a point on the image and press enter to confirm.')
        cv2.namedWindow("Select ROI by clicking and dragging, then press enter", cv2.WINDOW_NORMAL)
        coords = cv2.selectROI("Select ROI by clicking and dragging, then press enter", img_left_g)
        cv2.destroyWindow("Select ROI by clicking and dragging, then press enter")
        x = coords[0]
        y = coords[1]
        width = coords[2]
        height = coords[3]
        img_left_g_crop = img_left_g[y:y + height, x:x + width]
        image_right_g_g_crop = image_right_g[y:y + height, x:x + width]


        sift = cv2.SIFT_create()

        # Find keypoints and descriptors for the images
        keypoints1, descriptors1 = sift.detectAndCompute(img_left_g_crop, None)
        keypoints2, descriptors2 = sift.detectAndCompute(image_right_g_g_crop, None)

        # Create a BFMatcher object
        bf = cv2.BFMatcher()

        # Match the descriptors
        matches = bf.match(descriptors1, descriptors2)

        # Sort the matches by distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Select the top matches (you can adjust the number according to your needs)
        num_matches = 10
        selected_matches = matches[:num_matches]

        # Extract the corresponding keypoints from the matches
        src_pts  = np.float32([keypoints1[m.queryIdx].pt for m in selected_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in selected_matches]).reshape(-1, 1, 2)
        # Estimate the transformation matrix
        transformation_matrix, _ = cv2.estimateAffinePartial2D(dst_pts, src_pts)# cv2.RANSAC)

        # Remove scaling component from the transformation matrix
        # transformation_matrix = transformation_matrix[:2, :]
        transformation_matrix[0, 0] = 1
        transformation_matrix[1, 1] = 1

        # Warp the second image using the estimated transformation matrix
        aligned_image = cv2.warpAffine(image_right, transformation_matrix, (img_left.shape[1], img_left.shape[0]))


        # debug warped image
        list_for_animate = [img_left[:,:,1], image_right[:,:,1]]
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        # Define your two images
        image1 = img_left # Replace with your image data or file path
        image2 =  aligned_image# Replace with your image data or file path

        # Create the figure and axis
        fig, ax = plt.subplots()

        # Initialize the image plot
        im = ax.imshow(image1)
        # Define the update function for the animation
        def update(frame):
            # Alternate between the two images
            if frame % 2 == 0:
                im.set_array(image1)
            else:
                im.set_array(image2)

            return im,
        # Create the animation
        ani = animation.FuncAnimation(fig, update, frames=10, interval=500, blit=True)

        # Show the figure
        plt.show()
        time.sleep(3)
        plt.close()
        inp = input("Accept Alginment? (Y/N)")
        if inp.lower() == 'y' or inp.lower == 'yes':
            break
        else:
            continue

    #crop off black lines


    return img_left, aligned_image


if __name__ == '__main__':


    mpo_2_vid_gif(r"C:\Users\giles\Pictures\20230528_Italy_wiggles\card2\DSCF1282.MPO")

    # p = Path(r"C:\Users\giles\Pictures\mpo_backup").rglob('*.MPO')
    # for m in p:
    #     mpo_2_vid_gif(str(m))