import tensorflow as tf
import tensorflow_hub as hub

import requests
import numpy as np
import moviepy.editor as mpy
import mpo_split

from typing import Generator, Iterable, List, Optional

# Todo add environment details anaconda, tensorflow etc to readnme


_UINT8_MAX_F = float(np.iinfo(np.uint8).max)
model = hub.load("https://tfhub.dev/google/film/1")

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
    speedup = f.fx(mpy.vfx.speedx, 2)
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

    # load and downsample, resolution is too high otherwise and memory runs out
    image1 = load_image(str(filename_left))[::4, ::4, :]
    image2 = load_image(str(filename_right))[::4, ::4, :]

    # generation of interpolated frames

    times_to_interpolate = 6
    interpolator = Interpolator()
    input_frames = [image1, image2]
    frames = list(interpolate_recursively(input_frames, times_to_interpolate,interpolator))

    # interpolated frames have range outside of 0-1. Normalizing gives weird results, we must clip instead.
    # I discovered this by looking at the histogram and noticing that the bulk of normalized pixel values are 0-1
    clipped = clip_frames(frames)

    output_file = dir_new.parent / (dir_new.name + '_wiggle.mp4')
    create_video_from_images(clipped, str(output_file), fps=30)
    print(f'video with {len(frames)} frames')
    # media.show_video(frames, fps=30, title='FILM interpolated video')

    return



if __name__ == '__main__':
    mpo_2_vid_gif(r"C:\Users\giles\Pictures\20230528_Italy_wiggles\DSCF1341.MPO")