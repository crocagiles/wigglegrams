# turn 10s mp4 from runway into looping gif

from moviepy.editor import *
from pathlib import Path
import random
import datetime

def runway_to_vid_and_gif(pth_mp4_input):

    video = VideoFileClip(str(pth_mp4_input))
    clip = video.subclip(0, 5)  # 10s clip from runway has 5 second static frame at end of video. trim it off/

    speedup = clip.fx(vfx.speedx, 7) # speed up the first 5 seconds since it's pretty slow
    reversed = speedup.fx(vfx.time_symmetrize) # add reversed video to end to make smooth loop

    # Export
    pth_mp4_input_P = Path(pth_mp4_input)

    nm_out_mp4 = pth_mp4_input_P.stem + '_mvpy_out.mp4'
    nm_out_gif = pth_mp4_input_P.stem + '_mvpy_out.gif'

    pth_mp4_input_P = Path(pth_mp4_input)
    path_out_mp4, path_out_gif = pth_mp4_input_P.parent / nm_out_mp4, pth_mp4_input_P.parent / nm_out_gif

    reversed.write_videofile(str(path_out_mp4))
    reversed.write_gif(str(path_out_gif), program='imageio', opt='nq', fuzz=1, )

    return True

def compile_wiggles(dir_with_wiggles):

    # compile several mp4 wiggles (from the other function in this file) into single video
    pths_wiggle = dir_with_wiggles.rglob('*_mvpy_out.mp4') # wiggles should all have this pattern
    path_wiggle = [str(p) for p in pths_wiggle] # since moviepy doesn't like path objects..

    # each wiggle mp4 is only a single loop. Let's duplicate a few to make the end result look more dynamic
    final_vid_list = duplicate_half(path_wiggle)
    final_clips = concatenate_videoclips([VideoFileClip(f) for f in final_vid_list])

    # Generate a datetime stamp
    now = datetime.datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Construct the filename with the datetime stamp and output
    name_out = "wigglegram_compilation_" + dt_string + ".mp4"
    path_out = Path(dir_with_wiggles) / name_out
    final_clips.write_videofile(str(path_out))

    return

def duplicate_half(lst):
    # Determine the number of items to duplicate
    num_to_duplicate = len(lst) // 2

    # Get a list of random indices to duplicate
    indices_to_duplicate = random.sample(range(len(lst)), num_to_duplicate)

    # Create a new list with duplicates
    new_lst = []
    for i, item in enumerate(lst):
        new_lst.append(item)
        if i in indices_to_duplicate:
            new_lst.append(item)

    return new_lst



    return True

def batch_runway_to_vidgif(dir_root):

    # Find all runway mp4 files. Filter out mp4s that have been created locally by other wigglegram funcrtions..
    runway_mp4s = []
    all_mp4s = list(Path(dir_root).rglob('*.mp4'))
    for filename in all_mp4s:
        if '_mvpy_out' in str(filename) or 'wigglegram_compilation' in str(filename):
            print(f'filtered out {filename}')
            continue
        filepath = os.path.join(dir_root, filename)
        filesize = os.path.getsize(filepath) // 1000  # convert to KB
        if filesize >= 3000 and filesize <= 5000:
            runway_mp4s.append(filename)

    # call runway_to_vid_and_gif on the videos we found
    for r in runway_mp4s:
        print(f'Running: {r.name}')
        runway_to_vid_and_gif(r)

    return True




if __name__ == '__main__':

    # input = Path(r"C:\Users\giles\Downloads\wiggletest\DSCF1120\DSCF1120.mp4")
    # runway_to_vid_and_gif(input)

    input = Path(r"C:\Users\giles\Downloads\wiggletest")
    batch_runway_to_vidgif(input) # look for all runway mp4 files and run em all
    compile_wiggles(input)
