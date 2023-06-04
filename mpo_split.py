#!/usr/bin/env python

from __future__ import print_function
from optparse import OptionParser
from PIL import Image
from pathlib import Path

try:
    import StringIO
except ImportError:
    from io import BytesIO
import errno, glob, os, sys


class MPOError(Exception):
    """Error class to distinguish improper file errors from possible IOErrors"""

    def __init__(self, value): self.value = value

    def __str__(self): return repr(self.value)


def split_mpo(filename):
    """Reads a given MPO file and finds the break between the two JPEG images."""

    with open(filename, 'rb') as f:
        data = f.read()

        # Look for the hex string 0xFFD9FFD8FFE1:
        #   0xFFD9 represents the end of the first JPEG image
        #   0xFFD8FFE1 marks the start of the appended JPEG image
        idx = data.find(b'\xFF\xD8\xFF\xE1', 1)

        if idx > 0:
            return Image.open(BytesIO(data[: idx])), Image.open(BytesIO(data[idx:]))
        else:
            raise MPOError(filename)


def main(args, options=False):
    # Process the given MPO files
    for i, f in enumerate(args):
        try:
            # Load the right and left images (ordered for crosseye stereo)
            img_left, img_right  = split_mpo(f)

            # Create the stereo image
            size = (2 * img_right.size[0], img_right.size[1])
            img_stereo = Image.new('RGB', size)

            # if options.parallel:
            #     img_stereo.paste(img_right, (0, 0))
            #     img_stereo.paste(img_left, (img_right.size[0], 0))
            # else:
            #     img_stereo.paste(img_left, (0, 0))
            #     img_stereo.paste(img_right, (img_right.size[0], 0))

            # Save the stereo image
            stereo_type = 'parallel'  # if options.parallel else 'crosseye'
            filename = f[:-4] + '_' + stereo_type + '.jpg'

            print('Writing ' + filename + ' (%d/%d)' % (i + 1, len(args)))

            # modify to save individual images instead of combined image
            # img_stereo.save(filename) # this is the stereo image in single jpeg format

            # make newdir to store two jpegs that form stereo image
            path_mpo = Path(f)
            nm_new_dir = path_mpo.stem
            dir_new = path_mpo.parent / nm_new_dir
            if not dir_new.exists():
                dir_new.mkdir()

            filename_left = dir_new / f'{nm_new_dir}_left.jpg'
            filename_right = dir_new / f'{nm_new_dir}_right.jpg'

            img_right.save(filename_right)
            img_left.save(filename_left)

        except MPOError:
            print(filename + ' is not a valid MPO file')
        except IOError as e:
            print(filename + ':')
            print('errno:', e.errno)
            print('err code:', errno.errorcode[e.errno])
            print('err message:', os.strerror(e.errno))

    return dir_new, filename_left, filename_right

if __name__ == '__main__':
    # Parse arguments
    parser = OptionParser('usage: %prog [options] mpofiles(s)')
    parser.add_option("-p", '--parallel', action='store_true', dest='parallel',
                      default=False, help='produce parallel rather than crosseye stereos.')
    (options, args) = parser.parse_args()

    if len(args) == 0:
        parser.error('invalid argument - requires at least one MPO file to read')
    elif len(args) == 1 and '*' in args[0]:
        args = glob.glob(args[0])

    main(args)

