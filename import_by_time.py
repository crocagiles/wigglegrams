import time
from pathlib import Path
import exifread
from collections import OrderedDict
from shutil import copy

def get_date(jpg_path):
    with open(jpg_path, 'rb') as fh:
        tags = exifread.process_file(fh, stop_tag="EXIF DateTimeOriginal")
        dateTaken = tags["EXIF DateTimeOriginal"]
        return dateTaken

outputDir = Path('C:/Users/giles/Pictures/WIGGLEGRAMS/20200818_test')

if not outputDir.exists():
    outputDir.mkdir()

user_set = {
    'cam1': 'E:/',
    'cam2': 'G:/',
    'cam3': 'F:/'
}
num_sds = len(user_set.keys())

master_d = {}
for sd, item in user_set.items():
    master_d[sd] = {}
    master_d[sd]['drive'] = Path(item)
    master_d[sd]['jpg_dir'] = Path(item) / r'\DCIM\100GOPRO'
    jpgs = master_d[sd]['jpg_dir'].rglob('*.JPG')
    master_d[sd]['jpgs'] = {}
    for j in jpgs:
        master_d[sd]['jpgs'][j] = str(get_date(j))


all_time_stamps = []

for key in master_d.keys():
    for val in master_d[key]['jpgs'].values():
        all_time_stamps.append(str(val))

unique_stamps = list(OrderedDict.fromkeys(all_time_stamps))

stamps_w_full_set = []
for s in unique_stamps:
    count = all_time_stamps.count(s)
    if not count == num_sds:
        continue
    stamps_w_full_set.append(s)

sets = []
for stamp in stamps_w_full_set:
    set = []
    for sd, val in master_d.items():
        for j, s in val['jpgs'].items():
            if s == stamp:
                set.append(j)
        sets.append(set)

for s in sets:
    nm_stamp = str(get_date(s[0])).replace(':','').replace(' ','_')
    new_dir = outputDir / nm_stamp
    if not new_dir.exists():
        new_dir.mkdir()
    for i, j in enumerate(s):
        new_nm = 'cam_' + str(i+1) + '_' + nm_stamp + '.JPG'
        full_out = new_dir / new_nm
        if not full_out.exists():
            copy(j, full_out)



print('done')