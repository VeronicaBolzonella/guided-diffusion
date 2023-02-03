import sys, os
from PIL import Image as im
import numpy as np

npz_file_path = sys.argv[1]
output_handle = sys.argv[2]
# npz_file_path = '/src/bash_scripts/installation_test/results/samples_10x128x128x3.npz'
# output_handle =  '/src/test'
try:
    verbose = sys.argv[3]
except:
    verbose = False

output_dir = os.path.dirname(output_handle)
npz_file = np.load(npz_file_path)
print(f"Loading {npz_file_path}", file = sys.stderr)

show_res = True
img_idx = 0
for now_key in npz_file.files:
    arrays = npz_file[now_key]
    
    for i in range(len(arrays)):
        img_idx += 1
        try:
            data = im.fromarray(arrays[i])
        except TypeError:
            print(f'Current array: {now_key} has the size of {arrays.shape}. No output will be generated', file = sys.stderr)
            sys.exit()

        if show_res:
            print(f"Generating {data.size} images", file = sys.stderr)
            print(f"Saving imgs to {output_dir}", file = sys.stderr)
            show_res = False
        data.save(f'{output_handle}_{img_idx}.png')
        if verbose:
            print(f"Saving img {output_handle}_{img_idx}.png", file = sys.stderr)
             
  
