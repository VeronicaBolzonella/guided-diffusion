'''

import torch as th
mdl_path = './pretained_models/64x64_diffusion.pt'
model = th.load(mdl_path)
model.eval()
# the above code runs..
# MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --dropout 0.1 --image_size 64 --learn_sigma True --noise_schedule cosine --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True"

'''
# ran run_provided_examples.sh 

import numpy as np
output = np.load("/tmp/openai-2023-01-12-22-44-14-222700/samples_10x64x64x3.npz")
print(output.files)
arr_0 = output["arr_0"]
from PIL import Image as im
for i in range(len(arr_0)):
    data = im.fromarray(arr_0[i])
    print(data.size)
    data.save(f'/src/generated_images/sample_generated_pic_{i}.png')      
    print(f'/src/generated_images/sample_generated_pic_{i}.png')      
    