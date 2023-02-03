(set -x; SAMPLE_FLAGS="--batch_size 4 --num_samples 10 --timestep_respacing 250"

MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False \
--diffusion_steps 1000 --image_size 256 --learn_sigma True \
--noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 \
--resblock_updown True --use_fp16 True --use_scale_shift_norm True"

python3 /src/image_sample.py $MODEL_FLAGS \
--model_path /src/pretrained_models/256x256_diffusion_uncond.pt $SAMPLE_FLAGS
) > ./log/256x256_model_uncond_011823.log 2>&1
 