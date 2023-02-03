MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 \
--large_size 80 --small_size 64 --learn_sigma True --noise_schedule linear --num_channels 192 \
--num_heads 4  --num_res_blocks 2 --resblock_updown True \
--use_fp16 True --use_scale_shift_norm True"

SAMPLE_FLAGS="--batch_size 4 --num_samples 10 --timestep_respacing 250"

python3 super_res_sample.py $MODEL_FLAGS --model_path pretrained_models/64_256_upsampler.pt \
--base_samples 64_samples.npz $SAMPLE_FLAGS

 
# need to run base samples

# output to /tmp/openai-2023-01-12-22-44-14-222700/samples_10x64x64x3.npz