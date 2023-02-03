MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --dropout 0.1 --image_size 64 --learn_sigma True --noise_schedule cosine --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True"
SAMPLE_FLAGS="--batch_size 4 --num_samples 10 --timestep_respacing 250"
python3 image_sample.py $MODEL_FLAGS --model_path pretrained_models/64x64_diffusion.pt $SAMPLE_FLAGS

# output to /tmp/openai-2023-01-12-22-44-14-222700/samples_10x64x64x3.npz