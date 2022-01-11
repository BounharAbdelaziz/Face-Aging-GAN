export CUDA_VISIBLE_DEVICES=2
export CUDA_LAUNCH_BLOCKING=1

python -W ignore train_cycleGAN.py --img_dir ../datasets/ffhq_mini/images/ \
                --batch_size 2 --n_epochs 500 --norm_type in2d --experiment_name FACE_cycleGAN_bs_2_lr_2e_4_bicubic_in2d_cosine_ffhq \
                --print_freq 25 --lr 0.0002 --lr_policy cosine --warmup_period 0 \
                --lambda_G 1 --lambda_D 1 --lambda_L1 0 --lambda_MSE 0 --lambda_PCP 0 --lambda_AGE 0 --lambda_ID 0 --lambda_cycle 10 --lambda_ID_cycle 10
                # --process_ffhq 