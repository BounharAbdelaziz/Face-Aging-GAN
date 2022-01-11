export CUDA_VISIBLE_DEVICES=2
export CUDA_LAUNCH_BLOCKING=1


python -W ignore train.py \
                --batch_size 10 --n_epochs 500 --norm_type bn2d --experiment_name FACE_AGE_GAN_bs_D_lr_2e_4_bicubic_bn2d_cosine_ffhq \
                --print_freq 10 --lr 0.0002 --lr_policy cosine --warmup_period 0 \
                --lambda_G 1 --lambda_D 1 --lambda_L1 0 --lambda_MSE 10 --lambda_PCP 0 --lambda_AGE 0 --lambda_ID 0 --lambda_cycle 0 --lambda_ID_cycle 0 \
                --img_size 256 --img_dir ../datasets/ffhq_mini/images/ 
                #--verbose