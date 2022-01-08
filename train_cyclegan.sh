export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1

python -W ignore train_cycleGAN.py --batch_size 3 --n_epochs 500 --experiment_name FACE_cycleGAN_bs_3_lr_1e_5_bicubic --print_freq 10 --lr 1e-3 --lr_policy plateau --warmup_period 0 --lambda_G 1 --lambda_D 1 --lambda_L1 0 --lambda_MSE 10 --lambda_PCP 0.2 --lambda_AGE 0 --lambda_ID 0 --lambda_cycle 10 --lambda_ID_cycle 10