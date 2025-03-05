


python3 -m modules.training.train --training_type xfeat_synthetic --batch_size 16  \
        --synthetic_root_path /home/wenhuanyao/Dataset/coco_20k  \
        --ckpt_save_path /home/wenhuanyao/accelerated_features/checkpoints/synthetic  \
        --save_ckpt_every 5000 --device_num 1 --n_steps 160000 --lr 3e-4 --gamma_steplr 0.5


