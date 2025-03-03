
# python3 -m modules.training.train --training_type xfeat_default  --megadepth_root_path /home/wenhuanyao/Dataset/MegaDepth   \
#         --synthetic_root_path /home/wenhuanyao/Dataset/coco_20k --ckpt_save_path /home/wenhuanyao/accelerated_features/checkpoints \
#         --batch_size 16 --device_num 0 --save_ckpt_every 500



python3 -m modules.training.train --training_type xfeat_synthetic --batch_size 16  \
        --synthetic_root_path /home/wenhuanyao/Dataset/coco_20k --training_res (800, 608)  \
        --ckpt_save_path /home/wenhuanyao/accelerated_features/checkpoints  \
        --save_ckpt_every 5000 --device_num 0 --n_steps 160000 --lr 3e-4 --gamma_steplr 0.5