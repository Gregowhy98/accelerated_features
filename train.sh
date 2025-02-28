
# python3 -m modules.training.train --training_type xfeat_default  --megadepth_root_path /home/wenhuanyao/Dataset/MegaDepth   \
#         --synthetic_root_path /home/wenhuanyao/Dataset/coco_20k --ckpt_save_path /home/wenhuanyao/accelerated_features/checkpoints \
#         --batch_size 16 --device_num 0 --save_ckpt_every 500



python3 -m modules.training.train --training_type xfeat_synthetic --synthetic_root_path /home/wenhuanyao/Dataset/coco_20k --ckpt_save_path /home/wenhuanyao/accelerated_features/checkpoints