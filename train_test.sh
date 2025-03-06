
python3 -m train_line --training_type xfeat_megadepth  --megadepth_root_path /home/wenhuanyao/Dataset/MegaDepth   \
        --ckpt_save_path /home/wenhuanyao/accelerated_features/checkpoints/megadepth \
        --batch_size 16 --device_num 2 --save_ckpt_every 2000