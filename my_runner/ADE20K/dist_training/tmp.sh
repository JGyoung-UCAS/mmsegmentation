#!/usr/bin/env bash

echo "Starting the first sh..."
nohup my_runner/ADE20K/dist_training/gcnet_r50-d8_4xb4-80k_ade20k-512x512_original.sh >> my_runner/ADE20K/dist_training/gcnet_r50-d8_4xb4-80k_ade20k-512x512_original.out &

echo "Starting the second sh..."
nohup my_runner/ADE20K/dist_training/gcnet_r50-d8_4xb4-80k_ade20k-512x512_suppressor_e5_all_frozen.sh >> my_runner/ADE20K/dist_training/gcnet_r50-d8_4xb4-80k_ade20k-512x512_suppressor_e5_all_frozen.out &

echo "Starting the third sh..."
nohup my_runner/ADE20K/dist_training/psanet_r50-d8_4xb4-80k_ade20k-512x512_original.sh >> my_runner/ADE20K/dist_training/psanet_r50-d8_4xb4-80k_ade20k-512x512_original.out &

echo "Starting the fourth sh..."
nohup my_runner/ADE20K/dist_training/psanet_r50-d8_4xb4-80k_ade20k-512x512_suppressor_e5_all_frozen.sh >> my_runner/ADE20K/dist_training/psanet_r50-d8_4xb4-80k_ade20k-512x512_suppressor_e5_all_frozen.out &

echo "Starting the fiveth sh..."
nohup my_runner/ADE20K/dist_training/pspnet_r50-d8_4xb4-80k_ade20k-512x512_original.sh >> my_runner/ADE20K/dist_training/pspnet_r50-d8_4xb4-80k_ade20k-512x512_original.out &

echo "Starting the sixth sh..."
nohup my_runner/ADE20K/dist_training/pspnet_r50-d8_4xb4-80k_ade20k-512x512_suppressor_e5_all_frozen.sh >> my_runner/ADE20K/dist_training/pspnet_r50-d8_4xb4-80k_ade20k-512x512_suppressor_e5_all_frozen.out &

echo "Completed."