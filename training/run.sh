gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Num of GPUs: $gpu_count"
echo "Running Task: $task"
torchrun --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --nnodes=$WORLD_SIZE --node_rank=$RANK --nproc_per_node ${gpu_count} \
        train_on_coco_img_iter.py \
        --config ./configs/caption_coco_config.py \
        --output_dir ./outputs \
        --world_size ${gpu_count} \
        --dist_url env:// \
        --distributed True 
