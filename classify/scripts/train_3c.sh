python train.py \
    -train data/train_c3.json \
    -val data/val_c3.json \
    -e ViT_B_Vehicle_c3 \
    --task_name ViT_B_Vehicle_c3 \
    --num_class 3 \
    --class_names "truck" "freight_car" "bg" \
    --num_epoch 100 \
    --num_worker 16 \
    --batch_size 128 \
    --save_interval 4 \
    --lr 1e-4 \
    --evaluate_first \
    --amp
