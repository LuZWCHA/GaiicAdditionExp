export CUDA_VISIBLE_DEVICES=0
export OPENBLAS_NUM_THREADS=1
    # --resume /root/workspace/data/classify/checkpoints/dual_stream_vit_ds_custom/truck_freightcar_noise/1/dual_stream_vit_ds_custom.pth.tar \

python lnl/main.py \
    --net dual_stream_vit \
    --dataset ds_custom2 \
    --epochs 50 \
    --batch-size 256 \
    --num-class 6 \
    --exp-name six_classes_noise \
    --ood-noise 0.0 \
    --id-noise 0.2 \
    --lr 1e-3 \
    --mixup \
    --warmup 5 \
    --proj-size 768 \
    --cont