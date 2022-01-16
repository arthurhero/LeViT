OUT_DIR=./checkpoints/
CKPT_NAME=cifar_simp_aug_32.ckpt

CUDA_VISIBLE_DEVICES=6,7 WORLD_SIZE=2 python -m torch.distributed.launch --nproc_per_node=8 \
    --master_port 47771 --use_env main.py --model LeViT_128S_cifar\
    --data-path ../datasets/cifar10 --data-set CIFAR --input-size 32 \
    --output_dir $OUT_DIR --distillation-type none \
    --checkpoint_name $CKPT_NAME \
    --resume $OUT_DIR$CKPT_NAME \ 
    --eval
