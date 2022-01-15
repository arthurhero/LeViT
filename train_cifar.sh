python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model LeViT_128S_cifar\
    --data-path ../datasets/cifar10 --data-set CIFAR --input-size 32 \
    --output_dir ./checkpoints/ --distillation-type none
