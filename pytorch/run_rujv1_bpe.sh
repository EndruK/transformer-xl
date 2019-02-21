#!/bin/bash

if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python train.py \
        --cuda \
        --data ../data/rujv1_bpe/ \
        --dataset generic_dataset \
        --adaptive \
        --n_layer 8 \
        --d_model 512 \
        --n_head 8 \
        --d_head 64 \
        --d_inner 2048 \
        --dropout 0.1 \
        --dropatt 0.0 \
        --optim adam \
        --lr 0.00025 \
        --warmup_step 0 \
        --max_step 200000 \
        --tgt_len 256 \
        --mem_len 256 \
        --eval_tgt_len 128 \
        --batch_size 32 \
        --multi_gpu \
        --gpu0_bsz -1 \
        ${@:2}
else
    echo 'unknown argment 1'
fi
