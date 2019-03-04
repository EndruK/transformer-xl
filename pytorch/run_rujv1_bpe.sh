#!/bin/bash

set -e

if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python train.py \
        --cuda \
        --data ../data/rujv1_bpe/ \
        --dataset generic_dataset \
        --n_layer 16 \
        --d_model 768 \
        --n_head 12 \
        --d_head 64 \
        --d_inner 2048 \
        --dropout 0.1 \
        --dropatt 0.0 \
        --optim adam \
        --lr 0.00025 \
        --warmup_step 2000 \
        --max_step 100000 \
        --tgt_len 128 \
        --mem_len 128 \
        --eval_tgt_len 128 \
        --batch_size 256 \
        --batch_chunk 8 \
        --eval-interval 1000 \
        --log-interval 100 \
        --multi_gpu \
        --gpu0_bsz 112 \
        ${@:2}
else
    echo 'unknown argument 1'
fi
