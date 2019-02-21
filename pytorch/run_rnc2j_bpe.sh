#!/bin/bash

if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python train.py \
        --cuda \
        --data ../data/rnc2j-bpe/ \
        --dataset generic_dataset \
        --adaptive \
        --n_layer 8 \
        --d_model 1024 \
        --n_head 8 \
        --d_head 128 \
        --d_inner 4096 \
        --dropout 0.1 \
        --dropatt 0.0 \
        --optim adam \
        --lr 0.00025 \
        --warmup_step 8000 \
        --max_step 200000 \
        --tgt_len 128 \
        --mem_len 128 \
        --eval_tgt_len 128 \
        --batch_size 32 \
        --multi_gpu \
        --gpu0_bsz -1 \
        ${@:2}
else
    echo 'unknown argment 1'
fi
