#!/bin/bash
set -e

# Path
LOCAL_DIR=../data/rnc2j
GSDATA=gs://kl-colab/transformer-xl/rnc2j-tfrecords/
GSEXP=gs://kl-colab/transformer-xl/rnc2j-exp-1/

# TPU setting
NUM_HOST=1
NUM_CORE=8 # TPUv2 -> 8 | TPUv3 -> 16

TEST_NUM_HOST=1
TEST_NUM_CORE=8 # TPUv2 -> 8 | TPUv3 -> 16

TRAIN_BSZ=256
VALID_BSZ=256

# Model
DIV_VAL=4
N_LAYER=12
D_MODEL=512
D_EMBED=512
N_HEAD=8
D_HEAD=64
D_INNER=4096

# Training
TGT_LEN=128
MEM_LEN=128

# Testing
TEST_TGT_LEN=128
TEST_MEM_LEN=128
TEST_CLAMP_LEN=-1
TEST_BSZ=8

if [[ $1 == 'train_data' ]]; then
    python data_utils.py \
        --data_dir=${LOCAL_DIR}/ \
        --dataset=generic_dataset \
        --tgt_len=${TGT_LEN} \
        --per_host_train_bsz=${TRAIN_BSZ} \
        --per_host_valid_bsz=${VALID_BSZ} \
        --num_core_per_host=${NUM_CORE} \
        --num_passes=2 \
        --use_tpu=True \
        ${@:2}

    SRC_PATTERN=train.bsz-${TRAIN_BSZ}.tlen-${TGT_LEN}.core-${NUM_CORE}*
    gsutil cp ${LOCAL_DIR}/tfrecords/${SRC_PATTERN} ${GSDATA}

    SRC_PATTERN=valid.bsz-${VALID_BSZ}.tlen-${TGT_LEN}.core-${NUM_CORE}*
    gsutil cp ${LOCAL_DIR}/tfrecords/${SRC_PATTERN} ${GSDATA}

elif [[ $1 == 'test_data' ]]; then
    python data_utils.py \
        --data_dir=${LOCAL_DIR}/ \
        --dataset=generic_dataset \
        --tgt_len=${TEST_TGT_LEN} \
        --per_host_test_bsz=${TEST_BSZ} \
        --num_core_per_host=${TEST_NUM_CORE} \
        --num_passes=1 \
        --use_tpu=True \
        ${@:2}

    SRC_PATTERN=test.bsz-${TEST_BSZ}.tlen-${TEST_TGT_LEN}.core-${TEST_NUM_CORE}*
    gsutil cp ${LOCAL_DIR}/tfrecords/${SRC_PATTERN} ${GSDATA}

elif [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python train.py \
        --data_dir=${GSDATA} \
        --record_info_dir=${LOCAL_DIR}/tfrecords/ \
        --corpus_info_path=${LOCAL_DIR}/corpus-info.json \
        --model_dir=${GSEXP} \
        --div_val=${DIV_VAL} \
        --untie_r=True \
        --proj_share_all_but_first=True \
        --proj_same_dim=True \
        --n_layer=${N_LAYER} \
        --d_model=${D_MODEL} \
        --d_embed=${D_EMBED} \
        --n_head=${N_HEAD} \
        --d_head=${D_HEAD} \
        --d_inner=${D_INNER} \
        --dropout=0.2 \
        --dropatt=0.2 \
        --init_std=0.005 \
        --learning_rate=0.00025 \
        --warmup_steps=8000 \
        --train_steps=4000000 \
        --tgt_len=${TGT_LEN} \
        --mem_len=${MEM_LEN} \
        --train_batch_size=${TRAIN_BSZ} \
        --num_hosts=${NUM_HOST} \
        --num_core_per_host=${NUM_CORE} \
        --iterations=1000 \
        --save_steps=1000 \
        --use_tpu=True \
        --do_eval=False \
        ${@:2}

elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
    python train.py \
        --data_dir=${GSDATA} \
        --record_info_dir=${LOCAL_DIR}/tfrecords/ \
        --corpus_info_path=${LOCAL_DIR}/corpus-info.json \
        --model_dir=${GSEXP} \
        --div_val=${DIV_VAL} \
        --untie_r=True \
        --proj_share_all_but_first=True \
        --proj_same_dim=True \
        --n_layer=${N_LAYER} \
        --d_model=${D_MODEL} \
        --d_embed=${D_EMBED} \
        --n_head=${N_HEAD} \
        --d_head=${D_HEAD} \
        --d_inner=${D_INNER} \
        --tgt_len=${TEST_TGT_LEN} \
        --mem_len=${TEST_MEM_LEN} \
        --clamp_len=${TEST_CLAMP_LEN} \
        --same_length=True \
        --eval_batch_size=${TEST_BSZ} \
        --num_host=${TEST_NUM_HOST} \
        --num_core_per_host=${TEST_NUM_CORE} \
        --use_tpu=True \
        --do_train=False \
        --do_eval_only=True \
        --eval_split=test \
        ${@:2}

else
    echo 'unknown argment 1'
fi
