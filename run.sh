#!/usr/bin/env bash
export PATH=/usr/local/cuda-9.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH
#set -ex
DATA_DIR='/data/disk1/private/wangmuzi/data/THUMT'
CODE='~/THUMT'

DEVICES=7
MODEL=${DATA_DIR}/checkpoints/lang8_transformer/model.ckpt-168000
VSRC=${DATA_DIR}/data/lang8_seg/src.vocab.32k.txt
VTRG=${DATA_DIR}/data/lang8_seg/trg.vocab.32k.txt
PAIR=${DATA_DIR}/data/lang8_seg/bpe32k
PORT=3001
echo ${PORT}, ${DEVICE}

CUDA_VISIBLE_DEVICES=${DEVICES} PYTHONPATH=${CODE} python demo.py --model transformer \
--vocabulary ${VSRC} ${VTRG} --pair ${PAIR} --checkpoint ${MODEL} --port ${PORT} \
--parameters "beam_size=4,device_list=[0],decode_length=80,decode_alpha=0.6"




#!/usr/bin/env bash
export PATH=/usr/local/cuda-9.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH
#set -ex
DATA_DIR='/data/disk1/private/wangmuzi/data/THUMT'
CODE='~/THUMT'

DEVICES=6
MODEL=${DATA_DIR}/checkpoints/cged_transformer/model.ckpt-9000
VSRC=${DATA_DIR}/data/cged/src.vocab.10k.txt
VTRG=${DATA_DIR}/data/cged/trg.vocab.10k.txt
PAIR=${DATA_DIR}/data/cged/bpe10k
PORT=3001
echo ${PORT}, ${DEVICE}

CUDA_VISIBLE_DEVICES=${DEVICES} PYTHONPATH=${CODE} python demo.py --model transformer \
--vocabulary ${VSRC} ${VTRG} --pair ${PAIR} --checkpoint ${MODEL} --port ${PORT} \
--parameters "beam_size=4,device_list=[0],decode_length=80,decode_alpha=0.6"



#!/usr/bin/env bash
export PATH=/usr/local/cuda-9.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH
#set -ex
DATA_DIR='/data/disk1/private/wangmuzi/data/THUMT'
CODE='~/THUMT'

DEVICES=6
MODEL=${DATA_DIR}/checkpoints/rmrb_transformer/model.ckpt-200000
VSRC=${DATA_DIR}/data/rmrb/src.vocab.32k.txt
VTRG=${DATA_DIR}/data/rmrb/trg.vocab.32k.txt
PAIR=${DATA_DIR}/data/rmrb/bpe32k
PORT=3001
echo ${PORT}, ${DEVICE}

CUDA_VISIBLE_DEVICES=${DEVICES} PYTHONPATH=${CODE} python demo.py --model transformer \
--vocabulary ${VSRC} ${VTRG} --pair ${PAIR} --checkpoint ${MODEL} --port ${PORT} \
--parameters "beam_size=4,device_list=[0],decode_length=80,decode_alpha=0.6"