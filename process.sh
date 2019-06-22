#!/usr/bin/env bash
-----------------lan8_seg----------------------------
DATA_DIR='/data/disk1/private/wangmuzi/data/THUMT'

python resource_mt/subword_nmt/learn_joint_bpe_and_vocab.py \
 --input ${DATA_DIR}/data/lang8_seg/lang8_train.src.seg ${DATA_DIR}/data/lang8_seg/lang8_train.trg.seg \
 -s 32000 -o bpe32k --write-vocabulary  ${DATA_DIR}/data/lang8_seg/src.vocab ${DATA_DIR}/data/lang8_seg/trg.vocab

python resource_mt/subword_nmt/apply_bpe.py --vocabulary ${DATA_DIR}/data/lang8_seg/src.vocab --vocabulary-threshold 25 \
-c bpe32k < ${DATA_DIR}/data/lang8_seg/lang8_train.src.seg > ${DATA_DIR}/data/lang8_seg/lang8_train.32k.src.seg

python resource_mt/subword_nmt/apply_bpe.py --vocabulary ${DATA_DIR}/data/lang8_seg/trg.vocab --vocabulary-threshold 25 \
-c bpe32k < ${DATA_DIR}/data/lang8_seg/lang8_train.trg.seg > ${DATA_DIR}/data/lang8_seg/lang8_train.32k.trg.seg


python resource_mt/subword_nmt/apply_bpe.py --vocabulary ${DATA_DIR}/data/lang8_seg/src.vocab --vocabulary-threshold 25 \
-c bpe32k < ${DATA_DIR}/data/lang8_seg/lang8_valid.src.seg > ${DATA_DIR}/data/lang8_seg/lang8_valid.32k.src.seg

python resource_mt/subword_nmt/apply_bpe.py --vocabulary ${DATA_DIR}/data/lang8_seg/trg.vocab --vocabulary-threshold 25 \
-c bpe32k < ${DATA_DIR}/data/lang8_seg/lang8_valid.trg.seg > ${DATA_DIR}/data/lang8_seg/lang8_valid.32k.trg.seg


python thumt/scripts/build_vocab.py ${DATA_DIR}/data/lang8_seg/lang8_train.32k.src.seg ${DATA_DIR}/data/lang8_seg/src.vocab.32k
python thumt/scripts/build_vocab.py ${DATA_DIR}/data/lang8_seg/lang8_train.32k.trg.seg ${DATA_DIR}/data/lang8_seg/trg.vocab.32k

export CUDA_VISIBLE_DEVICES=4,6
python trainer.py \
--input ${DATA_DIR}/data/lang8_seg/lang8_train.32k.src.seg ${DATA_DIR}/data/lang8_seg/lang8_train.32k.trg.seg \
--output ${DATA_DIR}/checkpoints/lang8_transformer/ \
--validation ${DATA_DIR}/data/lang8_seg/lang8_valid.32k.src.seg --references ${DATA_DIR}/data/lang8_seg/lang8_valid.32k.trg.seg \
--vocabulary ${DATA_DIR}/data/lang8_seg/src.vocab.32k.txt ${DATA_DIR}/data/lang8_seg/trg.vocab.32k.txt \
--model transformer \
--parameters "batch_size=6250,device_list=[0,1],train_steps=200000,save_checkpoint_steps=2000,eval_steps=2000"
#--checkpoint ${DATA_DIR}/checkpoints/lang8_transformer/

export CUDA_VISIBLE_DEVICES=7
python translator.py --models transformer \
--input ${DATA_DIR}/data/lang8_seg/test --output ${DATA_DIR}/data/lang8_seg/pred \
--vocabulary ${DATA_DIR}/data/lang8_seg/src.vocab.32k.txt ${DATA_DIR}/data/lang8_seg/trg.vocab.32k.txt \
--checkpoints ${DATA_DIR}/checkpoints/lang8_transformer/ \
--parameters "device_list=[0]"




--------------------------------- cged -------------------------
DATA_DIR='/data/disk1/private/wangmuzi/data/THUMT'

python resource_mt/subword-nmt/learn_joint_bpe_and_vocab.py \
 --input ${DATA_DIR}/data/cged/cged.trg \
 -s 32000 -o ${DATA_DIR}/data/cged/bpe32k --write-vocabulary  ${DATA_DIR}/data/cged/vocab

python resource_mt/subword-nmt/apply_bpe.py --vocabulary ${DATA_DIR}/data/cged/vocab --vocabulary-threshold 5 \
-c ${DATA_DIR}/data/cged/bpe32k < ${DATA_DIR}/data/cged/cged_train.src > ${DATA_DIR}/data/cged/cged_train.32k.src

python resource_mt/subword-nmt/apply_bpe.py --vocabulary ${DATA_DIR}/data/cged/vocab --vocabulary-threshold 5 \
-c ${DATA_DIR}/data/cged/bpe32k < ${DATA_DIR}/data/cged/cged_train.trg > ${DATA_DIR}/data/cged/cged_train.32k.trg

python resource_mt/subword-nmt/apply_bpe.py --vocabulary ${DATA_DIR}/data/cged/vocab --vocabulary-threshold 5 \
-c ${DATA_DIR}/data/cged/bpe32k < ${DATA_DIR}/data/cged/cged_valid.src > ${DATA_DIR}/data/cged/cged_valid.32k.src

python resource_mt/subword-nmt/apply_bpe.py --vocabulary ${DATA_DIR}/data/cged/vocab --vocabulary-threshold 5 \
-c ${DATA_DIR}/data/cged/bpe32k < ${DATA_DIR}/data/cged/cged_valid.trg > ${DATA_DIR}/data/cged/cged_valid.32k.trg


python thumt/scripts/build_vocab.py ${DATA_DIR}/data/cged/cged_train.32k.src ${DATA_DIR}/data/cged/src.vocab.32k
python thumt/scripts/build_vocab.py ${DATA_DIR}/data/cged/cged_train.32k.trg ${DATA_DIR}/data/cged/trg.vocab.32k

export CUDA_VISIBLE_DEVICES=4,6
python trainer.py \
--input ${DATA_DIR}/data/cged/cged_train.10k.src ${DATA_DIR}/data/cged/cged_train.10k.trg \
--output ${DATA_DIR}/checkpoints/cged_transformer/ \
--validation ${DATA_DIR}/data/cged/cged_valid.10k.src --references ${DATA_DIR}/data/cged/cged_valid.10k.trg \
--vocabulary ${DATA_DIR}/data/cged/src.vocab.10k.txt ${DATA_DIR}/data/cged/trg.vocab.10k.txt \
--model transformer \
--parameters "batch_size=10000,device_list=[0,1],train_steps=200000,save_checkpoint_steps=1000,eval_steps=1500"
#--checkpoint ${DATA_DIR}/checkpoints/lang8_transformer/


export CUDA_VISIBLE_DEVICES=1
python translator.py --models transformer \
--input ${DATA_DIR}/data/cged/test --output ${DATA_DIR}/data/cged/pred \
--vocabulary ${DATA_DIR}/data/cged/src.vocab.10k.txt ${DATA_DIR}/data/cged/trg.vocab.10k.txt \
--checkpoints ${DATA_DIR}/checkpoints/cged_transformer/ \
--parameters "device_list=[0]"






--------------------------------- rmrb -------------------------
DATA_DIR='/data/disk1/private/wangmuzi/data/THUMT'

python resource_mt/subword_nmt/learn_joint_bpe_and_vocab.py \
 --input ${DATA_DIR}/data/rmrb/rmrb \
 -s 32000 -o ${DATA_DIR}/data/rmrb/bpe32k --write-vocabulary  ${DATA_DIR}/data/rmrb/vocab

python resource_mt/subword_nmt/apply_bpe.py --vocabulary ${DATA_DIR}/data/rmrb/vocab --vocabulary-threshold 15 \
-c ${DATA_DIR}/data/rmrb/bpe32k < ${DATA_DIR}/data/rmrb/train.src > ${DATA_DIR}/data/rmrb/train.32k.src

python resource_mt/subword_nmt/apply_bpe.py --vocabulary ${DATA_DIR}/data/rmrb/vocab --vocabulary-threshold 15 \
-c ${DATA_DIR}/data/rmrb/bpe32k < ${DATA_DIR}/data/rmrb/train.trg > ${DATA_DIR}/data/rmrb/train.32k.trg

python resource_mt/subword_nmt/apply_bpe.py --vocabulary ${DATA_DIR}/data/rmrb/vocab --vocabulary-threshold 15 \
-c ${DATA_DIR}/data/rmrb/bpe32k < ${DATA_DIR}/data/rmrb/valid.src > ${DATA_DIR}/data/rmrb/valid.32k.src

python resource_mt/subword_nmt/apply_bpe.py --vocabulary ${DATA_DIR}/data/rmrb/vocab --vocabulary-threshold 15 \
-c ${DATA_DIR}/data/rmrb/bpe32k < ${DATA_DIR}/data/rmrb/valid.trg > ${DATA_DIR}/data/rmrb/valid.32k.trg


python thumt/scripts/build_vocab.py ${DATA_DIR}/data/rmrb/train.32k.src ${DATA_DIR}/data/rmrb/src.vocab.32k
python thumt/scripts/build_vocab.py ${DATA_DIR}/data/rmrb/train.32k.trg ${DATA_DIR}/data/rmrb/trg.vocab.32k

export CUDA_VISIBLE_DEVICES=6,7
python trainer.py \
--input ${DATA_DIR}/data/rmrb/train.32k.src ${DATA_DIR}/data/rmrb/train.32k.trg \
--output ${DATA_DIR}/checkpoints/rmrb_transformer/ \
--validation ${DATA_DIR}/data/rmrb/valid.32k.src --references ${DATA_DIR}/data/rmrb/valid.32k.trg \
--vocabulary ${DATA_DIR}/data/rmrb/src.vocab.32k.txt ${DATA_DIR}/data/rmrb/trg.vocab.32k.txt \
--model transformer \
--parameters "batch_size=6250,device_list=[0,1],train_steps=200000,save_checkpoint_steps=2000,eval_steps=3000"
#--checkpoint ${DATA_DIR}/checkpoints/lang8_transformer/


export CUDA_VISIBLE_DEVICES=1
python translator.py --models transformer \
--input ${DATA_DIR}/data/cged/test --output ${DATA_DIR}/data/cged/pred \
--vocabulary ${DATA_DIR}/data/cged/src.vocab.10k.txt ${DATA_DIR}/data/cged/trg.vocab.10k.txt \
--checkpoints ${DATA_DIR}/checkpoints/cged_transformer/ \
--parameters "device_list=[0]"


--------------------------------- lang8 -------------------------
DATA_DIR='/data/disk1/private/wangmuzi/data/THUMT'

python resource_mt/subword_nmt/learn_joint_bpe_and_vocab.py \
 --input ${DATA_DIR}/data/lang8/lang8 \
 -s 32000 -o ${DATA_DIR}/data/lang8/bpe32k --write-vocabulary  ${DATA_DIR}/data/lang8/vocab

python resource_mt/subword_nmt/apply_bpe.py --vocabulary ${DATA_DIR}/data/lang8/vocab --vocabulary-threshold 15 \
-c ${DATA_DIR}/data/lang8/bpe32k < ${DATA_DIR}/data/lang8/train.src > ${DATA_DIR}/data/lang8/train.32k.src

python resource_mt/subword_nmt/apply_bpe.py --vocabulary ${DATA_DIR}/data/lang8/vocab --vocabulary-threshold 15 \
-c ${DATA_DIR}/data/lang8/bpe32k < ${DATA_DIR}/data/lang8/train.trg > ${DATA_DIR}/data/lang8/train.32k.trg

python resource_mt/subword_nmt/apply_bpe.py --vocabulary ${DATA_DIR}/data/lang8/vocab --vocabulary-threshold 15 \
-c ${DATA_DIR}/data/lang8/bpe32k < ${DATA_DIR}/data/lang8/valid.src > ${DATA_DIR}/data/lang8/valid.32k.src

python resource_mt/subword_nmt/apply_bpe.py --vocabulary ${DATA_DIR}/data/lang8/vocab --vocabulary-threshold 15 \
-c ${DATA_DIR}/data/lang8/bpe32k < ${DATA_DIR}/data/lang8/valid.trg > ${DATA_DIR}/data/lang8/valid.32k.trg


python thumt/scripts/build_vocab.py ${DATA_DIR}/data/lang8/train.32k.src ${DATA_DIR}/data/lang8/src.vocab.32k
python thumt/scripts/build_vocab.py ${DATA_DIR}/data/lang8/train.32k.trg ${DATA_DIR}/data/lang8/trg.vocab.32k

export CUDA_VISIBLE_DEVICES=4,5
python trainer.py \
--input ${DATA_DIR}/data/lang8/train.32k.src ${DATA_DIR}/data/lang8/train.32k.trg \
--output ${DATA_DIR}/checkpoints/lang8_transformer1/ \
--validation ${DATA_DIR}/data/lang8/valid.32k.src --references ${DATA_DIR}/data/lang8/valid.32k.trg \
--vocabulary ${DATA_DIR}/data/lang8/src.vocab.32k.txt ${DATA_DIR}/data/lang8/trg.vocab.32k.txt \
--model transformer \
--parameters "batch_size=6250,device_list=[0,1],train_steps=200000,save_checkpoint_steps=1000,eval_steps=1000"
#--checkpoint ${DATA_DIR}/checkpoints/lang8_transformer/

--------------------------lang8_seg test------------------------------
export CUDA_VISIBLE_DEVICES=1
python translator.py --models transformer \
--input ${DATA_DIR}/data/cged/test --output ${DATA_DIR}/data/cged/pred \
--vocabulary ${DATA_DIR}/data/cged/src.vocab.10k.txt ${DATA_DIR}/data/cged/trg.vocab.10k.txt \
--checkpoints ${DATA_DIR}/checkpoints/cged_transformer/ \
--parameters "batch_size=6250,device_list=[0,1],train_steps=200000,save_checkpoint_steps=1000,eval_steps=1000"


sed -r 's/(@@ )|(@@ ?$)//g' < ${DATA_DIR}/data/lang8_seg/pred > ${DATA_DIR}/data/lang8_seg/pred.norm
sed -r 's/(@@ )|(@@ ?$)//g' < ${DATA_DIR}/data/lang8_seg/test_result > ${DATA_DIR}/data/lang8_seg/test_result.norm
sed -r 's/(@@ )|(@@ ?$)//g' < ${DATA_DIR}/data/lang8_seg/test > ${DATA_DIR}/data/lang8_seg/test.norm
perl resource_mt/subword-nmt/multi-bleu.perl -lc ${DATA_DIR}/data/lang8_seg/test_result.norm < ${DATA_DIR}/data/lang8_seg/pred.norm > ${DATA_DIR}/data/lang8_seg/evalResult


python get_relevance.py \
--model transformer --input ${DATA_DIR}/data/lang8_seg/test --output ${DATA_DIR}/data/lang8_seg/pred \
--vocabulary ${DATA_DIR}/data/lang8_seg/src.vocab.32k.txt ${DATA_DIR}/data/lang8_seg/trg.vocab.32k.txt \
--checkpoints ${DATA_DIR}/checkpoints/lang8_transformer/ --relevances  ${DATA_DIR}/data/lang8_seg/lrp \
--parameters "device_list=[0],decode_batch_size=1"

python visualize.py ${DATA_DIR}/data/lang8_seg/lrp/n