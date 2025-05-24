#!/bin/bash

scp_file="/Netdata/2021/zb/data/LibriMix/Libri2Mix/wav16k/min/lists/test/s1.scp"
nq=2 # first 2 layer output
output="output_sv/libri2mix_clean/nq_$nq"

model="/DKUdata/tangbl/FunCodec/egs/LibriTTS/codec/exp/audio_codec-encodec-zh_en-general-16k-nq32ds640-pytorch/model.pth"
config="/DKUdata/tangbl/FunCodec/egs/LibriTTS/codec/exp/audio_codec-encodec-zh_en-general-16k-nq32ds640-pytorch/config.yaml"

num_proc=8
gpus="cuda:4"

python utils/export_funcodec.py --scp_file $scp_file --nq $nq \
    --output $output \
    --model $model --config $config \
    --num_proc $num_proc --gpus $gpus