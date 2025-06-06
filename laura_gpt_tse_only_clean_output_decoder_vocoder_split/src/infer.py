## Inference python scripts
import os
import sys 
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import argparse

import logging
import torch
import torch.multiprocessing as mp
import tqdm
import time
import numpy as np
from pathlib import Path

import torchaudio
import soundfile as sf
from utils.utils import AttrDict, update_args, setup_seed
from bin.tse_inference import TSExtraction
from utils.utils import get_source_list
from utils.mel_spectrogram import MelSpec


def parse_args():
    parser = argparse.ArgumentParser()
    ## laura gpt related
    parser.add_argument("--sampling", default=25, type=int)
    parser.add_argument("--beam_size", default=1, type=int)

    parser.add_argument("--mix_wav_scp", type=str, default = None)
    parser.add_argument("--ref_wav_scp", type=str, default = None)

    parser.add_argument("--decoder_config", type=str)
    parser.add_argument("--encoder_config", type=str)
    parser.add_argument("--decoder_only", action='store_true', help = 'if inference only using decoder')
    parser.add_argument("--decoder_ckpt", type=str)
    parser.add_argument("--encoder_ckpt", type=str)
    parser.add_argument("--output_dir", type=str)
    ## DDP
    parser.add_argument("--num_proc", type=int, default=4)
    parser.add_argument(
        "--gpus", nargs="+", default=["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    )
    ## Inference methods:
    ## DDP
    parser.add_argument("-i", "--infer", choices=['offline', 'trunk',], default='offline', help= 
                        """ 
                        Inference methods type
                        offline: Infer the whole audio at the same time
                        trunk: splits the audio into multiple trunks with overlap 50\\%, and inference from there
                        one: Output one frame of raw audio form at a time.
                        """)
    parser.add_argument("--hop_ds", type=int, default=2, help="Only useful when infer type is trunk. the Hop Size to move: default: 2 seconds")
    args = parser.parse_args()
    return args


def main(args):
    print(args)
    print(f"Running decoder only: {args.decoder_only}")
    os.makedirs(args.output_dir, exist_ok=True)
    setup_seed(1234, 0)
    mp.spawn(inference, args=(args,), nprocs=args.num_proc, join=True)
    print("done!")


def inference(rank, args):
    # update args to contain config
    update_args(args, args.decoder_config)
    args_encoder = argparse.Namespace()
    update_args(args_encoder, args.encoder_config)
    args = AttrDict(**vars(args))
    args_encoder = AttrDict(**vars(args_encoder))
    args.output_dir = Path(args.output_dir)
    print(f"args: {args}")
    # device setup
    device = args.gpus[rank % len(args.gpus)]
    # data for each process setup

    mix_wav_ids, mix_wav_paths = get_source_list(args.mix_wav_scp, ret_name=True)
    ref_wav_ids, ref_wav_paths = get_source_list(args.ref_wav_scp, ret_name=True)
    

    scp_list = [] # [ [mix_wav, ref_wav, ref_codec], [...] ]
    for id in mix_wav_ids:
        mix_wav_path = mix_wav_paths[mix_wav_ids.index(id)]
        ref_wav_path = ref_wav_paths[ref_wav_ids.index(id)]
        scp_list.append([mix_wav_path, ref_wav_path])

    scp = scp_list[rank::args.num_proc]
    # logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # load model
    logger.info("this")
    tse = TSExtraction(args, args_encoder, args.decoder_ckpt, args.encoder_ckpt, device, logger)
    logger.info(tse)

    # Inference
    total_rtf = 0.0
    with torch.no_grad(), tqdm.tqdm(scp, desc=f"[inferencing...rank {rank}]") as pbar:
        for paths in pbar:
            mix_wav_path, ref_wav_path = paths

            # 0. Mix Mel -> [1, T,]
            mix_audio, sr = torchaudio.load(mix_wav_path)  # [1,T]
            mix_audio = mix_audio.to(device)
            # mask = torch.tensor([mix_audio.size(1)], dtype=torch.long)
            # mix_mel, _ = mel_spec.mel(mix_audio, mask)
            # mix_mel = mix_mel.to(device)

            # 1. Ref Mel -> [1,T,D]
            ref_audio, sr = torchaudio.load(ref_wav_path)  # [1,T]
            if args.max_aux_ds is not None:
                ref_audio = ref_audio[:, -int(args.max_aux_ds * 16000):]
            ref_audio = ref_audio.to(device)
            # mask = torch.tensor([ref_audio.size(1)], dtype=torch.long)
            # ref_mel, _ = mel_spec.mel(ref_audio, mask)
            # ref_mel = ref_mel.to(device)
            ## Limit the reference mel length

            # # 2. Ref Codec ->
            # ref_codec = np.load(ref_codec_path) # [T,N]
            # ref_codec = torch.from_numpy(ref_codec).to(torch.long)# [T,N]
            # ## Limit the reference mel length
            # if args.max_aux_ds is not None:
            #     ref_codec = ref_codec[-int(args.max_aux_ds * args.codec_token_rate):]

            # 1. Inference
            start = time.time()
            output = tse(mix_audio, ref_audio, decoder_only = args.decoder_only)[0]["gen"].squeeze()  # [T]
            rtf = (time.time() - start) / (len(output) / sr)
            pbar.set_postfix({"RTF": rtf})
            total_rtf += rtf

            # 2. Save audio
            base_name = Path(mix_wav_path).stem + ".wav"
            save_path = args.output_dir / base_name

            sf.write(
                save_path,
                normalize(output.cpu().numpy(), ref_audio.cpu().numpy().squeeze()),
                samplerate=sr,
            )
    logger.info(
        f"Finished generation of {len(scp)} utterances (RTF = {total_rtf / len(scp):.03f})."
    )

def normalize(output: np.ndarray, mixture: np.ndarray):
    norm = np.linalg.norm(mixture, np.inf)
    return output * norm / np.max(np.abs(output))


if __name__ == "__main__":
    args = parse_args()
    main(args)
