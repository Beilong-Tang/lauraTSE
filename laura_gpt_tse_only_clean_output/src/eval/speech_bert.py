#
# Wrapper for SpeechBert Score: https://github.com/Takaaki-Saeki/DiscreteSpeechMetrics 
#

import os 
import sys
import os.path as op
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import glob
from pathlib import Path
import argparse
from discrete_speech_metrics import SpeechBERTScore



def parse_argss():
    p = argparse.ArgumentParser()
    # Model specific
    p.add_argument("--model", default='hubert-base')
    p.add_argument("--layer", default = 11)

    # Data specific
    p.add_argument('--test_dir', type = str, required=True)
    p.add_argument('--ref_dir', type = str, required=True)

    p.add_argument('--ref_suffix', type = str, default = 'wav')

    # Output specific
    p.add_argument('--out_path',type=str, required=True)

    return p.parse_args()

def main(args):
    metrics = SpeechBERTScore(sr=16000,
                              model_type=args.model,
                              layer=args.layer,
                              use_gpu=True)
    
    suffix = args.test_suffix

    ref_audio_paths = glob.glob(op.join(args.ref_dir, f'*.{suffix}'))
    ref_audio_path_dict = dict([(Path(p).stem, p) for p in ref_audio_paths])

    out_audio_paths = glob.glob(op.join(args.test_dir, ".wav"))
    out_audio_path_dict = dict([(Path(p).stem, p) for p in out_audio_paths])

    for _k, _out_path in out_audio_path_dict:
        _ref_path = ref_audio_path_dict.get(_k)
        assert _ref_path is not None
        
    







if __name__ == "__main__":
    args = parse_argss()
    main(args)
    pass