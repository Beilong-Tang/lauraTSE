from typing import Dict, Union
import os
import pickle
import yaml
from collections import defaultdict


import fire 
from pathlib import Path

def read_2column_text(path: Union[Path, str]) -> Dict[str, str]:
    """Read a text file having 2 column as dict object.

    Examples:
        wav.scp:
            key1 /some/path/a.wav
            key2 /some/path/b.wav

        >>> read_2column_text('wav.scp')
        {'key1': '/some/path/a.wav', 'key2': '/some/path/b.wav'}

    """

    data = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for linenum, line in enumerate(f, 1):
            sps = line.rstrip().split(maxsplit=1)
            if len(sps) == 1:
                k, v = sps[0], ""
            else:
                k, v = sps
            if k in data:
                raise RuntimeError(f"{k} is duplicated ({path}:{linenum})")
            data[k] = v
    return data

def main(scp_list_config: str,output_dir:str):
    """
    Arguments:
        scp_list_config: path to yaml spk list. It should contain a `data` field where each field in `data` contains `clean`, `codec`, and `shape` 
        spk_scp_dict: {"key1": scp_path, "key2": scp_path_2 }. key should be the unique id for the dataset. 
            Note that each scp_path's utt's key should be split by ('-'), and the first element is the unique spk id of the wav.
    Returns:
        clean.scp
        codec.scp
        shape.scp
        spk_dict.scp

    """
    with open(scp_list_config, "r") as f:
        scp_dict:dict = yaml.safe_load(f)['data']
        pass

    res_clean = []
    res_codec = []
    res_shape = []
    res_spk_dict = defaultdict(list)
    
    for _name, _values in scp_dict.items():
        _clean_scp = _values['clean']
        _codec_scp = _values['codec']
        _shape_scp = _values['shape']
        _clean_scp_dict = read_2column_text(_clean_scp)
        _codec_scp_dict = read_2column_text(_codec_scp)
        _shape_scp_dict = read_2column_text(_shape_scp)

        for _utt_id in _clean_scp_dict.keys():
            _utt_id_new = f"{_name}{_utt_id}"
            _spk_id = f"{_name}{_utt_id.split('-')[0]}"
            res_clean.append(f"{_utt_id_new} {_clean_scp_dict[_utt_id]}\n")
            res_codec.append(f"{_utt_id_new} {_codec_scp_dict[_utt_id]}\n")
            res_shape.append(f"{_utt_id_new} {_shape_scp_dict[_utt_id]}\n")
        
            res_spk_dict[_spk_id].append(_clean_scp_dict[_utt_id])

    with open(os.path.join(output_dir, "clean.scp"), "w") as f:
        f.writelines(res_clean)
    with open(os.path.join(output_dir, "codec.scp"), "w") as f:
        f.writelines(res_codec)
    with open(os.path.join(output_dir, "shape.scp"), "w") as f:
        f.writelines(res_shape)
    with open(os.path.join(output_dir, "clean_spk_dict.pkl"), "wb") as f:
        pickle.dump(res_spk_dict, f)
    print("Done...")

if __name__ == "__main__":
    fire.Fire(main)