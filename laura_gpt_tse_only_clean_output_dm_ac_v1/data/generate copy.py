from typing import Dict, Union
import os
import pickle
import yaml


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

def main(scp_list_config: str, spk_scp_dict:Dict[str,str],output_dir:str):
    """
    Arguments:
        scp_list_config: path to yaml spk list. It should contain a `data` field where each field in `data` should contain `clean`, `codec`, and `shape` 
        spk_scp_dict: {"key1": scp_path, "key2": scp_path_2 }. key should be the unique id for the dataset. 
            Note that each scp_path's utt's key should be split by ('-'), and the first element is the unique spk id of the wav.
    """
    with open(scp_list_config, "r") as f:
        pass

    os.makedirs(output_dir, exist_ok=True)
    res = []
    res_spk_dict = {}
    for _name, _path in spk_scp_dict.items():
        _id_path_dict = read_2column_text(_path)
        for _k,_v in _id_path_dict.items():
            res.append(f"{_name}{_k} {_v}\n")
            _spk_id = f"{_name}{_k.split('-')[0]}"
            if res_spk_dict.get(_spk_id) is None:
                res_spk_dict[_spk_id] = [_v]
            else:
                res_spk_dict[_spk_id] = res_spk_dict[_spk_id] + [_v]
    with open(os.path.join(output_dir, "clean.scp"), "w") as f:
        f.writelines(res)
    with open(os.path.join(output_dir, "clean_spk_dict.pkl"), "wb") as f:
        pickle.dump(res_spk_dict, f)
    print("Done...")

if __name__ == "__main__":
    fire.Fire(main)