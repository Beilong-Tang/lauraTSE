# Copyright ESPnet (https://github.com/espnet/espnet). All Rights Reserved.
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import collections
import functools
import logging
import numbers
import re
from typing import Mapping
from typing import Union

import h5py
import kaldiio
import numpy as np
import torch
import copy
import humanfriendly
from typeguard import check_argument_types

from funcodec.fileio.npy_scp import NpyScpReader
from funcodec.fileio.rand_gen_dataset import FloatRandomGenerateDataset
from funcodec.fileio.rand_gen_dataset import IntRandomGenerateDataset
from funcodec.fileio.read_text import load_num_sequence_text
from funcodec.fileio.read_text import read_2column_text
from funcodec.fileio.sound_scp import SoundScpReader
from funcodec.fileio.read_text import read_2column_text
from funcodec.datasets.dataset import ESPnetDataset
from funcodec.utils.sized_dict import SizedDict
from typing import Any
from typing import Callable
from typing import Collection
from typing import Dict
from typing import Mapping
from typing import Tuple
from typing import Union

import yaml


# from dataset.augmentation import generate_from_config, generate_augmentations_config

from utils.hinter import hint_once


class AdapterForSoundScpReader(collections.abc.Mapping):
    def __init__(self, loader, dtype=None):
        assert check_argument_types()
        self.loader = loader
        self.dtype = dtype
        self.rate = None

    def keys(self):
        return self.loader.keys()

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        return iter(self.loader)

    def __getitem__(self, key: str) -> np.ndarray:
        retval = self.loader[key]

        if isinstance(retval, tuple):
            assert len(retval) == 2, len(retval)
            if isinstance(retval[0], int) and isinstance(retval[1], np.ndarray):
                # sound scp case
                rate, array = retval
            elif isinstance(retval[1], int) and isinstance(retval[0], np.ndarray):
                # Extended ark format case
                array, rate = retval
            else:
                raise RuntimeError(
                    f"Unexpected type: {type(retval[0])}, {type(retval[1])}"
                )

            if self.rate is not None and self.rate != rate:
                raise RuntimeError(
                    f"Sampling rates are mismatched: {self.rate} != {rate}"
                )
            self.rate = rate
            # Multichannel wave file
            # array: (NSample, Channel) or (Nsample)
            if array.dtype == np.int16:
                array = array / (2 ** 15)
            elif array.dtype == np.int32:
                array = array / (2 ** 31)
            if self.dtype is not None:
                array = array.astype(self.dtype)

        else:
            # Normal ark case
            assert isinstance(retval, np.ndarray), type(retval)
            array = retval
            if self.dtype is not None:
                array = array.astype(self.dtype)

        assert isinstance(array, np.ndarray), type(array)
        return array


class H5FileWrapper:
    def __init__(self, path: str):
        self.path = path
        self.h5_file = h5py.File(path, "r")

    def __repr__(self) -> str:
        return str(self.h5_file)

    def __len__(self) -> int:
        return len(self.h5_file)

    def __iter__(self):
        return iter(self.h5_file)

    def __getitem__(self, key) -> np.ndarray:
        value = self.h5_file[key]
        return value[()]


def sound_loader(path, float_dtype=None):
    # The file is as follows:
    #   utterance_id_A /some/where/a.wav
    #   utterance_id_B /some/where/a.flac

    # NOTE(kamo): SoundScpReader doesn't support pipe-fashion
    # like Kaldi e.g. "cat a.wav |".
    # NOTE(kamo): The audio signal is normalized to [-1,1] range.
    loader = SoundScpReader(path, normalize=True, always_2d=False)

    # SoundScpReader.__getitem__() returns Tuple[int, ndarray],
    # but ndarray is desired, so Adapter class is inserted here
    return AdapterForSoundScpReader(loader, float_dtype)


def kaldi_loader(path, float_dtype=None, max_cache_fd: int = 0):
    loader = kaldiio.load_scp(path, max_cache_fd=max_cache_fd)
    return AdapterForSoundScpReader(loader, float_dtype)


def rand_int_loader(filepath, loader_type):
    # e.g. rand_int_3_10
    try:
        low, high = map(int, loader_type[len("rand_int_") :].split("_"))
    except ValueError:
        raise RuntimeError(f"e.g rand_int_3_10: but got {loader_type}")
    return IntRandomGenerateDataset(filepath, low, high)


def build_codec_loader(filepath, quant_groups=32, file_type="ark"):
    from funcodec.fileio.codec_loader import CodecLoader
    return CodecLoader(filepath, quant_groups=quant_groups, file_type=file_type)
    
import random
import librosa
import pickle
from pathlib import Path
from src.utils.mel_spectrogram import MelSpec

def normalize(audio):
    max_value = np.max(np.abs(audio))
    return audio * (1 / (max_value + 1e-8))

class DmMixSpkReader:
    def __init__(self, clean_path, spk_dict_path:str, mel_config:dict, snr = 5):
        # snr in [0,5]
        self.clean_scp = read_2column_text(clean_path)
        with open(spk_dict_path, "rb") as f:
            self.spk_dict = pickle.load(f)
        self.snr = snr
        self.mel_proc = MelSpec(**mel_config)
        pass 

    def __len__(self):
        return len(self.clean_scp)
    
    def __iter__(self):
        return iter(self.clean_scp)
    
    def __getitem__(self, uid):
        # load the path
        clean_path = self.clean_scp[uid]
        clean_spk_id = uid.split("-")[0]

        intf_spk = random.choice(list(self.spk_dict.keys()))
        while intf_spk == clean_spk_id:
            intf_spk = random.choice(list(self.spk_dict.keys()))
        intf_path = random.choice(self.spk_dict[intf_spk])

        # load the audio
        clean_audio, _ = librosa.load(clean_path, sr=None)
        intf_audio, _ = librosa.load(intf_path, sr=None)
        ## pad the length 
        if clean_audio.shape[0] > intf_audio[0]:
            ## repeat intf_audio 
            new_intf_audio = np.tile(intf_audio, len(clean_audio) // len(intf_audio) + 1)
            intf_audio = new_intf_audio[:len(clean_audio)]
        elif clean_audio.shape[0] < intf_audio[0]:
            offset = random.randint(0, len(intf_audio) - len(clean_audio) - 1)
            intf_audio = intf_audio[offset: offset + len(clean_audio)]
        assert intf_audio.shape == clean_audio.shape
        ## normalize 
        clean_audio, intf_audio = normalize(clean_audio), normalize(intf_audio)
        ## snr
        _snr = random.random() * self.snr
        intf_audio = intf_audio * 10 ** (-_snr / 20)
        mix = clean_audio + intf_audio
        return self.mel_proc.mel_one_np(mix)

        
class DmRefReader:
    def __init__(self, clean_path, spk_dict_path:str, mel_config:dict, ref_ds:Union[int, tuple]):
        with open(spk_dict_path, "rb") as f:
            self.spk_dict = pickle.load(f)
        self.clean_scp = read_2column_text(clean_path)
        self.ds = ref_ds
        self.mel_proc = MelSpec(**mel_config)
    
    def __len__(self):
        return len(self.clean_scp)

    def __iter__(self):
        return iter(self.clean_scp)


    def _clip_wav(self, wav:np.ndarray, ds: Union[int, list], sr):
        def _random_select_wav(wav:np.ndarray, length):
            offset = random.randint(0, len(wav) - length - 1)
            return wav[offset: offset + length]
        if isinstance(ds, int):
            if len(wav) <= int(ds * sr):
                return wav
            else:
                return _random_select_wav(wav, int(ds * sr))
        elif isinstance(ds, list):
            lower = int(ds[0] * sr)
            upper = int(ds[-1] * sr)
            if len(wav) <= lower:
                return wav
            elif len(wav) <= upper:
                length = random.randint(lower, len(wav)-1)
                return _random_select_wav(wav, length)
            else:
                return _random_select_wav(wav, upper)

    def __getitem__(self, uid):
        spk = uid.split("-")[0]

        ref_path = random.choice(self.spk_dict[spk])
        while Path(ref_path).stem == uid:
            ref_path = random.choice(self.spk_dict[spk])
        
        ref_speech, sr = librosa.load(ref_path, sr = None) # [T]

        ref_speech = self._clip_wav(ref_speech, self.ds, sr)
        
        ref_speech = normalize(ref_speech)
        return self.mel_proc.mel_one_np(ref_speech)

class MelReader:
    def __init__(self, scp_path, mel_config:dict, ref_ds = None):
        """
        Convert the input audio to mel spectrogram,
        if ref_ds is not None, then clip the audio and only choose the last ds seconds
        """
        self.scp_dict = read_2column_text(scp_path)
        self.mel_proc = MelSpec(**mel_config)
        self.ds =  ref_ds
    
    def __len__(self):
        return len(self.scp_dict)

    def __iter__(self):
        return iter(self.scp_dict)

    def __getitem__(self, uid):
        audio_path = self.scp_dict[uid]
        audio, sr = librosa.load(audio_path, sr = None)
        if self.ds is not None:
            audio = audio[-int(sr * self.ds):]
        audio = normalize(audio)
        return self.mel_proc.mel_one_np(audio)


DATA_TYPES = {
    "dm_libri_mix": dict(
        func=DmMixSpkReader, 
        kwargs=['spk_dict_path', "mel_config"],
        help="Dynamic Mixing for librispeech intference speech"
        ), ## Newly added for dynamic mixing noise 
    "dm_libri_ref": dict(
        func=DmRefReader, 
        kwargs=['spk_dict_path', "mel_config", "ref_ds"],
        help="Dynamic Mixing for reference speech for librispeech"
        ), ## Newly added for dynamic mixing noise 
    "mixture_eval": dict(
        func=MelReader, 
        kwargs=["mel_config"],
        help="audio to mel"
        ), ## Mel spectrogram for the audio
    "ref_eval": dict(
        func=MelReader, 
        kwargs=["mel_config", "ref_ds"],
        help="audio to mel"
        ), ## Mel spectrogram for the audio for ref_eval
    "sound": dict(
        func=sound_loader,
        kwargs=["float_dtype"],
        help="Audio format types which supported by sndfile wav, flac, etc."
        "\n\n"
        "   utterance_id_a a.wav\n"
        "   utterance_id_b b.wav\n"
        "   ...",
    ),
    "kaldi_ark": dict(
        func=kaldi_loader,
        kwargs=["max_cache_fd"],
        help="Kaldi-ark file type."
        "\n\n"
        "   utterance_id_A /some/where/a.ark:123\n"
        "   utterance_id_B /some/where/a.ark:456\n"
        "   ...",
    ),
    "npy": dict(
        func=NpyScpReader,
        kwargs=[],
        help="Npy file format."
        "\n\n"
        "   utterance_id_A /some/where/a.npy\n"
        "   utterance_id_B /some/where/b.npy\n"
        "   ...",
    ),
    "text_int": dict(
        func=functools.partial(load_num_sequence_text, loader_type="text_int"),
        kwargs=[],
        help="A text file in which is written a sequence of interger numbers "
        "separated by space."
        "\n\n"
        "   utterance_id_A 12 0 1 3\n"
        "   utterance_id_B 3 3 1\n"
        "   ...",
    ),
    "csv_int": dict(
        func=functools.partial(load_num_sequence_text, loader_type="csv_int"),
        kwargs=[],
        help="A text file in which is written a sequence of interger numbers "
        "separated by comma."
        "\n\n"
        "   utterance_id_A 100,80\n"
        "   utterance_id_B 143,80\n"
        "   ...",
    ),
    "text_float": dict(
        func=functools.partial(load_num_sequence_text, loader_type="text_float"),
        kwargs=[],
        help="A text file in which is written a sequence of float numbers "
        "separated by space."
        "\n\n"
        "   utterance_id_A 12. 3.1 3.4 4.4\n"
        "   utterance_id_B 3. 3.12 1.1\n"
        "   ...",
    ),
    "csv_float": dict(
        func=functools.partial(load_num_sequence_text, loader_type="csv_float"),
        kwargs=[],
        help="A text file in which is written a sequence of float numbers "
        "separated by comma."
        "\n\n"
        "   utterance_id_A 12.,3.1,3.4,4.4\n"
        "   utterance_id_B 3.,3.12,1.1\n"
        "   ...",
    ),
    "text": dict(
        func=read_2column_text,
        kwargs=[],
        help="Return text as is. The text must be converted to ndarray "
        "by 'preprocess'."
        "\n\n"
        "   utterance_id_A hello world\n"
        "   utterance_id_B foo bar\n"
        "   ...",
    ),
    "hdf5": dict(
        func=H5FileWrapper,
        kwargs=[],
        help="A HDF5 file which contains arrays at the first level or the second level."
        "   >>> f = h5py.File('file.h5')\n"
        "   >>> array1 = f['utterance_id_A']\n"
        "   >>> array2 = f['utterance_id_B']\n",
    ),
    "rand_float": dict(
        func=FloatRandomGenerateDataset,
        kwargs=[],
        help="Generate random float-ndarray which has the given shapes "
        "in the file."
        "\n\n"
        "   utterance_id_A 3,4\n"
        "   utterance_id_B 10,4\n"
        "   ...",
    ),
    "rand_int_\\d+_\\d+": dict(
        func=rand_int_loader,
        kwargs=["loader_type"],
        help="e.g. 'rand_int_0_10'. Generate random int-ndarray which has the given "
        "shapes in the path. "
        "Give the lower and upper value by the file type. e.g. "
        "rand_int_0_10 -> Generate integers from 0 to 10."
        "\n\n"
        "   utterance_id_A 3,4\n"
        "   utterance_id_B 10,4\n"
        "   ...",
    ),
}

class DMESPnetDataset(ESPnetDataset):
    """Pytorch Dataset class for ESPNet.

    Added only the DATA_TYPES the mix standing for mixing

    Examples:
        >>> dataset = ESPnetDataset([('wav.scp', 'input', 'sound'),
        ...                          ('token_int', 'output', 'text_int')],
        ...                         )
        ... uttid, data = dataset['uttid']
        {'input': per_utt_array, 'output': per_utt_array}
    """
    def __init__(
        self,
        path_name_type_list: Collection[Tuple[str, str, str]],
        preprocess: Callable[
            [str, Dict[str, np.ndarray]], Dict[str, np.ndarray]
        ] = None,
        float_dtype: str = "float32",
        int_dtype: str = "long",
        max_cache_size: Union[float, int, str] = 0.0,
        max_cache_fd: int = 0,
        spk_dict_path:str = None, ## Note that this cannot be None
        mel_config: dict = None,
        ref_ds: Union[list, int] = 5 # This can be either an int or a list representing the range of the reference speech.
    ):
        assert spk_dict_path is not None
        self.mel_config = mel_config
        self.spk_dict_path = spk_dict_path
        self.ref_ds = ref_ds


        if len(path_name_type_list) == 0:
            raise ValueError(
                '1 or more elements are required for "path_name_type_list"'
            )

        path_name_type_list = copy.deepcopy(path_name_type_list)
        self.preprocess = preprocess

        self.float_dtype = float_dtype
        self.int_dtype = int_dtype
        self.max_cache_fd = max_cache_fd

        self.loader_dict = {}
        self.debug_info = {}
        for path, name, _type in path_name_type_list:
            if name in self.loader_dict:
                raise RuntimeError(f'"{name}" is duplicated for data-key')
            loader = self._build_loader(path, _type)
            self.loader_dict[name] = loader
            self.debug_info[name] = path, _type
            if len(self.loader_dict[name]) == 0:
                raise RuntimeError(f"{path} has no samples")

            # TODO(kamo): Should check consistency of each utt-keys?

        if isinstance(max_cache_size, str):
            max_cache_size = humanfriendly.parse_size(max_cache_size)
        self.max_cache_size = max_cache_size
        if max_cache_size > 0:
            self.cache = SizedDict(shared=True)
        else:
            self.cache = None

    def _build_loader(
        self, path: str, loader_type: str
    ) -> Mapping[str, Union[np.ndarray, torch.Tensor, str, numbers.Number]]:
        """Helper function to instantiate Loader.

        Args:
            path:  The file path
            loader_type:  loader_type. sound, npy, text_int, text_float, etc
        """
        for key, dic in DATA_TYPES.items():
            # e.g. loader_type="sound"
            # -> return DATA_TYPES["sound"]["func"](path)
            if re.match(key, loader_type):
                kwargs = {}
                for key2 in dic["kwargs"]:
                    if key2 == "loader_type":
                        kwargs["loader_type"] = loader_type
                    elif key2 == "float_dtype":
                        kwargs["float_dtype"] = self.float_dtype
                    elif key2 == "int_dtype":
                        kwargs["int_dtype"] = self.int_dtype
                    elif key2 == "max_cache_fd":
                        kwargs["max_cache_fd"] = self.max_cache_fd
                    
                    ## Add the dynamic mixing here
                    elif key2 == "spk_dict_path":
                        kwargs['spk_dict_path'] = self.spk_dict_path 
                    elif key2 == "mel_config":
                        kwargs['mel_config'] = self.mel_config
                    elif key2 =="ref_ds":
                        kwargs['ref_ds'] = self.ref_ds
                    
                    else:
                        raise RuntimeError(f"Not implemented keyword argument: {key2}")

                func = dic["func"]
                try:
                    return func(path, **kwargs) # returns a value
                except Exception:
                    if hasattr(func, "__name__"):
                        name = func.__name__
                    else:
                        name = str(func)
                    logging.error(f"An error happened with {name}({path})")
                    raise
        else:
            raise RuntimeError(f"Not supported: loader_type={loader_type}")

    # def has_name(self, name) -> bool:
    #     return name in self.loader_dict

    # def names(self) -> Tuple[str, ...]:
    #     return tuple(self.loader_dict)

    # def __iter__(self):
    #     return iter(next(iter(self.loader_dict.values())))

    # def __repr__(self):
    #     _mes = self.__class__.__name__
    #     _mes += "("
    #     for name, (path, _type) in self.debug_info.items():
    #         _mes += f'\n  {name}: {{"path": "{path}", "type": "{_type}"}}'
    #     _mes += f"\n  preprocess: {self.preprocess})"
    #     return _mes

    # def __getitem__(self, uid: Union[str, int]) -> Tuple[str, Dict[str, np.ndarray]]:
    #     assert check_argument_types()

    #     # Change integer-id to string-id
    #     if isinstance(uid, int):
    #         d = next(iter(self.loader_dict.values()))
    #         uid = list(d)[uid]

    #     if self.cache is not None and uid in self.cache:
    #         data = self.cache[uid]
    #         return uid, data

    #     data = {}
    #     # 1. Load data from each loaders
    #     for name, loader in self.loader_dict.items():
    #         try:
    #             value = loader[uid]
    #             if isinstance(value, (list, tuple)):
    #                 value = np.array(value)
    #             if not isinstance(
    #                 value, (np.ndarray, torch.Tensor, str, numbers.Number)
    #             ):
    #                 raise TypeError(
    #                     f"Must be ndarray, torch.Tensor, str or Number: {type(value)}"
    #                 )
    #         except Exception:
    #             path, _type = self.debug_info[name]
    #             logging.error(
    #                 f"Error happened with path={path}, type={_type}, id={uid}"
    #             )
    #             raise

    #         # torch.Tensor is converted to ndarray
    #         if isinstance(value, torch.Tensor):
    #             value = value.numpy()
    #         elif isinstance(value, numbers.Number):
    #             value = np.array([value])
    #         data[name] = value

    #     # 2. [Option] Apply preprocessing
    #     #   e.g. funcodec.train.preprocessor:CommonPreprocessor
    #     if self.preprocess is not None:
    #         data = self.preprocess(uid, data)

    #     # 3. Force data-precision
    #     for name in data:
    #         value = data[name]
    #         if not isinstance(value, np.ndarray):
    #             raise RuntimeError(
    #                 f"All values must be converted to np.ndarray object "
    #                 f'by preprocessing, but "{name}" is still {type(value)}.'
    #             )

    #         # Cast to desired type
    #         if value.dtype.kind == "f":
    #             value = value.astype(self.float_dtype)
    #         elif value.dtype.kind == "i":
    #             value = value.astype(self.int_dtype)
    #         else:
    #             raise NotImplementedError(f"Not supported dtype: {value.dtype}")
    #         data[name] = value

    #     if self.cache is not None and self.cache.size < self.max_cache_size:
    #         self.cache[uid] = data

    #     retval = uid, data
    #     assert check_return_type(retval)
    #     return retval
