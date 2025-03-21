### The base class for abstract trainer class
import torch
import os
import time
import torch.distributed as dist
import torch.distributed
from utils.mel_spectrogram import MelSpec
from datetime import timedelta

from funcodec.iterators.sequence_iter_factory import SequenceIterFactory
from funcodec.torch_utils.recursive_op import recursive_average
from utils.utils import Logger

from .helper import dict_to_str, save
from utils.hinter import hint_once 
from utils.postprocess import MaxLength
from funcodec.bin.codec_inference import Speech2Token
from funcodec.modules.nets_utils import pad_list

from utils.dprint import dprint


def gather_tensors(tensor):
    """
    Gather tensors from all GPUs.
    """
    tensor_list = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_list, tensor)
    return tensor_list


def get_avg_result(res: dict):
    new_res = {}
    for k, v in res.items():
        tensors = gather_tensors(v)
        value = sum(t.item() for t in tensors) / len(tensors)
        new_res[k] = value
    return new_res


def apply_weight_average(loss, stats, weight):
    loss = (loss * weight.type(loss.dtype)).sum()
    stats, weight = recursive_average(stats, weight, distributed=True)
    loss /= weight
    loss *= torch.distributed.get_world_size()
    return loss


class Trainer:
    def __init__(
        self,
        model,
        tr_data: SequenceIterFactory,
        cv_data: SequenceIterFactory,
        optim,
        scheduler,
        config,
        ckpt_dir,
        rank,
        logger: Logger,
        resume: str
    ):
        self.model = model
        self.tr_data = tr_data
        self.cv_data = cv_data
        self.config = config
        self.epoch_start = 0
        self.step = 0
        self.optim = optim
        self.rank = rank
        self.log_interval = config.log_interval
        self.logger = logger
        self.max_ckpt = config.max_ckpt
        self.best_field = config.best_field
        self.best_value = None
        self.best_save_type = config.best_save_type
        self.grad_clip = config.grad_clip
        self.ckpt_dir = ckpt_dir
        ###
        self.scheduler = scheduler
        self.new_bob = config.new_bob
        self.cv_log = {}
        # self.max_aux_ds = config.max_aux_ds # Maximum auxiliary audio length in seconds
        self.max_mix_ds = config.max_mix_ds
        assert config.mel_config is not None

        self.mix_process = MaxLength(['text'], max_len= int(self.max_mix_ds * 16000 / config.mel_config['hop_size']))
        self.codec_process = MaxLength(['codec'], max_len=int(self.max_mix_ds * 16000 / 640))

        self.mel_process = MelSpec(**config.mel_config)


        if resume != "":
            ## loading ckpt
            self._log(f"loading model from {resume}...")
            ckpt = torch.load(resume, map_location="cpu")
            self.model.module.load_state_dict(ckpt["model_state_dict"])
            self.optim.load_state_dict(ckpt["optim"])
            self.epoch_start = ckpt["epoch"] + 1
            self.step = ckpt["step"]
            self.cv_log = ckpt["cv_log"]
            self.best_value = ckpt[self.best_field]
            self.optim.load_state_dict(ckpt["optim"])
            self.scheduler = ckpt["scheduler"]
            self.new_bob = ckpt["new_bob"]
        
    
    def _post_process(self, _data:dict):
        """
        This process basically limits the length of mixture and codec
        """
        _data.update(self.mix_process(_data))
        _data.update(self.codec_process(_data))
        return _data

    def _train_one_batch(self, batch, data, optim, if_log) -> dict:
        uttid, _data = data

        ## Post process:
        _data_res = self._post_process(_data)

        ##  Apply Mel to data text
        # _data_res["text"], _data_res["text_lengths"] = self.mel_process.mel(
        #     _data_res["text"], _data_res["text_lengths"]
        # )
        # _data_res["aux"], _data_res["aux_lengths"] = self.mel_process.mel(
        #     _data_res["aux"], _data_res["aux_lengths"]
        # )
        # _data_res = _data
            
        data_shape = []
        for key, value in _data_res.items():
            data_shape.append(f"{key}:{value.shape}")
            _data_res[key] = value.cuda()
        hint_once(f"batch data shape {','.join(data_shape)} | text lengths {_data_res['text_lengths']}, aux lengths {_data_res['aux_lengths']} on rank {torch.distributed.get_rank()}", "data_after_shape")
        
        
        ## Process Mel Spectrogram ##
        loss, stats, weight = self.model(**_data_res)
        loss = apply_weight_average(loss, stats, weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        optim.step()
        optim.zero_grad()
        self.scheduler.step()
        torch.cuda.empty_cache()
        if if_log:
            stats["lr"] = optim.param_groups[0]["lr"]
            return stats
        return None

    def _eval_one_batch(self, data) -> dict:
        dprint("evaluating on one batch data!")
        uttid, _data = data

        # Post process:
        _data_res = self._post_process(_data)

        # #  Apply Mel to data text
        # _data_res["text"], _data_res["text_lengths"] = self.mel_process.mel(
        #     _data_res["text"], _data_res["text_lengths"]
        # )
        # _data_res["aux"], _data_res["aux_lengths"] = self.mel_process.mel(
        #     _data_res["aux"], _data_res["aux_lengths"]
        # )
        # _data_res = _data
        
        for key, value in _data_res.items():
            _data_res[key] = value.cuda()
        loss, stats, weight = self.model(**_data_res)
        loss = apply_weight_average(loss, stats, weight)
        return stats

    def _log(self, msg):
        if self.rank == 0:
            self.logger.info(msg)
        pass

    def _save(self, model, cv_log, epoch, optim, path, step, save_best: bool):
        if self.rank == 0:
            self._log(f"saving model... for epoch {epoch}")
            content = {
                "epoch": epoch,
                "step": step,
                "model_state_dict": model.module.state_dict(),
                "optim": optim.state_dict(),
                "cv_log": cv_log,
                "scheduler": self.scheduler,
                "new_bob": self.new_bob,
                self.best_field: self.best_value,
            }
            save(
                path,
                content,
                epoch,
                self.max_ckpt
            )
            if save_best:
                self._log(f"saving the best model of epoch {epoch}")
                torch.save(content, path.replace(f"epoch{epoch}.pth", f"best.pth"))
        pass

    def _train(self, optim, tr_data, epoch):
        self.model.train()
        total = int((len(tr_data) * 1) * self.config.epoch)
        start_time = time.time()
        for batch, data in enumerate(tr_data):
            if_log = batch % self.log_interval == 0
            res = self._train_one_batch(batch, data, optim, if_log)
            if if_log:
                res["epoch"] = epoch
                time_per_batch = (time.time() - start_time) / self.log_interval
                res[
                    "p"
                ] = f"[{self.step}/{total}|({str(timedelta(seconds=((total-self.step) * time_per_batch)))})]"
                res["time/batch"] = f"{time_per_batch}s"
                start_time = time.time()
                self._log(f"tr, {dict_to_str(res)}")
            self.step += 1

    def _eval(self, cv_data, epoch):
        self.model.eval()
        result = None
        if self.rank == 0:
            print(f"evaluating on cv_data of len {len(cv_data)* 1}")
        # with torch.no_grad():
        for data in cv_data:
            res = self._eval_one_batch(data)
            if result == None:
                result = res
            else:
                for key in result.keys():
                    result[key] += res[key]
        for key in result.keys():
            result[key] = result[key] / len(cv_data)
        ## gather all tensors onto the same device
        result = get_avg_result(result)
        self._log(f"eval epoch {epoch} {dict_to_str(result)}")
        if epoch != -1:
            self.cv_log[epoch] = result
        return result[self.best_field]

    def train(self):
        for epoch in range(self.epoch_start, self.config.epoch):
            self._log(f"...epoch {epoch}...")
            tr_data = self.tr_data.build_iter(epoch)
            cv_data = self.cv_data.build_iter(epoch, shuffle=False)
            ### training
            self._train(self.optim, tr_data, epoch)
            #### evaluation
            result = self._eval(cv_data, epoch)
            if self.best_value is None:
                save_best = True
                self.best_value = result
            else:
                save_best = (
                    result > self.best_value
                    if self.best_save_type == "ascend"
                    else result < self.best_value
                )
                if save_best:
                    self.best_value = result
            ### save model
            self._save(
                self.model,
                self.cv_log,
                epoch,
                self.optim,
                os.path.join(self.ckpt_dir, f"epoch{epoch}.pth"),
                self.step,
                save_best,
            )
            dist.barrier()
