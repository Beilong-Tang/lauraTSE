# from nemo.collections.asr.models import EncDecCTCModel
import random
import torch.nn as nn
import torch
import nemo.collections.asr as nemo_asr
class ConformerEncoder(nemo_asr.models.EncDecCTCModelBPE):
    

    def _forward_internal(
        conformer, audio_signal, length, cache_last_channel=None, cache_last_time=None, cache_last_channel_len=None, out_layer = None
    ):
        """
        Modified from
        """
        if length is None:
            length = audio_signal.new_full(
                (audio_signal.size(0),), audio_signal.size(-1), dtype=torch.int64, device=audio_signal.device
            )

        # select a random att_context_size with the distribution specified by att_context_probs during training
        # for non-validation cases like test, validation or inference, it uses the first mode in self.att_context_size
        if conformer.training and len(conformer.att_context_size_all) > 1:
            cur_att_context_size = random.choices(conformer.att_context_size_all, weights=conformer.att_context_probs)[0]
        else:
            cur_att_context_size = conformer.att_context_size

        audio_signal = torch.transpose(audio_signal, 1, 2)

        if isinstance(conformer.pre_encode, nn.Linear):
            audio_signal = conformer.pre_encode(audio_signal)
        else:
            audio_signal, length = conformer.pre_encode(x=audio_signal, lengths=length)
            length = length.to(torch.int64)
            # self.streaming_cfg is set by setup_streaming_cfg(), called in the init
            if conformer.streaming_cfg.drop_extra_pre_encoded > 0 and cache_last_channel is not None:
                audio_signal = audio_signal[:, conformer.streaming_cfg.drop_extra_pre_encoded :, :]
                length = (length - conformer.streaming_cfg.drop_extra_pre_encoded).clamp(min=0)

        if conformer.reduction_position is not None and cache_last_channel is not None:
            raise ValueError("Caching with reduction feature is not supported yet!")

        max_audio_length = audio_signal.size(1)
        if cache_last_channel is not None:
            cache_len = conformer.streaming_cfg.last_channel_cache_size
            cache_keep_size = max_audio_length - conformer.streaming_cfg.cache_drop_size
            max_audio_length = max_audio_length + cache_len
            padding_length = length + cache_len
            offset = torch.neg(cache_last_channel_len) + cache_len
        else:
            padding_length = length
            cache_last_channel_next = None
            cache_len = 0
            offset = None

        audio_signal, pos_emb = conformer.pos_enc(x=audio_signal, cache_len=cache_len)

        # Create the self-attention and padding masks
        pad_mask, att_mask = conformer._create_masks(
            att_context_size=cur_att_context_size,
            padding_length=padding_length,
            max_audio_length=max_audio_length,
            offset=offset,
            device=audio_signal.device,
        )

        if cache_last_channel is not None:
            pad_mask = pad_mask[:, cache_len:]
            if att_mask is not None:
                att_mask = att_mask[:, cache_len:]
            # Convert caches from the tensor to list
            cache_last_time_next = []
            cache_last_channel_next = []

        for lth, (drop_prob, layer) in enumerate(zip(conformer.layer_drop_probs, conformer.layers)):
            original_signal = audio_signal
            if cache_last_channel is not None:
                cache_last_channel_cur = cache_last_channel[lth]
                cache_last_time_cur = cache_last_time[lth]
            else:
                cache_last_channel_cur = None
                cache_last_time_cur = None
            audio_signal = layer(
                x=audio_signal,
                att_mask=att_mask,
                pos_emb=pos_emb,
                pad_mask=pad_mask,
                cache_last_channel=cache_last_channel_cur,
                cache_last_time=cache_last_time_cur,
            )

            if cache_last_channel_cur is not None:
                (audio_signal, cache_last_channel_cur, cache_last_time_cur) = audio_signal
                cache_last_channel_next.append(cache_last_channel_cur)
                cache_last_time_next.append(cache_last_time_cur)

            # applying stochastic depth logic from https://arxiv.org/abs/2102.03216
            if conformer.training and drop_prob > 0.0:
                should_drop = torch.rand(1) < drop_prob
                # adjusting to match expectation
                if should_drop:
                    # that's not efficient, but it's hard to implement distributed
                    # version of dropping layers without deadlock or random seed meddling
                    # so multiplying the signal by 0 to ensure all weights get gradients
                    audio_signal = audio_signal * 0.0 + original_signal
                else:
                    # not doing this operation if drop prob is 0 as it's identity in that case
                    audio_signal = (audio_signal - original_signal) / (1.0 - drop_prob) + original_signal

            if conformer.reduction_position == lth:
                audio_signal, length = conformer.reduction_subsampling(x=audio_signal, lengths=length)
                max_audio_length = audio_signal.size(1)
                # Don't update the audio_signal here because then it will again scale the audio_signal
                # and cause an increase in the WER
                _, pos_emb = conformer.pos_enc(x=audio_signal, cache_len=cache_len)
                pad_mask, att_mask = conformer._create_masks(
                    att_context_size=cur_att_context_size,
                    padding_length=length,
                    max_audio_length=max_audio_length,
                    offset=offset,
                    device=audio_signal.device,
                )

            # saving tensors if required for interctc loss
            if conformer.is_access_enabled(getattr(conformer, "model_guid", None)):
                if conformer.interctc_capture_at_layers is None:
                    conformer.interctc_capture_at_layers = conformer.access_cfg.get('interctc', {}).get('capture_layers', [])
                if lth in conformer.interctc_capture_at_layers:
                    lth_audio_signal = audio_signal
                    if conformer.out_proj is not None:
                        lth_audio_signal = conformer.out_proj(audio_signal)
                    # shape is the same as the shape of audio_signal output, i.e. [B, D, T]
                    conformer.register_accessible_tensor(
                        name=f'interctc/layer_output_{lth}', tensor=torch.transpose(lth_audio_signal, 1, 2)
                    )
                    conformer.register_accessible_tensor(name=f'interctc/layer_length_{lth}', tensor=length)
            

            ##########################
            ##  Output audio signal ##
            ##########################
            if lth == out_layer:
                return audio_signal, length

        if conformer.out_proj is not None:
            audio_signal = conformer.out_proj(audio_signal)

        # Reduction
        if conformer.reduction_position == -1:
            audio_signal, length = conformer.reduction_subsampling(x=audio_signal, lengths=length)

        audio_signal = torch.transpose(audio_signal, 1, 2)
        length = length.to(dtype=torch.int64)

        if cache_last_channel is not None:
            cache_last_channel_next = torch.stack(cache_last_channel_next, dim=0)
            cache_last_time_next = torch.stack(cache_last_time_next, dim=0)
            return (
                audio_signal,
                length,
                cache_last_channel_next,
                cache_last_time_next,
                torch.clamp(cache_last_channel_len + cache_keep_size, max=cache_len),
            )
        else:
            return audio_signal, length

    def _conformer_encode(self, audio_signal, length, layer):
        """
        layer: starting from 0 
        """
        return ConformerEncoder._forward_internal(self.encoder, audio_signal = audio_signal, length = length, out_layer= layer )
    
    def _set_layer(self, layer:int):
        self.layer = layer
    
    def init_from_pretrained(path, enc_layer):
        """
        Create a Nemo ConformerEncoder instance from a path
        """
        model:ConformerEncoder = ConformerEncoder.from_pretrained(path)
        model.preprocessor.featurizer.dither = 0.0 
        model.preprocessor.featurizer.pad_to = 0
        model._set_layer(enc_layer)
        model.eval()
        return model
    
    def encode(self, input_signal=None, input_signal_length=None, processed_signal=None, processed_signal_length=None):
        """
        Forward pass of the model.

        Args:
            input_signal: Tensor that represents a batch of raw audio signals,
                of shape [B, T]. T here represents timesteps, with 1 second of audio represented as
                `self.sample_rate` number of floating point values.
            input_signal_length: Vector of length B, that contains the individual lengths of the audio
                sequences.
            processed_signal: Tensor that represents a batch of processed audio signals,
                of shape (B, D, T) that has undergone processing via some DALI preprocessor.
            processed_signal_length: Vector of length B, that contains the individual lengths of the
                processed audio sequences.

        Returns:
            A tuple of 2 elements -
            1) The encoded tensor of shape [B, T, D].
            2) The lengths of the encoded sequence, of shape [B].
        """
        #####################
        try:
            layer = self.layer
        except:
            raise Exception("Please call set_layer(layer) to specify the output layer")
        #####################

        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) == False:
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal,
                length=input_signal_length,
            )

        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        encoder_output = self._conformer_encode(audio_signal=processed_signal, length=processed_signal_length, layer = layer)
        encoded = encoder_output[0]
        encoded_len = encoder_output[1]
        return encoded, encoded_len

    def get_num_params(self):
        def _cal_param(m):
            return sum([p.numel() for p in m.parameters()])
        
        try:
            layer = self.layer
        except:
            raise Exception("Please call set_layer(layer) to specify the output layer")

        total_num = 0 
        total_num += _cal_param(self.encoder.pre_encode)
        total_num += _cal_param(self.encoder.pos_enc)
        for i, l in enumerate(self.encoder.layers):
            if i > layer:
                break
            total_num += _cal_param(l)
        return total_num
