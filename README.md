## LauraTSE

Official code for LauraTSE: Target Speaker Extraction using Decoder-only Language Models

## Before you start!!

### !! Please note that this repository does not serve the purpose for readability and external usage as the code is mainly for my deployment on the server. You can read it if you want but it is by no means well-structured !!


Note that this code contains static config on the server

- [ ] add `spk_dict_path` in the config
- [ ] Rewrite Trainer (no need to monitor maximum length and the mel proc, dataloader directly returns mel spectrogram)
- [ ] Note that the `train` and `valid` ref path, the `path` should be the clean_scp (for compatibility reasons)
- [ ] All the references and mixtures are normalized to [-1,1], make sure to adjust the inference script to normalize first

