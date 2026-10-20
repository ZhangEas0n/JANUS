# Storage Report

Sizes can overlap where a file serves multiple roles (for example a checkpoint under logs).

| Category | Files | Size |
|---|---:|---:|
| datasets | 30,068 | 33.49 GiB |
| pretrained models | 10 | 449.81 MiB |
| checkpoints | 4 | 1.05 GiB |
| perturbations | 53 | 9.64 MiB |
| results | 276 | 219.27 MiB |
| recordings | 0 | 0.00 B |
| logs | 74 | 3.78 MiB |
| cache/temp | 854 | 28.19 MiB |

## Largest retained items

- `voxceleb_archives/vox1_dev_wav.zip`: ~30.4 GiB
- `model/wav2vec_base.ckpt`: ~1.04 GiB
- `voxceleb_archives/vox1_test_wav.zip`: ~1.04 GiB
- `model/wav2vec2-base-960h/pytorch_model.bin`: ~360 MiB
- `model/distilhubert/pytorch_model.bin`: ~89.6 MiB

Potential cleanup is documented separately; nothing was deleted.
