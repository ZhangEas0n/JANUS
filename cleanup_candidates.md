# Cleanup Candidates

Report only. No file was deleted. Manual confirmation and a fresh dependency check are required before cleanup.

| Candidate | Files | Size | Examples |
|---|---:|---:|---|
| IDE metadata | 5 | 1.14 KiB | `.idea/.gitignore`, `.idea/MarsCodeWorkspaceAppSettings.xml`, `.idea/advpre_sv.iml` |
| Local .venv (appears Windows-oriented) | 767 | 27.80 MiB | `.venv/Lib/site-packages/antlr4/BufferedTokenStream.py`, `.venv/Lib/site-packages/antlr4/CommonTokenFactory.py`, `.venv/Lib/site-packages/antlr4/CommonTokenStream.py` |
| Python bytecode/cache | 82 | 394.30 KiB | `btuap/__pycache__/__init__.cpython-38.pyc`, `btuap/__pycache__/common.cpython-38.pyc`, `btuap/__pycache__/config.cpython-38.pyc` |
| Historical/launch logs | 81 | 5.58 MiB | `logs/wav2vec_speaker_identification/run/2026-09-11_10-43-02/064c45f9cfb34d2c99cf408235d6d129/uap_fsc_joint.log`, `logs/wav2vec_speaker_identification/run/2026-09-11_10-44-19/3736b7e7b5654775908ba43ce859f170/uap_fsc_joint.log`, `logs/wav2vec_speaker_identification/run/2026-09-11_10-50-44/349c707c4097420a9a84a417cfcfcacd/uap_fsc_joint.log` |
| Source/config backup copies | 4 | 71.14 KiB | `archive/source_backups/train_fsc_joint_uap_experiment.py.before_carrier_20260916`, `archive/source_backups/train_fsc_joint_uap_experiment.py.before_melmimic_20260915`, `archive/source_backups/train_fsc_joint_uap_experiment.py.before_rms_balance_20260916` |

Hugging Face cache at `/root/.cache/huggingface` is currently empty. Empty directories such as `tmp/` and `pretrained_models/` were retained.
