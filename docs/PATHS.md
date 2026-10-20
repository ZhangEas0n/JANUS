# Paths

| Resource | Actual path | Status |
|---|---|---|
| Project root | `/root/autodl-tmp/btuap-backup-main` | active |
| VoxCeleb1 | `/root/autodl-tmp/btuap-backup-main/voxceleb_archives` | PASS sample read |
| VoxCeleb metadata | `/root/autodl-tmp/btuap-backup-main/voxceleb_meta` | present |
| FSC | `/root/autodl-tmp/btuap-backup-main/fluent_speech_commands_dataset` | PASS sample read |
| MUSAN | not found | SKIPPED |
| Environment sounds | `/root/autodl-tmp/btuap-backup-main/data/environmental_sounds` | PASS |
| Wav2Vec2-base | `/root/autodl-tmp/btuap-backup-main/model/wav2vec2-base-960h` | raw weights PASS |
| Wav2Vec2-large | not found | SKIPPED |
| HuBERT-base | not found | SKIPPED |
| WavLM-base | not found | SKIPPED |
| DistilHuBERT | `/root/autodl-tmp/btuap-backup-main/model/distilhubert` | raw weights PASS |
| SV checkpoint root | `/root/autodl-tmp/btuap-backup-main/model` | retained |
| SR head/checkpoint | `/root/autodl-tmp/btuap-backup-main/models/checkpoints/sr/intent_head_fsc.pt` | PASS |
| Perturbations | `/root/autodl-tmp/btuap-backup-main/perturbations` | canonical; active code uses this path |
| Results | `/root/autodl-tmp/btuap-backup-main/results` | canonical; active code writes here |
| Recordings | `/root/autodl-tmp/btuap-backup-main/recordings` | no OTA recordings found |
| Physical audio evaluator | `/root/autodl-tmp/btuap-backup-main/experiments/eval_physical_audio.py` | implemented; single and batch smoke tests PASS |
| Physical audio smoke outputs | `/root/autodl-tmp/btuap-backup-main/results/physical_ota/smoke_test` | digital WAV smoke test only; not an OTA recording |
| Logs | `/root/autodl-tmp/btuap-backup-main/logs` | canonical; historical logs retained |

`config/` remains the authoritative Hydra configuration directory. `configs/` is reserved for future path governance; it is not wired into runtime code to avoid broad changes.
