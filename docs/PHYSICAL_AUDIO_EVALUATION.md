# Physical Audio SV + SR Evaluation

## Entry point

`experiments/eval_physical_audio.py` evaluates an already-recorded audio file. It does not control playback or recording.

Single recording:

```bash
/root/autodl-tmp/btuap-env/bin/python experiments/eval_physical_audio.py \
  --audio /path/to/recorded.wav \
  --enrollment /path/to/enrollment.wav \
  --output results/physical_ota/result.json \
  --device cuda
```

Use repeated `--enrollment` options or `--enrollment-list` for Any-N enrollment. Ten enrollment files are reported as `Any-10`, and the highest frozen-threshold score determines acceptance.

Batch evaluation:

```bash
/root/autodl-tmp/btuap-env/bin/python experiments/eval_physical_audio.py \
  --audio-dir /path/to/recordings \
  --metadata /path/to/trials.csv \
  --max-files 50 \
  --enrollment-list /path/to/enrollments.csv \
  --output results/physical_ota/results.csv \
  --device cuda
```

The batch command also writes `<output_stem>_summary.json`. Conditional ASR fields are only calculated when a metadata row has `clean_audio_path`; otherwise the summary explicitly sets `conditional_asr_available` to false.

## Frozen evaluation protocol

- SV: Wav2Vec2 speaker encoder from `model/wav2vec_base.ckpt`, loaded from the exact inference state at `models/checkpoints/sv/wav2vec_base_inference_shards/index.json`.
- SV score: scaled cosine in `[0, 1]`.
- SV threshold: `0.6631416827452248`.
- SR: `SimpleSensitiveIntentHead` at `models/checkpoints/sr/intent_head_fsc.pt`.
- SR mode: binary sensitive intent.
- SR threshold: `0.35`.
- Threshold source: `results/joint/evaluation/fsc_joint_thresholds.json`.
- Joint decision: `sv_accepted AND is_sensitive`.

Input audio is converted to mono, resampled to 16 kHz, and normalized with the existing project pipeline. No denoising, source separation, threshold tuning, or training is performed.

## Smoke test status

Single JSON and batch CSV/summary output passed on one existing FSC WAV. This verifies the evaluation entry point but is not a physical OTA result. The server currently has no microphone re-recording under `recordings/`.

## Fifty-file folder workflow

Put the test recordings under `recordings/physical_batch_50/`. Folder mode scans supported audio files recursively in sorted path order. Use `--max-files 50` to prevent extra files from being evaluated.

When every recording uses the same enrollment:

```bash
/root/autodl-tmp/btuap-env/bin/python experiments/eval_physical_audio.py   --audio-dir recordings/physical_batch_50   --max-files 50   --enrollment recordings/enrollment/enrollment.wav   --output results/physical_ota/batch_50/results.csv   --log results/physical_ota/batch_50/evaluation.log   --device cuda
```

Use `--enrollment-list` for a shared Any-10 gallery. If recordings require different enrollment audio, provide a metadata CSV with an `enrollment_path` value for each row.
