# Post-organization Smoke Test

## Migration summary

- moved: 131
- copied/generated index group: 1
- kept in place: 8
- skipped: 1
- failed: 0
- retired `btuap/` artifact compatibility symlinks: 54
- preserved canonical artifact targets: 54
- retired remaining script/result compatibility symlinks: 70
- total retired compatibility symlinks: 124
- remaining symlinks: 0

## Tests

| Test | Status | Details |
|---|---|---|
| `import src` | PASS | Core package imports from original location. |
| `import btuap` | PASS | JANUS source package imports after artifact separation. |
| BTUAP default paths | PASS | Defaults point directly to canonical perturbation and SR-checkpoint directories. |
| Canonical Original JANUS | PASS | All canonical PT tensors load and contain finite values. |
| Canonical Mel-Mimic | PASS | Bird, fan, sea-waves, and bird-carrier groups load. |
| Perturbation count | PASS | 39 canonical PT perturbations validated. |
| Retired perturbation aliases | PASS | 54 temporary `btuap/` artifact links were removed after target and SHA-256 verification. |
| Intent head | PASS | Canonical checkpoint loads; one head forward is finite. |
| Result JSON | PASS | 62 JSON files parsed from canonical result tree. |
| Result CSV | PASS | 163 CSV files opened from canonical result tree. |
| Canonical results | PASS | Active code writes directly to categorized paths under `results/`. |
| Organized scripts | PASS | Canonical scripts compile and direct-path entry points load successfully. |
| Migration manifest | PASS | 131 moved destinations and required links validated. |
| Checkpoint SHA-256 | PASS | 45 original checkpoint/model hashes unchanged. |
| Symlink scan | PASS | 0 symlinks remain in the project tree. |
| FSC samples | PASS | Two 16 kHz samples and split metadata previously validated. |
| VoxCeleb1 samples | PASS | One member from each dev/test ZIP previously validated. |
| Wav2Vec2 raw state_dict | PASS | Raw weights load; hidden size 768. |
| DistilHuBERT raw state_dict | PASS | Raw weights load; hidden size 768. |
| Hydra YAML runtime load | BLOCKED | Current Python lacks `omegaconf`/PyYAML. |
| Full Wav2Vec2/DistilHuBERT object load | BLOCKED | Current Python lacks `transformers`. |
| SV checkpoint object load | BLOCKED | Current Python lacks `pytorch_lightning`. |
| Full audio → SV encoder/head | BLOCKED | Full model dependencies unavailable. |
| Full audio → SR encoder/head | BLOCKED | Full encoder dependencies unavailable; intent head forward passes. |
| Minimal joint trial | BLOCKED | Full SV/SR pipeline dependencies unavailable. |
| Full evaluation | SKIPPED | Prohibited for organization task. |

Runtime defaults, evaluation outputs, plotting outputs, and runnable command paths now point directly to their canonical directories. Model parameters, experimental parameters, dataset splits, and training logic were not changed. Historical result logs remain immutable.


## Canonical-path validation (2026-09-16)

- `btuap/config.py` defaults resolve directly to `perturbations/original/` and `models/checkpoints/sr/`.
- 54 canonical artifact targets match their pre-retirement SHA-256 hashes.
- 39 perturbation PT files load and all floating-point tensors are finite.
- The canonical intent-head checkpoint loads successfully.
- 37 physical Python files under `btuap/` and `scripts/` compile.
- Active Python/config/shell files contain 0 retired artifact paths.
- `src` and `btuap` imports pass.
- The final 70 compatibility symlinks were retired after target and hash validation; 0 symlinks remain.
- `btuap/` contains 0 PT/WAV artifact links.


## Full canonical-path activation (2026-09-17)

- Updated evaluation defaults from `paper_results/` to categorized locations under `results/`.
- Updated plotting defaults to categorized locations under `figures/`.
- Updated runnable commands to `scripts/data/`, `scripts/eval/`, and `scripts/utilities/`.
- Added direct-execution project-root bootstrapping to scripts that import project packages.
- Updated 16 Hydra evaluation entry points to resolve `../../config` from their physical locations.
- Retired 70 remaining compatibility symlinks after validating every target; 0 symlinks remain.
- Direct-path `--help` checks passed for the joint, Any-10, Kenansville, and protocol-building entry points.
- Python compilation passed for `btuap/`, `scripts/`, and the affected root training modules.
