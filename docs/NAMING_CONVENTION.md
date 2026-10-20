# Naming Convention

These rules apply to new artifacts only. Historical files are not renamed.

- Checkpoint: `<backbone>_<task>_<YYYYMMDD>.ckpt`
- Perturbation: `janus_<type>_<backbone>_<extra>.pt`
- Result: `<experiment>_<model>_<YYYYMMDD>.csv`
- Log: `<experiment>_<phase>_<YYYYMMDD-HHMMSS>.log`
- OTA recording: `<device>_<room>_<distance>_<take>.wav`

Use lowercase ASCII, underscores as separators, explicit units in tokens, and keep one stable run identifier across config, checkpoint, perturbation, result, and log names.
