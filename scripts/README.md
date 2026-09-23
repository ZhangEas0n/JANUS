# Scripts

Standalone scripts are grouped by purpose. Run these paths from the project root.

- `eval/`: evaluation entry points
- `data/`: protocol, label, and audio generation helpers
- `utilities/`: plotting, checkpoint combination, and result summaries
- `train/`: reserved for future self-contained training entry points

Examples:

```bash
python scripts/eval/evaluate_fsc_joint.py
python scripts/data/build_fsc_joint_protocol.py --help
python scripts/utilities/plot_quality_comparison.py
```

Evaluation scripts resolve the project package and Hydra configuration from their canonical locations. Core Hydra training entry points remain at the project root.
