"""Evaluate JANUS perturbations under unknown utterance start times."""

from __future__ import print_function

import argparse
import csv
import json
import subprocess
import sys
import random
import statistics
from datetime import datetime
from pathlib import Path

RESULT_FIELDS = [
    "method",
    "offset_seconds",
    "seed",
    "num_trials",
    "asr_sv",
    "asr_sr",
    "asr_joint",
    "far",
    "sensitive_rate",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint_trials_csv", type=Path)
    parser.add_argument("--delta_path", type=Path, required=True)
    parser.add_argument("--sv_only_delta_path", type=Path)
    parser.add_argument("--sr_only_delta_path", type=Path)
    parser.add_argument("--naive_delta_path", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--tau_sv", type=float)
    parser.add_argument("--t_sens", type=float, default=0.35)
    constraint = parser.add_mutually_exclusive_group()
    constraint.add_argument("--epsilon", type=float)
    constraint.add_argument("--target_snr_db", type=float)
    parser.add_argument(
        "--offset_seconds",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 1.0, 2.0],
    )
    parser.add_argument("--random_offset", action="store_true")
    parser.add_argument("--num_random_offsets", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max_trials", type=int)

    parser.add_argument("--project_root", type=Path, default=Path.cwd())
    parser.add_argument("--fsc_root", type=Path)
    parser.add_argument("--backbone", type=Path)
    parser.add_argument("--sv_ckpt", type=Path)
    parser.add_argument("--intent_head_path", type=Path)
    parser.add_argument("--sv_threshold_path", type=Path)
    parser.add_argument("--target_count", type=int, default=10)
    parser.add_argument("--enrollment_k", type=int, default=3)
    return parser.parse_args()


def load_delta_length(path):
    import torch

    payload = torch.load(str(path), map_location="cpu")
    if isinstance(payload, dict):
        for key in ("uap_delta", "delta", "perturbation", "uap"):
            if key in payload:
                payload = payload[key]
                break
    if not torch.is_tensor(payload):
        raise TypeError("Unsupported perturbation checkpoint: {}".format(path))
    return int(payload.shape[-1])


def choose_output_dir(path):
    path = path.resolve()
    if path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = path / timestamp
    path.mkdir(parents=True, exist_ok=False)
    return path


def method_paths(args):
    methods = [("JANUS", args.delta_path)]
    optional = [
        ("SV-only", args.sv_only_delta_path),
        ("SR-only", args.sr_only_delta_path),
        ("Independent SV+SR sum", args.naive_delta_path),
    ]
    methods.extend((name, path) for name, path in optional if path is not None)
    for _, path in methods:
        if not path.is_file():
            raise FileNotFoundError("Missing perturbation checkpoint: {}".format(path))
    return methods


def offsets_for_method(args, delta_path):
    if not args.random_offset:
        return [(args.seed, float(value)) for value in args.offset_seconds]

    seeds = args.seeds if args.seeds else [args.seed]
    duration = load_delta_length(delta_path) / float(args.sample_rate)
    offsets = []
    for seed in seeds:
        generator = random.Random(seed)
        for _ in range(args.num_random_offsets):
            offsets.append((seed, generator.uniform(0.0, duration)))
    return offsets


def add_override(command, key, value):
    if value is not None:
        if isinstance(value, (str, Path)):
            value = json.dumps(str(value))
        command.append("+fsc_any10_eval.{}={}".format(key, value))


def run_one(args, method, delta_path, offset, seed, output_dir, log_handle):
    slug = method.lower().replace(" ", "_").replace("+", "plus")
    run_dir = output_dir / "{}_seed{}_offset_{:.6f}".format(slug, seed, offset)
    evaluator = args.project_root / "scripts" / "eval" / "evaluate_fsc_any10.py"
    if not evaluator.is_file():
        raise FileNotFoundError("Missing evaluator: {}".format(evaluator))

    command = [sys.executable, str(evaluator)]
    add_override(command, "method", method)
    add_override(command, "uap_path", delta_path.resolve())
    add_override(command, "output_dir", run_dir)
    add_override(command, "sample_rate", args.sample_rate)
    add_override(command, "async_offset_seconds", "{:.9f}".format(offset))
    add_override(command, "device", args.device)
    add_override(command, "sr_threshold", args.t_sens)
    add_override(command, "sv_threshold", args.tau_sv)
    add_override(command, "target_snr_db", args.target_snr_db)
    add_override(command, "epsilon", args.epsilon)
    add_override(command, "max_trials", args.max_trials)
    add_override(command, "joint_trials_csv", args.joint_trials_csv)
    add_override(command, "fsc_root", args.fsc_root)
    add_override(command, "backbone", args.backbone)
    add_override(command, "sv_ckpt", args.sv_ckpt)
    add_override(command, "intent_head_path", args.intent_head_path)
    add_override(command, "sv_threshold_path", args.sv_threshold_path)
    add_override(command, "target_count", args.target_count)
    add_override(command, "enrollment_k", args.enrollment_k)

    header = "\n[RUN] {}\n".format(" ".join(str(item) for item in command))
    print(header.strip())
    log_handle.write(header)
    log_handle.flush()
    process = subprocess.Popen(
        command,
        cwd=str(args.project_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
    )
    for line in process.stdout:
        print(line, end="")
        log_handle.write(line)
    return_code = process.wait()
    log_handle.flush()
    if return_code != 0:
        raise RuntimeError("Evaluation failed with exit code {}".format(return_code))

    result_path = run_dir / "fsc_any10_results.json"
    with result_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    query = payload["query_level_any10"]
    return {
        "method": method,
        "offset_seconds": offset,
        "seed": seed,
        "num_trials": query["num_queries"],
        "asr_sv": query["asr_sv_any10"],
        "asr_sr": query["asr_sr"],
        "asr_joint": query["asr_joint_any10"],
        "far": query["adv_any10_acceptance_rate"],
        "sensitive_rate": query["adv_sensitive_rate"],
    }, query


def plot_asr(rows, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for metric, label, color in (
        ("asr_sv", "ASR-SV", "#4C78A8"),
        ("asr_sr", "ASR-SR", "#59A14F"),
        ("asr_joint", "ASR-Joint", "#E15759"),
    ):
        selected = sorted(
            [row for row in rows if row["method"] == "JANUS"],
            key=lambda row: row["offset_seconds"],
        )
        plt.plot(
            [row["offset_seconds"] for row in selected],
            [row[metric] for row in selected],
            marker="o",
            label=label,
            color=color,
        )
    plt.title("JANUS under temporal misalignment")
    plt.xlabel("Offset (seconds)")
    plt.ylabel("Attack success rate")
    plt.ylim(0.0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=300)
    plt.close()


def plot_methods(rows, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = sorted(set(row["method"] for row in rows))
    for method in methods:
        selected = sorted(
            [row for row in rows if row["method"] == method],
            key=lambda row: row["offset_seconds"],
        )
        plt.plot(
            [row["offset_seconds"] for row in selected],
            [row["asr_joint"] for row in selected],
            marker="o",
            label=method,
        )
    plt.title("Joint success under temporal misalignment")
    plt.xlabel("Offset (seconds)")
    plt.ylabel("ASR-Joint")
    plt.ylim(0.0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=300)
    plt.close()


def write_readme(path):
    path.write_text(
        """# Unknown Start Time Evaluation

This experiment measures sensitivity to temporal misalignment. An offset of
0 seconds is synchronized injection; positive offsets circularly shift the
perturbation relative to the utterance start. Stable ASR-Joint across offsets
indicates that JANUS does not require strict start-time synchronization. A
substantial decline indicates temporal sensitivity and motivates random-start
or shift-invariant training. FAR is the query-level Any-10 FAR.
""",
        encoding="utf-8",
    )


def aggregate_results(rows):
    summary = {}
    metrics = ("asr_sv", "asr_sr", "asr_joint", "far", "sensitive_rate")
    for method in sorted(set(row["method"] for row in rows)):
        selected = [row for row in rows if row["method"] == method]
        method_summary = {"num_runs": len(selected)}
        for metric in metrics:
            values = [float(row[metric]) for row in selected]
            method_summary[metric + "_mean"] = statistics.mean(values)
            method_summary[metric + "_std"] = (
                statistics.pstdev(values) if len(values) > 1 else 0.0
            )
        summary[method] = method_summary
    return summary


def main():
    args = parse_args()
    args.project_root = args.project_root.resolve()
    if args.sample_rate <= 0 or args.num_random_offsets <= 0:
        raise ValueError("sample_rate and num_random_offsets must be positive")
    if any(value < 0 for value in args.offset_seconds):
        raise ValueError("offset_seconds values must be non-negative")
    if args.joint_trials_csv is not None and not args.joint_trials_csv.is_file():
        raise FileNotFoundError("Missing joint trial CSV: {}".format(args.joint_trials_csv))

    output_dir = choose_output_dir(args.output_dir)
    rows = []
    clean_by_offset = {}
    log_path = output_dir / "async_injection.log"
    with log_path.open("w", encoding="utf-8") as log_handle:
        for method, delta_path in method_paths(args):
            for seed, offset in offsets_for_method(args, delta_path):
                row, query = run_one(
                    args, method, delta_path, offset, seed, output_dir, log_handle
                )
                rows.append(row)
                clean_by_offset[(seed, offset)] = {
                    "method": "Clean",
                    "offset_seconds": offset,
                    "seed": seed,
                    "num_trials": query["num_queries"],
                    "asr_sv": None,
                    "asr_sr": None,
                    "asr_joint": None,
                    "far": query["clean_any10_acceptance_rate"],
                    "sensitive_rate": query["clean_sensitive_rate"],
                }

    csv_path = output_dir / "async_injection_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(list(clean_by_offset.values()) + rows)

    summary = {
        "config": {
            "sample_rate": args.sample_rate,
            "tau_sv": args.tau_sv,
            "t_sens": args.t_sens,
            "epsilon": args.epsilon,
            "target_snr_db": args.target_snr_db,
            "random_offset": args.random_offset,
            "num_random_offsets": args.num_random_offsets,
            "seed": args.seed,
            "seeds": args.seeds,
            "joint_trials_csv": None
            if args.joint_trials_csv is None
            else str(args.joint_trials_csv),
        },
        "aggregate_over_offsets": aggregate_results(rows),
        "results": list(clean_by_offset.values()) + rows,
    }
    with (output_dir / "async_injection_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)

    plot_asr(rows, output_dir / "async_asr_vs_offset.png")
    if len(method_paths(args)) > 1:
        plot_methods(rows, output_dir / "async_methods_asr_joint.png")
    write_readme(output_dir / "README_async_injection.md")
    print("[+] Async injection results saved to {}".format(output_dir))


if __name__ == "__main__":
    main()
