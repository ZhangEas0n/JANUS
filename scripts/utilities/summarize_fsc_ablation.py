"""Summarize FSC Any-10 single-task and joint ablations."""

import argparse
import csv
import json
from pathlib import Path


METHODS = [
    ("SV-only perturbation", "sv_only"),
    ("SR-only perturbation", "sr_only"),
    ("JANUS", "janus"),
    ("Independent SV+SR sum", "naive_sum"),
]

FIELDS = [
    "method", "target_snr_db", "actual_snr_db", "asr_sv_any10", "asr_sr",
    "asr_joint_any10", "pair_far", "sensitive_rate",
    "any10_acceptance_rate", "frr_diagnostic", "num_queries",
    "eligible_queries",
]


def load_result(root, tag):
    path = root / tag / "fsc_any10_results.json"
    if not path.is_file():
        raise FileNotFoundError("Missing ablation result: {}".format(path))
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def clean_row(payload):
    query = payload["query_level_any10"]
    pair = payload["pair_level"]
    genuine = payload["genuine_rejection_diagnostics"]
    return {
        "method": "Clean", "target_snr_db": None, "actual_snr_db": None,
        "asr_sv_any10": None, "asr_sr": None, "asr_joint_any10": None,
        "pair_far": pair["clean_pair_far"],
        "sensitive_rate": query["clean_sensitive_rate"],
        "any10_acceptance_rate": query["clean_any10_acceptance_rate"],
        "frr_diagnostic": genuine["clean_frr"],
        "num_queries": query["num_queries"],
        "eligible_queries": query["eligible_queries"],
    }


def method_row(name, payload):
    query = payload["query_level_any10"]
    pair = payload["pair_level"]
    genuine = payload["genuine_rejection_diagnostics"]
    distortion = payload["distortion"]["test"]
    return {
        "method": name,
        "target_snr_db": payload.get("parameters", {}).get("target_snr_db"),
        "actual_snr_db": distortion["snr_db_mean"],
        "asr_sv_any10": query["asr_sv_any10"],
        "asr_sr": query["asr_sr"],
        "asr_joint_any10": query["asr_joint_any10"],
        "pair_far": pair["adv_pair_far"],
        "sensitive_rate": query["adv_sensitive_rate"],
        "any10_acceptance_rate": query["adv_any10_acceptance_rate"],
        "frr_diagnostic": genuine["adv_frr"],
        "num_queries": query["num_queries"],
        "eligible_queries": query["eligible_queries"],
    }


def validate(payloads):
    reference = payloads[0][1]
    keys = (
        reference["target_speakers"], reference["thresholds"],
        reference["query_level_any10"]["num_queries"],
        reference["query_level_any10"]["eligible_queries"],
    )
    for name, payload in payloads[1:]:
        current = (
            payload["target_speakers"], payload["thresholds"],
            payload["query_level_any10"]["num_queries"],
            payload["query_level_any10"]["eligible_queries"],
        )
        if current != keys:
            raise ValueError("Evaluation protocol mismatch for {}".format(name))


def percent(value):
    return "N/A" if value is None else "{:.2f}%".format(100.0 * float(value))


def write_markdown(rows, path):
    headers = ["Method", "ASR-SV", "ASR-SR", "ASR-Joint", "FAR", "Sensitive Rate"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        values = [
            row["method"], percent(row["asr_sv_any10"]),
            percent(row["asr_sr"]), percent(row["asr_joint_any10"]),
            percent(row["pair_far"]), percent(row["sensitive_rate"]),
        ]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    payloads = [(name, load_result(args.results_root, tag)) for name, tag in METHODS]
    validate(payloads)
    rows = [clean_row(payloads[0][1])]
    rows.extend(method_row(name, payload) for name, payload in payloads)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "fsc_any10_ablation_10db.csv"
    md_path = args.output_dir / "fsc_any10_ablation_10db.md"
    json_path = args.output_dir / "fsc_any10_ablation_10db.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(rows, md_path)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump({"snr_db": 10.0, "rows": rows}, handle, indent=2)
    print("[+] Saved {}, {}, {}".format(csv_path, md_path, json_path))


if __name__ == "__main__":
    main()
