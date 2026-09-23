"""Summarize aligned FSC Any-N evaluation JSON files."""

from __future__ import print_function

import argparse
import csv
import json
from pathlib import Path


FIELDS = (
    "method",
    "target_snr_db",
    "actual_snr_db",
    "linf_mean",
    "rms_mean",
    "num_queries",
    "eligible_queries",
    "asr_sv_any10",
    "asr_sr",
    "asr_joint_any10",
    "clean_any10_acceptance_rate",
    "adv_any10_acceptance_rate",
    "clean_pair_far",
    "adv_pair_far",
    "asr_sv_pair",
    "asr_sr_pair",
    "asr_joint_pair",
    "clean_frr",
    "adv_frr",
    "conditional_genuine_rejection_rate",
)


def parse_result(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use NAME=PATH for every --result value.")
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("Both NAME and PATH must be non-empty.")
    return name.strip(), Path(path.strip())


def load_row(name, path):
    if not path.is_file():
        raise FileNotFoundError("Missing result JSON: {}".format(path))
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    distortion = payload["distortion"]["test"]
    query = payload["query_level_any10"]
    pair = payload["pair_level"]
    genuine = payload["genuine_rejection_diagnostics"]
    parameters = payload.get("parameters", {})
    return {
        "method": name,
        "target_snr_db": parameters.get("target_snr_db"),
        "actual_snr_db": distortion["snr_db_mean"],
        "linf_mean": distortion["linf_mean"],
        "rms_mean": distortion["rms_mean"],
        "num_queries": query["num_queries"],
        "eligible_queries": query["eligible_queries"],
        "asr_sv_any10": query["asr_sv_any10"],
        "asr_sr": query["asr_sr"],
        "asr_joint_any10": query["asr_joint_any10"],
        "clean_any10_acceptance_rate": query["clean_any10_acceptance_rate"],
        "adv_any10_acceptance_rate": query["adv_any10_acceptance_rate"],
        "clean_pair_far": pair["clean_pair_far"],
        "adv_pair_far": pair["adv_pair_far"],
        "asr_sv_pair": pair["asr_sv_pair"],
        "asr_sr_pair": pair["asr_sr_pair"],
        "asr_joint_pair": pair["asr_joint_pair"],
        "clean_frr": genuine["clean_frr"],
        "adv_frr": genuine["adv_frr"],
        "conditional_genuine_rejection_rate": genuine[
            "conditional_genuine_rejection_rate"
        ],
    }


def percent(value):
    return "{:.2f}%".format(100.0 * float(value))


def write_markdown(rows, path):
    headers = (
        "Method",
        "SNR",
        "ASR-SV Any10",
        "ASR-SR",
        "ASR-Joint",
        "Adv. pair FAR",
        "Adv. FRR",
    )
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        values = (
            row["method"],
            "{:.2f} dB".format(row["actual_snr_db"]),
            percent(row["asr_sv_any10"]),
            percent(row["asr_sr"]),
            percent(row["asr_joint_any10"]),
            percent(row["adv_pair_far"]),
            percent(row["adv_frr"]),
        )
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        type=parse_result,
        help="Named result in NAME=PATH form; repeat for each method.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--output_prefix",
        default=None,
        help="Output filename prefix; inferred from a shared target SNR by default.",
    )
    args = parser.parse_args()

    rows = [load_row(name, path) for name, path in args.result]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = args.output_prefix
    if output_prefix is None:
        target_snrs = {row["target_snr_db"] for row in rows}
        if len(target_snrs) == 1 and None not in target_snrs:
            snr = next(iter(target_snrs))
            output_prefix = "fsc_any10_{:g}db_comparison".format(float(snr))
        else:
            output_prefix = "fsc_any10_comparison"
    csv_path = args.output_dir / (output_prefix + ".csv")
    markdown_path = args.output_dir / (output_prefix + ".md")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(rows, markdown_path)
    print("[+] Saved {}".format(csv_path))
    print("[+] Saved {}".format(markdown_path))


if __name__ == "__main__":
    main()
