#!/usr/bin/env python3
"""Download, extract, and validate the Fluent Speech Commands dataset.

The official Fluent Speech Commands download is often distributed through a
browser-mediated link. This script therefore supports both direct URLs and
already-downloaded archives.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import pathlib
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import wave
import zipfile


DATASET_DIRNAME = "fluent_speech_commands_dataset"
CSV_NAMES = ("train_data.csv", "valid_data.csv", "test_data.csv")
EXPECTED_WAVS = 30043


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download/extract and validate Fluent Speech Commands."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--url",
        help="Direct archive URL. If the official site gives you a signed URL, paste it here.",
    )
    source.add_argument(
        "--archive",
        type=pathlib.Path,
        help="Path to a manually downloaded .zip/.tar/.tar.gz/.tgz archive.",
    )
    parser.add_argument(
        "--dest",
        type=pathlib.Path,
        default=None,
        help=(
            "Output dataset directory. Defaults to "
            "$DATA_FOLDER/fluent_speech_commands_dataset if DATA_FOLDER is set, "
            "otherwise ./fluent_speech_commands_dataset."
        ),
    )
    parser.add_argument(
        "--checksum",
        help="Optional SHA256 checksum for the archive.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing destination directory.",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep the downloaded archive instead of deleting the temporary copy.",
    )
    parser.add_argument(
        "--skip-audio-check",
        action="store_true",
        help="Only check file existence/counts; do not open every wav file.",
    )
    parser.add_argument(
        "--expected-wavs",
        type=int,
        default=EXPECTED_WAVS,
        help=f"Expected wav count. Default: {EXPECTED_WAVS}.",
    )
    return parser.parse_args()


def default_dest() -> pathlib.Path:
    data_folder = os.environ.get("DATA_FOLDER")
    if data_folder:
        return pathlib.Path(data_folder) / DATASET_DIRNAME
    return pathlib.Path.cwd() / DATASET_DIRNAME


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, output_dir: pathlib.Path) -> pathlib.Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    url_path = urllib.request.urlparse(url).path
    filename = pathlib.Path(url_path).name or "fluent_speech_commands_dataset.archive"
    archive = output_dir / filename
    part = archive.with_suffix(archive.suffix + ".part")

    headers = {}
    existing = part.stat().st_size if part.exists() else 0
    if existing:
        headers["Range"] = f"bytes={existing}-"
        print(f"Resuming download at byte {existing}: {url}")
    else:
        print(f"Downloading: {url}")

    request = urllib.request.Request(url, headers=headers)
    mode = "ab" if existing else "wb"
    try:
        with urllib.request.urlopen(request) as response, part.open(mode) as handle:
            total = response.headers.get("Content-Length")
            total_text = f" ({total} bytes remaining)" if total else ""
            print(f"Writing to {part}{total_text}")
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
    except urllib.error.HTTPError as exc:
        if existing and exc.code == 416:
            print("Remote reports the partial file is already complete.")
        else:
            raise

    part.replace(archive)
    return archive


def ensure_safe_path(base: pathlib.Path, member_name: str) -> pathlib.Path:
    target = (base / member_name).resolve()
    base_resolved = base.resolve()
    if base_resolved != target and base_resolved not in target.parents:
        raise RuntimeError(f"Refusing to extract unsafe archive member: {member_name}")
    return target


def extract_archive(archive: pathlib.Path, extract_dir: pathlib.Path) -> None:
    print(f"Extracting {archive} ...")
    suffixes = "".join(archive.suffixes).lower()
    if suffixes.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                ensure_safe_path(extract_dir, info.filename)
            zf.extractall(extract_dir)
    elif (
        suffixes.endswith(".tar")
        or suffixes.endswith(".tar.gz")
        or suffixes.endswith(".tgz")
        or suffixes.endswith(".tar.bz2")
        or suffixes.endswith(".tbz2")
        or suffixes.endswith(".tar.xz")
        or suffixes.endswith(".txz")
    ):
        with tarfile.open(archive) as tf:
            for member in tf.getmembers():
                ensure_safe_path(extract_dir, member.name)
            tf.extractall(extract_dir)
    else:
        raise RuntimeError(f"Unsupported archive type: {archive}")


def find_dataset_root(extract_dir: pathlib.Path) -> pathlib.Path:
    candidates = []
    for path in [extract_dir, *extract_dir.rglob("*")]:
        if not path.is_dir():
            continue
        if (path / "data").is_dir() and (path / "wavs" / "speakers").is_dir():
            candidates.append(path)
    if not candidates:
        raise RuntimeError(
            "Could not find a Fluent Speech Commands root containing data/ and wavs/speakers/."
        )
    candidates.sort(key=lambda p: len(p.parts))
    return candidates[0]


def replace_dest(src: pathlib.Path, dest: pathlib.Path, force: bool) -> None:
    if dest.exists():
        if not force:
            raise RuntimeError(f"Destination already exists: {dest}. Use --force to replace it.")
        print(f"Replacing existing destination: {dest}")
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest))


def validate_dataset(root: pathlib.Path, expected_wavs: int, skip_audio_check: bool) -> None:
    print(f"Validating dataset at {root}")
    missing_paths = []
    csv_rows = 0

    for csv_name in CSV_NAMES:
        csv_path = root / "data" / csv_name
        if not csv_path.exists():
            raise RuntimeError(f"Missing CSV file: {csv_path}")
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if "path" not in (reader.fieldnames or []):
                raise RuntimeError(f"CSV has no 'path' column: {csv_path}")
            for row in reader:
                csv_rows += 1
                wav_path = root / row["path"]
                if not wav_path.exists():
                    missing_paths.append(wav_path)

    if missing_paths:
        preview = "\n".join(str(path) for path in missing_paths[:20])
        raise RuntimeError(
            f"{len(missing_paths)} CSV-referenced wav files are missing. First entries:\n{preview}"
        )

    wav_paths = sorted((root / "wavs" / "speakers").glob("*/*.wav"))
    wav_count = len(wav_paths)
    print(f"CSV rows: {csv_rows}")
    print(f"WAV files: {wav_count}")

    if csv_rows != expected_wavs:
        raise RuntimeError(f"Expected {expected_wavs} CSV rows, got {csv_rows}.")
    if wav_count != expected_wavs:
        raise RuntimeError(f"Expected {expected_wavs} wav files, got {wav_count}.")

    if skip_audio_check:
        print("Skipping per-file wav readability check.")
        return

    bad = []
    for index, wav_path in enumerate(wav_paths, start=1):
        try:
            with wave.open(str(wav_path), "rb") as wav:
                channels = wav.getnchannels()
                sample_rate = wav.getframerate()
                frames = wav.getnframes()
                if channels != 1 or sample_rate != 16000 or frames <= 0:
                    bad.append((wav_path, f"{channels}ch {sample_rate}Hz {frames}frames"))
        except Exception as exc:  # noqa: BLE001 - report any unreadable audio file.
            bad.append((wav_path, repr(exc)))
        if index % 1000 == 0:
            print(f"Checked {index}/{wav_count} wav files...")

    if bad:
        preview = "\n".join(f"{path}: {reason}" for path, reason in bad[:20])
        raise RuntimeError(f"{len(bad)} bad wav files found. First entries:\n{preview}")

    print("All wav files are readable 16 kHz mono audio.")


def main() -> int:
    args = parse_args()
    dest = (args.dest or default_dest()).resolve()

    with tempfile.TemporaryDirectory(prefix="fsc_download_") as tmp:
        tmp_dir = pathlib.Path(tmp)
        archive = args.archive.resolve() if args.archive else download(args.url, tmp_dir)
        if not archive.exists():
            raise RuntimeError(f"Archive does not exist: {archive}")

        if args.checksum:
            actual = sha256(archive)
            if actual.lower() != args.checksum.lower():
                raise RuntimeError(
                    f"SHA256 mismatch for {archive}\nExpected: {args.checksum}\nActual:   {actual}"
                )
            print("SHA256 checksum OK.")

        extract_dir = tmp_dir / "extracted"
        extract_dir.mkdir()
        extract_archive(archive, extract_dir)
        dataset_root = find_dataset_root(extract_dir)
        validate_dataset(dataset_root, args.expected_wavs, args.skip_audio_check)

        replace_dest(dataset_root, dest, args.force)
        print(f"Ready: {dest}")

        if args.keep_archive and not args.archive:
            kept = dest.parent / archive.name
            shutil.copy2(archive, kept)
            print(f"Archive kept at: {kept}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:  # noqa: BLE001 - keep CLI error concise.
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
