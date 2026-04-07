#!/usr/bin/env python
"""Build the local TEDWB1k HuggingFace release tree.

Layout produced (under --out, default /mnt/sdc/szj/Datasets/TEDWB1k_RELEASE):

    train.txt                  (materialized — full training pool, 1361 ids)
    train_subset_x1.txt        (materialized — 1 id, single-subject overfit; subset of train)
    train_subset_x12.txt       (materialized — 12 ids, small overfit; subset of train)
    train_val.txt              (materialized — 20 ids monitored during training; subset of train)
    test.txt                   (materialized — 70 ids, identity-disjoint from train)
    dataset_frames.json        -> source/shots_ehmx/dataset_frames.json
    metadata/
        subjects_train.parquet            (materialized w/ embedded thumbnails)
        subjects_train_subset_x1.parquet  (materialized)
        subjects_train_subset_x12.parquet (materialized)
        subjects_train_val.parquet        (materialized)
        subjects_test.parquet             (materialized)
        subjects.csv                      (materialized — all rows, programmatic)
        skipped.txt                       (materialized)
        previews/<video_id>.jpg           (resized track_smplx.jpg, materialized)
    subjects/<video_id>/
        tracking/
            optim_tracking_ehm.pkl -> source
            id_share_params.pkl    -> source
            videos_info.json       -> source
        track_smplx.jpg            -> source
        track_flame.jpg            -> source
        track_base.jpg             -> source
        frames                     -> source/shots_images/<video_id>   (dir symlink)
        mattes                     -> source/shots_rmbg2/<video_id>    (dir symlink)

All symlinks are RELATIVE (so the release tree is portable). Only small
files are materialized (parquets/csv/skipped.txt/split-txts + ~512px
thumbnails), so the tree footprint stays under ~100 MB even though it
represents ~198 GB of source data.

Re-runnable: existing symlinks are replaced; thumbnails are skipped if present.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

# track_*.jpg files are tall stacked grids (~150–250 MP) — disable PIL's
# decompression-bomb guard so the thumbnailer can read them.
Image.MAX_IMAGE_PIXELS = None

REQUIRED_TRACKING = [
    "optim_tracking_ehm.pkl",
    "id_share_params.pkl",
    "videos_info.json",
]
PREVIEW_JPGS = ["track_smplx.jpg", "track_flame.jpg", "track_base.jpg"]


def replace_symlink(target: Path, link: Path) -> None:
    """Create link → target as a RELATIVE symlink, replacing any existing entry.

    Relative symlinks make the release tree portable: moving/renaming
    TEDWB1k_RELEASE keeps the links valid as long as the relationship
    between the release dir and the source dir is unchanged.
    """
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        link.unlink()
    rel_target = os.path.relpath(target.resolve(), start=link.parent.resolve())
    link.symlink_to(rel_target)


def load_split(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {ln.strip() for ln in path.read_text().splitlines() if ln.strip()}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=Path("/mnt/sdc/szj/Datasets/TEDWB1k"))
    ap.add_argument("--out", type=Path, default=Path("/mnt/sdc/szj/Datasets/TEDWB1k_RELEASE"))
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only first N subjects (0 = all). For dry runs.")
    ap.add_argument("--thumb_size", type=int, default=512,
                    help="Long-side pixel size for metadata/previews thumbnails.")
    ap.add_argument("--skip_thumbnails", action="store_true",
                    help="Skip thumbnail generation (parquet still references them).")
    ap.add_argument("--thumb_workers", type=int, default=8,
                    help="Parallel workers for thumbnail generation.")
    return ap.parse_args()


def _make_one_thumbnail(args: tuple[str, Path, Path, int]) -> tuple[str, str]:
    """Worker: generate one thumbnail. Returns (vid, status)."""
    vid, src_jpg, dst_jpg, thumb_size = args
    if dst_jpg.exists():
        return vid, "cached"
    if not src_jpg.exists():
        return vid, "missing_src"
    try:
        with Image.open(src_jpg) as im:
            # JPEG native downscale on read (1/2, 1/4, 1/8) — much faster than
            # decoding the full ~200 MP grid before resizing.
            im.draft("RGB", (thumb_size * 2, thumb_size * 2))
            im = im.convert("RGB")
            w, h = im.size
            scale = thumb_size / max(w, h)
            if scale < 1.0:
                im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
            im.save(dst_jpg, "JPEG", quality=85, optimize=True)
        return vid, "ok"
    except Exception as exc:  # noqa: BLE001
        return vid, f"failed:{exc}"


def link_repo_files(out: Path, scripts_dir: Path) -> None:
    """Symlink README.md, LICENSE, and load_tedwb1k.py at the release root.

    These three files are part of the public HF repo and are kept in
    ehm-tracker/release/ so they're version-controlled alongside the build
    tooling. Symlinking them into TEDWB1k_RELEASE/ means `huggingface-cli upload`
    of the release dir picks them up automatically.
    """
    print("[0/5] Linking README + LICENSE + loader script ...")
    for fname in ("README.md", "LICENSE", "load_tedwb1k.py"):
        src = scripts_dir / fname
        if src.exists():
            replace_symlink(src, out / fname)
            print(f"   {fname:20s} -> {os.path.relpath(src, out)}")
        else:
            print(f"   {fname:20s} MISSING ({src})")


def write_split_txt(out_path: Path, ids: list[str]) -> None:
    """Materialize a release-side split txt file (sorted, one id per line)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.is_symlink() or out_path.exists():
        out_path.unlink()
    out_path.write_text("\n".join(sorted(ids)) + "\n")


def link_splits(src: Path, out: Path) -> dict[str, set[str]]:
    """Compute and write the 5 release-side splits.

    Source files used:
        single_train.txt        -> train_subset_x1
        train_12subjects.txt    -> train_subset_x12
        test.txt                -> test (unchanged)
        train_all.txt           -> train (full training pool, 1361 ids)
        shots_ehmx/dataset_frames.json -> source of `train_val` ids

    Final splits in the release (overlap structure):
        train             = train_all                (1361 — full training pool)
          ├── train_subset_x1   = single_train       (1, ⊂ train)
          ├── train_subset_x12  = train_12subjects   (12, ⊂ train)
          └── train_val         = dframes['valid']   (20, ⊂ train, monitored during training)
        test              = test.txt                  (70, identity-disjoint from train)
    """
    print("[1/5] Computing and writing split files at release root ...")

    src_train_all  = load_split(src / "train_all.txt")
    src_12         = load_split(src / "train_12subjects.txt")
    src_1          = load_split(src / "single_train.txt")
    src_test       = load_split(src / "test.txt")

    dframes_src = src / "shots_ehmx" / "dataset_frames.json"
    if not dframes_src.exists():
        raise FileNotFoundError(f"{dframes_src} is required for the train_val split")
    with open(dframes_src) as f:
        dframes = json.load(f)
    train_val = sorted({fk.split("/")[0] for fk in dframes.get("valid", [])})

    # Sanity: train_val must be a subset of train_all
    leak = set(train_val) - src_train_all
    if leak:
        print(f"   WARNING: {len(leak)} train_val ids not in train_all: {sorted(leak)[:5]}",
              file=sys.stderr)

    splits: dict[str, list[str]] = {
        "train":            sorted(src_train_all),
        "train_subset_x1":  sorted(src_1),
        "train_subset_x12": sorted(src_12),
        "train_val":        train_val,
        "test":             sorted(src_test),
    }

    # Sanity: train_subset_x{1,12} and train_val should all be subsets of train
    train_set = set(splits["train"])
    for sub_name in ("train_subset_x1", "train_subset_x12", "train_val"):
        not_in_train = set(splits[sub_name]) - train_set
        if not_in_train:
            print(f"   WARNING: {sub_name} has {len(not_in_train)} ids not in train: {sorted(not_in_train)[:5]}",
                  file=sys.stderr)

    # Informational: do the small overfit subsets overlap with train_val?
    for sub_name in ("train_subset_x1", "train_subset_x12"):
        overlap = set(splits[sub_name]) & set(train_val)
        if overlap:
            print(f"   note: {sub_name} shares {len(overlap)} ids with train_val: {sorted(overlap)}")

    for name, ids in splits.items():
        write_split_txt(out / f"{name}.txt", ids)
        print(f"   {name:18s} {len(ids):5d} ids  -> {name}.txt")

    replace_symlink(dframes_src, out / "dataset_frames.json")
    print(f"   dataset_frames.json  -> dataset_frames.json")

    return {name: set(ids) for name, ids in splits.items()}


def link_subjects(
    src: Path,
    out: Path,
    splits: dict[str, set[str]],
    limit: int,
) -> tuple[list[dict], list[tuple[str, str, object]]]:
    print(f"[2/5] Building subjects/ from {src}/shots_ehmx ...")
    ehmx_root = src / "shots_ehmx"
    images_root = src / "shots_images"
    mattes_root = src / "shots_rmbg2"

    subject_dirs = sorted(d for d in ehmx_root.iterdir() if d.is_dir() and not d.name.startswith("."))
    # Filter out the merge bundle dirs (these live alongside per-subject dirs in earlier layouts)
    subject_dirs = [d for d in subject_dirs if not d.name.startswith("shots_ehmx_overfit")]
    if limit > 0:
        subject_dirs = subject_dirs[:limit]
        print(f"   --limit {limit}: processing first {len(subject_dirs)} subjects")
    else:
        print(f"   {len(subject_dirs)} candidate subject dirs")

    rows: list[dict] = []
    skipped: list[tuple[str, str, object]] = []

    for sub_src in tqdm(subject_dirs, desc="link subjects"):
        vid = sub_src.name

        missing = [f for f in REQUIRED_TRACKING if not (sub_src / f).exists()]
        if missing:
            skipped.append((vid, "tracking_missing", missing))
            continue
        if not (images_root / vid).exists():
            skipped.append((vid, "no_frames_dir", str(images_root / vid)))
            continue
        if not (mattes_root / vid).exists():
            skipped.append((vid, "no_mattes_dir", str(mattes_root / vid)))
            continue

        sub_out = out / "subjects" / vid
        tracking_dir = sub_out / "tracking"
        tracking_dir.mkdir(parents=True, exist_ok=True)
        for fname in REQUIRED_TRACKING:
            replace_symlink(sub_src / fname, tracking_dir / fname)
        for fname in PREVIEW_JPGS:
            jpg = sub_src / fname
            if jpg.exists():
                replace_symlink(jpg, sub_out / fname)
        replace_symlink(images_root / vid, sub_out / "frames")
        replace_symlink(mattes_root / vid, sub_out / "mattes")

        try:
            with open(sub_src / "videos_info.json") as f:
                vi = json.load(f)
            entry = vi.get(vid, {})
            frame_keys = entry.get("frames_keys", [])
            num_frames = entry.get("frames_num", 0) or len(frame_keys)
            shots = sorted({k.split("/")[0] for k in frame_keys})
            num_shots = len(shots)
        except Exception as exc:  # noqa: BLE001
            num_frames = 0
            num_shots = 0
            skipped.append((vid, "videos_info_unreadable", str(exc)))

        rows.append({
            "video_id": vid,
            "num_frames": int(num_frames),
            "num_shots": int(num_shots),
            "track_smplx_path": f"subjects/{vid}/track_smplx.jpg",
            "track_flame_path": f"subjects/{vid}/track_flame.jpg",
            "track_base_path":  f"subjects/{vid}/track_base.jpg",
            "preview_path":     f"metadata/previews/{vid}.jpg",
        })

    print(f"   linked {len(rows)} subjects, skipped {len(skipped)}")
    # Sort rows by video_id for stable parquet output
    rows.sort(key=lambda r: r["video_id"])
    return rows, skipped


def make_thumbnails(rows: list[dict], src: Path, out: Path, thumb_size: int, workers: int) -> None:
    print(f"[3/5] Generating {thumb_size}px thumbnails -> metadata/previews/ ({workers} workers)...")
    previews_dir = out / "metadata" / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        (
            row["video_id"],
            src / "shots_ehmx" / row["video_id"] / "track_smplx.jpg",
            previews_dir / f"{row['video_id']}.jpg",
            thumb_size,
        )
        for row in rows
    ]

    n_ok = n_cached = n_missing = n_failed = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for vid, status in tqdm(
            ex.map(_make_one_thumbnail, tasks, chunksize=4),
            total=len(tasks),
            desc="thumbnails",
        ):
            if status == "ok":
                n_ok += 1
            elif status == "cached":
                n_cached += 1
            elif status == "missing_src":
                n_missing += 1
            else:
                n_failed += 1
                print(f"   {vid}: {status}", file=sys.stderr)
    print(f"   made={n_ok} cached={n_cached} missing_src={n_missing} failed={n_failed}")


def write_metadata(
    rows: list[dict],
    skipped: list[tuple[str, str, object]],
    splits: dict[str, set[str]],
    out: Path,
) -> None:
    print("[4/5] Writing metadata/subjects*.{parquet,csv} ...")
    meta_dir = out / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(meta_dir / "subjects.csv", index=False)

    # Build per-split parquets via the `datasets` library so the `preview`
    # column is cast to `Image()` with embedded jpg bytes. Five split parquets
    # (matching the 5 release splits) → HF Dataset Viewer shows 5 tabs.
    #
    # NOTE: cast_column to Image() with a path string stores
    # {bytes: None, path: <absolute>} which the HF viewer cannot resolve.
    # We have to load the raw jpg bytes and pass them as
    # {"bytes": ..., "path": None} so the parquet is fully self-contained.
    try:
        import datasets

        df_for_ds = df.copy()

        def _load_bytes(rel: str) -> dict:
            with open(out / rel, "rb") as fh:
                return {"bytes": fh.read(), "path": None}

        df_for_ds["preview"] = df_for_ds["preview_path"].apply(_load_bytes)

        for split_name in ("train", "train_subset_x1", "train_subset_x12", "train_val", "test"):
            ids_in_split = splits.get(split_name, set())
            split_df = df_for_ds[df_for_ds["video_id"].isin(ids_in_split)].reset_index(drop=True)
            ds = datasets.Dataset.from_pandas(split_df, preserve_index=False)
            ds = ds.cast_column("preview", datasets.Image())
            out_path = meta_dir / f"subjects_{split_name}.parquet"
            ds.to_parquet(str(out_path))
            pq_size = out_path.stat().st_size / 1e6
            print(f"   {len(split_df):4d} rows -> subjects_{split_name}.parquet ({pq_size:.1f} MB)")
    except Exception as exc:  # noqa: BLE001
        print(f"   WARNING: datasets-based parquet write failed ({exc}); falling back to plain pandas",
              file=sys.stderr)
        df.to_parquet(meta_dir / "subjects.parquet", index=False)

    print("[5/5] Writing metadata/skipped.txt ...")
    with open(meta_dir / "skipped.txt", "w") as f:
        for vid, reason, info in skipped:
            f.write(f"{vid}\t{reason}\t{info if info is not None else ''}\n")
    print(f"   {len(skipped)} skipped entries")


def print_summary(out: Path, rows: list[dict], skipped: list[tuple[str, str, object]], splits: dict[str, set[str]]) -> None:
    print()
    print("=" * 60)
    print(f"DONE. Output: {out}")
    print(f"  subjects linked: {len(rows)}")
    print(f"  subjects skipped: {len(skipped)}")
    print(f"  splits coverage:")
    linked_ids = {r["video_id"] for r in rows}
    for name in ["train", "train_subset_x1", "train_subset_x12", "train_val", "test"]:
        if name not in splits:
            continue
        in_release = len(splits[name] & linked_ids)
        total = len(splits[name])
        print(f"    {name:18s} {in_release}/{total}")


def main() -> int:
    args = parse_args()
    if not args.source.exists():
        print(f"ERROR: source not found: {args.source}", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)

    # README/LICENSE/loader live next to this script, in ehm-tracker/release/.
    scripts_dir = Path(__file__).resolve().parent
    link_repo_files(args.out, scripts_dir)

    splits = link_splits(args.source, args.out)
    rows, skipped = link_subjects(args.source, args.out, splits, args.limit)
    if not args.skip_thumbnails:
        make_thumbnails(rows, args.source, args.out, args.thumb_size, args.thumb_workers)
    write_metadata(rows, skipped, splits, args.out)
    print_summary(args.out, rows, skipped, splits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
