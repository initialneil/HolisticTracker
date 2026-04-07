#!/usr/bin/env python
"""Download and assemble a TEDWB1k split for HolisticAvatar.

Usage examples
--------------
# Smallest possible test (1 subject, ~80 MB):
python load_tedwb1k.py --split train_subset_x1 --out ~/data/tedwb1k_x1

# 12-subject overfit set (~1 GB):
python load_tedwb1k.py --split train_subset_x12 --out ~/data/tedwb1k_x12

# 20-subject training-monitor set (~2 GB, subset of train):
python load_tedwb1k.py --split train_val --out ~/data/tedwb1k_train_val

# 70-subject test set (~10 GB):
python load_tedwb1k.py --split test --out ~/data/tedwb1k_test

# Full training pool (1361 subjects, ~190 GB):
python load_tedwb1k.py --split train --out ~/data/tedwb1k_train

# Use already-downloaded HF cache, skip re-download:
python load_tedwb1k.py --split test --out ~/data/tedwb1k_test --hf_cache ~/.cache/huggingface

After it finishes, point your training config at --out:
    DATASET.data_path: <out>

The directory `<out>` will contain the same five files HolisticAvatar's
`TrackedData.__init__` expects:
    optim_tracking_ehm.pkl   # merged from per-subject pkls
    id_share_params.pkl      # merged from per-subject pkls
    videos_info.json         # merged from per-subject jsons
    dataset_frames.json      # copied from the release root
    extra_info.json          # generated locally with absolute frames_root/matte_root

…plus `frames_root/<vid>/...` and `matte_root/<vid>/...` containing the per-shot
JPGs that the dataloader reads at training time.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

REPO_ID = "initialneil/TEDWB1k"
REPO_TYPE = "dataset"

SPLIT_FILES = {
    "train":             "train.txt",
    "train_subset_x1":   "train_subset_x1.txt",
    "train_subset_x12":  "train_subset_x12.txt",
    "train_val":         "train_val.txt",
    "test":              "test.txt",
}

# Per-subject files we always need to feed TrackedData:
PER_SUBJECT_TRACKING = [
    "tracking/optim_tracking_ehm.pkl",
    "tracking/id_share_params.pkl",
    "tracking/videos_info.json",
]
# Per-subject preview/data dirs (frames + mattes are uploaded as tar; previews stay loose).
PER_SUBJECT_OPTIONAL = [
    "track_smplx.jpg",
    "track_flame.jpg",
    "track_base.jpg",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=list(SPLIT_FILES.keys()),
                    help="Which subject set to download.")
    ap.add_argument("--out", required=True, type=Path,
                    help="Local directory to assemble the dataset into.")
    ap.add_argument("--repo_id", default=REPO_ID,
                    help=f"HuggingFace dataset repo id (default: {REPO_ID}).")
    ap.add_argument("--hf_cache", type=Path, default=None,
                    help="Override HuggingFace cache dir (default: ~/.cache/huggingface).")
    ap.add_argument("--include_previews", action="store_true",
                    help="Also download per-subject track_*.jpg files (~20 MB/subject).")
    ap.add_argument("--keep_tars", action="store_true",
                    help="Keep frames.tar / mattes.tar after extraction (default: delete to save space).")
    ap.add_argument("--skip_download", action="store_true",
                    help="Skip download step (assume HF cache is already populated).")
    ap.add_argument("--skip_extract", action="store_true",
                    help="Skip frames/mattes extraction (just merge tracking pkls).")
    ap.add_argument("--local_snapshot", type=Path, default=None,
                    help="Skip HF download entirely; treat this local dir as the snapshot. "
                         "Useful for testing build_release.py output before upload, or if "
                         "the user already has a clone of the repo.")
    return ap.parse_args()


def read_subject_ids(
    split_name: str,
    repo_id: str,
    hf_cache: Path | None,
    local_snapshot: Path | None,
) -> list[str]:
    """Fetch and parse the split txt for the chosen split."""
    txt_name = SPLIT_FILES[split_name]
    if local_snapshot is not None:
        local_txt = local_snapshot / txt_name
        if not local_txt.exists():
            raise FileNotFoundError(f"{local_txt} not found in local snapshot")
        print(f"[1/5] Reading split file {txt_name} from local snapshot ...")
    else:
        from huggingface_hub import hf_hub_download
        print(f"[1/5] Fetching split file {txt_name} from {repo_id} ...")
        local_txt = Path(hf_hub_download(
            repo_id=repo_id,
            filename=txt_name,
            repo_type=REPO_TYPE,
            cache_dir=str(hf_cache) if hf_cache else None,
        ))
    ids = [ln.strip() for ln in Path(local_txt).read_text().splitlines() if ln.strip()]
    print(f"   {len(ids)} subject ids in '{split_name}'")
    return ids


def download_subject_files(
    repo_id: str,
    hf_cache: Path | None,
    subject_ids: list[str],
    include_previews: bool,
) -> Path:
    """Snapshot only the subject files we need. Returns the snapshot root."""
    from huggingface_hub import snapshot_download

    patterns: list[str] = []
    for vid in subject_ids:
        for f in PER_SUBJECT_TRACKING:
            patterns.append(f"subjects/{vid}/{f}")
        # frames.tar / mattes.tar are added at upload time (B9). The release tree
        # built by build_release.py exposes loose dirs which become tars on upload.
        patterns.append(f"subjects/{vid}/frames.tar")
        patterns.append(f"subjects/{vid}/mattes.tar")
        if include_previews:
            for f in PER_SUBJECT_OPTIONAL:
                patterns.append(f"subjects/{vid}/{f}")
    # Always grab dataset_frames.json (used for train/valid frame split inside TrackedData)
    patterns.append("dataset_frames.json")

    print(f"[2/5] snapshot_download from {repo_id} ({len(patterns)} patterns) ...")
    snap = snapshot_download(
        repo_id=repo_id,
        repo_type=REPO_TYPE,
        allow_patterns=patterns,
        cache_dir=str(hf_cache) if hf_cache else None,
    )
    print(f"   snapshot at: {snap}")
    return Path(snap)


def merge_tracking(
    snapshot: Path,
    subject_ids: list[str],
    out: Path,
) -> None:
    """Merge per-subject tracking files into the 5-file TrackedData bundle.

    Per-subject `optim_tracking_ehm.pkl` and `id_share_params.pkl` are FLAT
    (no top-level video_id key) — the merger wraps them under each video_id
    so the result matches the format produced by `merge_ehmx_dataset.py`.
    """
    print(f"[3/5] Merging tracking files for {len(subject_ids)} subjects ...")
    merged_optim: dict = {}
    merged_id_share: dict = {}
    merged_videos_info: dict = {}
    n_frames_total = 0
    missing: list[str] = []

    t0 = time.time()
    for i, vid in enumerate(subject_ids, 1):
        sub = snapshot / "subjects" / vid / "tracking"
        opt_p = sub / "optim_tracking_ehm.pkl"
        id_p  = sub / "id_share_params.pkl"
        vi_p  = sub / "videos_info.json"
        if not (opt_p.exists() and id_p.exists() and vi_p.exists()):
            missing.append(vid)
            continue
        with open(opt_p, "rb") as f:
            merged_optim[vid] = pickle.load(f)
        with open(id_p, "rb") as f:
            merged_id_share[vid] = pickle.load(f)
        with open(vi_p, "r") as f:
            vi = json.load(f)
        merged_videos_info.update(vi)
        n_frames_total += len(merged_optim[vid])
        if i % 50 == 0 or i == len(subject_ids):
            elapsed = time.time() - t0
            print(f"   merged {i}/{len(subject_ids)} subjects "
                  f"({n_frames_total} frames so far, {elapsed:.1f}s)")

    if missing:
        print(f"   WARNING: {len(missing)} subjects had missing tracking files: {missing[:5]}...", file=sys.stderr)

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "optim_tracking_ehm.pkl", "wb") as f:
        pickle.dump(merged_optim, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(out / "id_share_params.pkl", "wb") as f:
        pickle.dump(merged_id_share, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(out / "videos_info.json", "w") as f:
        json.dump(merged_videos_info, f)
    print(f"   wrote optim_tracking_ehm.pkl ({n_frames_total} frames)")
    print(f"   wrote id_share_params.pkl ({len(merged_id_share)} subjects)")
    print(f"   wrote videos_info.json ({len(merged_videos_info)} subjects)")

    # Copy dataset_frames.json from snapshot (used by train/valid splits inside TrackedData)
    src_frames = snapshot / "dataset_frames.json"
    if src_frames.exists():
        out_frames = out / "dataset_frames.json"
        out_frames.write_text(src_frames.read_text())
        print(f"   copied dataset_frames.json")
    else:
        print("   WARNING: dataset_frames.json missing in snapshot — train/valid splits won't work")


def setup_frame_dirs(
    snapshot: Path,
    subject_ids: list[str],
    out: Path,
    keep_tars: bool,
) -> tuple[Path, Path]:
    """Materialize per-subject frames + mattes under out/frames_root, out/matte_root.

    Handles both layouts:
    - Snapshot has `subjects/<vid>/frames.tar` (HF upload case): extract into
      out/frames_root/<vid>/ and (optionally) delete the tar to save disk.
    - Snapshot has `subjects/<vid>/frames/` as a real dir or symlink (local
      build_release.py output, or pre-extracted clone): symlink it from
      out/frames_root/<vid> -> resolved frames dir.
    """
    import tarfile

    frames_root = out / "frames_root"
    matte_root  = out / "matte_root"
    frames_root.mkdir(parents=True, exist_ok=True)
    matte_root.mkdir(parents=True, exist_ok=True)

    print(f"[4/5] Setting up frames + mattes for {len(subject_ids)} subjects ...")
    n_extracted = n_linked = n_missing = 0
    for vid in subject_ids:
        sub = snapshot / "subjects" / vid
        for kind, dest_root in [("frames", frames_root), ("mattes", matte_root)]:
            tar_path = sub / f"{kind}.tar"
            dir_path = sub / kind
            target   = dest_root / vid
            if target.exists() or target.is_symlink():
                continue  # idempotent
            if tar_path.exists():
                target.mkdir(parents=True, exist_ok=True)
                with tarfile.open(tar_path, "r") as tar:
                    tar.extractall(path=target)
                if not keep_tars:
                    tar_path.unlink()
                n_extracted += 1
            elif dir_path.exists():
                # Resolve through any symlinks so the link in out/ is stable.
                target.symlink_to(dir_path.resolve())
                n_linked += 1
            else:
                print(f"   WARNING: {vid}/{kind} not in snapshot (no .tar, no dir)", file=sys.stderr)
                n_missing += 1
    print(f"   extracted={n_extracted // 2}  linked={n_linked // 2}  missing={n_missing}")
    return frames_root, matte_root


def write_extra_info(out: Path, frames_root: Path, matte_root: Path) -> None:
    """Write extra_info.json with absolute paths to the local extracted dirs."""
    print("[5/5] Writing extra_info.json ...")
    extra = {
        "frames_root": str(frames_root.resolve()),
        "matte_root":  str(matte_root.resolve()),
        "pshuman_root": None,
    }
    with open(out / "extra_info.json", "w") as f:
        json.dump(extra, f, indent=2)
    print(f"   frames_root = {extra['frames_root']}")
    print(f"   matte_root  = {extra['matte_root']}")


def main() -> int:
    args = parse_args()
    out = args.out.expanduser().resolve()
    local_snapshot = args.local_snapshot.expanduser().resolve() if args.local_snapshot else None

    if local_snapshot is None:
        try:
            import huggingface_hub  # noqa: F401
        except ImportError:
            print("ERROR: huggingface_hub is required. Install with:", file=sys.stderr)
            print("  pip install huggingface_hub", file=sys.stderr)
            return 2

    subject_ids = read_subject_ids(args.split, args.repo_id, args.hf_cache, local_snapshot)

    if local_snapshot is not None:
        print(f"[2/5] Using local snapshot at {local_snapshot} (no download)")
        snapshot = local_snapshot
    elif args.skip_download:
        print("[2/5] --skip_download: assuming local snapshot is already populated")
        from huggingface_hub import snapshot_download
        snapshot = Path(snapshot_download(
            repo_id=args.repo_id,
            repo_type=REPO_TYPE,
            allow_patterns=["dataset_frames.json"],
            cache_dir=str(args.hf_cache) if args.hf_cache else None,
        ))
    else:
        snapshot = download_subject_files(
            repo_id=args.repo_id,
            hf_cache=args.hf_cache,
            subject_ids=subject_ids,
            include_previews=args.include_previews,
        )

    merge_tracking(snapshot, subject_ids, out)

    if args.skip_extract:
        print("[4/5] --skip_extract: skipping frames/mattes setup")
        frames_root = out / "frames_root"
        matte_root  = out / "matte_root"
    else:
        frames_root, matte_root = setup_frame_dirs(snapshot, subject_ids, out, args.keep_tars)

    write_extra_info(out, frames_root, matte_root)

    print()
    print("=" * 60)
    print(f"DONE. Local dataset assembled at: {out}")
    print(f"  Point training config at:  DATASET.data_path: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
