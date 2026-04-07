#!/usr/bin/env python
"""Build a fresh `assets/Docs/tracking_examples.jpg` for the HolisticTracker README.

The previous asset was generated before the bbox-smoothing fix. This script
re-creates it from the current (post-fix) tracking results, by pairing each
subject's source frame with the corresponding SMPL-X mesh render extracted from
that subject's `track_smplx.jpg` grid.

Layout
------
Reads `train_subset_x12.txt` from the release tree and tiles 12 subjects in a
4-col × 3-row grid. Each cell is a source-frame | mesh-render pair, so the
final image has 8 panels per row × 3 rows.

Usage
-----
    python release/make_tracking_examples.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Source track_smplx.jpg files have ~200 MP grids; disable PIL's bomb guard.
Image.MAX_IMAGE_PIXELS = None

DEFAULT_SUBJECTS = "/mnt/sdc/szj/Datasets/TEDWB1k_RELEASE/train_subset_x12.txt"
DEFAULT_FRAMES_ROOT = "/mnt/sdc/szj/Datasets/TEDWB1k/shots_images"
DEFAULT_EHMX_ROOT = "/mnt/sdc/szj/Datasets/TEDWB1k/shots_ehmx"
DEFAULT_OUT = "/mnt/sdc/szj/Playground/ehm-tracker/assets/Docs/tracking_examples.jpg"

PANEL_SIZE = 512        # each source-or-mesh panel
GRID_COLS = 4           # 4 subjects per row
GRID_ROWS = 3           # 3 rows of subjects
PAIR_GAP = 4            # px between source and mesh inside a cell
CELL_GAP = 8            # px between cells
TRACK_GRID_COLS = 8     # track_smplx.jpg has 8 cells per row


def find_first_frame(frames_root: Path, video_id: str) -> tuple[Path, str, int] | None:
    """Return (frame_path, shot_id, frame_idx_within_shot) for the first frame of the first shot."""
    vid_dir = frames_root / video_id
    if not vid_dir.exists():
        return None
    shots = sorted(d.name for d in vid_dir.iterdir() if d.is_dir())
    if not shots:
        return None
    first_shot = shots[0]
    frames = sorted((vid_dir / first_shot).glob("*.jpg"))
    if not frames:
        return None
    return frames[0], first_shot, 0


def extract_mesh_panel(track_smplx_path: Path, frame_idx: int) -> Image.Image:
    """Extract the panel for a specific frame index from a track_smplx.jpg grid.

    The grid is 8 columns wide; row N covers frames N*8 .. N*8+7. We use
    PIL's draft mode (1/4 native JPEG downscale) so we don't decode the full
    ~200 MP image into memory.
    """
    im = Image.open(track_smplx_path)
    im.draft("RGB", (4096, 4096))  # ~1/4 scale
    im = im.convert("RGB")
    full_w, full_h = im.size

    cell_w = full_w // TRACK_GRID_COLS
    n_rows = full_h // cell_w  # cells are roughly square
    if n_rows == 0:
        n_rows = 1
    cell_h = full_h // n_rows

    row = frame_idx // TRACK_GRID_COLS
    col = frame_idx % TRACK_GRID_COLS
    box = (col * cell_w, row * cell_h, (col + 1) * cell_w, (row + 1) * cell_h)
    panel = im.crop(box)

    # Each cell starts with a small "<shot_id>/<frame_id>" caption strip
    # at the top — trim ~3% of the cell height to remove it.
    trim = max(8, int(cell_h * 0.03))
    panel = panel.crop((0, trim, panel.size[0], panel.size[1]))
    return panel


def fit_square(im: Image.Image, size: int) -> Image.Image:
    """Resize and center-crop to size×size."""
    w, h = im.size
    scale = size / min(w, h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    im = im.resize((nw, nh), Image.LANCZOS)
    left = (nw - size) // 2
    top  = (nh - size) // 2
    return im.crop((left, top, left + size, top + size))


def build_cell(source_path: Path, mesh_panel: Image.Image, panel_size: int) -> Image.Image:
    """Make a 2-panel (source | mesh) horizontal strip."""
    src = Image.open(source_path).convert("RGB")
    src_sq = fit_square(src, panel_size)
    mesh_sq = fit_square(mesh_panel, panel_size)

    cell_w = panel_size * 2 + PAIR_GAP
    cell = Image.new("RGB", (cell_w, panel_size), (255, 255, 255))
    cell.paste(src_sq, (0, 0))
    cell.paste(mesh_sq, (panel_size + PAIR_GAP, 0))
    return cell


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", type=Path, default=Path(DEFAULT_SUBJECTS),
                    help="Text file with subject ids, one per line.")
    ap.add_argument("--frames_root", type=Path, default=Path(DEFAULT_FRAMES_ROOT))
    ap.add_argument("--ehmx_root",   type=Path, default=Path(DEFAULT_EHMX_ROOT))
    ap.add_argument("--out", type=Path, default=Path(DEFAULT_OUT))
    ap.add_argument("--panel_size", type=int, default=PANEL_SIZE)
    ap.add_argument("--grid_cols", type=int, default=GRID_COLS)
    ap.add_argument("--grid_rows", type=int, default=GRID_ROWS)
    ap.add_argument("--max_width", type=int, default=3140,
                    help="Final image is downsized so width <= max_width "
                         "(matches the original asset's resolution).")
    args = ap.parse_args()

    subject_ids = [ln.strip() for ln in args.subjects.read_text().splitlines() if ln.strip()]
    n_needed = args.grid_cols * args.grid_rows
    if len(subject_ids) < n_needed:
        print(f"WARNING: only {len(subject_ids)} subjects in {args.subjects}, need {n_needed}; "
              f"will pad with the first ones.")
        while len(subject_ids) < n_needed:
            subject_ids.append(subject_ids[len(subject_ids) % len(subject_ids)])
    subject_ids = subject_ids[:n_needed]
    print(f"Using {len(subject_ids)} subjects:")
    for vid in subject_ids:
        print(f"  - {vid}")

    cells: list[Image.Image] = []
    for vid in subject_ids:
        first = find_first_frame(args.frames_root, vid)
        if first is None:
            print(f"  SKIP {vid}: no frames")
            continue
        frame_path, shot_id, frame_idx = first
        track_smplx = args.ehmx_root / vid / "track_smplx.jpg"
        if not track_smplx.exists():
            print(f"  SKIP {vid}: no track_smplx.jpg")
            continue
        try:
            mesh = extract_mesh_panel(track_smplx, frame_idx)
        except Exception as exc:  # noqa: BLE001
            print(f"  SKIP {vid}: mesh extract failed ({exc})")
            continue
        cell = build_cell(frame_path, mesh, args.panel_size)
        cells.append(cell)
        print(f"  cell ok: {vid}  source={frame_path.name}  mesh from frame_idx={frame_idx}")

    if len(cells) < n_needed:
        print(f"ERROR: only built {len(cells)} cells, need {n_needed}")
        return 1

    cell_w = args.panel_size * 2 + PAIR_GAP
    cell_h = args.panel_size
    grid_w = cell_w * args.grid_cols + CELL_GAP * (args.grid_cols - 1)
    grid_h = cell_h * args.grid_rows + CELL_GAP * (args.grid_rows - 1)

    canvas = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    for i, cell in enumerate(cells):
        r, c = divmod(i, args.grid_cols)
        x = c * (cell_w + CELL_GAP)
        y = r * (cell_h + CELL_GAP)
        canvas.paste(cell, (x, y))

    if grid_w > args.max_width:
        scale = args.max_width / grid_w
        new_size = (int(round(grid_w * scale)), int(round(grid_h * scale)))
        canvas = canvas.resize(new_size, Image.LANCZOS)
        print(f"downsized to {new_size}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out, "JPEG", quality=88, optimize=True)
    size_kb = args.out.stat().st_size / 1024
    print(f"\nwrote {args.out} ({canvas.size[0]}x{canvas.size[1]}, {size_kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
