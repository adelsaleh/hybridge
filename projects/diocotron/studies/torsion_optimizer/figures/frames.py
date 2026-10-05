#!/usr/bin/env python3
"""Build a PNG-only storyboard from reduced-optimizer frame logs."""

from __future__ import annotations

# Support direct execution as well as python -m from a checkout.
if __package__ in {None, ""}:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[5]))

import argparse
import csv
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from projects.diocotron.paths import resolve_archive_path


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


_ITERATE_PANEL_LAYOUTS = {
    "OPT": (6, 3),
    "FINAL": (7, 3),
}


def candidate_only_frame(image: Image.Image, stage: str) -> Image.Image:
    """Crop a saved optimizer frame to its three evolving-state panels.

    Optimizer OPT frames contain three fixed target panels followed by
    potential, density, and mismatch. FINAL frames contain four fixed
    target panels followed by the same three evolving fields. The storyboard
    keeps those fixed fields in the separate design strip and uses this crop
    for every iterate, while the frame archive itself remains unchanged.
    """

    layout = _ITERATE_PANEL_LAYOUTS.get(str(stage).upper())
    if layout is None:
        return image.copy()
    panel_count, candidate_count = layout
    left = round(image.width * (panel_count - candidate_count) / panel_count)
    if not 0 < left < image.width:
        raise ValueError(f"cannot crop {stage} frame with width {image.width}")
    return image.crop((left, 0, image.width, image.height))


def _select_states(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    opt = sorted(
        (row for row in rows if row.get("stage") == "OPT"),
        key=lambda row: int(row.get("k", -1)),
    )
    selected: list[dict[str, str]] = []
    if opt:
        count = min(5, len(opt))
        indices = [
            round(index * (len(opt) - 1) / max(count - 1, 1))
            for index in range(count)
        ]
        selected.extend(opt[index] for index in dict.fromkeys(indices))
    final = [row for row in rows if row.get("stage") == "FINAL"]
    if final:
        selected.append(final[-1])
    return selected


def render(run_dir: Path, outputs: list[Path]) -> None:
    run_dir = resolve_archive_path(run_dir).resolve()
    rows = _read_rows(run_dir / "logs" / "frames.csv")
    design_rows = [row for row in rows if row.get("stage") == "DESIGN"]
    state_rows = _select_states(rows)
    if not design_rows or not state_rows:
        raise RuntimeError("frames.csv must contain DESIGN and OPT/FINAL rows")

    status_by_iteration: dict[int, str] = {}
    optimization_path = run_dir / "logs" / "optimization.csv"
    if optimization_path.is_file():
        for row in _read_rows(optimization_path):
            try:
                status_by_iteration[int(row.get("k", -1))] = row.get("status", "")
            except ValueError:
                continue

    with Image.open(resolve_archive_path(design_rows[-1]["filename"])) as source:
        design = source.convert("RGB")
    states: list[tuple[Image.Image, str]] = []
    for row in state_rows:
        with Image.open(resolve_archive_path(row["filename"])) as source:
            full_image = source.convert("RGB")
            image = candidate_only_frame(full_image, row.get("stage", ""))
        if row.get("stage") == "FINAL":
            label = "Final certified PDE projection"
        else:
            iteration = int(row.get("k", -1))
            status = status_by_iteration.get(iteration, "")
            label = f"Outer iteration {iteration}"
            if status:
                label += f" — {status.lower()}"
        states.append((image, label))

    width = design.width
    margin = max(24, width // 100)
    separator = max(28, width // 80)
    heading_height = max(52, width // 45)
    tile_gap = max(18, width // 140)
    columns = 2
    tile_width = (width - tile_gap) // columns
    tile_image_height = round(states[0][0].height * tile_width / states[0][0].width)
    tile_label_height = max(40, width // 62)
    tile_height = tile_label_height + tile_image_height
    row_count = math.ceil(len(states) / columns)
    state_height = row_count * tile_height + max(0, row_count - 1) * tile_gap
    canvas_height = (
        heading_height + design.height + separator
        + heading_height + state_height + margin
    )
    canvas = Image.new("RGB", (width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    heading_font = _font(max(24, width // 80))
    label_font = _font(max(18, width // 115))

    draw.text(
        (margin, (heading_height - heading_font.size) // 2),
        "Design fields (shown once)",
        fill="black",
        font=heading_font,
    )
    design_top = heading_height
    canvas.paste(design, (0, design_top))
    state_heading_top = design_top + design.height + separator
    draw.text(
        (margin, state_heading_top + (heading_height - heading_font.size) // 2),
        "Threshold-optimization evolution",
        fill="black",
        font=heading_font,
    )
    state_top = state_heading_top + heading_height
    for index, (image, label) in enumerate(states):
        row_index, column = divmod(index, columns)
        x = column * (tile_width + tile_gap)
        y = state_top + row_index * (tile_height + tile_gap)
        draw.rectangle((x, y, x + tile_width, y + tile_label_height), fill="#f3f4f6")
        draw.text(
            (x + 12, y + max(4, (tile_label_height - label_font.size) // 2)),
            label,
            fill="black",
            font=label_font,
        )
        resized = image.resize((tile_width, tile_image_height), Image.Resampling.LANCZOS)
        canvas.paste(resized, (x, y + tile_label_height))

    for output in outputs:
        output = output.expanduser().resolve()
        if output.suffix.lower() != ".png":
            raise ValueError(f"storyboard output must be PNG: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output, format="PNG", compress_level=6)
        print(output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, action="append", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    render(args.run_dir, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
