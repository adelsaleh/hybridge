from __future__ import annotations

from PIL import Image

from projects.diocotron.studies.torsion_optimizer.figures.frames import (
    candidate_only_frame,
)


def _striped_frame(panel_count: int) -> Image.Image:
    image = Image.new("RGB", (100 * panel_count, 40), "white")
    for index in range(panel_count):
        color = (20 + 30 * index, 10 + 20 * index, 5 + 10 * index)
        image.paste(
            Image.new("RGB", (100, image.height), color),
            (100 * index, 0),
        )
    return image


def test_opt_storyboard_crop_keeps_only_three_evolving_panels() -> None:
    full = _striped_frame(6)
    cropped = candidate_only_frame(full, "OPT")

    assert cropped.size == (300, 40)
    assert cropped.getpixel((50, 20)) == full.getpixel((350, 20))
    assert cropped.getpixel((250, 20)) == full.getpixel((550, 20))


def test_final_storyboard_crop_discards_four_fixed_target_panels() -> None:
    full = _striped_frame(7)
    cropped = candidate_only_frame(full, "FINAL")

    assert cropped.size == (300, 40)
    assert cropped.getpixel((50, 20)) == full.getpixel((450, 20))
    assert cropped.getpixel((250, 20)) == full.getpixel((650, 20))
