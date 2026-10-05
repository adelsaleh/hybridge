"""Render the README first-solve figure from the README's own code block.

Runs the Python block under "A first solve on the CPU" without a display and
saves the comparison as a transparent light/dark pair in
``docs/getting_started/media``. Run as
``python -m scripts.reports.make_readme_first_solve_figure``.
"""

from __future__ import annotations

import os
from pathlib import Path
import re

os.environ["MPLBACKEND"] = "Agg"   # the README's plot call would otherwise open a window
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from hybridge.io.figures import apply_figure_theme  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MEDIA = ROOT / "docs" / "getting_started" / "media"
THEMES = {"light": "#1f2328", "dark": "#e6edf3"}   # GitHub's light and dark page text colors


def readme_block():
    """The first Python block of the README's first-solve section."""
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    section = text.split("## A first solve on the CPU", 1)[1].split("\n## ", 1)[0]
    return re.search(r"```python\n(.*?)```", section, re.S).group(1)


def main():
    import hybridge.io

    plt.close("all")
    figures = []
    plot = hybridge.io.plot_solution_comparison
    # Keep the figure the README's plot call returns; the block does not assign it.
    hybridge.io.plot_solution_comparison = lambda *a, **k: figures.append(plot(*a, **k)) or figures[-1]
    try:
        exec(compile(readme_block(), "README.md", "exec"), {"__name__": "readme_first_solve"})
    finally:
        hybridge.io.plot_solution_comparison = plot
    figure, = figures
    MEDIA.mkdir(parents=True, exist_ok=True)
    for theme, foreground in THEMES.items():
        apply_figure_theme(figure, foreground)
        path = MEDIA / f"first_solve_{theme}.png"
        figure.savefig(path, dpi=120, transparent=True, bbox_inches="tight", pad_inches=0.05)
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
