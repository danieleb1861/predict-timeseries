"""Splash banner for the ``predict-timeseries`` command.

Pure rendering helpers: no I/O, no subprocess calls. The CLI decides when to
show the banner and passes in the version and git reference.
"""

from __future__ import annotations

from typing import Optional

from rich.cells import cell_len
from rich.console import Console, Group, RenderableType
from rich.table import Table
from rich.text import Text

ACCENT = "#FF5A1F"  # measured load (history)
FORECAST = "#FF9A6B"  # forecast centre line and uncertainty cone
FAINT = "#6B3A28"  # baseline and "now" marker
INK = "#D9D9D9"  # wordmark

WORDMARK = "PREDICT-TIMESERIES"
TAGLINE = "shipboard electric-load forecasting"

# 3x4 pixel font, rendered with half blocks: 4 pixel rows -> 2 text rows.
FONT = {
    "P": ["111", "101", "111", "100"],
    "R": ["111", "101", "110", "101"],
    "E": ["111", "110", "100", "111"],
    "D": ["110", "101", "101", "110"],
    "I": ["1", "1", "1", "1"],
    "C": ["111", "100", "100", "111"],
    "T": ["111", "010", "010", "010"],
    "M": ["10001", "11011", "10101", "10001"],
    "S": ["111", "110", "011", "111"],
    "-": ["00", "11", "00", "00"],
}
HALF_BLOCKS = {(False, False): " ", (True, False): "▀", (False, True): "▄", (True, True): "█"}

# Mascot: measured load, a "now" marker, then a forecast with a widening cone.
# Values are row indices, 0 = bottom row.
HISTORY = [2, 2, 3, 4, 5, 5, 4, 3, 3, 4, 5, 6]
FORECAST_CENTRE = [6, 6, 5, 5, 4, 4, 4, 5, 5, 6]
MASCOT_ROWS = 8


def _supports_unicode(console: Console) -> bool:
    """Return True if the console encoding can print block and dot glyphs."""
    return (console.encoding or "").lower().replace("-", "").startswith("utf")


def render_wordmark(word: str = WORDMARK) -> Text:
    """
    Render ``word`` as a two-row pixel wordmark.

    Inputs
    ----------
    word : str, optional
        Upper-case text. Each character must be a key of ``FONT``.

    Returns
    -------
    rich.text.Text
        Two text rows of half-block glyphs. Rows have equal width.
    """
    pixel_rows = ["0".join(FONT[char][row] for char in word) for row in range(4)]
    bits = [[pixel == "1" for pixel in row] for row in pixel_rows]
    lines = [
        "".join(HALF_BLOCKS[(bits[top][col], bits[top + 1][col])] for col in range(len(bits[0])))
        for top in (0, 2)
    ]
    return Text("\n".join(lines), style=INK)


def render_mascot() -> Text:
    """
    Render the load-forecast mascot.

    Inputs
    ----------
    None

    Returns
    -------
    rich.text.Text
        ``MASCOT_ROWS`` rows: history dots, a "now" marker, and a forecast
        centre line with an uncertainty cone that widens with the horizon.
    """
    n_history = len(HISTORY)
    out = Text()
    for row in range(MASCOT_ROWS - 1, -1, -1):
        for col in range(n_history + len(FORECAST_CENTRE)):
            if col < n_history:
                if HISTORY[col] == row:
                    glyph, style = "●", ACCENT
                elif row == 0:
                    glyph, style = "·", FAINT
                else:
                    glyph, style = " ", ""
            else:
                step = col - n_history
                spread = 1 + step // 3
                distance = abs(FORECAST_CENTRE[step] - row)
                if distance == 0:
                    glyph, style = "•", FORECAST
                elif distance <= spread:
                    glyph, style = "·", FORECAST
                else:
                    glyph, style = " ", ""
            if col == n_history and glyph == " ":
                glyph, style = "│", FAINT
            out.append(glyph + " ", style=style)
        if row:
            out.append("\n")
    return out


def _block_width(text: Text) -> int:
    """Return the widest line of a multi-line ``Text`` in terminal cells."""
    return max(cell_len(line) for line in text.plain.splitlines())


def _tagline(version: Optional[str], git_ref: Optional[str], separator: str = "·") -> Text:
    """Build the tagline with optional version and git reference."""
    text = Text(TAGLINE, style=f"bold {ACCENT}")
    for item in (f"v{version}" if version else None, git_ref):
        if item:
            text.append(f"  {separator}  {item}", style="dim")
    return text


def render_banner(
    console: Console,
    version: Optional[str] = None,
    git_ref: Optional[str] = None,
) -> RenderableType:
    """
    Build the splash banner that fits the console.

    Inputs
    ----------
    console : rich.console.Console
        Target console. Its width and encoding select the layout.
    version : str or None, optional
        Package version shown in the tagline.
    git_ref : str or None, optional
        Git reference shown in the tagline, for example ``main@3793742``.

    Returns
    -------
    rich.console.RenderableType
        Full banner (mascot + wordmark + tagline) when the console is wide
        enough and supports Unicode; otherwise a single compact line.
    """
    unicode_ok = _supports_unicode(console)
    tagline = _tagline(version, git_ref, separator="·" if unicode_ok else "|")

    if unicode_ok:
        mascot = render_mascot()
        wordmark = render_wordmark()
        gap = 4
        right_width = max(_block_width(wordmark), tagline.cell_len)
        full_width = _block_width(mascot) + gap + right_width
        if console.width >= full_width:
            grid = Table.grid(padding=(0, gap))
            grid.add_column(vertical="middle")
            grid.add_column(vertical="middle")
            grid.add_row(mascot, Group(wordmark, Text(), tagline))
            return grid

    # Compact line: drop the git ref, then the version, until it fits.
    prefix = Text("● • · " if unicode_ok else "", style=ACCENT)
    separator = "·" if unicode_ok else "|"
    candidates = [tagline, _tagline(version, None, separator), _tagline(None, None)]
    for candidate in candidates:
        line = Text.assemble(prefix, (WORDMARK, f"bold {INK}"), "  ", candidate)
        if line.cell_len <= console.width:
            return line
    return Text.assemble(prefix, (WORDMARK, f"bold {INK}"))
