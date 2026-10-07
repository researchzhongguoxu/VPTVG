"""Text and formula drawing helpers for Renderer v1."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

from PIL import ImageDraw, ImageFont


@lru_cache(maxsize=32)
def load_font(size: int, *, bold: bool = False) -> tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, str | None]:
    """Load a readable Chinese-capable font when available."""

    candidates = [
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if not path.exists():
            continue
        try:
            return ImageFont.truetype(str(path), size=size), str(path)
        except OSError:
            continue
    return ImageFont.load_default(), None


def wrap_text(
    text: str,
    *,
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    max_width: int,
    max_lines: int | None = None,
) -> list[str]:
    """Wrap mixed Chinese/ASCII text by measured pixel width."""

    paragraphs = str(text or "").splitlines() or [""]
    lines: list[str] = []
    for paragraph in paragraphs:
        current = ""
        for token in _tokens(paragraph):
            candidate = f"{current}{token}" if current else token
            if current and _text_width(draw, candidate, font) > max_width:
                lines.append(current)
                current = token
                if max_lines is not None and len(lines) >= max_lines:
                    return _truncate_lines(lines, draw=draw, font=font, max_width=max_width)
            else:
                current = candidate
        if current:
            lines.append(current)
            if max_lines is not None and len(lines) >= max_lines:
                return _truncate_lines(lines, draw=draw, font=font, max_width=max_width)
    return lines


def line_height(draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont) -> int:
    """Return a conservative line height for the selected font."""

    return _line_height(draw, font)


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    max_width: int,
    line_spacing: int = 10,
    max_lines: int | None = None,
) -> tuple[int, int, int, int]:
    """Draw wrapped text and return its bounding box."""

    x, y = xy
    lines = wrap_text(text, draw=draw, font=font, max_width=max_width, max_lines=max_lines)
    line_height = _line_height(draw, font)
    bottom = y
    widest = 0
    for index, line in enumerate(lines):
        top = y + index * (line_height + line_spacing)
        draw.text((x, top), line, font=font, fill=fill)
        widest = max(widest, _text_width(draw, line, font))
        bottom = top + line_height
    return x, y, x + widest, bottom


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), str(text), font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _tokens(text: str) -> Iterable[str]:
    buffer = ""
    for char in text:
        if char.isspace():
            if buffer:
                yield buffer
                buffer = ""
            yield char
        elif "\u4e00" <= char <= "\u9fff":
            if buffer:
                yield buffer
                buffer = ""
            yield char
        else:
            buffer += char
            if char in {",", ".", ";", ":", "，", "。", "；", "："}:
                yield buffer
                buffer = ""
    if buffer:
        yield buffer


def _text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    return text_size(draw, text, font)[0]


def _line_height(draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont) -> int:
    return max(text_size(draw, "国Ag", font)[1], getattr(font, "size", 24))


def _truncate_lines(
    lines: list[str],
    *,
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    if not lines:
        return lines
    last = lines[-1]
    suffix = "..."
    while last and _text_width(draw, last + suffix, font) > max_width:
        last = last[:-1]
    lines[-1] = (last + suffix) if last else suffix
    return lines
