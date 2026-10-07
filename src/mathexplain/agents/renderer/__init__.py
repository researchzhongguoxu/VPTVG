"""Renderer agent entry point package.

This package owns the Renderer agent boundary. Low-level rendering helpers live in
``mathexplain.rendering``.
"""

from mathexplain.agents.renderer.component import Renderer, RenderResult

__all__ = ["Renderer", "RenderResult"]
