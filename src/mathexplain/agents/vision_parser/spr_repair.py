"""SPR repair placeholder for SGVP."""

from __future__ import annotations

from mathexplain.schemas.spr import SPR


class SPRRepair:
    """Placeholder repair agent that does not mutate SPR."""

    def repair(self, spr: SPR, validation_report: dict | None = None) -> tuple[SPR, dict]:
        """Return the original SPR and a skipped repair report."""

        return spr, {
            "repair_executed": False,
            "target": "spr",
            "reason": "SPR repair is not implemented in the skeleton.",
            "validation_report": validation_report or {},
        }
