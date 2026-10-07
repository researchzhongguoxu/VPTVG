"""EMR repair placeholder for SGVP."""

from __future__ import annotations

from mathexplain.schemas.emr import EMR


class EMRRepair:
    """Placeholder repair agent that does not mutate EMR."""

    def repair(self, emr: EMR, validation_report: dict | None = None) -> tuple[EMR, dict]:
        """Return the original EMR and a skipped repair report."""

        return emr, {
            "repair_executed": False,
            "target": "emr",
            "reason": "EMR repair is not implemented in the skeleton.",
            "validation_report": validation_report or {},
        }
