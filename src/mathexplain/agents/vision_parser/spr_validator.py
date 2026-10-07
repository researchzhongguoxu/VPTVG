"""SPR validation adapter for SGVP."""

from __future__ import annotations

from mathexplain.schemas.spr import SPR, validate_spr_dict


class SPRValidator:
    """Run basic SPR schema validation."""

    def validate(self, spr: SPR) -> dict:
        """Validate an SPR object and return a compact validation report."""

        validate_spr_dict(spr.model_dump(mode="json"))
        return {
            "schema": "SPR-1.0",
            "valid": True,
            "validator": "pydantic",
        }
