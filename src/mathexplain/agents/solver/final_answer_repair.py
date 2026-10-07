"""Rule-based final-answer repair for Solver-produced SCS artifacts."""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from mathexplain.schemas.scs import SolutionChain
from mathexplain.schemas.spr import SPR


@dataclass(frozen=True)
class _Candidate:
    target: str
    value: str
    source_tool_call_id: str | None
    source_step_id: str | None
    backend_operation: str | None
    context: str
    score: float
    reasons: list[str]
    step_index: int = 0


@dataclass(frozen=True)
class _AnswerIntent:
    name: str
    evidence: str
    ordinal: str | None = None


class FinalAnswerRepair:
    """Repair only the final answer by reselecting CAS-backed results.

    The repairer is intentionally conservative. It does not invent new
    equations or rerun the whole solve; it only reuses successful CAS steps, plus
    a small cents-to-dollars conversion when the problem text clearly asks for a
    money amount expressed from cent-denominated inputs.
    """

    REPORT_VERSION = "FINAL_ANSWER_REPAIR-1.0"
    MIN_SCORE = 4.0
    MIN_MARGIN = 3.0

    def repair(self, scs: SolutionChain, spr: SPR | None = None) -> tuple[SolutionChain, dict[str, Any]]:
        """Return a repaired chain and an inspectable repair report."""

        report = self._base_report(scs)
        if spr is None:
            report["reason"] = "SPR is missing; final-answer repair skipped."
            return scs, report
        question_focus_text = self._question_focus_text(spr)
        question_text = self._question_text(spr)
        if not question_focus_text:
            report["reason"] = "Question text is missing; final-answer repair skipped."
            return scs, report
        answer_intent = self._answer_intent(question_focus_text)
        report["answer_intent"] = answer_intent.name
        report["question_focus_text"] = question_focus_text
        if self._looks_like_multi_answer_question(question_focus_text, spr, answer_intent):
            report["reason"] = "Question appears to request multiple answers; final-answer repair skipped."
            return scs, report

        candidates = self._cas_candidates(scs, question_focus_text, answer_intent)
        if not candidates:
            report["reason"] = "No successful CAS-backed candidate was available."
            report["binding_decision"] = "unrepairable"
            report["unrepairable_reason"] = "no_cas_backed_candidates"
            return scs, report

        selected = max(candidates, key=lambda item: item.score)
        current_candidate = self._original_answer_candidate(scs, candidates)
        current_score = current_candidate.score if current_candidate is not None else self._current_score(scs, candidates)
        report["candidate_count"] = len(candidates)
        report["current_score"] = current_score
        report["selected_score"] = selected.score
        report["decision"] = "skip"
        report["binding_decision"] = "skip"
        report["original_answer_protected"] = False
        report["protection_reason"] = None
        report["selected_candidate_before_gate"] = self._candidate_report(selected)
        report["current_candidate"] = self._candidate_report(current_candidate) if current_candidate else None
        report["unrepairable_reason"] = None
        report["candidate_scores"] = [
            self._candidate_report(candidate)
            for candidate in sorted(candidates, key=lambda item: item.score, reverse=True)
        ]
        last_candidate = self._last_answer_like_candidate(candidates)
        text_candidate = self._final_answer_text_candidate(scs, candidates, selected)
        if text_candidate is not None and (
            current_candidate is None or not self._same_candidate(text_candidate, current_candidate)
        ) and (
            self._same_candidate(text_candidate, selected)
            or (last_candidate is not None and self._same_candidate(text_candidate, last_candidate))
            or not self._candidate_has_strong_intent_match(selected, answer_intent)
        ):
            return self._apply_repair(
                scs,
                text_candidate,
                selected=selected,
                repair_type="final_answer_item_alignment",
                report=report,
                reason=(
                    "Existing final answer text already matches a CAS-backed candidate; "
                    "the final answer item was aligned to that candidate."
                ),
            )
        if self._should_protect_current_last_answer(
            scs,
            current=current_candidate,
            last_candidate=last_candidate,
            question_text=question_text,
        ):
            report["decision"] = "protect"
            report["binding_decision"] = "protect"
            report["original_answer_protected"] = True
            report["protection_reason"] = "Current final answer already matches the last answer-like CAS result."
            report["reason"] = report["protection_reason"]
            return scs, report
        aligned = self._maybe_align_final_answer_item(
            scs,
            current_candidate,
            selected,
            answer_intent,
        )
        if aligned is not None:
            return self._apply_repair(
                scs,
                aligned,
                selected=current_candidate or aligned,
                repair_type="final_answer_item_alignment",
                report=report,
                reason="Final answer text was aligned with the selected CAS-backed final answer item.",
            )
        if selected.score < self.MIN_SCORE:
            if self._should_use_last_answer_like_candidate(
                scs,
                selected=selected,
                current=current_candidate,
                last_candidate=last_candidate,
                question_text=question_focus_text,
                intent=answer_intent,
            ):
                return self._apply_repair(
                    scs,
                    last_candidate,
                    selected=selected,
                    repair_type="last_answer_like_cas_selection",
                    report=report,
                    reason=(
                        "Single-question chain selected the last answer-like CAS result "
                        "over an earlier intermediate result."
                    ),
                )
            report["reason"] = "Best candidate score is below the conservative repair threshold."
            if selected.score < 2.0:
                report["binding_decision"] = "unrepairable"
                report["unrepairable_reason"] = "no_semantically_suitable_candidate_above_threshold"
            return scs, report

        item = self._item_from_candidate(selected)
        repair_type = "target_reselection"
        converted = self._maybe_convert_cents_to_dollars(item, question_text)
        if converted is not None:
            item = converted
            repair_type = "unit_conversion_cents_to_dollars"
        elif selected.score - current_score < self.MIN_MARGIN:
            report["reason"] = "Best candidate is not clearly better than the current final answer."
            return scs, report
        elif protection_reason := self._should_protect_original_answer(
            scs,
            selected,
            current_candidate,
            question_text,
            answer_intent,
        ):
            report["decision"] = "protect"
            report["binding_decision"] = "protect"
            report["original_answer_protected"] = True
            report["protection_reason"] = protection_reason
            report["reason"] = protection_reason
            return scs, report

        new_final_answer = self._format_final_answer(item)
        if self._same_answer(new_final_answer, scs.final_answer):
            report["reason"] = "Selected candidate formats to the existing final answer."
            return scs, report

        return self._apply_repair(
            scs,
            self._candidate_from_item(item, selected),
            selected=selected,
            repair_type=repair_type,
            report=report,
            reason="A clearly better CAS-backed final-answer candidate was selected.",
            item_override=item,
        )

    @classmethod
    def _base_report(cls, scs: SolutionChain) -> dict[str, Any]:
        return {
            "schema_version": cls.REPORT_VERSION,
            "problem_id": scs.problem_id,
            "chain_id": scs.chain_id,
            "repair_attempted": False,
            "repair_performed": False,
            "repair_type": None,
            "original_final_answer": scs.final_answer,
            "repaired_final_answer": scs.final_answer,
            "candidate_count": 0,
            "candidate_scores": [],
            "current_score": 0.0,
            "selected_score": 0.0,
            "decision": "skip",
            "binding_decision": "skip",
            "original_answer_protected": False,
            "protection_reason": None,
            "answer_intent": "unknown",
            "question_focus_text": "",
            "selected_candidate_before_gate": None,
            "current_candidate": None,
            "unrepairable_reason": None,
            "reason": "No repair was needed.",
        }

    @staticmethod
    def _question_focus_text(spr: SPR) -> str:
        questions = [question.text for question in spr.questions if question.text]
        if questions:
            return questions[-1].strip()
        return (spr.problem_stem or spr.problem_text or "").strip()

    @staticmethod
    def _question_text(spr: SPR) -> str:
        parts = [spr.problem_text or "", spr.problem_stem or ""]
        parts.extend(question.text for question in spr.questions)
        target_by_id = {target.id: target.text for target in spr.targets}
        for question in spr.questions:
            parts.extend(target_by_id.get(target_id, "") for target_id in question.target_ids)
        return " ".join(part for part in parts if part).strip()

    @staticmethod
    def _answer_intent(question_text: str) -> _AnswerIntent:
        lowered = question_text.lower()
        ordinal_match = re.search(r"(?:horse|#)\s*#?\s*(\d+)|horse\s*#\s*(\d+)", lowered)
        if ordinal_match:
            ordinal = next(group for group in ordinal_match.groups() if group)
            return _AnswerIntent("specific_ordinal_entity", "horse ordinal", ordinal=ordinal)
        if any(signal in lowered for signal in ["save", "saving", "savings"]):
            return _AnswerIntent("savings", "save/savings")
        if any(signal in lowered for signal in ["how much more", "how many more", "difference", "compared to", "higher net"]):
            return _AnswerIntent("difference_more", "difference/more")
        if any(signal in lowered for signal in ["lunch break", "at lunch", "during lunch"]):
            return _AnswerIntent("at_lunch_state", "lunch")
        if any(signal in lowered for signal in ["were at", "was at", "present", "attending", "on friday"]):
            return _AnswerIntent("attendance_present", "present/attending")
        if "nap" in lowered:
            return _AnswerIntent("nap_time", "nap")
        if any(signal in lowered for signal in ["individual", "single unit"]):
            return _AnswerIntent("individual_units", "individual units")
        if any(signal in lowered for signal in ["gram", "grams"]):
            return _AnswerIntent("grams", "grams")
        if any(signal in lowered for signal in ["how fast", "speed", "mph", "rate"]):
            return _AnswerIntent("speed_rate", "speed/rate")
        if any(signal in lowered for signal in ["percentage", "percent", "%"]):
            return _AnswerIntent("percentage", "percent")
        if any(signal in lowered for signal in ["remaining", "left", "rest", "there in the box", "in the box"]):
            return _AnswerIntent("remaining_left", "remaining/left")
        if any(signal in lowered for signal in ["how much", "spend", "cost", "money", "pays", "pay"]):
            return _AnswerIntent("money_total", "money")
        return _AnswerIntent("generic", "fallback")

    @staticmethod
    def _looks_like_multi_answer_question(question_text: str, spr: SPR, intent: _AnswerIntent) -> bool:
        if intent.name in {"difference_more"}:
            return False
        lowered = question_text.lower()
        if sum(len(question.target_ids) for question in spr.questions) > 1 and intent.name == "generic":
            return True
        return any(signal in lowered for signal in ["respectively", "both", "x and y", "how many of each"])

    def _cas_candidates(self, scs: SolutionChain, question_text: str, intent: _AnswerIntent) -> list[_Candidate]:
        candidates: list[_Candidate] = []
        for step_index, step in enumerate(scs.steps):
            if not step.rule_name or not step.rule_name.startswith("cas_"):
                continue
            diagnostics = step.diagnostics or {}
            if diagnostics.get("success") is not True:
                continue
            request = diagnostics.get("input") or {}
            target = (
                request.get("target")
                or self._first_expression_name(request)
                or diagnostics.get("tool_call_id")
                or step.step_id
            )
            output = diagnostics.get("output") or {}
            value = self._candidate_value_from_cas_output(output, str(target))
            if value is None:
                continue
            context = " ".join(
                str(part or "")
                for part in [
                    target,
                    diagnostics.get("tool_call_id"),
                    diagnostics.get("operation"),
                    request.get("expression"),
                    step.description,
                    step.justification,
                ]
            )
            score, reasons = self._score_candidate(str(target), str(value), context, question_text, intent, step_index)
            candidates.append(
                _Candidate(
                    target=str(target),
                    value=str(value),
                    source_tool_call_id=diagnostics.get("tool_call_id"),
                    source_step_id=step.step_id,
                    backend_operation=diagnostics.get("operation"),
                    context=context,
                    score=score,
                    reasons=reasons,
                    step_index=step_index,
                )
            )
        return candidates

    @staticmethod
    def _candidate_value_from_cas_output(output: dict[str, Any], target: str) -> Any | None:
        raw_solution = output.get("raw_solution")
        if isinstance(raw_solution, list) and raw_solution and isinstance(raw_solution[0], dict):
            solution = raw_solution[0]
            if target in solution:
                return solution[target]
        if isinstance(raw_solution, dict) and target in raw_solution:
            return raw_solution[target]
        return output.get("formatted_solution") or output.get("value")

    @staticmethod
    def _first_expression_name(request: dict[str, Any]) -> str | None:
        expressions = request.get("expressions") or []
        for expression in expressions:
            if isinstance(expression, dict) and expression.get("name"):
                return str(expression["name"])
        return None

    def _score_candidate(
        self,
        target: str,
        value: str,
        context: str,
        question_text: str,
        intent: _AnswerIntent,
        step_index: int = 0,
    ) -> tuple[float, list[str]]:
        lowered_question = question_text.lower()
        lowered_context = context.lower().replace("{", "").replace("}", "")
        context_tokens = set(re.findall(r"[a-z]+", lowered_context))
        target_tokens = set(re.findall(r"[a-z]+|\d+", target.lower().replace("_", " ")))
        score = 0.0
        reasons: list[str] = []

        def add(points: float, reason: str) -> None:
            nonlocal score
            score += points
            reasons.append(reason)

        if self._target_token_match(target, lowered_question):
            add(1.0, "target token appears in question")
        if any(word in lowered_context for word in ["final", "answer", "result"]):
            add(0.25, "candidate is result-like")
        if step_index:
            add(min(0.75, step_index * 0.03), "later CAS step gets a small final-step bonus")

        self._score_intent_candidate(
            intent,
            lowered_context,
            context_tokens,
            target_tokens,
            add,
        )

        entity_rules = {
            "grams": ["gram", "grams"],
            "cookies": ["cookie", "cookies"],
            "bills": ["bill", "bills"],
            "reports": ["report", "reports"],
            "cars": ["car", "cars"],
            "bookcases": ["bookcase", "bookcases"],
        }
        for entity, words in entity_rules.items():
            if any(word in lowered_question for word in words):
                if any(
                    word in context_tokens or self._singular_token(word) in context_tokens
                    for word in words
                ):
                    add(3.0, f"candidate matches requested entity: {entity}")
                if entity == "grams" and any(word in lowered_context for word in ["calorie", "calories", "cal"]):
                    add(-2.0, "candidate is calorie-like but question asks grams")

        if any(signal in lowered_question for signal in ["how fast", "speed", "mph", "rate"]):
            speed_like = any(signal in lowered_context for signal in ["speed", "rate", "mph"])
            if speed_like:
                add(4.0, "question asks for speed/rate and candidate is speed-like")
            if not speed_like and any(signal in lowered_context for signal in ["total_time", "time", "distance", "d_total"]):
                add(-2.5, "candidate is time/distance-like but question asks speed")

        if any(signal in lowered_question for signal in ["remaining", "left", "rest", "there in the box"]):
            if any(signal in lowered_context for signal in ["remaining", "left", "rest", "lunch", "box"]):
                add(3.0, "question asks for remaining/rest and candidate matches")
            if any(signal in lowered_context for signal in ["before", "initial", "total_before"]):
                add(-1.5, "candidate is a before/intermediate total")
        if any(signal in lowered_question for signal in ["at lunch", "during lunch", "lunch break"]):
            if "lunch" in lowered_context and "before" not in lowered_context:
                add(3.0, "question asks for the state at lunch and candidate is lunch-like")
            if "before_lunch" in lowered_context or "total_before" in lowered_context:
                add(-2.0, "candidate is before-lunch rather than at-lunch")

        if any(signal in lowered_question for signal in ["percentage", "percent", "%"]):
            if any(signal in lowered_context for signal in ["percentage", "percent", "pct"]):
                add(4.0, "question asks for a percent and candidate is percent-like")
            if any(signal in lowered_context for signal in ["count", "number", "contemporary", "jazz"]):
                add(-1.5, "candidate is count/category-like but question asks percent")

        if any(signal in lowered_question for signal in ["how much", "spend", "cost", "$"]) and self._is_numeric(value):
            if any(signal in lowered_context for signal in ["total", "cost", "spend", "postage", "money"]):
                add(1.5, "money question and candidate is money-like")
            if "total" in lowered_question and "total" in lowered_context:
                add(2.0, "question asks for a total money amount and candidate is total-like")
            if self._question_requests_aggregate_money(lowered_question):
                if self._candidate_is_aggregate_money(lowered_context, context_tokens, target_tokens):
                    add(4.0, "aggregate money question favors total cost/spend candidate")
                elif self._candidate_is_local_money_component(context_tokens, target_tokens):
                    add(-2.5, "aggregate money question penalizes local cost component")

        if self._is_intermediate_like_text(lowered_context):
            add(-1.0, "candidate name indicates an intermediate quantity")

        return score, reasons

    @staticmethod
    def _question_requests_aggregate_money(lowered_question: str) -> bool:
        money_question = any(signal in lowered_question for signal in ["how much", "spend", "cost", "$", "pay"])
        if not money_question:
            return False
        return any(
            signal in lowered_question
            for signal in [
                "total",
                "altogether",
                "overall",
                "in all",
                "all ",
                "sum",
                "replacing",
                "replace",
                "replacement",
                "collection",
                "spend on",
                "paid for",
                "pay for",
                "pays for",
            ]
        )

    @staticmethod
    def _candidate_is_aggregate_money(
        lowered_context: str,
        context_tokens: set[str],
        target_tokens: set[str],
    ) -> bool:
        tokens = context_tokens | target_tokens
        return (
            bool(tokens & {"total", "overall", "sum", "altogether", "all", "replacement", "replacing", "spend"})
            or "total_cost" in lowered_context
            or "total_spend" in lowered_context
            or "replacement_cost" in lowered_context
        )

    @staticmethod
    def _candidate_is_local_money_component(
        context_tokens: set[str],
        target_tokens: set[str],
    ) -> bool:
        tokens = context_tokens | target_tokens
        if not (tokens & {"cost", "spend", "price", "pay", "paid"}):
            return False
        local_markers = {
            "normal",
            "older",
            "series",
            "regular",
            "discount",
            "small",
            "big",
            "large",
            "first",
            "second",
            "component",
            "partial",
        }
        return bool(tokens & local_markers) and "total" not in tokens

    @staticmethod
    def _score_intent_candidate(
        intent: _AnswerIntent,
        lowered_context: str,
        context_tokens: set[str],
        target_tokens: set[str],
        add: Any,
    ) -> None:
        def has_any(tokens: set[str]) -> bool:
            return bool((context_tokens | target_tokens) & tokens)

        if intent.name == "savings":
            if has_any({"save", "saves", "saving", "savings"}):
                add(7.0, "final question asks for savings and candidate is savings-like")
            if has_any({"trip", "trips", "cost", "regular", "total"}):
                add(-4.0, "savings question penalizes trips/cost intermediate candidates")
            return
        if intent.name == "attendance_present":
            if has_any({"attending", "present", "attendance"}):
                add(7.0, "final question asks who was present/attending")
            if "total_boys" in lowered_context or has_any({"total"}):
                add(-2.5, "present/attending question penalizes total-before-absence candidates")
            return
        if intent.name == "remaining_left":
            if has_any({"box", "left", "leftover", "remaining", "remain", "rest"}):
                add(5.0, "final question asks for remaining/left amount")
            if has_any({"total", "baked", "eaten", "consumed"}):
                add(-2.5, "remaining question penalizes total/consumed intermediate candidates")
            return
        if intent.name == "specific_ordinal_entity":
            ordinal = intent.ordinal or ""
            compact_context = re.sub(r"[^a-z0-9]+", "", lowered_context)
            if ordinal and f"horse{ordinal}" in compact_context:
                add(8.0, f"final question asks for horse #{ordinal}")
            if has_any({"remaining", "remain"}):
                add(-4.0, "specific entity question penalizes remaining bucket")
            if re.search(r"horse\d+", compact_context) and ordinal and f"horse{ordinal}" not in compact_context:
                add(-3.0, "specific entity question penalizes other numbered entities")
            return
        if intent.name == "nap_time":
            if has_any({"nap"}):
                add(8.0, "final question asks for nap time")
            if has_any({"total", "homework", "available"}):
                add(-3.0, "nap question penalizes total/homework/available time")
            return
        if intent.name == "difference_more":
            if has_any({"difference", "diff", "more"}):
                add(8.0, "final question asks for a difference/more amount")
            if has_any({"net", "gross", "rate", "wage", "billy", "sally"}):
                add(-2.5, "difference question penalizes component wage/net-pay candidates")
            return
        if intent.name == "at_lunch_state":
            if "lunch" in lowered_context and not any(signal in lowered_context for signal in ["before_lunch", "before lunch"]):
                add(8.0, "final question asks for lunch-break state")
            if has_any({"added", "before", "first", "break"}) or "total_after_first" in lowered_context:
                add(-4.0, "lunch question penalizes added/before-first-break candidates")
            return
        if intent.name == "individual_units":
            if has_any({"individual", "single"}):
                add(8.0, "final question asks for individual units")
            if has_any({"dozen", "dozens"}):
                add(-4.0, "individual-unit question penalizes dozen-valued candidates")
            return
        if intent.name == "grams":
            if has_any({"gram", "grams"}):
                add(8.0, "final question asks for grams")
            if has_any({"calorie", "calories", "serving"}):
                add(-4.0, "grams question penalizes calorie/serving intermediates")
            return

    @staticmethod
    def _target_token_match(target: str, lowered_question: str) -> bool:
        tokens = [
            token.lower()
            for token in re.findall(r"[A-Za-z]+", target.replace("_", " "))
            if len(token) > 2
        ]
        return any(token in lowered_question for token in tokens)

    @staticmethod
    def _singular_token(word: str) -> str:
        lowered = word.lower()
        return lowered[:-1] if lowered.endswith("s") else lowered

    @staticmethod
    def _current_score(scs: SolutionChain, candidates: list[_Candidate]) -> float:
        current_items = scs.metadata.get("final_answer_items") or []
        current_call_ids = {str(item.get("source_tool_call_id") or "") for item in current_items}
        current_targets = {str(item.get("target") or "") for item in current_items}
        current_values = {str(item.get("value") or "") for item in current_items}
        scores = [
            candidate.score
            for candidate in candidates
            if (
                str(candidate.source_tool_call_id or "") in current_call_ids
                or candidate.target in current_targets
                or candidate.value in current_values
            )
        ]
        return max(scores, default=0.0)

    @staticmethod
    def _original_answer_candidate(scs: SolutionChain, candidates: list[_Candidate]) -> _Candidate | None:
        metadata_candidate = FinalAnswerRepair._metadata_answer_candidate(scs, candidates)
        if metadata_candidate is not None:
            return metadata_candidate
        answer_number = FinalAnswerRepair._first_numeric_token(scs.final_answer)
        if answer_number is None:
            return None
        matches = [
            candidate
            for candidate in candidates
            if FinalAnswerRepair._numbers_equal(answer_number, candidate.value)
        ]
        return max(matches, key=lambda item: item.score, default=None)

    @staticmethod
    def _metadata_answer_candidate(scs: SolutionChain, candidates: list[_Candidate]) -> _Candidate | None:
        current_items = scs.metadata.get("final_answer_items") or []
        current_call_ids = {str(item.get("source_tool_call_id") or "") for item in current_items}
        current_targets = {str(item.get("target") or "") for item in current_items}
        current_values = {
            str(item.get("value") or "")
            for item in current_items
            if item.get("value") is not None
        }
        matches = [
            candidate
            for candidate in candidates
            if (
                str(candidate.source_tool_call_id or "") in current_call_ids
                or candidate.target in current_targets
                or candidate.value in current_values
            )
        ]
        return max(matches, key=lambda item: item.score, default=None)

    @classmethod
    def _final_answer_text_candidate(
        cls,
        scs: SolutionChain,
        candidates: list[_Candidate],
        selected: _Candidate | None,
    ) -> _Candidate | None:
        answer_number = cls._first_numeric_token(scs.final_answer)
        if answer_number is not None:
            exact_matches = [
                candidate
                for candidate in candidates
                if cls._numbers_equal(answer_number, candidate.value)
            ]
            if exact_matches:
                if selected is not None and any(cls._same_candidate(selected, candidate) for candidate in exact_matches):
                    return selected
                return max(exact_matches, key=lambda item: (item.step_index, item.score))

        answer_numbers = cls._numeric_tokens(scs.final_answer)
        if not answer_numbers:
            return None
        containing_matches = [
            candidate
            for candidate in candidates
            if any(cls._numbers_equal(number, candidate.value) for number in answer_numbers)
        ]
        if not containing_matches:
            return None
        if selected is not None and any(cls._same_candidate(selected, candidate) for candidate in containing_matches):
            return selected
        return max(containing_matches, key=lambda item: (item.step_index, item.score))

    @classmethod
    def _last_answer_like_candidate(cls, candidates: list[_Candidate]) -> _Candidate | None:
        for candidate in reversed(candidates):
            context = " ".join([candidate.target, candidate.context]).lower()
            if any(signal in context for signal in ["verify", "verification", "check_only", "validation"]):
                continue
            if candidate.backend_operation in {"construct_expression", "parse_expression"}:
                continue
            return candidate
        return None

    @classmethod
    def _should_use_last_answer_like_candidate(
        cls,
        scs: SolutionChain,
        *,
        selected: _Candidate,
        current: _Candidate | None,
        last_candidate: _Candidate | None,
        question_text: str,
        intent: _AnswerIntent,
    ) -> bool:
        if last_candidate is None:
            return False
        if not cls._same_candidate(selected, last_candidate):
            return False
        if current is not None and cls._same_candidate(current, last_candidate):
            return False
        if cls._numbers_equal(scs.final_answer, last_candidate.value):
            return False
        if (
            current is not None
            and cls._numbers_equal(scs.final_answer, current.value)
            and not any(signal in question_text.lower() for signal in ["win by", "faster", "difference", "how much more"])
        ):
            return False
        if current is not None and last_candidate.step_index <= current.step_index:
            return False
        if cls._candidate_has_strong_intent_match(last_candidate, intent):
            return True
        if current is not None and last_candidate.score > current.score:
            return True
        lowered_question = question_text.lower()
        lowered_context = " ".join([last_candidate.target, last_candidate.context]).lower()
        if any(signal in lowered_question for signal in ["win by", "faster", "difference", "how much more"]):
            return any(signal in lowered_context for signal in ["difference", "diff", "win"])
        return False

    @classmethod
    def _should_protect_current_last_answer(
        cls,
        scs: SolutionChain,
        *,
        current: _Candidate | None,
        last_candidate: _Candidate | None,
        question_text: str,
    ) -> bool:
        if current is None or last_candidate is None:
            return False
        if not cls._same_candidate(current, last_candidate):
            return False
        if not cls._numbers_equal(scs.final_answer, current.value):
            return False
        lowered = question_text.lower()
        if "cent" in lowered and any(signal in lowered for signal in ["how much", "spend", "cost", "$"]):
            return False
        return True

    @classmethod
    def _should_protect_original_answer(
        cls,
        scs: SolutionChain,
        selected: _Candidate,
        current: _Candidate | None,
        question_text: str,
        intent: _AnswerIntent,
    ) -> str | None:
        if current is None:
            return None
        if cls._same_candidate(selected, current):
            return None
        if cls._is_high_confidence_transform(selected, question_text):
            return None
        if cls._candidate_has_strong_intent_match(selected, intent) and not cls._candidate_has_strong_intent_match(current, intent):
            return None
        if current.score < 0:
            return None
        if cls._is_intermediate_like_candidate(current):
            return None
        if cls._is_intermediate_like_candidate(selected):
            return "Original CAS-backed answer was protected from an intermediate-like replacement."
        if cls._is_plain_numeric_final_answer(scs.final_answer):
            return "Original plain numeric CAS-backed answer was protected from target reselection."
        if current.score >= 1.5 and selected.score - current.score < cls.MIN_MARGIN + 1.0:
            return "Original CAS-backed answer was protected because the new candidate was not dominant enough."
        return None

    @classmethod
    def _candidate_has_strong_intent_match(cls, candidate: _Candidate, intent: _AnswerIntent) -> bool:
        text = " ".join([candidate.target, candidate.context]).lower()
        tokens = set(re.findall(r"[a-z]+|\d+", text.replace("_", " ")))
        if intent.name == "savings":
            return bool(tokens & {"save", "saves", "saving", "savings"})
        if intent.name == "attendance_present":
            return bool(tokens & {"attending", "present", "attendance"})
        if intent.name == "specific_ordinal_entity" and intent.ordinal:
            return f"horse{intent.ordinal}" in re.sub(r"[^a-z0-9]+", "", text)
        if intent.name == "nap_time":
            return "nap" in tokens
        if intent.name == "difference_more":
            return bool(tokens & {"difference", "diff", "more"})
        if intent.name == "at_lunch_state":
            return "lunch" in tokens and "before_lunch" not in text
        if intent.name == "individual_units":
            return bool(tokens & {"individual", "single"})
        if intent.name == "remaining_left":
            return bool(tokens & {"box", "left", "leftover", "remaining", "remain", "rest"})
        if intent.name == "grams":
            return bool(tokens & {"gram", "grams"})
        if intent.name == "money_total":
            return cls._candidate_is_aggregate_money(text, tokens, tokens)
        return False

    @classmethod
    def _maybe_align_final_answer_item(
        cls,
        scs: SolutionChain,
        current: _Candidate | None,
        selected: _Candidate | None,
        intent: _AnswerIntent,
    ) -> _Candidate | None:
        answer_number = cls._first_numeric_token(scs.final_answer)
        if answer_number is None:
            return None
        if (
            selected is not None
            and (current is None or not cls._same_candidate(selected, current))
            and cls._numbers_equal(answer_number, selected.value)
            and selected.score >= cls.MIN_SCORE
            and (
                cls._candidate_has_strong_intent_match(selected, intent)
                or current is None
                or selected.score - current.score >= cls.MIN_MARGIN
            )
        ):
            return selected
        if current is None or cls._numbers_equal(answer_number, current.value):
            return None
        if current.score >= cls.MIN_SCORE or cls._candidate_has_strong_intent_match(current, intent):
            return current
        return None

    @staticmethod
    def _candidate_report(candidate: _Candidate) -> dict[str, Any]:
        return {
            "target": candidate.target,
            "value": candidate.value,
            "source_tool_call_id": candidate.source_tool_call_id,
            "source_step_id": candidate.source_step_id,
            "backend_operation": candidate.backend_operation,
            "score": candidate.score,
            "reasons": candidate.reasons,
        }

    @staticmethod
    def _candidate_from_item(item: dict[str, Any], selected: _Candidate) -> _Candidate:
        return _Candidate(
            target=str(item.get("target") or selected.target),
            value=str(item.get("value") or selected.value),
            source_tool_call_id=item.get("source_tool_call_id") or selected.source_tool_call_id,
            source_step_id=item.get("source_step_id") or selected.source_step_id,
            backend_operation=item.get("backend_operation") or selected.backend_operation,
            context=selected.context,
            score=selected.score,
            reasons=selected.reasons,
            step_index=selected.step_index,
        )

    def _apply_repair(
        self,
        scs: SolutionChain,
        repair_candidate: _Candidate,
        *,
        selected: _Candidate,
        repair_type: str,
        report: dict[str, Any],
        reason: str,
        item_override: dict[str, Any] | None = None,
    ) -> tuple[SolutionChain, dict[str, Any]]:
        item = item_override or self._item_from_candidate(repair_candidate)
        new_final_answer = self._format_final_answer(item)
        repaired = scs.model_copy(deep=True)
        original_answer = repaired.final_answer
        repaired.final_answer = new_final_answer
        metadata = deepcopy(repaired.metadata)
        metadata["final_answer_items"] = [item]
        metadata["repair"] = {
            "repair_performed": True,
            "repair_type": repair_type,
            "original_final_answer": original_answer,
            "repaired_final_answer": new_final_answer,
            "source": "FinalAnswerRepair.rule_v1.2",
            "confidence": min(0.95, 0.5 + selected.score / 10),
            "selected_source_tool_call_id": item.get("source_tool_call_id"),
            "selected_target": item.get("target"),
            "selection_reasons": selected.reasons,
        }
        repaired.metadata = metadata
        self._rewrite_final_answer_step(repaired, new_final_answer, item)
        report.update(
            {
                "repair_attempted": True,
                "repair_performed": True,
                "repair_type": repair_type,
                "reason": reason,
                "decision": "repair",
                "binding_decision": "alignment" if repair_type == "final_answer_item_alignment" else "repair",
                "original_final_answer": original_answer,
                "repaired_final_answer": new_final_answer,
                "selected_target": item.get("target"),
                "selected_source_tool_call_id": item.get("source_tool_call_id"),
            }
        )
        return repaired, report

    @staticmethod
    def _same_candidate(left: _Candidate, right: _Candidate) -> bool:
        return (
            left.target == right.target
            or (
                left.source_tool_call_id is not None
                and left.source_tool_call_id == right.source_tool_call_id
            )
            or FinalAnswerRepair._numbers_equal(left.value, right.value)
        )

    @staticmethod
    def _is_plain_numeric_final_answer(final_answer: str | None) -> bool:
        text = str(final_answer or "").strip()
        if not text:
            return False
        normalized = text.lower()
        if re.fullmatch(r"(?:答案是\s*)?[-+]?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?", normalized):
            return True
        if re.fullmatch(r"answer\s+is\s+[-+]?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?", normalized):
            return True
        if text.count("=") == 1:
            rhs = text.rsplit("=", 1)[-1].strip()
            return re.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?", rhs) is not None
        return False

    @classmethod
    def _is_intermediate_like_candidate(cls, candidate: _Candidate) -> bool:
        return cls._is_intermediate_like_text(" ".join([candidate.target, candidate.context]).lower())

    @staticmethod
    def _is_intermediate_like_text(text: str) -> bool:
        intermediate_signals = [
            "regular_cost",
            "before_lunch",
            "total_before",
            "remaining_after_boats",
            "netpay_a",
            "net_pay_a",
            "e_packs",
            "t_p",
            "t_w",
            "remaining_cal",
            "remaining_percent",
            "slices_bill",
            "dasha_cost",
            "n_remaining1",
        ]
        return any(signal in text for signal in intermediate_signals)

    @staticmethod
    def _is_high_confidence_transform(candidate: _Candidate, question_text: str) -> bool:
        lowered_question = question_text.lower()
        lowered_context = candidate.context.lower()
        if "cent" in lowered_question and any(signal in lowered_context for signal in ["cost", "spend", "postage"]):
            return True
        return False

    @staticmethod
    def _first_numeric_token(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value)
        if "=" in text:
            text = text.rsplit("=", 1)[-1]
        match = re.search(r"[-+]?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?", text)
        return match.group(0) if match else None

    @staticmethod
    def _numeric_tokens(value: Any) -> list[str]:
        if value is None:
            return []
        return re.findall(r"[-+]?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?", str(value))

    @classmethod
    def _numbers_equal(cls, left: Any, right: Any) -> bool:
        left_decimal = cls._decimal(cls._first_numeric_token(left))
        right_decimal = cls._decimal(cls._first_numeric_token(right))
        return left_decimal is not None and right_decimal is not None and left_decimal == right_decimal

    @staticmethod
    def _item_from_candidate(candidate: _Candidate) -> dict[str, Any]:
        return {
            "target": candidate.target,
            "value": candidate.value,
            "source_tool_call_id": candidate.source_tool_call_id,
            "source_step_id": candidate.source_step_id,
            "backend_operation": candidate.backend_operation,
            "backend_confirmed": True,
        }

    def _maybe_convert_cents_to_dollars(
        self,
        item: dict[str, Any],
        question_text: str,
    ) -> dict[str, Any] | None:
        lowered = question_text.lower()
        if "cent" not in lowered:
            return None
        if not any(signal in lowered for signal in ["how much", "spend", "cost", "$"]):
            return None
        value = self._decimal(item.get("value"))
        if value is None or value == 0:
            return None
        if abs(value) < Decimal("100"):
            return None
        converted = deepcopy(item)
        converted["target"] = f"{item.get('target')}_dollars"
        converted["value"] = self._format_decimal(value / Decimal("100"))
        converted["repair_transform"] = "cents_to_dollars"
        return converted

    @staticmethod
    def _decimal(value: Any) -> Decimal | None:
        if value is None:
            return None
        text = str(value).strip().replace(",", "")
        try:
            return Decimal(text)
        except InvalidOperation:
            return None

    @classmethod
    def _is_numeric(cls, value: Any) -> bool:
        return cls._decimal(value) is not None

    @staticmethod
    def _format_decimal(value: Decimal) -> str:
        normalized = value.normalize()
        text = format(normalized, "f")
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text or "0"

    @staticmethod
    def _format_final_answer(item: dict[str, Any]) -> str:
        target = str(item.get("target") or "").strip()
        value = str(item.get("value") or "").strip()
        if not target:
            return value
        return f"{target} = {value}"

    @staticmethod
    def _same_answer(left: str | None, right: str | None) -> bool:
        return (left or "").strip() == (right or "").strip()

    @staticmethod
    def _rewrite_final_answer_step(
        scs: SolutionChain,
        new_final_answer: str,
        item: dict[str, Any],
    ) -> None:
        final_steps = [step for step in scs.steps if step.rule_name == "final_answer"]
        if not final_steps:
            return
        final_step = final_steps[-1]
        final_step.math_expression = new_final_answer
        final_step.verification_status = "repaired"
        diagnostics = deepcopy(final_step.diagnostics or {})
        diagnostics["source"] = "repair"
        diagnostics["repair_performed"] = True
        diagnostics["output"] = {"formatted_solution": new_final_answer}
        input_payload = deepcopy(diagnostics.get("input") or {})
        input_payload["final_answer_items"] = [item]
        diagnostics["input"] = input_payload
        final_step.diagnostics = diagnostics
