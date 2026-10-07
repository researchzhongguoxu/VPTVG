"""Public interface for the Solver component."""

from mathexplain.agents.solver.algebra_solver import AlgebraSolver
from mathexplain.agents.solver.component import Solver
from mathexplain.agents.solver.final_answer_repair import FinalAnswerRepair
from mathexplain.agents.solver.planner import DeepSeekPlanner
from mathexplain.agents.solver.router import SolverRouter
from mathexplain.agents.solver.verifier_guided_replanner import VerifierGuidedReplanner

__all__ = [
    "AlgebraSolver",
    "DeepSeekPlanner",
    "FinalAnswerRepair",
    "Solver",
    "SolverRouter",
    "VerifierGuidedReplanner",
]
