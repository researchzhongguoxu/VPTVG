from pathlib import Path

from mathexplain.agents.problem_classifier import ProblemClassifier
from mathexplain.agents.solver.algebra_solver import AlgebraSolver
from mathexplain.agents.solver.calculation_request import CalculationRequestValidator
from mathexplain.agents.solver.final_answer_repair import FinalAnswerRepair
from mathexplain.agents.solver import DeepSeekPlanner, Solver
from mathexplain.agents.verifier import Verifier
from mathexplain.agents.vision_parser import EMRBuilder, VisionParser
from mathexplain.schemas.classification import ClassificationResult
from mathexplain.schemas.emr import EMR
from mathexplain.schemas.scs import SolutionChain, validate_solution_chain_dict
from mathexplain.schemas.spr import SPR
from mathexplain.services.cas import CASService


class FakePlanningClient:
    model = "fake-deepseek-v4-pro"

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        assert system_prompt
        assert user_prompt
        return (
            '{"tool_calls":[{"operation":"solve_for","variables":["x"],'
            '"equations":[{"lhs":"2*x + 3","rhs":"7"}],"target":"x",'
            '"reason":"Solve the provided equation.","evidence_span":"2*x + 3 = 7"}]}'
        )


class FakeEquationSystemAggregatePlanningClient:
    model = "fake-deepseek-v4-flash"

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        assert "tool_calls" in system_prompt + user_prompt
        return (
            '{"tool_calls":[{"id":"solve_system","operation":"solve_equation_system",'
            '"variables":["x","y"],'
            '"equations":[{"lhs":"x+8*y","rhs":"15"},{"lhs":"4*x+y","rhs":"29"}],'
            '"target":"x_and_y","reason":"Solve the system for x and y.",'
            '"evidence_span":"x + 8y = 15; 4x + y = 29"}]}'
        )


class FakeRatioTargetExpressionPlanningClient:
    model = "fake-qwen-plus"

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        assert "tool_calls" in system_prompt + user_prompt
        return (
            '{"tool_calls":[{"id":"solve_ratios","operation":"solve_equation_system",'
            '"variables":["a","b","c","d"],'
            '"equations":[{"lhs":"a/b","rhs":"4"},{"lhs":"b/c","rhs":"1/3"},{"lhs":"c/d","rhs":"6"}],'
            '"target":"a","reason":"Solve the ratio chain first.",'
            '"evidence_span":"a/b=4, b/c=1/3, c/d=6"}]}'
        )


class FakeRatioTargetExpressionWithInvalidFollowupPlanningClient:
    model = "fake-qwen-plus"

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        assert "tool_calls" in system_prompt + user_prompt
        return (
            '{"tool_calls":[{"id":"solve_ratio_chain","operation":"solve_equation_system",'
            '"variables":["a","b","c","d"],'
            '"equations":[{"lhs":"a/b","rhs":"4"},{"lhs":"b/c","rhs":"1/3"},{"lhs":"c/d","rhs":"6"}],'
            '"target":"d/a","reason":"Solve the ratio chain first.",'
            '"evidence_span":"a/b=4, b/c=1/3, c/d=6"},'
            '{"id":"final_answer","operation":"evaluate","variables":["a","b","c","d"],'
            '"expressions":[{"name":"ratio_d_over_a","expression":"d/a"}],'
            '"substitutions":{"a":"$solve_ratio_chain.a","b":"$solve_ratio_chain.b",'
            '"c":"$solve_ratio_chain.c","d":"$solve_ratio_chain.d"},'
            '"target":"ratio_d_over_a","reason":"Evaluate d/a.",'
            '"evidence_span":"what is d/a?"}]}'
        )


class FakeCoordinateMidpointPlanningClient:
    model = "fake-qwen-plus"

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        assert "tool_calls" in system_prompt + user_prompt
        return (
            '{"tool_calls":['
            '{"id":"solve_xB","operation":"solve_for","variables":["x_B"],'
            '"equations":[{"lhs":"(8 + x_B) / 2","rhs":"4"}],"target":"x_B",'
            '"reason":"Use the midpoint x-coordinate equation.",'
            '"evidence_span":"M(4,4), A(8,4)"}'
            ',{"id":"solve_yB","operation":"solve_for","variables":["y_B"],'
            '"equations":[{"lhs":"(4 + y_B) / 2","rhs":"4"}],"target":"y_B",'
            '"reason":"Use the midpoint y-coordinate equation.",'
            '"evidence_span":"M(4,4), A(8,4)"}'
            ',{"id":"sum_coords","operation":"evaluate","variables":[],'
            '"expressions":[{"name":"sum_coords","expression":"$solve_xB.x_B + $solve_yB.y_B"}],'
            '"target":"sum_coords","reason":"Sum point B coordinates.",'
            '"evidence_span":"sum of the coordinates of point B"}'
            ']}'
        )


def test_deepseek_planner_repairs_unescaped_quotes_inside_reason() -> None:
    raw = """
{
  "tool_calls": [
    {
      "id": "solve_system",
      "operation": "solve_equation_system",
      "variables": ["C", "R"],
      "expressions": [],
      "equations": [
        {"lhs": "C + R", "rhs": "35"},
        {"lhs": "2*C + 4*R", "rhs": "94"}
      ],
      "substitutions": {},
      "target": "",
      "reason": "根据问题"鸡兔同笼，共有头35个，脚94只"，设鸡有C只，兔有R只。",
      "evidence_span": "共有头35个，脚94只，问鸡和兔各有多少只"
    }
  ]
}
"""

    parsed = DeepSeekPlanner._parse_tool_call_json(raw)

    assert parsed["parse_error"] is None
    assert parsed["tool_calls"][0]["id"] == "solve_system"
    assert parsed["tool_calls"][0]["equations"][1]["lhs"] == "2*C + 4*R"
    assert '问题"鸡兔同笼' in parsed["tool_calls"][0]["reason"]


class FakeEquationSystemDerivedTotalPlanningClient:
    model = "fake-deepseek-v4-flash"

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        assert "tool_calls" in system_prompt + user_prompt
        return (
            '{"tool_calls":['
            '{"id":"solve_system","operation":"solve_equation_system",'
            '"variables":["x","y"],'
            '"equations":[{"lhs":"x+y","rhs":"452"},{"lhs":"y","rhs":"92"}],'
            '"target":"x","reason":"Solve for x from the system.",'
            '"evidence_span":"x + y = 452, y = 92"},'
            '{"id":"compute_total","operation":"evaluate",'
            '"variables":["x","y","total"],'
            '"expressions":[{"name":"total","expression":"x+y"}],'
            '"substitutions":{"x":"$solve_system.x","y":"$solve_system.y"},'
            '"target":"total","reason":"Compute the requested total.",'
            '"evidence_span":"total = x + y"}'
            ']}'
        )


class FakeLinearFunctionListTargetPlanningClient:
    model = "fake-deepseek-v4-flash"

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        assert "tool_calls" in system_prompt + user_prompt
        return (
            '{"tool_calls":['
            '{"id":"solve_kb","operation":"solve_equation_system",'
            '"variables":["k","b"],'
            '"equations":[{"lhs":"-k+b","rhs":"3"},{"lhs":"2*k+b","rhs":"0"}],'
            '"target":["k","b"],"reason":"Solve k and b from two points.",'
            '"evidence_span":"(-1, 3), (2, 0)"},'
            '{"id":"final_expr","operation":"substitute","variables":["x","k","b"],'
            '"expressions":[{"name":"y_expr","expression":"k*x+b"}],'
            '"substitutions":{"k":"$solve_kb.k","b":"$solve_kb.b"},'
            '"target":"y_expr","reason":"Substitute k and b into y = kx + b.",'
            '"evidence_span":"Find the explicit equation."}'
            ']}'
        )


class FakeLinearFunctionMissingSubstituteExpressionPlanningClient:
    model = "fake-deepseek-v4-flash"

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        assert "tool_calls" in system_prompt + user_prompt
        return (
            '{"tool_calls":['
            '{"id":"solve_kb","operation":"solve_equation_system",'
            '"variables":["k","b"],'
            '"equations":[{"lhs":"5","rhs":"2*k+b"},{"lhs":"1","rhs":"b"}],'
            '"target":"","reason":"Solve k and b from conditions.",'
            '"evidence_span":"(2, 5), x = 0 gives y = 1"},'
            '{"id":"construct_expr","operation":"construct_expression",'
            '"variables":["k","b","x"],'
            '"expressions":[{"name":"line_expr","expression":"k*x+b"}],'
            '"target":"line_expr","reason":"Construct the line expression.",'
            '"evidence_span":"y = kx + b"},'
            '{"id":"substitute_vals","operation":"substitute",'
            '"variables":["x"],'
            '"expressions":[],"substitutions":{"k":"$solve_kb.k","b":"$solve_kb.b"},'
            '"target":"line_expr","reason":"Substitute k and b into the line expression.",'
            '"evidence_span":"Find the explicit equation."}'
            ']}'
        )


class FakeLinearFunctionCoefficientOnlyPlanningClient:
    model = "fake-deepseek-v4-flash"

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        assert "tool_calls" in system_prompt + user_prompt
        return (
            '{"tool_calls":['
            '{"id":"solve_b","operation":"solve_for",'
            '"variables":["b"],"equations":[{"lhs":"b","rhs":"1"}],'
            '"target":"b","reason":"Solve b from x=0,y=1.",'
            '"evidence_span":"x=0,y=1"},'
            '{"id":"solve_k","operation":"solve_for",'
            '"variables":["k","b"],"equations":[{"lhs":"5","rhs":"2*k+b"}],'
            '"substitutions":{"b":"$solve_b.b"},"target":"k",'
            '"reason":"Solve k from point (2,5).",'
            '"evidence_span":"(2,5)"}'
            ']}'
        )


class FakeDerivativePointValuePlanningClient:
    model = "fake-deepseek-v4-flash"

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        assert "tool_calls" in system_prompt + user_prompt
        return (
            '{"tool_calls":['
            '{"id":"construct_v","operation":"construct_expression",'
            '"variables":["t"],'
            '"expressions":[{"name":"v","expression":"3*t^2 - 6*t"}],'
            '"target":"v","reason":"Construct the velocity expression.",'
            '"evidence_span":"v(t) = s prime (t)"},'
            '{"id":"eval_v_at_2","operation":"evaluate",'
            '"variables":["t"],'
            '"expressions":[{"name":"v","expression":"$construct_v.v"}],'
            '"substitutions":{"t":"2"},"target":"v",'
            '"reason":"Evaluate velocity at t = 2.",'
            '"evidence_span":"t = 2"}'
            ']}'
        )


class FakeSymbolicEvaluateFallbackPlanningClient:
    model = "fake-deepseek-v4-flash"

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        assert "tool_calls" in system_prompt + user_prompt
        return (
            '{"tool_calls":[{"id":"eval_param_expr","operation":"evaluate",'
            '"variables":["x","a"],'
            '"expressions":[{"name":"param_expr_at_1","expression":"x - 2*a"}],'
            '"substitutions":{"x":"1"},"target":"param_expr_at_1",'
            '"reason":"Substitute x = 1 and keep the parameter expression.",'
            '"evidence_span":"x = 1"}]}'
        )


class FakeDerivativeFunctionNotationPlanningClient:
    model = "fake-deepseek-v4-flash"

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        assert "tool_calls" in system_prompt + user_prompt
        return (
            '{"tool_calls":[{"id":"deriv","operation":"evaluate",'
            '"variables":["x"],'
            '"expressions":[{"name":"deriv_result","expression":"derivative(x^4 - 2*x^3 + 5*x, x)"}],'
            '"target":"deriv_result","reason":"Differentiate the polynomial.",'
            '"evidence_span":"y = x^4 - 2x^3 + 5x"}]}'
        )


class FakeLinearFunctionLocalCoefficientPlanningClient:
    model = "fake-deepseek-v4-flash"

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        assert "tool_calls" in system_prompt + user_prompt
        return (
            '{"tool_calls":['
            '{"id":"solve_kb","operation":"solve_equation_system",'
            '"variables":["k","b"],'
            '"equations":[{"lhs":"2*k + b","rhs":"90"},{"lhs":"5*k + b","rhs":"180"}],'
            '"target":"k_and_b","reason":"Solve local coefficients for y = kx + b.",'
            '"evidence_span":"(2,90), (5,180)"},'
            '{"id":"final_line","operation":"substitute",'
            '"variables":["x","k","b"],'
            '"expressions":[{"name":"y_expr","expression":"k*x + b"}],'
            '"substitutions":{"k":"$solve_kb.k","b":"$solve_kb.b"},'
            '"target":"y_expr","reason":"Build explicit function expression.",'
            '"evidence_span":"求 y 与 x 的函数解析式"}'
            ']}'
        )


class FakeBareExpressionNameVertexPlanningClient:
    model = "fake-deepseek-v4-flash"

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        assert "tool_calls" in system_prompt + user_prompt
        return (
            '{"tool_calls":['
            '{"id":"construct_quadratic","operation":"construct_expression",'
            '"variables":["x"],'
            '"expressions":[{"name":"quadratic","expression":"x^2 - 4*x + 3"}],'
            '"reason":"Construct the quadratic expression.",'
            '"evidence_span":"y = x^2 - 4x + 3"},'
            '{"id":"calc_h","operation":"evaluate",'
            '"variables":[],"expressions":[{"name":"h_value","expression":"-(-4)/(2*1)"}],'
            '"target":"h_value","reason":"Compute vertex x-coordinate.",'
            '"evidence_span":"顶点坐标"},'
            '{"id":"substitute_y","operation":"substitute",'
            '"variables":["x"],'
            '"expressions":[{"name":"k_value","expression":"quadratic"}],'
            '"substitutions":{"x":"$calc_h.h_value"},'
            '"target":"k_value","reason":"Evaluate y at the vertex x-coordinate.",'
            '"evidence_span":"顶点坐标"}'
            ']}'
        )


class FakeTangentEquationSubstitutePlanningClient:
    model = "fake-deepseek-v4-flash"

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        assert "tool_calls" in system_prompt + user_prompt
        return (
            '{"tool_calls":['
            '{"id":"eval_y0","operation":"evaluate",'
            '"variables":["x"],'
            '"expressions":[{"name":"y0","expression":"x^2 + 2*x + 1"}],'
            '"substitutions":{"x":"1"},"target":"y0",'
            '"reason":"Evaluate y at x=1.","evidence_span":"x = 1"},'
            '{"id":"eval_m","operation":"evaluate",'
            '"variables":["x"],'
            '"expressions":[{"name":"m","expression":"2*x + 2"}],'
            '"substitutions":{"x":"1"},"target":"m",'
            '"reason":"Evaluate slope at x=1.","evidence_span":"切线斜率"},'
            '{"id":"substitute_tangent","operation":"substitute",'
            '"variables":["x","M","Y0"],'
            '"expressions":[{"name":"tangent_eq","expression":"y = M*(x - 1) + Y0"}],'
            '"substitutions":{"M":"$eval_m.m","Y0":"$eval_y0.y0"},'
            '"target":"tangent_eq","reason":"Substitute slope and point into tangent equation.",'
            '"evidence_span":"切线方程"}'
            ']}'
        )


class FakeLinearFunctionFinalExpressionAsCoefficientPlanningClient:
    model = "fake-deepseek-v4-flash"

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        assert "tool_calls" in system_prompt + user_prompt
        return (
            '{"tool_calls":['
            '{"id":"solve_k","operation":"solve_for",'
            '"variables":["k"],"equations":[{"lhs":"2*k + 1","rhs":"0"}],'
            '"target":"k","reason":"Solve the slope.",'
            '"evidence_span":"x-intercept condition"},'
            '{"id":"substitute_final","operation":"substitute",'
            '"variables":["x","k"],'
            '"expressions":[{"name":"linear_expr","expression":"k*x + 3"}],'
            '"substitutions":{"k":"$solve_k.k"},'
            '"target":"k","reason":"Substitute the slope to get the function expression.",'
            '"evidence_span":"Find the function expression."}'
            ']}'
        )


class FakeRevenueExpressionAndMaximumPricePlanningClient:
    model = "fake-deepseek-v4-flash"

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        assert "tool_calls" in system_prompt + user_prompt
        return (
            '{"tool_calls":['
            '{"id":"build_revenue","operation":"construct_expression",'
            '"variables":["x"],'
            '"expressions":[{"name":"y_expr","expression":"200*x - 4*x^2"}],'
            '"target":"y_expr","reason":"Revenue is price times quantity.",'
            '"evidence_span":"price is x and quantity is 200 - 4x"},'
            '{"id":"solve_max_price","operation":"solve_for",'
            '"variables":["x"],'
            '"equations":[{"lhs":"-8*x + 200","rhs":"0"}],'
            '"target":"x","reason":"Set the derivative to zero for maximum revenue.",'
            '"evidence_span":"maximum revenue"}'
            ']}'
        )


class FakeQuadraticVertexCoordinatesPlanningClient:
    model = "fake-deepseek-v4-flash"

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        assert "tool_calls" in system_prompt + user_prompt
        return (
            '{"tool_calls":['
            '{"id":"solve_c","operation":"solve_for",'
            '"variables":["c"],'
            '"equations":[{"lhs":"2","rhs":"1**2 - 4*1 + c"}],'
            '"target":"c","reason":"Use point (1, 2) to solve c.",'
            '"evidence_span":"passes through (1, 2)"},'
            '{"id":"step2_x_vertex","operation":"evaluate",'
            '"variables":[],"expressions":[{"name":"vertex_x_expr","expression":"(-(-4))/(2*1)"}],'
            '"target":"vertex_x_expr","reason":"Compute vertex x-coordinate.",'
            '"evidence_span":"vertex coordinates"},'
            '{"id":"step3_y_vertex","operation":"substitute",'
            '"variables":["x","c"],'
            '"expressions":[{"name":"y_vertex_expr","expression":"x**2 - 4*x + c"}],'
            '"substitutions":{"x":"2","c":"$solve_c.c"},'
            '"target":"y_vertex_expr","reason":"Substitute vertex x and c to get vertex y.",'
            '"evidence_span":"vertex coordinates"}'
            ']}'
        )


class FakeQuadraticVertexLocalSymbolPlanningClient:
    model = "fake-deepseek-v4-flash"

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        assert "tool_calls" in system_prompt + user_prompt
        return (
            '{"tool_calls":['
            '{"id":"solve_c","operation":"solve_equation_system",'
            '"variables":["c"],'
            '"equations":[{"lhs":"2","rhs":"1 - 4 + c"}],'
            '"target":"c","reason":"Solve c from point (1, 2).",'
            '"evidence_span":"passes through (1, 2)"},'
            '{"id":"vertex_x","operation":"evaluate",'
            '"variables":[],"expressions":[{"name":"vertex_x","expression":"-(-4)/(2*1)"}],'
            '"target":"vertex_x","reason":"Compute vertex x-coordinate.",'
            '"evidence_span":"vertex coordinates"},'
            '{"id":"vertex_y","operation":"evaluate",'
            '"variables":["x_v","c"],'
            '"expressions":[{"name":"vertex_y","expression":"x_v^2 - 4*x_v + c"}],'
            '"substitutions":{"x_v":"$vertex_x.vertex_x","c":"$solve_c.c"},'
            '"target":"vertex_y","reason":"Evaluate the vertex y-coordinate.",'
            '"evidence_span":"vertex coordinates"}'
            ']}'
        )


class FakeQuadraticInterceptPartialPlanningClient:
    model = "fake-deepseek-v4-flash"

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        assert "tool_calls" in system_prompt + user_prompt
        return (
            '{"tool_calls":['
            '{"id":"solve_x_partial","operation":"solve_for",'
            '"variables":["x"],'
            '"equations":[{"lhs":"x**2 - 5*x + 6","rhs":"0"}],'
            '"target":"x","reason":"Find x-axis intercept x values.",'
            '"evidence_span":"x-axis intercept coordinates"},'
            '{"id":"compute_y_partial","operation":"evaluate",'
            '"variables":["x"],'
            '"expressions":[{"name":"expr_y","expression":"x**2 - 5*x + 6"}],'
            '"substitutions":{"x":"0"},'
            '"target":"expr_y","reason":"Find y-axis intercept value.",'
            '"evidence_span":"y-axis intercept coordinates"}'
            ']}'
        )


class FakeTangentLinePartialPlanningClient:
    model = "fake-deepseek-v4-flash"

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        assert "tool_calls" in system_prompt + user_prompt
        return (
            '{"tool_calls":['
            '{"id":"calc_y0","operation":"evaluate",'
            '"variables":["x"],'
            '"expressions":[{"name":"y_at_1","expression":"x^2 + 2*x + 1"}],'
            '"substitutions":{"x":"1"},"target":"y_at_1",'
            '"reason":"Evaluate y at x = 1.","evidence_span":"x = 1"},'
            '{"id":"calc_m","operation":"evaluate",'
            '"variables":["x"],'
            '"expressions":[{"name":"m","expression":"2*x + 2"}],'
            '"substitutions":{"x":"1"},"target":"m",'
            '"reason":"Evaluate derivative at x = 1.","evidence_span":"slope at x = 1"},'
            '{"id":"construct_tangent","operation":"differentiate",'
            '"variables":["x","y"],'
            '"expressions":[{"name":"tangent_line","expression":"y = m*x + (y0 - m*1)"}],'
            '"substitutions":{"m":"$calc_m.m","y0":"$calc_y0.y_at_1"},'
            '"target":"tangent_line","reason":"Construct tangent line.",'
            '"evidence_span":"tangent line"}'
            ']}'
        )


class FakeTimeoutPlanningClient:
    model = "fake-deepseek-timeout"

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        assert system_prompt
        assert user_prompt
        raise TimeoutError("Request timed out.")


class FakeMath5PlanningClient:
    model = "fake-deepseek-v4-pro"

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        assert "construct_expression" in system_prompt + user_prompt
        return (
            '{"tool_calls":[{"operation":"construct_expression","variables":["x"],'
            '"expressions":[{"name":"father_age","expression":"3*x + 4"}],'
            '"target":"father_age","reason":"The father is four years older than '
            'three times Xiao Ming age.","evidence_span":"3 倍多 4 岁"}]}'
        )


class FakeInvalidExpressionPlanningClient:
    model = "fake-deepseek-v4-pro"

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        assert system_prompt
        assert user_prompt
        return (
            '{"tool_calls":[{"operation":"construct_expression","variables":["x"],'
            '"expressions":[{"name":"father_age","expression":"3**"}],'
            '"target":"father_age","reason":"bad expression","evidence_span":"bad"}]}'
        )


class FakeUndeclaredVariablePlanningClient:
    model = "fake-deepseek-v4-pro"

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        assert system_prompt
        assert user_prompt
        return (
            '{"tool_calls":[{"operation":"construct_expression","variables":["x"],'
            '"expressions":[{"name":"father_age","expression":"3*y + 4"}],'
            '"target":"father_age","reason":"bad variable","evidence_span":"bad"}]}'
        )


class FakeMath11TwoStepSpeedPlanningClient:
    model = "fake-deepseek-v4-flash"

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        assert "tool_calls" in system_prompt + user_prompt
        return (
            '{"tool_calls":['
            '{"id":"calc_sum","operation":"evaluate","variables":[],'
            '"expressions":[{"name":"relative_speed","expression":"900/5"}],'
            '"target":"relative_speed","reason":"计算客车和货车的速度和",'
            '"evidence_span":"甲、乙两地相距900千米，5小时相遇"},'
            '{"id":"solve","operation":"solve_equation_system",'
            '"variables":["v_bus","v_truck","sum_val"],'
            '"equations":['
            '{"lhs":"v_bus + v_truck","rhs":"sum_val"},'
            '{"lhs":"4*v_bus","rhs":"5*v_truck"}'
            '],'
            '"substitutions":{"sum_val":"$calc_sum.relative_speed"},'
            '"target":"v_bus","reason":"利用速度比和速度和求出客车速度",'
            '"evidence_span":"客车和货车的速度比是5:4"}'
            ']}'
        )


class FakeChoiceScaleAreaPlanningClient:
    model = "fake-deepseek-v4-flash"

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        assert "tool_calls" in system_prompt + user_prompt
        return (
            '{"tool_calls":['
            '{"id":"compute_new_side","operation":"evaluate","variables":["new_side"],'
            '"expressions":[{"name":"new_side","expression":"2*4"}],'
            '"target":"new_side","reason":"Scale the square side by 4.","evidence_span":"side 2, scale 4:1"},'
            '{"id":"compute_area","operation":"evaluate","variables":["new_side","area"],'
            '"expressions":[{"name":"area","expression":"new_side^2"}],'
            '"substitutions":{"new_side":"$compute_new_side.new_side"},'
            '"target":"area","reason":"Area of a square is side squared.","evidence_span":"square area"}'
            ']}'
        )


class FakeMath7PlanningClient:
    model = "fake-deepseek-v4-pro"

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        assert "tool_calls" in system_prompt + user_prompt
        return (
            '{"tool_calls":['
            '{"id":"solve_x","operation":"solve_equation_system","variables":["x"],'
            '"equations":[{"lhs":"28+(28+x)+2*(28+x)","rhs":"120"}],'
            '"target":"x","reason":"根据三天总页数列方程。","evidence_span":"三天一共看了120页"},'
            '{"id":"day2_pages","operation":"substitute","variables":["x"],'
            '"expressions":[{"name":"second_day_pages","expression":"28+x"}],'
            '"substitutions":{"x":"$solve_x.x"},"target":"second_day_pages",'
            '"reason":"第二天比第一天多 x 页，所以第二天为 28+x。",'
            '"evidence_span":"第二天比第一天多看了x页"}]}'
        )


class FakeMath9PlanningClient:
    model = "fake-deepseek-v4-pro"

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        assert "tool_calls" in system_prompt + user_prompt
        return (
            '{"tool_calls":['
            '{"id":"solve_x","operation":"solve_for","variables":["x"],'
            '"equations":[{"lhs":"6*x+12","rhs":"8*(x-3)"}],'
            '"target":"x","reason":"solve per-class pencils","evidence_span":"6 classes and 8 classes"},'
            '{"id":"calculate_T","operation":"evaluate","variables":["T"],'
            '"expressions":[{"name":"T","expression":"6*x+12"}],'
            '"substitutions":{"x":"$solve_x.x"},"target":"T",'
            '"reason":"substitute x into total pencils","evidence_span":"total pencils"}]}'
        )


class FakeMath10PlanningClient:
    model = "fake-deepseek-v4-pro"

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        assert "tool_calls" in system_prompt + user_prompt
        return (
            '{"tool_calls":['
            '{"id":"solve_T","operation":"solve_equation_system","variables":["T"],'
            '"equations":[{"lhs":"T","rhs":"(T/2 - 4) + ((T - (T/2 - 4))/2 - 2) + 18"}],'
            '"target":"T","reason":"solve total apples","evidence_span":"remaining 18 kg"},'
            '{"id":"calc_boxes","operation":"evaluate","variables":["T"],'
            '"expressions":[{"name":"N_boxes","expression":"T/8"}],'
            '"substitutions":{"T":"$solve_T.T"},"target":"N_boxes",'
            '"reason":"calculate boxes","evidence_span":"8 kg per box"}]}'
        )


class FakeMath10VerbosePlanningClient:
    model = "fake-deepseek-v4-pro"

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        assert "tool_calls" in system_prompt + user_prompt
        return (
            '{"tool_calls":['
            '{"id":"s1","operation":"construct_expression","variables":["T"],'
            '"expressions":[{"name":"S_m","expression":"T/2 - 4"}],'
            '"equations":[],"substitutions":null,"target":null,'
            '"reason":"上午卖出总数的一半少4千克","evidence_span":"上午卖出总数的一半少4千克"},'
            '{"id":"s2","operation":"construct_expression","variables":["T","S_m"],'
            '"expressions":[{"name":"S_a","expression":"(T - S_m)/2 - 2"}],'
            '"equations":[],"substitutions":null,"target":null,'
            '"reason":"下午卖出剩下的一半少2千克","evidence_span":"下午卖出剩下的一半少2千克"},'
            '{"id":"s3","operation":"substitute","variables":["T","S_m"],'
            '"expressions":[{"name":"S_a_explicit","expression":"$s2.S_a"}],'
            '"equations":[],"substitutions":{"S_m":"$s1.S_m"},"target":null,'
            '"reason":"代入上午表达式","evidence_span":"下午卖出剩下的一半少2千克"},'
            '{"id":"s4","operation":"construct_expression","variables":["T","S_m","S_a"],'
            '"expressions":[{"name":"R","expression":"T - S_m - S_a"}],'
            '"equations":[],"substitutions":null,"target":null,'
            '"reason":"构建剩余表达式","evidence_span":"这时还剩18千克苹果"},'
            '{"id":"s5","operation":"substitute","variables":["T","S_m","S_a"],'
            '"expressions":[{"name":"R_T","expression":"$s4.R"}],'
            '"equations":[],"substitutions":{"S_m":"$s1.S_m","S_a":"$s3.S_a_explicit"},'
            '"target":null,"reason":"代入得到只含T的表达式","evidence_span":"这时还剩18千克苹果"},'
            '{"id":"s6","operation":"solve_for","variables":["T"],'
            '"expressions":[],"equations":[{"lhs":"$s5.R_T","rhs":"18"}],'
            '"substitutions":null,"target":"T","reason":"求总重量","evidence_span":"还剩18千克"},'
            '{"id":"s7","operation":"evaluate","variables":["T"],'
            '"expressions":[{"name":"N","expression":"T/8"}],"equations":[],'
            '"substitutions":{"T":"$s6.T"},"target":null,'
            '"reason":"计算箱数","evidence_span":"每箱装8千克"}'
            ']}'
        )


class FakeMath10NumBoxesPlanningClient:
    model = "fake-deepseek-v4-pro"

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        assert "tool_calls" in system_prompt + user_prompt
        return (
            '{"tool_calls":['
            '{"id":"solve_T","operation":"solve_for","variables":["T"],'
            '"equations":[{"lhs":"T - (T/2 - 4) - ((T - (T/2 - 4))/2 - 2)","rhs":"18"}],'
            '"target":"T","reason":"solve total apples","evidence_span":"remaining 18 kg"},'
            '{"id":"compute_boxes","operation":"evaluate","variables":["T"],'
            '"expressions":[{"name":"NumBoxes","expression":"T/8"}],'
            '"substitutions":{"T":"$solve_T.T"},"target":"NumBoxes",'
            '"reason":"calculate boxes","evidence_span":"How many boxes if each box holds 8 kg?"}'
            ']}'
        )


class FakeMath10RemainingAliasPlanningClient:
    model = "fake-deepseek-v4-pro"

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        assert "tool_calls" in system_prompt + user_prompt
        return (
            '{"tool_calls":['
            '{"id":"solve_R","operation":"solve_for","variables":["R"],'
            '"equations":[{"lhs":"R/2 + 2","rhs":"18"}],'
            '"target":"R","reason":"solve remaining after morning sales",'
            '"evidence_span":"下午卖出剩下的一半少2千克，这时还剩18千克苹果"},'
            '{"id":"solve_x","operation":"solve_for","variables":["x"],'
            '"equations":[{"lhs":"x/2 + 4","rhs":"$solve_R.R"}],'
            '"target":"x","reason":"solve original total using remaining amount",'
            '"evidence_span":"上午卖出总数的一半少4千克"},'
            '{"id":"compute_boxes","operation":"evaluate","variables":["x","boxes"],'
            '"expressions":[{"name":"boxes","expression":"x/8"}],'
            '"substitutions":{"x":"$solve_x.x"},"target":"boxes",'
            '"reason":"calculate boxes","evidence_span":"每箱装8千克"}'
            ']}'
        )


class FakeMath10TotalRemainingAliasPlanningClient:
    model = "fake-deepseek-v4-pro"

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        assert "tool_calls" in system_prompt + user_prompt
        return (
            '{"tool_calls":['
            '{"id":"s1","operation":"solve_for","variables":["Total_remaining"],'
            '"equations":[{"lhs":"Total_remaining/2 + 2","rhs":"18"}],'
            '"target":"Total_remaining","reason":"solve final remaining relation",'
            '"evidence_span":"(剩余总量)/2 + 2 = 18"},'
            '{"id":"s2","operation":"solve_for","variables":["x","Total_remaining"],'
            '"equations":[{"lhs":"x/2 + 4","rhs":"Total_remaining"}],'
            '"substitutions":{"Total_remaining":"$s1.Total_remaining"},'
            '"target":"x","reason":"solve original total","evidence_span":"x/2 + 4 = 剩余总量"},'
            '{"id":"s3","operation":"evaluate","variables":["x"],'
            '"expressions":[{"name":"Boxes","expression":"x/8"}],'
            '"substitutions":{"x":"$s2.x"},"target":"Boxes",'
            '"reason":"calculate boxes","evidence_span":"每箱装8千克"}'
            ']}'
        )


class FakeMath10MissingSubstitutionPlanningClient:
    model = "fake-deepseek-v4-pro"

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        assert "tool_calls" in system_prompt + user_prompt
        return (
            '{"tool_calls":['
            '{"id":"solve_x","operation":"solve_for","variables":["x"],'
            '"equations":[{"lhs":"x/4 + 4","rhs":"18"}],'
            '"target":"x","reason":"solve original total","evidence_span":"remaining 18 kg"},'
            '{"id":"compute_boxes","operation":"evaluate","variables":["x","n"],'
            '"expressions":[{"name":"n","expression":"x/8"}],'
            '"substitutions":{},"target":"n",'
            '"reason":"calculate boxes","evidence_span":"8 kg per box"}'
            ']}'
        )


class FakeMath10SoldBreakdownPlanningClient:
    model = "fake-deepseek-v4-pro"

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        assert "tool_calls" in system_prompt + user_prompt
        return (
            '{"tool_calls":['
            '{"id":"solve_x","operation":"solve_equation_system",'
            '"variables":["x","Sold_AM","Sold_PM"],'
            '"equations":['
            '{"lhs":"Sold_AM","rhs":"x/2 - 4"},'
            '{"lhs":"Sold_PM","rhs":"(x - Sold_AM)/2 - 2"},'
            '{"lhs":"x - Sold_AM - Sold_PM","rhs":"18"}'
            '],'
            '"target":"x","reason":"求解原来运来的苹果总重量",'
            '"evidence_span":"上午卖出总数的一半少4千克，下午卖出剩下的一半少2千克，这时还剩18千克苹果"},'
            '{"id":"compute_boxes","operation":"evaluate","variables":["x"],'
            '"expressions":[{"name":"boxes","expression":"x/8"}],'
            '"substitutions":{"x":"$solve_x.x"},"target":"boxes",'
            '"reason":"计算箱数","evidence_span":"如果每箱装8千克苹果"}'
            ']}'
        )


class FakeQwenSPRGenerator:
    mode = "qwen"

    def generate(self, problem_id: str, image_path: str) -> SPR:
        return SPR.model_validate(
            {
                "schema_version": "SPR-1.0",
                "source_image": {"image_path": image_path},
                "problem_text": "A fake qwen word problem.",
                "problem_type": "word_problem",
                "knowledge_units": ["linear equations"],
                "variables": [{"symbol": "x", "domain": "integer"}],
                "confidence": {"overall": 0.9},
                "metadata": {"problem_id": problem_id, "parser_mode": "qwen"},
            }
        )


def math3_spr() -> SPR:
    return SPR.model_validate(
        {
            "schema_version": "SPR-1.0",
            "source_image": {"image_path": "data/inputs/images/math3.jpg"},
            "problem_text": (
                "7. 已知正实数 a,b 满足 2a+b=1. "
                "则 (5a+b)/(a^2+ab) 的最小值为 ( )"
            ),
            "problem_stem": "已知正实数 a,b 满足 2a+b=1. 则 (5a+b)/(a^2+ab) 的最小值为",
            "conditions": [
                {"id": "c1", "text": "a, b 为正实数", "linked_formula_ids": []},
                {"id": "c2", "text": "2a+b=1", "linked_formula_ids": ["f1"]},
            ],
            "questions": [
                {
                    "id": "q1",
                    "text": "求 (5a+b)/(a^2+ab) 的最小值",
                    "question_type": "select",
                    "target_ids": ["t1"],
                }
            ],
            "formulas": [
                {"id": "f1", "raw_text": "2a+b=1", "latex": "2a+b=1", "role": "condition"},
                {
                    "id": "f2",
                    "raw_text": "(5a+b)/(a^2+ab)",
                    "latex": "\\frac{5a+b}{a^2+ab}",
                    "role": "goal",
                },
            ],
            "variables": [
                {"symbol": "a", "description": "positive real number", "domain": "(0, \\infty)"},
                {"symbol": "b", "description": "positive real number", "domain": "(0, \\infty)"},
            ],
            "problem_type": "algebra",
            "knowledge_units": ["inequality", "algebraic_expression", "minimum_value"],
            "targets": [{"id": "t1", "text": "最小值", "target_type": "select"}],
            "confidence": {"overall": 0.95},
            "metadata": {"problem_id": "problem_math3", "parser_mode": "qwen"},
        }
    )


def geometry_classification_result() -> ClassificationResult:
    return ClassificationResult.model_validate(
        {
            "problem_id": "problem_geometry",
            "primary_type": "geometry",
            "task_type": "proof",
            "difficulty": "intermediate",
            "knowledge_units": ["circle_geometry"],
            "solver_route": "geometry_stub",
            "solver_config": {
                "solver_name": "GeometrySolverStub",
                "expected_emr_type": "unknown",
                "task_type": "proof",
                "strategy": "not_implemented_stub",
            },
            "requires_solver": True,
            "requires_verifier": True,
            "confidence": 0.7,
        }
    )


def unknown_classification_result() -> ClassificationResult:
    return ClassificationResult.model_validate(
        {
            "problem_id": "problem_unknown",
            "primary_type": "unknown",
            "task_type": "unknown",
            "difficulty": "unknown",
            "knowledge_units": [],
            "solver_route": "unknown",
            "solver_config": {
                "solver_name": "UnknownSolver",
                "expected_emr_type": "unknown",
                "task_type": "unknown",
                "strategy": "no_route",
            },
            "requires_solver": False,
            "requires_verifier": False,
            "confidence": 0.2,
        }
    )


def algebra_solver_classification_result(problem_id: str = "problem_eqsys") -> ClassificationResult:
    return ClassificationResult.model_validate(
        {
            "problem_id": problem_id,
            "primary_type": "algebra",
            "task_type": "equation_solving",
            "difficulty": "intermediate",
            "knowledge_units": ["equation_system"],
            "solver_route": "algebra_solver",
            "solver_config": {
                "solver_name": "AlgebraSolver",
                "expected_emr_type": "unknown",
                "task_type": "equation_solving",
                "strategy": "planner_tool_calls",
            },
            "requires_solver": True,
            "requires_verifier": True,
            "confidence": 0.8,
        }
    )


def math5_spr() -> SPR:
    return SPR.model_validate(
        {
            "schema_version": "SPR-1.0",
            "source_image": {"image_path": "data/inputs/images/math5.png"},
            "problem_text": "小明今年 x 岁，爸爸年龄是小明年龄的 3 倍多 4 岁，爸爸今年多少岁？",
            "problem_stem": "小明今年 x 岁，爸爸年龄是小明年龄的 3 倍多 4 岁",
            "questions": [
                {
                    "id": "q1",
                    "text": "爸爸今年多少岁？",
                    "question_type": "compute",
                    "target_ids": ["t1"],
                }
            ],
            "variables": [
                {"symbol": "x", "description": "小明今年年龄", "domain": "positive integer"}
            ],
            "problem_type": "algebra",
            "knowledge_units": ["word_problem", "algebraic_expression"],
            "targets": [{"id": "t1", "text": "爸爸今年年龄", "target_type": "compute"}],
            "confidence": {"overall": 0.95},
            "metadata": {"problem_id": "problem_math5", "parser_mode": "manual_test"},
        }
    )


def math5_emr() -> EMR:
    return EMR.model_validate(
        {
            "problem_id": "problem_math5",
            "source_spr_version": "SPR-1.0",
            "source_spr_id": "problem_math5",
            "source_problem_type": "algebra",
            "representation_type": "unknown",
            "variables": [
                {
                    "id": "var_x",
                    "symbol": "x",
                    "domain": "integer",
                    "description": "小明今年年龄",
                }
            ],
            "goals": [
                {
                    "id": "goal_1",
                    "goal_type": "compute",
                    "target": "爸爸今年年龄",
                    "target_variable_ids": ["var_x"],
                    "description": "构造爸爸今年年龄表达式",
                }
            ],
        }
    )


def math5_classification_result() -> ClassificationResult:
    return ClassificationResult.model_validate(
        {
            "problem_id": "problem_math5",
            "primary_type": "algebra",
            "task_type": "unsupported",
            "difficulty": "elementary",
            "knowledge_units": ["word_problem", "algebraic_expression"],
            "solver_route": "unsupported",
            "solver_config": {
                "solver_name": "UnsupportedSolver",
                "expected_emr_type": "unknown",
                "task_type": "unsupported",
                "strategy": "unsupported_route",
            },
            "requires_solver": False,
            "requires_verifier": False,
            "confidence": 0.2,
        }
    )


def math7_spr() -> SPR:
    return SPR.model_validate(
        {
            "schema_version": "SPR-1.0",
            "source_image": {"image_path": "data/inputs/images/math7.png"},
            "problem_text": "一本故事书，小明第一天看了28页，第二天比第一天多看了x页，第三天看的页数是第二天的2倍。这三天一共看了120页。第二天看了多少页？",
            "problem_stem": "小明第一天看了28页，第二天比第一天多看了x页，第三天看的页数是第二天的2倍。",
            "questions": [
                {
                    "id": "q1",
                    "text": "第二天看了多少页？",
                    "question_type": "compute",
                    "target_ids": ["t1"],
                }
            ],
            "variables": [
                {"symbol": "x", "description": "第二天比第一天多看的页数", "domain": "positive integer"}
            ],
            "problem_type": "word_problem",
            "knowledge_units": ["linear_equations", "algebraic_modeling"],
            "targets": [{"id": "t1", "text": "第二天看的页数", "target_type": "compute"}],
            "confidence": {"overall": 0.95},
            "metadata": {"problem_id": "problem_math7", "parser_mode": "manual_test"},
        }
    )


def math7_unknown_emr() -> EMR:
    return EMR.model_validate(
        {
            "problem_id": "problem_math7",
            "source_spr_version": "SPR-1.0",
            "source_spr_id": "problem_math7",
            "source_problem_type": "word_problem",
            "representation_type": "unknown",
        }
    )


def math7_classification_result() -> ClassificationResult:
    return ClassificationResult.model_validate(
        {
            "problem_id": "problem_math7",
            "primary_type": "word_problem",
            "task_type": "unknown",
            "difficulty": "unknown",
            "knowledge_units": ["linear_equations", "algebraic_modeling"],
            "solver_route": "unknown",
            "solver_config": {
                "solver_name": "UnknownSolver",
                "expected_emr_type": "unknown",
                "task_type": "unknown",
                "strategy": "no_route",
            },
            "requires_solver": False,
            "requires_verifier": False,
            "confidence": 0.2,
        }
    )


def math9_spr() -> SPR:
    return SPR.model_validate(
        {
            "schema_version": "SPR-1.0",
            "source_image": {"image_path": "data/inputs/images/math9.png"},
            "problem_text": (
                "学校买来一些铅笔，如果平均分给6个班，还剩12支；"
                "如果平均分给8个班，则每班比原来少3支。学校一共买来多少支铅笔？"
            ),
            "problem_stem": (
                "学校买来一些铅笔，如果平均分给6个班，还剩12支；"
                "如果平均分给8个班，则每班比原来少3支。"
            ),
            "questions": [
                {
                    "id": "q1",
                    "text": "学校一共买来多少支铅笔？",
                    "question_type": "compute",
                    "target_ids": ["t1"],
                }
            ],
            "variables": [
                {
                    "symbol": "T",
                    "description": "Total number of pencils",
                    "domain": "integer",
                },
                {
                    "symbol": "x",
                    "description": "Pencils per class when distributed to 6 classes",
                    "domain": "integer",
                },
            ],
            "problem_type": "word_problem",
            "knowledge_units": ["linear equations", "arithmetic operations"],
            "targets": [{"id": "t1", "text": "Total number of pencils", "target_type": "compute"}],
            "confidence": {"overall": 0.95},
            "metadata": {"problem_id": "problem_math9", "parser_mode": "manual_test"},
        }
    )


def math9_unknown_emr() -> EMR:
    return EMR.model_validate(
        {
            "problem_id": "problem_math9",
            "source_spr_version": "SPR-1.0",
            "source_spr_id": "problem_math9",
            "source_problem_type": "word_problem",
            "representation_type": "unknown",
        }
    )


def math10_spr() -> SPR:
    return SPR.model_validate(
        {
            "schema_version": "SPR-1.0",
            "source_image": {"image_path": "data/inputs/images/math10.png"},
            "problem_text": (
                "Fruit store apple word problem. First ask original weight; "
                "second ask number of boxes."
            ),
            "problem_stem": "Fruit store sold apples in morning and afternoon.",
            "questions": [
                {
                    "id": "q1",
                    "text": "How many kg apples were originally transported?",
                    "question_type": "compute",
                    "target_ids": ["t1"],
                },
                {
                    "id": "q2",
                    "text": "How many boxes if each box holds 8 kg?",
                    "question_type": "compute",
                    "target_ids": ["t2"],
                },
            ],
            "variables": [
                {"symbol": "T", "description": "Total apple weight", "domain": "integer"},
                {"symbol": "N_boxes", "description": "Number of boxes", "domain": "integer"},
            ],
            "problem_type": "word_problem",
            "knowledge_units": ["linear equations", "arithmetic operations"],
            "targets": [
                {"id": "t1", "text": "Total apple weight", "target_type": "compute"},
                {"id": "t2", "text": "Number of boxes", "target_type": "compute"},
            ],
            "confidence": {"overall": 0.95},
            "metadata": {"problem_id": "problem_math10", "parser_mode": "manual_test"},
        }
    )


def math10_unknown_emr() -> EMR:
    return EMR.model_validate(
        {
            "problem_id": "problem_math10",
            "source_spr_version": "SPR-1.0",
            "source_spr_id": "problem_math10",
            "source_problem_type": "word_problem",
            "representation_type": "unknown",
        }
    )


def test_env_example_contains_deepseek_placeholders() -> None:
    env_example = Path(".env.example").read_text(encoding="utf-8")

    assert "DEEPSEEK_API_KEY=your_deepseek_api_key_here" in env_example
    assert "DEEPSEEK_BASE_URL=https://api.deepseek.com" in env_example
    assert "DEEPSEEK_MODEL=deepseek-v4-pro" in env_example


def test_gitignore_ignores_dotenv() -> None:
    gitignore = Path(".gitignore").read_text(encoding="utf-8")

    assert ".env" in {line.strip() for line in gitignore.splitlines()}


def test_cas_service_solves_linear_equation() -> None:
    result = CASService().solve_equation_system(
        equations=[{"id": "eq_1", "lhs_sympy": "2*x + 3", "rhs_sympy": "7"}],
        variables=["x"],
        target="x",
    )

    assert result["success"] is True
    assert result["backend"] == "sympy"
    assert result["operation"] == "solve_equation_system"
    assert result["output"]["formatted_solution"] == "x = 2"
    assert result["output"]["raw_solution"] == [{"x": "2"}]


def test_cas_service_confirms_constructed_expression() -> None:
    result = CASService().construct_expression("3*x + 4", ["x"], target="father_age")

    assert result["success"] is True
    assert result["backend"] == "sympy"
    assert result["operation"] == "construct_expression"
    assert result["output"]["confirmed_expression"] == "3*x + 4"
    assert result["output"]["free_symbols"] == ["x"]


def test_cas_service_substitutes_expression() -> None:
    result = CASService().substitute("28+x", ["x"], {"x": "8/3"}, target="second_day_pages")

    assert result["success"] is True
    assert result["operation"] == "substitute"
    assert result["output"]["value"] == "92/3"


def test_cas_service_evaluates_expression_with_effective_variables() -> None:
    result = CASService().evaluate_expression(
        "6*x+12",
        ["T", "x"],
        {"x": "18"},
        target="T",
    )

    assert result["success"] is True
    assert result["operation"] == "evaluate"
    assert result["output"]["value"] == "120"


def test_cas_service_treats_caret_as_power() -> None:
    result = CASService().evaluate_expression(
        "new_side^2",
        ["new_side", "area"],
        {"new_side": "8"},
        target="area",
    )

    assert result["success"] is True
    assert result["operation"] == "evaluate"
    assert result["output"]["value"] == "64"


def test_cas_service_accepts_implicit_multiplication() -> None:
    result = CASService().evaluate_expression(
        "-x^2 + 4x + 2(x + 1)",
        ["x", "value"],
        {"x": "3"},
        target="value",
    )

    assert result["success"] is True
    assert result["operation"] == "evaluate"
    assert result["output"]["value"] == "11"


def test_cas_service_rejects_symbolic_evaluate_result() -> None:
    result = CASService().evaluate_expression(
        "x/8",
        ["x"],
        {},
        target="boxes",
    )

    assert result["success"] is False
    assert result["output"]["value"] == "x/8"
    assert result["missing_symbols"] == ["x"]
    assert "unresolved symbols" in result["errors"][0]


def test_cas_service_reports_missing_substitution_symbol() -> None:
    result = CASService().evaluate_expression(
        "6*x+12",
        ["T"],
        {"x": "18"},
        target="T",
    )

    assert result["success"] is False
    assert result["missing_symbols"] == ["x"]
    assert "Missing symbols" in result["errors"][0]


def test_cas_service_reports_tuple_parse_without_raising() -> None:
    result = CASService().parse_expression("x, y", ["x", "y"])

    assert result["success"] is False
    assert "tuple/list" in result["errors"][0]


def test_solver_returns_solution_chain_for_mock_linear_equation() -> None:
    parser_output = VisionParser().run("any/path.png")
    classification = ProblemClassifier().classify(parser_output.spr, parser_output.emr)

    chain = Solver().solve(parser_output.emr, classification)

    assert isinstance(chain, SolutionChain)
    assert chain.chain_id == "chain_main"
    assert chain.final_answer == "x = 2"
    assert "define_variable" in [step.rule_name for step in chain.steps]
    assert chain.steps[0].rule_name == "define_variable"
    assert chain.steps[1].rule_name == "given_equation"
    assert chain.steps[2].rule_name == "cas_solve_equation_system"
    assert chain.steps[3].rule_name == "final_answer"
    assert chain.steps[2].diagnostics["backend"] == "sympy"
    assert chain.steps[2].diagnostics["operation"] == "solve_equation_system"
    assert chain.steps[2].diagnostics["input"]["variables"] == ["x"]
    assert chain.steps[2].diagnostics["output"]["formatted_solution"] == "x = 2"
    assert chain.metadata["backend_confirmed"] is True
    assert chain.metadata["final_answer_items"] == [
        {
            "target": "x",
            "value": "2",
            "source_step_id": "chain_main_step_2",
            "backend_operation": "solve_equation_system",
            "backend_confirmed": True,
        }
    ]
    assert chain.metadata["variable_bindings"]["x"]["description"]
    assert chain.metadata["answer_bindings"][0]["target"] == "x"
    assert chain.metadata["solution_outline"]
    assert validate_solution_chain_dict(chain.model_dump(mode="json"))
    assert "x = 2" in chain.model_dump_json()


def test_solver_prefers_executable_emr_over_planner_tool_calls() -> None:
    parser_output = VisionParser().run("any/path.png")
    classification = ProblemClassifier().classify(parser_output.spr, parser_output.emr)
    planner = DeepSeekPlanner(client=FakePlanningClient())

    chain = Solver(planner=planner, use_planner=True).solve(
        parser_output.emr,
        classification,
        spr=parser_output.spr,
    )

    assert chain.final_answer == "x = 2"
    assert [step.rule_name for step in chain.steps] == [
        "define_variable",
        "given_equation",
        "cas_solve_equation_system",
        "final_answer",
    ]
    assert "planner_model" not in chain.metadata
    assert chain.metadata["backend_confirmed"] is True


def test_solver_evaluates_direct_emr_target_expression_after_solving() -> None:
    spr = SPR.model_validate(
        {
            "source_image": {"image_path": "math8.png"},
            "problem_text": "A school bought books. 5*x+18=8*x-12. Find total books.",
            "problem_type": "algebra",
            "knowledge_units": ["linear equations"],
            "variables": [{"symbol": "x", "domain": "integer"}],
            "confidence": {"overall": 0.95},
        }
    )
    emr = EMR.model_validate(
        {
            "problem_id": "problem_math8_direct",
            "source_problem_type": "algebra",
            "representation_type": "equation_system",
            "variables": [{"id": "var_x", "symbol": "x", "domain": "integer"}],
            "expressions": [
                {
                    "id": "expr_1",
                    "sympy": "5*x+18",
                    "raw_text": "5x+18",
                    "source_formula_id": "f_expr_lhs",
                }
            ],
            "equations": [
                {
                    "id": "eq_1",
                    "lhs_sympy": "5*x+18",
                    "rhs_sympy": "8*x-12",
                    "source_formula_ids": ["f_equation"],
                }
            ],
            "goals": [
                {
                    "id": "goal_1",
                    "goal_type": "compute",
                    "target": "total books",
                    "target_expression_id": "expr_1",
                    "target_variable_ids": ["var_x"],
                }
            ],
        }
    )
    classification = ProblemClassifier().classify(spr, emr)

    chain = Solver().solve(emr, classification, spr=spr)

    assert chain.final_answer == "total books = 68"
    assert [step.rule_name for step in chain.steps] == [
        "define_variable",
        "given_equation",
        "cas_solve_equation_system",
        "cas_evaluate",
        "final_answer",
    ]
    assert chain.steps[3].diagnostics["operation"] == "evaluate"
    assert chain.steps[3].diagnostics["input"]["expression"] == "5*x+18"
    assert chain.steps[3].diagnostics["input"]["substitutions"] == {"x": "10"}
    assert chain.metadata["final_answer_items"][-1] == {
        "target": "total books",
        "value": "68",
        "source_step_id": "chain_main_step_3",
        "backend_operation": "evaluate",
        "backend_confirmed": True,
    }


def test_emr_builder_does_not_treat_condition_fraction_as_final_target() -> None:
    spr = SPR.model_validate(
        {
            "source_image": {"image_path": "math13.png"},
            "problem_text": "A student read 1/4 of a book. After reading 15 more pages, read:unread is 2:3. How many pages are in the book?",
            "problem_stem": "A student read 1/4 of a book and then 15 more pages.",
            "conditions": [
                {"id": "c1", "text": "already read 1/4", "linked_formula_ids": ["f1"]},
                {"id": "c2", "text": "read to unread is 2:3", "linked_formula_ids": ["f2"]},
                {"id": "c3", "text": "model equation", "linked_formula_ids": ["f3"]},
            ],
            "questions": [
                {
                    "id": "q1",
                    "text": "How many pages are in the book?",
                    "question_type": "compute",
                    "target_ids": ["t1"],
                }
            ],
            "targets": [{"id": "t1", "text": "Total number of pages", "target_type": "compute"}],
            "formulas": [
                {"id": "f1", "raw_text": "1/4", "latex": "\\frac{1}{4}", "role": "condition"},
                {"id": "f2", "raw_text": "2:3", "latex": "2:3", "role": "condition"},
                {"id": "f3", "raw_text": "x*(1/4)+15=x*(2/5)", "latex": "", "role": "condition"},
            ],
            "variables": [
                {
                    "symbol": "x",
                    "description": "Total number of pages in the book",
                    "domain": "integer",
                }
            ],
            "problem_type": "algebra",
            "knowledge_units": ["linear equations"],
            "confidence": {"overall": 0.95},
        }
    )
    emr = EMRBuilder().build("problem_math13", spr)
    classification = ProblemClassifier().classify(spr, emr)

    chain = Solver().solve(emr, classification, spr=spr)

    assert emr.expressions == []
    assert emr.goals[0].target == "Total number of pages"
    assert emr.goals[0].target_expression_id is None
    assert chain.final_answer == "x = 100"
    assert chain.metadata["final_answer_items"] == [
        {
            "target": "x",
            "value": "100",
            "source_step_id": "chain_main_step_2",
            "backend_operation": "solve_equation_system",
            "backend_confirmed": True,
        }
    ]


def test_solver_binds_numeric_choice_option_after_planner_calculation() -> None:
    spr = SPR.model_validate(
        {
            "source_image": {"image_path": "math16.png"},
            "problem_text": (
                "A square has side length 2 cm. It is enlarged by scale 4:1. "
                "What is the enlarged area? A. 8 B. 32 C. 64 D. 128"
            ),
            "problem_stem": "A square has side length 2 cm and is enlarged by scale 4:1.",
            "conditions": [
                {"id": "c_side", "text": "Original side length is 2 cm"},
                {"id": "c_scale", "text": "Scale is 4:1"},
            ],
            "questions": [
                {
                    "id": "q_area",
                    "text": "Choose the enlarged area.",
                    "question_type": "select",
                    "target_ids": ["target_area"],
                }
            ],
            "targets": [{"id": "target_area", "text": "enlarged area", "target_type": "select"}],
            "formulas": [
                {"id": "opt_a", "raw_text": "A. 8", "role": "option"},
                {"id": "opt_b", "raw_text": "B. 32", "role": "option"},
                {"id": "opt_c", "raw_text": "C. 64", "role": "option"},
                {"id": "opt_d", "raw_text": "D. 128", "role": "option"},
            ],
            "variables": [
                {"symbol": "new_side", "description": "new side length"},
                {"symbol": "area", "description": "enlarged square area"},
            ],
            "problem_type": "geometry",
            "knowledge_units": ["scale", "square area"],
            "confidence": {"overall": 0.95},
        }
    )
    emr = EMRBuilder().build("problem_math16_choice", spr)
    classification = ProblemClassifier().classify(spr, emr)
    planner = DeepSeekPlanner(client=FakeChoiceScaleAreaPlanningClient())

    chain = Solver(planner=planner, use_planner=True).solve(emr, classification, spr=spr)

    assert chain.metadata["partial_failure"] is False
    assert chain.final_answer == "答案是 64"
    final_item = chain.metadata["final_answer_items"][0]
    assert final_item["value"] == "64"
    assert final_item["selected_option_id"] == "opt_c"
    assert final_item["selected_option_label"] == "C"


def test_solver_returns_unsupported_chain_for_math3_optimization() -> None:
    spr = math3_spr()
    emr = EMRBuilder().build("problem_math3", spr)
    classification = ProblemClassifier().classify(spr, emr)

    chain = Solver().solve(emr, classification)

    assert chain.chain_id == "chain_main"
    assert chain.final_answer is None
    assert "unsupported_route" in [step.rule_name for step in chain.steps]
    assert all(step.diagnostics.get("explainer_ready") is True for step in chain.steps)
    assert chain.metadata["solver_route"] == "algebra_optimization_solver"
    assert chain.metadata["task_type"] == "optimization"
    assert chain.metadata["unsupported_reason"]
    unsupported_step = next(step for step in chain.steps if step.rule_name == "unsupported_route")
    assert unsupported_step.diagnostics["solver_route"] == "algebra_optimization_solver"
    assert unsupported_step.diagnostics["task_type"] == "optimization"


def test_solver_constructs_math5_expression_from_fake_planner() -> None:
    planner = DeepSeekPlanner(client=FakeMath5PlanningClient())

    chain = Solver(planner=planner, use_planner=True).solve(
        math5_emr(),
        math5_classification_result(),
        spr=math5_spr(),
    )

    assert chain.chain_id == "chain_main"
    assert chain.final_answer == "3*x + 4"
    assert chain.steps[0].rule_name == "read_problem_and_variables"
    assert "define_variable" in [step.rule_name for step in chain.steps]
    assert "modeling_from_tool_call" in [step.rule_name for step in chain.steps]
    planner_step = next(step for step in chain.steps if step.rule_name == "planner_construct_expression")
    cas_step = next(step for step in chain.steps if step.rule_name == "cas_construct_expression")
    assert chain.steps[-1].rule_name == "final_answer"
    assert planner_step.diagnostics["normalized_request"]["expressions"][0]["expression"] == "3*x + 4"
    assert cas_step.diagnostics["backend"] == "sympy"
    assert cas_step.diagnostics["backend_confirmed"] is True
    assert chain.metadata["backend_confirmed"] is True
    assert chain.metadata["variable_bindings"]
    assert chain.metadata["answer_bindings"]
    assert validate_solution_chain_dict(chain.model_dump(mode="json"))


def test_solver_executes_math7_multistep_tool_calls_with_spr_variables() -> None:
    planner = DeepSeekPlanner(client=FakeMath7PlanningClient())

    chain = Solver(planner=planner, use_planner=True).solve(
        math7_unknown_emr(),
        math7_classification_result(),
        spr=math7_spr(),
    )

    assert chain.chain_id == "chain_main"
    assert chain.final_answer == "第二天看了 92/3 页"
    assert chain.metadata["tool_call_count"] == 2
    assert chain.metadata["completed_tool_call_count"] == 2
    assert chain.metadata["execution_context"]["variables"]["x"] == "8/3"
    assert "cas_solve_equation_system" in [step.rule_name for step in chain.steps]
    assert "cas_substitute" in [step.rule_name for step in chain.steps]
    assert all(step.description for step in chain.steps)
    assert any("读取题意" in step.description for step in chain.steps)
    assert validate_solution_chain_dict(chain.model_dump(mode="json"))


def test_solver_normalizes_math9_multistep_tool_call_variables() -> None:
    spr = math9_spr()
    emr = math9_unknown_emr()
    classification = ProblemClassifier().classify(spr, emr)
    planner = DeepSeekPlanner(client=FakeMath9PlanningClient())

    chain = Solver(planner=planner, use_planner=True).solve(
        emr,
        classification,
        spr=spr,
    )

    assert chain.chain_id == "chain_main"
    assert "120" in (chain.final_answer or "")
    assert chain.metadata["solver_route"] == "algebra_word_problem_solver"
    assert chain.metadata["tool_call_count"] == 2
    assert chain.metadata["completed_tool_call_count"] == 2
    assert chain.metadata["execution_context"]["variables"]["x"] == "18"
    evaluate_report = chain.metadata["validation_reports"][1]
    assert evaluate_report["planner_variables"] == ["T"]
    assert evaluate_report["effective_variables"] == ["T", "x"]
    assert evaluate_report["normalization_warnings"]
    evaluate_step = next(
        step
        for step in chain.steps
        if step.rule_name == "planner_evaluate"
        and step.diagnostics.get("tool_call_id") == "calculate_T"
    )
    cas_evaluate_step = next(
        step
        for step in chain.steps
        if step.rule_name == "cas_evaluate"
        and step.diagnostics.get("tool_call_id") == "calculate_T"
    )
    assert evaluate_step.diagnostics["planner_variables"] == ["T"]
    assert evaluate_step.diagnostics["effective_variables"] == ["T", "x"]
    assert cas_evaluate_step.diagnostics["success"] is True
    assert cas_evaluate_step.diagnostics["output"]["value"] == "120"
    assert validate_solution_chain_dict(chain.model_dump(mode="json"))


def test_solver_aggregates_math10_multi_question_final_answer() -> None:
    spr = math10_spr()
    emr = math10_unknown_emr()
    classification = ProblemClassifier().classify(spr, emr)
    planner = DeepSeekPlanner(client=FakeMath10PlanningClient())

    chain = Solver(planner=planner, use_planner=True).solve(
        emr,
        classification,
        spr=spr,
    )

    assert chain.chain_id == "chain_main"
    assert "T = 56" in (chain.final_answer or "")
    assert "N_boxes = 7" in (chain.final_answer or "")
    assert chain.metadata["backend_confirmed"] is True
    assert chain.metadata["partial_failure"] is False
    assert "define_variable" in [step.rule_name for step in chain.steps]
    assert "modeling_from_tool_call" in [step.rule_name for step in chain.steps]
    assert chain.metadata["variable_bindings"]["T"]["description"] == "Total apple weight"
    assert chain.metadata["variable_bindings"]["T"]["variable_role"] == "primary_unknown"
    assert chain.metadata["variable_bindings"]["T"]["used_in_computation"] is True
    assert chain.metadata["variable_bindings"]["N_boxes"]["variable_role"] == "final_target"
    assert chain.metadata["variable_bindings"]["N_boxes"]["explainer_priority"] == 9
    assert chain.metadata["target_bindings"]["t1"]["text"] == "Total apple weight"
    assert chain.metadata["answer_bindings"][0]["display_name"] == "Total apple weight"
    assert chain.metadata["answer_bindings"][1]["display_name"] == "Number of boxes"
    assert chain.metadata["answer_bindings"][1]["question_id"] == "q2"
    assert {group["id"] for group in chain.metadata["solution_outline"]} >= {
        "understand",
        "model",
        "solve",
        "answer",
    }
    assert chain.metadata["final_answer_items"] == [
        {
            "target": "T",
            "value": "56",
            "source_tool_call_id": "solve_T",
            "backend_operation": "solve_equation_system",
            "backend_confirmed": True,
        },
        {
            "target": "N_boxes",
            "value": "7",
            "source_tool_call_id": "calc_boxes",
            "backend_operation": "evaluate",
            "backend_confirmed": True,
        },
    ]
    assert chain.steps[-1].diagnostics["input"]["final_answer_items"] == chain.metadata["final_answer_items"]
    assert validate_solution_chain_dict(chain.model_dump(mode="json"))


def test_solver_handles_math10_verbose_named_expression_tool_calls() -> None:
    spr = math10_spr()
    emr = math10_unknown_emr()
    classification = ProblemClassifier().classify(spr, emr)
    planner = DeepSeekPlanner(client=FakeMath10VerbosePlanningClient())

    chain = Solver(planner=planner, use_planner=True).solve(
        emr,
        classification,
        spr=spr,
    )

    assert chain.chain_id == "chain_main"
    assert chain.metadata["backend_confirmed"] is True
    assert chain.metadata["partial_failure"] is False
    assert chain.metadata["completed_tool_call_count"] == 7
    assert chain.metadata["execution_context"]["s1"]["S_m"] == "T/2 - 4"
    assert "R_T" in chain.metadata["execution_context"]["s5"]
    assert chain.metadata["execution_context"]["variables"]["T"] == "56"
    assert chain.metadata["final_answer_items"] == [
        {
            "target": "T",
            "value": "56",
            "source_tool_call_id": "s6",
            "backend_operation": "solve_for",
            "backend_confirmed": True,
        },
        {
            "target": "N_boxes",
            "value": "7",
            "source_tool_call_id": "s7",
            "backend_operation": "evaluate",
            "backend_confirmed": True,
        },
    ]
    assert "T = 56" in (chain.final_answer or "")
    assert "N_boxes = 7" in (chain.final_answer or "")
    assert validate_solution_chain_dict(chain.model_dump(mode="json"))


def test_solver_binds_semantic_box_answer_to_second_question() -> None:
    spr_data = math10_spr().model_dump(mode="json")
    spr_data["variables"][1]["symbol"] = "NumBoxes"
    spr_data["variables"][1]["description"] = "Number of boxes needed"
    spr_data["targets"][1]["text"] = "Total number of boxes"
    spr = SPR.model_validate(spr_data)
    emr = math10_unknown_emr()
    classification = ProblemClassifier().classify(spr, emr)
    planner = DeepSeekPlanner(client=FakeMath10NumBoxesPlanningClient())

    chain = Solver(planner=planner, use_planner=True).solve(emr, classification, spr=spr)

    assert chain.metadata["answer_bindings"][1]["target"] == "NumBoxes"
    assert chain.metadata["answer_bindings"][1]["question_id"] == "q2"
    assert chain.metadata["variable_bindings"]["NumBoxes"]["variable_role"] == "final_target"
    assert chain.metadata["variable_bindings"]["NumBoxes"]["used_in_computation"] is True


def test_solver_keeps_all_variables_for_aggregate_equation_system_target() -> None:
    spr = SPR.model_validate(
        {
            "source_image": {"image_path": "eqsys_003.png"},
            "problem_text": "Solve the system: x + 8y = 15, 4x + y = 29.",
            "problem_type": "algebra",
            "conditions": [
                {"id": "c1", "text": "x + 8y = 15"},
                {"id": "c2", "text": "4x + y = 29"},
            ],
            "questions": [
                {
                    "id": "q1",
                    "text": "Find the solution set (x, y).",
                    "question_type": "compute",
                    "target_ids": ["t1"],
                }
            ],
            "variables": [
                {"symbol": "x", "description": "Unknown variable x", "domain": "real"},
                {"symbol": "y", "description": "Unknown variable y", "domain": "real"},
            ],
            "targets": [{"id": "t1", "text": "Solution set (x, y)", "target_type": "compute"}],
            "confidence": {"overall": 0.95},
        }
    )
    emr = EMR.model_validate(
        {
            "problem_id": "problem_eqsys",
            "source_problem_type": "algebra",
            "representation_type": "unknown",
        }
    )
    classification = algebra_solver_classification_result(problem_id="problem_eqsys")
    planner = DeepSeekPlanner(client=FakeEquationSystemAggregatePlanningClient())

    chain = Solver(planner=planner, use_planner=True).solve(emr, classification, spr=spr)
    verification = Verifier().verify(chain, spr=spr, emr=emr, classification_result=classification)

    assert chain.final_answer == "x = 7, y = 1"
    assert chain.metadata["final_answer_items"] == [
        {
            "target": "x",
            "value": "7",
            "source_tool_call_id": "solve_system",
            "backend_operation": "solve_equation_system",
            "backend_confirmed": True,
        },
        {
            "target": "y",
            "value": "1",
            "source_tool_call_id": "solve_system",
            "backend_operation": "solve_equation_system",
            "backend_confirmed": True,
        },
    ]
    assert chain.steps[-1].diagnostics["input"]["final_answer_items"] == chain.metadata["final_answer_items"]
    assert verification.valid is True


def test_solver_recovers_tangent_line_after_planner_timeout() -> None:
    spr = SPR.model_validate(
        {
            "source_image": {"image_path": "deriv_006.png"},
            "problem_text": "15. 已知函数 y = -x^2 + 4x 求函数图像在 x = 3 处的切线方程。",
            "problem_stem": "已知函数 y = -x^2 + 4x，求函数图像在 x = 3 处的切线方程。",
            "problem_type": "calculus",
            "knowledge_units": ["derivative", "tangent_line_equation"],
            "conditions": [{"id": "c1", "text": "y = -x^2 + 4x"}],
            "questions": [
                {
                    "id": "q1",
                    "text": "求函数图像在 x = 3 处的切线方程",
                    "question_type": "compute",
                    "target_ids": ["t1"],
                }
            ],
            "formulas": [
                {"id": "f1", "raw_text": "y = -x^2 + 4x", "latex": "y = -x^2 + 4x", "role": "condition"},
                {"id": "x1", "raw_text": "x = 3", "latex": "x = 3", "role": "condition"},
            ],
            "variables": [
                {"symbol": "x", "description": "independent variable", "domain": "real"},
                {"symbol": "y", "description": "dependent variable", "domain": "real"},
            ],
            "targets": [{"id": "t1", "text": "切线方程", "target_type": "compute"}],
            "confidence": {"overall": 0.95},
        }
    )
    emr = EMR.model_validate(
        {
            "problem_id": "problem_deriv_006",
            "source_problem_type": "calculus",
            "representation_type": "unknown",
        }
    )
    classification = unknown_classification_result()
    planner = DeepSeekPlanner(client=FakeTimeoutPlanningClient())

    chain = Solver(planner=planner, use_planner=True).solve(emr, classification, spr=spr)
    verification = Verifier().verify(chain, spr=spr, emr=emr, classification_result=classification)

    assert chain.metadata["backend_confirmed"] is True
    assert chain.metadata["partial_failure"] is False
    assert chain.metadata["final_answer_items"] == [
        {
            "target": "tangent_line",
            "value": "y = 9 - 2*x",
            "source_tool_call_id": "derived_tangent_line",
            "backend_operation": "substitute",
            "backend_confirmed": True,
        }
    ]
    assert "y = 9 - 2*x" in (chain.final_answer or "")
    assert verification.valid is True


def test_solver_recovers_polynomial_extrema_after_planner_timeout() -> None:
    spr = SPR.model_validate(
        {
            "source_image": {"image_path": "deriv_008.png"},
            "problem_text": "24. 求函数 y = x^3 - 6x^2 + 9x + 1 的极值。",
            "problem_stem": "求函数 y = x^3 - 6x^2 + 9x + 1 的极值。",
            "problem_type": "calculus",
            "knowledge_units": ["derivatives", "critical_points", "local_extrema"],
            "conditions": [{"id": "c1", "text": "y = x^3 - 6x^2 + 9x + 1"}],
            "questions": [
                {
                    "id": "q1",
                    "text": "求该函数的极值",
                    "question_type": "compute",
                    "target_ids": ["t1"],
                }
            ],
            "formulas": [
                {
                    "id": "f1",
                    "raw_text": "y = x^3 - 6x^2 + 9x + 1",
                    "latex": "y = x^3 - 6x^2 + 9x + 1",
                    "role": "condition",
                }
            ],
            "variables": [
                {"symbol": "x", "description": "independent variable", "domain": "real"},
                {"symbol": "y", "description": "dependent variable", "domain": "real"},
            ],
            "targets": [{"id": "t1", "text": "极值", "target_type": "compute"}],
            "confidence": {"overall": 0.95},
        }
    )
    emr = EMR.model_validate(
        {
            "problem_id": "problem_deriv_008",
            "source_problem_type": "calculus",
            "representation_type": "unknown",
        }
    )
    classification = unknown_classification_result()
    planner = DeepSeekPlanner(client=FakeTimeoutPlanningClient())

    chain = Solver(planner=planner, use_planner=True).solve(emr, classification, spr=spr)
    verification = Verifier().verify(chain, spr=spr, emr=emr, classification_result=classification)

    assert chain.metadata["backend_confirmed"] is True
    assert chain.metadata["partial_failure"] is False
    assert chain.metadata["final_answer_items"] == [
        {
            "target": "local_extrema",
            "value": "x = 1, y = 5 (local maximum); x = 3, y = 1 (local minimum)",
            "source_tool_call_id": "derived_extrema_value_2",
            "backend_operation": "evaluate",
            "backend_confirmed": True,
        }
    ]
    assert "local_extrema = x = 1, y = 5" in (chain.final_answer or "")
    assert verification.valid is True
    assert verification.final_answer_valid is True


def test_solver_keeps_chicken_rabbit_multi_target_system_answers() -> None:
    spr = SPR.model_validate(
        {
            "source_image": {"image_path": "math19.png"},
            "problem_text": "鸡兔同笼，共有头35个，脚94只，问鸡和兔各有多少只。",
            "problem_type": "algebra",
            "conditions": [
                {"id": "c1", "text": "共有头35个"},
                {"id": "c2", "text": "共有脚94只"},
            ],
            "questions": [
                {
                    "id": "q1",
                    "text": "问鸡和兔各有多少只",
                    "question_type": "compute",
                    "target_ids": ["t1"],
                }
            ],
            "variables": [
                {"symbol": "x", "description": "number of chickens", "domain": "integer"},
                {"symbol": "y", "description": "number of rabbits", "domain": "integer"},
            ],
            "targets": [
                {"id": "t1", "text": "number of chickens", "target_type": "unknown"},
                {"id": "t2", "text": "number of rabbits", "target_type": "unknown"},
            ],
            "confidence": {"overall": 0.98},
        }
    )
    backend_results = [
        {
            "success": True,
            "operation": "solve_equation_system",
            "output": {
                "raw_solution": [{"x": "23", "y": "12"}],
                "formatted_solution": "x = 23, y = 12",
            },
        }
    ]
    validation_reports = [
        {
            "normalized_request": {
                "id": "solve_system",
                "operation": "solve_equation_system",
                "target": "x",
            }
        }
    ]

    items = AlgebraSolver._final_answer_items(backend_results, validation_reports, spr)

    assert items == [
        {
            "target": "x",
            "value": "23",
            "source_tool_call_id": "solve_system",
            "backend_operation": "solve_equation_system",
            "backend_confirmed": True,
        },
        {
            "target": "y",
            "value": "12",
            "source_tool_call_id": "solve_system",
            "backend_operation": "solve_equation_system",
            "backend_confirmed": True,
        },
    ]


def test_solver_traces_equation_system_solution_and_derived_total_answer() -> None:
    spr = SPR.model_validate(
        {
            "source_image": {"image_path": "eqsys_007.png"},
            "problem_text": "A system gives x + y = 452 and y = 92. Find x and the total.",
            "problem_type": "algebra",
            "conditions": [
                {"id": "c1", "text": "x + y = 452"},
                {"id": "c2", "text": "y = 92"},
            ],
            "questions": [
                {
                    "id": "q1",
                    "text": "Find x and the total.",
                    "question_type": "compute",
                    "target_ids": ["t1", "t2"],
                }
            ],
            "variables": [
                {"symbol": "x", "description": "Unknown x", "domain": "real"},
                {"symbol": "y", "description": "Known related quantity", "domain": "real"},
                {"symbol": "total", "description": "Total amount", "domain": "real"},
            ],
            "targets": [
                {"id": "t1", "text": "x", "target_type": "compute"},
                {"id": "t2", "text": "total", "target_type": "compute"},
            ],
            "confidence": {"overall": 0.95},
        }
    )
    emr = EMR.model_validate(
        {
            "problem_id": "problem_eqsys_derived_total",
            "source_problem_type": "algebra",
            "representation_type": "unknown",
        }
    )
    classification = algebra_solver_classification_result(problem_id="problem_eqsys_derived_total")
    planner = DeepSeekPlanner(client=FakeEquationSystemDerivedTotalPlanningClient())

    chain = Solver(planner=planner, use_planner=True).solve(emr, classification, spr=spr)
    verification = Verifier().verify(chain, spr=spr, emr=emr, classification_result=classification)

    assert chain.metadata["final_answer_items"] == [
        {
            "target": "x",
            "value": "360",
            "source_tool_call_id": "solve_system",
            "backend_operation": "solve_equation_system",
            "backend_confirmed": True,
        },
        {
            "target": "total",
            "value": "452",
            "source_tool_call_id": "compute_total",
            "backend_operation": "evaluate",
            "backend_confirmed": True,
        },
    ]
    assert chain.final_answer == "x = 360；total = 452"
    assert verification.valid is True
    assert verification.final_answer_valid is True


def test_solver_filters_intermediate_solve_system_values_from_final_answer_items() -> None:
    spr_data = math10_spr().model_dump(mode="json")
    spr_data["problem_text"] = (
        "水果店运来一批苹果。上午卖出总数的一半少4千克，"
        "下午卖出剩下的一半少2千克，这时还剩18千克苹果。"
    )
    spr_data["questions"] = [
        {
            "id": "q1",
            "text": "水果店原来运来多少千克苹果？",
            "question_type": "compute",
            "target_ids": ["t1"],
        },
        {
            "id": "q2",
            "text": "如果每箱装8千克苹果，这些苹果一共可以装几箱？",
            "question_type": "compute",
            "target_ids": ["t2"],
        },
    ]
    spr_data["variables"] = [
        {"symbol": "x", "description": "原来运来的苹果总重量", "domain": "real_positive"},
        {"symbol": "Sold_AM", "description": "上午卖出的苹果重量", "domain": "real_positive"},
        {"symbol": "Sold_PM", "description": "下午卖出的苹果重量", "domain": "real_positive"},
    ]
    spr_data["targets"] = [
        {"id": "t1", "text": "原来运来的苹果总重量", "target_type": "compute"},
        {"id": "t2", "text": "可以装的箱数", "target_type": "compute"},
    ]
    spr = SPR.model_validate(spr_data)
    emr = math10_unknown_emr()
    classification = ProblemClassifier().classify(spr, emr)
    planner = DeepSeekPlanner(client=FakeMath10SoldBreakdownPlanningClient())

    chain = Solver(planner=planner, use_planner=True).solve(emr, classification, spr=spr)

    assert chain.metadata["final_answer_items"] == [
        {
            "target": "x",
            "value": "56",
            "source_tool_call_id": "solve_x",
            "backend_operation": "solve_equation_system",
            "backend_confirmed": True,
        },
        {
            "target": "boxes",
            "value": "7",
            "source_tool_call_id": "compute_boxes",
            "backend_operation": "evaluate",
            "backend_confirmed": True,
        },
    ]
    assert "Sold_AM" not in (chain.final_answer or "")
    assert "Sold_PM" not in (chain.final_answer or "")


def test_solver_auto_substitutes_solved_variables_before_evaluate_final_answer() -> None:
    spr = math10_spr()
    emr = math10_unknown_emr()
    classification = ProblemClassifier().classify(spr, emr)
    planner = DeepSeekPlanner(client=FakeMath10MissingSubstitutionPlanningClient())

    chain = Solver(planner=planner, use_planner=True).solve(emr, classification, spr=spr)

    assert chain.metadata["partial_failure"] is False
    assert chain.metadata["execution_context"]["variables"]["x"] == "56"
    evaluate_step = next(
        step
        for step in chain.steps
        if step.rule_name == "cas_evaluate"
        and step.diagnostics.get("tool_call_id") == "compute_boxes"
    )
    assert evaluate_step.diagnostics["success"] is True
    assert evaluate_step.diagnostics["input"]["substitutions"] == {"x": "56"}
    assert evaluate_step.diagnostics["output"]["value"] == "7"
    assert chain.metadata["final_answer_items"] == [
        {
            "target": "T",
            "value": "56",
            "source_tool_call_id": "solve_x",
            "backend_operation": "solve_for",
            "backend_confirmed": True,
        },
        {
            "target": "N_boxes",
            "value": "7",
            "source_tool_call_id": "compute_boxes",
            "backend_operation": "evaluate",
            "backend_confirmed": True,
        },
    ]


def test_solver_does_not_match_single_letter_x_to_boxes_question() -> None:
    spr = math10_spr()
    emr = math10_unknown_emr()
    classification = ProblemClassifier().classify(spr, emr)
    planner = DeepSeekPlanner(client=FakeMath10PlanningClient())

    chain = Solver(planner=planner, use_planner=True).solve(emr, classification, spr=spr)

    assert chain.metadata["answer_bindings"][0]["target"] == "T"
    assert chain.metadata["answer_bindings"][0]["question_id"] == "q1"
    assert chain.metadata["target_bindings"]["t2"]["unit"] == "box"


def test_vision_parser_report_records_qwen_mode_for_qwen_generator(tmp_path: Path) -> None:
    output = VisionParser(spr_generator=FakeQwenSPRGenerator()).run(
        "fake_qwen_image.png",
        save_artifacts=True,
        output_dir=tmp_path,
    )

    assert output.parsing_report.metadata["parser_mode"] == "qwen"
    spr_generate = next(
        stage for stage in output.parsing_report.stages if stage.stage_name == "spr_generate"
    )
    emr_build = next(
        stage for stage in output.parsing_report.stages if stage.stage_name == "emr_build"
    )
    assert spr_generate.metadata["generator_mode"] == "qwen"
    assert "Qwen" in spr_generate.message
    assert emr_build.metadata["builder_mode"] == "fallback_unknown"
    assert "deterministic mock" not in emr_build.message.lower()


def test_calculation_request_validator_normalizes_missing_request_variables() -> None:
    spr = math9_spr()
    emr = math9_unknown_emr()
    context = {"solve_x": {"x": "18"}, "variables": {"x": "18"}}
    report = CalculationRequestValidator().validate(
        {
            "operation": "evaluate",
            "variables": ["T"],
            "expressions": [{"name": "T", "expression": "6*x+12"}],
            "substitutions": {"x": "$solve_x.x"},
            "target": "T",
        },
        emr,
        spr=spr,
        execution_context=context,
    )

    assert report["valid"] is True
    assert report["planner_variables"] == ["T"]
    assert report["effective_variables"] == ["T", "x"]
    assert report["normalization_warnings"]


def test_calculation_request_validator_resolves_embedded_prior_result_reference() -> None:
    spr = math7_spr()
    emr = math7_unknown_emr()
    report = CalculationRequestValidator().validate(
        {
            "id": "calc_total",
            "operation": "evaluate",
            "variables": [],
            "expressions": [
                {"name": "L_total", "expression": "18 + $calc_remaining.L_remaining"}
            ],
            "target": "L_total",
        },
        emr,
        spr=spr,
        execution_context={
            "variables": {"L_remaining": "54"},
            "calc_remaining": {"L_remaining": "54"},
        },
    )

    assert report["valid"] is True
    assert not report["errors"]
    assert "L_total" in report["effective_variables"]


def test_calculation_request_validator_normalizes_qwen_expression_in_target() -> None:
    spr = math7_spr()
    emr = math7_unknown_emr()
    validator = CalculationRequestValidator()
    first = validator.validate(
        {
            "id": "calc_unripe",
            "operation": "evaluate",
            "variables": [],
            "expressions": [],
            "target": "25 * 0.20",
            "reason": "Compute the unripe count.",
        },
        emr,
        spr=spr,
        execution_context={},
    )

    assert first["valid"] is True
    assert first["normalized_request"]["target"] == "calc_unripe"
    assert first["normalized_request"]["expressions"] == [
        {"name": "calc_unripe", "expression": "25 * 0.20"}
    ]

    second = validator.validate(
        {
            "id": "calc_good",
            "operation": "evaluate",
            "variables": [],
            "expressions": [],
            "target": "25 - 1 - $calc_unripe - 2",
            "reason": "Compute the good count.",
        },
        emr,
        spr=spr,
        execution_context={
            "variables": {"calc_unripe": "5.00000000000000"},
            "calc_unripe": {"value": "5.00000000000000"},
        },
    )

    assert second["valid"] is True
    assert second["normalized_request"]["target"] == "calc_good"
    assert second["normalized_request"]["expressions"] == [
        {"name": "calc_good", "expression": "25 - 1 - $calc_unripe - 2"}
    ]


def test_calculation_request_validator_resolves_latex_symbol_alias_reference() -> None:
    assert (
        CalculationRequestValidator.resolve_reference(
            "$calc_purple.N_purple",
            {"calc_purple": {"N_{purple}": "18"}},
        )
        == "18"
    )


def test_algebra_solver_stores_tool_result_target_aliases() -> None:
    context = {"variables": {}}
    AlgebraSolver._store_tool_result(
        "calc_purple",
        {
            "operation": "evaluate",
            "output": {"value": "18", "formatted_solution": "18"},
        },
        context,
        {
            "target": "N_purple",
            "expressions": [{"name": "purple_expr", "expression": "10 + 0.8*10"}],
        },
    )

    assert context["calc_purple"]["N_purple"] == "18"
    assert context["calc_purple"]["N_{purple}"] == "18"
    assert CalculationRequestValidator.resolve_reference("$calc_purple.N_purple", context) == "18"


def test_calculation_request_validator_rejects_undeclared_symbol() -> None:
    spr = math9_spr()
    emr = math9_unknown_emr()
    report = CalculationRequestValidator().validate(
        {
            "operation": "evaluate",
            "variables": ["T"],
            "expressions": [{"name": "T", "expression": "6*y+12"}],
            "substitutions": {},
            "target": "T",
        },
        emr,
        spr=spr,
    )

    assert report["valid"] is False
    assert any(error.get("undeclared_symbols") == ["y"] for error in report["errors"])


def test_calculation_request_validator_accepts_word_problem_local_symbols_when_spr_has_no_variables() -> None:
    spr_data = math10_spr().model_dump(mode="json")
    spr_data["variables"] = []
    spr = SPR.model_validate(spr_data)
    emr = math10_unknown_emr()

    report = CalculationRequestValidator().validate(
        {
            "operation": "solve_equation_system",
            "variables": ["x"],
            "equations": [{"lhs": "x/4 + 4", "rhs": "18"}],
            "target": "x",
        },
        emr,
        spr=spr,
    )

    assert report["valid"] is True
    assert report["declared_variables"] == ["x"]
    assert any("planner-local" in warning["message"] for warning in report["normalization_warnings"])


def test_calculation_request_validator_accepts_derived_evaluate_target_name() -> None:
    spr_data = math10_spr().model_dump(mode="json")
    spr_data["variables"] = [
        {
            "symbol": "x",
            "description": "水果店原来运来的苹果总重量（千克）",
            "domain": "positive_real",
        }
    ]
    spr = SPR.model_validate(spr_data)
    emr = math10_unknown_emr()

    report = CalculationRequestValidator().validate(
        {
            "operation": "evaluate",
            "variables": ["x", "boxes"],
            "expressions": [{"name": "boxes", "expression": "x/8"}],
            "substitutions": {"x": "$solve_x.x"},
            "target": "boxes",
        },
        emr,
        spr=spr,
        execution_context={"variables": {"x": "56"}, "solve_x": {"x": "56"}},
    )

    assert report["valid"] is True
    assert report["declared_variables"] == ["boxes", "x"]
    assert any("planner-local" in warning["message"] for warning in report["normalization_warnings"])


def test_calculation_request_validator_accepts_remaining_alias_symbol() -> None:
    spr_data = math10_spr().model_dump(mode="json")
    spr_data["variables"] = [
        {
            "symbol": "x",
            "description": "水果店原来运来的苹果总重量",
            "domain": "positive_real",
        },
        {
            "symbol": "remaining",
            "description": "上午卖出后剩下的苹果重量",
            "domain": "positive_real",
        },
    ]
    spr = SPR.model_validate(spr_data)
    emr = math10_unknown_emr()

    report = CalculationRequestValidator().validate(
        {
            "operation": "solve_for",
            "variables": ["R"],
            "equations": [{"lhs": "R/2 + 2", "rhs": "18"}],
            "target": "R",
        },
        emr,
        spr=spr,
    )

    assert report["valid"] is True
    assert report["semantic_symbol_aliases"] == {"R": "remaining"}
    assert any("semantically aligned" in warning["message"] for warning in report["normalization_warnings"])


def test_calculation_request_validator_accepts_total_remaining_alias_symbol() -> None:
    spr_data = math10_spr().model_dump(mode="json")
    spr_data["variables"] = [
        {
            "symbol": "x",
            "description": "水果店原来运来的苹果总重量",
            "domain": "positive_real",
        },
        {
            "symbol": "Remaining",
            "description": "最后剩下的苹果重量",
            "domain": "positive_real",
        },
    ]
    spr = SPR.model_validate(spr_data)
    emr = math10_unknown_emr()

    report = CalculationRequestValidator().validate(
        {
            "operation": "solve_for",
            "variables": ["Total_remaining"],
            "equations": [{"lhs": "Total_remaining/2 + 2", "rhs": "18"}],
            "target": "Total_remaining",
        },
        emr,
        spr=spr,
    )

    assert report["valid"] is True
    assert report["semantic_symbol_aliases"] == {"Total_remaining": "Remaining"}


def test_solver_accepts_math10_remaining_alias_tool_trace() -> None:
    spr_data = math10_spr().model_dump(mode="json")
    spr_data["variables"] = [
        {
            "symbol": "x",
            "description": "水果店原来运来的苹果总重量",
            "domain": "positive_real",
        },
        {
            "symbol": "remaining",
            "description": "上午卖出后剩下的苹果重量",
            "domain": "positive_real",
        },
        {
            "symbol": "n",
            "description": "苹果的箱数",
            "domain": "integer",
        },
    ]
    spr = SPR.model_validate(spr_data)
    emr = math10_unknown_emr()
    classification = ProblemClassifier().classify(spr, emr)

    chain = Solver(
        planner=DeepSeekPlanner(client=FakeMath10RemainingAliasPlanningClient()),
        use_planner=True,
    ).solve(emr, classification, spr=spr)

    assert chain.steps[-1].rule_name == "final_answer"
    assert chain.metadata["backend_confirmed"] is True
    assert chain.metadata["completed_tool_call_count"] == 3
    assert chain.metadata["final_answer_items"] == [
        {
            "target": "x",
            "value": "56",
            "source_tool_call_id": "solve_x",
            "backend_operation": "solve_for",
            "backend_confirmed": True,
        },
        {
            "target": "n",
            "value": "7",
            "source_tool_call_id": "compute_boxes",
            "backend_operation": "evaluate",
            "backend_confirmed": True,
        },
    ]
    assert chain.metadata["validation_reports"][0]["semantic_symbol_aliases"] == {"R": "remaining"}
    assert validate_solution_chain_dict(chain.model_dump(mode="json"))


def test_solver_accepts_math10_total_remaining_alias_tool_trace() -> None:
    spr_data = math10_spr().model_dump(mode="json")
    spr_data["variables"] = [
        {
            "symbol": "x",
            "description": "水果店原来运来的苹果总重量",
            "domain": "positive_real",
        },
        {
            "symbol": "Remaining",
            "description": "最后剩下的苹果重量",
            "domain": "positive_real",
        },
        {
            "symbol": "Boxes",
            "description": "一共可以装的箱数",
            "domain": "integer",
        },
    ]
    spr = SPR.model_validate(spr_data)
    emr = math10_unknown_emr()
    classification = ProblemClassifier().classify(spr, emr)

    chain = Solver(
        planner=DeepSeekPlanner(client=FakeMath10TotalRemainingAliasPlanningClient()),
        use_planner=True,
    ).solve(emr, classification, spr=spr)

    assert chain.steps[-1].rule_name == "final_answer"
    assert chain.metadata["backend_confirmed"] is True
    assert chain.metadata["partial_failure"] is False
    assert chain.metadata["completed_tool_call_count"] == 3
    assert chain.metadata["validation_reports"][0]["semantic_symbol_aliases"] == {
        "Total_remaining": "Remaining"
    }
    assert chain.metadata["final_answer_items"] == [
        {
            "target": "x",
            "value": "56",
            "source_tool_call_id": "s2",
            "backend_operation": "solve_for",
            "backend_confirmed": True,
        },
        {
            "target": "Boxes",
            "value": "7",
            "source_tool_call_id": "s3",
            "backend_operation": "evaluate",
            "backend_confirmed": True,
        },
    ]
    assert validate_solution_chain_dict(chain.model_dump(mode="json"))


def test_calculation_request_validator_rejects_unresolved_substitution_reference() -> None:
    spr = math9_spr()
    emr = math9_unknown_emr()
    report = CalculationRequestValidator().validate(
        {
            "operation": "evaluate",
            "variables": ["T", "x"],
            "expressions": [{"name": "T", "expression": "6*x+12"}],
            "substitutions": {"x": "$missing_step.x"},
            "target": "T",
        },
        emr,
        spr=spr,
        execution_context={"variables": {}},
    )

    assert report["valid"] is False
    assert any(error["field_path"] == "substitutions.x" for error in report["errors"])


def test_calculation_request_validator_accepts_resolvable_local_substitution_symbol() -> None:
    spr = SPR.model_validate(
        {
            "source_image": {"image_path": "math11.png"},
            "problem_text": "客车和货车相对开出，5小时相遇，速度比是5:4。",
            "problem_type": "word_problem",
            "variables": [
                {"symbol": "v_{bus}", "description": "客车平均速度", "domain": "positive_real"},
                {"symbol": "v_{truck}", "description": "货车平均速度", "domain": "positive_real"},
            ],
            "confidence": {"overall": 0.95},
        }
    )
    emr = EMR.model_validate(
        {
            "problem_id": "problem_math11",
            "source_problem_type": "word_problem",
            "representation_type": "unknown",
            "variables": [
                {"id": "var_bus", "symbol": "v_{bus}", "domain": "positive_real"},
                {"id": "var_truck", "symbol": "v_{truck}", "domain": "positive_real"},
            ],
        }
    )

    report = CalculationRequestValidator().validate(
        {
            "id": "solve",
            "operation": "solve_equation_system",
            "variables": ["v_bus", "v_truck", "sum_val"],
            "equations": [
                {"lhs": "v_bus + v_truck", "rhs": "sum_val"},
                {"lhs": "4*v_bus", "rhs": "5*v_truck"},
            ],
            "substitutions": {"sum_val": "$calc_sum.sum_speed"},
            "target": "v_bus",
        },
        emr,
        spr=spr,
        execution_context={"calc_sum": {"sum_speed": "180"}, "variables": {}},
    )

    assert report["valid"] is True
    assert "sum_val" in report["declared_variables"]
    assert report["semantic_symbol_aliases"] == {
        "v_bus": "v_{bus}",
        "v_truck": "v_{truck}",
    }
    assert any(
        "planner-local" in warning["message"]
        for warning in report["normalization_warnings"]
    )


def test_calculation_request_validator_accepts_concrete_local_substitution_symbols() -> None:
    spr = SPR.model_validate(
        {
            "source_image": {"image_path": "func_011.png"},
            "problem_text": "Given y = x^2 - 2x - 3, find the vertex.",
            "problem_type": "algebra",
            "variables": [
                {"symbol": "x", "description": "independent variable", "domain": "real"},
                {"symbol": "y", "description": "dependent variable", "domain": "real"},
            ],
            "confidence": {"overall": 0.95},
        }
    )
    emr = EMR.model_validate(
        {
            "problem_id": "problem_func_011",
            "source_problem_type": "algebra",
            "representation_type": "unknown",
            "variables": [
                {"id": "var_x", "symbol": "x", "domain": "real"},
                {"id": "var_y", "symbol": "y", "domain": "real"},
            ],
        }
    )

    report = CalculationRequestValidator().validate(
        {
            "id": "compute_h",
            "operation": "evaluate",
            "variables": ["a", "b"],
            "expressions": [{"name": "h", "expression": "-b/(2*a)"}],
            "substitutions": {"a": "1", "b": "-2"},
            "target": "h",
        },
        emr,
        spr=spr,
        execution_context={"variables": {}},
    )

    assert report["valid"] is True
    assert {"a", "b"} <= set(report["declared_variables"])
    assert any(
        "planner-local" in warning["message"]
        for warning in report["normalization_warnings"]
    )


def test_solver_handles_math11_two_step_speed_plan_with_local_sum_symbol() -> None:
    spr = SPR.model_validate(
        {
            "source_image": {"image_path": "math11.png"},
            "problem_text": "甲、乙两地相距900千米，一辆客车和一辆货车同时从两地相对开出，5小时相遇。已知客车和货车的速度比是5:4，客车平均每小时行多少千米？",
            "problem_type": "word_problem",
            "variables": [
                {"symbol": "S", "description": "Distance between location A and B", "domain": "real_positive"},
                {"symbol": "t", "description": "Time until meeting", "domain": "real_positive"},
                {"symbol": "v_{bus}", "description": "Speed of the passenger bus", "domain": "real_positive"},
                {"symbol": "v_{truck}", "description": "Speed of the freight truck", "domain": "real_positive"},
            ],
            "questions": [
                {
                    "id": "q1",
                    "text": "客车平均每小时行多少千米？",
                    "target_ids": ["t1"],
                }
            ],
            "targets": [
                {
                    "id": "t1",
                    "text": "Speed of the passenger bus in km/h",
                    "target_type": "unknown",
                }
            ],
            "confidence": {"overall": 0.95},
        }
    )
    emr = EMR.model_validate(
        {
            "problem_id": "problem_math11",
            "source_problem_type": "word_problem",
            "representation_type": "unknown",
        }
    )
    classification = ProblemClassifier().classify(spr, emr)

    chain = Solver(
        planner=DeepSeekPlanner(client=FakeMath11TwoStepSpeedPlanningClient()),
        use_planner=True,
    ).solve(emr, classification, spr=spr)

    assert chain.final_answer == "v_{bus} = 100"
    assert chain.metadata["partial_failure"] is False
    assert chain.metadata["final_answer_items"] == [
        {
            "target": "v_{bus}",
            "value": "100",
            "source_tool_call_id": "solve",
            "backend_operation": "solve_equation_system",
            "backend_confirmed": True,
        }
    ]
    assert validate_solution_chain_dict(chain.model_dump(mode="json"))


def test_solver_returns_unsupported_for_invalid_planner_expression() -> None:
    planner = DeepSeekPlanner(client=FakeInvalidExpressionPlanningClient())

    chain = Solver(planner=planner, use_planner=True).solve(
        math5_emr(),
        math5_classification_result(),
        spr=math5_spr(),
    )

    assert chain.final_answer is None
    assert chain.steps[-1].rule_name == "unsupported_route"
    assert chain.steps[-1].diagnostics["backend_confirmed"] is False
    assert chain.steps[-1].diagnostics["validation_reports"][0]["errors"]
    assert validate_solution_chain_dict(chain.model_dump(mode="json"))


def test_solver_returns_unsupported_for_undeclared_planner_variable() -> None:
    planner = DeepSeekPlanner(client=FakeUndeclaredVariablePlanningClient())

    chain = Solver(planner=planner, use_planner=True).solve(
        math5_emr(),
        math5_classification_result(),
        spr=math5_spr(),
    )

    assert chain.final_answer is None
    assert chain.steps[-1].rule_name == "unsupported_route"
    errors = chain.steps[-1].diagnostics["validation_reports"][0]["errors"]
    assert any(error.get("undeclared_symbols") == ["y"] for error in errors)
    assert validate_solution_chain_dict(chain.model_dump(mode="json"))


def test_calculation_request_validator_accepts_plain_alias_for_latex_subscript_symbols() -> None:
    spr = SPR.model_validate(
        {
            "source_image": {"image_path": "math11.png"},
            "problem_text": "甲、乙两地相距900千米，客车和货车同时相对开出，5小时相遇。速度比是5:4。",
            "problem_type": "word_problem",
            "variables": [
                {"symbol": "v_{bus}", "description": "客车的平均速度", "domain": "positive_real"},
                {"symbol": "v_{truck}", "description": "货车的平均速度", "domain": "positive_real"},
            ],
            "confidence": {"overall": 0.95},
        }
    )
    emr = EMR.model_validate(
        {
            "problem_id": "problem_math11",
            "source_problem_type": "word_problem",
            "representation_type": "unknown",
            "variables": [
                {"id": "var_bus", "symbol": "v_{bus}", "domain": "positive_real"},
                {"id": "var_truck", "symbol": "v_{truck}", "domain": "positive_real"},
            ],
        }
    )

    report = CalculationRequestValidator().validate(
        {
            "operation": "solve_equation_system",
            "variables": ["v_bus", "v_truck"],
            "equations": [
                {"lhs": "(v_bus + v_truck) * 5", "rhs": "900"},
                {"lhs": "v_bus / v_truck", "rhs": "5/4"},
            ],
            "target": "v_bus",
        },
        emr,
        spr=spr,
    )

    assert report["valid"] is True
    assert report["semantic_symbol_aliases"] == {
        "v_bus": "v_{bus}",
        "v_truck": "v_{truck}",
    }
    assert not report["errors"]


def test_calculation_request_validator_accepts_english_color_semantic_alias() -> None:
    spr = SPR.model_validate(
        {
            "source_image": {"image_path": "flowers.png"},
            "problem_text": "Ten flowers are yellow.",
            "problem_type": "word_problem",
            "variables": [
                {
                    "symbol": "N_{yellow}",
                    "description": "Number of yellow flowers",
                    "domain": "integer",
                }
            ],
            "confidence": {"overall": 0.95},
        }
    )
    emr = EMR.model_validate(
        {
            "problem_id": "problem_flowers",
            "source_problem_type": "algebra",
            "representation_type": "unknown",
            "variables": [
                {
                    "id": "var_yellow",
                    "symbol": "N_{yellow}",
                    "description": "Number of yellow flowers",
                    "domain": "integer",
                }
            ],
        }
    )

    report = CalculationRequestValidator().validate(
        {
            "id": "solve_yellow",
            "operation": "solve_for",
            "variables": ["yellow"],
            "equations": [{"lhs": "yellow", "rhs": "10"}],
            "target": "yellow",
        },
        emr,
        spr=spr,
    )

    assert report["valid"] is True
    assert report["semantic_symbol_aliases"]["yellow"] == "N_{yellow}"


def test_calculation_request_validator_accepts_passenger_truck_short_aliases() -> None:
    spr = SPR.model_validate(
        {
            "source_image": {"image_path": "math11.png"},
            "problem_text": "客车和货车相对开出，速度比是5:4。",
            "problem_type": "word_problem",
            "variables": [
                {"symbol": "v_{passenger}", "description": "Speed of the passenger bus", "domain": "real"},
                {"symbol": "v_{truck}", "description": "Speed of the truck", "domain": "real"},
            ],
            "confidence": {"overall": 0.95},
        }
    )
    emr = EMR.model_validate(
        {
            "problem_id": "problem_math11",
            "source_problem_type": "word_problem",
            "representation_type": "unknown",
        }
    )

    report = CalculationRequestValidator().validate(
        {
            "operation": "solve_equation_system",
            "variables": ["v_p", "v_t"],
            "equations": [
                {"lhs": "v_p/v_t", "rhs": "5/4"},
                {"lhs": "(v_p + v_t)*5", "rhs": "900"},
            ],
            "target": "v_p",
        },
        emr,
        spr=spr,
    )

    assert report["valid"] is True
    assert report["semantic_symbol_aliases"] == {
        "v_p": "v_{passenger}",
        "v_t": "v_{truck}",
    }
    assert not report["errors"]


def test_calculation_request_validator_accepts_ratio_coefficient_for_word_problem() -> None:
    spr = SPR.model_validate(
        {
            "source_image": {"image_path": "math11.png"},
            "problem_text": "甲、乙两地相距900千米，客车和货车同时相对开出，5小时相遇。速度比是5:4。",
            "problem_type": "word_problem",
            "variables": [
                {"symbol": "v_{bus}", "description": "客车的平均速度", "domain": "positive_real"},
                {"symbol": "v_{truck}", "description": "货车的平均速度", "domain": "positive_real"},
            ],
            "confidence": {"overall": 0.95},
        }
    )
    emr = EMR.model_validate(
        {
            "problem_id": "problem_math11",
            "source_problem_type": "word_problem",
            "representation_type": "unknown",
            "variables": [
                {"id": "var_bus", "symbol": "v_{bus}", "domain": "positive_real"},
                {"id": "var_truck", "symbol": "v_{truck}", "domain": "positive_real"},
            ],
        }
    )

    report = CalculationRequestValidator().validate(
        {
            "operation": "solve_equation_system",
            "variables": ["k"],
            "equations": [{"lhs": "45*k", "rhs": "900"}],
            "target": "k",
        },
        emr,
        spr=spr,
    )

    assert report["valid"] is True
    assert "k" in report["declared_variables"]
    assert any("planner-local temporary symbols" in warning["message"] for warning in report["normalization_warnings"])


def test_solver_does_not_canonicalize_speed_sum_to_passenger_speed() -> None:
    spr = SPR.model_validate(
        {
            "source_image": {"image_path": "math11.png"},
            "problem_text": "客车和货车相对开出，求客车平均速度。",
            "problem_type": "word_problem",
            "variables": [
                {"symbol": "v_p", "description": "Average speed of the passenger car", "domain": "positive_real"},
                {"symbol": "v_f", "description": "Average speed of the freight truck", "domain": "positive_real"},
            ],
            "confidence": {"overall": 0.95},
        }
    )

    ribbon_spr = SPR.model_validate(
        {
            "source_image": {"image_path": "math7.png"},
            "problem_text": "一根彩带剪去 18 米后，求原来的长度。",
            "problem_type": "word_problem",
            "variables": [
                {"symbol": "L_{remaining}", "description": "Length of the remaining part", "domain": "positive_real"},
                {"symbol": "L_{total}", "description": "Original total length of the ribbon", "domain": "positive_real"},
            ],
            "confidence": {"overall": 0.95},
        }
    )

    assert AlgebraSolver._canonical_target_name("sum_speed", spr) == "sum_speed"
    assert AlgebraSolver._canonical_target_name("relative_speed", spr) == "relative_speed"
    assert AlgebraSolver._canonical_target_name("total_length", ribbon_spr) == "L_{total}"


def test_calculation_request_validator_accepts_xy_local_system_for_speed_word_problem() -> None:
    spr = SPR.model_validate(
        {
            "source_image": {"image_path": "math11.png"},
            "problem_text": "A and B are 900 km apart. A bus and a truck meet after 5 hours. Their speed ratio is 5:4.",
            "problem_type": "word_problem",
            "variables": [
                {"symbol": "S", "description": "Distance between Place A and Place B", "domain": "real"},
                {"symbol": "t", "description": "Time taken to meet", "domain": "real"},
                {"symbol": "v_{bus}", "description": "Speed of the passenger bus", "domain": "positive_real"},
                {"symbol": "v_{truck}", "description": "Speed of the freight truck", "domain": "positive_real"},
            ],
            "confidence": {"overall": 0.95},
        }
    )
    emr = EMR.model_validate(
        {
            "problem_id": "problem_math11",
            "source_problem_type": "word_problem",
            "representation_type": "unknown",
        }
    )

    report = CalculationRequestValidator().validate(
        {
            "id": "solve_system",
            "operation": "solve_equation_system",
            "variables": ["x", "y"],
            "equations": [
                {"lhs": "(x + y)*5", "rhs": "900"},
                {"lhs": "4*x", "rhs": "5*y"},
            ],
            "target": "x",
        },
        emr,
        spr=spr,
    )

    assert report["valid"] is True
    assert "x" in report["declared_variables"]
    assert "y" in report["declared_variables"]
    assert report["semantic_symbol_aliases"] == {}
    assert not report["errors"]


def test_solver_returns_unsupported_chain_for_geometry_stub() -> None:
    emr = EMR.model_validate(
        {
            "problem_id": "problem_geometry",
            "source_problem_type": "geometry",
            "representation_type": "unknown",
        }
    )

    chain = Solver().solve(emr, geometry_classification_result())

    assert chain.final_answer is None
    assert chain.steps[0].rule_name == "unsupported_route"
    assert chain.metadata["solver_route"] == "geometry_stub"
    assert "geometry_stub" in chain.steps[0].diagnostics["solver_route"]


def test_solver_returns_unsupported_chain_for_unknown_route() -> None:
    emr = EMR.model_validate(
        {
            "problem_id": "problem_unknown",
            "source_problem_type": "unknown",
            "representation_type": "unknown",
        }
    )

    chain = Solver().solve(emr, unknown_classification_result())

    assert chain.final_answer is None
    assert chain.steps[0].rule_name == "unsupported_route"
    assert chain.metadata["solver_route"] == "unknown"
    assert chain.metadata["unsupported_reason"]


def test_deepseek_fake_planner_draft_does_not_enter_final_answer() -> None:
    parser_output = VisionParser().run("any/path.png")
    classification = ProblemClassifier().classify(parser_output.spr, parser_output.emr)
    planner = DeepSeekPlanner(client=FakePlanningClient())

    chain = Solver(planner=planner, use_planner=True).solve(parser_output.emr, classification)

    assert chain.final_answer == "x = 2"
    assert chain.metadata["backend_confirmed"] is True
    assert "planner_model" not in chain.metadata
    assert all(not (step.rule_name or "").startswith("planner_") for step in chain.steps)


def test_solver_accepts_list_target_for_linear_function_parameters() -> None:
    spr = SPR.model_validate(
        {
            "source_image": {"image_path": "func_002.png"},
            "problem_text": "A line y = kx + b passes through (-1, 3) and crosses the x-axis at (2, 0). Find the explicit equation.",
            "problem_type": "algebra",
            "conditions": [
                {"id": "c1", "text": "y = kx + b"},
                {"id": "c2", "text": "The line passes through (-1, 3)."},
                {"id": "c3", "text": "The line crosses the x-axis at (2, 0)."},
            ],
            "questions": [
                {
                    "id": "q1",
                    "text": "Find the explicit equation of the line.",
                    "question_type": "compute",
                    "target_ids": ["t1"],
                }
            ],
            "variables": [
                {"symbol": "x", "description": "independent variable", "domain": "real"},
                {"symbol": "y", "description": "dependent variable", "domain": "real"},
                {"symbol": "k", "description": "slope", "domain": "real"},
                {"symbol": "b", "description": "intercept", "domain": "real"},
            ],
            "targets": [
                {"id": "t1", "text": "Determine the values of coefficients k and b", "target_type": "compute"},
                {"id": "t2", "text": "Write the explicit equation of the line", "target_type": "compute"},
            ],
            "confidence": {"overall": 0.95},
        }
    )
    emr = EMR.model_validate(
        {
            "problem_id": "problem_func_002",
            "source_problem_type": "algebra",
            "representation_type": "unknown",
        }
    )
    classification = algebra_solver_classification_result(problem_id="problem_func_002")
    planner = DeepSeekPlanner(client=FakeLinearFunctionListTargetPlanningClient())

    chain = Solver(planner=planner, use_planner=True).solve(emr, classification, spr=spr)
    verification = Verifier().verify(chain, spr=spr, emr=emr, classification_result=classification)

    assert chain.metadata["completed_tool_call_count"] == 2
    assert any(
        report["normalized_request"]["target"] == "k_and_b"
        for report in chain.metadata["validation_reports"]
    )
    assert chain.metadata["final_answer_items"] == [
        {
            "target": "y",
            "value": "2 - x",
            "source_tool_call_id": "final_expr",
            "backend_operation": "substitute",
            "backend_confirmed": True,
        }
    ]
    assert chain.final_answer == "答案是 2 - x"
    assert verification.valid is True
    assert verification.final_answer_valid is True


def test_solver_recovers_missing_substitute_expression_from_context() -> None:
    spr = SPR.model_validate(
        {
            "source_image": {"image_path": "func_001.png"},
            "problem_text": "A line y = kx + b passes through (2, 5), and when x = 0, y = 1. Find the explicit equation.",
            "problem_type": "algebra",
            "conditions": [
                {"id": "c1", "text": "y = kx + b"},
                {"id": "c2", "text": "The line passes through (2, 5)."},
                {"id": "c3", "text": "When x = 0, y = 1."},
            ],
            "questions": [
                {
                    "id": "q1",
                    "text": "Find the explicit equation of the line.",
                    "question_type": "compute",
                    "target_ids": ["t1"],
                }
            ],
            "variables": [
                {"symbol": "x", "description": "independent variable", "domain": "real"},
                {"symbol": "y", "description": "dependent variable", "domain": "real"},
                {"symbol": "k", "description": "slope", "domain": "real"},
                {"symbol": "b", "description": "intercept", "domain": "real"},
            ],
            "targets": [{"id": "t1", "text": "Write the explicit equation of the line", "target_type": "compute"}],
            "confidence": {"overall": 0.95},
        }
    )
    emr = EMR.model_validate(
        {
            "problem_id": "problem_func_001",
            "source_problem_type": "algebra",
            "representation_type": "unknown",
        }
    )
    classification = algebra_solver_classification_result(problem_id="problem_func_001")
    planner = DeepSeekPlanner(client=FakeLinearFunctionMissingSubstituteExpressionPlanningClient())

    chain = Solver(planner=planner, use_planner=True).solve(emr, classification, spr=spr)
    verification = Verifier().verify(chain, spr=spr, emr=emr, classification_result=classification)

    assert chain.metadata["completed_tool_call_count"] == 3
    substitute_report = chain.metadata["validation_reports"][-1]
    assert substitute_report["valid"] is True
    assert substitute_report["normalized_request"]["expressions"] == [
        {"expression": "k*x+b", "name": "line_expr"}
    ]
    assert chain.metadata["final_answer_items"] == [
        {
            "target": "y",
            "value": "2*x + 1",
            "source_tool_call_id": "substitute_vals",
            "backend_operation": "substitute",
            "backend_confirmed": True,
        }
    ]
    assert chain.final_answer == "答案是 2*x + 1"
    assert verification.valid is True


def test_solver_prefers_evaluated_derivative_point_value_over_symbolic_expression() -> None:
    spr = SPR.model_validate(
        {
            "source_image": {"image_path": "deriv_010.png"},
            "problem_text": "The displacement is s(t)=t^3-3t^2. Find the instantaneous velocity at t = 2.",
            "problem_type": "calculus",
            "conditions": [{"id": "c1", "text": "s(t)=t^3-3t^2"}],
            "questions": [
                {
                    "id": "q1",
                    "text": "Find the instantaneous velocity at t = 2.",
                    "question_type": "compute",
                    "target_ids": ["t1"],
                }
            ],
            "variables": [
                {"symbol": "t", "description": "time", "domain": "real"},
                {"symbol": "v", "description": "instantaneous velocity", "domain": "real"},
            ],
            "targets": [{"id": "t1", "text": "velocity at t = 2", "target_type": "compute"}],
            "confidence": {"overall": 0.95},
        }
    )
    emr = EMR.model_validate(
        {
            "problem_id": "problem_deriv_010",
            "source_problem_type": "calculus",
            "representation_type": "unknown",
        }
    )
    classification = algebra_solver_classification_result(problem_id="problem_deriv_010")
    planner = DeepSeekPlanner(client=FakeDerivativePointValuePlanningClient())

    chain = Solver(planner=planner, use_planner=True).solve(emr, classification, spr=spr)
    verification = Verifier().verify(chain, spr=spr, emr=emr, classification_result=classification)

    assert chain.metadata["final_answer_items"] == [
        {
            "target": "v",
            "value": "0",
            "source_tool_call_id": "eval_v_at_2",
            "backend_operation": "evaluate",
            "backend_confirmed": True,
        }
    ]
    assert "0" in (chain.final_answer or "")
    assert "3*t**2 - 6*t" not in (chain.final_answer or "")
    assert verification.valid is True
    assert verification.final_answer_valid is True


def test_solver_treats_symbolic_evaluate_result_as_substitution() -> None:
    spr = SPR.model_validate(
        {
            "source_image": {"image_path": "derivative_param.png"},
            "problem_text": "Given f''(x) = x - 2a, compute f''(1) as an expression in a.",
            "problem_type": "calculus",
            "conditions": [{"id": "c1", "text": "f''(x) = x - 2a"}],
            "questions": [
                {
                    "id": "q1",
                    "text": "Compute f''(1) as an expression in a.",
                    "question_type": "compute",
                    "target_ids": ["t1"],
                }
            ],
            "variables": [
                {"symbol": "x", "description": "independent variable", "domain": "real"},
                {"symbol": "a", "description": "parameter", "domain": "real"},
            ],
            "targets": [{"id": "t1", "text": "expression for f''(1)", "target_type": "compute"}],
            "confidence": {"overall": 0.95},
        }
    )
    emr = EMR.model_validate(
        {
            "problem_id": "problem_derivative_param",
            "source_problem_type": "calculus",
            "representation_type": "unknown",
        }
    )
    classification = algebra_solver_classification_result(problem_id="problem_derivative_param")
    planner = DeepSeekPlanner(client=FakeSymbolicEvaluateFallbackPlanningClient())

    chain = Solver(planner=planner, use_planner=True).solve(emr, classification, spr=spr)
    verification = Verifier().verify(chain, spr=spr, emr=emr, classification_result=classification)

    assert chain.metadata["partial_failure"] is False
    assert chain.metadata["final_answer_items"] == [
        {
            "target": "param_expr_at_1",
            "value": "1 - 2*a",
            "source_tool_call_id": "eval_param_expr",
            "backend_operation": "substitute",
            "backend_confirmed": True,
        }
    ]
    assert verification.valid is True


def test_solver_normalizes_derivative_function_notation() -> None:
    spr = SPR.model_validate(
        {
            "source_image": {"image_path": "deriv_003.png"},
            "problem_text": "Given y = x^4 - 2x^3 + 5x, find the derivative.",
            "problem_type": "calculus",
            "conditions": [{"id": "c1", "text": "y = x^4 - 2x^3 + 5x"}],
            "questions": [
                {
                    "id": "q1",
                    "text": "Find the derivative.",
                    "question_type": "compute",
                    "target_ids": ["t1"],
                }
            ],
            "variables": [{"symbol": "x", "description": "independent variable", "domain": "real"}],
            "targets": [{"id": "t1", "text": "derivative", "target_type": "compute"}],
            "confidence": {"overall": 0.95},
        }
    )
    emr = EMR.model_validate(
        {
            "problem_id": "problem_deriv_003",
            "source_problem_type": "calculus",
            "representation_type": "unknown",
        }
    )
    classification = algebra_solver_classification_result(problem_id="problem_deriv_003")
    planner = DeepSeekPlanner(client=FakeDerivativeFunctionNotationPlanningClient())

    chain = Solver(planner=planner, use_planner=True).solve(emr, classification, spr=spr)
    verification = Verifier().verify(chain, spr=spr, emr=emr, classification_result=classification)

    assert chain.metadata["partial_failure"] is False
    assert chain.metadata["final_answer_items"] == [
        {
            "target": "deriv_result",
            "value": "4*x**3 - 6*x**2 + 5",
            "source_tool_call_id": "deriv",
            "backend_operation": "construct_expression",
            "backend_confirmed": True,
        }
    ]
    assert verification.valid is True


def test_solver_retains_function_expression_target_when_planner_uses_coefficient_target() -> None:
    spr = SPR.model_validate(
        {
            "source_image": {"image_path": "func_003.png"},
            "problem_text": "Find the function expression of the line.",
            "problem_type": "algebra",
            "conditions": [
                {"id": "c1", "text": "The line has intercept 3."},
                {"id": "c2", "text": "The x-intercept condition determines the slope."},
            ],
            "questions": [
                {
                    "id": "q1",
                    "text": "Find the function expression.",
                    "question_type": "compute",
                    "target_ids": ["t1"],
                }
            ],
            "variables": [
                {"symbol": "x", "description": "independent variable", "domain": "real"},
                {"symbol": "y", "description": "dependent variable", "domain": "real"},
                {"symbol": "k", "description": "slope", "domain": "real"},
            ],
            "targets": [{"id": "t1", "text": "function expression", "target_type": "compute"}],
            "confidence": {"overall": 0.95},
        }
    )
    emr = EMR.model_validate(
        {
            "problem_id": "problem_func_003",
            "source_problem_type": "algebra",
            "representation_type": "unknown",
        }
    )
    classification = algebra_solver_classification_result(problem_id="problem_func_003")
    planner = DeepSeekPlanner(client=FakeLinearFunctionFinalExpressionAsCoefficientPlanningClient())

    chain = Solver(planner=planner, use_planner=True).solve(emr, classification, spr=spr)
    verification = Verifier().verify(chain, spr=spr, emr=emr, classification_result=classification)

    assert chain.metadata["final_answer_items"] == [
        {
            "target": "y",
            "value": "3 - x/2",
            "source_tool_call_id": "substitute_final",
            "backend_operation": "substitute",
            "backend_confirmed": True,
        }
    ]
    assert verification.valid is True


def test_solver_keeps_function_expression_and_optimization_value_for_separate_questions() -> None:
    spr = SPR.model_validate(
        {
            "source_image": {"image_path": "func_021.png"},
            "problem_text": (
                "A company sells a product at price x yuan. Daily sales quantity is 200 - 4x. "
                "Let daily revenue be y. Find the function expression of y in terms of x, "
                "and find the selling price when revenue is maximized."
            ),
            "problem_type": "algebra",
            "conditions": [
                {"id": "c1", "text": "Price is x."},
                {"id": "c2", "text": "Daily sales quantity is 200 - 4x."},
                {"id": "c3", "text": "Daily revenue is y."},
            ],
            "questions": [
                {
                    "id": "q1",
                    "text": "Find the function expression of y in terms of x.",
                    "question_type": "compute",
                    "target_ids": ["t1"],
                },
                {
                    "id": "q2",
                    "text": "Find the selling price when revenue is maximized.",
                    "question_type": "compute",
                    "target_ids": ["t2"],
                },
            ],
            "variables": [
                {"symbol": "x", "description": "selling price", "domain": "positive real"},
                {"symbol": "y", "description": "daily revenue", "domain": "real"},
            ],
            "targets": [
                {"id": "t1", "text": "Function expression between y and x", "target_type": "compute"},
                {"id": "t2", "text": "Selling price when revenue is maximized", "target_type": "compute"},
            ],
            "confidence": {"overall": 0.95},
        }
    )
    emr = EMR.model_validate(
        {
            "problem_id": "problem_func_021",
            "source_problem_type": "algebra",
            "representation_type": "unknown",
        }
    )
    classification = algebra_solver_classification_result(problem_id="problem_func_021")
    planner = DeepSeekPlanner(client=FakeRevenueExpressionAndMaximumPricePlanningClient())

    chain = Solver(planner=planner, use_planner=True).solve(emr, classification, spr=spr)
    verification = Verifier().verify(chain, spr=spr, emr=emr, classification_result=classification)

    assert chain.metadata["final_answer_items"] == [
        {
            "target": "y",
            "value": "200*x - 4*x^2",
            "source_tool_call_id": "build_revenue",
            "backend_operation": "construct_expression",
            "backend_confirmed": True,
        },
        {
            "target": "x",
            "value": "25",
            "source_tool_call_id": "solve_max_price",
            "backend_operation": "solve_for",
            "backend_confirmed": True,
        },
    ]
    assert "200*x - 4*x^2" in (chain.final_answer or "")
    assert "x = 25" in (chain.final_answer or "")
    assert verification.valid is True
    assert verification.final_answer_valid is True


def test_solver_keeps_both_vertex_coordinates_when_planner_computes_them() -> None:
    spr = SPR.model_validate(
        {
            "source_image": {"image_path": "func_012.png"},
            "problem_text": (
                "Given the quadratic function y = x^2 - 4x + c passes through (1, 2), "
                "find c and the vertex coordinates."
            ),
            "problem_type": "algebra",
            "conditions": [
                {"id": "c1", "text": "y = x^2 - 4x + c"},
                {"id": "c2", "text": "The graph passes through (1, 2)."},
            ],
            "questions": [
                {
                    "id": "q1",
                    "text": "Find c.",
                    "question_type": "compute",
                    "target_ids": ["t1"],
                },
                {
                    "id": "q2",
                    "text": "Find the vertex coordinates.",
                    "question_type": "compute",
                    "target_ids": ["t2"],
                },
            ],
            "variables": [
                {"symbol": "x", "description": "independent variable", "domain": "real"},
                {"symbol": "y", "description": "dependent variable", "domain": "real"},
                {"symbol": "c", "description": "constant term", "domain": "real"},
            ],
            "targets": [
                {"id": "t1", "text": "value of c", "target_type": "compute"},
                {"id": "t2", "text": "coordinates of the vertex", "target_type": "compute"},
            ],
            "confidence": {"overall": 0.95},
        }
    )
    emr = EMR.model_validate(
        {
            "problem_id": "problem_func_012",
            "source_problem_type": "algebra",
            "representation_type": "unknown",
        }
    )
    classification = algebra_solver_classification_result(problem_id="problem_func_012")
    planner = DeepSeekPlanner(client=FakeQuadraticVertexCoordinatesPlanningClient())

    chain = Solver(planner=planner, use_planner=True).solve(emr, classification, spr=spr)
    verification = Verifier().verify(chain, spr=spr, emr=emr, classification_result=classification)

    assert chain.metadata["final_answer_items"] == [
        {
            "target": "c",
            "value": "5",
            "source_tool_call_id": "solve_c",
            "backend_operation": "solve_for",
            "backend_confirmed": True,
        },
        {
            "target": "vertex_x",
            "value": "2",
            "source_tool_call_id": "step2_x_vertex",
            "backend_operation": "evaluate",
            "backend_confirmed": True,
        },
        {
            "target": "vertex_y",
            "value": "1",
            "source_tool_call_id": "step3_y_vertex",
            "backend_operation": "substitute",
            "backend_confirmed": True,
        },
    ]
    assert "c = 5" in (chain.final_answer or "")
    assert "vertex_x = 2" in (chain.final_answer or "")
    assert "vertex_y = 1" in (chain.final_answer or "")
    assert verification.valid is True
    assert verification.final_answer_valid is True


def test_solver_accepts_vertex_y_local_symbol_bound_to_prior_result() -> None:
    spr = SPR.model_validate(
        {
            "source_image": {"image_path": "func_012.png"},
            "problem_text": (
                "Given the quadratic function y = x^2 - 4x + c passes through (1, 2), "
                "find c and the vertex coordinates."
            ),
            "problem_type": "algebra",
            "conditions": [
                {"id": "c1", "text": "y = x^2 - 4x + c"},
                {"id": "c2", "text": "The graph passes through (1, 2)."},
            ],
            "questions": [
                {"id": "q1", "text": "Find c.", "question_type": "compute", "target_ids": ["t1"]},
                {
                    "id": "q2",
                    "text": "Find the vertex coordinates.",
                    "question_type": "compute",
                    "target_ids": ["t2"],
                },
            ],
            "variables": [
                {"symbol": "x", "description": "independent variable", "domain": "real"},
                {"symbol": "y", "description": "dependent variable", "domain": "real"},
                {"symbol": "c", "description": "constant term", "domain": "real"},
            ],
            "targets": [
                {"id": "t1", "text": "value of c", "target_type": "compute"},
                {"id": "t2", "text": "coordinates of the vertex", "target_type": "compute"},
            ],
            "confidence": {"overall": 0.95},
        }
    )
    emr = EMR.model_validate(
        {
            "problem_id": "problem_func_012",
            "source_problem_type": "algebra",
            "representation_type": "unknown",
        }
    )
    classification = algebra_solver_classification_result(problem_id="problem_func_012")
    planner = DeepSeekPlanner(client=FakeQuadraticVertexLocalSymbolPlanningClient())

    chain = Solver(planner=planner, use_planner=True).solve(emr, classification, spr=spr)
    verification = Verifier().verify(chain, spr=spr, emr=emr, classification_result=classification)

    assert chain.metadata["partial_failure"] is False
    assert chain.metadata["final_answer_items"] == [
        {
            "target": "c",
            "value": "5",
            "source_tool_call_id": "solve_c",
            "backend_operation": "solve_equation_system",
            "backend_confirmed": True,
        },
        {
            "target": "vertex_x",
            "value": "2",
            "source_tool_call_id": "vertex_x",
            "backend_operation": "evaluate",
            "backend_confirmed": True,
        },
        {
            "target": "vertex_y",
            "value": "1",
            "source_tool_call_id": "vertex_y",
            "backend_operation": "evaluate",
            "backend_confirmed": True,
        },
    ]
    assert verification.valid is True


def test_solver_deterministically_completes_quadratic_intercepts() -> None:
    spr = SPR.model_validate(
        {
            "source_image": {"image_path": "func_009.png"},
            "problem_text": "Find the x-axis and y-axis intercept coordinates of y = x^2 - 5x + 6.",
            "problem_type": "algebra",
            "conditions": [{"id": "c1", "text": "Given y = x^2 - 5x + 6", "linked_formula_ids": ["f1"]}],
            "questions": [
                {
                    "id": "q1",
                    "text": "Find the x-axis intercept coordinates.",
                    "question_type": "compute",
                    "target_ids": ["t1"],
                },
                {
                    "id": "q2",
                    "text": "Find the y-axis intercept coordinate.",
                    "question_type": "compute",
                    "target_ids": ["t2"],
                },
            ],
            "formulas": [
                {"id": "f1", "raw_text": "y = x^2 - 5x + 6", "latex": "y = x^2 - 5x + 6", "role": "condition"}
            ],
            "variables": [
                {"symbol": "x", "description": "independent variable", "domain": "real"},
                {"symbol": "y", "description": "dependent variable", "domain": "real"},
            ],
            "targets": [
                {"id": "t1", "text": "x-axis intercept coordinates", "target_type": "compute"},
                {"id": "t2", "text": "y-axis intercept coordinate", "target_type": "compute"},
            ],
            "confidence": {"overall": 0.95},
        }
    )
    emr = EMR.model_validate(
        {
            "problem_id": "problem_func_009",
            "source_problem_type": "algebra",
            "representation_type": "unknown",
        }
    )
    classification = algebra_solver_classification_result(problem_id="problem_func_009")
    planner = DeepSeekPlanner(client=FakeQuadraticInterceptPartialPlanningClient())

    chain = Solver(planner=planner, use_planner=True).solve(emr, classification, spr=spr)
    verification = Verifier().verify(chain, spr=spr, emr=emr, classification_result=classification)

    answer_pairs = {
        (item["target"], item["value"])
        for item in chain.metadata["final_answer_items"]
    }
    assert ("x_axis_intercepts", "(2, 0), (3, 0)") in answer_pairs
    assert ("y_axis_intercept", "(0, 6)") in answer_pairs
    assert verification.valid is True


def test_solver_resolves_bare_expression_name_for_vertex_y_substitution() -> None:
    spr = SPR.model_validate(
        {
            "source_image": {"image_path": "func_004.png"},
            "problem_text": "Find the vertex coordinates of y = x^2 - 4x + 3.",
            "problem_type": "algebra",
            "conditions": [{"id": "c1", "text": "y = x^2 - 4x + 3"}],
            "questions": [
                {
                    "id": "q1",
                    "text": "Find the vertex coordinates.",
                    "question_type": "compute",
                    "target_ids": ["t1"],
                }
            ],
            "variables": [
                {"symbol": "x", "description": "independent variable", "domain": "real"},
                {"symbol": "y", "description": "dependent variable", "domain": "real"},
            ],
            "targets": [{"id": "t1", "text": "coordinates of the vertex", "target_type": "compute"}],
            "confidence": {"overall": 0.95},
        }
    )
    emr = EMR.model_validate(
        {
            "problem_id": "problem_func_004",
            "source_problem_type": "algebra",
            "representation_type": "unknown",
        }
    )
    classification = algebra_solver_classification_result(problem_id="problem_func_004")
    planner = DeepSeekPlanner(client=FakeBareExpressionNameVertexPlanningClient())

    chain = Solver(planner=planner, use_planner=True).solve(emr, classification, spr=spr)
    verification = Verifier().verify(chain, spr=spr, emr=emr, classification_result=classification)

    assert chain.metadata["partial_failure"] is False
    assert {
        (item["target"], item["value"], item["source_tool_call_id"])
        for item in chain.metadata["final_answer_items"]
    } >= {("h_value", "2", "calc_h"), ("k_value", "-1", "substitute_y")}
    assert verification.valid is True


def test_solver_partial_failure_keeps_completed_cas_answer_items() -> None:
    spr = SPR.model_validate(
        {
            "source_image": {"image_path": "deriv_005.png"},
            "problem_text": "Given y = x^2 + 2x + 1, find the tangent line equation at x = 1.",
            "problem_type": "calculus",
            "conditions": [{"id": "c1", "text": "y = x^2 + 2x + 1"}],
            "questions": [
                {
                    "id": "q1",
                    "text": "Find the tangent line equation at x = 1.",
                    "question_type": "compute",
                    "target_ids": ["t1"],
                }
            ],
            "variables": [
                {"symbol": "x", "description": "independent variable", "domain": "real"},
                {"symbol": "y", "description": "dependent variable", "domain": "real"},
            ],
            "targets": [{"id": "t1", "text": "equation of the tangent line", "target_type": "compute"}],
            "confidence": {"overall": 0.95},
        }
    )
    emr = EMR.model_validate(
        {
            "problem_id": "problem_deriv_005",
            "source_problem_type": "calculus",
            "representation_type": "unknown",
        }
    )
    classification = algebra_solver_classification_result(problem_id="problem_deriv_005")
    planner = DeepSeekPlanner(client=FakeTangentLinePartialPlanningClient())

    chain = Solver(planner=planner, use_planner=True).solve(emr, classification, spr=spr)
    verification = Verifier().verify(chain, spr=spr, emr=emr, classification_result=classification)

    assert chain.metadata["partial_failure"] is True
    assert chain.metadata["backend_confirmed"] is True
    assert chain.metadata["final_answer_items"] == [
        {
            "target": "y",
            "value": "4",
            "source_tool_call_id": "calc_y0",
            "backend_operation": "evaluate",
            "backend_confirmed": True,
        },
        {
            "target": "m",
            "value": "4",
            "source_tool_call_id": "calc_m",
            "backend_operation": "evaluate",
            "backend_confirmed": True,
        },
    ]
    assert "y = 4" in (chain.final_answer or "")
    assert "m = 4" in (chain.final_answer or "")
    assert verification.valid is True


def test_solver_substitutes_values_into_equation_statement() -> None:
    spr = SPR.model_validate(
        {
            "source_image": {"image_path": "deriv_005.png"},
            "problem_text": "Given y = x^2 + 2x + 1, find the tangent line equation at x = 1.",
            "problem_type": "calculus",
            "conditions": [{"id": "c1", "text": "y = x^2 + 2x + 1"}],
            "questions": [
                {
                    "id": "q1",
                    "text": "Find the tangent line equation at x = 1.",
                    "question_type": "compute",
                    "target_ids": ["t1"],
                }
            ],
            "variables": [
                {"symbol": "x", "description": "independent variable", "domain": "real"},
                {"symbol": "y", "description": "dependent variable", "domain": "real"},
            ],
            "targets": [{"id": "t1", "text": "tangent line equation", "target_type": "compute"}],
            "confidence": {"overall": 0.95},
        }
    )
    emr = EMR.model_validate(
        {
            "problem_id": "problem_deriv_005",
            "source_problem_type": "calculus",
            "representation_type": "unknown",
        }
    )
    classification = algebra_solver_classification_result(problem_id="problem_deriv_005")
    planner = DeepSeekPlanner(client=FakeTangentEquationSubstitutePlanningClient())

    chain = Solver(planner=planner, use_planner=True).solve(emr, classification, spr=spr)
    verification = Verifier().verify(chain, spr=spr, emr=emr, classification_result=classification)

    assert chain.metadata["partial_failure"] is False
    tangent_step = next(
        step
        for step in chain.steps
        if step.diagnostics.get("tool_call_id") == "substitute_tangent"
        and step.rule_name == "cas_substitute"
    )
    assert tangent_step.diagnostics["output"]["value"] == "y = 4*x"
    assert verification.valid is True


def test_solver_derives_linear_function_expression_from_coefficients() -> None:
    spr = SPR.model_validate(
        {
            "source_image": {"image_path": "func_001.png"},
            "problem_text": "A line y = kx + b passes through (2, 5), and when x = 0, y = 1. Find the explicit equation.",
            "problem_type": "algebra",
            "conditions": [
                {"id": "c1", "text": "y = kx + b"},
                {"id": "c2", "text": "The line passes through (2, 5)."},
                {"id": "c3", "text": "When x = 0, y = 1."},
            ],
            "questions": [
                {
                    "id": "q1",
                    "text": "Find the explicit equation of the line.",
                    "question_type": "compute",
                    "target_ids": ["t1"],
                }
            ],
            "variables": [
                {"symbol": "x", "description": "independent variable", "domain": "real"},
                {"symbol": "y", "description": "dependent variable", "domain": "real"},
                {"symbol": "k", "description": "slope", "domain": "real"},
                {"symbol": "b", "description": "intercept", "domain": "real"},
            ],
            "targets": [{"id": "t1", "text": "Write the explicit equation of the line", "target_type": "compute"}],
            "confidence": {"overall": 0.95},
        }
    )
    emr = EMR.model_validate(
        {
            "problem_id": "problem_func_001",
            "source_problem_type": "algebra",
            "representation_type": "unknown",
        }
    )
    classification = algebra_solver_classification_result(problem_id="problem_func_001")
    planner = DeepSeekPlanner(client=FakeLinearFunctionCoefficientOnlyPlanningClient())

    chain = Solver(planner=planner, use_planner=True).solve(emr, classification, spr=spr)
    verification = Verifier().verify(chain, spr=spr, emr=emr, classification_result=classification)

    assert any(
        step.rule_name == "cas_substitute"
        and step.diagnostics.get("tool_call_id") == "derived_linear_function_expression"
        for step in chain.steps
    )
    assert chain.metadata["final_answer_items"] == [
        {
            "target": "y",
            "value": "2*x + 1",
            "source_tool_call_id": "derived_linear_function_expression",
            "backend_operation": "substitute",
            "backend_confirmed": True,
        }
    ]
    assert chain.final_answer == "答案是 2*x + 1"
    assert verification.valid is True


def test_calculation_request_validator_accepts_short_reference_with_substitution() -> None:
    spr_data = math10_spr().model_dump(mode="json")
    spr_data["variables"] = [
        {"symbol": "N_total", "description": "Total number of oranges", "domain": "integer"},
        {"symbol": "N_good", "description": "Number of good oranges", "domain": "integer"},
    ]
    spr = SPR.model_validate(spr_data)
    emr = math10_unknown_emr()

    report = CalculationRequestValidator().validate(
        {
            "id": "good_count",
            "operation": "evaluate",
            "variables": ["unripe_count"],
            "expressions": [{"name": "good_count", "expression": "25 - 1 - $unripe_count - 2"}],
            "substitutions": {"unripe_count": "$unripe_count.unripe_count"},
            "target": "good_count",
        },
        emr,
        spr=spr,
        execution_context={
            "variables": {"unripe_count": "5"},
            "unripe_count": {"unripe_count": "5", "value": "5"},
        },
    )

    assert report["valid"] is True
    assert report["normalized_request"]["expressions"][0]["expression"] == "25 - 1 - unripe_count - 2"


class FakeGSM8KChargePlanningClient:
    model = "fake-deepseek-v4-flash"

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        assert "tool_calls" in system_prompt + user_prompt
        return (
            '{"tool_calls":['
            '{"id":"remaining_charge","operation":"evaluate","variables":[],'
            '"expressions":[{"name":"remaining","expression":"100 - 60"}],'
            '"substitutions":{},"target":"remaining"},'
            '{"id":"time_minutes","operation":"evaluate","variables":["remaining"],'
            '"expressions":[{"name":"t_min","expression":"remaining * 3"}],'
            '"substitutions":{"remaining":"$remaining_charge.remaining"},"target":"t_min"},'
            '{"id":"time_hours","operation":"evaluate","variables":["t_min"],'
            '"expressions":[{"name":"t_hours","expression":"t_min / 60"}],'
            '"substitutions":{"t_min":"$time_minutes.t_min"},"target":"t_hours"}'
            ']}'
        )


class FakeGSM8KChargeConstructExpressionClient:
    model = "fake-qwen3.6-plus"

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        assert "tool_calls" in system_prompt + user_prompt
        return (
            '{"tool_calls":['
            '{"id":"calc_remaining_charge","operation":"construct_expression","variables":[],'
            '"expressions":[{"name":"remaining_charge","expression":"100 - 60"}],'
            '"substitutions":{},"target":"remaining_charge"},'
            '{"id":"calc_time_minutes","operation":"construct_expression","variables":[],'
            '"expressions":[{"name":"time_minutes","expression":"$calc_remaining_charge.remaining_charge * 3"}],'
            '"substitutions":{},"target":"time_minutes"},'
            '{"id":"final_answer","operation":"construct_expression","variables":[],'
            '"expressions":[{"name":"time_hours","expression":"$calc_time_minutes.time_minutes / 60"}],'
            '"substitutions":{},"target":"time_hours"}'
            ']}'
        )


def test_solver_prefers_requested_hours_over_intermediate_minutes() -> None:
    spr = SPR.model_validate(
        {
            "source_image": {"image_path": "charge.png"},
            "problem_text": (
                "The cell-phone recharges at the rate of 1 percentage-point of charge "
                "per 3 minutes. Now, the phone is at 60% charged. How long will it "
                "take to fully charge, in hours?"
            ),
            "problem_stem": "Phone charging word problem.",
            "conditions": [],
            "questions": [
                {
                    "id": "q1",
                    "text": "How long will it take to fully charge, in hours?",
                    "question_type": "compute",
                    "target_ids": ["t1", "t2", "t3"],
                }
            ],
            "formulas": [],
            "variables": [
                {"symbol": "Rate", "description": "Charging rate", "domain": "positive"},
                {
                    "symbol": "t_hours",
                    "description": "Time required to fully charge in hours",
                    "domain": "positive",
                },
            ],
            "problem_type": "word_problem",
            "knowledge_units": ["rates"],
            "targets": [
                {"id": "t1", "text": "Calculate remaining charge needed", "target_type": "compute"},
                {"id": "t2", "text": "Calculate time in minutes", "target_type": "compute"},
                {"id": "t3", "text": "Convert time to hours", "target_type": "compute"},
            ],
            "confidence": {"overall": 1.0},
            "metadata": {"problem_id": "charge"},
        }
    )
    emr = EMR.model_validate(
        {
            "problem_id": "charge",
            "source_problem_type": "word_problem",
            "representation_type": "unknown",
        }
    )
    classification = ProblemClassifier().classify(spr, emr)

    chain = Solver(
        planner=DeepSeekPlanner(client=FakeGSM8KChargePlanningClient()),
        use_planner=True,
    ).solve(emr, classification, spr=spr)

    assert chain.final_answer == "t_hours = 2"
    assert chain.metadata["final_answer_items"] == [
        {
            "target": "t_hours",
            "value": "2",
            "source_tool_call_id": "time_hours",
            "backend_operation": "evaluate",
            "backend_confirmed": True,
        }
    ]


def test_solver_parenthesizes_construct_expression_references_and_selects_final_hours() -> None:
    spr = SPR.model_validate(
        {
            "source_image": {"image_path": "charge.png"},
            "problem_text": (
                "The cell-phone recharges at the rate of 1 percentage-point of charge "
                "per 3 minutes. Now, the phone is at 60% charged. How long will it "
                "take to fully charge, in hours?"
            ),
            "problem_stem": "Phone charging word problem.",
            "conditions": [],
            "questions": [
                {
                    "id": "q1",
                    "text": "How long will it take to fully charge, in hours?",
                    "question_type": "compute",
                    "target_ids": ["t1"],
                }
            ],
            "formulas": [],
            "variables": [
                {
                    "symbol": "t",
                    "description": "Time required to fully charge",
                    "domain": "time",
                },
            ],
            "problem_type": "word_problem",
            "knowledge_units": ["rates", "unit_conversion_minutes_to_hours"],
            "targets": [
                {
                    "id": "t1",
                    "text": "Calculate the time in hours required for the battery to reach 100% from 60%.",
                    "target_type": "compute",
                },
            ],
            "confidence": {"overall": 1.0},
            "metadata": {"problem_id": "charge_construct"},
        }
    )
    emr = EMR.model_validate(
        {
            "problem_id": "charge_construct",
            "source_problem_type": "word_problem",
            "representation_type": "unknown",
        }
    )
    classification = ProblemClassifier().classify(spr, emr)

    chain = Solver(
        planner=DeepSeekPlanner(client=FakeGSM8KChargeConstructExpressionClient()),
        use_planner=True,
    ).solve(emr, classification, spr=spr)

    assert chain.final_answer == "2"
    assert chain.metadata["final_answer_items"] == [
        {
            "target": "t",
            "value": "2",
            "source_tool_call_id": "final_answer",
            "backend_operation": "construct_expression",
            "backend_confirmed": True,
        }
    ]
    assert chain.metadata["execution_context"]["calc_time_minutes"]["time_minutes"] == "120"
    assert chain.metadata["execution_context"]["final_answer"]["time_hours"] == "2"

    verification = Verifier().verify(chain, spr=spr, emr=emr)
    assert verification.valid is True
    assert verification.status == "passed"


class FakeGSM8KAgePlanningClient:
    model = "fake-deepseek-v4-flash"

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        assert "tool_calls" in system_prompt + user_prompt
        return (
            '{"tool_calls":[{"id":"solve_sam_age","operation":"solve_for",'
            '"variables":["R","S"],'
            '"equations":[{"lhs":"R","rhs":"6*S"},{"lhs":"R+9","rhs":"3*(S+9)"}],'
            '"substitutions":{},"target":"S"}]}'
        )


def test_solver_and_verifier_accept_multiequation_solve_for_as_system() -> None:
    spr = SPR.model_validate(
        {
            "source_image": {"image_path": "age.png"},
            "problem_text": (
                "Ruby is 6 times older than Sam. In 9 years, Ruby will be 3 times "
                "as old as Sam. How old is Sam now?"
            ),
            "problem_stem": "Age word problem.",
            "conditions": [],
            "questions": [
                {
                    "id": "q1",
                    "text": "How old is Sam now?",
                    "question_type": "solve",
                    "target_ids": ["t1"],
                }
            ],
            "formulas": [],
            "variables": [
                {"symbol": "R", "description": "Ruby's current age", "domain": "positive"},
                {"symbol": "S", "description": "Sam's current age", "domain": "positive"},
            ],
            "problem_type": "word_problem",
            "knowledge_units": ["linear equations"],
            "targets": [{"id": "t1", "text": "Sam's current age", "target_type": "solve"}],
            "confidence": {"overall": 1.0},
            "metadata": {"problem_id": "age"},
        }
    )
    emr = EMR.model_validate(
        {
            "problem_id": "age",
            "source_problem_type": "word_problem",
            "representation_type": "unknown",
        }
    )
    classification = ProblemClassifier().classify(spr, emr)

    chain = Solver(
        planner=DeepSeekPlanner(client=FakeGSM8KAgePlanningClient()),
        use_planner=True,
    ).solve(emr, classification, spr=spr)
    report = Verifier().verify(chain, spr=spr, emr=emr, classification_result=classification)

    assert chain.final_answer == "S = 6"
    assert chain.metadata["partial_failure"] is False
    assert report.valid is True


class FakeGSM8KSavingsPlanningClient:
    model = "fake-deepseek-v4-flash"

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        assert "tool_calls" in system_prompt + user_prompt
        return (
            '{"tool_calls":['
            '{"id":"compute_total_cost","operation":"evaluate","variables":[],'
            '"expressions":[{"name":"total_cost","expression":"2*5*2.20"}],'
            '"substitutions":{},"target":"total_cost"},'
            '{"id":"compute_savings","operation":"evaluate","variables":["total_cost"],'
            '"expressions":[{"name":"savings","expression":"total_cost - 20"}],'
            '"substitutions":{"total_cost":"$compute_total_cost.total_cost"},"target":"savings"}'
            ']}'
        )


def test_solver_prefers_savings_over_total_cost() -> None:
    spr = SPR.model_validate(
        {
            "source_image": {"image_path": "bus.png"},
            "problem_text": (
                "Janet takes two bus trips five days a week. If each bus trip costs "
                "her $2.20, how much would she save by buying a weekly bus pass for $20?"
            ),
            "conditions": [],
            "questions": [
                {"id": "q1", "text": "How much would she save?", "question_type": "compute", "target_ids": ["t1"]}
            ],
            "variables": [{"symbol": "S", "description": "savings", "domain": "real"}],
            "problem_type": "word_problem",
            "knowledge_units": ["money"],
            "targets": [{"id": "t1", "text": "savings", "target_type": "compute"}],
            "confidence": {"overall": 1.0},
        }
    )
    emr = EMR.model_validate(
        {"problem_id": "bus", "source_problem_type": "word_problem", "representation_type": "unknown"}
    )
    classification = ProblemClassifier().classify(spr, emr)

    chain = Solver(
        planner=DeepSeekPlanner(client=FakeGSM8KSavingsPlanningClient()),
        use_planner=True,
    ).solve(emr, classification, spr=spr)

    assert "2.00000000000000" in (chain.final_answer or "")
    assert chain.metadata["final_answer_items"][0]["source_tool_call_id"] == "compute_savings"


class FakeGSM8KLeftPlanningClient:
    model = "fake-deepseek-v4-flash"

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        assert "tool_calls" in system_prompt + user_prompt
        return (
            '{"tool_calls":['
            '{"id":"find_bill","operation":"evaluate","variables":[],'
            '"expressions":[{"name":"S_bill","expression":"1/3 * 12"}],'
            '"substitutions":{},"target":"S_bill"},'
            '{"id":"find_mark","operation":"evaluate","variables":[],'
            '"expressions":[{"name":"S_mark","expression":"1/4 * 12"}],'
            '"substitutions":{},"target":"S_mark"},'
            '{"id":"find_left","operation":"evaluate",'
            '"variables":["S_total","S_bill","S_mark","S_jenny"],'
            '"expressions":[{"name":"S_left","expression":"S_total - S_bill - S_mark - S_jenny"}],'
            '"substitutions":{"S_total":"12","S_bill":"$find_bill.S_bill",'
            '"S_mark":"$find_mark.S_mark","S_jenny":"2"},"target":"S_left"}'
            ']}'
        )


def test_solver_prefers_leftover_answer_over_given_slices() -> None:
    spr = SPR.model_validate(
        {
            "source_image": {"image_path": "pizza.png"},
            "problem_text": (
                "Jenny is dividing up a pizza with 12 slices. She gives 1/3 to Bill "
                "and 1/4 to Mark. If Jenny eats 2 slices, how many slices are left?"
            ),
            "conditions": [],
            "questions": [
                {"id": "q1", "text": "How many slices are left?", "question_type": "compute", "target_ids": ["t1"]}
            ],
            "variables": [{"symbol": "S_{left}", "description": "slices left", "domain": "integer"}],
            "problem_type": "word_problem",
            "knowledge_units": ["fractions"],
            "targets": [{"id": "t1", "text": "slices left", "target_type": "compute"}],
            "confidence": {"overall": 1.0},
        }
    )
    emr = EMR.model_validate(
        {"problem_id": "pizza", "source_problem_type": "word_problem", "representation_type": "unknown"}
    )
    classification = ProblemClassifier().classify(spr, emr)

    chain = Solver(
        planner=DeepSeekPlanner(client=FakeGSM8KLeftPlanningClient()),
        use_planner=True,
    ).solve(emr, classification, spr=spr)

    assert "3" in (chain.final_answer or "")
    assert chain.metadata["final_answer_items"][0]["source_tool_call_id"] == "find_left"


def _gsm8k_word_spr(
    problem_id: str,
    question_text: str,
    target_text: str,
    variables: list[dict[str, str]] | None = None,
) -> SPR:
    return SPR.model_validate(
        {
            "source_image": {"image_path": f"{problem_id}.png"},
            "problem_text": question_text,
            "conditions": [],
            "questions": [
                {
                    "id": "q1",
                    "text": question_text,
                    "question_type": "compute",
                    "target_ids": ["t1"],
                }
            ],
            "variables": variables or [],
            "problem_type": "word_problem",
            "knowledge_units": ["gsm8k"],
            "targets": [{"id": "t1", "text": target_text, "target_type": "compute"}],
            "confidence": {"overall": 1.0},
            "metadata": {"problem_id": problem_id},
        }
    )


def _unknown_word_emr(problem_id: str) -> EMR:
    return EMR.model_validate(
        {
            "problem_id": problem_id,
            "source_problem_type": "word_problem",
            "representation_type": "unknown",
        }
    )


def test_calculation_request_validator_normalizes_dollar_substitution_key_and_bare_reference() -> None:
    spr = _gsm8k_word_spr(
        "eggs",
        "How much does Lloyd make on eggs per week?",
        "weekly earnings",
        [{"symbol": "E_week", "description": "weekly earnings", "domain": "real"}],
    )
    emr = _unknown_word_emr("eggs")

    report = CalculationRequestValidator().validate(
        {
            "id": "weekly_earnings",
            "operation": "evaluate",
            "variables": [],
            "expressions": [{"name": "E_week", "expression": "$daily_earnings.E_day * 7"}],
            "substitutions": {"$daily_earnings.E_day": "$daily_earnings.E_day"},
            "target": "E_week",
        },
        emr,
        spr=spr,
        execution_context={
            "variables": {"E_day": "42"},
            "daily_earnings": {"E_day": "42", "value": "42"},
        },
    )

    assert report["valid"] is True
    assert report["normalized_request"]["substitutions"] == {}

    assert CalculationRequestValidator.resolve_reference(
        "$daily_earnings",
        {"daily_earnings": {"E_day": "42", "value": "42"}},
    ) == "42"


def test_calculation_request_validator_resolves_result_alias_reference() -> None:
    context = {
        "calc_total_time": {
            "operation": "evaluate",
            "output": {"value": "180", "formatted_solution": "180"},
            "value": "180",
        }
    }

    assert CalculationRequestValidator.resolve_reference("$calc_total_time.result", context) == "180"


class FakeGSM8KResultReferencePlanningClient:
    model = "fake-qwen3.6-plus"
    provider = "qwen"

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        assert "last tool call directly computes" in system_prompt
        return (
            '{"tool_calls":['
            '{"id":"calc_total_time","operation":"evaluate","variables":[],'
            '"expressions":[{"name":"total_time","expression":"(11 - 8) * 60"}],'
            '"substitutions":{},"target":"total_time"},'
            '{"id":"calc_work_time","operation":"evaluate","variables":[],'
            '"expressions":[{"name":"work_time","expression":"$calc_total_time.result - 30"}],'
            '"substitutions":{},"target":"work_time"},'
            '{"id":"compute_final_answer","operation":"evaluate","variables":[],'
            '"expressions":[{"name":"earnings","expression":"$calc_work_time.result / 10 * 5"}],'
            '"substitutions":{},"target":"earnings"}'
            ']}'
        )


def test_solver_resolves_qwen_result_reference_chain_to_final_answer() -> None:
    spr = _gsm8k_word_spr(
        "susan_pay",
        (
            "Susan earns $5 every 10 minutes. If she works between 8 a.m. and "
            "11 a.m. and pauses for half an hour, how much money does she earn?"
        ),
        "money earned",
        [{"symbol": "earnings", "description": "money earned", "domain": "real"}],
    )
    emr = _unknown_word_emr("susan_pay")
    classification = ProblemClassifier().classify(spr, emr)

    chain = Solver(
        planner=DeepSeekPlanner(client=FakeGSM8KResultReferencePlanningClient()),
        use_planner=True,
    ).solve(emr, classification, spr=spr)

    assert chain.metadata["partial_failure"] is False
    assert chain.final_answer == "答案是 75"


def test_calculation_request_validator_rejects_ungrounded_numeric_word_problem_expression() -> None:
    spr = _gsm8k_word_spr(
        "jen_april",
        "Jen works 4 weeks in April. How much will she receive?",
        "total amount received",
        [],
    )
    emr = _unknown_word_emr("jen_april")

    report = CalculationRequestValidator().validate(
        {
            "id": "calc_total_hours",
            "operation": "substitute",
            "variables": ["hours_per_day", "days_per_week", "weeks", "total_hours"],
            "expressions": [{"name": "total_hours", "expression": "hours_per_day * days_per_week * weeks"}],
            "substitutions": {},
            "target": "total_hours",
        },
        emr,
        spr=spr,
    )

    assert report["valid"] is False
    assert any("must ground every expression symbol" in error["message"] for error in report["errors"])


class FakeGSM8KWeeklyEarningsPlanningClient:
    model = "fake-deepseek-v4-flash"

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        assert "tool_calls" in system_prompt + user_prompt
        return (
            '{"tool_calls":['
            '{"id":"daily_earnings","operation":"evaluate","variables":[],'
            '"expressions":[{"name":"E_day","expression":"252 * 2 / 12"}],'
            '"substitutions":{},"target":"E_day"},'
            '{"id":"weekly_earnings","operation":"evaluate","variables":[],'
            '"expressions":[{"name":"E_week","expression":"$daily_earnings.E_day * 7"}],'
            '"substitutions":{"$daily_earnings.E_day":"$daily_earnings.E_day"},'
            '"target":"E_week"}'
            ']}'
        )


def test_solver_prefers_weekly_earnings_over_daily_earnings() -> None:
    spr = _gsm8k_word_spr(
        "eggs",
        "How much does Lloyd make on eggs per week?",
        "weekly earnings",
        [
            {"symbol": "E_day", "description": "daily earnings", "domain": "real"},
            {"symbol": "E_week", "description": "weekly earnings", "domain": "real"},
        ],
    )
    emr = _unknown_word_emr("eggs")
    classification = ProblemClassifier().classify(spr, emr)

    chain = Solver(
        planner=DeepSeekPlanner(client=FakeGSM8KWeeklyEarningsPlanningClient()),
        use_planner=True,
    ).solve(emr, classification, spr=spr)

    assert "294" in (chain.final_answer or "")
    assert chain.metadata["partial_failure"] is False
    assert chain.metadata["final_answer_items"][0]["source_tool_call_id"] == "weekly_earnings"


class FakeGSM8KPercentPlanningClient:
    model = "fake-deepseek-v4-flash"

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        assert "tool_calls" in system_prompt + user_prompt
        return (
            '{"tool_calls":['
            '{"id":"calc_contemp","operation":"evaluate","variables":[],'
            '"expressions":[{"name":"contemp_count","expression":"0.20 * 20"}],'
            '"substitutions":{},"target":"contemp_count"},'
            '{"id":"calc_hiphop_pct","operation":"evaluate","variables":["hiphop_count","total"],'
            '"expressions":[{"name":"hiphop_pct","expression":"(hiphop_count / total) * 100"}],'
            '"substitutions":{"hiphop_count":"12","total":"20"},"target":"hiphop_pct"}'
            ']}'
        )


def test_solver_prefers_percentage_answer_over_count() -> None:
    spr = _gsm8k_word_spr(
        "dance",
        "What percentage of the entire students enrolled in hip-hop dance?",
        "hip-hop percentage",
        [{"symbol": "P_hiphop", "description": "hip-hop percentage", "domain": "real"}],
    )
    emr = _unknown_word_emr("dance")
    classification = ProblemClassifier().classify(spr, emr)

    chain = Solver(
        planner=DeepSeekPlanner(client=FakeGSM8KPercentPlanningClient()),
        use_planner=True,
    ).solve(emr, classification, spr=spr)

    assert "60" in (chain.final_answer or "")
    assert chain.metadata["final_answer_items"][0]["source_tool_call_id"] == "calc_hiphop_pct"


class FakeGSM8KBookcasePlanningClient:
    model = "fake-deepseek-v4-flash"

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        assert "tool_calls" in system_prompt + user_prompt
        return (
            '{"tool_calls":['
            '{"id":"case_capacity","operation":"evaluate","variables":[],"expressions":'
            '[{"name":"C_case","expression":"55"}],"substitutions":{},"target":"C_case"},'
            '{"id":"number_of_cases","operation":"evaluate","variables":["B_total","C_case"],'
            '"expressions":[{"name":"N_cases","expression":"ceil(B_total / C_case)"}],'
            '"substitutions":{"B_total":"110","C_case":"$case_capacity.C_case"},'
            '"target":"N_cases"}'
            ']}'
        )


def test_solver_accepts_ceil_and_prefers_number_of_cases() -> None:
    spr = _gsm8k_word_spr(
        "bookcase",
        "If she has 110 books, how many bookcases does she need to hold all of them?",
        "number of bookcases",
        [{"symbol": "N_cases", "description": "number of bookcases needed", "domain": "integer"}],
    )
    emr = _unknown_word_emr("bookcase")
    classification = ProblemClassifier().classify(spr, emr)

    chain = Solver(
        planner=DeepSeekPlanner(client=FakeGSM8KBookcasePlanningClient()),
        use_planner=True,
    ).solve(emr, classification, spr=spr)
    verification = Verifier().verify(chain, spr=spr, emr=emr, classification_result=classification)

    assert "2" in (chain.final_answer or "")
    assert chain.metadata["partial_failure"] is False
    assert verification.valid is True


def _repair_scs(problem_id: str, final_answer: str, cas_steps: list[dict[str, str]]) -> SolutionChain:
    steps = [
        {
            "step_id": f"step_{index}",
            "step_index": index,
            "description": f"CAS step {index}",
            "math_expression": item["value"],
            "rule_name": "cas_evaluate",
            "verification_status": "unverified",
            "diagnostics": {
                "tool_call_id": item["id"],
                "operation": "evaluate",
                "input": {
                    "expression": item.get("expression", item["value"]),
                    "variables": [],
                    "substitutions": {},
                    "target": item["target"],
                },
                "output": {"value": item["value"], "formatted_solution": item["value"]},
                "success": True,
                "backend_confirmed": True,
            },
        }
        for index, item in enumerate(cas_steps)
    ]
    steps.append(
        {
            "step_id": f"step_{len(steps)}",
            "step_index": len(steps),
            "description": "Final answer.",
            "math_expression": final_answer,
            "rule_name": "final_answer",
            "verification_status": "unverified",
            "diagnostics": {"input": {"final_answer_items": []}, "output": {"formatted_solution": final_answer}},
        }
    )
    first = cas_steps[0]
    return SolutionChain.model_validate(
        {
            "problem_id": problem_id,
            "chain_id": "chain_main",
            "steps": steps,
            "final_answer": final_answer,
            "confidence": 0.85,
            "metadata": {
                "solver_route": "algebra_word_problem_solver",
                "final_answer_items": [
                    {
                        "target": first["target"],
                        "value": first["value"],
                        "source_tool_call_id": first["id"],
                        "backend_operation": "evaluate",
                        "backend_confirmed": True,
                    }
                ],
            },
        }
    )


def test_final_answer_repair_reselects_grams_over_remaining_calories() -> None:
    spr = _gsm8k_word_spr(
        "chips",
        (
            "A bag has 250 calories per serving. A 300g bag has 5 servings. "
            "How many grams can you eat if you have 200 calories remaining?"
        ),
        "grams can eat",
    )
    chain = _repair_scs(
        "chips",
        "remaining_cal = 200",
        [
            {"id": "remaining_cal", "target": "remaining_cal", "value": "200"},
            {"id": "grams", "target": "grams", "value": "48"},
        ],
    )

    repaired, report = FinalAnswerRepair().repair(chain, spr=spr)

    assert report["repair_performed"] is True
    assert repaired.final_answer == "grams = 48"
    assert repaired.metadata["final_answer_items"][0]["source_tool_call_id"] == "grams"


def test_final_answer_repair_converts_cents_to_dollars() -> None:
    spr = _gsm8k_word_spr(
        "postage",
        (
            "He sends 700 small coupons and twice as many big coupons. Small coupons "
            "cost 5 cents and big coupons cost 15 cents. How much does he spend on postage total?"
        ),
        "total postage cost",
    )
    chain = _repair_scs(
        "postage",
        "total_cost = 24500",
        [{"id": "total_cost", "target": "total_cost", "value": "24500"}],
    )

    repaired, report = FinalAnswerRepair().repair(chain, spr=spr)

    assert report["repair_performed"] is True
    assert report["repair_type"] == "unit_conversion_cents_to_dollars"
    assert repaired.final_answer == "total_cost_dollars = 245"
    assert repaired.metadata["final_answer_items"][0]["repair_transform"] == "cents_to_dollars"


def test_final_answer_repair_reselects_speed_over_total_time() -> None:
    spr = _gsm8k_word_spr(
        "running",
        "John runs 60 miles a week in 6 total hours. How fast does he run?",
        "speed",
    )
    chain = _repair_scs(
        "running",
        "total_time = 6",
        [
            {"id": "total_time", "target": "total_time", "value": "6"},
            {"id": "speed", "target": "speed", "value": "10"},
        ],
    )

    repaired, report = FinalAnswerRepair().repair(chain, spr=spr)

    assert report["repair_performed"] is True
    assert repaired.final_answer == "speed = 10"


def test_final_answer_repair_reselects_remaining_after_lunch() -> None:
    spr = _gsm8k_word_spr(
        "cars",
        (
            "Hunter counted 50 cars and then 20 more. At lunch, half the cars had gone. "
            "What's the total number of cars in the parking lot at lunch?"
        ),
        "cars at lunch",
    )
    chain = _repair_scs(
        "cars",
        "total_before_lunch = 70",
        [
            {"id": "total_before", "target": "total_before_lunch", "value": "70"},
            {"id": "lunch_count", "target": "lunch_count", "value": "35"},
        ],
    )

    repaired, report = FinalAnswerRepair().repair(chain, spr=spr)

    assert report["repair_performed"] is True
    assert repaired.final_answer == "lunch_count = 35"


def test_final_answer_repair_reselects_percentage_over_count() -> None:
    spr = _gsm8k_word_spr(
        "dance_repair",
        "What percentage of the entire students enrolled in hip-hop dance?",
        "hip-hop percentage",
    )
    chain = _repair_scs(
        "dance_repair",
        "contemporary = 4",
        [
            {"id": "contemporary", "target": "contemporary", "value": "4"},
            {"id": "percentage", "target": "percentage_hiphop", "value": "60"},
        ],
    )

    repaired, report = FinalAnswerRepair().repair(chain, spr=spr)

    assert report["repair_performed"] is True
    assert repaired.final_answer == "percentage_hiphop = 60"


def test_final_answer_repair_does_not_change_clear_correct_answer() -> None:
    spr = _gsm8k_word_spr(
        "age_repair",
        "Ruby is older than Sam. How old is Sam now?",
        "Sam's current age",
    )
    chain = _repair_scs(
        "age_repair",
        "S = 6",
        [{"id": "sam_age", "target": "S", "value": "6"}],
    )

    repaired, report = FinalAnswerRepair().repair(chain, spr=spr)

    assert report["repair_performed"] is False
    assert repaired.final_answer == "S = 6"


def test_final_answer_repair_aligns_metadata_without_degrading_existing_answer() -> None:
    spr = _gsm8k_word_spr(
        "shoe_cost_alignment",
        "John buys 2 pairs of shoes for each of his 3 children. They cost $60 each. How much did he pay?",
        "total amount paid",
    )
    chain = _repair_scs(
        "shoe_cost_alignment",
        "答案是 360",
        [
            {"id": "compute_pairs", "target": "total_pairs", "value": "6"},
            {"id": "compute_total", "target": "total_cost", "value": "360"},
        ],
    )
    chain.metadata["final_answer_items"] = [
        {
            "target": "total_pairs",
            "value": "6",
            "source_tool_call_id": "compute_pairs",
            "backend_operation": "evaluate",
            "backend_confirmed": True,
        }
    ]

    repaired, report = FinalAnswerRepair().repair(chain, spr=spr)

    assert report["repair_performed"] is True
    assert report["repair_type"] == "final_answer_item_alignment"
    assert repaired.final_answer == "total_cost = 360"
    assert repaired.metadata["final_answer_items"][0]["source_tool_call_id"] == "compute_total"
    assert repaired.metadata["final_answer_items"][0]["value"] == "360"


def test_final_answer_repair_reselects_last_answer_like_cas_result_for_win_difference() -> None:
    spr = _gsm8k_word_spr(
        "relay_win_difference",
        (
            "One relay team finishes in 220 seconds and the other finishes in 222 seconds. "
            "How many seconds will the faster team win by?"
        ),
        "seconds faster",
    )
    chain = _repair_scs(
        "relay_win_difference",
        "t_A = 220",
        [
            {"id": "team_a_total", "target": "t_A", "value": "220"},
            {"id": "team_b_total", "target": "t_B", "value": "222"},
            {"id": "win_difference", "target": "win_difference", "value": "2"},
        ],
    )
    chain.metadata["final_answer_items"] = [
        {
            "target": "t_A",
            "value": "220",
            "source_tool_call_id": "team_a_total",
            "backend_operation": "evaluate",
            "backend_confirmed": True,
        }
    ]

    repaired, report = FinalAnswerRepair().repair(chain, spr=spr)

    assert report["repair_performed"] is True
    assert report["repair_type"] == "last_answer_like_cas_selection"
    assert repaired.final_answer == "win_difference = 2"
    assert repaired.metadata["final_answer_items"][0]["source_tool_call_id"] == "win_difference"


def test_final_answer_repair_protects_plain_numeric_savings_answer() -> None:
    spr = _gsm8k_word_spr(
        "bus_repair",
        "Each regular bus trip costs $2.20. How much would she save with a weekly pass?",
        "savings",
    )
    chain = _repair_scs(
        "bus_repair",
        "答案是 2.00000000000000",
        [
            {"id": "regular_cost", "target": "regular_cost", "value": "22.0000000000000"},
            {"id": "savings", "target": "savings", "value": "2.00000000000000"},
        ],
    )
    chain.metadata["final_answer_items"] = [
        {
            "target": "savings",
            "value": "2.00000000000000",
            "source_tool_call_id": "savings",
            "backend_operation": "evaluate",
            "backend_confirmed": True,
        }
    ]

    repaired, report = FinalAnswerRepair().repair(chain, spr=spr)

    assert report["repair_performed"] is False
    assert report["decision"] in {"skip", "protect"}
    assert repaired.final_answer == "答案是 2.00000000000000"


def test_final_answer_repair_protects_leftover_answer_from_bill_slices() -> None:
    spr = _gsm8k_word_spr(
        "pizza_repair",
        "Bill eats 4 slices, Mark eats 3 slices, and Jenny eats 2. How many slices are left?",
        "slices left",
    )
    chain = _repair_scs(
        "pizza_repair",
        "S_remaining = 3",
        [
            {"id": "slices_bill", "target": "slices_bill", "value": "4"},
            {"id": "remaining", "target": "S_remaining", "value": "3"},
        ],
    )
    chain.metadata["final_answer_items"] = [
        {
            "target": "S_remaining",
            "value": "3",
            "source_tool_call_id": "remaining",
            "backend_operation": "evaluate",
            "backend_confirmed": True,
        }
    ]

    repaired, report = FinalAnswerRepair().repair(chain, spr=spr)

    assert report["repair_performed"] is False
    assert report["decision"] in {"skip", "protect"}
    assert repaired.final_answer == "S_remaining = 3"


def test_final_answer_repair_protects_left_answer_from_after_boats_intermediate() -> None:
    spr = _gsm8k_word_spr(
        "cats_repair",
        "Some cats were carried away by boats and some ran away. How many cats were left?",
        "cats left",
    )
    chain = _repair_scs(
        "cats_repair",
        "C_left = 12",
        [
            {"id": "remaining_after_boats", "target": "remaining_after_boats", "value": "30"},
            {"id": "cats_left", "target": "C_left", "value": "12"},
        ],
    )
    chain.metadata["final_answer_items"] = [
        {
            "target": "C_left",
            "value": "12",
            "source_tool_call_id": "cats_left",
            "backend_operation": "evaluate",
            "backend_confirmed": True,
        }
    ]

    repaired, report = FinalAnswerRepair().repair(chain, spr=spr)

    assert report["repair_performed"] is False
    assert report["decision"] in {"skip", "protect"}
    assert repaired.final_answer == "C_left = 12"


def test_final_answer_repair_protects_plain_numeric_savings_from_cost() -> None:
    spr = _gsm8k_word_spr(
        "dasha_repair",
        "Dasha paid $36. How much did she save?",
        "savings",
    )
    chain = _repair_scs(
        "dasha_repair",
        "答案是 6",
        [
            {"id": "dasha_cost", "target": "dasha_cost", "value": "36"},
            {"id": "savings", "target": "savings", "value": "6"},
        ],
    )
    chain.metadata["final_answer_items"] = [
        {
            "target": "savings",
            "value": "6",
            "source_tool_call_id": "savings",
            "backend_operation": "evaluate",
            "backend_confirmed": True,
        }
    ]

    repaired, report = FinalAnswerRepair().repair(chain, spr=spr)

    assert report["repair_performed"] is False
    assert report["decision"] in {"skip", "protect"}
    assert repaired.final_answer == "答案是 6"


def test_final_answer_repair_protects_difference_from_net_pay_intermediate() -> None:
    spr = _gsm8k_word_spr(
        "net_pay_repair",
        "Alex and Bea have different net pay after taxes. What is the difference?",
        "difference",
    )
    chain = _repair_scs(
        "net_pay_repair",
        "difference = 8400",
        [
            {"id": "netA", "target": "NetPay_A", "value": "24000"},
            {"id": "diff", "target": "difference", "value": "8400"},
        ],
    )
    chain.metadata["final_answer_items"] = [
        {
            "target": "difference",
            "value": "8400",
            "source_tool_call_id": "diff",
            "backend_operation": "evaluate",
            "backend_confirmed": True,
        }
    ]

    repaired, report = FinalAnswerRepair().repair(chain, spr=spr)

    assert report["repair_performed"] is False
    assert report["decision"] in {"skip", "protect"}
    assert repaired.final_answer == "difference = 8400"


def test_final_answer_repair_reports_original_answer_protection() -> None:
    spr = _gsm8k_word_spr(
        "plain_numeric_protection",
        "A clerk handled many reports. How many reports were finally approved?",
        "approved reports",
    )
    chain = _repair_scs(
        "plain_numeric_protection",
        "答案是 12",
        [
            {"id": "answer_result", "target": "answer_result", "value": "12"},
            {"id": "total_reports", "target": "total_reports", "value": "60"},
        ],
    )
    chain.metadata["final_answer_items"] = [
        {
            "target": "answer_result",
            "value": "12",
            "source_tool_call_id": "answer_result",
            "backend_operation": "evaluate",
            "backend_confirmed": True,
        }
    ]

    repaired, report = FinalAnswerRepair().repair(chain, spr=spr)

    assert report["repair_performed"] is False
    assert report["decision"] == "protect"
    assert report["original_answer_protected"] is True
    assert repaired.final_answer == "答案是 12"


def test_final_answer_repair_v12_keeps_savings_over_total_trips() -> None:
    spr = _gsm8k_word_spr(
        "bus_savings_v12",
        "How much would she save by buying a weekly bus pass for $20?",
        "savings",
    )
    chain = _repair_scs(
        "bus_savings_v12",
        "savings = 2",
        [
            {"id": "trips", "target": "total_trips", "value": "10"},
            {"id": "regular_cost", "target": "total_regular_cost", "value": "22"},
            {"id": "savings", "target": "savings", "value": "2"},
        ],
    )
    chain.metadata["final_answer_items"] = [
        {
            "target": "savings",
            "value": "2",
            "source_tool_call_id": "savings",
            "backend_operation": "evaluate",
            "backend_confirmed": True,
        }
    ]

    repaired, report = FinalAnswerRepair().repair(chain, spr=spr)

    assert report["repair_performed"] is False
    assert repaired.final_answer == "savings = 2"


def test_final_answer_repair_v12_reselects_attending_boys() -> None:
    spr = _gsm8k_word_spr(
        "attending_boys_v12",
        "How many fourth grade boys were at Small Tree School on Friday?",
        "boys present on Friday",
    )
    chain = _repair_scs(
        "attending_boys_v12",
        "B = 53",
        [
            {"id": "total_boys", "target": "total_boys", "value": "53"},
            {"id": "attending_boys", "target": "attending_boys", "value": "49"},
        ],
    )

    repaired, report = FinalAnswerRepair().repair(chain, spr=spr)

    assert report["repair_performed"] is True
    assert repaired.final_answer == "attending_boys = 49"


def test_final_answer_repair_v12_reselects_box_count() -> None:
    spr = _gsm8k_word_spr(
        "cookies_box_v12",
        "How many cookies are there in the box?",
        "cookies in the box",
    )
    chain = _repair_scs(
        "cookies_box_v12",
        "G = 90",
        [
            {"id": "total", "target": "total_baked", "value": "90"},
            {"id": "box", "target": "box_count", "value": "80"},
        ],
    )

    repaired, report = FinalAnswerRepair().repair(chain, spr=spr)

    assert report["repair_performed"] is True
    assert repaired.final_answer == "box_count = 80"


def test_final_answer_repair_v12_reselects_specific_horse() -> None:
    spr = _gsm8k_word_spr(
        "horse12_v12",
        "How many people think that horse #12 will win?",
        "horse 12 voters",
    )
    chain = _repair_scs(
        "horse12_v12",
        "n_remaining = 40",
        [
            {"id": "remaining", "target": "n_remaining", "value": "40"},
            {"id": "horse7", "target": "n_horse7", "value": "24"},
            {"id": "horse12", "target": "n_horse12", "value": "16"},
        ],
    )

    repaired, report = FinalAnswerRepair().repair(chain, spr=spr)

    assert report["repair_performed"] is True
    assert repaired.final_answer == "n_horse12 = 16"


def test_final_answer_repair_v12_reselects_nap_time() -> None:
    spr = _gsm8k_word_spr(
        "nap_v12",
        "How many minutes does John have to nap?",
        "nap time",
    )
    chain = _repair_scs(
        "nap_v12",
        "T_total = 180",
        [
            {"id": "homework", "target": "total_homework", "value": "80"},
            {"id": "available", "target": "total_available", "value": "180"},
            {"id": "nap", "target": "nap_time", "value": "100"},
        ],
    )

    repaired, report = FinalAnswerRepair().repair(chain, spr=spr)

    assert report["repair_performed"] is True
    assert repaired.final_answer == "nap_time = 100"


def test_final_answer_repair_v12_allows_difference_with_many_targets() -> None:
    spr = _gsm8k_word_spr(
        "difference_v12",
        "How much more money will Billy earn than Sally, in dollars?",
        "difference",
    )
    spr.questions[0].target_ids = ["t1", "t2", "t3"]
    chain = _repair_scs(
        "difference_v12",
        "R_B_final = 11.5",
        [
            {"id": "billy_rate", "target": "R_B_final", "value": "11.5"},
            {"id": "sally_rate", "target": "R_S_final", "value": "10.5"},
            {"id": "difference", "target": "difference", "value": "20"},
        ],
    )

    repaired, report = FinalAnswerRepair().repair(chain, spr=spr)

    assert report["repair_performed"] is True
    assert repaired.final_answer == "difference = 20"


def test_final_answer_repair_v12_reselects_net_pay_difference() -> None:
    spr = _gsm8k_word_spr(
        "net_pay_difference_v12",
        "How much more money will Nick make at the job with a higher net pay rate, compared to the other job?",
        "difference",
    )
    chain = _repair_scs(
        "net_pay_difference_v12",
        "I_A_net = 24000",
        [
            {"id": "net_a", "target": "I_A_net", "value": "24000"},
            {"id": "net_b", "target": "I_B_net", "value": "32400"},
            {"id": "diff", "target": "diff", "value": "8400"},
        ],
    )

    repaired, report = FinalAnswerRepair().repair(chain, spr=spr)

    assert report["repair_performed"] is True
    assert repaired.final_answer == "diff = 8400"


def test_final_answer_repair_v12_aligns_final_answer_item() -> None:
    spr = _gsm8k_word_spr(
        "lunch_alignment_v12",
        "What's the total number of cars he counted during lunch break?",
        "cars at lunch",
    )
    chain = _repair_scs(
        "lunch_alignment_v12",
        "C_added = 70",
        [
            {"id": "added", "target": "total_after_first_break", "value": "70"},
            {"id": "lunch", "target": "cars_at_lunch", "value": "35"},
        ],
    )
    chain.metadata["final_answer_items"] = [
        {
            "target": "cars_at_lunch",
            "value": "35",
            "source_tool_call_id": "lunch",
            "backend_operation": "evaluate",
            "backend_confirmed": True,
        }
    ]

    repaired, report = FinalAnswerRepair().repair(chain, spr=spr)

    assert report["repair_performed"] is True
    assert report["repair_type"] == "final_answer_item_alignment"
    assert repaired.final_answer == "cars_at_lunch = 35"


def test_final_answer_repair_v12_reselects_individual_units() -> None:
    spr = _gsm8k_word_spr(
        "brownies_v12",
        "How many individual brownies did Greta have left over from the entire day?",
        "individual brownies left",
    )
    chain = _repair_scs(
        "brownies_v12",
        "total_dozens = 11/2",
        [
            {"id": "total_dozens", "target": "total_dozens", "value": "11/2"},
            {"id": "leftover_dozens", "target": "leftover_dozens", "value": "4"},
            {"id": "individual_leftover", "target": "individual_leftover", "value": "48"},
        ],
    )

    repaired, report = FinalAnswerRepair().repair(chain, spr=spr)

    assert report["repair_performed"] is True
    assert repaired.final_answer == "individual_leftover = 48"


def test_final_answer_repair_v12_uses_target_value_from_equation_system_output() -> None:
    spr = _gsm8k_word_spr(
        "brownies_equation_system_v12",
        "How many individual brownies did Greta have left over from the entire day?",
        "individual brownies left",
    )
    chain = _repair_scs(
        "brownies_equation_system_v12",
        "B_left = 48",
        [{"id": "solve_B_left", "target": "B_left", "value": "48"}],
    )
    diagnostics = chain.steps[0].diagnostics
    diagnostics["operation"] = "solve_equation_system"
    diagnostics["input"]["target"] = "B_left"
    diagnostics["output"] = {
        "raw_solution": [
            {
                "B_eaten": "18",
                "B_home": "48",
                "B_left": "48",
                "B_made": "12",
                "B_office": "6",
                "B_total": "66",
            }
        ],
        "formatted_solution": (
            "B_eaten = 18, B_home = 48, B_left = 48, "
            "B_made = 12, B_office = 6, B_total = 66"
        ),
    }

    repaired, report = FinalAnswerRepair().repair(chain, spr=spr)

    assert report["repair_performed"] is False
    assert repaired.final_answer == "B_left = 48"


def test_final_answer_repair_v12_reselects_total_replacement_cost() -> None:
    spr = _gsm8k_word_spr(
        "movie_replacement_cost_v12",
        "How much does replacing the movies cost?",
        "Calculate the number of series movies.",
    )
    chain = _repair_scs(
        "movie_replacement_cost_v12",
        "N_{series} = 200",
        [
            {"id": "step1", "target": "N_{series}", "value": "200"},
            {"id": "step2", "target": "series_cost", "value": "1200"},
            {"id": "step3", "target": "N_remaining", "value": "400"},
            {"id": "step4", "target": "N_older", "value": "160"},
            {"id": "step5", "target": "older_cost", "value": "800"},
            {"id": "step6", "target": "N_normal", "value": "240"},
            {"id": "step7", "target": "normal_cost", "value": "2400"},
            {"id": "step8", "target": "total_cost", "value": "4400"},
        ],
    )

    repaired, report = FinalAnswerRepair().repair(chain, spr=spr)

    assert report["repair_performed"] is True
    assert report["answer_intent"] == "money_total"
    assert repaired.final_answer == "total_cost = 4400"


def test_final_answer_repair_v12_aligns_metadata_to_correct_text_answer() -> None:
    spr = _gsm8k_word_spr(
        "movie_replacement_text_alignment_v12",
        "How much does replacing the movies cost?",
        "Calculate the total cost of replacing all movies.",
    )
    chain = _repair_scs(
        "movie_replacement_text_alignment_v12",
        "答案是 4400",
        [
            {"id": "step1", "target": "N_series", "value": "200"},
            {"id": "step2", "target": "N_remaining", "value": "400"},
            {"id": "step3", "target": "N_older", "value": "160"},
            {"id": "step4", "target": "N_normal", "value": "240"},
            {"id": "step5", "target": "cost_series", "value": "1200"},
            {"id": "step6", "target": "cost_older", "value": "800"},
            {"id": "step7", "target": "cost_normal", "value": "2400"},
            {"id": "step8", "target": "total_cost", "value": "4400"},
        ],
    )
    chain.metadata["final_answer_items"] = [
        {
            "target": "N_normal",
            "value": "240",
            "source_tool_call_id": "step4",
            "backend_operation": "evaluate",
            "backend_confirmed": True,
        }
    ]

    repaired, report = FinalAnswerRepair().repair(chain, spr=spr)

    assert report["repair_performed"] is True
    assert report["repair_type"] == "final_answer_item_alignment"
    assert repaired.final_answer == "total_cost = 4400"
    assert repaired.metadata["final_answer_items"][0]["target"] == "total_cost"
    assert repaired.metadata["final_answer_items"][0]["source_tool_call_id"] == "step8"


def test_final_answer_repair_v12_does_not_replace_requested_local_cost_with_total() -> None:
    spr = _gsm8k_word_spr(
        "normal_movie_cost_v12",
        "How much do the normal movies cost?",
        "normal movie cost",
    )
    chain = _repair_scs(
        "normal_movie_cost_v12",
        "normal_cost = 2400",
        [
            {"id": "normal", "target": "normal_cost", "value": "2400"},
            {"id": "total", "target": "total_cost", "value": "4400"},
        ],
    )
    chain.metadata["final_answer_items"] = [
        {
            "target": "normal_cost",
            "value": "2400",
            "source_tool_call_id": "normal",
            "backend_operation": "evaluate",
            "backend_confirmed": True,
        }
    ]

    repaired, report = FinalAnswerRepair().repair(chain, spr=spr)

    assert report["repair_performed"] is False
    assert repaired.final_answer == "normal_cost = 2400"


class FakeGSM8KNamedEntityTotalPlanningClient:
    model = "fake-deepseek-v4-flash"

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        assert "tool_calls" in system_prompt + user_prompt
        return (
            '{"tool_calls":['
            '{"id":"calc_T_Rose","operation":"evaluate","variables":["O_Rose","P_Rose"],'
            '"expressions":[{"name":"T_Rose","expression":"O_Rose + P_Rose"}],'
            '"substitutions":{"O_Rose":"12","P_Rose":"4"},"target":"T_Rose"},'
            '{"id":"solve_T_Sophia","operation":"solve_for","variables":["T_Rose","T_Sophia"],'
            '"equations":[{"lhs":"T_Rose","rhs":"4 * T_Sophia"}],'
            '"substitutions":{"T_Rose":"$calc_T_Rose.T_Rose"},"target":"T_Sophia"}'
            ']}'
        )


def test_solver_prefers_named_entity_requested_in_question_over_other_total() -> None:
    spr = _gsm8k_word_spr(
        "sophia_rose",
        (
            "Sophia and Rose went together to the market to buy onions and potatoes. "
            "Rose bought 4 times the number of onions and potatoes Sophia bought. "
            "If Rose bought 12 onions and 4 potatoes, how many onions and potatoes "
            "in total did Sophia buy at the market?"
        ),
        "Total number of onions and potatoes bought by Sophia",
        [
            {"symbol": "T_{Rose}", "description": "Total items bought by Rose", "domain": "integer"},
            {"symbol": "T_{Sophia}", "description": "Total items bought by Sophia", "domain": "integer"},
        ],
    )
    emr = _unknown_word_emr("sophia_rose")
    classification = ProblemClassifier().classify(spr, emr)

    chain = Solver(
        planner=DeepSeekPlanner(client=FakeGSM8KNamedEntityTotalPlanningClient()),
        use_planner=True,
    ).solve(emr, classification, spr=spr)

    assert chain.final_answer == "T_Sophia = 4"
    assert chain.metadata["final_answer_items"] == [
        {
            "target": "T_{Sophia}",
            "value": "4",
            "source_tool_call_id": "solve_T_Sophia",
            "backend_operation": "solve_for",
            "backend_confirmed": True,
        }
    ]


def test_solver_keeps_last_repeated_single_answer_target() -> None:
    spr_data = math7_spr().model_dump(mode="json")
    spr_data["questions"] = [
        {
            "id": "q1",
            "text": "What is the distance, in miles, between the bird's northern and southern homes?",
            "question_type": "compute",
            "target_ids": ["t1"],
        }
    ]
    spr_data["targets"] = [{"id": "t1", "text": "distance between homes", "target_type": "compute"}]
    spr = SPR.model_validate(spr_data)
    items = [
        {"target": "d_north_south_1", "value": "300"},
        {"target": "d_north_south_1", "value": "36"},
        {"target": "d_north_south_1", "value": "110"},
        {"target": "d_north_south_1", "value": "374"},
    ]

    selected = AlgebraSolver._select_final_answer_items(items, spr)

    assert selected == [{"target": "d_north_south_1", "value": "374"}]


def test_solver_repairs_middle_school_mixed_number_from_spr_text() -> None:
    spr = SPR.model_validate(
        {
            "schema_version": "SPR-1.0",
            "source_image": {"image_path": "data/experiments/math/images/mixed_number.png"},
            "problem_text": "Solve for w and express as a common fraction: (1 1/6) / w = 42/3.",
            "problem_stem": "Solve for w.",
            "formulas": [{"id": "f1", "raw_text": "(1 1/6) / w = 42/3", "role": "condition"}],
            "questions": [{"id": "q1", "text": "What is w?", "question_type": "compute", "target_ids": ["t1"]}],
            "variables": [{"symbol": "w", "domain": "real"}],
            "problem_type": "algebra",
            "knowledge_units": ["fraction_rational", "equation"],
            "targets": [{"id": "t1", "text": "w", "target_type": "compute"}],
            "confidence": {"overall": 0.95},
        }
    )
    emr = EMR.model_validate(
        {
            "problem_id": "math_middle_mixed_number",
            "source_problem_type": "algebra",
            "representation_type": "equation_system",
            "variables": [{"id": "var_w", "symbol": "w", "domain": "real"}],
            "equations": [{"id": "eq1", "lhs_sympy": "11/6/w", "rhs_sympy": "42/3"}],
            "goals": [{"id": "g1", "goal_type": "compute", "target": "w", "target_variable_ids": ["var_w"]}],
        }
    )
    classification = algebra_solver_classification_result(problem_id="math_middle_mixed_number")

    chain = Solver(use_planner=False).solve(emr, classification, spr=spr)

    assert chain.final_answer == "w = 1/12"
    assert chain.metadata["final_answer_items"][0]["value"] == "1/12"


def test_solver_derives_product_of_two_quadratic_solutions() -> None:
    spr = SPR.model_validate(
        {
            "schema_version": "SPR-1.0",
            "source_image": {"image_path": "data/experiments/math/images/quadratic_roots.png"},
            "problem_text": "If x^2 - x - 6 = 0, what is the product of the two solutions?",
            "problem_stem": "Find the product of the two solutions.",
            "questions": [
                {
                    "id": "q1",
                    "text": "What is the product of the two solutions?",
                    "question_type": "compute",
                    "target_ids": ["t1"],
                }
            ],
            "variables": [{"symbol": "x", "domain": "real"}],
            "problem_type": "algebra",
            "knowledge_units": ["quadratic_equation"],
            "targets": [{"id": "t1", "text": "product of the two solutions", "target_type": "compute"}],
            "confidence": {"overall": 0.95},
        }
    )
    emr = EMR.model_validate(
        {
            "problem_id": "math_middle_quadratic_product",
            "source_problem_type": "algebra",
            "representation_type": "equation_system",
            "variables": [{"id": "var_x", "symbol": "x", "domain": "real"}],
            "equations": [{"id": "eq1", "lhs_sympy": "x**2 - x - 6", "rhs_sympy": "0"}],
            "goals": [{"id": "g1", "goal_type": "compute", "target": "product_of_solutions"}],
        }
    )
    classification = algebra_solver_classification_result(problem_id="math_middle_quadratic_product")

    chain = Solver(use_planner=False).solve(emr, classification, spr=spr)

    assert chain.final_answer == "product_of_solutions = -6"
    assert any(
        item["target"] == "product_of_solutions" and item["value"] == "-6"
        for item in chain.metadata["final_answer_items"]
    )


def test_solver_completes_emr_target_expression_after_ratio_system() -> None:
    spr = SPR.model_validate(
        {
            "schema_version": "SPR-1.0",
            "source_image": {"image_path": "data/experiments/math/images/ratio_chain.png"},
            "problem_text": "Given a/b = 4, b/c = 1/3, and c/d = 6, find d/a.",
            "problem_stem": "Find d/a from the ratio chain.",
            "questions": [{"id": "q1", "text": "What is d/a?", "question_type": "compute", "target_ids": ["t1"]}],
            "variables": [
                {"symbol": "a", "domain": "real"},
                {"symbol": "b", "domain": "real"},
                {"symbol": "c", "domain": "real"},
                {"symbol": "d", "domain": "real"},
            ],
            "problem_type": "algebra",
            "knowledge_units": ["ratio", "equation_system"],
            "targets": [{"id": "t1", "text": "d/a", "target_type": "compute"}],
            "confidence": {"overall": 0.95},
        }
    )
    emr = EMR.model_validate(
        {
            "problem_id": "math_middle_ratio_target",
            "source_problem_type": "algebra",
            "representation_type": "unknown",
            "variables": [
                {"id": "var_a", "symbol": "a", "domain": "real"},
                {"id": "var_b", "symbol": "b", "domain": "real"},
                {"id": "var_c", "symbol": "c", "domain": "real"},
                {"id": "var_d", "symbol": "d", "domain": "real"},
            ],
            "expressions": [{"id": "expr_target", "sympy": "d/a", "raw_text": "d/a"}],
            "goals": [
                {
                    "id": "g1",
                    "goal_type": "compute",
                    "target": "d_over_a",
                    "target_expression_id": "expr_target",
                    "description": "find d/a",
                }
            ],
        }
    )
    classification = algebra_solver_classification_result(problem_id="math_middle_ratio_target")
    planner = DeepSeekPlanner(client=FakeRatioTargetExpressionPlanningClient())

    chain = Solver(planner=planner, use_planner=True).solve(emr, classification, spr=spr)

    assert "1/8" in (chain.final_answer or "")
    assert any(
        item["backend_operation"] == "evaluate" and item["value"] == "1/8"
        for item in chain.metadata["final_answer_items"]
    )


def test_solver_falls_back_to_emr_target_when_followup_tool_call_is_invalid() -> None:
    spr = SPR.model_validate(
        {
            "schema_version": "SPR-1.0",
            "source_image": {"image_path": "data/experiments/math/images/ratio_chain.png"},
            "problem_text": "Given a/b = 4, b/c = 1/3, and c/d = 6, find d/a.",
            "problem_stem": "Find d/a from the ratio chain.",
            "questions": [{"id": "q1", "text": "What is d/a?", "question_type": "compute", "target_ids": ["t1"]}],
            "variables": [
                {"symbol": "a", "domain": "real"},
                {"symbol": "b", "domain": "real"},
                {"symbol": "c", "domain": "real"},
                {"symbol": "d", "domain": "real"},
            ],
            "problem_type": "algebra",
            "knowledge_units": ["ratio", "equation_system"],
            "targets": [{"id": "t1", "text": "d/a", "target_type": "compute"}],
            "confidence": {"overall": 0.95},
        }
    )
    emr = EMR.model_validate(
        {
            "problem_id": "math_middle_ratio_target_invalid_followup",
            "source_problem_type": "algebra",
            "representation_type": "unknown",
            "variables": [
                {"id": "var_a", "symbol": "a", "domain": "real"},
                {"id": "var_b", "symbol": "b", "domain": "real"},
                {"id": "var_c", "symbol": "c", "domain": "real"},
                {"id": "var_d", "symbol": "d", "domain": "real"},
            ],
            "expressions": [{"id": "expr_target", "sympy": "d/a", "raw_text": "d/a"}],
            "goals": [
                {
                    "id": "g1",
                    "goal_type": "compute",
                    "target": "d_over_a",
                    "target_expression_id": "expr_target",
                    "description": "find d/a",
                }
            ],
        }
    )
    classification = algebra_solver_classification_result(problem_id="math_middle_ratio_target_invalid_followup")
    planner = DeepSeekPlanner(client=FakeRatioTargetExpressionWithInvalidFollowupPlanningClient())

    chain = Solver(planner=planner, use_planner=True).solve(emr, classification, spr=spr)

    assert chain.final_answer == "d_over_a = 1/8"
    assert any(
        item["backend_operation"] == "evaluate" and item["value"] == "1/8"
        for item in chain.metadata["final_answer_items"]
    )


def test_solver_allows_planner_local_symbols_for_coordinate_midpoint_task() -> None:
    spr = SPR.model_validate(
        {
            "schema_version": "SPR-1.0",
            "source_image": {"image_path": "data/experiments/math/images/midpoint.png"},
            "problem_text": (
                "Point M(4, 4) is the midpoint of AB. If point A has coordinates (8, 4), "
                "what is the sum of the coordinates of point B?"
            ),
            "problem_stem": "Find the sum of the coordinates of point B.",
            "questions": [
                {
                    "id": "q1",
                    "text": "What is the sum of the coordinates of point B?",
                    "question_type": "compute",
                    "target_ids": ["t1"],
                }
            ],
            "variables": [
                {"symbol": "M", "description": "Midpoint of segment AB", "domain": "coordinates"},
                {"symbol": "A", "description": "One endpoint of segment AB", "domain": "coordinates"},
                {"symbol": "B", "description": "The other endpoint of segment AB", "domain": "coordinates"},
            ],
            "problem_type": "geometry",
            "knowledge_units": ["midpoint_formula", "coordinate_geometry"],
            "targets": [{"id": "t1", "text": "sum of the coordinates of point B", "target_type": "compute"}],
            "confidence": {"overall": 0.95},
        }
    )
    emr = EMR.model_validate(
        {
            "problem_id": "math_middle_coordinate_midpoint",
            "source_problem_type": "geometry",
            "representation_type": "unknown",
        }
    )
    classification = ProblemClassifier().classify(spr, emr)
    planner = DeepSeekPlanner(client=FakeCoordinateMidpointPlanningClient())

    chain = Solver(planner=planner, use_planner=True).solve(emr, classification, spr=spr)

    assert chain.final_answer and "4" in chain.final_answer
    assert any(
        item["backend_operation"] == "evaluate" and item["value"] == "4"
        for item in chain.metadata["final_answer_items"]
    )


class FakeExponentBranchPlanningClient:
    model = "fake-qwen3.6-plus"

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        assert "tool_calls" in system_prompt + user_prompt
        return (
            '{"tool_calls":['
            '{"id":"solve_b","operation":"solve_for","variables":["b"],'
            '"equations":[{"lhs":"125**b","rhs":"5"}],"target":"b"},'
            '{"id":"compute_c","operation":"substitute","variables":["b"],'
            '"expressions":[{"name":"c","expression":"27**b"}],'
            '"substitutions":{"b":"$solve_b.b"},"target":"c"}'
            ']}'
        )


class FakeMidpointOnlyXPlanningClient:
    model = "fake-qwen3.6-plus"

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        assert "tool_calls" in system_prompt + user_prompt
        return (
            '{"tool_calls":[{"id":"sum_coords","operation":"evaluate","variables":[],'
            '"expressions":[{"name":"sum_coords","expression":"(8+2)/2"}],'
            '"target":"sum_coords","reason":"Incorrectly computes only the midpoint x-coordinate."}]}'
        )


class FakeWrongEquivalentUnitPlanningClient:
    model = "fake-qwen3.6-plus"

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        assert "tool_calls" in system_prompt + user_prompt
        return (
            '{"tool_calls":[{"id":"solve_for_ligs","operation":"evaluate","variables":[],'
            '"expressions":[{"name":"ligs","expression":"6400/63"}],'
            '"target":"ligs","reason":"Uses the inverse conversion direction."}]}'
        )


class FakeIrrelevantPlanningClient:
    model = "fake-qwen3.6-plus"

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        assert "tool_calls" in system_prompt + user_prompt
        return (
            '{"tool_calls":[{"id":"irrelevant","operation":"evaluate","variables":[],'
            '"expressions":[{"name":"wrong","expression":"999"}],"target":"wrong"}]}'
        )


def test_solver_filters_nonzero_candidate_in_direct_quadratic_solve() -> None:
    spr = SPR.model_validate(
        {
            "source_image": {"image_path": "math_middle_0100.png"},
            "problem_text": "If $x^2 - 2x = 0$ and $x \\neq 0$, what is the value of $x$?",
            "problem_type": "algebra",
            "variables": [{"symbol": "x", "description": "The variable to solve for", "domain": "real"}],
            "questions": [{"id": "q1", "text": "What is the value of x?", "target_ids": ["t1"]}],
            "targets": [{"id": "t1", "text": "x", "target_type": "compute"}],
            "confidence": {"overall": 1.0},
        }
    )
    emr = EMR.model_validate(
        {
            "problem_id": "math_middle_nonzero_root",
            "source_problem_type": "algebra",
            "representation_type": "equation_system",
            "variables": [{"id": "var_x", "symbol": "x", "domain": "real"}],
            "equations": [{"id": "eq1", "lhs_sympy": "x**2 - 2*x", "rhs_sympy": "0", "relation": "eq"}],
        }
    )

    chain = AlgebraSolver().solve(emr, algebra_solver_classification_result("math_middle_nonzero_root"), spr=spr)

    assert chain.final_answer == "x = 2"
    assert chain.metadata["final_answer_items"] == [
        {
            "target": "x",
            "value": "2",
            "source_step_id": "chain_main_step_2",
            "backend_operation": "solve_equation_system",
            "backend_confirmed": True,
        }
    ]


def test_solver_uses_principal_real_solution_for_exponent_substitution_context() -> None:
    spr = SPR.model_validate(
        {
            "source_image": {"image_path": "math_middle_0122.png"},
            "problem_text": "If $125^b = 5$ and $27^b = c$, what is the value of $c$?",
            "problem_type": "algebra",
            "variables": [
                {"symbol": "b", "description": "exponent variable", "domain": "real"},
                {"symbol": "c", "description": "value to be computed", "domain": "real"},
            ],
            "questions": [{"id": "q1", "text": "What is the value of c?", "target_ids": ["t1"]}],
            "targets": [{"id": "t1", "text": "c", "target_type": "compute"}],
            "confidence": {"overall": 1.0},
        }
    )
    emr = EMR.model_validate(
        {
            "problem_id": "math_middle_exponent_branch",
            "source_problem_type": "algebra",
            "representation_type": "unknown",
            "variables": [
                {"id": "var_b", "symbol": "b", "domain": "real"},
                {"id": "var_c", "symbol": "c", "domain": "real"},
            ],
            "equations": [
                {"id": "eq1", "lhs_sympy": "125**b", "rhs_sympy": "5", "relation": "eq"},
                {"id": "eq2", "lhs_sympy": "27**b", "rhs_sympy": "c", "relation": "eq"},
            ],
        }
    )
    classification = ProblemClassifier().classify(spr, emr)

    chain = Solver(
        planner=DeepSeekPlanner(client=FakeExponentBranchPlanningClient()),
        use_planner=True,
    ).solve(emr, classification, spr=spr)

    assert chain.final_answer == "答案是 3"
    assert chain.metadata["execution_context"]["solve_b"]["b"] == "1/3"
    assert chain.metadata["final_answer_items"] == [
        {
            "target": "c",
            "value": "3",
            "source_tool_call_id": "compute_c",
            "backend_operation": "substitute",
            "backend_confirmed": True,
        }
    ]


def test_solver_completes_midpoint_coordinate_sum_when_planner_only_computes_x_coordinate() -> None:
    spr = SPR.model_validate(
        {
            "source_image": {"image_path": "math_middle_0147.png"},
            "problem_text": "What is the sum of the coordinates of the midpoint of the segment with endpoints (8, 5) and (2, -1)?",
            "problem_type": "geometry",
            "variables": [{"symbol": "M", "description": "Midpoint", "domain": "point"}],
            "questions": [{"id": "q1", "text": "What is the sum of the coordinates of the midpoint?", "target_ids": ["t1"]}],
            "targets": [{"id": "t1", "text": "sum of midpoint coordinates", "target_type": "compute"}],
            "confidence": {"overall": 1.0},
        }
    )
    emr = EMR.model_validate(
        {
            "problem_id": "math_middle_midpoint_sum",
            "source_problem_type": "geometry",
            "representation_type": "unknown",
        }
    )
    classification = ProblemClassifier().classify(spr, emr)

    chain = Solver(
        planner=DeepSeekPlanner(client=FakeMidpointOnlyXPlanningClient()),
        use_planner=True,
    ).solve(emr, classification, spr=spr)

    assert chain.final_answer == "sum_coords = 7"
    assert chain.metadata["final_answer_items"] == [
        {
            "target": "sum_coords",
            "value": "7",
            "source_tool_call_id": "derived_midpoint_coordinate_sum",
            "backend_operation": "evaluate",
            "backend_confirmed": True,
        }
    ]


def test_solver_completes_equivalent_unit_chain_with_requested_conversion_direction() -> None:
    spr = SPR.model_validate(
        {
            "source_image": {"image_path": "math_middle_0188.png"},
            "problem_text": "On planet Larky, 7 ligs = 4 lags, and 9 lags = 20 lugs. How many ligs are equivalent to 80 lugs?",
            "problem_type": "word_problem",
            "variables": [
                {"symbol": "ligs", "description": "unit lig", "domain": "positive"},
                {"symbol": "lags", "description": "unit lag", "domain": "positive"},
                {"symbol": "lugs", "description": "unit lug", "domain": "positive"},
            ],
            "questions": [{"id": "q1", "text": "How many ligs are equivalent to 80 lugs?", "target_ids": ["t1"]}],
            "targets": [{"id": "t1", "text": "ligs equivalent to 80 lugs", "target_type": "compute"}],
            "confidence": {"overall": 1.0},
        }
    )
    emr = EMR.model_validate(
        {
            "problem_id": "math_middle_equivalent_ligs",
            "source_problem_type": "word_problem",
            "representation_type": "unknown",
        }
    )
    classification = ProblemClassifier().classify(spr, emr)

    chain = Solver(
        planner=DeepSeekPlanner(client=FakeWrongEquivalentUnitPlanningClient()),
        use_planner=True,
    ).solve(emr, classification, spr=spr)

    assert chain.final_answer == "ligs = 63"
    assert chain.metadata["final_answer_items"] == [
        {
            "target": "ligs",
            "value": "63",
            "source_tool_call_id": "derived_equivalent_unit_chain",
            "backend_operation": "evaluate",
            "backend_confirmed": True,
        }
    ]


def test_solver_completes_equivalent_price_chain_with_number_words() -> None:
    solver = AlgebraSolver()
    spr = SPR.model_validate(
        {
            "source_image": {"image_path": "math_middle_0126.png"},
            "problem_text": (
                "If six cookies cost the same as 2 brownies, and four brownies cost the same as "
                "10 cupcakes, how many cupcakes can Bob buy for the price of eighteen cookies?"
            ),
            "problem_type": "word_problem",
            "confidence": {"overall": 1.0},
        }
    )

    package = solver._derive_equivalent_unit_chain_backend_result(spr)

    assert package is not None
    assert package["backend_result"]["output"]["value"] == "15"
    assert package["validation_report"]["normalized_request"]["target"] == "cupcakes"


def test_solver_evaluates_absolute_value_target_expression_after_direct_solve() -> None:
    spr = SPR.model_validate(
        {
            "source_image": {"image_path": "math_middle_0065.png"},
            "problem_text": (
                "The average of the numbers 23 and x is 27. "
                "What is the positive difference between 23 and x?"
            ),
            "problem_type": "algebra",
            "variables": [{"symbol": "x", "description": "unknown number", "domain": "real"}],
            "questions": [{"id": "q1", "text": "What is the positive difference between 23 and x?", "target_ids": ["t1"]}],
            "targets": [{"id": "t1", "text": "positive difference between 23 and x", "target_type": "compute"}],
            "confidence": {"overall": 1.0},
        }
    )
    emr = EMR.model_validate(
        {
            "problem_id": "math_middle_abs_difference",
            "source_problem_type": "algebra",
            "representation_type": "equation_system",
            "variables": [{"id": "var_x", "symbol": "x", "domain": "real"}],
            "equations": [{"id": "eq1", "lhs_sympy": "(23+x)/2", "rhs_sympy": "27", "relation": "eq"}],
            "expressions": [{"id": "expr1", "sympy": "|x-23|", "raw_text": "|x-23|"}],
            "goals": [
                {
                    "id": "g1",
                    "goal_type": "compute",
                    "target": "positive difference",
                    "target_expression_id": "expr1",
                }
            ],
        }
    )

    chain = AlgebraSolver().solve(emr, algebra_solver_classification_result("math_middle_abs_difference"), spr=spr)

    assert chain.final_answer == "positive difference = 8"
    assert chain.metadata["final_answer_items"][-1]["value"] == "8"


def test_solver_prefers_deterministic_two_set_neither_inclusion_exclusion() -> None:
    spr = SPR.model_validate(
        {
            "source_image": {"image_path": "math_middle_0058.png"},
            "problem_text": (
                "Of the 100 students in the science club, 65 take mathematics, "
                "43 take physics and 10 students take both mathematics and physics. "
                "How many science club students take neither mathematics nor physics?"
            ),
            "problem_type": "word_problem",
            "confidence": {"overall": 1.0},
        }
    )
    emr = EMR.model_validate(
        {
            "problem_id": "math_middle_two_set_neither",
            "source_problem_type": "word_problem",
            "representation_type": "unknown",
        }
    )
    classification = ProblemClassifier().classify(spr, emr)

    chain = Solver(
        planner=DeepSeekPlanner(client=FakeIrrelevantPlanningClient()),
        use_planner=True,
    ).solve(emr, classification, spr=spr)

    assert chain.final_answer == "neither = 2"
    assert chain.metadata["final_answer_items"] == [
        {
            "target": "neither",
            "value": "2",
            "source_tool_call_id": "derived_two_set_neither",
            "backend_operation": "evaluate",
            "backend_confirmed": True,
        }
    ]


def test_solver_prefers_deterministic_ratio_share_total() -> None:
    spr = SPR.model_validate(
        {
            "source_image": {"image_path": "math_middle_0084.png"},
            "problem_text": (
                "Amanda, Ben, and Carlos share a sum of money. Their portions are in the ratio "
                "of 1:2:7, respectively. If Amanda's portion is $20, what is the total amount of money shared?"
            ),
            "problem_type": "word_problem",
            "confidence": {"overall": 1.0},
        }
    )
    emr = EMR.model_validate(
        {
            "problem_id": "math_middle_ratio_share_total",
            "source_problem_type": "word_problem",
            "representation_type": "unknown",
        }
    )
    classification = ProblemClassifier().classify(spr, emr)

    chain = Solver(
        planner=DeepSeekPlanner(client=FakeIrrelevantPlanningClient()),
        use_planner=True,
    ).solve(emr, classification, spr=spr)

    assert chain.final_answer == "total = 200"
    assert chain.metadata["final_answer_items"] == [
        {
            "target": "total",
            "value": "200",
            "source_tool_call_id": "derived_ratio_share_total",
            "backend_operation": "evaluate",
            "backend_confirmed": True,
        }
    ]


def test_solver_completes_square_sum_identity_from_emr_conditions() -> None:
    spr = SPR.model_validate(
        {
            "source_image": {"image_path": "math_middle_0108.png"},
            "problem_text": "If $(x + y)^2 = 25$ and $xy = 6$, what is the value of $x^2 + y^2$?",
            "problem_type": "algebra",
            "variables": [
                {"symbol": "x", "description": "variable x", "domain": "real"},
                {"symbol": "y", "description": "variable y", "domain": "real"},
            ],
            "questions": [{"id": "q1", "text": "What is the value of x^2 + y^2?", "target_ids": ["t1"]}],
            "targets": [{"id": "t1", "text": "x^2 + y^2", "target_type": "compute"}],
            "confidence": {"overall": 1.0},
        }
    )
    emr = EMR.model_validate(
        {
            "problem_id": "math_middle_square_sum_identity",
            "source_problem_type": "algebra",
            "representation_type": "unknown",
            "variables": [
                {"id": "var_x", "symbol": "x", "domain": "real"},
                {"id": "var_y", "symbol": "y", "domain": "real"},
            ],
            "equations": [
                {"id": "eq1", "lhs_sympy": "(x+y)**2", "rhs_sympy": "25", "relation": "eq"},
                {"id": "eq2", "lhs_sympy": "x*y", "rhs_sympy": "6", "relation": "eq"},
            ],
            "expressions": [{"id": "expr1", "sympy": "x**2+y**2", "raw_text": "x^2+y^2"}],
            "goals": [
                {
                    "id": "g1",
                    "goal_type": "compute",
                    "target": "x_squared_plus_y_squared",
                    "target_expression_id": "expr1",
                }
            ],
        }
    )
    classification = ProblemClassifier().classify(spr, emr)

    chain = Solver(
        planner=DeepSeekPlanner(client=FakeIrrelevantPlanningClient()),
        use_planner=True,
    ).solve(emr, classification, spr=spr)

    assert chain.final_answer == "x_squared_plus_y_squared = 13"
    assert chain.metadata["final_answer_items"] == [
        {
            "target": "x_squared_plus_y_squared",
            "value": "13",
            "source_tool_call_id": "derived_square_sum_identity",
            "backend_operation": "evaluate",
            "backend_confirmed": True,
        }
    ]
