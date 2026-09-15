"""Benchmark scenario definitions.

Each scenario is a static, deterministic workload that can be
repeated across runs for comparable measurements.

Scenarios use *text input* (not microphone audio) so that the
LLM / Tool / TTS portions are benchmarked consistently.  STT is
explicitly **not** measured in this phase.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BenchmarkScenario:
    """A single deterministic benchmark workload."""

    scenario_id: str
    name: str
    description: str
    input_text: str
    expected_tool_calls: int
    expected_success: bool = True
    category: str = "baseline"
    include_tts: bool = True
    notes: str = ""
    # Optional: restrict which tools the scenario may use
    allowed_tools: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Scenario catalogue
# ---------------------------------------------------------------------------

B1_SIMPLE_CONVERSATION = BenchmarkScenario(
    scenario_id="simple_conversation",
    name="Simple Conversation",
    description="Basic greeting — no tool calls expected. Measures LLM baseline latency.",
    input_text="Hello, how are you?",
    expected_tool_calls=0,
    category="baseline",
)

B2_CONTEXT_CONVERSATION = BenchmarkScenario(
    scenario_id="context_conversation",
    name="Context Conversation",
    description=(
        "Short factual statement — no tool calls. "
        "Tests LLM response generation and context handling."
    ),
    input_text=(
        "Please remember this for later: my employee ID is E001. "
        "Do not look anything up right now — just acknowledge."
    ),
    expected_tool_calls=0,
    category="baseline",
)

B3_SINGLE_TOOL = BenchmarkScenario(
    scenario_id="single_tool",
    name="Single Tool",
    description="Requests exactly one employee lookup. Validates tool-call pipeline.",
    input_text="Look up employee E001.",
    expected_tool_calls=1,
    category="tool",
    allowed_tools=["get_employee"],
)

B4_MULTI_TOOL = BenchmarkScenario(
    scenario_id="multi_tool",
    name="Multi Tool",
    description="Requests employee lookup followed by leave balance. Two tool calls expected.",
    input_text="Look up employee E001 and then check their PTO balance.",
    expected_tool_calls=2,
    category="tool",
    allowed_tools=["get_employee", "get_leave_balance"],
)

B5_LONG_RESPONSE = BenchmarkScenario(
    scenario_id="long_response",
    name="Long Response",
    description=(
        "Requests a detailed explanation — no tools. "
        "Produces a longer LLM response for TTS measurement."
    ),
    input_text=(
        "Explain in detail how a voice AI pipeline works, "
        "covering STT, LLM, tool calling, and TTS stages."
    ),
    expected_tool_calls=0,
    category="stress",
    include_tts=True,
)

# Ordered catalogue for iteration / API listing
ALL_SCENARIOS: list[BenchmarkScenario] = [
    B1_SIMPLE_CONVERSATION,
    B2_CONTEXT_CONVERSATION,
    B3_SINGLE_TOOL,
    B4_MULTI_TOOL,
    B5_LONG_RESPONSE,
]

_SCENARIO_MAP: dict[str, BenchmarkScenario] = {
    s.scenario_id: s for s in ALL_SCENARIOS
}


def get_scenario(scenario_id: str) -> BenchmarkScenario | None:
    """Look up a scenario by ID."""
    return _SCENARIO_MAP.get(scenario_id)


def list_scenarios() -> list[BenchmarkScenario]:
    """Return all available scenarios."""
    return list(ALL_SCENARIOS)
