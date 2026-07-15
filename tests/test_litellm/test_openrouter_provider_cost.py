from litellm.cost_calculator import response_cost_calculator
from litellm.types.utils import ModelResponse, Usage


def test_cost_calculator_uses_openrouter_streaming_usage_cost():
    response = ModelResponse(
        id="openrouter-audio",
        model="google/lyria-3-clip-preview",
        choices=[],
        usage=Usage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            cost=0.04,
        ),
    )

    result = response_cost_calculator(
        response_object=response,
        model="google/lyria-3-clip-preview",
        custom_llm_provider="openrouter",
        call_type="completion",
        optional_params={},
    )

    assert result == 0.04
