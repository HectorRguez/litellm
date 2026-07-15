import copy

import litellm
from litellm.cost_calculator import response_cost_calculator
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.types.utils import ModelResponse, Usage
from litellm.utils import _invalidate_model_cost_lowercase_map


def test_cost_calculator_uses_openrouter_streaming_usage_cost():
    response = ModelResponse(
        id="openrouter-audio",
        model="google/lyria-3-clip-preview",
        choices=[],
        usage=Usage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            cost=0.03,
        ),
    )

    result = response_cost_calculator(
        response_object=response,
        model="google/lyria-3-clip-preview",
        custom_llm_provider="openrouter",
        call_type="completion",
        optional_params={},
        custom_pricing=True,
    )

    assert result == 0.03


def test_cost_calculator_uses_router_flat_request_pricing():
    deployment_id = "openrouter-flat-request-pricing-test"
    original_entry = copy.deepcopy(litellm.model_cost.get(deployment_id))

    try:
        litellm.model_cost[deployment_id] = {
            "litellm_provider": "openrouter",
            "mode": "chat",
            "input_cost_per_request": 0.04,
        }
        _invalidate_model_cost_lowercase_map()
        response = ModelResponse(
            id="openrouter-audio",
            model="unmapped-flat-request-model",
            choices=[],
            usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )

        result = response_cost_calculator(
            response_object=response,
            model="unmapped-flat-request-model",
            custom_llm_provider="openrouter",
            call_type="completion",
            optional_params={},
            custom_pricing=True,
            router_model_id=deployment_id,
        )

        assert result == 0.04
    finally:
        if original_entry is None:
            litellm.model_cost.pop(deployment_id, None)
        else:
            litellm.model_cost[deployment_id] = original_entry
        _invalidate_model_cost_lowercase_map()


def test_logging_recalculates_zero_hidden_cost_with_custom_pricing():
    deployment_id = "openrouter-flat-request-streaming-test"
    original_entry = copy.deepcopy(litellm.model_cost.get(deployment_id))

    try:
        litellm.model_cost[deployment_id] = {
            "litellm_provider": "openrouter",
            "mode": "chat",
            "input_cost_per_request": 0.04,
        }
        _invalidate_model_cost_lowercase_map()
        logging_obj = LiteLLMLoggingObj(
            model="unmapped-flat-request-model",
            messages=[{"role": "user", "content": "Music"}],
            stream=True,
            call_type="acompletion",
            start_time=0,
            litellm_call_id="openrouter-flat-request-streaming",
            function_id="openrouter-flat-request-streaming",
        )
        logging_obj.update_environment_variables(
            model="unmapped-flat-request-model",
            user="",
            optional_params={},
            litellm_params={
                "input_cost_per_request": 0.04,
                "metadata": {
                    "model_info": {
                        "id": deployment_id,
                        "input_cost_per_request": 0.04,
                    }
                },
            },
        )
        logging_obj.model_call_details["custom_llm_provider"] = "openrouter"
        response = ModelResponse(
            id="openrouter-audio",
            model="unmapped-flat-request-model",
            choices=[],
            usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )
        response._hidden_params = {
            "model_id": deployment_id,
            "response_cost": 0.0,
        }

        assert logging_obj._response_cost_calculator(result=response) == 0.04
    finally:
        if original_entry is None:
            litellm.model_cost.pop(deployment_id, None)
        else:
            litellm.model_cost[deployment_id] = original_entry
        _invalidate_model_cost_lowercase_map()
