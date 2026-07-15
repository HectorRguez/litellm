import copy

import litellm
from litellm.cost_calculator import response_cost_calculator
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


def test_router_stream_recalculates_zero_hidden_cost_with_custom_pricing():
    backend_model = "openrouter/custom-flat-request-streaming-test"
    original_shared_entry = copy.deepcopy(litellm.model_cost.get(backend_model))
    deployment_id = None
    try:
        router = litellm.Router(
            model_list=[
                {
                    "model_name": "flat-request-streaming-test",
                    "litellm_params": {
                        "model": backend_model,
                        "api_key": "fake-key",
                        "input_cost_per_request": 0.04,
                    },
                }
            ]
        )
        deployment_id = router.model_list[0]["model_info"]["id"]
        stream = router.completion(
            model="flat-request-streaming-test",
            messages=[{"role": "user", "content": "Music"}],
            stream=True,
            mock_response="audio",
        )
        response = ModelResponse(
            id="openrouter-audio",
            model="custom-flat-request-streaming-test",
            choices=[],
            usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )
        response._hidden_params = {
            "model_id": deployment_id,
            "response_cost": 0.0,
        }

        assert stream.logging_obj._response_cost_calculator(result=response) == 0.04
    finally:
        if deployment_id is not None:
            litellm.model_cost.pop(deployment_id, None)
        if original_shared_entry is None:
            litellm.model_cost.pop(backend_model, None)
        else:
            litellm.model_cost[backend_model] = original_shared_entry
        _invalidate_model_cost_lowercase_map()
