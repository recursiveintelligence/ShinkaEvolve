import backoff
import openai
from .pricing import DEEPSEEK_MODELS
from .result import QueryResult
import logging

logger = logging.getLogger(__name__)


def backoff_handler(details):
    exc = details.get("exception")
    if exc:
        logger.info(
            f"DeepSeek - Retry {details['tries']} due to error: {exc}. Waiting {details['wait']:0.1f}s..."
        )


@backoff.on_exception(
    backoff.expo,
    (
        openai.APIConnectionError,
        openai.APIStatusError,
        openai.RateLimitError,
        openai.APITimeoutError,
    ),
    max_tries=5,
    max_value=20,
    on_backoff=backoff_handler,
)
def query_deepseek(
    client,
    model,
    msg,
    system_msg,
    msg_history,
    output_model,
    model_posteriors=None,
    **kwargs,
) -> QueryResult:
    """Query DeepSeek model."""
    if output_model is not None:
        raise NotImplementedError("Structured output not supported for DeepSeek.")
    new_msg_history = msg_history + [{"role": "user", "content": msg}]
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            *new_msg_history,
        ],
        **kwargs,
        n=1,
        stop=None,
    )
    content = response.choices[0].message.content
    try:
        thought = response.choices[0].message.reasoning_content
    except:
        thought = ""
    new_msg_history.append({"role": "assistant", "content": content})
    input_cost = DEEPSEEK_MODELS[model]["input_price"] * response.usage.prompt_tokens
    output_cost = (
        DEEPSEEK_MODELS[model]["output_price"] * response.usage.completion_tokens
    )
    return QueryResult(
        content=content,
        msg=msg,
        system_msg=system_msg,
        new_msg_history=new_msg_history,
        model_name=model,
        kwargs=kwargs,
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
        cost=input_cost + output_cost,
        input_cost=input_cost,
        output_cost=output_cost,
        thought=thought,
        model_posteriors=model_posteriors,
    )
