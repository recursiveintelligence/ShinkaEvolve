import backoff
import anthropic
from .pricing import CLAUDE_MODELS
from .result import QueryResult
import logging

logger = logging.getLogger(__name__)


MAX_TRIES = 20
MAX_VALUE = 20


def backoff_handler(details):
    exc = details.get("exception")
    if exc:
        logger.info(
            f"Anthropic - Retry {details['tries']} due to error: {exc}. Waiting {details['wait']:0.1f}s..."
        )


@backoff.on_exception(
    backoff.expo,
    (
        anthropic.APIConnectionError,
        anthropic.APIStatusError,
        anthropic.RateLimitError,
        anthropic.APITimeoutError,
    ),
    max_tries=MAX_TRIES,
    max_value=MAX_VALUE,
    on_backoff=backoff_handler,
)
def query_anthropic(
    client,
    model,
    msg,
    system_msg,
    msg_history,
    output_model,
    model_posteriors=None,
    **kwargs,
) -> QueryResult:
    """Query Anthropic/Bedrock model."""
    new_msg_history = msg_history + [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": msg,
                }
            ],
        }
    ]
    if output_model is None:
        response = client.messages.create(
            model=model,
            system=system_msg,
            messages=new_msg_history,
            **kwargs,
        )
        # Separate thinking from non-thinking content
        if len(response.content) == 1:
            thought = ""
            content = response.content[0].text
        else:
            thought = response.content[0].thinking
            content = response.content[1].text
    else:
        raise NotImplementedError("Structured output not supported for Anthropic.")
    new_msg_history.append(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": content,
                }
            ],
        }
    )
    input_cost = CLAUDE_MODELS[model]["input_price"] * response.usage.input_tokens
    output_cost = CLAUDE_MODELS[model]["output_price"] * response.usage.output_tokens
    result = QueryResult(
        content=content,
        msg=msg,
        system_msg=system_msg,
        new_msg_history=new_msg_history,
        model_name=model,
        kwargs=kwargs,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        cost=input_cost + output_cost,
        input_cost=input_cost,
        output_cost=output_cost,
        thought=thought,
        model_posteriors=model_posteriors,
    )
    return result
