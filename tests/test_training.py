import pytest

from training.train_sft import to_prompt_completion


def test_to_prompt_completion_separates_the_assistant_answer():
    example = {
        "messages": [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "How should I warm up?"},
            {"role": "assistant", "content": "Start with easy movement."},
        ],
        "metadata": {"topic": "programming"},
    }

    converted = to_prompt_completion(example)

    assert converted["prompt"] == example["messages"][:-1]
    assert converted["completion"] == [example["messages"][-1]]


def test_to_prompt_completion_rejects_a_non_assistant_final_message():
    with pytest.raises(ValueError, match="assistant role"):
        to_prompt_completion(
            {
                "messages": [
                    {"role": "system", "content": "Be helpful."},
                    {"role": "user", "content": "Hi"},
                ]
            }
        )
