import pytest
from jinja2.exceptions import TemplateError

from mistral_common.integrations.chat_templates.chat_templates import generate_chat_template
from mistral_common.tokens.tokenizers.base import TokenizerVersion
from tests.integrations.chat_templates.helpers import encode_transformers


class TestV15ReasoningEffort:
    @pytest.mark.parametrize(
        ("spm", "version", "image", "audio", "think"),
        [
            (False, TokenizerVersion.v15, False, False, False),
            (False, TokenizerVersion.v15, True, False, False),
            (False, TokenizerVersion.v15, False, False, True),
            (False, TokenizerVersion.v15, True, False, True),
        ],
    )
    def test_reasoning_effort_validation(
        self,
        spm: bool,
        version: TokenizerVersion,
        image: bool,
        audio: bool,
        think: bool,
    ) -> None:
        chat_template = generate_chat_template(
            spm=spm, tokenizer_version=version, image_support=image, audio_support=audio, thinking_support=think
        )

        # Test valid reasoning_effort values -- v15 always emits [MODEL_SETTINGS]
        # (None/undefined defaults to 'none' in the template)
        valid_conversations = [
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"},
                ],
                "reasoning_effort": "none",
            },
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"},
                ],
                "reasoning_effort": "high",
            },
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"},
                ]
                # No reasoning_effort -- template defaults to 'none'
            },
        ]

        for conv in valid_conversations:
            result = encode_transformers(chat_template, conv)  # type: ignore[arg-type]
            assert result is not None
            assert "[MODEL_SETTINGS]" in result

        # Test invalid reasoning_effort values
        invalid_conversations = [
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"},
                ],
                "reasoning_effort": "low",
            },
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"},
                ],
                "reasoning_effort": "medium",
            },
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"},
                ],
                "reasoning_effort": "invalid_value",
            },
        ]

        for conv in invalid_conversations:
            with pytest.raises(TemplateError, match='reasoning_effort must be either "none" or "high"'):
                encode_transformers(chat_template, conv)
