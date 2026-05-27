from typing import Any

import pytest

from mistral_common.integrations.chat_templates.chat_templates import generate_chat_template
from mistral_common.tokens.tokenizers.base import TokenizerVersion
from tests.integrations.chat_templates.helpers import _load_golden_template, _make_config, render_template


class TestV15ModelSettings:
    def test_v15_reasoning_effort(self) -> None:
        # Test v15 template generation
        template = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v15,
            image_support=False,
            audio_support=False,
            thinking_support=False,
        )

        # Verify the template contains MODEL_SETTINGS logic
        assert "[MODEL_SETTINGS]" in template
        assert "reasoning_effort" in template

        # Test rendering with different reasoning effort values
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        # Test with no reasoning effort (None/undefined — defaults to 'none')
        output_none = render_template(template, messages, reasoning_effort=None)
        assert '[MODEL_SETTINGS]{"reasoning_effort": "none"}[/MODEL_SETTINGS]' in output_none

        # Test with reasoning effort='high'
        output_high = render_template(template, messages, reasoning_effort="high")
        assert '[MODEL_SETTINGS]{"reasoning_effort": "high"}[/MODEL_SETTINGS]' in output_high

        # Test with reasoning effort='none' (explicit string)
        output_none_explicit = render_template(template, messages, reasoning_effort="none")
        assert '[MODEL_SETTINGS]{"reasoning_effort": "none"}[/MODEL_SETTINGS]' in output_none_explicit

        # Test that v15 static and dynamic templates produce same output
        config = _make_config((TokenizerVersion.v15, False, False, False, False, False))
        static_template = _load_golden_template(config)

        static_output = render_template(static_template, messages, reasoning_effort="high")
        dynamic_output = render_template(template, messages, reasoning_effort="high")

        assert static_output == dynamic_output

    def test_v15_with_features(self) -> None:
        # Test v15 with image support
        template_image = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v15,
            image_support=True,
            audio_support=False,
            thinking_support=False,
        )

        messages_with_image = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image_url", "image_url": "http://example.com/image.png"},
                ],
            },
            {"role": "assistant", "content": "It's an image."},
        ]

        output = render_template(template_image, messages_with_image, reasoning_effort="high")
        assert '[MODEL_SETTINGS]{"reasoning_effort": "high"}[/MODEL_SETTINGS]' in output
        assert "[IMG]" in output

        # Test v15 with thinking support
        template_think = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v15,
            image_support=False,
            audio_support=False,
            thinking_support=True,
        )

        messages_with_thinking = [
            {"role": "user", "content": "Solve this problem"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Let me think..."},
                    {"type": "text", "text": "The answer is 42."},
                ],
            },
        ]

        output = render_template(template_think, messages_with_thinking, reasoning_effort="none")
        assert '[MODEL_SETTINGS]{"reasoning_effort": "none"}[/MODEL_SETTINGS]' in output
        assert "[THINK]Let me think...[/THINK]" in output

    def test_v15_with_tools(self) -> None:
        template = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v15,
            image_support=False,
            audio_support=False,
            thinking_support=False,
        )

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string", "description": "The city and state"}},
                        "required": ["location"],
                    },
                },
            }
        ]

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather in Paris?"},
        ]

        output = render_template(template, messages, tools=tools, reasoning_effort="high")
        assert '[MODEL_SETTINGS]{"reasoning_effort": "high"}[/MODEL_SETTINGS]' in output
        assert "[AVAILABLE_TOOLS]" in output
        assert "get_weather" in output

    @pytest.mark.parametrize(
        ("has_system", "has_tools", "reasoning_effort"),
        [
            (True, True, "high"),
            (True, False, "high"),
            (False, True, "high"),
            (False, False, "high"),
            (True, True, "none"),
            (False, False, "none"),
            (True, True, None),
            (False, False, None),
        ],
    )
    def test_v15_available_tools_and_settings_ordering(
        self, has_system: bool, has_tools: bool, reasoning_effort: str | None
    ) -> None:
        template = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v15,
            image_support=False,
            audio_support=False,
            thinking_support=False,
        )

        messages: list[dict[str, Any]] = []
        if has_system:
            messages.append({"role": "system", "content": "You are helpful."})
        messages.extend(
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ]
        )

        tools: list[dict[str, Any]] | None = None
        if has_tools:
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "tool1",
                        "description": "",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]

        output = render_template(template, messages, tools=tools, reasoning_effort=reasoning_effort)

        # Verify ordering of special blocks
        if has_system:
            assert "[SYSTEM_PROMPT]You are helpful.[/SYSTEM_PROMPT]" in output
            sp_pos = output.index("[SYSTEM_PROMPT]")
        else:
            assert "[SYSTEM_PROMPT]" not in output
            sp_pos = -1

        if has_tools:
            assert "[AVAILABLE_TOOLS]" in output
            tools_pos = output.index("[AVAILABLE_TOOLS]")
            assert tools_pos > sp_pos
        else:
            assert "[AVAILABLE_TOOLS]" not in output
            tools_pos = sp_pos

        # MODEL_SETTINGS is always emitted for v15 (reasoning_effort is always provided)
        assert "[MODEL_SETTINGS]" in output
        settings_pos = output.index("[MODEL_SETTINGS]")
        assert settings_pos > tools_pos

        inst_pos = output.index("[INST]")
        assert inst_pos > settings_pos

    @pytest.mark.parametrize("image", [False, True])
    def test_v15_think_template_rejects_think_in_system(self, image: bool) -> None:
        template = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v15,
            image_support=image,
            audio_support=False,
            thinking_support=True,
        )

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "System text."},
                    {"type": "thinking", "thinking": "System thinking."},
                ],
            },
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

        with pytest.raises(ValueError, match="Only text chunks are supported in system message contents"):
            render_template(template, messages)
