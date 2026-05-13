from __future__ import annotations

from typing import Any

import pytest

from mistral_common.integrations.chat_templates.chat_templates import generate_chat_template
from mistral_common.integrations.chat_templates.template_generator import TemplateConfig, build_chat_template
from mistral_common.tokens.tokenizers.base import TokenizerVersion
from tests.integrations.chat_templates.conftest import (
    ALL_CONFIGS,
    _load_golden_template,
    _make_config,
    render_template,
)


class TestTemplateConfigValidation:
    def test_invalid_config_spm_v11(self) -> None:
        """Test that SPM with v11+ raises error."""
        with pytest.raises(ValueError, match="SPM tokenizer is not supported"):
            TemplateConfig(version=TokenizerVersion.v11, spm=True)

    def test_invalid_config_image_audio(self) -> None:
        """Test that image and audio together raises error."""
        with pytest.raises(ValueError, match="Image and audio support are mutually exclusive"):
            TemplateConfig(version=TokenizerVersion.v7, image_support=True, audio_support=True)

    def test_invalid_config_image_v1(self) -> None:
        """Test that image support with v1/v2 raises error."""
        with pytest.raises(ValueError, match="Image support is only available"):
            TemplateConfig(version=TokenizerVersion.v1, image_support=True)

    def test_invalid_config_audio_v3(self) -> None:
        """Test that audio support with v3 raises error."""
        with pytest.raises(ValueError, match="Audio support is only available"):
            TemplateConfig(version=TokenizerVersion.v3, audio_support=True)

    def test_invalid_config_thinking_v7(self) -> None:
        """Test that thinking support with non-v13 raises error."""
        with pytest.raises(ValueError, match="Thinking support is only available"):
            TemplateConfig(version=TokenizerVersion.v7, thinking_support=True)

    def test_invalid_config_audio_thinking(self) -> None:
        """Test that audio and thinking together raises error."""
        with pytest.raises(ValueError, match="Audio and thinking support are mutually exclusive"):
            TemplateConfig(version=TokenizerVersion.v13, audio_support=True, thinking_support=True)

    def test_invalid_config_plain_thinking_with_thinking(self) -> None:
        r"""`plain_thinking_support` and `thinking_support` are mutually exclusive."""
        with pytest.raises(ValueError, match="Plain thinking support and thinking support are mutually exclusive"):
            TemplateConfig(version=TokenizerVersion.v11, thinking_support=True, plain_thinking_support=True)

    def test_invalid_config_plain_thinking_non_v11(self) -> None:
        r"""`plain_thinking_support` only works with v11."""
        with pytest.raises(ValueError, match="Plain thinking support is only available for tokenizer version v11"):
            TemplateConfig(version=TokenizerVersion.v15, plain_thinking_support=True)

    def test_invalid_config_plain_thinking_with_audio(self) -> None:
        r"""`plain_thinking_support` and `audio_support` are mutually exclusive."""
        with pytest.raises(ValueError, match="Audio and plain thinking support are mutually exclusive"):
            TemplateConfig(version=TokenizerVersion.v11, audio_support=True, plain_thinking_support=True)


def test_generate_chat_template_function() -> None:
    """Test that generate_chat_template function works correctly."""
    # Test basic generation
    template = generate_chat_template(
        spm=False,
        tokenizer_version=TokenizerVersion.v13,
        image_support=False,
        audio_support=False,
        thinking_support=False,
    )
    assert "{%- set default_system_message = '' %}" in template
    assert "bos_token" in template

    # Test with default system prompt
    template_with_prompt = generate_chat_template(
        spm=False,
        tokenizer_version=TokenizerVersion.v13,
        image_support=False,
        audio_support=False,
        thinking_support=False,
        default_system_prompt="You are a helpful assistant.",
    )
    assert "You are a helpful assistant." in template_with_prompt

    # Test that both static and dynamic produce same output
    config = _make_config((TokenizerVersion.v13, False, False, False, False, False))
    static_template = _load_golden_template(config)

    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]

    static_output = render_template(static_template, messages)
    dynamic_output = render_template(template, messages)

    assert static_output == dynamic_output


def test_v15_reasoning_effort() -> None:
    """Test that v15 templates correctly handle reasoning effort."""
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


def test_v15_with_features() -> None:
    """Test that v15 templates work correctly with images and thinking."""
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


def test_v15_with_tools() -> None:
    """Test that v15 templates work correctly with tools."""
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
    ("version", "spm"),
    [
        (TokenizerVersion.v7, False),
        (TokenizerVersion.v7, True),
        (TokenizerVersion.v13, False),
        (TokenizerVersion.v15, False),
    ],
)
def test_aggregation_consecutive_assistants_both_tool_calls(version: TokenizerVersion, spm: bool) -> None:
    r"""Test consecutive assistant messages where both have tool_calls.

    This pattern is rejected by the validator but the normalizer handles it.
    We test the template directly to ensure the Jinja aggregation logic works.
    """
    template = generate_chat_template(
        spm=spm,
        tokenizer_version=version,
        image_support=False,
        audio_support=False,
        thinking_support=False,
    )

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Weather?"},
        {
            "role": "assistant",
            "content": "Checking Paris.",
            "tool_calls": [
                {
                    "id": "123456789",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                },
            ],
        },
        {
            "role": "assistant",
            "content": "And London.",
            "tool_calls": [
                {
                    "id": "023456789",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "London"}'},
                },
            ],
        },
        {"role": "tool", "name": "get_weather", "content": "22", "tool_call_id": "123456789"},
        {"role": "tool", "name": "get_weather", "content": "15", "tool_call_id": "023456789"},
        {"role": "assistant", "content": "Paris: 22, London: 15"},
        {"role": "user", "content": "Thanks"},
        {"role": "assistant", "content": "Welcome"},
    ]

    tools: list[dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]

    output = render_template(template, messages, tools=tools)

    # Both assistant messages should be merged: content joined with \n\n
    assert "Checking Paris.\n\nAnd London." in output
    # Both tool calls should be present in the output
    assert '"city": "Paris"' in output
    assert '"city": "London"' in output


@pytest.mark.parametrize(
    ("version", "spm"),
    [
        (TokenizerVersion.v2, False),
        (TokenizerVersion.v2, True),
        (TokenizerVersion.v3, False),
        (TokenizerVersion.v3, True),
        (TokenizerVersion.v7, False),
        (TokenizerVersion.v7, True),
        (TokenizerVersion.v11, False),
        (TokenizerVersion.v13, False),
        (TokenizerVersion.v15, False),
    ],
)
def test_user_after_tool_accepted(version: TokenizerVersion, spm: bool) -> None:
    r"""User message after tool results is accepted by the ordering check."""
    template = generate_chat_template(
        spm=spm,
        tokenizer_version=version,
        image_support=False,
        audio_support=False,
        thinking_support=False,
    )

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Search for info"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "aaaaaaaaa",
                    "type": "function",
                    "function": {"name": "search", "arguments": '{"q": "info"}'},
                },
            ],
        },
        {"role": "tool", "name": "search", "content": "result1", "tool_call_id": "aaaaaaaaa"},
        {"role": "user", "content": "Now refine"},
        {"role": "assistant", "content": "Here is the refined answer"},
    ]

    tools: list[dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "",
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
            },
        }
    ]

    output = render_template(template, messages, tools=tools)
    assert "Now refine" in output
    assert "Here is the refined answer" in output


@pytest.mark.parametrize(
    ("version", "spm"),
    [
        (TokenizerVersion.v7, False),
        (TokenizerVersion.v7, True),
        (TokenizerVersion.v13, False),
        (TokenizerVersion.v15, False),
    ],
)
def test_user_after_tool_static_matches_dynamic(version: TokenizerVersion, spm: bool) -> None:
    r"""Static and dynamic templates produce same output for tool->user transitions."""
    config = _make_config((version, spm, False, False, False, False))
    static_template = _load_golden_template(config)
    dynamic_template = generate_chat_template(
        spm=spm,
        tokenizer_version=version,
        image_support=False,
        audio_support=False,
        thinking_support=False,
    )

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Search for info"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "aaaaaaaaa",
                    "type": "function",
                    "function": {"name": "search", "arguments": '{"q": "info"}'},
                },
            ],
        },
        {"role": "tool", "name": "search", "content": "result1", "tool_call_id": "aaaaaaaaa"},
        {"role": "user", "content": "Now refine"},
        {"role": "assistant", "content": "Here is the refined answer"},
    ]

    tools: list[dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "",
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
            },
        }
    ]

    static_output = render_template(static_template, messages, tools=tools)
    dynamic_output = render_template(dynamic_template, messages, tools=tools)

    assert static_output == dynamic_output, (
        f"Output mismatch for version={version}, spm={spm}\n\n"
        f"Static output: {static_output}\n"
        f"Dynamic output: {dynamic_output}"
    )


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
    has_system: bool, has_tools: bool, reasoning_effort: str | None
) -> None:
    r"""Test that v15 emits system, available_tools, and model_settings in the correct order.

    Expected order: `[SYSTEM_PROMPT]...[/SYSTEM_PROMPT]` (if system) then
    `[AVAILABLE_TOOLS]...[/AVAILABLE_TOOLS]` (if tools) then
    `[MODEL_SETTINGS]...[/MODEL_SETTINGS]` (always, None defaults to 'none') then
    `[INST]...[/INST]`.
    """
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


# ── reasoning / reasoning_content → thinking chunk conversion ──────────


THINK_CONFIGS = [
    (TokenizerVersion.v13, False, False),
    (TokenizerVersion.v13, True, False),
    (TokenizerVersion.v15, False, False),
    (TokenizerVersion.v15, True, False),
]


@pytest.mark.parametrize(("version", "image", "audio"), THINK_CONFIGS)
def test_reasoning_content_to_thinking_chunk(version: TokenizerVersion, image: bool, audio: bool) -> None:
    r"""`reasoning_content` on an assistant message produces a leading `[THINK]...[/THINK]`."""
    template = generate_chat_template(
        spm=False,
        tokenizer_version=version,
        image_support=image,
        audio_support=audio,
        thinking_support=True,
    )

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "reasoning_content": "Let me add 2 and 2.", "content": "The answer is 4."},
    ]

    output = render_template(template, messages)
    assert "[THINK]Let me add 2 and 2.[/THINK]The answer is 4." in output


@pytest.mark.parametrize(("version", "image", "audio"), THINK_CONFIGS)
def test_reasoning_field_to_thinking_chunk(version: TokenizerVersion, image: bool, audio: bool) -> None:
    r"""`reasoning` field (alias) produces the same `[THINK]...[/THINK]`."""
    template = generate_chat_template(
        spm=False,
        tokenizer_version=version,
        image_support=image,
        audio_support=audio,
        thinking_support=True,
    )

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "What is 3+3?"},
        {"role": "assistant", "reasoning": "Adding three and three.", "content": "The answer is 6."},
    ]

    output = render_template(template, messages)
    assert "[THINK]Adding three and three.[/THINK]The answer is 6." in output


@pytest.mark.parametrize(("version", "image", "audio"), THINK_CONFIGS)
def test_reasoning_content_takes_precedence_over_reasoning(version: TokenizerVersion, image: bool, audio: bool) -> None:
    r"""When both `reasoning_content` and `reasoning` are present, `reasoning_content` wins."""
    template = generate_chat_template(
        spm=False,
        tokenizer_version=version,
        image_support=image,
        audio_support=audio,
        thinking_support=True,
    )

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Hi"},
        {
            "role": "assistant",
            "reasoning_content": "RC wins",
            "reasoning": "R loses",
            "content": "Hello!",
        },
    ]

    output = render_template(template, messages)
    assert "[THINK]RC wins[/THINK]Hello!" in output
    assert "R loses" not in output


@pytest.mark.parametrize(("version", "image", "audio"), THINK_CONFIGS)
def test_reasoning_with_existing_think_chunks(version: TokenizerVersion, image: bool, audio: bool) -> None:
    r"""`reasoning_content` is prepended before existing inline `ThinkChunk`\s."""
    template = generate_chat_template(
        spm=False,
        tokenizer_version=version,
        image_support=image,
        audio_support=audio,
        thinking_support=True,
    )

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Hi"},
        {
            "role": "assistant",
            "reasoning_content": "Top-level reasoning",
            "content": [
                {"type": "thinking", "thinking": "Inline thinking"},
                {"type": "text", "text": "Hello!"},
            ],
        },
    ]

    output = render_template(template, messages)
    assert "[THINK]Top-level reasoning[/THINK][THINK]Inline thinking[/THINK]Hello!" in output


@pytest.mark.parametrize(("version", "image", "audio"), THINK_CONFIGS)
def test_reasoning_content_only_no_text_content(version: TokenizerVersion, image: bool, audio: bool) -> None:
    r"""`reasoning_content` with empty/null text content produces just the thinking chunk."""
    template = generate_chat_template(
        spm=False,
        tokenizer_version=version,
        image_support=image,
        audio_support=audio,
        thinking_support=True,
    )

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "reasoning_content": "Just thinking...", "content": ""},
    ]

    output = render_template(template, messages)
    assert "[THINK]Just thinking...[/THINK]" in output


@pytest.mark.parametrize(("version", "image", "audio"), THINK_CONFIGS)
def test_reasoning_content_with_tool_calls(version: TokenizerVersion, image: bool, audio: bool) -> None:
    r"""`reasoning_content` is preserved alongside `tool_calls`."""
    template = generate_chat_template(
        spm=False,
        tokenizer_version=version,
        image_support=image,
        audio_support=audio,
        thinking_support=True,
    )

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "What's the weather?"},
        {
            "role": "assistant",
            "reasoning_content": "I should call a tool.",
            "content": "",
            "tool_calls": [
                {
                    "id": "abc123",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                },
            ],
        },
        {"role": "tool", "name": "get_weather", "content": "22C", "tool_call_id": "abc123"},
        {"role": "assistant", "content": "It's 22C in Paris."},
    ]

    tools: list[dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]

    output = render_template(template, messages, tools=tools)
    assert "[THINK]I should call a tool.[/THINK]" in output
    assert "get_weather" in output


@pytest.mark.parametrize(("version", "image", "audio"), THINK_CONFIGS)
def test_reasoning_aggregation_consecutive_assistants(version: TokenizerVersion, image: bool, audio: bool) -> None:
    r"""Consecutive assistant messages with `reasoning_content` aggregate correctly."""
    template = generate_chat_template(
        spm=False,
        tokenizer_version=version,
        image_support=image,
        audio_support=audio,
        thinking_support=True,
    )

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Solve this."},
        {"role": "assistant", "reasoning_content": "Step 1 reasoning", "content": "Partial answer."},
        {"role": "assistant", "reasoning_content": "Step 2 reasoning", "content": "Final answer."},
    ]

    output = render_template(template, messages)
    # Both reasoning traces should appear as thinking chunks
    assert "[THINK]Step 1 reasoning[/THINK]" in output
    assert "[THINK]Step 2 reasoning[/THINK]" in output
    # Text from both messages should be aggregated
    assert "Partial answer." in output
    assert "Final answer." in output


@pytest.mark.parametrize(("version", "image", "audio"), THINK_CONFIGS)
def test_reasoning_static_dynamic_parity(version: TokenizerVersion, image: bool, audio: bool) -> None:
    r"""Static and dynamic templates produce identical output for `reasoning_content` input."""
    config = _make_config((version, False, image, audio, True, False))
    static_template = _load_golden_template(config)
    dynamic_template = generate_chat_template(
        spm=False,
        tokenizer_version=version,
        image_support=image,
        audio_support=audio,
        thinking_support=True,
    )

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "reasoning_content": "Thinking...", "content": "Hi!"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "reasoning": "Considering the question.", "content": "I'm well!"},
    ]

    static_output = render_template(static_template, messages)
    dynamic_output = render_template(dynamic_template, messages)

    assert static_output == dynamic_output


def test_non_think_template_ignores_reasoning_field() -> None:
    r"""Non-think templates silently ignore `reasoning_content` (no conversion happens)."""
    template = generate_chat_template(
        spm=False,
        tokenizer_version=TokenizerVersion.v13,
        image_support=False,
        audio_support=False,
        thinking_support=False,
    )

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "reasoning_content": "This is ignored", "content": "Hello!"},
    ]

    output = render_template(template, messages)
    assert "[THINK]" not in output
    assert "Hello!" in output


def test_empty_reasoning_content_is_ignored() -> None:
    r"""Empty-string `reasoning_content` does not produce a thinking chunk."""
    template = generate_chat_template(
        spm=False,
        tokenizer_version=TokenizerVersion.v13,
        image_support=False,
        audio_support=False,
        thinking_support=True,
    )

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "reasoning_content": "", "content": "Hello!"},
    ]

    output = render_template(template, messages)
    assert "[THINK]" not in output
    assert "Hello!" in output


def test_none_reasoning_content_is_ignored() -> None:
    r"""`None` value for `reasoning_content` does not produce a thinking chunk."""
    template = generate_chat_template(
        spm=False,
        tokenizer_version=TokenizerVersion.v13,
        image_support=False,
        audio_support=False,
        thinking_support=True,
    )

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "reasoning_content": None, "content": "Hello!"},
    ]

    output = render_template(template, messages)
    assert "[THINK]" not in output
    assert "Hello!" in output


def test_plain_think_template_produces_correct_output() -> None:
    r"""Plain think template emits `<think>`/`</think>` tags, not `[THINK]`/`[/THINK]`."""
    template = generate_chat_template(
        spm=False,
        tokenizer_version=TokenizerVersion.v11,
        image_support=False,
        audio_support=False,
        thinking_support=False,
        plain_thinking_support=True,
    )

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Solve this"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Let me think..."},
                {"type": "text", "text": "The answer is 42."},
            ],
        },
    ]

    output = render_template(template, messages)
    assert "<think>Let me think...</think>The answer is 42." in output
    assert "[THINK]" not in output
    assert "[/THINK]" not in output


@pytest.mark.parametrize("image", [False, True])
def test_plain_think_static_dynamic_parity(image: bool) -> None:
    r"""Static and dynamic plain think templates produce identical output."""
    config = _make_config((TokenizerVersion.v11, False, image, False, False, True))
    static_template = _load_golden_template(config)
    dynamic_template = generate_chat_template(
        spm=False,
        tokenizer_version=TokenizerVersion.v11,
        image_support=image,
        audio_support=False,
        thinking_support=False,
        plain_thinking_support=True,
    )

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Hello"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Thinking..."},
                {"type": "text", "text": "Hi!"},
            ],
        },
    ]

    static_output = render_template(static_template, messages)
    dynamic_output = render_template(dynamic_template, messages)
    assert static_output == dynamic_output


def test_plain_think_reasoning_content_conversion() -> None:
    r"""`reasoning_content` produces `<think>` tags in plain think mode."""
    template = generate_chat_template(
        spm=False,
        tokenizer_version=TokenizerVersion.v11,
        image_support=False,
        audio_support=False,
        thinking_support=False,
        plain_thinking_support=True,
    )

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "reasoning_content": "Let me add.", "content": "4."},
    ]

    output = render_template(template, messages)
    assert "<think>Let me add.</think>4." in output
    assert "[THINK]" not in output


def test_plain_think_closed_false() -> None:
    r"""`closed: false` omits the `</think>` closing tag in plain think mode."""
    template = generate_chat_template(
        spm=False,
        tokenizer_version=TokenizerVersion.v11,
        image_support=False,
        audio_support=False,
        thinking_support=False,
        plain_thinking_support=True,
    )

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Hi"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Still thinking...", "closed": False},
            ],
        },
    ]

    output = render_template(template, messages)
    assert "<think>Still thinking..." in output
    assert "</think>" not in output


def test_plain_think_image_template() -> None:
    r"""Image + plain think template handles both `[IMG]` and `<think>` tags."""
    template = generate_chat_template(
        spm=False,
        tokenizer_version=TokenizerVersion.v11,
        image_support=True,
        audio_support=False,
        thinking_support=False,
        plain_thinking_support=True,
    )

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is this?"},
                {"type": "image_url", "image_url": "http://example.com/img.png"},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "It looks like..."},
                {"type": "text", "text": "A red square."},
            ],
        },
    ]

    output = render_template(template, messages)
    assert "[IMG]" in output
    assert "<think>It looks like...</think>A red square." in output
    assert "[THINK]" not in output


# ── closed attribute handling for special token think ──────────────────


@pytest.mark.parametrize(
    ("version", "image"),
    [
        (TokenizerVersion.v13, False),
        (TokenizerVersion.v13, True),
        (TokenizerVersion.v15, False),
        (TokenizerVersion.v15, True),
    ],
)
def test_closed_false_special_token_think(version: TokenizerVersion, image: bool) -> None:
    r"""`closed: false` omits `[/THINK]` for special token think templates."""
    template = generate_chat_template(
        spm=False,
        tokenizer_version=version,
        image_support=image,
        audio_support=False,
        thinking_support=True,
    )

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Hi"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Ongoing thought", "closed": False},
            ],
        },
    ]

    output = render_template(template, messages)
    assert "[THINK]Ongoing thought" in output
    assert "[/THINK]" not in output


@pytest.mark.parametrize(
    ("version", "image"),
    [
        (TokenizerVersion.v13, False),
        (TokenizerVersion.v13, True),
        (TokenizerVersion.v15, False),
        (TokenizerVersion.v15, True),
    ],
)
def test_closed_true_special_token_think(version: TokenizerVersion, image: bool) -> None:
    r"""`closed: true` (or absent) emits `[/THINK]` for special token think templates."""
    template = generate_chat_template(
        spm=False,
        tokenizer_version=version,
        image_support=image,
        audio_support=False,
        thinking_support=True,
    )

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Hi"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Done thinking"},
                {"type": "text", "text": "Hello!"},
            ],
        },
    ]

    output = render_template(template, messages)
    assert "[THINK]Done thinking[/THINK]Hello!" in output


# ── audio + system prompt error path ──────────────────────────────────


@pytest.mark.parametrize(
    "version",
    [TokenizerVersion.v7, TokenizerVersion.v11, TokenizerVersion.v13],
)
def test_audio_with_system_prompt_raises(version: TokenizerVersion) -> None:
    r"""Audio chunks are rejected when a system prompt is present (v7-v13)."""
    template = generate_chat_template(
        spm=False,
        tokenizer_version=version,
        image_support=False,
        audio_support=True,
        thinking_support=False,
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "You are helpful."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is this audio?"},
                {"type": "input_audio", "input_audio": {"data": "base64data", "format": "wav"}},
            ],
        },
        {"role": "assistant", "content": "An audio clip."},
    ]

    with pytest.raises(
        ValueError, match="Audio chunks are not supported in user message content when system prompt is provided"
    ):
        render_template(template, messages)


def test_v15_audio_with_system_prompt_succeeds() -> None:
    r"""v15 allows audio chunks even when a system prompt is present."""
    template = generate_chat_template(
        spm=False,
        tokenizer_version=TokenizerVersion.v15,
        image_support=False,
        audio_support=True,
        thinking_support=False,
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "You are helpful."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is this audio?"},
                {"type": "input_audio", "input_audio": {"data": "base64data", "format": "wav"}},
            ],
        },
        {"role": "assistant", "content": "An audio clip."},
    ]

    output = render_template(template, messages)
    assert "[AUDIO]" in output
    assert "[SYSTEM_PROMPT]" in output


@pytest.mark.parametrize("image", [False, True])
def test_v15_think_template_rejects_think_in_system(image: bool) -> None:
    r"""v15 think template raises on think chunks in system messages."""
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


def test_v1_empty_assistant_content_raises() -> None:
    r"""v1 template raises on assistant with empty content list."""
    template = generate_chat_template(
        spm=False,
        tokenizer_version=TokenizerVersion.v1,
        image_support=False,
        audio_support=False,
        thinking_support=False,
    )

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": []},
    ]

    with pytest.raises(ValueError, match="Assistant message content must be non-empty"):
        render_template(template, messages)


def test_v1_invalid_assistant_chunk_type_raises() -> None:
    r"""v1 template raises on non-text chunk in assistant message."""
    template = generate_chat_template(
        spm=False,
        tokenizer_version=TokenizerVersion.v1,
        image_support=False,
        audio_support=False,
        thinking_support=False,
    )

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": [{"type": "image", "url": "http://example.com/img.png"}]},
    ]

    with pytest.raises(ValueError, match="Only text chunks are supported in assistant message contents"):
        render_template(template, messages)


class TestV3ToolResultJsonParsing:
    """Tests for v3 (non-SPM) tool result JSON formatting edge cases."""

    @pytest.fixture()
    def v3_template(self) -> str:
        return generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v3,
            image_support=False,
            audio_support=False,
            thinking_support=False,
        )

    def test_numeric_int_tool_result(self, v3_template: str) -> None:
        r"""v3 template parses integer tool results as int (not string)."""
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "abc123def",
                        "type": "function",
                        "function": {"name": "add", "arguments": '{"a": 2, "b": 2}'},
                    }
                ],
            },
            {"role": "tool", "content": "4", "tool_call_id": "abc123def", "name": "add"},
            {"role": "assistant", "content": "4"},
        ]

        output = render_template(
            v3_template,
            messages,
            tools=[{"type": "function", "function": {"name": "add", "parameters": {}}}],
        )
        assert '"content": 4' in output

    def test_numeric_float_tool_result(self, v3_template: str) -> None:
        r"""v3 template parses float tool results as float."""
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "What is pi?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "abc123def",
                        "type": "function",
                        "function": {"name": "pi", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "content": "3.14", "tool_call_id": "abc123def", "name": "pi"},
            {"role": "assistant", "content": "3.14"},
        ]

        output = render_template(
            v3_template,
            messages,
            tools=[{"type": "function", "function": {"name": "pi", "parameters": {}}}],
        )
        assert '"content": 3.14' in output

    def test_missing_call_id_raises(self, v3_template: str) -> None:
        r"""v3 template raises when tool message has no valid call_id."""
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "abc123def",
                        "type": "function",
                        "function": {"name": "greet", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "content": "hi", "name": "greet"},
            {"role": "assistant", "content": "hi"},
        ]

        with pytest.raises(ValueError, match="call_id or tool_call_id"):
            render_template(
                v3_template,
                messages,
                tools=[{"type": "function", "function": {"name": "greet", "parameters": {}}}],
            )

    def test_invalid_length_call_id_raises(self, v3_template: str) -> None:
        r"""v3 template raises when tool result call_id is not 9 characters."""
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "abc123def",
                        "type": "function",
                        "function": {"name": "greet", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "content": "hi", "tool_call_id": "short", "name": "greet"},
            {"role": "assistant", "content": "hi"},
        ]

        with pytest.raises(ValueError, match="call_id or tool_call_id"):
            render_template(
                v3_template,
                messages,
                tools=[{"type": "function", "function": {"name": "greet", "parameters": {}}}],
            )


def test_v7_content_and_tool_calls_accepted() -> None:
    r"""v7 template accepts assistant messages with both content and tool_calls."""
    template = generate_chat_template(
        spm=False,
        tokenizer_version=TokenizerVersion.v7,
        image_support=False,
        audio_support=False,
        thinking_support=False,
    )

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "What's the weather?"},
        {
            "role": "assistant",
            "content": "Let me check the weather for you.",
            "tool_calls": [
                {
                    "id": "abc123def",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                }
            ],
        },
        {"role": "tool", "content": "Sunny, 25C", "tool_call_id": "abc123def", "name": "get_weather"},
        {"role": "assistant", "content": "It's sunny and 25C in Paris."},
    ]

    output = render_template(
        template,
        messages,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }
        ],
    )
    assert "Let me check the weather for you." in output
    assert "[TOOL_CALLS]" in output


@pytest.mark.parametrize("version", [TokenizerVersion.v13, TokenizerVersion.v15])
def test_v13_v15_accept_short_tool_call_id(version: TokenizerVersion) -> None:
    r"""v13+ templates accept tool call IDs of any length (no 9-char constraint)."""
    template = generate_chat_template(
        spm=False,
        tokenizer_version=version,
        image_support=False,
        audio_support=False,
        thinking_support=False,
    )

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Hello"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "x",
                    "type": "function",
                    "function": {"name": "greet", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "content": "hi", "tool_call_id": "x", "name": "greet"},
        {"role": "assistant", "content": "Done"},
    ]

    output = render_template(
        template,
        messages,
        tools=[{"type": "function", "function": {"name": "greet", "parameters": {}}}],
    )
    assert "greet" in output
    assert "Done" in output


def test_v1_tool_role_rejected() -> None:
    r"""v1 template raises on tool role messages."""
    template = generate_chat_template(
        spm=False,
        tokenizer_version=TokenizerVersion.v1,
        image_support=False,
        audio_support=False,
        thinking_support=False,
    )

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
        {"role": "tool", "content": "result"},
    ]

    with pytest.raises(ValueError, match="Unexpected role 'tool' after role 'assistant'"):
        render_template(template, messages)


@pytest.mark.parametrize("config_tuple", ALL_CONFIGS)
def test_valid_config_construction(config_tuple: tuple[TokenizerVersion, bool, bool, bool, bool, bool]) -> None:
    """Valid configs from ALL_CONFIGS construct without error and have accessible properties."""
    config = _make_config(config_tuple)
    assert isinstance(config.has_tools, bool)
    assert isinstance(config.any_thinking_support, bool)


def test_empty_messages_raises() -> None:
    """Empty messages list raises an error."""
    config = TemplateConfig(version=TokenizerVersion.v15)
    template = build_chat_template(config)
    with pytest.raises(Exception):
        render_template(template, messages=[])


def test_reasoning_content_with_none_content() -> None:
    """Assistant message with content=None and reasoning_content works."""
    config = TemplateConfig(version=TokenizerVersion.v15, thinking_support=True)
    template = build_chat_template(config)
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Hello"},
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "Let me think about this.",
        },
    ]
    result = render_template(template, messages=messages)
    assert "[THINK]" in result
    assert "Let me think about this." in result


def test_double_quotes_in_default_system_prompt() -> None:
    """Double quotes in default_system_prompt are preserved correctly."""
    template = generate_chat_template(
        spm=False,
        tokenizer_version=TokenizerVersion.v7,
        image_support=False,
        audio_support=False,
        thinking_support=False,
        default_system_prompt='You are "the best" assistant.',
    )
    messages: list[dict[str, Any]] = [{"role": "user", "content": "Hello"}]
    result = render_template(template, messages=messages)
    assert 'You are "the best" assistant.' in result


def test_use_token_variables_true_emits_variable_refs() -> None:
    """With use_token_variables=True, template references bos_token and eos_token variables."""
    config = TemplateConfig(version=TokenizerVersion.v7, use_token_variables=True)
    template = build_chat_template(config)
    assert "bos_token" in template
    assert "eos_token" in template
    assert "{{- '<s>' }}" not in template
    assert "{{- '</s>' }}" not in template


def test_use_token_variables_false_emits_literals() -> None:
    """With use_token_variables=False, template embeds literal token values."""
    config = TemplateConfig(version=TokenizerVersion.v7, use_token_variables=False)
    template = build_chat_template(config)
    assert "'<s>'" in template
    assert "'</s>'" in template
    assert "bos_token" not in template
    assert "eos_token" not in template


def test_use_token_variables_false_renders_correctly() -> None:
    """Template with literal tokens renders without passing bos_token/eos_token kwargs."""
    config = TemplateConfig(version=TokenizerVersion.v15, use_token_variables=False)
    template = build_chat_template(config)
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    result = render_template(template, messages=messages)
    assert "<s>" in result
    assert "</s>" in result


def test_use_token_variables_default_is_true() -> None:
    """Default use_token_variables is True."""
    config = TemplateConfig(version=TokenizerVersion.v7)
    assert config.use_token_variables is True
