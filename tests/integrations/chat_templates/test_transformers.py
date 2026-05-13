"""Layer 3: Integration tests comparing transformers rendering vs mistral-common tokenizer.

These tests require `transformers` to be installed (test-integrations dependency group).
"""

from __future__ import annotations

from typing import Any

import pytest

transformers = pytest.importorskip("transformers")

from jinja2.exceptions import TemplateError  # noqa: E402

from mistral_common.integrations.chat_templates.chat_templates import generate_chat_template  # noqa: E402
from mistral_common.protocol.instruct.chunk import TextChunk, ThinkChunk  # noqa: E402
from mistral_common.protocol.instruct.messages import (  # noqa: E402
    AssistantMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from mistral_common.protocol.instruct.request import ChatCompletionRequest  # noqa: E402
from mistral_common.protocol.instruct.validator import ValidationMode  # noqa: E402
from mistral_common.tokens.tokenizers.base import TokenizerVersion  # noqa: E402
from tests.integrations.chat_templates.conftest import (  # noqa: E402
    ALL_TRANSFORMERS_CONFIGS,
    _get_conversations,
    _get_mistral_tokenizer,
    encode_mistral_common,
    encode_transformers,
)


@pytest.mark.parametrize(
    ("spm", "version", "image", "audio", "think"),
    ALL_TRANSFORMERS_CONFIGS,
)
@pytest.mark.parametrize("mode", [ValidationMode.test, ValidationMode.finetuning])
def test_chat_template(
    spm: bool,
    version: TokenizerVersion,
    mode: ValidationMode,
    image: bool,
    audio: bool,
    think: bool,
) -> None:
    conversations = _get_conversations(version, mode, image, audio, think)

    mistral_tokenizer = _get_mistral_tokenizer(
        spm=spm, tokenizer_version=version, validation_mode=mode, image=image, audio=audio, think=think
    )
    chat_template = generate_chat_template(
        spm=spm, tokenizer_version=version, image_support=image, audio_support=audio, thinking_support=think
    )
    if version <= TokenizerVersion.v2:
        for conv in conversations:
            for message in conv.messages:
                if isinstance(message, (UserMessage, AssistantMessage)) and isinstance(message.content, list):
                    assert len(message.content) == 1 and isinstance(message.content[0], TextChunk), (
                        "Only text content is supported for v1 and v2"
                    )
                    message.content = str(message.content[0].text)
    for conversation in conversations:
        if version == TokenizerVersion.v2:
            for message in conversation.messages:
                if isinstance(message, ToolMessage):
                    message.name = "tool"
        # Run transformers first since encode_mistral_common may mutate the conversation in-place
        transformers_encoded = encode_transformers(
            chat_template, conversation, keep_name_for_tools=version == TokenizerVersion.v2
        )
        mistral_common_encoded = encode_mistral_common(mistral_tokenizer, conversation, spm)

        assert mistral_common_encoded == transformers_encoded


@pytest.mark.parametrize(
    ("spm", "version", "image", "audio", "think"),
    ALL_TRANSFORMERS_CONFIGS,
)
def test_role_error(
    spm: bool,
    version: TokenizerVersion,
    image: bool,
    audio: bool,
    think: bool,
) -> None:
    chat_template = generate_chat_template(
        spm=spm, tokenizer_version=version, image_support=image, audio_support=audio, thinking_support=think
    )

    # Consecutive user messages should be aggregated (not raise an error)
    VALID_CONSECUTIVE_USERS = {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "World"},
            {"role": "assistant", "content": "Hi"},
        ]
    }
    # This should not raise
    encode_transformers(chat_template, VALID_CONSECUTIVE_USERS)

    # Consecutive assistants get aggregated: user, assistant*3, user → user, assistant, user
    VALID_CONSECUTIVE_ASSISTANTS = {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "assistant", "content": "Help?"},
            {"role": "assistant", "content": "More"},
            {"role": "user", "content": "Thanks"},
        ]
    }
    encode_transformers(chat_template, VALID_CONSECUTIVE_ASSISTANTS)

    # Starting with assistant is rejected by the first-message constraint
    INVALID_STARTS_WITH_ASSISTANT = {
        "messages": [
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Hello"},
        ]
    }

    if version >= TokenizerVersion.v7:
        first_msg_match = r"Conversation must start with a user or system message, got assistant\."
    else:
        first_msg_match = r"Conversation must start with a user message, got assistant\."

    with pytest.raises(TemplateError, match=first_msg_match):
        encode_transformers(chat_template, INVALID_STARTS_WITH_ASSISTANT)

    # Invalid role after user is caught by the transition table
    INVALID_ROLE = {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "invalid", "content": "Hello"},
        ]
    }

    with pytest.raises(TemplateError, match=r"Unexpected role 'invalid' after role 'user'"):
        encode_transformers(chat_template, INVALID_ROLE)

    # Tool after user is rejected by the transition table (tool can only follow assistant or tool)
    if version >= TokenizerVersion.v2:
        INVALID_TOOL_AFTER_USER = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "tool", "content": "result", "tool_call_id": "123456789"},
            ]
        }
        with pytest.raises(TemplateError, match=r"Unexpected role 'tool' after role 'user'"):
            encode_transformers(chat_template, INVALID_TOOL_AFTER_USER)

    # User after tool is accepted (user can follow tool results)
    if version >= TokenizerVersion.v2:
        VALID_USER_AFTER_TOOL: dict[str, Any] = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "123456789", "function": {"name": "func", "arguments": "{}"}}],
                },
                {"role": "tool", "content": "result", "tool_call_id": "123456789"},
                {"role": "user", "content": "continue with this context"},
            ]
        }
        encode_transformers(chat_template, VALID_USER_AFTER_TOOL)

    # System after assistant is rejected for v7+ (system stays in loop_messages)
    if version >= TokenizerVersion.v7:
        INVALID_SYSTEM_AFTER_ASSISTANT = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
                {"role": "system", "content": "New system prompt"},
                {"role": "user", "content": "World"},
            ]
        }
        with pytest.raises(TemplateError, match=r"Unexpected role 'system' after role 'assistant'"):
            encode_transformers(chat_template, INVALID_SYSTEM_AFTER_ASSISTANT)


@pytest.mark.parametrize(
    ("spm", "version", "image", "audio", "think"),
    ALL_TRANSFORMERS_CONFIGS,
)
def test_invalid_chunks(
    spm: bool,
    version: TokenizerVersion,
    image: bool,
    audio: bool,
    think: bool,
) -> None:
    INVALID_SP_THINK = {
        "messages": [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "think", "thinking": "Hello"},
                ],
            }
        ]
    }

    INVALID_SP_RANDOM = {
        "messages": [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "random", "random": "Hello"},
                ],
            }
        ]
    }

    INVALID_ASSISTANT_THINK = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "think", "thinking": "Hello"},
                ],
            },
        ]
    }

    INVALID_ASSISTANT_RANDOM = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "random", "random": "Hello"},
                ],
            },
        ]
    }

    INVALID_USER_IMAGE = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "image", "image_url": "Hello"},
                ],
            }
        ]
    }

    INVALID_USER_AUDIO = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "audio", "audio_url": "Hello"},
                ],
            }
        ]
    }

    INVALID_USER_RANDOM = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "random", "random": "Hello"},
                ],
            }
        ]
    }

    SP_INVALIDS = [INVALID_SP_RANDOM, INVALID_SP_THINK]
    ASSISTANT_INVALIDS = [INVALID_ASSISTANT_RANDOM, INVALID_ASSISTANT_THINK]
    USER_INVALIDS = [INVALID_USER_IMAGE, INVALID_USER_AUDIO, INVALID_USER_RANDOM]

    invalid_convs = [INVALID_SP_RANDOM, INVALID_USER_RANDOM, INVALID_ASSISTANT_RANDOM]
    if not think:
        invalid_convs += [INVALID_SP_THINK, INVALID_ASSISTANT_THINK]
    if not image:
        invalid_convs += [INVALID_USER_IMAGE]
    if not audio:
        invalid_convs += [INVALID_USER_AUDIO]

    chat_template = generate_chat_template(
        spm=spm, tokenizer_version=version, image_support=image, audio_support=audio, thinking_support=think
    )
    for conv in invalid_convs:
        msg_template = "Only {chunks} chunks are supported in {role} message content."
        if conv in SP_INVALIDS:
            chunks = "text and thinking" if think and version < TokenizerVersion.v15 else "text"
            role = "system"
        elif conv in USER_INVALIDS:
            chunks = "text"
            if image:
                chunks += ", image and image_url"
            if audio:
                chunks += ", input_audio and audio_url"
            role = "user"
        elif conv in ASSISTANT_INVALIDS:
            chunks = "text and thinking" if think else "text"
            role = "assistant"

        err_msg = msg_template.format(chunks=chunks, role=role)
        with pytest.raises(TemplateError, match=err_msg):
            encode_transformers(chat_template, conv)


@pytest.mark.parametrize(
    ("spm", "version", "image", "audio", "think"),
    [
        (False, TokenizerVersion.v3, False, False, False),
        (True, TokenizerVersion.v3, False, False, False),
        (False, TokenizerVersion.v3, True, False, False),
        (True, TokenizerVersion.v3, True, False, False),
        (False, TokenizerVersion.v7, False, False, False),
        (True, TokenizerVersion.v7, False, False, False),
        (False, TokenizerVersion.v7, True, False, False),
        (True, TokenizerVersion.v7, True, False, False),
        (False, TokenizerVersion.v7, False, True, False),
        (False, TokenizerVersion.v11, False, False, False),
        (False, TokenizerVersion.v11, True, False, False),
        (False, TokenizerVersion.v11, False, True, False),
    ],
)
def test_tool_call_errors(
    spm: bool,
    version: TokenizerVersion,
    image: bool,
    audio: bool,
    think: bool,
) -> None:
    invalid_id_conv = {
        "messages": [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "1", "function": {"name": "func", "arguments": "{}"}}],
            },
        ]
    }

    chat_template = generate_chat_template(
        spm=spm, tokenizer_version=version, image_support=image, audio_support=audio, thinking_support=think
    )
    with pytest.raises(TemplateError, match="Tool call must have an id of 9 characters or numbers."):
        encode_transformers(chat_template, invalid_id_conv)


@pytest.mark.parametrize(
    ("spm", "version", "image", "audio", "think"),
    [
        (False, TokenizerVersion.v2, False, False, False),
        (True, TokenizerVersion.v2, False, False, False),
        (False, TokenizerVersion.v3, False, False, False),
        (True, TokenizerVersion.v3, False, False, False),
        (False, TokenizerVersion.v3, True, False, False),
        (True, TokenizerVersion.v3, True, False, False),
    ],
)
def test_invalid_assistant(
    spm: bool,
    version: TokenizerVersion,
    image: bool,
    audio: bool,
    think: bool,
) -> None:
    invalid_message_conv = {
        "messages": [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": "hey",
                "tool_calls": [{"id": "123456789", "function": {"name": "func", "arguments": "{}"}}],
            },
        ]
    }

    chat_template = generate_chat_template(
        spm=spm, tokenizer_version=version, image_support=image, audio_support=audio, thinking_support=think
    )
    with pytest.raises(TemplateError, match="Assistant message cannot have both content and tool calls."):
        encode_transformers(chat_template, invalid_message_conv)


class TestDefaultSystemPrompt:
    """Tests for the default_system_prompt parameter in generate_chat_template.

    There are two different system prompt handling methods:
    1. Legacy style (v1-v3): System message injected into user messages
    2. Modern style (v7+): Uses [SYSTEM_PROMPT]...[/SYSTEM_PROMPT] tokens

    Tests cover representative versions for each method.
    """

    DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant."

    CONV_NO_SYSTEM = {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
    }

    CONV_WITH_SYSTEM = {
        "messages": [
            {"role": "system", "content": "You are a custom assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
    }

    def test_legacy_style_default_system_prompt_used(self) -> None:
        chat_template = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v3,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=self.DEFAULT_SYSTEM_PROMPT,
        )

        output = encode_transformers(chat_template, self.CONV_NO_SYSTEM)

        assert self.DEFAULT_SYSTEM_PROMPT in output
        assert f"[INST]{self.DEFAULT_SYSTEM_PROMPT}\n\nHello[/INST]" in output

    def test_legacy_style_default_system_prompt_ignored_when_provided(self) -> None:
        chat_template = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v3,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=self.DEFAULT_SYSTEM_PROMPT,
        )

        output = encode_transformers(chat_template, self.CONV_WITH_SYSTEM)

        assert self.DEFAULT_SYSTEM_PROMPT not in output
        assert "You are a custom assistant." in output

    def test_legacy_style_no_default_system_prompt(self) -> None:
        chat_template = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v3,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=None,
        )

        output = encode_transformers(chat_template, self.CONV_NO_SYSTEM)

        assert "[INST]Hello[/INST]" in output

    def test_modern_style_default_system_prompt_used(self) -> None:
        chat_template = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v7,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=self.DEFAULT_SYSTEM_PROMPT,
        )

        output = encode_transformers(chat_template, self.CONV_NO_SYSTEM)

        assert f"[SYSTEM_PROMPT]{self.DEFAULT_SYSTEM_PROMPT}[/SYSTEM_PROMPT]" in output

    def test_modern_style_default_system_prompt_ignored_when_provided(self) -> None:
        chat_template = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v7,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=self.DEFAULT_SYSTEM_PROMPT,
        )

        output = encode_transformers(chat_template, self.CONV_WITH_SYSTEM)

        assert self.DEFAULT_SYSTEM_PROMPT not in output
        assert "[SYSTEM_PROMPT]You are a custom assistant.[/SYSTEM_PROMPT]" in output

    def test_modern_style_no_default_system_prompt(self) -> None:
        chat_template = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v7,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=None,
        )

        output = encode_transformers(chat_template, self.CONV_NO_SYSTEM)

        assert "[SYSTEM_PROMPT]" not in output
        assert "[/SYSTEM_PROMPT]" not in output

    def test_legacy_spm_style_default_system_prompt(self) -> None:
        """Test default system prompt with SPM tokenizer (v3 SPM style)."""
        chat_template = generate_chat_template(
            spm=True,
            tokenizer_version=TokenizerVersion.v3,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=self.DEFAULT_SYSTEM_PROMPT,
        )

        output = encode_transformers(chat_template, self.CONV_NO_SYSTEM)

        assert self.DEFAULT_SYSTEM_PROMPT in output
        assert f"[INST] {self.DEFAULT_SYSTEM_PROMPT}\n\nHello[/INST]" in output

    def test_modern_spm_style_default_system_prompt(self) -> None:
        chat_template = generate_chat_template(
            spm=True,
            tokenizer_version=TokenizerVersion.v7,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=self.DEFAULT_SYSTEM_PROMPT,
        )

        output = encode_transformers(chat_template, self.CONV_NO_SYSTEM)

        assert f"[SYSTEM_PROMPT]{self.DEFAULT_SYSTEM_PROMPT}[/SYSTEM_PROMPT]" in output

    def test_v11_style_default_system_prompt_used(self) -> None:
        """v11 uses default system prompt when no system message provided."""
        chat_template = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v11,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=self.DEFAULT_SYSTEM_PROMPT,
        )
        output = encode_transformers(chat_template, self.CONV_NO_SYSTEM)
        assert f"[SYSTEM_PROMPT]{self.DEFAULT_SYSTEM_PROMPT}[/SYSTEM_PROMPT]" in output

    def test_v11_style_default_system_prompt_ignored_when_provided(self) -> None:
        """v11 ignores default system prompt when user provides one."""
        chat_template = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v11,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=self.DEFAULT_SYSTEM_PROMPT,
        )
        output = encode_transformers(chat_template, self.CONV_WITH_SYSTEM)
        assert self.DEFAULT_SYSTEM_PROMPT not in output
        assert "You are a custom assistant." in output

    def test_v13_style_default_system_prompt_used(self) -> None:
        """v13 uses default system prompt when no system message provided."""
        chat_template = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v13,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=self.DEFAULT_SYSTEM_PROMPT,
        )
        output = encode_transformers(chat_template, self.CONV_NO_SYSTEM)
        assert f"[SYSTEM_PROMPT]{self.DEFAULT_SYSTEM_PROMPT}[/SYSTEM_PROMPT]" in output

    def test_v13_style_default_system_prompt_ignored_when_provided(self) -> None:
        """v13 ignores default system prompt when user provides one."""
        chat_template = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v13,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=self.DEFAULT_SYSTEM_PROMPT,
        )
        output = encode_transformers(chat_template, self.CONV_WITH_SYSTEM)
        assert self.DEFAULT_SYSTEM_PROMPT not in output
        assert "You are a custom assistant." in output

    def test_special_characters_in_system_prompt(self) -> None:
        special_prompt = "You're a helpful assistant! Use \"quotes\" and 'apostrophes'."
        chat_template = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v7,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=special_prompt,
        )

        output = encode_transformers(chat_template, self.CONV_NO_SYSTEM)

        assert special_prompt in output

    def test_v15_style_default_system_prompt(self) -> None:
        """v15 default system prompt works alongside MODEL_SETTINGS."""
        template = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v15,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=self.DEFAULT_SYSTEM_PROMPT,
        )

        result = encode_transformers(template, self.CONV_NO_SYSTEM)
        assert self.DEFAULT_SYSTEM_PROMPT in result
        assert "[SYSTEM_PROMPT]" in result
        assert "[MODEL_SETTINGS]" in result

        result_with_system = encode_transformers(template, self.CONV_WITH_SYSTEM)
        assert self.DEFAULT_SYSTEM_PROMPT not in result_with_system
        assert "You are a custom assistant." in result_with_system

    def test_backslash_in_system_prompt(self) -> None:
        """Backslashes in default system prompt are preserved through escaping."""
        prompt = "use \\ for escaping"
        template = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v7,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=prompt,
        )
        result = encode_transformers(template, self.CONV_NO_SYSTEM)
        assert prompt in result

    def test_newline_in_system_prompt(self) -> None:
        """Newlines in default system prompt are preserved."""
        prompt = "Line 1\nLine 2\nLine 3"
        template = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v7,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=prompt,
        )
        result = encode_transformers(template, self.CONV_NO_SYSTEM)
        assert "Line 1\nLine 2\nLine 3" in result

    def test_backslash_digit_in_system_prompt(self) -> None:
        """Backslash-digit sequences (regex backreferences) are preserved."""
        prompt = r"Step \1: initialize, Step \2: run"
        template = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v7,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=prompt,
        )
        result = encode_transformers(template, self.CONV_NO_SYSTEM)
        assert prompt in result

    def test_jinja_syntax_in_system_prompt(self) -> None:
        """Jinja2 template syntax in default system prompt is literal, not executed."""
        prompt = "Use {{ variable }} and {% if true %}block{% endif %}"
        template = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v7,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=prompt,
        )
        result = encode_transformers(template, self.CONV_NO_SYSTEM)
        assert prompt in result


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
    spm: bool,
    version: TokenizerVersion,
    image: bool,
    audio: bool,
    think: bool,
) -> None:
    """Test that reasoning_effort must be either 'none' or 'high' for v15 templates."""
    chat_template = generate_chat_template(
        spm=spm, tokenizer_version=version, image_support=image, audio_support=audio, thinking_support=think
    )

    # Test valid reasoning_effort values — v15 always emits [MODEL_SETTINGS]
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
            # No reasoning_effort — template defaults to 'none'
        },
    ]

    for conv in valid_conversations:
        result = encode_transformers(chat_template, conv)  # type: ignore
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


class TestGenerateChatTemplatePlainThinkingValidation:
    r"""Validation tests for the `plain_thinking_support` parameter."""

    def test_plain_thinking_with_thinking_raises(self) -> None:
        r"""`plain_thinking_support` and `thinking_support` are mutually exclusive."""
        with pytest.raises(ValueError, match="Plain thinking support and thinking support are mutually exclusive"):
            generate_chat_template(
                spm=False,
                tokenizer_version=TokenizerVersion.v13,
                image_support=False,
                audio_support=False,
                thinking_support=True,
                plain_thinking_support=True,
            )

    def test_plain_thinking_non_v11_raises(self) -> None:
        r"""`plain_thinking_support` only works with v11."""
        with pytest.raises(ValueError, match="Plain thinking support is only available for tokenizer version v11"):
            generate_chat_template(
                spm=False,
                tokenizer_version=TokenizerVersion.v15,
                image_support=False,
                audio_support=False,
                thinking_support=False,
                plain_thinking_support=True,
            )

    def test_plain_thinking_with_audio_raises(self) -> None:
        r"""`plain_thinking_support` and `audio_support` are mutually exclusive."""
        with pytest.raises(ValueError, match="Audio and plain thinking support are mutually exclusive"):
            generate_chat_template(
                spm=False,
                tokenizer_version=TokenizerVersion.v11,
                image_support=False,
                audio_support=True,
                thinking_support=False,
                plain_thinking_support=True,
            )

    def test_plain_thinking_v11_works(self) -> None:
        r"""`plain_thinking_support` with v11 returns a valid template."""
        template = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v11,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            plain_thinking_support=True,
        )
        assert "<think>" in template
        assert "</think>" in template
        assert "[THINK]" not in template


@pytest.mark.parametrize(
    ("image", "mode"),
    [
        (False, ValidationMode.test),
        (False, ValidationMode.finetuning),
        (True, ValidationMode.test),
        (True, ValidationMode.finetuning),
    ],
)
def test_plain_think_vs_mistral_common_baseline(image: bool, mode: ValidationMode) -> None:
    r"""v11 plain think template matches mistral-common v11 on non-think conversations.

    Verifies that the v11 plain think Jinja template produces identical output
    to the v11 mistral-common tokenizer for all standard conversations. This
    ensures adding thinking support did not break the base v11 template.
    """
    version = TokenizerVersion.v11
    conversations = _get_conversations(version, mode, image, audio=False, think=False)

    mistral_tok = _get_mistral_tokenizer(
        spm=False, tokenizer_version=version, validation_mode=mode, image=image, audio=False, think=False
    )
    plain_think_template = generate_chat_template(
        spm=False,
        tokenizer_version=version,
        image_support=image,
        audio_support=False,
        thinking_support=False,
        plain_thinking_support=True,
    )

    for conversation in conversations:
        plain_think_encoded = encode_transformers(plain_think_template, conversation)
        mistral_encoded = encode_mistral_common(mistral_tok, conversation, spm=False)

        assert plain_think_encoded == mistral_encoded


@pytest.mark.parametrize("image", [False, True])
def test_plain_think_vs_mistral_common_with_thinking(image: bool) -> None:
    r"""v11 plain think template renders thinking chunks as `<think>`/`</think>`.

    Uses the v11 mistral-common tokenizer on a stripped version of the
    conversation (think chunks removed) as the baseline, then verifies
    that the v11 plain think template output equals the baseline with
    `<think>...</think>` tags inserted at the expected positions.
    """
    version = TokenizerVersion.v11

    mistral_tok = _get_mistral_tokenizer(
        spm=False,
        tokenizer_version=version,
        validation_mode=ValidationMode.finetuning,
        image=image,
        audio=False,
        think=False,
    )
    plain_think_template = generate_chat_template(
        spm=False,
        tokenizer_version=version,
        image_support=image,
        audio_support=False,
        thinking_support=False,
        plain_thinking_support=True,
    )

    # Build a conversation with thinking chunks
    think_conversation = ChatCompletionRequest(  # type: ignore[type-var]
        messages=[
            SystemMessage(
                content=[
                    TextChunk(text="You are a helpful assistant."),
                    ThinkChunk(thinking="System reasoning."),
                    TextChunk(text="Be concise."),
                ],
            ),
            UserMessage(content=[TextChunk(text="What is 2+2?")]),
            AssistantMessage(
                content=[
                    ThinkChunk(thinking="Simple arithmetic."),
                    TextChunk(text="4."),
                ],
                tool_calls=[],
            ),
            UserMessage(content=[TextChunk(text="Thanks.")]),
            AssistantMessage(content=[TextChunk(text="You're welcome.")]),
        ],
    )

    # Same conversation without thinking chunks
    no_think_conversation = ChatCompletionRequest(  # type: ignore[type-var]
        messages=[
            SystemMessage(content="You are a helpful assistant.Be concise."),
            UserMessage(content=[TextChunk(text="What is 2+2?")]),
            AssistantMessage(content="4.", tool_calls=[]),
            UserMessage(content=[TextChunk(text="Thanks.")]),
            AssistantMessage(content=[TextChunk(text="You're welcome.")]),
        ],
    )

    # Render with plain think template (has <think> tags)
    plain_think_encoded = encode_transformers(plain_think_template, think_conversation)

    # Render baseline without think chunks via mistral-common
    baseline_encoded = encode_mistral_common(mistral_tok, no_think_conversation, spm=False)

    # Verify think tags are present
    assert "<think>System reasoning.</think>" in plain_think_encoded
    assert "<think>Simple arithmetic.</think>" in plain_think_encoded
    assert "[THINK]" not in plain_think_encoded

    # Verify stripping think tags recovers the baseline
    stripped = plain_think_encoded.replace("<think>System reasoning.</think>", "").replace(
        "<think>Simple arithmetic.</think>", ""
    )
    assert stripped == baseline_encoded


@pytest.mark.parametrize("image", [False, True])
@pytest.mark.parametrize("mode", [ValidationMode.test, ValidationMode.finetuning])
def test_plain_think_comprehensive(image: bool, mode: ValidationMode) -> None:
    r"""v11 plain think template produces valid output for all conversation types."""
    version = TokenizerVersion.v11
    # Use non-thinking conversations (plain think uses same base conversations)
    conversations = _get_conversations(version, mode, image, audio=False, think=False)

    mistral_tok = _get_mistral_tokenizer(
        spm=False,
        tokenizer_version=version,
        validation_mode=mode,
        image=image,
        audio=False,
        think=False,
    )
    plain_think_template = generate_chat_template(
        spm=False,
        tokenizer_version=version,
        image_support=image,
        audio_support=False,
        thinking_support=False,
        plain_thinking_support=True,
    )

    for conversation in conversations:
        # Plain think template with non-think conversations should match v11 base
        plain_encoded = encode_transformers(plain_think_template, conversation)
        mistral_encoded = encode_mistral_common(mistral_tok, conversation, spm=False)
        assert plain_encoded == mistral_encoded
