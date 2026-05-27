from typing import Any

import pytest
from jinja2.exceptions import TemplateError

from mistral_common.integrations.chat_templates.chat_templates import generate_chat_template
from mistral_common.protocol.instruct.chunk import TextChunk
from mistral_common.protocol.instruct.messages import AssistantMessage, ToolMessage, UserMessage
from mistral_common.protocol.instruct.validator import ValidationMode
from mistral_common.tokens.tokenizers.base import TokenizerVersion
from tests.integrations.chat_templates.conftest import ALL_TRANSFORMERS_CONFIGS
from tests.integrations.chat_templates.fixtures_data import _get_conversations
from tests.integrations.chat_templates.helpers import _get_mistral_tokenizer, encode_mistral_common, encode_transformers


class TestTransformersMistralCommonParity:
    @pytest.mark.parametrize(
        ("spm", "version", "image", "audio", "think"),
        ALL_TRANSFORMERS_CONFIGS,
    )
    @pytest.mark.parametrize("mode", [ValidationMode.test, ValidationMode.finetuning])
    def test_chat_template(
        self,
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

        # Consecutive assistants get aggregated: user, assistant*3, user -> user, assistant, user
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
        self,
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
