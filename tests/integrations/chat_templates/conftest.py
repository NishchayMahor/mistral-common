import json
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

try:
    from jinja2 import BaseLoader
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    _HAS_JINJA2 = True
except ImportError:
    _HAS_JINJA2 = False

try:
    from transformers.utils.chat_template_utils import render_jinja_template

    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

from mistral_common.audio import Audio
from mistral_common.integrations.chat_templates.template_generator import TemplateConfig
from mistral_common.protocol.instruct.chunk import (
    AudioChunk,
    AudioURLChunk,
    ImageChunk,
    ImageURLChunk,
    RawAudio,
    TextChunk,
    ThinkChunk,
)
from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from mistral_common.protocol.instruct.normalize import get_normalizer
from mistral_common.protocol.instruct.request import ChatCompletionRequest, ReasoningEffort
from mistral_common.protocol.instruct.tool_calls import Function, FunctionCall, Tool, ToolCall
from mistral_common.protocol.instruct.validator import ValidationMode, get_validator
from mistral_common.tokens.tokenizers.audio import (
    AudioConfig,
    AudioEncoder,
    AudioSpectrogramConfig,
    SpecialAudioIDs,
)
from mistral_common.tokens.tokenizers.base import InstructTokenizer, Tokenizer, TokenizerVersion
from mistral_common.tokens.tokenizers.image import ImageConfig, ImageEncoder, SpecialImageIDs
from mistral_common.tokens.tokenizers.instruct import (
    InstructTokenizerV1,
    InstructTokenizerV2,
    InstructTokenizerV3,
    InstructTokenizerV7,
    InstructTokenizerV11,
    InstructTokenizerV13,
    InstructTokenizerV15,
)
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.model_settings_builder import EnumBuilder, ModelSettingsBuilder
from mistral_common.tokens.tokenizers.sentencepiece import SentencePieceTokenizer
from mistral_common.tokens.tokenizers.tekken import Tekkenizer
from tests.test_tekken import get_special_tokens

# Golden template files live in the integrations/ tree (outside src/)
_GOLDEN_DIR = Path(__file__).parent.parent.parent / "data" / "chat_templates"

# All configurations for output comparison tests (including SPM).
# Tuples: (version, spm, image, audio, think, plain_think)
ALL_CONFIGS: list[tuple[TokenizerVersion, bool, bool, bool, bool, bool]] = [
    # Non-SPM
    (TokenizerVersion.v1, False, False, False, False, False),
    (TokenizerVersion.v2, False, False, False, False, False),
    (TokenizerVersion.v3, False, False, False, False, False),
    (TokenizerVersion.v3, False, True, False, False, False),
    (TokenizerVersion.v7, False, False, False, False, False),
    (TokenizerVersion.v7, False, True, False, False, False),
    (TokenizerVersion.v7, False, False, True, False, False),
    (TokenizerVersion.v11, False, False, False, False, False),
    (TokenizerVersion.v11, False, True, False, False, False),
    (TokenizerVersion.v11, False, False, True, False, False),
    (TokenizerVersion.v13, False, False, False, False, False),
    (TokenizerVersion.v13, False, True, False, False, False),
    (TokenizerVersion.v13, False, False, True, False, False),
    (TokenizerVersion.v13, False, False, False, True, False),
    (TokenizerVersion.v13, False, True, False, True, False),
    (TokenizerVersion.v15, False, False, False, False, False),
    (TokenizerVersion.v15, False, True, False, False, False),
    (TokenizerVersion.v15, False, False, True, False, False),
    (TokenizerVersion.v15, False, False, False, True, False),
    (TokenizerVersion.v15, False, True, False, True, False),
    # SPM
    (TokenizerVersion.v1, True, False, False, False, False),
    (TokenizerVersion.v2, True, False, False, False, False),
    (TokenizerVersion.v3, True, False, False, False, False),
    (TokenizerVersion.v3, True, True, False, False, False),
    (TokenizerVersion.v7, True, False, False, False, False),
    (TokenizerVersion.v7, True, True, False, False, False),
    # Plain thinking (v11 only)
    (TokenizerVersion.v11, False, False, False, False, True),
    (TokenizerVersion.v11, False, True, False, False, True),
]

# Parametrization configs for test_transformers.py, formatted as (spm, version, image, audio, think)
ALL_TRANSFORMERS_CONFIGS: list[tuple[bool, TokenizerVersion, bool, bool, bool]] = [
    (spm, version, image, audio, think)
    for version, spm, image, audio, think, plain_think in ALL_CONFIGS
    if not plain_think  # plain_think has separate tests
]


def _config_id(c: tuple[TokenizerVersion, bool, bool, bool, bool, bool]) -> str:
    """Generate a human-readable test ID for a config tuple."""
    v, spm, img, aud, think, plain = c
    parts = [v.value]
    if spm:
        parts.append("spm")
    if img:
        parts.append("img")
    if aud:
        parts.append("aud")
    if think:
        parts.append("think")
    if plain:
        parts.append("plain_think")
    return "_".join(parts)


def _make_config(c: tuple[TokenizerVersion, bool, bool, bool, bool, bool]) -> TemplateConfig:
    """Create a TemplateConfig from a config tuple."""
    version, spm, image, audio, think, plain_think = c
    return TemplateConfig(
        version=version,
        spm=spm,
        image_support=image,
        audio_support=audio,
        thinking_support=think,
        plain_thinking_support=plain_think,
    )


def _load_golden_template(config: TemplateConfig) -> str:
    """Load the static golden template for a config."""
    parts = [config.version.value]
    if config.image_support and config.any_thinking_support:
        parts.append("image_think")
    elif config.image_support:
        parts.append("image")
    elif config.audio_support:
        parts.append("audio")
    elif config.any_thinking_support:
        parts.append("think")
    if config.spm:
        parts.append("spm")
    filename = "_".join(parts) + ".jinja"
    path = _GOLDEN_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Golden template not found: {path}")
    return path.read_text()


_IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/7/78/Red_Square_%282x2_Pixel%29.png"

SPM_SPECIAL_WHITESPACE = "▁"
SPM_WHITESPACE = "▁"


def _create_dummy_image() -> Image.Image:
    """Create a simple dummy 2x2 red square image for testing."""
    return Image.new("RGB", (2, 2), color="red")


_IMAGE = _create_dummy_image()


def _sin_wave(sampling_rate: int, duration: float) -> np.ndarray:
    return np.sin(np.ones([int(duration * sampling_rate)]))


def _sample_audio() -> Audio:
    sampling_rate = 44100
    original_array = _sin_wave(sampling_rate, 1)
    return Audio(
        audio_array=original_array,
        sampling_rate=sampling_rate,
        format="wav",
    )


_AUDIO_URL = _sample_audio().to_base64("wav")
_AUDIO = RawAudio(data=_AUDIO_URL, format="wav")


@pytest.fixture(autouse=True, scope="package")
def mock_download_image() -> Generator[None, None, None]:
    """Mock the download_image function to return a dummy image for all tests."""
    with patch("mistral_common.image.download_image") as mock_download:
        mock_download.return_value = _IMAGE
        with patch("mistral_common.tokens.tokenizers.image.download_image") as mock_download2:
            mock_download2.return_value = _IMAGE
            yield


def render_template(
    template: str, messages: list[Any], tools: list[Any] | None = None, reasoning_effort: str | None = None
) -> str:
    """Render a jinja2 template with the given messages."""
    assert _HAS_JINJA2, "jinja2 is required"

    def raise_exception(msg: str) -> None:
        raise ValueError(msg)

    env = ImmutableSandboxedEnvironment(loader=BaseLoader())
    env.globals["raise_exception"] = raise_exception
    jinja_template = env.from_string(template)

    render_kwargs: dict[str, Any] = {
        "messages": messages,
        "tools": tools,
        "bos_token": "<s>",
        "eos_token": "</s>",
    }

    # Only add reasoning_effort for v15+ templates that support it
    if reasoning_effort is not None or "reasoning_effort" in template:
        render_kwargs["reasoning_effort"] = reasoning_effort

    return jinja_template.render(**render_kwargs)


def encode_mistral_common(mistral_tokenizer: MistralTokenizer, chat_request: ChatCompletionRequest, spm: bool) -> str:
    """Encode a chat request using mistral-common tokenizer."""
    mistral_encoded = str(mistral_tokenizer.encode_chat_completion(chat_request).text)
    # Remove image tokens except one per image
    mistral_encoded = mistral_encoded.replace("[IMG]", "").replace("[IMG_BREAK]", "").replace("[IMG_END]", "[IMG]")
    # Remove audio tokens except one per audio
    mistral_encoded = mistral_encoded.replace("[AUDIO]", "").replace("[BEGIN_AUDIO]", "[AUDIO]")
    if spm:
        mistral_encoded = (
            mistral_encoded.replace(SPM_SPECIAL_WHITESPACE, " ").replace(SPM_WHITESPACE, " ").replace("<0x0A>", "\n")
        )
    return mistral_encoded


def encode_transformers(
    chat_template: str, chat_request: ChatCompletionRequest | dict[str, Any], keep_name_for_tools: bool = False
) -> str:
    """Encode a chat request using the transformers render_jinja_template."""
    assert _HAS_TRANSFORMERS, "transformers is required"
    if isinstance(chat_request, ChatCompletionRequest):
        openai_request = chat_request.to_openai()
        if keep_name_for_tools:
            for openai_message, chat_message in zip(openai_request["messages"], chat_request.messages):
                if chat_message.role == "tool":
                    openai_message["name"] = chat_message.name
    else:
        openai_request = chat_request
    for tool in openai_request.get("tools", []):
        tool["function"].pop("strict", False)

    reasoning_effort = openai_request.get("reasoning_effort")
    template_kwargs: dict[str, Any] = {}
    if reasoning_effort is not None:
        template_kwargs["reasoning_effort"] = reasoning_effort

    encoded = render_jinja_template(
        [openai_request["messages"]],
        tools=openai_request.get("tools", None),
        chat_template=chat_template,
        bos_token="<s>",
        eos_token="</s>",
        **template_kwargs,
    )[0][0]
    assert isinstance(encoded, str), type(encoded)
    return encoded


def _get_image_encoder(tokenizer: Tokenizer) -> ImageEncoder:
    image_config = ImageConfig(image_patch_size=2, max_image_size=10, spatial_merge_size=1)
    return ImageEncoder(
        image_config=image_config,
        special_ids=SpecialImageIDs(
            img=tokenizer.get_special_token("[IMG]"),
            img_break=tokenizer.get_special_token("[IMG_BREAK]"),
            img_end=tokenizer.get_special_token("[IMG_END]"),
        ),
    )


def _get_audio_encoder() -> AudioEncoder:
    audio_config = AudioConfig(
        sampling_rate=24_000,
        frame_rate=12.5,
        encoding_config=AudioSpectrogramConfig(
            num_mel_bins=128,
            window_size=400,
            hop_length=160,
        ),
    )
    return AudioEncoder(
        audio_config=audio_config,
        special_ids=SpecialAudioIDs(audio=24, begin_audio=25, streaming_pad=26, text_to_audio=27, audio_to_text=28),
    )


def _get_mistral_tekkenizer(
    tokenizer_version: TokenizerVersion, validation_mode: ValidationMode, image: bool, audio: bool, think: bool
) -> MistralTokenizer:
    special_tokens = get_special_tokens(tokenizer_version=tokenizer_version, add_audio=audio, add_think=think)
    with open(MistralTokenizer._data_path() / "tekken_240911.json", "r", encoding="utf-8") as f:
        json_tekkenizer = json.load(f)
    vocab = json_tekkenizer["vocab"]
    vocab_size = json_tekkenizer["config"]["default_vocab_size"]
    pattern = json_tekkenizer["config"]["pattern"]
    model_settings_builder = (
        ModelSettingsBuilder(
            reasoning_effort=EnumBuilder(
                accepts_none=True, default=ReasoningEffort.none, values=[ReasoningEffort.none, ReasoningEffort.high]
            )
        )
        if tokenizer_version.supports_model_settings
        else None
    )
    tokenizer = Tekkenizer(
        vocab,
        special_tokens,
        pattern=pattern,
        vocab_size=vocab_size,
        num_special_tokens=100,
        version=tokenizer_version,
        model_settings_builder=model_settings_builder,
    )

    audio_encoder = _get_audio_encoder() if audio else None
    image_encoder = _get_image_encoder(tokenizer) if image else None

    if tokenizer_version == TokenizerVersion.v1:
        instruct_cls = InstructTokenizerV1
    elif tokenizer_version == TokenizerVersion.v2:
        instruct_cls = InstructTokenizerV2
    elif tokenizer_version == TokenizerVersion.v3:
        instruct_cls = InstructTokenizerV3
    elif tokenizer_version == TokenizerVersion.v7:
        instruct_cls = InstructTokenizerV7
    elif tokenizer_version == TokenizerVersion.v11:
        instruct_cls = InstructTokenizerV11
    elif tokenizer_version == TokenizerVersion.v13:
        instruct_cls = InstructTokenizerV13
    elif tokenizer_version == TokenizerVersion.v15:
        instruct_cls = InstructTokenizerV15
    else:
        raise ValueError(f"Unknown tokenizer version: {tokenizer_version}")

    instruct_tokenizer: InstructTokenizer = instruct_cls(
        tokenizer=tokenizer,
        image_encoder=image_encoder,
        audio_encoder=audio_encoder,
    )
    model_settings_builder = tokenizer.model_settings_builder if isinstance(tokenizer, Tekkenizer) else None
    return MistralTokenizer(
        instruct_tokenizer,
        validator=get_validator(mode=validation_mode, version=tokenizer_version),
        request_normalizer=get_normalizer(version=tokenizer_version, model_settings_builder=model_settings_builder),
    )


def _get_mistral_sentencepiece(
    tokenizer_version: TokenizerVersion, validation_mode: ValidationMode, image: bool
) -> MistralTokenizer:
    tokenizer = SentencePieceTokenizer(
        MistralTokenizer._data_path() / "mistral_instruct_tokenizer_241114.model.v7m1",
        tokenizer_version,
    )

    image_encoder = _get_image_encoder(tokenizer) if image else None

    if tokenizer_version == TokenizerVersion.v1:
        instruct_cls = InstructTokenizerV1
    elif tokenizer_version == TokenizerVersion.v2:
        instruct_cls = InstructTokenizerV2
    elif tokenizer_version == TokenizerVersion.v3:
        instruct_cls = InstructTokenizerV3
    elif tokenizer_version == TokenizerVersion.v7:
        instruct_cls = InstructTokenizerV7
    else:
        raise ValueError(f"Unknown tokenizer version: {tokenizer_version}")

    instruct_tokenizer: InstructTokenizer = instruct_cls(
        tokenizer=tokenizer,
        image_encoder=image_encoder,
    )
    return MistralTokenizer(
        instruct_tokenizer,
        validator=get_validator(mode=validation_mode, version=tokenizer_version),
        request_normalizer=get_normalizer(version=tokenizer_version),
    )


def _get_mistral_tokenizer(
    spm: bool,
    tokenizer_version: TokenizerVersion,
    validation_mode: ValidationMode,
    image: bool,
    audio: bool,
    think: bool,
) -> MistralTokenizer:
    """Get a mistral tokenizer instance for testing."""
    if spm:
        return _get_mistral_sentencepiece(tokenizer_version, validation_mode, image)
    else:
        return _get_mistral_tekkenizer(tokenizer_version, validation_mode, image, audio, think)


# -- Request fixtures --

REQUEST_ONE_TURN_TEST = ChatCompletionRequest(
    messages=[
        UserMessage(content="User says hello"),
    ]
)

REQUEST_ONE_TURN_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        UserMessage(content="User says hello"),
        AssistantMessage(content="Assistant says hi"),
    ]
)

REQUEST_ONE_TURN_WITH_SYSTEM_TEST = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="User says hello"),
    ]
)

REQUEST_ONE_TURN_WITH_SYSTEM_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="User says hello"),
        AssistantMessage(content="Assistant says hi"),
    ]
)

REQUEST_MULTI_TURN_TEST = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        UserMessage(content="User says hello"),
        AssistantMessage(content="Assistant says hi"),
        UserMessage(content="User says how are you ?"),
    ]
)

REQUEST_MULTI_TURN_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        UserMessage(content="User says hello"),
        AssistantMessage(content="Assistant says hi"),
        UserMessage(content="User says how are you ?"),
        AssistantMessage(content="Assistant says hi"),
    ]
)

REQUEST_MULTI_TURN_WITH_SYSTEM_TEST = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="User says hello"),
        AssistantMessage(content="Assistant says hi"),
        UserMessage(content="User says how are you ?"),
    ]
)

REQUEST_MULTI_TURN_WITH_SYSTEM_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="User says hello"),
        AssistantMessage(content="Assistant says hi"),
        UserMessage(content="User says how are you ?"),
        AssistantMessage(content="Assistant says hi"),
    ]
)

REQUEST_MULTI_TURN_WITH_TOOLS_TEST = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="User says hello"),
        AssistantMessage(content="Assistant says hi"),
        UserMessage(content="User says how are you ?"),
    ],
    tools=[
        Tool(
            function=Function(
                name="tool1",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                            "required": ["location"],
                        }
                    },
                },
            )
        ),
        Tool(function=Function(name="tool2", parameters={})),
    ],
)

REQUEST_MULTI_TURN_WITH_TOOLS_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="User says hello"),
        AssistantMessage(content="Assistant says hi"),
        UserMessage(content="User says how are you ?"),
        AssistantMessage(content="Assistant says hi"),
    ],
    tools=[
        Tool(
            function=Function(
                name="tool1",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                            "required": ["location"],
                        }
                    },
                },
            )
        ),
        Tool(function=Function(name="tool2", parameters={})),
    ],
)

REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TEST = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="User says hello"),
        AssistantMessage(
            content=None,
            tool_calls=[
                ToolCall(
                    id="123456789",
                    function=FunctionCall(
                        name="tool1",
                        arguments={  # type: ignore[arg-type]
                            "location": "San Francisco, CA",
                        },
                    ),
                ),
            ],
        ),
        ToolMessage(content="32", tool_call_id="123456789"),
        AssistantMessage(content="Whether is 32 degrees in San Francisco, CA"),
        UserMessage(content=[TextChunk(text="User says thanks.")]),
    ],
    tools=[
        Tool(
            function=Function(
                name="tool1",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                            "required": ["location"],
                        }
                    },
                },
            )
        ),
        Tool(function=Function(name="tool2", parameters={})),
    ],
)

REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TEST_2 = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="User says hello"),
        AssistantMessage(content="Whether is 32 degrees in San Francisco, CA"),
        UserMessage(content=[TextChunk(text="User says thanks.")]),
        AssistantMessage(content=[TextChunk(text="Assistant says you're welcome.")]),
        UserMessage(content="bye"),
        AssistantMessage(
            content=None,
            tool_calls=[
                ToolCall(
                    id="123456789",
                    function=FunctionCall(
                        name="tool1",
                        arguments={  # type: ignore[arg-type]
                            "location": "San Francisco, CA",
                        },
                    ),
                ),
                ToolCall(
                    id="023456789",
                    function=FunctionCall(
                        name="tool2",
                        arguments={},  # type: ignore[arg-type]
                    ),
                ),
            ],
        ),
        ToolMessage(content="32", tool_call_id="123456789"),
    ],
    tools=[
        Tool(
            function=Function(
                name="tool1",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                            "required": ["location"],
                        }
                    },
                },
            )
        ),
        Tool(function=Function(name="tool2", parameters={})),
    ],
)

REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="User says hello"),
        AssistantMessage(
            content=None,
            tool_calls=[
                ToolCall(
                    id="123456789",
                    function=FunctionCall(
                        name="tool1",
                        arguments={  # type: ignore[arg-type]
                            "location": "San Francisco, CA",
                        },
                    ),
                ),
            ],
        ),
        ToolMessage(content="32", tool_call_id="123456789"),
        AssistantMessage(content="Whether is 32 degrees in San Francisco, CA"),
        UserMessage(content=[TextChunk(text="User says thanks.")]),
        AssistantMessage(content=[TextChunk(text="Assistant says you're welcome.")]),
    ],
    tools=[
        Tool(
            function=Function(
                name="tool1",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                            "required": ["location"],
                        }
                    },
                },
            )
        ),
        Tool(function=Function(name="tool2", parameters={})),
    ],
)

REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TRAIN_2 = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="User says hello"),
        AssistantMessage(content="Whether is 32 degrees in San Francisco, CA"),
        UserMessage(content=[TextChunk(text="User says thanks.")]),
        AssistantMessage(content=[TextChunk(text="Assistant says you're welcome.")]),
        UserMessage(content="bye"),
        AssistantMessage(
            content=None,
            tool_calls=[
                ToolCall(
                    id="123456789",
                    function=FunctionCall(
                        name="tool1",
                        arguments={  # type: ignore[arg-type]
                            "location": "San Francisco, CA",
                        },
                    ),
                ),
                ToolCall(
                    id="023456789",
                    function=FunctionCall(
                        name="tool2",
                        arguments={},  # type: ignore[arg-type]
                    ),
                ),
            ],
        ),
        ToolMessage(content="32", tool_call_id="123456789"),
        ToolMessage(content="aya", tool_call_id="023456789"),
        AssistantMessage(content="wow 32", tool_calls=[]),
    ],
    tools=[
        Tool(
            function=Function(
                name="tool1",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                            "required": ["location"],
                        }
                    },
                },
            )
        ),
        Tool(function=Function(name="tool2", parameters={})),
    ],
)

REQUEST_MULTI_TURN_WITH_CONTENT_AND_TOOLS_CALLS_TEST = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="User says hello"),
        AssistantMessage(content="Whether is 32 degrees in San Francisco, CA"),
        UserMessage(content=[TextChunk(text="User says thanks.")]),
        AssistantMessage(content=[TextChunk(text="Assistant says you're welcome.")]),
        UserMessage(content="bye"),
        AssistantMessage(
            content="Assistant says hi, let me fetch the weather for you.",
            tool_calls=[
                ToolCall(
                    id="123456789",
                    function=FunctionCall(
                        name="tool1",
                        arguments={  # type: ignore[arg-type]
                            "location": "San Francisco, CA",
                        },
                    ),
                ),
                ToolCall(
                    id="023456789",
                    function=FunctionCall(
                        name="tool2",
                        arguments={},  # type: ignore[arg-type]
                    ),
                ),
            ],
        ),
        ToolMessage(content="32", tool_call_id="123456789"),
    ],
    tools=[
        Tool(
            function=Function(
                name="tool1",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                            "required": ["location"],
                        }
                    },
                },
            )
        ),
        Tool(function=Function(name="tool2", parameters={})),
    ],
)

REQUEST_MULTI_TURN_WITH_CONTENT_AND_TOOLS_CALLS_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="User says hello"),
        AssistantMessage(content="Whether is 32 degrees in San Francisco, CA"),
        UserMessage(content=[TextChunk(text="User says thanks.")]),
        AssistantMessage(content=[TextChunk(text="Assistant says you're welcome.")]),
        UserMessage(content="bye"),
        AssistantMessage(
            content="Assistant says hi, let me fetch the weather for you.",
            tool_calls=[
                ToolCall(
                    id="123456789",
                    function=FunctionCall(
                        name="tool1",
                        arguments={  # type: ignore[arg-type]
                            "location": "San Francisco, CA",
                        },
                    ),
                ),
                ToolCall(
                    id="023456789",
                    function=FunctionCall(
                        name="tool2",
                        arguments={},  # type: ignore[arg-type]
                    ),
                ),
            ],
        ),
        ToolMessage(content="32", tool_call_id="123456789"),
        ToolMessage(content="aya", tool_call_id="023456789"),
        AssistantMessage(content="wow 32", tool_calls=[]),
    ],
    tools=[
        Tool(
            function=Function(
                name="tool1",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                            "required": ["location"],
                        }
                    },
                },
            )
        ),
        Tool(function=Function(name="tool2", parameters={})),
    ],
)

REQUEST_TOOL_THEN_USER_TEST = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="User says hello"),
        AssistantMessage(
            content=None,
            tool_calls=[
                ToolCall(
                    id="123456789",
                    function=FunctionCall(
                        name="tool1",
                        arguments={  # type: ignore[arg-type]
                            "location": "San Francisco, CA",
                        },
                    ),
                ),
            ],
        ),
        ToolMessage(content="32", tool_call_id="123456789"),
        UserMessage(content="What does that mean?"),
    ],
    tools=[
        Tool(
            function=Function(
                name="tool1",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                            "required": ["location"],
                        }
                    },
                },
            )
        ),
        Tool(function=Function(name="tool2", parameters={})),
    ],
)

REQUEST_TOOL_THEN_USER_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="User says hello"),
        AssistantMessage(
            content=None,
            tool_calls=[
                ToolCall(
                    id="123456789",
                    function=FunctionCall(
                        name="tool1",
                        arguments={  # type: ignore[arg-type]
                            "location": "San Francisco, CA",
                        },
                    ),
                ),
            ],
        ),
        ToolMessage(content="32", tool_call_id="123456789"),
        UserMessage(content="What does that mean?"),
        AssistantMessage(content="The temperature is 32 degrees in San Francisco."),
    ],
    tools=[
        Tool(
            function=Function(
                name="tool1",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                            "required": ["location"],
                        }
                    },
                },
            )
        ),
        Tool(function=Function(name="tool2", parameters={})),
    ],
)

REQUEST_TOOL_THEN_USER_FULL_LOOP_TEST = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="User says hello"),
        AssistantMessage(
            content=None,
            tool_calls=[
                ToolCall(
                    id="123456789",
                    function=FunctionCall(
                        name="tool1",
                        arguments={  # type: ignore[arg-type]
                            "location": "San Francisco, CA",
                        },
                    ),
                ),
                ToolCall(
                    id="023456789",
                    function=FunctionCall(
                        name="tool2",
                        arguments={},  # type: ignore[arg-type]
                    ),
                ),
            ],
        ),
        ToolMessage(content="32", tool_call_id="123456789"),
        ToolMessage(content="sunny", tool_call_id="023456789"),
        UserMessage(content="Now what about Tokyo?"),
        AssistantMessage(
            content=None,
            tool_calls=[
                ToolCall(
                    id="234567890",
                    function=FunctionCall(
                        name="tool1",
                        arguments={  # type: ignore[arg-type]
                            "location": "Tokyo, JP",
                        },
                    ),
                ),
            ],
        ),
        ToolMessage(content="28", tool_call_id="234567890"),
        AssistantMessage(content="San Francisco is 32 and sunny, Tokyo is 28."),
        UserMessage(content="Thanks!"),
    ],
    tools=[
        Tool(
            function=Function(
                name="tool1",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                            "required": ["location"],
                        }
                    },
                },
            )
        ),
        Tool(function=Function(name="tool2", parameters={})),
    ],
)

REQUEST_TOOL_THEN_USER_FULL_LOOP_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="User says hello"),
        AssistantMessage(
            content=None,
            tool_calls=[
                ToolCall(
                    id="123456789",
                    function=FunctionCall(
                        name="tool1",
                        arguments={  # type: ignore[arg-type]
                            "location": "San Francisco, CA",
                        },
                    ),
                ),
                ToolCall(
                    id="023456789",
                    function=FunctionCall(
                        name="tool2",
                        arguments={},  # type: ignore[arg-type]
                    ),
                ),
            ],
        ),
        ToolMessage(content="32", tool_call_id="123456789"),
        ToolMessage(content="sunny", tool_call_id="023456789"),
        UserMessage(content="Now what about Tokyo?"),
        AssistantMessage(
            content=None,
            tool_calls=[
                ToolCall(
                    id="234567890",
                    function=FunctionCall(
                        name="tool1",
                        arguments={  # type: ignore[arg-type]
                            "location": "Tokyo, JP",
                        },
                    ),
                ),
            ],
        ),
        ToolMessage(content="28", tool_call_id="234567890"),
        AssistantMessage(content="San Francisco is 32 and sunny, Tokyo is 28."),
    ],
    tools=[
        Tool(
            function=Function(
                name="tool1",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                            "required": ["location"],
                        }
                    },
                },
            )
        ),
        Tool(function=Function(name="tool2", parameters={})),
    ],
)

REQUEST_TOOL_THEN_USER_WITH_CONTENT_TEST = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="User says hello"),
        AssistantMessage(
            content="Let me check the weather for you.",
            tool_calls=[
                ToolCall(
                    id="123456789",
                    function=FunctionCall(
                        name="tool1",
                        arguments={  # type: ignore[arg-type]
                            "location": "San Francisco, CA",
                        },
                    ),
                ),
                ToolCall(
                    id="023456789",
                    function=FunctionCall(
                        name="tool2",
                        arguments={},  # type: ignore[arg-type]
                    ),
                ),
            ],
        ),
        ToolMessage(content="32", tool_call_id="123456789"),
        ToolMessage(content="sunny", tool_call_id="023456789"),
        UserMessage(content="What does that mean?"),
    ],
    tools=[
        Tool(
            function=Function(
                name="tool1",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                            "required": ["location"],
                        }
                    },
                },
            )
        ),
        Tool(function=Function(name="tool2", parameters={})),
    ],
)

REQUEST_TOOL_THEN_USER_WITH_CONTENT_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="User says hello"),
        AssistantMessage(
            content="Let me check the weather for you.",
            tool_calls=[
                ToolCall(
                    id="123456789",
                    function=FunctionCall(
                        name="tool1",
                        arguments={  # type: ignore[arg-type]
                            "location": "San Francisco, CA",
                        },
                    ),
                ),
                ToolCall(
                    id="023456789",
                    function=FunctionCall(
                        name="tool2",
                        arguments={},  # type: ignore[arg-type]
                    ),
                ),
            ],
        ),
        ToolMessage(content="32", tool_call_id="123456789"),
        ToolMessage(content="sunny", tool_call_id="023456789"),
        UserMessage(content="What does that mean?"),
        AssistantMessage(content="It is 32 degrees and sunny in San Francisco."),
    ],
    tools=[
        Tool(
            function=Function(
                name="tool1",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                            "required": ["location"],
                        }
                    },
                },
            )
        ),
        Tool(function=Function(name="tool2", parameters={})),
    ],
)

REQUEST_MULTI_TURN_IMAGE_URL_TEST = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(
            content=[
                TextChunk(text="User asks what is this image ?"),
                ImageURLChunk(image_url=_IMAGE_URL),
                ImageURLChunk(image_url=_IMAGE_URL),
            ]
        ),
        AssistantMessage(content="Assistant answers It is a red square."),
        UserMessage(content=[TextChunk(text="User says thanks.")]),
    ]
)

REQUEST_MULTI_TURN_IMAGE_URL_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(
            content=[
                TextChunk(text="User asks what is this image ?"),
                ImageURLChunk(image_url=_IMAGE_URL),
                ImageURLChunk(image_url=_IMAGE_URL),
            ]
        ),
        AssistantMessage(content="Assistant answers It is a red square."),
        UserMessage(content=[TextChunk(text="User says thanks.")]),
        AssistantMessage(content=[TextChunk(text="Assistant says you're welcome.")]),
    ]
)

REQUEST_MULTI_TURN_IMAGE_TEST = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(
            content=[
                TextChunk(text="User asks what is this image ?"),
                ImageChunk(image=_IMAGE),
            ]
        ),
        AssistantMessage(content="Assistant answers It is a red square."),
        UserMessage(content=[TextChunk(text="User says thanks.")]),
    ]
)

REQUEST_MULTI_TURN_IMAGE_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(
            content=[
                TextChunk(text="User asks what is this image ?"),
                ImageChunk(image=_IMAGE),
            ]
        ),
        AssistantMessage(content="Assistant answers It is a red square."),
        UserMessage(content=[TextChunk(text="User says thanks.")]),
        AssistantMessage(content=[TextChunk(text="Assistant says you're welcome.")]),
    ]
)

REQUEST_MULTI_TURN_AUDIO_URL_TEST = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        UserMessage(
            content=[
                TextChunk(text="Users asks what is this audio ?"),
                AudioURLChunk(audio_url=_AUDIO_URL),
                AudioURLChunk(audio_url=_AUDIO_URL),
            ]
        ),
        AssistantMessage(content="Assistant answers it is a music."),
        UserMessage(content=[TextChunk(text="User says thanks.")]),
    ]
)

REQUEST_MULTI_TURN_AUDIO_URL_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        UserMessage(
            content=[
                TextChunk(text="Users asks what is this audio ?"),
                AudioURLChunk(audio_url=_AUDIO_URL),
            ]
        ),
        AssistantMessage(content="Assistant answers it is a music."),
        UserMessage(content=[TextChunk(text="User says thanks.")]),
        AssistantMessage(content=[TextChunk(text="Assistant says you're welcome.")]),
    ]
)

REQUEST_MULTI_TURN_AUDIO_TEST = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        UserMessage(
            content=[
                TextChunk(text="Users asks what is this audio ?"),
                AudioChunk(input_audio=_AUDIO),
            ]
        ),
        AssistantMessage(content="Assistant answers it is a music."),
        UserMessage(content=[TextChunk(text="User says thanks.")]),
    ]
)

REQUEST_MULTI_TURN_AUDIO_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        UserMessage(
            content=[
                TextChunk(text="Users asks what is this audio ?"),
                AudioChunk(input_audio=_AUDIO),
                AudioChunk(input_audio=_AUDIO),
            ]
        ),
        AssistantMessage(content="Assistant answers it is a music."),
        UserMessage(content=[TextChunk(text="User says thanks.")]),
        AssistantMessage(content=[TextChunk(text="Assistant says you're welcome.")]),
    ]
)

REQUEST_MULTI_TURN_IMAGE_AND_THINKING_TEST = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(
            content=[
                TextChunk(text="You are a helpful assistant that can think."),
                ThinkChunk(thinking="You need to think here."),
                TextChunk(text="Here you need to answer."),
            ],
        ),
        UserMessage(
            content=[
                TextChunk(text="User asks what is this image ?"),
                ImageURLChunk(image_url=_IMAGE_URL),
            ]
        ),
        AssistantMessage(
            content=[
                TextChunk(text="Assistant says wow I need to think."),
                ThinkChunk(thinking="Assistant thinks it's a red square."),
                TextChunk(text="Assistant says it is a red square."),
            ],
            tool_calls=[],
        ),
        UserMessage(content=[TextChunk(text="User says thanks.")]),
    ],
)

REQUEST_MULTI_TURN_THINKING_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(
            content=[
                TextChunk(text="You are a helpful assistant that can think."),
                ThinkChunk(thinking="You need to think here."),
                TextChunk(text="Here you need to answer."),
            ],
        ),
        UserMessage(
            content=[
                TextChunk(text="User asks what is this image ?"),
            ]
        ),
        AssistantMessage(
            content=[
                TextChunk(text="Assistant says wow I need to think."),
                ThinkChunk(thinking="Assistant thinks it's a red square."),
                TextChunk(text="Assistant says it is a red square."),
            ],
            tool_calls=[],
        ),
        UserMessage(content=[TextChunk(text="User says thanks.")]),
        AssistantMessage(content=[TextChunk(text="Assistant says you're welcome.")]),
    ],
)

REQUEST_MULTI_TURN_THINKING_TEST = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(
            content=[
                TextChunk(text="You are a helpful assistant that can think."),
                ThinkChunk(thinking="You need to think here."),
                TextChunk(text="Here you need to answer."),
            ],
        ),
        UserMessage(
            content=[
                TextChunk(text="User asks what is this image ?"),
            ]
        ),
        AssistantMessage(
            content=[
                TextChunk(text="Assistant says wow I need to think."),
                ThinkChunk(thinking="Assistant thinks it's a red square."),
                TextChunk(text="Assistant says it is a red square."),
            ],
            tool_calls=[],
        ),
        UserMessage(content=[TextChunk(text="User says thanks.")]),
    ],
)

REQUEST_MULTI_TURN_IMAGE_AND_THINKING_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(
            content=[
                TextChunk(text="You are a helpful assistant that can think."),
                ThinkChunk(thinking="You need to think here."),
                TextChunk(text="Here you need to answer."),
            ],
        ),
        UserMessage(
            content=[
                TextChunk(text="User asks what is this image ?"),
                ImageURLChunk(image_url=_IMAGE_URL),
            ]
        ),
        AssistantMessage(
            content=[
                TextChunk(text="Assistant says wow I need to think."),
                ThinkChunk(thinking="Assistant thinks it's a red square."),
                TextChunk(text="Assistant says it is a red square."),
            ],
            tool_calls=[],
        ),
        UserMessage(content=[TextChunk(text="User says thanks.")]),
        AssistantMessage(content=[TextChunk(text="Assistant says you're welcome.")]),
    ],
)


# -- Message aggregation test fixtures --

REQUEST_CONSECUTIVE_USERS_TEST = ChatCompletionRequest(
    messages=[
        UserMessage(content="Hello"),
        UserMessage(content="World"),
    ]
)

REQUEST_CONSECUTIVE_USERS_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        UserMessage(content="Hello"),
        UserMessage(content="World"),
        AssistantMessage(content="Hi there"),
    ]
)

REQUEST_CONSECUTIVE_USERS_WITH_SYSTEM_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(content="You are helpful."),
        UserMessage(content="Hello"),
        UserMessage(content="World"),
        AssistantMessage(content="Hi there"),
    ]
)

REQUEST_CONSECUTIVE_ASSISTANTS_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        UserMessage(content="Hello"),
        AssistantMessage(content="Hi"),
        AssistantMessage(content="How can I help?"),
        UserMessage(content="Thanks"),
        AssistantMessage(content="You're welcome"),
    ]
)

REQUEST_MULTIPLE_SYSTEMS_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(content="System prompt 1."),
        SystemMessage(content="System prompt 2."),
        UserMessage(content="Hello"),
        AssistantMessage(content="Hi"),
    ]
)

REQUEST_CONSECUTIVE_USERS_IMAGE_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        UserMessage(content="What is this?"),
        UserMessage(
            content=[
                ImageChunk(image=_IMAGE),
                TextChunk(text="Describe it"),
            ]
        ),
        AssistantMessage(content="It's an image."),
    ]
)

# -- Multi-chunk aggregation test fixtures --

REQUEST_CONSECUTIVE_USERS_TEXT_CHUNKS_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        UserMessage(content="First as string"),
        UserMessage(content=[TextChunk(text="Second as chunk")]),
        UserMessage(content=[TextChunk(text="Third part A"), TextChunk(text="Third part B")]),
        AssistantMessage(content="Response"),
    ]
)

REQUEST_CONSECUTIVE_USERS_MULTI_IMAGE_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        UserMessage(content=[TextChunk(text="Describe this"), ImageChunk(image=_IMAGE), TextChunk(text="What color?")]),
        UserMessage(content=[TextChunk(text="Also this"), ImageChunk(image=_IMAGE), TextChunk(text="What shape?")]),
        AssistantMessage(content="Both are red squares."),
    ]
)

REQUEST_CONSECUTIVE_USERS_AUDIO_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        UserMessage(
            content=[
                TextChunk(text="Listen to this"),
                AudioURLChunk(audio_url=_AUDIO_URL),
                TextChunk(text="What language?"),
            ]
        ),
        UserMessage(
            content=[
                TextChunk(text="And this"),
                AudioURLChunk(audio_url=_AUDIO_URL),
                TextChunk(text="Transcribe it"),
            ]
        ),
        AssistantMessage(content="Both are in English."),
    ]
)

REQUEST_CONSECUTIVE_ASSISTANTS_THINK_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        UserMessage(content="Solve this problem"),
        AssistantMessage(
            content=[
                TextChunk(text="Hmm."),
                ThinkChunk(thinking="Let me think..."),
                TextChunk(text="I need more context."),
            ]
        ),
        AssistantMessage(
            content=[
                TextChunk(text="OK."),
                ThinkChunk(thinking="Now I understand."),
                TextChunk(text="The answer is 42."),
            ]
        ),
        UserMessage(content="Thanks"),
        AssistantMessage(content="You're welcome"),
    ]
)

REQUEST_CONSECUTIVE_ASSISTANTS_TOOL_CALLS_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="What's the weather?"),
        AssistantMessage(content="Let me check."),
        AssistantMessage(
            content="Fetching data.",
            tool_calls=[
                ToolCall(
                    id="123456789",
                    function=FunctionCall(name="tool1", arguments={"location": "Paris"}),  # type: ignore[arg-type]
                ),
                ToolCall(
                    id="023456789",
                    function=FunctionCall(name="tool1", arguments={"location": "London"}),  # type: ignore[arg-type]
                ),
            ],
        ),
        ToolMessage(content="22", tool_call_id="123456789"),
        ToolMessage(content="15", tool_call_id="023456789"),
        AssistantMessage(content="Paris: 22, London: 15"),
        UserMessage(content="Thanks"),
        AssistantMessage(content="Welcome"),
    ],
    tools=[
        Tool(
            function=Function(
                name="tool1",
                parameters={"type": "object", "properties": {"location": {"type": "string"}}},
            )
        ),
    ],
)

REQUEST_SYSTEM_TEXT_CHUNKS_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(content=[TextChunk(text="You are helpful."), TextChunk(text="Be concise.")]),
        UserMessage(content="Hello"),
        AssistantMessage(content="Hi"),
    ]
)

REQUEST_CONSECUTIVE_SYSTEMS_THINK_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(content=[TextChunk(text="Rule A"), TextChunk(text="Rule B"), ThinkChunk(thinking="Think 1")]),
        SystemMessage(content=[ThinkChunk(thinking="Think 2"), TextChunk(text="Rule C"), TextChunk(text="Rule D")]),
        UserMessage(content="Hello"),
        AssistantMessage(content="Hi"),
    ]
)

REQUEST_MID_CONV_SYSTEM_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        UserMessage(content="Hello"),
        SystemMessage(content="New instruction."),
        AssistantMessage(content="Got it"),
    ]
)

REQUEST_MID_CONV_SYSTEM_WITH_CONSECUTIVE_USERS_TRAIN = ChatCompletionRequest(  # type: ignore[type-var]
    messages=[
        SystemMessage(content="Be helpful."),
        UserMessage(content="Hello"),
        UserMessage(content="World"),
        SystemMessage(content="Now be concise."),
        AssistantMessage(content="Got it"),
    ]
)


def _get_conversations(
    tokenizer_version: TokenizerVersion,
    validation_mode: ValidationMode,
    image: bool,
    audio: bool,
    think: bool,
) -> list[ChatCompletionRequest]:
    """Build a list of test conversations for the given configuration."""
    conversations: list[ChatCompletionRequest] = (
        [
            REQUEST_ONE_TURN_TEST,
            REQUEST_ONE_TURN_WITH_SYSTEM_TEST,
            REQUEST_MULTI_TURN_TEST,
            REQUEST_MULTI_TURN_WITH_SYSTEM_TEST,
        ]
        if validation_mode == ValidationMode.test
        else [
            REQUEST_ONE_TURN_TRAIN,
            REQUEST_ONE_TURN_WITH_SYSTEM_TRAIN,
            REQUEST_MULTI_TURN_TRAIN,
            REQUEST_MULTI_TURN_WITH_SYSTEM_TRAIN,
        ]
    )

    if tokenizer_version > TokenizerVersion.v1:
        if validation_mode == ValidationMode.test:
            conversations.extend(
                [
                    REQUEST_MULTI_TURN_WITH_TOOLS_TEST,
                    REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TEST,
                    REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TEST_2,
                    REQUEST_TOOL_THEN_USER_TEST,
                    REQUEST_TOOL_THEN_USER_FULL_LOOP_TEST,
                ]
            )
        else:
            conversations.extend(
                [
                    REQUEST_MULTI_TURN_WITH_TOOLS_TRAIN,
                    REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TRAIN,
                    REQUEST_MULTI_TURN_WITH_TOOLS_CALLS_TRAIN_2,
                    REQUEST_TOOL_THEN_USER_TRAIN,
                    REQUEST_TOOL_THEN_USER_FULL_LOOP_TRAIN,
                ]
            )
    if tokenizer_version > TokenizerVersion.v7:
        if validation_mode == ValidationMode.test:
            conversations.extend(
                [
                    REQUEST_MULTI_TURN_WITH_CONTENT_AND_TOOLS_CALLS_TEST,
                    REQUEST_TOOL_THEN_USER_WITH_CONTENT_TEST,
                ]
            )
        else:
            conversations.extend(
                [
                    REQUEST_MULTI_TURN_WITH_CONTENT_AND_TOOLS_CALLS_TRAIN,
                    REQUEST_TOOL_THEN_USER_WITH_CONTENT_TRAIN,
                ]
            )

    if image:
        if validation_mode == ValidationMode.test:
            conversations.extend([REQUEST_MULTI_TURN_IMAGE_URL_TEST, REQUEST_MULTI_TURN_IMAGE_TEST])
        else:
            conversations.extend([REQUEST_MULTI_TURN_IMAGE_URL_TRAIN, REQUEST_MULTI_TURN_IMAGE_TRAIN])

    if audio:
        if validation_mode == ValidationMode.test:
            conversations.extend([REQUEST_MULTI_TURN_AUDIO_URL_TEST, REQUEST_MULTI_TURN_AUDIO_TEST])
        else:
            conversations.extend([REQUEST_MULTI_TURN_AUDIO_URL_TRAIN, REQUEST_MULTI_TURN_AUDIO_TRAIN])

    if think:
        if validation_mode == ValidationMode.test:
            conversations.extend([REQUEST_MULTI_TURN_THINKING_TEST])
        else:
            conversations.extend([REQUEST_MULTI_TURN_THINKING_TRAIN])

    if image and think:
        if validation_mode == ValidationMode.test:
            conversations.extend([REQUEST_MULTI_TURN_IMAGE_AND_THINKING_TEST])
        else:
            conversations.extend([REQUEST_MULTI_TURN_IMAGE_AND_THINKING_TRAIN])

    # Message aggregation test fixtures (finetuning only since last msg must be assistant)
    if validation_mode == ValidationMode.finetuning:
        conversations.extend(
            [
                REQUEST_CONSECUTIVE_USERS_TRAIN,
                REQUEST_CONSECUTIVE_USERS_WITH_SYSTEM_TRAIN,
                REQUEST_CONSECUTIVE_ASSISTANTS_TRAIN,
                REQUEST_MULTIPLE_SYSTEMS_TRAIN,
            ]
        )
        if tokenizer_version >= TokenizerVersion.v3:
            conversations.extend(
                [
                    REQUEST_CONSECUTIVE_USERS_TEXT_CHUNKS_TRAIN,
                    REQUEST_SYSTEM_TEXT_CHUNKS_TRAIN,
                ]
            )
        if image:
            conversations.extend(
                [
                    REQUEST_CONSECUTIVE_USERS_IMAGE_TRAIN,
                    REQUEST_CONSECUTIVE_USERS_MULTI_IMAGE_TRAIN,
                ]
            )
        if audio:
            conversations.append(REQUEST_CONSECUTIVE_USERS_AUDIO_TRAIN)
        if think:
            conversations.extend(
                [
                    REQUEST_CONSECUTIVE_ASSISTANTS_THINK_TRAIN,
                    REQUEST_CONSECUTIVE_SYSTEMS_THINK_TRAIN,
                ]
            )
    else:
        conversations.append(REQUEST_CONSECUTIVE_USERS_TEST)

    # v7+ only: mid-conversation system messages and combined aggregation scenarios
    if tokenizer_version >= TokenizerVersion.v7 and validation_mode == ValidationMode.finetuning:
        conversations.extend(
            [
                REQUEST_MID_CONV_SYSTEM_TRAIN,
                REQUEST_MID_CONV_SYSTEM_WITH_CONSECUTIVE_USERS_TRAIN,
                REQUEST_CONSECUTIVE_ASSISTANTS_TOOL_CALLS_TRAIN,
            ]
        )

    conversations = [c.model_copy(deep=True) for c in conversations]

    if think and tokenizer_version >= TokenizerVersion.v15:
        for conv in conversations:
            for message in conv.messages:
                if isinstance(message, SystemMessage) and isinstance(message.content, list):
                    message.content = [
                        TextChunk(text="\n".join([c.text for c in message.content if isinstance(c, TextChunk)]))
                    ]

    return conversations
