"""Reusable helper functions for chat template tests.

This module contains utility functions used by test files across the
``unit/`` and ``transformers/`` subdirectories.  It intentionally has no
fixtures or test data -- those live in ``conftest.py`` and ``fixtures_data.py``.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
from jinja2 import BaseLoader
from jinja2.sandbox import ImmutableSandboxedEnvironment
from PIL import Image

try:
    from transformers.utils.chat_template_utils import render_jinja_template

    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

from mistral_common.audio import Audio
from mistral_common.integrations.chat_templates.template_generator import TemplateConfig
from mistral_common.protocol.instruct.normalize import get_normalizer
from mistral_common.protocol.instruct.request import ChatCompletionRequest, ReasoningEffort
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

# Golden template files live in the data/ tree (outside src/).
_GOLDEN_DIR = Path(__file__).parent.parent.parent / "data" / "chat_templates"

SPM_SPECIAL_WHITESPACE = "▁"
SPM_WHITESPACE = "▁"


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


def render_template(
    template: str, messages: list[Any], tools: list[Any] | None = None, reasoning_effort: str | None = None
) -> str:
    """Render a jinja2 template with the given messages."""

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


def _create_dummy_image() -> Image.Image:
    """Create a simple dummy 2x2 red square image for testing."""
    return Image.new("RGB", (2, 2), color="red")


def _sin_wave(sampling_rate: int, duration: float) -> np.ndarray:
    """Generate a sine wave numpy array."""
    return np.sin(np.ones([int(duration * sampling_rate)]))


def _sample_audio() -> Audio:
    """Create a sample Audio instance for testing."""
    sampling_rate = 44100
    original_array = _sin_wave(sampling_rate, 1)
    return Audio(
        audio_array=original_array,
        sampling_rate=sampling_rate,
        format="wav",
    )


def _get_image_encoder(tokenizer: Tokenizer) -> ImageEncoder:
    """Create an ImageEncoder for testing."""
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
    """Create an AudioEncoder for testing."""
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
    """Build a MistralTokenizer with Tekken backend."""
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
    """Build a MistralTokenizer with SentencePiece backend."""
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
