from __future__ import annotations

import re
import os
import tempfile

from typing import Optional, Union, List, Dict, Any, TYPE_CHECKING
from typing_extensions import Literal, Self

from .util import (
    audio_to_bct_tensor,
    audio_write,
    concatenate_audio,
    generate_id,
    is_numpy_array,
    is_torch_tensor,
    load_config,
    logger,
    seed_everything,
    simplify_quotations,
    trim_silence,
    maybe_use_tqdm
)

if TYPE_CHECKING:
    import torch
    import numpy as np
    from vocos import Vocos # type: ignore[import-untyped]
    from .models import ConditionalFlowMatching 
    from .hinting import AudioType, AudioResultType, SeedType

__all__ = ["F5TTSPipeline"]

AUDIO_OUTPUT_FORMAT_LITERAL = Literal["wav", "mp3", "ogg", "flac", "float", "int"]

class F5TTSPipeline:
    """
    Generate speech from text.
    """
    sample_rate: int = 24000
    hop_length: int = 256

    def __init__(
        self,
        vocoder: Vocos,
        model: ConditionalFlowMatching,
        save_dir: Optional[str]=None
    ) -> None:
        """
        Initialize the speech synthesis task.

        :param vocoder: The vocoder model to use for synthesis.
        :param model: The model to use for synthesis.
        """
        self.vocoder = vocoder
        self.model = model
        self.save_dir = save_dir if save_dir is not None else os.getcwd()

    @classmethod
    def from_pretrained(
        cls,
        directory_or_repository: str="benjamin-paine/fairytaler",
        device: Optional[Union[str, int, torch.device]]="auto",
        variant: Optional[Literal["fp16", "fp32"]]=None,
        local_dir: Optional[str]=None, # For HF hub download
        cache_dir: Optional[str]=None, # For HF hub download
        save_dir: Optional[str]=None, # For saving audio outputs
    ) -> F5TTSPipeline:
        """
        Load the speech synthesis pipeline from a pretrained model.
        """
        from transformers.modeling_utils import no_init_weights # type: ignore[import-untyped]
        from accelerate import load_checkpoint_and_dispatch # type: ignore[import-untyped]
        from huggingface_hub import snapshot_download # type: ignore[import-untyped]
        from vocos import Vocos

        from .models import ConditionalFlowMatching

        # Download the model if it's a HF hub model
        if not os.path.isdir(directory_or_repository):
            allow_patterns: Optional[List[str]] = None
            if variant == "fp16":
                allow_patterns = [
                    "**/model.fp16.safetensors",
                    "**/config.yaml",
                    "vocab.txt"
                ]
            directory_or_repository = snapshot_download(
                directory_or_repository,
                cache_dir=cache_dir,
                local_dir=local_dir,
                allow_patterns=allow_patterns
            )

        variant_ext = "" if variant != "fp16" else ".fp16"

        # Gather required files
        vocab_file = os.path.join(directory_or_repository, "vocab.txt")
        assert os.path.exists(vocab_file), f"Vocabulary file {vocab_file} not found."

        vocoder_dir = os.path.join(directory_or_repository, "vocoder")
        vocoder_config_file = os.path.join(vocoder_dir, "config.yaml")
        assert os.path.exists(vocoder_config_file), f"Vocoder config file {vocoder_config_file} not found."
        vocoder_model_file = os.path.join(vocoder_dir, f"model{variant_ext}.safetensors")
        assert os.path.exists(vocoder_model_file), f"Vocoder model file {vocoder_model_file} not found."

        transformer_dir = os.path.join(directory_or_repository, "transformer")
        transformer_config_file = os.path.join(transformer_dir, "config.yaml")
        assert os.path.exists(transformer_config_file), f"Transformer config file {transformer_config_file} not found."
        transformer_model_file = os.path.join(transformer_dir, f"model{variant_ext}.safetensors")
        assert os.path.exists(transformer_model_file), f"Transformer model file {transformer_model_file} not found."

        # Load transformer config
        transformer_config = load_config(transformer_config_file)

        # Load transformer vocab
        vocab: Dict[str, int] = {}
        with open(vocab_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                vocab[line[:-1]] = i

        transformer_config["vocab_char_map"] = vocab

        # Initialize models
        with no_init_weights():
            vocoder = Vocos.from_hparams(vocoder_config_file)
            model = ConditionalFlowMatching.dit(**transformer_config)

        # Get device map
        device_map: Optional[Union[str, Dict[str, int]]] = None
        if device == "auto":
            device_map = "auto"
        elif device is not None:
            device_map = {}
            if isinstance(device, int):
                device_map[""] = device
            elif isinstance(device, str):
                if ":" in device:
                    device_id = int(device.split(":")[1])
                else:
                    device_id = 0
                device_map[""] = device_id
            elif isinstance(device, torch.device):
                device_map[""] = device.index

        # Load state dicts
        vocoder = load_checkpoint_and_dispatch(vocoder, checkpoint=vocoder_model_file, device_map=device_map)
        vocoder.eval()
        model = load_checkpoint_and_dispatch(model, checkpoint=transformer_model_file, device_map=device_map)
        model.eval()

        return cls(vocoder, model)

    @property
    def device(self) -> torch.device:
        """
        Get the device of the model.
        """
        return self.model.device

    @property
    def dtype(self) -> torch.dtype:
        """
        Get the data type of the model.
        """
        return self.model.dtype

    def to(self, *args: Any, **kwargs: Any) -> Self:
        """
        Move the model to a new device.
        """
        self.vocoder.to(*args, **kwargs)
        self.model.to(*args, **kwargs)
        return self

    def ends_with_multi_byte_character(self, text: str) -> bool:
        """
        Check if the text ends with a multi-byte character.
        """
        return len(text[-1].encode("utf-8")) > 1

    def get_punctuation_pause(self, text: str) -> float:
        """
        Check if the text ends with punctuation and return the pause duration ratio.
        """
        char = text.strip()[-1]
        if char in [".", "!", "?", "。", "，", "！", "？"]: 
            return 1.0
        if char in [",", ";", "；"]:
            return 0.5
        return 0.0

    def format_text(
        self,
        text: str,
        is_reference_text: bool=False
    ) -> str:
        """
        Format text for synthesis.

        :param text: The text to format.
        :param is_reference_text: Whether the text is a reference text. Default is False.
        :return: The formatted text.
        """
        text = simplify_quotations(text).replace(";", ",")
        if is_reference_text:
            text = text.strip(",;- ")
            if not text.endswith("."):
                return f"{text}. "
            return f"{text} "
        return text

    def chunk_text(
        self,
        text: str,
        max_length: int=135
    ) -> List[str]:
        """
        Chunk text into smaller pieces.
        """
        chunks = []

        current_chunk = ""
        current_chunk_length = 0

        # Split the text into sentences based on punctuation followed by whitespace
        sentences = re.split(r"(?<=[;:,.!?])\s+|(?<=[；：，。！？])", text)

        for sentence in sentences:
            encoded = sentence.encode("utf-8")
            encoded_length = len(encoded)

            if encoded_length == 0:
                continue

            if current_chunk_length + encoded_length <= max_length:
                current_chunk += sentence
                current_chunk_length += encoded_length
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
                current_chunk_length = encoded_length

            if not self.ends_with_multi_byte_character(sentence):
                current_chunk += " "
                current_chunk_length += 1

        if current_chunk_length > 0:
            chunks.append(current_chunk.strip())

        return chunks

    def get_output_from_audio_result(
        self,
        result: Union[torch.Tensor, np.ndarray[Any, Any]],
        output_format: Optional[AUDIO_OUTPUT_FORMAT_LITERAL]="wav",
        normalization_strategy: Optional[Literal["clip", "peak", "rms", "loudness", "none"]]="loudness",
        strip_silence: bool=True,
        output_save: bool=False,
        return_first_item: bool=False
    ) -> AudioResultType:
        """
        Get the output from the result based on the output type requested.
        Should not be called if the output of the model is already a file.
        """
        if strip_silence:
            result = trim_silence(result, raise_when_all_silent=False)

        if output_format is None or output_format in ["wav", "mp3", "ogg", "flac"]:
            # Save each audio
            result_uris: List[str] = []
            result_bytes: List[bytes] = []
            for audio in result:
                if is_numpy_array(audio):
                    audio = torch.from_numpy(audio)
                if len(audio.shape) == 1:
                    audio = audio.unsqueeze(0)
                num_channels, num_samples = audio.shape
                duration = num_samples / self.sample_rate
                output_file = tempfile.mktemp()
                output_file = str(
                    audio_write(
                        output_file,
                        audio,
                        sample_rate=self.sample_rate,
                        format="wav" if output_format is None else output_format,
                        normalize=normalization_strategy is not None,
                        strategy=normalization_strategy,
                    )
                )
                if output_save:
                    result_uris.append(output_file)
                else:
                    with open(output_file, "rb") as f:
                        result_bytes.append(f.read())
            if output_save:
                result = result_uris # type: ignore[assignment]
            else:
                result = result_bytes # type: ignore[assignment]
        elif output_format == "float":
            # Float tensor
            if is_numpy_array(result):
                result = torch.from_numpy(result)
            if len(result.shape) == 1:
                result = result.unsqueeze(0) # type: ignore[union-attr]
            if len(result.shape) == 2:
                result = result.unsqueeze(0) # type: ignore[union-attr]
        elif output_format == "int":
            # NP int16 array
            import numpy as np
            if is_torch_tensor(result):
                result = result.numpy()
            if len(result.shape) == 1:
                result = result[np.newaxis, :] # Add channel dimension
            if len(result.shape) == 2:
                result = result[np.newaxis, :] # Add batch dimension
        else:
            raise ValueError(f"Output type {output_format} not recognized.")

        if output_save:
            result = self.save_audio_output( # type: ignore[assignment]
                result,
                output_format="wav" if output_format is None else output_format
            )

        if return_first_item:
            return result[0]
        return result

    def save_audio_output(
        self,
        audios: AudioType,
        output_format: AUDIO_OUTPUT_FORMAT_LITERAL="wav",
    ) -> List[str]:
        """
        Saves one or more audios to the configured directory.
        """
        audio_filenames: List[str] = []
        if output_format in ["wav", "mp3", "ogg", "flac"]:
            for audio in audios: # type: ignore[union-attr]
                assert isinstance(audio, str), "Audio output should have been saved to a temporary file."
                audio_filename = f"{generate_id()}.{output_format}"
                audio_filenames.append(audio_filename)
                output_file = os.path.join(self.save_dir, audio_filename)
                os.rename(audio, output_file)
        elif output_format == "float":
            for audio in audios: # type: ignore[union-attr]
                audio_filename = f"{generate_id()}.pt"
                save_path = os.path.join(self.save_dir, audio_filename)
                import torch
                torch.save(audio, save_path)
                audio_filenames.append(audio_filename)
        elif output_format == "int":
            for audio in audios: # type: ignore[union-attr]
                audio_filename = f"{generate_id()}.npy"
                save_path = os.path.join(self.save_dir, audio_filename)
                import numpy as np
                np.save(save_path, audio)
                audio_filenames.append(audio_filename)
        else:
            raise ValueError(f"Format {output_format} not recognized.")
        return audio_filenames

    def synthesize(
        self,
        texts: List[str],
        reference_text: str,
        reference_audio: torch.Tensor,
        reference_sample_rate: int,
        seed: SeedType=None,
        speed: float=1.0,
        sway_sampling_coef: float=-1.0,
        target_rms: float=0.1,
        cross_fade_duration: float=0.15,
        punctuation_pause_duration: float=0.0,
        num_steps: int=32,
        cfg_strength: float=2.0,
        fix_duration: Optional[float]=None,
        use_tqdm: bool=False,
        chunk_callback: Optional[Callable[[AudioResultType], None]]=None,
        chunk_callback_format: AUDIO_OUTPUT_FORMAT_LITERAL="float",
    ) -> torch.Tensor:
        """
        Synthesize audio from text.
        """
        import torch
        import torchaudio # type: ignore[import-untyped]

        rms = torch.sqrt(torch.mean(torch.square(reference_audio)))
        if rms < target_rms:
            reference_audio = reference_audio * (target_rms / rms)

        if reference_sample_rate != self.sample_rate:
            reference_audio = torchaudio.transforms.Resample(
                orig_freq=reference_sample_rate,
                new_freq=self.sample_rate
            )(reference_audio)

        reference_audio = reference_audio.to(self.model.device)
        reference_audio_length = reference_audio.shape[-1] // self.hop_length
        reference_text = self.format_text(reference_text, is_reference_text=True)
        reference_text_length = len(reference_text.encode("utf-8"))

        if reference_audio.ndim == 3:
            # Remove batch dimension
            reference_audio = reference_audio.squeeze(0)

        audios: List[torch.Tensor] = []
        num_texts = len(texts)

        for i, text in enumerate(texts):
            text = self.format_text(text)

            text_chunks = self.chunk_text(text)
            num_text_chunks = len(text_chunks)

            for j, text_chunk in maybe_use_tqdm(enumerate(text_chunks), desc="Synthesizing Text Chunks", use_tqdm=use_tqdm, total=num_text_chunks):

                if fix_duration is not None:
                    duration = int(fix_duration * self.sample_rate / self.hop_length)
                else:
                    # Estimate duration
                    reference_text_length = len(reference_text.encode("utf-8"))
                    generate_text_length = len(text_chunk.encode("utf-8"))
                    duration = reference_audio_length + int(reference_audio_length / reference_text_length * generate_text_length / speed)

                logger.debug(f"Generating audio from text chunk {j + 1} of {num_text_chunks} for text {i + 1} of {num_texts} with duration {duration}. Reference text: {reference_text}, text chunk: {text_chunk}")

                audio, trajectory = self.model.sample(
                    cond=reference_audio,
                    text=[reference_text + text_chunk],
                    duration=duration,
                    steps=num_steps,
                    cfg_strength=cfg_strength,
                    sway_sampling_coef=sway_sampling_coef
                )

                audio = audio.to(torch.float32)[:, reference_audio_length:, :]
                audio = audio.permute(0, 2, 1)
                audio = self.vocoder.decode(audio)

                if rms < target_rms:
                    audio = audio * rms / target_rms

                audio = audio.cpu()

                if chunk_callback is not None:
                    callback_audio = self.get_output_from_audio_result(
                        audio,
                        output_format=chunk_callback_format,
                        output_save=False,
                        return_first_item=True
                    )
                    chunk_callback(callback_audio)

                audio = audio.squeeze(0) # Remove channel dimension

                if (i < num_texts - 1 or j < num_text_chunks - 1):
                    pause = self.get_punctuation_pause(text_chunk)
                    if pause > 0:
                        pause_duration = punctuation_pause_duration * pause
                        logger.debug(f"Adding pause of {pause_duration} seconds to the end of the audio.")
                        num_pause_samples = int(pause_duration * self.sample_rate)
                        pause_samples = torch.zeros(num_pause_samples).to(dtype=audio.dtype, device=audio.device)
                        audio = torch.cat([audio, pause_samples])

                audios.append(audio)

        # Combine audios
        return concatenate_audio(
            audios,
            cross_fade_duration=cross_fade_duration,
            sample_rate=self.sample_rate
        )

    def __call__(
        self,
        text: Union[str, List[str]],
        reference_audio: AudioType,
        reference_text: str,
        reference_sample_rate: Optional[int]=None,
        seed: SeedType=None,
        speed: float=1.0,
        sway_sampling_coef: float=-1.0,
        target_rms: float=0.1,
        cross_fade_duration: float=0.15,
        punctuation_pause_duration: float=0.10,
        num_steps: int=32,
        cfg_strength: float=2.0,
        fix_duration: Optional[float]=None,
        use_tqdm: bool=False,
        chunk_callback: Optional[Callable[[AudioResultType], None]]=None,
        chunk_callback_format: AUDIO_OUTPUT_FORMAT_LITERAL="float",
        output_format: AUDIO_OUTPUT_FORMAT_LITERAL="wav",
        output_save: bool=False,
    ) -> AudioResultType:
        """
        Generate speech from text and reference audio.

        :param text: The text to synthesize.
        :param reference_audio: The reference audio to use for synthesis.
        :param reference_text: The reference text to use for synthesis.
        :param seed: The seed to use for random number generation.
        :param speed: The speed of the synthesized audio.
        :param sway_sampling_coef: The sampling coefficient for sway sampling.
        :param target_rms: The target RMS value for the synthesized audio.
        :param cross_fade_duration: The duration of the cross-fade between audio chunks.
        :param punctuation_pause_duration: The duration of the pause after punctuation.
        :param num_steps: The number of flow estimation steps.
        :param cfg_strength: The strength of classifier-free guidance.
        :param fix_duration: The fixed duration of the synthesized audio.
        :param output_format: The format of the output audio.
        :param output_save: Whether to save the output audio, or return the audio data directly.
        :return: The synthesized audio.
        """
        import torch
        if not text:
            raise ValueError("Text is required for synthesis.")
        if not reference_audio:
            raise ValueError("Reference audio is required for synthesis.")
        if not reference_text:
            raise ValueError("Reference text is required for synthesis.")

        if seed is not None:
            seed_everything(seed)

        # Read the reference audio
        reference_audio, reference_sample_rate = audio_to_bct_tensor(
            reference_audio,
            sample_rate=reference_sample_rate,
            target_sample_rate=self.sample_rate
        )

        # Read the reference text if it's a file
        if os.path.exists(reference_text):
            with open(reference_text, "r", encoding="utf-8") as f:
                reference_text = f.read()

        # clamp at 15s
        max_reference_samples = reference_sample_rate * 15 # type: ignore[operator]
        reference_audio = reference_audio[:, :, :max_reference_samples]

        # Make sure the reference audio is a single channel
        if reference_audio.shape[1] > 1:
            reference_audio = reference_audio.mean(dim=1, keepdim=True)

        # Synthesize the audio
        with torch.inference_mode():
            results = self.synthesize(
                texts=[text] if isinstance(text, str) else text,
                reference_audio=reference_audio,
                reference_text=reference_text,
                reference_sample_rate=reference_sample_rate, # type: ignore[arg-type]
                seed=seed,
                speed=speed,
                sway_sampling_coef=sway_sampling_coef,
                target_rms=target_rms,
                cross_fade_duration=cross_fade_duration,
                punctuation_pause_duration=punctuation_pause_duration,
                num_steps=num_steps,
                cfg_strength=cfg_strength,
                fix_duration=fix_duration,
                use_tqdm=use_tqdm,
                chunk_callback=chunk_callback,
                chunk_callback_format=chunk_callback_format
            )

            # This utility method will get the requested format
            return self.get_output_from_audio_result(
                results.unsqueeze(0),
                output_format=output_format,
                output_save=output_save,
                return_first_item=True
            )
