This is a re-implementation of [F5-TTS](https://github.com/SWivid/F5-TTS) aimed at reducing dependencies, increasing speed, reducing model size and improving usability.

# Installation

Fairytaler assumes you have a working CUDA environment to install into.

```
pip install fairytaler
```

# How to Use

You do not need to pre-download anything, necessary data will be downloaded at runtime. Weights will be fetched from [HuggingFace.](https://huggingface.co/benjamin-paine/fairytaler)

## Command Line

Use the `fairytaler` binary from the command line like so:

```sh
fairytaler examples/reference.wav examples/reference.txt "Hello, this is some test audio!"
```

Many options are available, for complete documentation run `fairytaler --help`.

## Python

```py
from fairytaler import F5TTSPipeline

pipeline = F5TTSPipeline.from_pretrained(
  "benjamin-paine/fairytaler",
  variant="fp16", # Omit for float32
  device="auto"
)
output_wav_file = pipeline(
  text="Hello, this is some test audio!",
  reference_audio="examples/reference.wav",
  reference_text="examples/reference.txt",
  output_save=True
)
print(f"Output saved to {output_wav_file}")
```

The full execution signature is:

```py
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
    output_format: AUDIO_OUTPUT_FORMAT_LITERAL="wav",
    output_save: bool=False,
) -> AudioResultType
```

Format values are `wav`, `ogg`, `flac`, `mp3`, `float` and `int`. Passing `output_save=True` will save to file, not passing it will return the data directly.

# Citation

```
@misc{chen2024f5ttsfairytalerfakesfluent,
      title={F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching}, 
      author={Yushen Chen and Zhikang Niu and Ziyang Ma and Keqi Deng and Chunhui Wang and Jian Zhao and Kai Yu and Xie Chen},
      year={2024},
      eprint={2410.06885},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2410.06885}, 
}
```
