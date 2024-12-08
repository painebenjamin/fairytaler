import os
import click

from typing import Optional

@click.command()
@click.argument("reference_audio", type=click.Path(exists=True))
@click.argument("reference_text", type=str)
@click.argument("text", type=str)
@click.option("--model", "-m", type=str, default="benjamin-paine/fairytaler", help="Path to the model directory or the model hub name.")
@click.option("--cache-dir", "-cd", type=click.Path(), default=None, help="When downloading from the hub, where to cache the models.")
@click.option("--local-dir", "-ld", type=click.Path(), default=None, help="Where to store the downloaded models.")
@click.option("--output", "-o", type=click.Path(), default=None, help="Output file path")
@click.option("--seed", "-s", type=int, default=None, help="Seed for random number generator")
@click.option("--speed", "-sp", type=float, default=1.0, help="Speed of the generated audio")
@click.option("--sway", "-sw", type=float, default=-1.0, help="Sway sampling coefficient.")
@click.option("--cross-fade", "-cf", type=float, default=0.15, help="Cross-fade duration in seconds")
@click.option("--punctuation-pause", "-pp", type=float, default=0.1, help="Punctuation pause duration in seconds")
@click.option("--num-steps", "-ns", type=int, default=32, help="The number of sampling steps.")
@click.option("--cfg-strength", "-cfg", type=float, default=2.0, help="Classifier-free Guidance strength.")
@click.option("--duration", "-dur", type=float, default=None, help="A fixed duration to use instead of dynamic.")
@click.option("--target-rms", "-rms", type=float, default=0.1, help="Target RMS value for the generated audio.")
@click.option("--debug/--no-debug", type=bool, is_flag=True, default=False, help="Enable debug mode.")
@click.option("--float16/--float32", type=bool, is_flag=True, default=True, help="Enable float16 mode.")
def main(
    reference_audio: str,
    reference_text: str,
    text: str,
    model: str="benjamin-paine/fairytaler",
    cache_dir: Optional[str]=None,
    local_dir: Optional[str]=None,
    output: Optional[str]=None,
    seed: Optional[int]=None,
    speed: float=1.0,
    sway: float=-1.0,
    cross_fade: float=0.15,
    punctuation_pause: float=0.1,
    num_steps: int=32,
    cfg_strength: float=2.0,
    duration: Optional[float]=None,
    target_rms: float=0.1,
    debug: bool=False,
    float16: bool=True,
) -> None:
    """
    Performs sampling from the model with the given reference audio and text.
    """
    from .pipeline import F5TTSPipeline
    from .util import green, red, debug_logger

    with debug_logger("ERROR" if not debug else "DEBUG"):
        try:
            pipeline = F5TTSPipeline.from_pretrained(
                model,
                variant="fp16" if float16 else None,
                cache_dir=cache_dir,
                local_dir=local_dir,
            )
            output_format = "wav"
            if output is not None:
                _, ext = os.path.splitext(output)
                assert ext in [".wav", ".mp3", ".ogg", ".flac"], f"Unsupported output format: {ext}"
                output_format = ext[1:]

            if os.path.isfile(text):
                with open(text, "r") as f:
                    text = f.read()

            result = pipeline(
                text,
                reference_audio,
                reference_text,
                seed=seed,
                speed=speed,
                sway_sampling_coef=sway,
                target_rms=target_rms,
                cross_fade_duration=cross_fade,
                punctuation_pause_duration=punctuation_pause,
                num_steps=num_steps,
                cfg_strength=cfg_strength,
                fix_duration=duration,
                output_format=output_format, # type: ignore[arg-type]
                output_save=True,
                use_tqdm=True
            )
            if output is not None:
                os.rename(result, output) # type: ignore[arg-type]

            click.echo(f"Generated audio saved to: {green(result)}") # type: ignore[arg-type]
        except Exception as e:
            click.echo(f"Error: {red(str(e))}")
            if debug:
                raise e

if __name__ == "__main__":
    main()
