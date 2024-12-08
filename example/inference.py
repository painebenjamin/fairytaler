import os
import sys

HERE = os.path.abspath(os.path.dirname(__file__))

sys.path.insert(0, os.path.join(HERE, '..', 'src'))
from fairytaler import F5TTSPipeline
from fairytaler.util import debug_logger

with debug_logger() as logger:
    pipeline = F5TTSPipeline.from_pretrained(
        variant="fp16",
        device=0
    )
    output_file = pipeline(
        text="Once upon a time, there was a beautiful princess named Cinderella.",
        reference_audio=os.path.join(HERE, 'reference.wav'),
        reference_text=os.path.join(HERE, 'reference.txt'),
        output_save=True
    )
    print(f"Output audio file: {output_file}")
