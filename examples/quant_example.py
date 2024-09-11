import scipy
from audio_task.inference import run_task


audio, sr = run_task(
    model="facebook/musicgen-small",
    prompt="80s pop track with bassy drums and synth",
    generation_config={
        "do_sample": True,
    },
    seed=1337,
    quantize_bits=8
)
scipy.io.wavfile.write("example.wav", rate=sr, data=audio)
