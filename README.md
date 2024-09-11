# audio-task
A general framework to define and execute the audio generation task


### Features

* A generalized schema to define a text-to-audio generation task.
* Model quantizing (INT4 or INT8)
* Fine grained control generation arguments

### Example

Here is an example of the facebook/musicgen-small music generation:

```python
import scipy
from audio_task.inference import run_task


audio, sr = run_task(
    model="facebook/musicgen-small",
    prompt="80s pop track with bassy drums and synth",
    generation_config={
        "do_sample": True,
    },
    seed=1337
)
scipy.io.wavfile.write("example.wav", rate=sr, data=audio)
```


### Get started

Create and activate the virtual environment:
```shell
$ python -m venv ./venv
$ source ./venv/bin/activate
```

Install the dependencies and the library.

If your computer support cuda:
```shell
(venv) $ pip install -r requirments_cuda.txt && pip install -e .
```

else:
```shell
(venv) $ pip install -r requirments_macos.txt && pip install -e .
```


Check and run the examples:
```shell
(venv) $ python ./examples/example.py
```

More explanations can be found in the doc:

[https://docs.crynux.ai/application-development/music-tasks/text-to-music-task](https://docs.crynux.ai/application-development/music-tasks/text-to-music-task)

### Task Definition

The complete task definition is `AudioTaskArgs` in the file [```./src/audio_task/models/args.py```](src/audio_task/models/args.py)

### Task Response

The task response is a tuple of generated audio waveform and its sampling rate. The audio waveform is a `np.ndarray` of shape `(audio_length, channels)`. The sampling rate is an integer.

### JSON Schema

The JSON schemas for the tasks could be used to validate the task arguments by other projects.
The schemas are given under [```./schema```](./schema). Projects could use the URL to load the JSON schema files directly.
