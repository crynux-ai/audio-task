import json

from audio_task.models import AudioTaskArgs

if __name__ == "__main__":
    schema = AudioTaskArgs.model_json_schema()

    with open("schema/text-to-audio-task.json", mode="w") as f:
        json.dump(schema, f)

