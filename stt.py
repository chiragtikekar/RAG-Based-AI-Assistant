# import whisper

# model = whisper.load_model("large-v2") 

# result = model.trahttps://github.com/microsoft/pyright/blob/main/docs/configuration.md#reportMissingImportsncribe(audio = "auidos/1.mp3",language = "hi", task="translate")

# print(result["text"])


import sys
sys.stdout.reconfigure(encoding='utf-8')

from faster_whisper import WhisperModel

# Use int8 for lower RAM usage
model = WhisperModel("base", device="cpu", compute_type="int8")

segments, info = model.transcribe(
    "audios/00_9_Minutes__codewithharry.mp3",
    language="hi"
)

for segment in segments:
    print(segment.text)
