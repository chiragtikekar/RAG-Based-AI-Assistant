import os
import json
from faster_whisper import WhisperModel

# Load model
model = WhisperModel("base", device="cpu", compute_type="int8")

# Make sure json folder exists
os.makedirs("jsons", exist_ok=True)

audios = os.listdir("audios")

for audio in audios:
    if "_" in audio:

        number = audio.split("_")[0]
        title = audio.split("_")[1].rsplit(".", 1)[0]

        print("Processing:", number, title)

        segments, info = model.transcribe(
            f"audios/{audio}",
            task="transcribe"
        )

        chunks = []

        for segment in segments:
            chunks.append({
                "number": number,
                "title": title,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text
            })

        chunks_with_metadata = {
            "language": info.language,
            "chunks": chunks
        }

        with open(f"jsons/{audio}.json", "w", encoding="utf-8") as f:
            json.dump(chunks_with_metadata, f, ensure_ascii=False, indent=4)

print(" All files processed successfully!")
