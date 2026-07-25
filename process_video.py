import os
import subprocess

files = os.listdir("videos")

for file in files:
    print(file)

    # remove extension
    name = os.path.splitext(file)[0]

    # extract tutorial number from "#1"
    tutorial_number = name.split(" ")[0].replace("#", "").zfill(2)

    # extract title after "HTML - "
    file_name = name.split(" - ", 1)[1]
    file_name = file_name.replace(" - W3Schools.com", "")
    file_name = file_name.replace(" ", "_").replace("-", "")

    print(tutorial_number, file_name)

    subprocess.run([
        "ffmpeg",
        "-i", f"videos/{file}",
        "-ac", "1",
        f"audios/{tutorial_number}_{file_name}.mp3"
    ])
    