from inspect import trace
import os
import pandas as pd
import pdb


output_file = "filelists/bible_data_10k_test.txt"
data_folder = "/app/chris/flowtron/tacotron2/waveglow/data/bible_data/test/"
transcript_file = "/app/chris/flowtron/transcript.txt"

files = os.listdir(data_folder)
files = [x for x in files if x.endswith(".wav")]

df = open(transcript_file, "r").read().split("\n")
df = [x.split("\t") for x in df]

annot = {}
for name, text, _ in df:
    key = name.split("/")[1]
    if key in annot:
        print("Repeating!!!!!", key + "\n", annot[key] + "\n", text)
    annot[key] = text

with open(output_file, "w") as out:
    outputs = []
    for f in files:
        text = annot[f.replace(".wav", "")]
        filepath = os.path.join(data_folder, f)
        speaker_id = "0"
        outputs.append(f"{filepath}|{text}|{speaker_id}")
    # pdb.set_trace()
    out.write("\n".join(outputs))
