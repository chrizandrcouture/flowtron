from nis import match
import subprocess
import re
import os
from tqdm import tqdm
import pandas as pd
import pdb


def get_duration(path_of_wav_file):
    process = subprocess.Popen(['ffmpeg',  '-i', path_of_wav_file], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = process.communicate()
    matches = re.search(r"Duration:\s{1}(?P<hours>\d+?):(?P<minutes>\d+?):(?P<seconds>\d+\.\d+?),", stdout.decode(), re.DOTALL).groupdict()
    total_length = float(matches['hours'])*3600 + float(matches["minutes"])*60 + float(matches["seconds"])
    return total_length


def resample_audio(input_file, sr=24000, output_folder=""):
    data = open(input_file).read().split("\n")
    for entry in data:
        f = entry.split("|")[0]
        command = ['ffmpeg',  '-i', f"'{f}'", '-ac', '1', '-ar', str(sr), '-q:a', '0',
                   f"'{os.path.join(output_folder, os.path.basename(f))}'"]
        process = os.system(" ".join(command))


def calculate_lens(folder, output_file):
    wav_files = [x for x in os.listdir(folder) if x.endswith(".wav")]
    output = pd.DataFrame()
    for f in tqdm(wav_files):
        path = os.path.join(folder, f)
        duration = get_duration(path)
        entry = pd.DataFrame({"filename": [path], "duration": [duration]})
        output = output.append(entry)
    output.to_csv(output_file, index=False)


def filter_files_by_len(datafile, duration_file, num_sample=None, output_file=""):
    df = pd.read_csv(duration_file)
    df2 = df.loc[df['duration'] < 10]
    samples = df2.sample(num_sample)
    data = pd.read_csv(datafile, names=["file", "annot", "id"], sep="|")
    sampled_data = data[data['file'].isin(samples['filename'])]
    sampled_data.to_csv(output_file, header=False, index=False, sep='|')


def compute_stats(datafile):
    data = pd.read_csv(datafile, header=None, sep="|")
    print(data.groupby(2).size())


def read_data_file(datafile):
    data = open(input_file).read().split("\n")
    df = []
    for entry in data:
        path, annot, id_ = entry.split("|")
        df.append(path, annot, id_)
    return df


def replace_audio_files(libri_file, new_data_file, replace_id, output_file):
    libri_data = read_data_file(libri_file)
    new_data = read_data_file(new_data_file)
    with open(output_file, "w") as out:
        old_entries = [f"{path}|{annot}|{id_}" for path, annot, id_ in libri_data
                       if int(id_.strip()) != replace_id]
        new_entries = [f"{path}|{annot}|{replace_id}" for path, annot, id_ in new_data]
        out.write("\n".join(old_entries + new_entries))


if __name__ == "__main__":
    REPLACE_ID = 6209
    output_file = "test_duration.csv"
    # datafile = "libritts_train_clean_100_audiopath_text_sid_atleast5min_val_filelist.txt"
    datafile = "bible_data_10k_test.txt"
    folder = "/app/chris/flowtron/tacotron2/waveglow/data/bible_data/test/"
    output_folder = "/app/chris/flowtron/tacotron2/waveglow/data/bible_data/train_24khz_fine/"
    output_file="bible_data_10k_lessthan10s_train.txt"
    # calculate_lens(folder, output_file)
    # compute_stats(datafile)
    # ID with most entries 6209, num speakers = 123, 12 samples in test, 412 samples in train.
    # filter_files_by_len(datafile, output_file, num_sample=20, output_file)
    resample_audio(output_file, output_folder=output_folder)
