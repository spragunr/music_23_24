"""Process MusicCPR dataset.

Author: Nathan Sprague
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import datasets
import pydub
import warnings


import torchaudio.transforms as T
from transformers import AutoModel
from transformers import Wav2Vec2FeatureExtractor
import datasets
from datasets import load_dataset
import torch
from torch import nn


warnings.filterwarnings("ignore")

FILE_PATH = "/cs/home/stu/mohammau/cs497/music_23_24/audio/teleband-export/teleband_downloads"
OUT_FILE_PATH = "/cs/home/stu/mohammau/cs497/music_23_24/audio/teleband-export/teleband_wavs"
DATA_CSV = "/cs/home/stu/mohammau/cs497/music_23_24/audio/teleband-export/_select_from_teachers_where_id_13_and_id_not_in_21_23_24_25_26_s_202306061508.csv"

def show_score_histograms(df):
    grades = [9, 10, 11, 12]
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.hist(
            df[df["grade"] == grades[i]]["tone"][:], np.arange(0.5, 6, 1), rwidth=0.7
        )
        plt.suptitle("Histograms of Tone by Grade")
        plt.title("Grade " + str(grades[i]))
    plt.show()

    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.hist(
            df[df["grade"] == grades[i]]["rhythm"][:], np.arange(0.5, 6, 1), rwidth=0.7
        )
        plt.suptitle("Histograms of Rhythm by Grade")
        plt.title("Grade " + str(grades[i]))
    plt.show()


def show_bad_recordings(df):
    for i in range(df.shape[0]):
        y, sr = librosa.load(FILE_PATH + "/" + df.iloc[i]["key"])
        length = librosa.get_duration(y=y, sr=sr)
        if np.max(y) < 0.01 or length < 5:
            print(np.max(y))
            print(
                df.iloc[i]["student_id"],
                df.iloc[i]["created_at"],
                df.iloc[i]["updated_at"],
                df.iloc[i]["grade"],
                df.iloc[i]["rhythm"],
                df.iloc[i]["tone"],
                df.iloc[i]["key"],
                df.iloc[i]["title"],
            )
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            librosa.display.specshow(
                librosa.power_to_db(S, ref=np.max), y_axis="mel", x_axis="time"
            )
            plt.colorbar(format="%+2.0f dB")
            plt.title("Mel spectrogram")
            plt.show()


def remove_bad_recordings(df):
    bad_indices = []
    for i in range(df.shape[0]):
        print(FILE_PATH + "/" + df.iloc[i]["key"])
        y, sr = librosa.load(FILE_PATH + "/" + df.iloc[i]["key"])
        length = librosa.get_duration(y=y, sr=sr)
        if np.max(y) < 0.01 or length < 5:
            bad_indices.append(i)
            print("dropping row", i)
    df = df.drop(df.index[bad_indices])
    df = df.reset_index(drop=True)
    return df


def build_dataset(df, gen_wavs=False):
    for i in range(df.shape[0]):
        sound = pydub.AudioSegment.from_file(FILE_PATH + "/" + df.iloc[i]["key"])
        sound.export(OUT_FILE_PATH + "/" + df.iloc[i]["key"] + ".mp3", format="mp3")

    df["key"] = df["key"].apply(lambda x: OUT_FILE_PATH + "/" + x + ".mp3")
    ds = datasets.Dataset.from_pandas(df)
    ds = ds.cast_column("key", datasets.features.Audio(sampling_rate=16000))
    ds.save_to_disk("processed_dataset/teleband_dataset_mp3")
    return ds


def main():
    df = pd.read_csv(DATA_CSV)




    # Print column header names
    print(df.columns)

    df = df[
        (df["grade"] >= 9)
        & df["graded"]
        & ~df["title"].str.contains("Creativity")
        & ~df["title"].str.contains("Celebration")
    ]

    df = df.drop_duplicates(subset=["checksum"], keep="first")
    print("data size", df.shape[0])
    print(df["assignment_id"].value_counts())

    df = df.reset_index(drop=True)

    df = df[["key", "grade", "rhythm", "tone", "expression", "title", "assignment_id"]]
    print(df)

    ## Create the feature, tell it what shape to be, loop through and add it to each individual dataset entry
    ## Feature is the mert reperesntation of the data 
    ## Add another field after its converted for each entry

    #df = remove_bad_recordings(df)
    #ds = build_dataset(df)

    if False:
        show_score_histograms(df)

        # print the counts for the grade column
        print(df["grade"].value_counts())

        print(df["tone"].value_counts())
        print(df["rhythm"].value_counts())
        print(df["expression"].value_counts())

        # print all titles
        print(df["title"].value_counts())

        # print the titles of all assignment 21 entries
        print(df[df["assignment_id"] == 21]["title"])

    # print(df)



def add_mert():

    ## Produced with the assistance of Generative AI

    model = AutoModel.from_pretrained("m-a-p/MERT-v0")
    processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v0")
    # Load your dataset (replace with your dataset loading code)
    dataset = datasets.load_from_disk("processed_dataset/teleband_dataset_mp3")
    dataset = dataset.sort("assignment_id")
    # Define a resampler to match the model's sampling rate
    sampling_rate = dataset.features["key"].sampling_rate

    resample_rate = processor.sampling_rate
    
    # make sure the sample_rate aligned
    if resample_rate != sampling_rate:
        resampler = T.Resample(sampling_rate, resample_rate)
    else:
        resampler = None

    resampler = T.Resample(sampling_rate, resample_rate)
    # Create an empty list to store the hidden states for each sample
    all_layer_hidden_states_list = []
    # Process and extract hidden states for each audio sample in the dataset
    for i in range(3):
        if resampler is None:
            input_audio = dataset[i]["key"]["array"]
        else:
            input_audio = resampler(torch.from_numpy(dataset[i]["key"]["array"]))

        # Process the audio data using the feature extractor
        inputs = processor(input_audio, sampling_rate=resample_rate, return_tensors="pt")

        # Perform inference with the model
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Access the hidden states
        all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()

        # Append the hidden states to the list
        all_layer_hidden_states_list.append(all_layer_hidden_states[-1, :, :].T)
        print("Here!")
    # Convert the list of hidden states to a torch.Tensor

    ## fail here
    ## are some of the mert features too big
    # Turn into a numpy array instead, as we're using datasets and not torch
    #    all_layer_hidden_states_tensor = torch.stack(all_layer_hidden_states_list)
    # Add the list of mert representations to the dataset


    dataset = dataset.add_column("mert_feature", np.array(all_layer_hidden_states_list))
    print("Here! as well")

    # Resave the dataset
    dataset.save_to_disk("processed_dataset/teleband_dataset_mp3")
    # Traceback (most recent call last):
    # File "/cs/home/stu/mohammau/cs497/music_23_24/build_dataset.py", line 206, in <module>
    # add_mert()
    # File "/cs/home/stu/mohammau/cs497/music_23_24/build_dataset.py", line 197, in add_mert
    # all_layer_hidden_states_tensor = torch.stack(all_layer_hidden_states_list)
    # RuntimeError: stack expects each tensor to be equal size, but got [768, 2975] at entry 0 and [768, 2864] at entry 1





if __name__ == "__main__":
    main()
    add_mert()







#processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v0",trust_remote_code=True)
#
#    sampling_rate = dataset.features["key"].sampling_rate
#
#    resample_rate = processor.sampling_rate
#    # make sure the sample_rate aligned
#    if resample_rate != sampling_rate:
#        print(f'setting rate from {sampling_rate} to {resample_rate}')
#        resampler = T.Resample(sampling_rate, resample_rate)
#    else:
#        resampler = None
#
#    inputs = processor(input_audio, sampling_rate=resample_rate, return_tensors="pt")
#
#    with torch.no_grad():
#        outputs = model(**inputs, output_hidden_states=True)
#    all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()  
