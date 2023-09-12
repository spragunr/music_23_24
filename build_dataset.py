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

warnings.filterwarnings("ignore")

FILE_PATH = "/home/spragunr/nobackup/data/teleband-export/teleband_downloads"
OUT_FILE_PATH = "/home/spragunr/nobackup/data/teleband-export/teleband_wavs"
DATA_CSV = "/home/spragunr/nobackup/data/teleband-export/_select_from_teachers_where_id_13_and_id_not_in_21_23_24_25_26_s_202306061508.csv"


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
    if gen_wavs:
        for i in range(df.shape[0]):
            sound = pydub.AudioSegment.from_file(FILE_PATH + "/" + df.iloc[0]["key"])
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

    df = remove_bad_recordings(df)
    ds = build_dataset(df)

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


if __name__ == "__main__":
    main()
