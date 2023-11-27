import datasets
import librosa
import os
import numpy as np
import matplotlib.pyplot as plt
from dtw_helper import *

PROCESSED_DATASET_PATH = "D:/Documents/VsCode/Python/Research-Project/music_23_24/processed_dataset/teleband_dataset_mp3"
SAMPLE_PATH = "D:/Documents/VsCode/Python/Research-Project/tele.band-playing-samples"
USE_NORMALIZATION = True
TRIM_SILENCE = True
TRY_DIFFERENT_DECIBEL_CUTOFFS = False

# Our current method of calculating a score is measuring the difference between each increase on the warping path.
def calculate_score(warping_path):
    total = 0.   
    for i in range(len(warping_path) - 1):
            original_score_left, original_score_right = warping_path[i][0], warping_path[i][1]
            if (original_score_left == (warping_path[i + 1][0]) + 1 and original_score_right == (warping_path[i + 1][1]) + 1):
                total += 0       
            else:
                total += 1        
    return total / len(warping_path)

def main():
    # Intialize a playing sample map that maps a title to a processed chromagram
    playing_sample_map = intialize_playing_sample_map(SAMPLE_PATH, True, True)

    dataset = datasets.load_from_disk(PROCESSED_DATASET_PATH)
    # Initialize the two points
    x, y = [], []

    for student_recording in dataset:
        student_recording_data, student_recording_sr = student_recording["key"]["array"], student_recording["key"]["sampling_rate"]
        D, wp = produce_warping_path(student_recording_data, student_recording_sr, 
                                     student_recording["title"], playing_sample_map, True, True)
        
        if wp.size == 0: continue

        actual_score = student_recording["rhythm"]
        score = calculate_score(wp)
        x.append([score])
        y.append([actual_score])

    fig, ax = plt.subplots()

    ax.scatter(x, y, picker = 5)
    plt.title("Rhythm and New Score Measure")
    plt.show()

main()