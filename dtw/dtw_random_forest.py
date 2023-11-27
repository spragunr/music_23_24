from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from dtw_helper import *
import datasets

PROCESSED_DATASET_PATH = "/cs/home/stu/greer2jl/research/lorenzo_stuff/Music24/processed_dataset/teleband_dataset_mp3"
SAMPLE_PATH = "/cs/home/stu/greer2jl/research/lorenzo_stuff/Music24/tele.band-playing-samples"

USE_NORMALIZATION = True
TRIM_SILENCE = True
USE_CROSS_VALIDATION = True

def map_warping_path(wp) -> np.ndarray:
    mapped_wp = []
    wp = wp[::-1]
    for i in range(1, len(wp)):
        if wp[i-1][0] + 1 == wp[i][0] and wp[i-1][1] + 1 == wp[i][1]:
            mapped_wp.append(0)
        elif wp[i-1][0] + 1 == wp[i][0]:
            mapped_wp.append(1)
        else:
            mapped_wp.append(-1)
    return np.array(mapped_wp)

def main():
    # Intialize a playing sample map that maps a title to a processed chromagram
    playing_sample_map = intialize_playing_sample_map(SAMPLE_PATH, True, True)

    dataset = datasets.load_from_disk(PROCESSED_DATASET_PATH)
    X, y = [], []

    for student_recording in dataset:
        s_data, s_sr = student_recording["key"]["array"], student_recording["key"]["sampling_rate"]
        D, wp = produce_warping_path(s_data, s_sr, student_recording["title"], 
                                     playing_sample_map, USE_NORMALIZATION, TRIM_SILENCE)


        # Take the first 500 items, since the minimum warping path is 501 in the dataset.
        wp = map_warping_path(wp)[:500]
        # Split into 5
        sub_arrays = np.split(wp, 10)

        # Get mean and standard deviation of each subarray
        mean_std = []
        for sub_array in sub_arrays:
            mean_std.extend([np.mean(sub_array), np.std(sub_array)])

        X.append(mean_std)
        y.append(student_recording["rhythm"])
    
    rf = RandomForestClassifier()

    if USE_CROSS_VALIDATION:
        scores = cross_val_score(rf, X, y, cv=5)
        print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
    else:
        # We don't really have enough data to do this split, but we are going to anyways because numbers.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
        # Train model
        rf.fit(X_train, y_train)

        # Now get data from test set.
        y_pred = rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")
    
main()

