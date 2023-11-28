from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from dtw_helper import *
import datasets

PROCESSED_DATASET_PATH = "D:/Documents/VsCode/Python/Research-Project/music_23_24/processed_dataset/teleband_dataset_mp3"
SAMPLE_PATH = "D:/Documents/VsCode/Python/Research-Project/tele.band-playing-samples"
CHUNK_SIZES = [2, 4, 5, 10, 20, 25, 50, 100, 125, 250]

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
    data = [([], []) for _ in range(len(CHUNK_SIZES))]

    for student_recording in dataset:
        s_data, s_sr = student_recording["key"]["array"], student_recording["key"]["sampling_rate"]
        D, wp = produce_warping_path(s_data, s_sr, student_recording["title"], 
                                     playing_sample_map, USE_NORMALIZATION, TRIM_SILENCE)


        # Take the first 500 items, since the minimum warping path is 501 in the dataset.
        wp = map_warping_path(wp)[:500]

        for i in range(len(CHUNK_SIZES)):
            sub_arrays = np.split(wp, CHUNK_SIZES[i])
            # Get mean and standard deviation of each subarray
            mean_std = []
            for sub_array in sub_arrays:
                mean_std.extend([np.mean(sub_array), np.std(sub_array)])
            data[i][0].append(mean_std)
            data[i][1].append(student_recording["rhythm"])
    

    if USE_CROSS_VALIDATION:
        x, y = [], []
        for i in range(len(CHUNK_SIZES)):
            rf = RandomForestClassifier()
            scores = cross_val_score(rf, data[i][0], data[i][1], cv=5)
            x.append(CHUNK_SIZES[i])
            y.append(scores.mean())
            # print("%0.2f accuracy with a standard deviation of %0.2f with chunk size %d" % (scores.mean(), scores.std(), CHUNK_SIZES[i]))
        plt.plot(x, y)

        plt.xlabel("Chunk Size")
        plt.ylabel("Accuracy")
        plt.show()
        
    else:
        # We don't really have enough data to do this split, but we are going to anyways because numbers.
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
        # # Train model
        # rf.fit(X_train, y_train)

        # # Now get data from test set.
        # y_pred = rf.predict(X_test)
        # accuracy = accuracy_score(y_test, y_pred)
        # print(f"Accuracy: {accuracy}")
        print("not available")
    
main()