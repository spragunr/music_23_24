import librosa
import os
import numpy as np

def intialize_playing_sample_map(sample_path, use_normalization, trim_silence):
    playing_sample_map = {}
    for name_of_song in os.listdir(sample_path):
        # Load in data and create the chromagram
        full_song_path = os.path.join(sample_path, name_of_song)

        if os.path.isdir(full_song_path):
            continue

        playing_sample_data, playing_sample_sr = librosa.load(full_song_path)

        if use_normalization: 
            playing_sample_data = librosa.util.normalize(playing_sample_data)
        if trim_silence:
            playing_sample_data, _ = librosa.effects.trim(playing_sample_data)

        playing_sample_chroma = librosa.feature.chroma_cqt(y=playing_sample_data, sr=playing_sample_sr,
                                                 hop_length=1024)
        
        # Change the mapping for two special cases.
        song_name_key = name_of_song.replace(".mp3", "")
        playing_sample_map[song_name_key] = playing_sample_chroma
    return playing_sample_map

def produce_warping_path(s_data, s_sr, s_title, sample_map, use_normalization, trim_silence):

    if use_normalization: 
        s_data= librosa.util.normalize(s_data)
    if trim_silence:
        old_data = s_data
        
        # This is the original trim calls. The default DB cutoff is set to 60 which isnt a problem just dont find it as accurate.
        # s_data, _ = librosa.effects.trim(s_data)
        
        # This is my updated way of doing it. ERM i went through and looked at the code that I put into the initialize playing sample thing.
        # Basically I was able to produde figures that allowed me too look at the decibal range that these pieces are being played at.
        # From those figures I was able to decude that the metranome is generally being played at 30 Db
        s_data, _ = librosa.effects.trim(s_data, top_db=30)
        #print(f'Before Trim: {librosa.get_duration(S=old_data, sr=s_sr)}')
        # print(f'After Trim: {librosa.get_duration(S=s_data, sr=s_sr)}\n')

    student_recording_chroma = librosa.feature.chroma_cqt(y=s_data, sr=s_sr ,
                                                hop_length=1024)

    # Retrieve the playing sample chroma from the hashmap.
    if not (s_title in sample_map):
        print(f"Couldn't find {s_title} in playing sample map.")
        return (np.empty([2, 2]), np.empty([2, 2]))
    playing_sample_chroma = sample_map[s_title]
    D, wp = librosa.sequence.dtw(X=playing_sample_chroma, Y=student_recording_chroma, metric='cosine')
    return (D, wp)
