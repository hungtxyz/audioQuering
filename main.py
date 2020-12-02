from dtaidistance import dtw
import os
import glob
import librosa
from tqdm import tqdm
import numpy as np
from python_speech_features import mfcc, fbank, logfbank
from scipy.spatial import distance



# # y, sr = librosa.load('./motbuoc.mp3')
# # feat_mby = extract_features(y)

y, sr = librosa.load('./lquery.mp3')
query_feature = librosa.feature.mfcc(y, sr)

b =  query_feature.reshape(-1)
# distan = 0
# count = 0
# for i in range(0, fcc_tdp.shape[1]-fcc_q.shape[1]):
#     a = fcc_tdp[:, i:i+fcc_q.shape[1]]
#     a = a.reshape(-1)
#     b = fcc_q.reshape(-1)
#     a = a.astype(np.double)
#     b = b.astype(np.double)
#     count += 1
#     # distan += dtw.distance_fast(a, b)
#     distan += distance.cosine(a, b)
#     # distan  += np.linalg.norm(a-b)

# distan /= count
# print(distan)

distance_list = []
song_list = []
for song in os.listdir("./data/features/"):
    song_path = os.path.join("./data/features/", song)
    try:
        song_feature = np.load(song_path, allow_pickle=True)
        count = 0
        distan = 0
        for i in range(song_feature.shape[1]-query_feature.shape[1]):
            a = song_feature[:,i+query_feature.shape[1]]
            count += 1
            distan += distance.cosine(a, b)
        song_list.append(song)
        distance_list.append(distan/count)
    except:
        pass

a = min(distance_list)
print(song_list[a.index(a)])
