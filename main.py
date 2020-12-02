from dtaidistance import dtw
import os
import glob
import librosa
from tqdm import tqdm
import numpy as np
from python_speech_features import mfcc, fbank, logfbank
from scipy.spatial import distance

from numba import jit

# # y, sr = librosa.load('./motbuoc.mp3')
# # feat_mby = extract_features(y)
print("loading")
y, sr = librosa.load('./cilu_q.mp3')
query_feature = librosa.feature.mfcc(y, sr)
print("done!")
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
@jit
def search():

    res = {}
    for song in os.listdir("./data/features/"):
        song_path = os.path.join("./data/features/", song)
        try:
            song_feature = np.load(song_path, allow_pickle=True)
            count = 0
            distan = 0
            distan_min = 999999999
            for i in range(song_feature.shape[1]-query_feature.shape[1]):
                a = song_feature[:,i:i+query_feature.shape[1]]
                a = a.reshape(-1)
                count += 1
                distan = distance.cosine(a, b)
                if distan_min>distan:
                    distan_min = distan
            # song_list.append(song)
            # distance_list.append(distan/count)
            res[song] = distan_min
        except:
            pass
    return res
    
res = search()
res = dict(sorted(res.items(), key=lambda item: item[1]))
print(res)

