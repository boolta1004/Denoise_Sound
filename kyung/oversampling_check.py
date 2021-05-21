import librosa
import librosa.display
# import pyaudio #마이크를 사용하기 위한 라이브러리
import wave
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression#텐서플로우로 바꿀예정
import os
import csv
from imblearn.over_sampling import *

CHANNELS = 1
RATE = 44100 #비트레이트 설정
CHUNK = int(RATE / 10) # 버퍼 사이즈 1초당 44100비트레이트 이므로 100ms단위
RECORD_SECONDS = 1 #녹음할 시간 설정
WAVE_OUTPUT_FILENAME = "output.wav"
DATA_PATH = "./sound_data/"
train_data=[]#train_date 저장할 공강
train_label=[]#train_label 저장할
test_data=[]#train_date 저장할 공강
test_label=[]#train_label 저장할

최 = []
유 = []
경 = []
최_label = []
유_label = []
경_label = []


def load_wave_generator(path):
    batch_waves = []
    labels = []
    # input_width=CHUNK*6 # wow, big!!
    folders = os.listdir(path)
    # folders = path
    # while True:
    # print("loaded batch of %d files" % len(files))
    for folder in folders:
        if not os.path.isdir(path): continue  # 폴더가 아니면 continue
        files = os.listdir(path + "/" + folder)
        print("Foldername :", folder, ", - file count : ", len(files))  # 폴더 이름과 그 폴더에 속하는 파일 갯수 출력
        if (folder == "0"):
            for wav in files:
                if not wav.endswith(".wav"):
                    continue
                else:
                    global 최, 최_label  # 전역변수를 사용하겠다.
                    print("Filename :", wav)  # .wav 파일이 아니면 continue
                    y, sr = librosa.load(path + "/" + folder + "/" + wav)
                    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=45, hop_length=int(sr * 0.01),
                                                n_fft=int(sr * 0.02)).T

                    if (len(최) == 0):
                        최 = mfcc
                        최_label = np.full(len(mfcc), folder)
                    else:
                        최 = np.concatenate((최, mfcc), axis=0)
                        최_label = np.concatenate((최_label, np.full(len(mfcc), int(folder))), axis=0)
                        # print("mfcc :",mfcc.shape)
        if (folder == "1"):
            for wav in files:
                if not wav.endswith(".wav"):
                    continue
                else:
                    global 유, 유_label
                    print("Filename :", wav)
                    y, sr = librosa.load(path + "/" + folder + "/" + wav)
                    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=45, hop_length=int(sr * 0.01),
                                                n_fft=int(sr * 0.02)).T

                    if (len(유) == 0):
                        유 = mfcc
                        유_label = np.full(len(mfcc), folder)
                    else:
                        유 = np.concatenate((유, mfcc), axis=0)
                        유_label = np.concatenate((유_label, np.full(len(mfcc), int(folder))), axis=0)
                        # print("mfcc :",mfcc.shape)
        if (folder == "2"):
            for wav in files:
                if not wav.endswith(".wav"):
                    continue
                else:
                    global 경, 경_label  # 전역변수를 사용하겠다.
                    print("Filename :", wav)  # .wav 파일이 아니면 continue
                    y, sr = librosa.load(path + "/" + folder + "/" + wav)
                    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=45, hop_length=int(sr * 0.01),
                                                n_fft=int(sr * 0.02)).T

                    if (len(경) == 0):
                        경 = mfcc
                        경_label = np.full(len(mfcc), '2')
                        # print(경_label)
                        # print(경_label.shape)
                        # print(경)
                        # print(경.shape)
                    else:
                        경 = np.concatenate((경, mfcc), axis=0)
                        경_label = np.concatenate((경_label, np.full(len(mfcc), int(folder))), axis=0)
                        # print("mfcc :",mfcc.shape)

def make_data(data, data_label):
    a = []
    b = []
    for j, row in enumerate(data):
        if(j+100 == len(data)):
            a = np.array(a)
            b = np.array(b)
            return a,b
        a.append(data[j:j+100])
        b.append(data_label[j+100])

load_wave_generator(DATA_PATH)

choi_update = []
choi_update_label = []

yu_update = []
yu_update_label = []

jyeong_update = []
jyeong_update_label = []

for i in range(len(최)):
    if 최[i][1] != 0:
        choi_update.append(최[i])
        choi_update_label.append(최_label[i])
for i in range(len(유)):
    if 유[i][1] != 0:
        yu_update.append(유[i])
        yu_update_label.append(유_label[i])
for i in range(len(경)):
    if 경[i][1] != 0:
        jyeong_update.append(경[i])
        jyeong_update_label.append(경_label[i])

choi_update = np.array(choi_update)
yu_update = np.array(yu_update)
jyeong_update = np.array(jyeong_update)
choi_update_label = np.array(choi_update_label)
yu_update_label = np.array(yu_update_label)
jyeong_update_label = np.array(jyeong_update_label)

print(yu_update_label.shape)
print(jyeong_update_label.shape)
a_x, a_y = [], []
a_x = np.concatenate((yu_update, jyeong_update), axis=0)
a_y = np.concatenate((yu_update_label, jyeong_update_label), axis=0)
print(a_y.shape)

# f1 = open('x_kyung.csv', 'w', newline='')
# wr = csv.writer(f1)
# wr.writerows(jyeong_update)
# f1.close()
#
# f2 = open('y_kyung.csv', 'w', newline='')
# wr = csv.writer(f2)
# wr.writerows(jyeong_update_label)
# f2.close()
#
# f3 = open('x_ryu.csv', 'w', newline='')
# wr = csv.writer(f3)
# wr.writerows(yu_update)
# f3.close()
#
# f4 = open('y_ryu.csv', 'w', newline='')
# wr = csv.writer(f4)
# wr.writerows(yu_update_label)
# f4.close()
#
# f3 = open('x_a.csv', 'w', newline='')
# wr = csv.writer(f3)
# wr.writerows(a_x)
# f3.close()
#
# f4 = open('y_a.csv', 'w', newline='')
# wr = csv.writer(f4)
# wr.writerows(a_y)
# f4.close()

X_s, y_s = SMOTE().fit_resample(a_x, a_y)

X_s = X_s[31467:]
y_s = y_s[31467:]

# f3 = open('x_s.csv', 'w', newline='')
# wr = csv.writer(f3)
# wr.writerows(X_s)
# f3.close()
#
# f4 = open('y_s.csv', 'w', newline='')
# wr = csv.writer(f4)
# wr.writerows(y_s)
# f4.close()
# exit()

from tensorflow.keras.models import load_model
model2 = load_model('gru_test.h5', compile = False)

def test_voice(X_s, model):
    test = X_s

    temp_label = np.full(len(X_s), 0)
    test, test_label = make_data(X_s, y_s)

    y_pred = np.argmax(model.predict(test), axis=-1)
    return who(y_pred)

def who(y_pred):
  aaa = 0
  bbb = 0
  ccc = 0
  for i in range(0, len(y_pred)):
    if(y_pred[i] == 0):
      aaa = aaa + 1
    elif(y_pred[i] == 1):
      bbb = bbb + 1
    else:
      ccc = ccc + 1

  result = [aaa,bbb,ccc]
  return result.index(max(result)), result


def result(path, model):
  total = 0
  count = 0
  folders = os.listdir(path)

  for folder in folders:
      if not os.path.isdir(path):continue #폴더가 아니면 continue
      files = os.listdir(path+"/"+folder)
      print("Foldername :",folder,", - file count : ",len(files))#폴더 이름과 그 폴더에 속하는 파일 갯수 출력
      total = total + len(files)
      for wav in files:
        print(wav)
        if not wav.endswith(".wav"):continue
        else:
          wav = path+"/"+folder+"/"+wav
          result_index, result = test_voice(wav, model)
          print("result : ", result_index , " real : ", int(folder), "  count : ", result)
          if(result_index == int(folder)):
            count = count + 1

  return str(count/total) + "%"

def model_predict_return(X_s, model):
  # files = os.listdir(path)
    result_index, result = test_voice(X_s, model)

    if result_index == 0:
      return "최창준 목소리 입니다"
    elif result_index == 1:
      return "유일권 목소리 입니다"
    else:
      return "경재원 목소리 입니다"

print(model_predict_return(X_s, model2))