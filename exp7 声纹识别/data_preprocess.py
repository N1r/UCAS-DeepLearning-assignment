import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from configuration import get_config
from utils import keyword_spot

config = get_config()   # get arguments from parser

# downloaded dataset path
audio_path =  './VCTK-Corpus/wav48'


def save_spectrogram_tdsv():
    """ Select text specific utterance and perform STFT with the audio file.
        Audio spectrogram files are divided as train set and test set and saved as numpy file.
        Need : utterance data set (VTCK)
    """
    print("start text dependent utterance selection")
    os.makedirs(config.train_path, exist_ok=True)   # make folder to save train file
    os.makedirs(config.test_path, exist_ok=True)    # make folder to save test file

    utterances_spec = []
    for folder in os.listdir(audio_path):
        utter_path= os.path.join(audio_path, folder, os.listdir(os.path.join(audio_path, folder))[0])
        if os.path.splitext(os.path.basename(utter_path))[0][-3:] != '001':  # if the text utterance doesn't exist pass
            print(os.path.basename(utter_path)[:4], "001 file doesn't exist")
            continue

        utter, sr = librosa.core.load(utter_path, config.sr)               # load the utterance audio
        utter_trim, index = librosa.effects.trim(utter, top_db=14)         # trim the beginning and end blank
        if utter_trim.shape[0]/sr <= config.hop*(config.tdsv_frame+2):     # if trimmed file is too short, then pass
            print(os.path.basename(utter_path), "voice trim fail")
            continue

        S = librosa.core.stft(y=utter_trim, n_fft=config.nfft,
                              win_length=int(config.window * sr), hop_length=int(config.hop * sr))  # perform STFT
        S = keyword_spot(S)          # keyword spot (for now, just slice last 80 frames which contains "Call Stella")
        utterances_spec.append(S)    # make spectrograms list

    utterances_spec = np.array(utterances_spec)  # list to numpy array
    np.random.shuffle(utterances_spec)           # shuffle spectrogram (by person)
    total_num = utterances_spec.shape[0]
    train_num = (total_num//10)*9                # split total data 90% train and 10% test
    print("selection is end")
    print("total utterances number : %d"%total_num, ", shape : ", utterances_spec.shape)
    print("train : %d, test : %d"%(train_num, total_num- train_num))
    np.save(os.path.join(config.train_path, "train.npy"), utterances_spec[:train_num])  # save spectrogram as numpy file
    np.save(os.path.join(config.test_path, "test.npy"), utterances_spec[train_num:])


def save_spectrogram_tisv():
    """ Full preprocess of text independent utterance. The log-mel-spectrogram is saved as numpy file.
        Each partial utterance is splitted by voice detection using DB
        and the first and the last 180 frames from each partial utterance are saved.
        Need : utterance data set (VTCK)
    """
    print("开始文本不独立语言特征提取...")
    ind = 0
    if not os.path.isdir(config.train_path):
        os.makedirs(config.train_path, exist_ok=True)   # make folder to save train file
    else:
        ind += len(os.listdir(config.train_path))
    if not os.path.isdir(config.test_path):
        os.makedirs(config.test_path, exist_ok=True)    # make folder to save test file
    else:
        ind += len(os.listdir(config.test_path))

    utter_min_len = (config.tisv_frame * config.hop + config.window) * config.sr    # lower bound of utterance length
    total_speaker_num = len(os.listdir(audio_path))
    train_speaker_num= (total_speaker_num//10)*9            # split total data 90% train and 10% test
    print("总的 speaker 数为 : %d"%total_speaker_num)
    print("训练 : %d, 测试 : %d"%(train_speaker_num, total_speaker_num-train_speaker_num))
    print(os.listdir(audio_path))

    for i, folder in enumerate(os.listdir(audio_path)):
        # 从原来断开处继续处理
        if i < ind:
            continue
        print("从断开处继续处理...")
        speaker_path = os.path.join(audio_path, folder)     # path of each speaker
        print("第 %d 个 speaker 处理..." % i)
        utterances_spec = []
        k=0
        for utter_name in os.listdir(speaker_path):
            utter_path = os.path.join(speaker_path, utter_name)         # path of each utterance
            utter, sr = librosa.core.load(utter_path, config.sr)        # load utterance audio
            utter_trim, index = librosa.effects.trim(utter, top_db=20)  # voice activity detection, only trim

            cur_slide = 0
            mfcc_win_sample = int(config.sr*config.hop*config.tisv_frame)
            while(True):
                if(cur_slide + mfcc_win_sample > utter_trim.shape[0]):
                    break
                slide_win = utter_trim[cur_slide : cur_slide+mfcc_win_sample]

                S = librosa.feature.mfcc(y=slide_win, sr=config.sr, n_mfcc=40)
                utterances_spec.append(S)

                cur_slide += int(mfcc_win_sample/2)

        utterances_spec = np.array(utterances_spec)
        print('utterances_spec.shape = {}'.format(utterances_spec.shape))

        if i<train_speaker_num:      # save spectrogram as numpy file
            np.save(os.path.join(config.train_path, "speaker%d.npy"%i), utterances_spec)
        else:
            np.save(os.path.join(config.test_path, "speaker%d.npy"%(i-train_speaker_num)), utterances_spec)


if __name__ == "__main__":
    if config.tdsv:
        save_spectrogram_tdsv()
    else:
        print("保存频谱tisv...")
        save_spectrogram_tisv()