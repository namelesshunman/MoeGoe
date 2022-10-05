import sys, re
import torch.cuda
from torch import no_grad, LongTensor
import logging
import make_ffmpeg_outputfile

logging.getLogger('numba').setLevel(logging.WARNING)

import commons
import utils
from models import SynthesizerTrn
from text import text_to_sequence, _clean_text
from mel_processing import spectrogram_torch
from scipy.io.wavfile import write

def get_text(text, hps, cleaned=False):
    if cleaned:
        text_norm = text_to_sequence(text, hps.symbols, [])
    else:
        text_norm = text_to_sequence(text, hps.symbols, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm

def ask_if_continue():
    while True:
        answer = input('Continue? (y/n): ')
        if answer == 'y':
            break
        elif answer == 'n':
            sys.exit(0)

def print_speakers(speakers):
    print('ID\tSpeaker')
    for id, name in enumerate(speakers):
        print(str(id) + '\t' + name)

def get_speaker_id(message):
    speaker_id = input(message)
    try:
        speaker_id = int(speaker_id)
    except:
        print(str(speaker_id) + ' is not a valid ID!')
        sys.exit(1)
    return speaker_id

def get_label_value(text, label, default, warning_name='value'):
    value = re.search(rf'\[{label}=(.+?)\]', text)
    if value:
        try:
            text = re.sub(rf'\[{label}=(.+?)\]', '', text, 1)
            value = float(value.group(1))
        except:
            print(f'Invalid {warning_name}!')
            sys.exit(1)
    else:
        value = default
    return value, text

def get_label(text, label):
    if f'[{label}]' in text:
        return True, text.replace(f'[{label}]', '')
    else:
        return False, text

class my_class:
    def __init__(self, *x):  # *x是不定长参数
        if len(x) == 2:
            self.seq = x[0]
            self.text = x[1]
            self.filenumber = 0  # 用来决定生成文件的序号，默认是1，一般不用到，只在重新生成创建失败的音频时用
        elif len(x) == 0:
            self.seq = ""
            self.text = ""
            self.filenumber = 0  # 用来决定生成文件的序号，默认是零，一般不用到，只在重新生成创建失败的音频时用
        elif len(x) == 3:
            self.seq = x[0]
            self.text = x[1]
            self.filenumber = x[2]  # 用来决定生成文件的序号，默认是零，一般不用到，只在重新生成创建失败的音频时用
        else:
            # print("必须给无参数或者两个参数，否则报错")
            raise AssertionError("必须给无参数或者二、三个参数，否则报错")


def my_get_txtflie(txtpath):
    # 用它来读取txt文件，按行读取，去除空行，要拿到每行的文本和对应说话人序号，文本要求经过处理，去空行，用|在前面隔开人物序号和文本
    # 最后的返回值要是一个类列表，该类要有两个属性，一个是序号，一个是文本
    class_list = []
    req = open(txtpath, encoding="utf-8").readlines()
    for line in req:
        line = str(line)
        if len(line) != 0:
            seq, line_text = line.split("|", 1)
            line_class = my_class()
            line_class.seq = seq
            line_class.text = line_text
            class_list.append(line_class)
        else:
            print("该行为空")
    return class_list


def my_voice_maker(character_seq, text, output_path):
    length_scale, text = get_label_value(text, 'LENGTH', 1, 'length scale')
    noise_scale, text = get_label_value(text, 'NOISE', 0.667, 'noise scale')
    noise_scale_w, text = get_label_value(text, 'NOISEW', 0.8, 'deviation of noise')
    cleaned, text = get_label(text, 'CLEANED')

    stn_tst = get_text(text, hps_ms, cleaned=cleaned)

    character_seq = int(character_seq)  # get_speaker_id这个函数有转int功能，但是它需要输入
    # print_speakers(speakers)
    speaker_id = character_seq
    out_path = output_path  #

    print("Raw_Text:", text)
    print("Cleaned_Text:", _clean_text(text, hps_ms.data.text_cleaners))

    with no_grad():
        x_tst = stn_tst.unsqueeze(0).to(dev)
        x_tst_lengths = LongTensor([stn_tst.size(0)]).to(dev)
        sid = LongTensor([speaker_id]).to(dev)
        audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale, noise_scale_w=noise_scale_w,
                               length_scale=length_scale)[0][0, 0].data.cpu().float().numpy()
    write(out_path, hps_ms.data.sampling_rate, audio)


if __name__ == '__main__':
    # model = input('Path of a VITS model: ')
    # config = input('Path of a config file: ')
    model = "/content/drive/MyDrive/gpu3_path/vits_from_Jiutian/koihime_vits_2000/G.pth"
    config = "/content/drive/MyDrive/gpu3_path/vits_from_Jiutian/koihime_vits_2000/config.json"

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    hps_ms = utils.get_hparams_from_file(config)
    n_speakers = hps_ms.data.n_speakers if 'n_speakers' in hps_ms.data.keys() else 0
    n_symbols = len(hps_ms.symbols) if 'symbols' in hps_ms.keys() else 0
    speakers = hps_ms.speakers if 'speakers' in hps_ms.keys() else ['0']
    # use_f0 = hps_ms.data.use_f0 if 'use_f0' in hps_ms.data.keys() else False

    net_g_ms = SynthesizerTrn(
        n_symbols,
        hps_ms.data.filter_length // 2 + 1,
        hps_ms.train.segment_size // hps_ms.data.hop_length,
        n_speakers=n_speakers,
        **hps_ms.model).to(dev)
    _ = net_g_ms.eval()
    utils.load_checkpoint(model, net_g_ms)

    while True:
        txt_path = input('Path of a txt file: ')
        output_path = input('Path of a Output: ')

        class_from_txt_list = my_get_txtflie(txt_path)
        class_get_loss_list = []

        # output_path = "E:\Voice_Maker_Output\keifa"
        file_number = 1

        flag = input("Do you want to set start number of wav file?y or n:")
        if flag == "y":
            set_number = input('wav file start number: ')
            try:
                set_number = int(set_number)
                file_number = set_number
            except:
                print("input number:", set_number, "is Invild")
        else:
            pass

        for class_fromtxt in class_from_txt_list:
            str_file_number = str(file_number).zfill(4)  # 设置为0001的格式，不足的位数补零
            try:
                output_path_name = output_path + "/" + str_file_number + ".wav"
                my_voice_maker(class_fromtxt.seq, class_fromtxt.text, output_path_name)
            except:
                print("文件名：", output_path_name, "句子：", class_fromtxt.text, "未生成成功")
                class_file_loss = my_class(class_fromtxt.seq, class_fromtxt.text, file_number)
                class_get_loss_list.append(class_file_loss)
            file_number = file_number + 1

        if class_get_loss_list:
            for class_loss in class_get_loss_list:
                str_file_number_loss = str(class_loss.filenumber).zfill(4)  # 设置为0001的格式，不足的位数补零
                try:
                    output_path_name2 = output_path + "/" + str_file_number_loss + ".wav"
                    my_voice_maker(class_loss.seq, class_loss.text, output_path_name2)
                except:
                    print("文件名：", output_path_name2, "句子：", class_loss.text, "未生成成功")
                    class_file_loss2 = my_class(class_loss.seq, class_loss.text, class_loss.filenumber)
                    class_get_loss_list.append(class_file_loss2)

        print("Voice Generated Sucessful")
        silent_audiofile_path = "/content/drive/MyDrive/gpu3_path/ffmpeg_path/silent-audio07_re_32bit.wav"
        # ffmpeg_path = "ffmpeg_path" #请在linux中直接安装ffmpeg，将它加入到路径变量中，不再需要相对路径，而且linux也不能用exe
        # make_ffmpeg_outputfile.use_ffmpeg_make_output_file(wav_raw_path=output_path, silent_audiofile_path=silent_audiofile_path, ffmpeg_path=ffmpeg_path)
        make_ffmpeg_outputfile.use_ffmpeg_make_output_file(wav_raw_path=output_path, silent_audiofile_path=silent_audiofile_path)
        ask_if_continue()