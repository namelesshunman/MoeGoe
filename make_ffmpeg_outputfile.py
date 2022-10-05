#############################################################
#本py程序只需改变路径即可生成output.wav，比只生成mylist.txt要更加方便
#############################################################
import os
import shutil

def make_ffmpeg_filelists(wavpath):
    FileName_Output = wavpath + "/" + "mylist.txt"
    dir_filelist = os.listdir(wavpath)
    dir_filelist.sort()
    for filename in dir_filelist:
        filename = str(filename)
        if filename.endswith(".wav"):#固定选择wav文件作为连接对象
            # print("True")
            with open(FileName_Output, 'a', encoding='utf-8')as file:
                file.write("file '" + filename + "'" + "\n")
                file.write("file 'silent-audio07_re_32bit.wav'" + "\n")
    print("myfilelist生成成功")

def muti_folder_make_filelists(raw_path):#只能对规则的根文件夹用，里面必须是子文件夹，不能有文件
    # raw_path = "E:\Voice_Maker_Output\PCR"
    sub_path = os.listdir(raw_path)
    for path_name in sub_path:
        true_path = raw_path + "/" + path_name
        print(true_path)
        make_ffmpeg_filelists(true_path)

# def use_ffmpeg_make_output_file(wav_raw_path, silent_audiofile_path , ffmpeg_path,):
def use_ffmpeg_make_output_file(wav_raw_path, silent_audiofile_path):
    #对有wav文件的文件夹生成mylist.txt
    # wav_raw_path = "E:\Voice_Maker_Output\恋昌无双\\2000ver_1-3.5"
    make_ffmpeg_filelists(wav_raw_path) # 生成需连接文件列表需要在复制静音文件前

    #复制静音文件
    # silent_audiofile_path = "D:\\ffmpeg\\ffmpeg-2022-07-14-git-882aac99d2-full_build\\ffmpeg-2022-07-14-git-882aac99d2-full_build\\bin\silent-audio07_re_32bit.wav"
    shutil.copy(silent_audiofile_path, wav_raw_path)
    print("静音文件复制成功")

    #用cmd命令连接文件
    # ffmpeg_path = "D:\\ffmpeg\\ffmpeg-2022-07-14-git-882aac99d2-full_build\\ffmpeg-2022-07-14-git-882aac99d2-full_build\\bin"
    rsplit_string = wav_raw_path.rsplit("/", 1)[1]
    connected_files_outputpath = "/content/drive/MyDrive/gpu3_path/gpu3_output/connect_all_output"
    # cmd = ffmpeg_path + "/" + "ffmpeg -f concat -i " + wav_raw_path + "/" + "mylist.txt -c copy " + connected_files_outputpath + "/" + rsplit_string + "_all.wav"
    cmd = "ffmpeg -f concat -i " + wav_raw_path + "/" + "mylist.txt -c copy " + connected_files_outputpath + "/" + rsplit_string + "_all.wav"
    #更改此项可以替换输出位置
    print(cmd)
    os.system(cmd)