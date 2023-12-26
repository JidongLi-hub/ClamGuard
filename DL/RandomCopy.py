import os
import random
import shutil
from tqdm import tqdm


def copy_random_files(source_folder, destination_folder, n):
    # 获取源文件夹中的所有文件列表
    all_files = os.listdir(source_folder)

    # 从文件列表中随机选择N个文件
    selected_files = random.sample(all_files, n)

    # 创建目标文件夹（如果不存在）
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # 复制选定的文件到目标文件夹
    for file_name in selected_files:
        source_path = os.path.join(source_folder, file_name)
        destination_path = os.path.join(destination_folder, file_name)
        shutil.copy2(source_path, destination_path)
        print(f"File '{file_name}' copied to '{destination_folder}'")

def copy_random_files2(source_folder, Train_folder, Test_folder, Dev_folder):
    # 获取源文件夹中的所有文件列表
    all_files = os.listdir(source_folder)

    random.shuffle(all_files)
    selected_files1 = all_files[:1600]
    selected_files2 = all_files[1600:1800]
    selected_files3 = all_files[1800:]

    # 复制选定的文件到目标文件夹
    for file_name in tqdm(selected_files1):
        source_path = os.path.join(source_folder, file_name)
        destination_path = os.path.join(Train_folder, file_name)
        shutil.copy2(source_path, destination_path)

    for file_name in tqdm(selected_files2):
        source_path = os.path.join(source_folder, file_name)
        destination_path = os.path.join(Test_folder, file_name)
        shutil.copy2(source_path, destination_path)

    for file_name in tqdm(selected_files3):
        source_path = os.path.join(source_folder, file_name)
        destination_path = os.path.join(Dev_folder, file_name)
        shutil.copy2(source_path, destination_path)

if __name__=="__main__":
    source_folder = "./DateSet/CPS_imgs/"
    destination_folder = "./DateSet/Train/"
    number_of_files_to_copy = 100

    #copy_random_files(source_folder, destination_folder, number_of_files_to_copy)
    copy_random_files2(source_folder, "./DateSet/Train/","./DateSet/Test/","./DateSet/Dev/")