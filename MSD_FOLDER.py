import os
import shutil
import json

# 文件夹路径
msd_dir = "/mnt/SSD2/YHR/dataset/MSD/Task01_BrainTumour"  # MSD数据集根目录
train_txt = "/mnt/SSD2/YHR/dataset/MSD_KFold/0/train.txt"
test_txt = "/mnt/SSD2/YHR/dataset/MSD_KFold/0/test.txt"

# 新的文件夹
train_dir = "/mnt/HDLV1/YHR/BraTSUMamba/BraTSUMamba_raw/Task0001_MSD/imagesTr"
test_dir = "/mnt/HDLV1/YHR/BraTSUMamba/BraTSUMamba_raw/Task0001_MSD/imagesTs"
label_dir = "/mnt/HDLV1/YHR/BraTSUMamba/BraTSUMamba_raw/Task0001_MSD/labelsTr"  # 新的标签文件夹路径

# 保存新的 JSON 文件
output_json_file = "/mnt/HDLV1/YHR/BraTSUMamba/BraTSUMamba_raw/Task0001_MSD/dataset.json"

# 确保新的文件夹存在
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(label_dir, exist_ok=True)  # 创建标签文件夹


# 从train.txt和test.txt中读取文件名
def read_file_list(file_path):
    with open(file_path, "r") as f:
        return [line.strip() for line in f]


train_files = read_file_list(train_txt)
test_files = read_file_list(test_txt)


# 复制文件
def copy_files(file_list, dest_dir, is_train=False):
    for file_name in file_list:
        # 原文件路径
        nii_file = os.path.join(msd_dir,"imagesTr", file_name + ".nii.gz")
        if os.path.exists(nii_file):
            # 复制到目标文件夹
            shutil.copy(nii_file, os.path.join(dest_dir, file_name + ".nii.gz"))

        if is_train:
            # 如果是train.txt，还需要复制label文件到新的标签文件夹
            label_file = os.path.join(msd_dir, "labelsTr", file_name + ".nii.gz")  # 假设标签文件夹位于MSD的根目录下
            if os.path.exists(label_file):
                shutil.copy(label_file, os.path.join(label_dir, file_name + ".nii.gz"))


# 复制train和test数据
copy_files(train_files, train_dir, is_train=True)
copy_files(test_files, test_dir, is_train=False)

print("文件复制完成！")



# 设置文件路径

new_train_folder = train_dir
new_test_folder = test_dir
new_label_folder = label_dir

# 读取 train.txt 和 test.txt 中的文件名
def read_file_list(file_path):
    with open(file_path, "r") as f:
        return [line.strip() for line in f]

train_files = read_file_list(train_txt)
test_files = read_file_list(test_txt)

# 创建新的 JSON 结构
new_json = {
    "name": "BRATS",
    "description": "Gliomas segmentation tumour and oedema in brain images",
    "reference": "https://www.med.upenn.edu/sbia/brats2017.html",
    "licence": "CC-BY-SA 4.0",
    "release": "2.0 04/05/2018",
    "tensorImageSize": "4D",
    "modality": {
        "0": "T1c",
        "1": "T1n",
        "2": "Flair",
        "3": "T2w"
    },
    "labels": {
        "0": "background",
        "1": "edema",
        "2": "non-enhancing tumor",
        "3": "enhancing tumour"
    },
    "numTraining": len(train_files),
    "numTest": len(test_files),
    "training": [],
    "test": []
}

# 更新 training 部分
for file_name in train_files:
    new_json["training"].append({
        "image": f"{new_train_folder}/{file_name}.nii.gz",
        "label": f"{new_label_folder}/{file_name}.nii.gz"
    })

# 更新 test 部分
for file_name in test_files:
    new_json["test"].append(f"{new_test_folder}/{file_name}.nii.gz")


with open(output_json_file, "w") as json_file:
    json.dump(new_json, json_file, indent=4)

print("JSON 文件已成功生成！")

