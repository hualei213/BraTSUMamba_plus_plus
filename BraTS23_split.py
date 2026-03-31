import os
import shutil


def split_dataset():
    # 1. 定义路径
    source_data_dir = '/mnt/SSD2/YHR/dataset/BraTS2023-GLI/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData'
    txt_dir = '/mnt/SSD2/YHR/dataset/BraTS2023-GLI/KFold/3'

    # 定义你要存放的新文件夹路径
    target_base_dir = '/mnt/SSD2/YHR/dataset/BraTS2023-GLI/BraTS2023_Fold_3'
    target_train_dir = os.path.join(target_base_dir, 'imagesTr')
    target_test_dir = os.path.join(target_base_dir, 'imagesTs')

    # 2. 创建目标文件夹
    os.makedirs(target_train_dir, exist_ok=True)
    os.makedirs(target_test_dir, exist_ok=True)

    # 3. 定义复制功能
    def copy_cases_from_txt(txt_filename, dest_dir):
        txt_path = os.path.join(txt_dir, txt_filename)

        if not os.path.exists(txt_path):
            print(f"❌ 找不到文件: {txt_path}")
            return

        with open(txt_path, 'r') as f:
            # 读取每一行，去除换行符和首尾空格
            cases = [line.strip() for line in f.readlines() if line.strip()]

        print(f"📄 开始根据 {txt_filename} 复制数据，共 {len(cases)} 个病例...")

        for case in cases:
            src_case_path = os.path.join(source_data_dir, case)
            dst_case_path = os.path.join(dest_dir, case)

            if os.path.exists(src_case_path):
                # 如果目标路径已经存在该文件夹，跳过以节省时间
                if not os.path.exists(dst_case_path):
                    shutil.copytree(src_case_path, dst_case_path)
                else:
                    print(f"⚠️ {case} 已存在于目标文件夹，跳过。")
            else:
                print(f"❌ 警告: 在原始数据中找不到 {case} 的文件夹！({src_case_path})")

        print(f"✅ {txt_filename} 的数据已成功复制到 {dest_dir}\n")

    # 4. 执行复制
    copy_cases_from_txt('train.txt', target_train_dir)
    copy_cases_from_txt('test.txt', target_test_dir)

    print("🎉 所有数据划分完毕！")


if __name__ == '__main__':
    split_dataset()