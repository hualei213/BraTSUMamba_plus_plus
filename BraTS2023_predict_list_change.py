import multiprocessing
import os
from multiprocessing import Pool
import SimpleITK as sitk
import numpy as np
# 注意这里增加了 subdirs 的导入
from batchgenerators.utilities.file_and_folder_operations import join, subfiles, maybe_mkdir_p, subdirs


def convert_2023_labels_back(seg: np.ndarray):
    """专门针对 BraTS 2023 的逆向转换：只互换 1(ED) 和 2(NCR)，3(ET) 不变"""
    new_seg = np.zeros_like(seg)
    new_seg[seg == 1] = 2  # 把模型预测的 1(水肿) 变回官方的 2
    new_seg[seg == 2] = 1  # 把模型预测的 2(坏死) 变回官方的 1
    new_seg[seg == 3] = 3  # 3(增强) 保持不变
    return new_seg


def load_convert_labels_back_to_BraTS23(filename, input_folder, output_folder):
    a = sitk.ReadImage(join(input_folder, filename))
    b = sitk.GetArrayFromImage(a)
    c = convert_2023_labels_back(b)
    d = sitk.GetImageFromArray(c)
    d.CopyInformation(a)
    sitk.WriteImage(d, join(output_folder, filename))


def restore_brats23_predictions_multiprocess(input_folder: str, output_folder: str, num_processes: int = 8):
    """多进程批量处理文件夹内的预测结果"""
    maybe_mkdir_p(output_folder)
    nii = subfiles(input_folder, suffix='.nii.gz', join=False)

    if len(nii) == 0:
        print(f"⚠️ 警告: 文件夹 {input_folder} 中没有找到 .nii.gz 文件，跳过。")
        return

    print(f"🚀 开始多进程还原 {len(nii)} 个 BraTS 2023 预测结果...")
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        p.starmap(load_convert_labels_back_to_BraTS23, zip(nii, [input_folder] * len(nii), [output_folder] * len(nii)))
    print(f"✅ 还原完成！结果保存在: {output_folder}")


if __name__ == '__main__':
    # 你存放所有预测结果文件夹的父目录
    base_dir = '/mnt/HDLV1/YHR/nnUNet-BraTSUMamba/nnUNet_results/Dataset2023004_BraTS2023_GLI'

    # 使用 batchgenerators 的 subdirs 自动获取所有以 'predict_' 开头的文件夹名
    # 如果你的系统环境里有其他非模型文件夹，prefix='predict_' 会精准过滤出你需要处理的目标
    pred_folders = subdirs(base_dir, prefix='predict_', join=False)

    if not pred_folders:
        print(f"❌ 在 {base_dir} 下没有找到任何以 'predict_' 开头的文件夹。请检查路径。")
    else:
        print(f"🔍 共找到 {len(pred_folders)} 个模型预测文件夹需要处理。")

        for folder_name in pred_folders:
            print(f"\n--- 正在处理: {folder_name} ---")

            # 自动拼接输入路径
            pred_dir = join(base_dir, folder_name)

            # 自动拼接输出路径，加上 'changed_' 前缀
            out_dir = join(base_dir, f"changed/{folder_name}")

            # 调用转换函数
            restore_brats23_predictions_multiprocess(pred_dir, out_dir, num_processes=8)

        print("\n🎉 所有预测文件夹的数据已全部批量转换完毕！")