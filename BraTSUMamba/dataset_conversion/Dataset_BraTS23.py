import os
import shutil
from batchgenerators.utilities.file_and_folder_operations import join, subdirs, maybe_mkdir_p
from BraTSUMamba.dataset_conversion.generate_dataset_json import generate_dataset_json
from BraTSUMamba.paths import nnUNet_raw

if __name__ == '__main__':
    brats23_data_dir = '/mnt/SSD2/YHR/dataset/BraTS2023-GLI/BraTS2023_Fold_3/imagesTr'

    # 2. 设定 nnU-Net 任务 ID 和名称
    task_id = 2023004
    task_name = "BraTS2023_GLI"

    foldername = "Dataset%03.0d_%s" % (task_id, task_name)

    # 3. 建立 nnU-Net 标准文件夹结构
    out_base = join(nnUNet_raw, foldername)
    imagestr = join(out_base, "imagesTr")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)

    # 获取所有病例文件夹名
    case_ids = subdirs(brats23_data_dir, join=False)

    print(f"🚀 开始物理拷贝数据至 {out_base}，共 {len(case_ids)} 个病例...")

    for c in case_ids:
        # 拷贝 4 个模态图像 (注意 23 版是短横线命名，如 -t1n.nii.gz)
        shutil.copy(join(brats23_data_dir, c, c + "-t1n.nii.gz"), join(imagestr, c + '_0000.nii.gz'))
        shutil.copy(join(brats23_data_dir, c, c + "-t1c.nii.gz"), join(imagestr, c + '_0001.nii.gz'))
        shutil.copy(join(brats23_data_dir, c, c + "-t2w.nii.gz"), join(imagestr, c + '_0002.nii.gz'))
        shutil.copy(join(brats23_data_dir, c, c + "-t2f.nii.gz"), join(imagestr, c + '_0003.nii.gz'))

        # 直接拷贝标签，不打开文件，不修改数值！
        shutil.copy(join(brats23_data_dir, c, c + "-seg.nii.gz"), join(labelstr, c + '.nii.gz'))

    print("✅ 数据拷贝完成，正在生成 dataset.json ...")

    # 4. 生成配置文件 (直接定义评价区域，模型会自动用 Dice Loss 优化这些区域)
    generate_dataset_json(
        out_base,
        channel_names={0: 'T1n', 1: 'T1c', 2: 'T2w', 3: 'T2f'},
        labels={
            'background': 0,
            'whole tumor': (1, 2, 3),   # WT = NCR(1) + ED(2) + ET(3)
            'tumor core': (1, 3),       # TC = NCR(1) + ET(3) (非连续组合，完美支持)
            'enhancing tumor': (3, )    # ET = ET(3)
        },
        num_training_cases=len(case_ids),
        file_ending='.nii.gz',
        regions_class_order=(1, 2, 3),
        license='BraTS',
        reference='BraTS 2023 GLI',
        dataset_release='1.0'
    )

    print("🎉 dataset.json 生成成功！")