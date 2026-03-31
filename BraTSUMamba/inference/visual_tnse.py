import argparse
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
from typing import Tuple, Union, List
from batchgenerators.utilities.file_and_folder_operations import join, isdir, maybe_mkdir_p, subdirs, load_json, \
    save_json
from BraTSUMamba.inference.predict_from_raw_data import nnUNetPredictor
from BraTSUMamba.utilities.file_path_utilities import get_output_folder
import SimpleITK as sitk


# ==========================================
# 1. 特征提取 Hook
# ==========================================
class FeatureHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.feature = None

    def hook_fn(self, module, input, output):
        # 自动处理输出可能是 tuple 的情况 (有些模块返回 (output, skip))
        if isinstance(output, tuple):
            self.feature = output[0].detach().cpu()
        else:
            self.feature = output.detach().cpu()

    def remove(self):
        self.hook.remove()


# ==========================================
# 2. 采样与 T-SNE 核心逻辑
# ==========================================
def sample_features(feature_map, mask_data, num_samples=300):
    """
    feature_map: [C, D, H, W]
    mask_data: [D_orig, H_orig, W_orig] (原始分辨率的 label)
    """
    C, Df, Hf, Wf = feature_map.shape

    # 将 Mask 缩放到特征图大小
    mask_tensor = torch.from_numpy(mask_data).float().unsqueeze(0).unsqueeze(0)
    mask_resized = torch.nn.functional.interpolate(
        mask_tensor, size=(Df, Hf, Wf), mode='nearest'
    ).squeeze().long()

    # 展平
    features_flat = feature_map.permute(1, 2, 3, 0).reshape(-1, C)
    labels_flat = mask_resized.reshape(-1)

    # 类别映射 (BraTS 默认)
    # 如果您的标签定义不同，请在此处修改
    label_dict = {
        0: 'Background',
        1: 'NCR (Core)',
        2: 'ED (Edema)',
        3: 'ET (Enhancing)'
    }

    sampled_X = []
    sampled_y = []

    unique_labels = torch.unique(labels_flat)

    for lbl in unique_labels:
        lbl_idx = lbl.item()
        if lbl_idx not in label_dict: continue

        # 找到该类别的所有像素索引
        indices = (labels_flat == lbl).nonzero(as_tuple=True)[0]

        # 采样策略：背景采样少一点，肿瘤采样多一点
        current_n = num_samples if lbl_idx != 0 else num_samples // 3

        if len(indices) > current_n:
            perm = torch.randperm(len(indices))[:current_n]
            indices = indices[perm]

        sampled_X.append(features_flat[indices])
        sampled_y.extend([label_dict[lbl_idx]] * len(indices))

    if not sampled_X:
        return None, None

    return torch.cat(sampled_X, dim=0), sampled_y


# ==========================================
# 3. 主流程
# ==========================================
def predict_and_visualize_entry_point():
    parser = argparse.ArgumentParser(
        description='Run inference and generating T-SNE visualization using existing nnU-Net arguments.')

    # --- 标准 nnU-Net 参数 ---
    parser.add_argument('-i', type=str, required=True, help='Input folder (imagesTs)')
    parser.add_argument('-o', type=str, required=True, help='Output folder')
    parser.add_argument('-d', type=str, required=True, help='Dataset ID (e.g., 1003)')
    parser.add_argument('-c', type=str, required=True, help='Configuration (e.g., 3d_fullres)')
    parser.add_argument('-f', nargs='+', type=str, required=False, default=(0, 1, 2, 3, 4), help='Folds')
    parser.add_argument('-chk', type=str, required=False, default='checkpoint_final.pth', help='Checkpoint name')
    parser.add_argument('-p', type=str, required=False, default='nnUNetPlans', help='Plans identifier')
    parser.add_argument('-tr', type=str, required=False, default='nnUNetTrainer', help='Trainer class')
    parser.add_argument('-device', type=str, default='cuda', required=False, help='Device')
    parser.add_argument('--disable_tta', action='store_true', help='Disable TTA')

    # --- T-SNE 特有参数 ---
    parser.add_argument('--layer', type=str, default='encoder.stages.5',
                        help='Layer to visualize (default: encoder.stages.5)')
    parser.add_argument('--max_cases', type=int, default=3,
                        help='Max number of cases to process for T-SNE to avoid OOM (default: 3)')
    parser.add_argument('--num_samples', type=int, default=500,
                        help='Number of points to sample per class per image')

    args = parser.parse_args()

    # 处理参数
    args.f = [i if i == 'all' else int(i) for i in args.f]
    maybe_mkdir_p(args.o)

    # 1. 自动定位 Ground Truth 文件夹
    # 假设输入文件夹叫 'imagesTs'，我们尝试找同级的 'labelsTs'
    gt_folder = None
    if 'imagesTs' in args.i:
        potential_gt = args.i.replace('imagesTs', 'labelsTs')
        if isdir(potential_gt):
            gt_folder = potential_gt
            print(f"[INFO] Auto-detected Ground Truth folder: {potential_gt}")
        else:
            print(f"[WARNING] Could not find 'labelsTs' folder. T-SNE will use PREDICTED labels for coloring.")
    else:
        print(f"[WARNING] Input folder does not contain 'imagesTs'. Cannot auto-guess labels folder.")

    # 2. 初始化 Predictor
    model_folder = get_output_folder(args.d, args.tr, args.p, args.c)
    device = torch.device(args.device)

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=not args.disable_tta,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )

    print(f"Loading model from: {model_folder}")
    predictor.initialize_from_trained_model_folder(
        model_folder,
        args.f,
        checkpoint_name=args.chk
    )

    # 3. 获取输入文件列表
    # nnU-Net 文件通常以 _0000.nii.gz 结尾，需要处理
    file_ending = predictor.dataset_json['file_ending']
    # 扫描文件夹
    all_files = [f for f in os.listdir(args.i) if f.endswith(file_ending) and "_0000" in f]
    # 提取 Case ID (去除 _0000 和后缀)
    case_ids = sorted(list(set([f.split("_0000")[0] for f in all_files])))

    print(f"Found {len(case_ids)} cases in {args.i}")

    # 限制处理数量，随机抽取
    if len(case_ids) > args.max_cases:
        import random
        selected_cases = random.sample(case_ids, args.max_cases)
        print(f"Randomly selected {args.max_cases} cases for T-SNE: {selected_cases}")
    else:
        selected_cases = case_ids

    # 4. 注册 Hook
    model = predictor.network
    model.to(device)
    target_layer = args.layer
    hook_handle = None

    for name, module in model.named_modules():
        if name == target_layer:
            hook_handle = FeatureHook(module)
            print(f"Hook registered on: {name}")
            break

    if hook_handle is None:
        raise ValueError(f"Could not find layer {target_layer} in model.")

    # 5. 循环推理并收集数据
    global_X = []
    global_y = []

    # 获取 Preprocessor
    preprocessor = predictor.configuration_manager.preprocessor_class(verbose=False)

    model.eval()

    for case_id in selected_cases:
        print(f"Processing {case_id}...")

        # 构造输入文件列表 (支持多模态)
        # 假设 dataset.json 里的 channel_names 决定了模态数量
        num_channels = len(predictor.dataset_json['channel_names'])
        input_files = []
        for c in range(num_channels):
            input_files.append(join(args.i, f"{case_id}_{c:04d}{file_ending}"))

        # 尝试加载 GT
        gt_file = None
        if gt_folder:
            potential_gt_file = join(gt_folder, f"{case_id}{file_ending}")
            if os.path.isfile(potential_gt_file):
                gt_file = [potential_gt_file]  # run_case expects list

        # 运行预处理
        # run_case 返回: data [C, D, H, W], seg [C, D, H, W] (if loaded), properties
        try:
            data, seg, data_properties = preprocessor.run_case(
                input_files,
                gt_file,  # 如果是 None，则 run_case 不会加载 seg
                predictor.plans_manager,
                predictor.configuration_manager,
                predictor.dataset_json
            )
        except Exception as e:
            print(f"Error preprocessing {case_id}: {e}")
            continue

        # 转换为 Tensor 并推理
        data_tensor = torch.from_numpy(data).unsqueeze(0).to(device)

        with torch.no_grad():
            # 我们只需要跑一次前向传播来触发 Hook，不需要完整的 sliding window 预测 (太慢且无法 hook)
            # 注意：如果显存不足，这里可能会炸。大图可能需要裁剪。
            # 为了 T-SNE，我们简单地取一个包含前景的 crop 或者直接输入（如果显存够）
            try:
                # 简单的前向传播
                _ = model(data_tensor)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(f"OOM error for {case_id}, skipping full forward pass. Trying center crop...")
                    torch.cuda.empty_cache()
                    # 简单的中心裁剪策略
                    D, H, W = data_tensor.shape[2:]
                    crop_size = 128
                    d_s, h_s, w_s = max(0, D // 2 - 64), max(0, H // 2 - 64), max(0, W // 2 - 64)
                    data_crop = data_tensor[:, :, d_s:d_s + crop_size, h_s:h_s + crop_size, w_s:w_s + crop_size]
                    _ = model(data_crop)
                    # 同时裁剪 seg
                    if seg is not None:
                        seg = seg[:, d_s:d_s + crop_size, h_s:h_s + crop_size, w_s:w_s + crop_size]
                else:
                    raise e

        # 获取特征
        features = hook_handle.feature[0]  # [C_feat, D, H, W]

        # 获取用于染色的 Label
        if seg is not None:
            # 使用 Ground Truth
            label_map = seg[0]  # [D, H, W]
        else:
            # 如果没有 GT，我们必须使用预测结果作为 Label
            # 这时候需要从 logits 计算 argmax，但上面的 forward 并没有返回完整的 logits (因为可能被 crop 了)
            # 为了简化，如果没有 GT，我们跳过这个 case 或者报错
            print(f"No Ground Truth found for {case_id}, skipping T-SNE sampling for this case.")
            continue

        # 采样
        X_sample, y_sample = sample_features(features, label_map, num_samples=args.num_samples)

        if X_sample is not None:
            global_X.append(X_sample)
            global_y.extend(y_sample)

    hook_handle.remove()

    # 6. 运行 T-SNE 并绘图
    if len(global_X) == 0:
        print("No samples collected. Check if labels match images or if images are all background.")
        return

    print("Running T-SNE on collected data...")
    X_all = torch.cat(global_X, dim=0).numpy()

    # 降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X_all)

    # 绘图
    df = pd.DataFrame({
        'tsne_1': X_embedded[:, 0],
        'tsne_2': X_embedded[:, 1],
        'Class': global_y
    })

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df, x='tsne_1', y='tsne_2', hue='Class',
        palette='bright', s=40, alpha=0.7
    )
    plt.title(f"Feature Space Visualization (Layer: {args.layer})\nDataset: {args.d} | Cases: {len(selected_cases)}")

    save_path = join(args.o, f'tsne_visualization_{args.layer.replace(".", "_")}.png')
    plt.savefig(save_path, dpi=300)
    print(f"Visualization saved to: {save_path}")


if __name__ == '__main__':
    predict_and_visualize_entry_point()