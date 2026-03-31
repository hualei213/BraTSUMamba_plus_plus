import os
import numpy as np
import glob
import SimpleITK as sitk
import csv
import argparse
from metric_ import dice, hd, get_percentile_distance


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate nnUNet model predictions')
    parser.add_argument('--task_id', type=int, required=True, help='Task ID')
    parser.add_argument('--model_epoch', type=str, required=True, help='Model epoch to evaluate')
    parser.add_argument('--label_dir', type=str, required=True, help='Directory containing ground truth labels')
    parser.add_argument('--pred_base_dir', type=str, required=True, help='Base directory for predictions')
    return parser.parse_args()


# Initialize lists for storing metrics
dice_list = []  # Dice list
hd95_list = []  # HD95 list
wt_dice = []
wt_hd95 = []
et_dice = []
et_hd95 = []
tc_dice = []
tc_hd95 = []


# Functions
def process_label(label):
    ncr = label == 1
    ed = label == 2
    et = label == 3
    ET = et
    TC = ncr + et
    WT = ncr + et + ed
    return ET, TC, WT


def get_test_list(test_label_dir):
    """
    Get list of test samples.
    :param test_label_dir: Directory containing labels of all test instances.
    :return: List containing IDs of test instances.
    """
    list_test_ids = []

    # Get path of all test label files
    glob_pattern = os.path.join(test_label_dir, '*.nii.gz')  # For .nii.gz files
    list_of_test_label_files = glob.glob(glob_pattern)

    for paths in list_of_test_label_files:
        # Get file base name
        file_basename = os.path.basename(paths)
        # Get id by removing the label suffix
        file_id = file_basename.replace('.nii.gz', '')  # 预测文件没有-label，所以直接去掉.nii.gz
        # Append file id to the list
        list_test_ids.append(file_id)

    return list_test_ids


def load_nifti_file(file_path):
    """
    Load .nii.gz file and return it as a numpy array.
    :param file_path: Path to the .nii.gz file.
    :return: numpy array containing the image data.
    """
    img = sitk.ReadImage(file_path)
    img_array = sitk.GetArrayFromImage(img)
    return img_array


def Evaluate(label_dir, pred_dir, pred_id, checkpoint_num):
    """
    Evaluate the performance for each subject.
    :param label_dir: Directory where the labels are stored.
    :param pred_dir: Directory where the predictions are stored.
    :param pred_id: Subject ID for which the evaluation is performed.
    :param checkpoint_num: The checkpoint number.
    :param overlap_step: Overlap step size.
    :return: None
    """
    print('Perform evaluation for subject-%s:' % pred_id)

    # Load label
    print('Loading label...')
    # 标签文件命名方式：{id}-label.nii.gz
    label_file = os.path.join(label_dir, pred_id + '.nii.gz')
    if not os.path.isfile(label_file):
        print(f'Label file {label_file} not found. Please generate the label file.')
        return

    label = load_nifti_file(label_file)
    print('Check label: ', label.shape, np.max(label))

    # Load prediction
    print('Loading prediction...')
    # 预测文件命名方式：{id}.nii.gz
    pred_file = os.path.join(pred_dir, pred_id + '.nii.gz')
    if not os.path.isfile(pred_file):
        print(f'Prediction file {pred_file} not found. Please generate the prediction result.')
        return

    pred = load_nifti_file(pred_file)
    print('Check pred: ', pred.shape, np.max(pred))

    # Process label and prediction for each class
    label_et, label_tc, label_wt = process_label(label)
    infer_et, infer_tc, infer_wt = process_label(pred)

    # Calculate Dice scores
    dice_et = dice(infer_et, label_et)
    dice_tc = dice(infer_tc, label_tc)
    dice_wt = dice(infer_wt, label_wt)

    # Calculate Hausdorff distance (95th percentile)
    hd95_et = hd(infer_et, label_et)
    hd95_tc = hd(infer_tc, label_tc)
    hd95_wt = hd(infer_wt, label_wt)

    # Append to lists
    et_dice.append(dice_et)
    tc_dice.append(dice_tc)
    wt_dice.append(dice_wt)

    et_hd95.append(hd95_et)
    tc_hd95.append(hd95_tc)
    wt_hd95.append(hd95_wt)

    # Calculate average Dice and HD95 scores
    avg_dice = (dice_et + dice_tc + dice_wt) / 3
    avg_hd = (hd95_et + hd95_tc + hd95_wt) / 3

    # Print the results for each sample
    print("\tDice: {}, HD95: {}".format(avg_dice, avg_hd))

    # Append to global lists for averaging
    dice_list.append(avg_dice)
    hd95_list.append(avg_hd)

    # Save results to CSV file
    csv_file_path = os.path.join(pred_dir, 'results_stat_%d.csv' % checkpoint_num)
    if not os.path.isfile(csv_file_path):
        f = open(csv_file_path, "w+")
        writer = csv.writer(f)
        writer.writerow(["scan ID", "Dice", "95 Percentile Distance"])
        f.close()

    with open(csv_file_path, "a+") as f:
        writer = csv.writer(f)
        writer.writerow([str(pred_id), str(avg_dice), str(avg_hd)])

    print('Evaluation for subject-%s completed.' % pred_id)


def get_evl(pred_dir, checkpoint_num, method_name, param_list):
    """
    Get evaluation metrics for a given method.
    :param pred_dir: Prediction directory to save results
    :param checkpoint_num: Checkpoint number
    :param method_name: Name of the evaluation method.
    :param param_list: List of evaluation values (e.g., Dice scores).
    :return: mean, std, max, min of the evaluation metrics.
    """
    mean = np.mean(param_list)
    std = np.std(param_list)
    max_val = np.max(param_list)
    min_val = np.min(param_list)

    # Save summary to CSV file
    csv_file_path = os.path.join(pred_dir, 'results_summary_%d.csv' % checkpoint_num)
    if not os.path.isfile(csv_file_path):
        f = open(csv_file_path, "w+")
        writer = csv.writer(f)
        writer.writerow(["Method", "Mean", "Std", "Max", "Min"])
        f.close()

    with open(csv_file_path, "a+") as f:
        writer = csv.writer(f)
        writer.writerow([str(method_name), str(mean), str(std), str(max_val), str(min_val)])

    return mean, std, max_val, min_val


if __name__ == '__main__':
    args = parse_args()

    # Set parameters from command line arguments
    task_id = args.task_id
    model_epoch = args.model_epoch
    LABEL_DIR = args.label_dir
    PRED_DIR = args.pred_base_dir  # 直接使用传入的预测目录
    CHECKPOINT_NUM = task_id

    print(f"Evaluating model: task_id={task_id}, epoch={model_epoch}")
    print(f"Label directory: {LABEL_DIR}")
    print(f"Prediction directory: {PRED_DIR}")

    # Check if prediction directory exists
    if not os.path.exists(PRED_DIR):
        print(f"Error: Prediction directory {PRED_DIR} does not exist!")
        exit(1)

    # Get the test sample IDs from prediction directory
    PRED_ID = get_test_list(PRED_DIR)

    # Loop through each test sample and perform evaluation
    for pred_id_single in PRED_ID:
        Evaluate(
            label_dir=LABEL_DIR,
            pred_dir=PRED_DIR,
            pred_id=pred_id_single,
            checkpoint_num=CHECKPOINT_NUM)

    # Get overall evaluation statistics
    print("\n=== Overall Evaluation Results ===")
    print("\tmean\tstd\tmax\tmin")
    wt_mdice, std, max_data, min_data = get_evl(PRED_DIR, CHECKPOINT_NUM, "WT_DICE", wt_dice)
    print("{}\t\t\t{}\t{}\t{}\t{}".format("WT_DICE", wt_mdice, std, max_data, min_data))
    wt_mhd95, std, max_data, min_data = get_evl(PRED_DIR, CHECKPOINT_NUM, "WT_HD95", wt_hd95)
    print("{}\t\t\t{}\t{}\t{}\t{}".format("WT_HD95", wt_mhd95, std, max_data, min_data))

    tc_mdice, std, max_data, min_data = get_evl(PRED_DIR, CHECKPOINT_NUM, "TC_DICE", tc_dice)
    print("{}\t\t\t{}\t{}\t{}\t{}".format("TC_DICE", tc_mdice, std, max_data, min_data))
    tc_mhd95, std, max_data, min_data = get_evl(PRED_DIR, CHECKPOINT_NUM, "TC_HD95", tc_hd95)
    print("{}\t\t\t{}\t{}\t{}\t{}".format("TC_HD95", tc_mhd95, std, max_data, min_data))

    et_mdice, std, max_data, min_data = get_evl(PRED_DIR, CHECKPOINT_NUM, "ET_DICE", et_dice)
    print("{}\t\t\t{}\t{}\t{}\t{}".format("ET_DICE", et_mdice, std, max_data, min_data))
    et_mhd95, std, max_data, min_data = get_evl(PRED_DIR, CHECKPOINT_NUM, "ET_HD95", et_hd95)
    print("{}\t\t\t{}\t{}\t{}\t{}".format("ET_HD95", et_mhd95, std, max_data, min_data))

    mean, std, max_data, min_data = get_evl(PRED_DIR, CHECKPOINT_NUM, "DICE", dice_list)
    print("{}\t\t\t{}\t{}\t{}\t{}".format("DICE", mean, std, max_data, min_data))

    mean, std, max_data, min_data = get_evl(PRED_DIR, CHECKPOINT_NUM, "95 percentile Distance", hd95_list)
    print("{}\t{}\t{}\t{}\t{}".format("95 percentile Distance", mean, std, max_data, min_data))
