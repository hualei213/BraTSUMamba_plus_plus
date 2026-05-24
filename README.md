<table>
<tr>
<td align="left">
  <h1 style="margin: 0;">
    BraTS-UMamba++: A Consistency-Driven Framework for Multimodal Brain Tumor Segmentation
  </h1>
</td>
<td align="right">
  <img src='Fig_BraTSUMamba_plus_plus/BraTSUMamba++_logo.png' alt="logo" width="280">
</td>
</tr>
</table>
<p align="left">
  <a href="#overview">
    <img src="https://img.shields.io/badge/Overview-Read-orange.svg" alt="Overview">
  </a>
  <a href="#datasets">
    <img src="https://img.shields.io/badge/Datasets-Download-yellow.svg" alt="Datasets">
  </a>
  <a href="#training">
    <img src="https://img.shields.io/badge/Training-Evaluation-purple.svg" alt="Training and Evaluation">
  </a
</p>

<a id="overview"></a>

## Overview
<div align=center>  
<img src='Fig_BraTSUMamba_plus_plus/ararchitecture.png' width="100%">
</div>

(a) Overall architecture of **BraTS-UMamba++**, a consistency-driven framework for multimodal brain tumor segmentation. The encoder performs **topology-aware sequence modeling**, skip connections enable **semantic alignment via subregion-aware fusion**, and the decoder **refines reconstruction using tri-plane frequency-guided structural cues**, (b) Attention-Refined Tri-Plane Frequency Enhancement (AR-TPFE) module, and (c) Frequency Specific Attention Refinement (FSAR) block in the AR-TPFE module.

<div align=center>  
<img src='Fig_BraTSUMamba_plus_plus/Fig_all.png' width="100%">
</div>
Some experimental results for illustration. For more details, please refer to our paper.


---

## 🛠️ Install Dependencies
Clone this repo and install environment:
```
git clone https://github.com/hualei213/BraTSUMamba_plus_plus.git
conda create -n BraTSUMamba_plus_plus python=3.8
conda activate BraTSUMamba_plus_plus
pip install -r requirements.txt
```
<a id="datasets"></a>

## 📦 Datasets Preparation

### 1. Download
 Datasets Preparation
Please download and prepare the following training datasets:


- [MSD-BTS](https://decathlon-10.grand-challenge.org/) Brain Tumor (Task 01_BrainTumour)  is available from via [AWS](http://medicaldecathlon.com/dataaws/) or [Google Drive](https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2).
- [BraTS 2020](https://www.med.upenn.edu/cbica/brats2020/data.html) is also now available for download on [Kaggle](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation).
- To download [BraTS 2023-GLI](https://www.synapse.org/Synapse:syn51156910/wiki/621282), simply create an account on Synapse and register for the challenge on the official website.

### 2. Preprocessing


Example preprocessing code for Colorectal dataset:
```
import os
import shutil
import json

msd_dir = "/dataset/MSD/Task01_BrainTumour"
train_txt = "/dataset/MSD_KFold/2/train.txt"
test_txt = "/dataset/MSD_KFold/2/test.txt"

train_dir = "/BraTSUMambav2_raw/Task0002_MSD/imagesTr"
test_dir = "/BraTSUMambav2/Task0002_MSD/imagesTs"
label_dir = "/BraTSUMambav2/Task0002_MSD/labelsTr"

output_json_file = "/BraTSUMambav2/Task0002_MSD/dataset.json"
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(label_dir, exist_ok=True)
def read_file_list(file_path):
    with open(file_path, "r") as f:
        return [line.strip() for line in f]
train_files = read_file_list(train_txt)
test_files = read_file_list(test_txt)
def copy_files(file_list, dest_dir, is_train=False):
    for file_name in file_list:
        nii_file = os.path.join(msd_dir,"imagesTr", file_name + ".nii.gz")
        if os.path.exists(nii_file):
            shutil.copy(nii_file, os.path.join(dest_dir, file_name + ".nii.gz"))

        if is_train:
            label_file = os.path.join(msd_dir, "labelsTr", file_name + ".nii.gz")
            if os.path.exists(label_file):
                shutil.copy(label_file, os.path.join(label_dir, file_name + ".nii.gz"))
copy_files(train_files, train_dir, is_train=True)
copy_files(test_files, test_dir, is_train=False)
new_train_folder = train_dir
new_test_folder = test_dir
new_label_folder = label_dir
def read_file_list(file_path):
    with open(file_path, "r") as f:
        return [line.strip() for line in f]

train_files = read_file_list(train_txt)
test_files = read_file_list(test_txt)
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
for file_name in train_files:
    new_json["training"].append({
        "image": f"{new_train_folder}/{file_name}.nii.gz",
        "label": f"{new_label_folder}/{file_name}.nii.gz"
    })
for file_name in test_files:
    new_json["test"].append(f"{new_test_folder}/{file_name}.nii.gz")
with open(output_json_file, "w") as json_file:
    json.dump(new_json, json_file, indent=4)
```

**Usage**:

The above script reorganizes the original MSD-BTS dataset according to the predefined fold split. It copies the selected training and testing images into the corresponding `imagesTr` and `imagesTs` folders, copies the training labels into `labelsTr`, and generates the `dataset.json` file required for dataset description.

Then, `/BraTSUMamba_plus_plus/BraTSUMamba/dataset_conversion/convert_MSD_dataset.py` is used to convert the reorganized MSD-BTS dataset into the BraTS-UMamba++ compatible raw-data format.

Finally, `/BraTSUMamba_plus_plus/BraTSUMamba/experiment_planning/plan_and_preprocess_entrypoints_bratsumamba.py` is used to conduct dataset planning and preprocessing, including dataset integrity checking, plan generation, and preprocessing of the raw NIfTI images for subsequent model training.

<a id="training"></a>

## 🚀 Training & Evaluation
```
#!/bin/bash
set -e

# ==============================================================================
# BraTS-UMamba++ training / inference / evaluation script
# ==============================================================================

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=1

# ==============================================================================
# Basic settings
# ==============================================================================

TASK_ID=0002
CV="all"
DATASET="MSD"

CONFIG="3d_fullres"
TRAINER="nnUNetTrainerBraTSUMamba_plus_plus"
PLANS="nnUNetPlans_BraTSUMambaPlusPlus"

RAW_ROOT="/BraTSUMambav2_raw"
RESULTS_ROOT="/BraTSUMambav2_results"

INPUT_IMG_DIR="${RAW_ROOT}/Dataset${TASK_ID}_${DATASET}/imagesTs"
LABEL_DIR="${RAW_ROOT}/Dataset${TASK_ID}_${DATASET}/labelsTs"

OUTPUT_ROOT="${RESULTS_ROOT}/Dataset${TASK_ID}_${DATASET}"
CHECKPOINT_DIR="${OUTPUT_ROOT}/${TRAINER}__${PLANS}__${CONFIG}/fold_${CV}"


PRED_ROOT="${OUTPUT_ROOT}/predictions"

# ==============================================================================
# Train
# ==============================================================================

echo "################################################################"
echo "START TRAINING"
echo "Dataset    : Dataset${TASK_ID}_${DATASET}"
echo "Config     : ${CONFIG}"
echo "Fold       : ${CV}"
echo "Trainer    : ${TRAINER}"
echo "Plans      : ${PLANS}"
echo "################################################################"

python -m nnunetv2.run.run_training \
    ${TASK_ID} ${CONFIG} ${CV} \
    -tr ${TRAINER} \
    -p ${PLANS}

echo "Training finished."

# ==============================================================================
# Predict and evaluate checkpoints
# ==============================================================================

mkdir -p "${PRED_ROOT}"

for epoch in $(seq 50 50 1000); do
    echo "################################################################"
    echo "STARTING PROCESS FOR EPOCH: ${epoch}"
    echo "################################################################"

    CURRENT_CHECKPOINT="${CHECKPOINT_DIR}/checkpoint_epoch_${epoch}.pth"
    PRED_DIR="${PRED_ROOT}/predict_${TASK_ID}_epoch_${epoch}"

    if [ ! -f "${CURRENT_CHECKPOINT}" ]; then
        echo "[Skip] Checkpoint not found: ${CURRENT_CHECKPOINT}"
        continue
    fi

    echo "[Step 1/2] Running inference with checkpoint_epoch_${epoch}.pth"

    python -m nnunetv2.inference.predict_from_raw_data \
        -i "${INPUT_IMG_DIR}" \
        -o "${PRED_DIR}" \
        -d ${TASK_ID} \
        -c ${CONFIG} \
        -f ${CV} \
        -tr ${TRAINER} \
        -p ${PLANS} \
        -chk "checkpoint_epoch_${epoch}.pth"

    echo "[Step 2/2] Running evaluation"

    if [ -d "${LABEL_DIR}" ]; then
        python eval.py \
            --task_id "${TASK_ID}" \
            --model_epoch "${epoch}" \
            --label_dir "${LABEL_DIR}" \
            --pred_base_dir "${PRED_DIR}"
    else
        echo "[Warning] LABEL_DIR does not exist: ${LABEL_DIR}"
        echo "Skip evaluation for epoch ${epoch}."
    fi

    echo "Evaluation for epoch ${epoch} completed!"
    echo ""
done

echo "All epochs processed successfully!"
```

## Acknowledgments
Thank the authors of [nnU-Net](https://github.com/MIC-DKFZ/nnUNet), [Mamba](https://github.com/state-spaces/mamba) and [SegMamba](https://github.com/ge-xing/SegMamba) for making their valuable code publicly available.
