from nnunetv2.configuration import default_num_processes
from nnunetv2.experiment_planning.plan_and_preprocess_api import extract_fingerprints, plan_experiments, preprocess


BRATSUMAMBA_NETWORK_CLASS = "dynamic_network_architectures.architectures.BraTS_UMamba_plus_plus.BraTSUMamba_plus_plus"


def _bratsumamba_architecture_dict():
    """
    Architecture metadata written into plans.json.

    Note:
    - n_stages/strides are kept in nnU-Net 6-stage form so that nnU-Net deep-supervision
      target downsampling remains: full, 1/2, 1/4, 1/8, 1/16.
    - The actual BraTS-UMamba++ AdM encoder is controlled by depths/dims and has 5 stages:
      16, 32, 64, 128, 256.
    """
    return {
        "network_class_name": BRATSUMAMBA_NETWORK_CLASS,
        "arch_kwargs": {
            "n_stages": 6,
            "features_per_stage": [16, 32, 64, 128, 256, 256],
            "conv_op": "torch.nn.modules.conv.Conv3d",
            "kernel_sizes": [[3, 3, 3]] * 6,
            "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            "n_conv_per_stage": [2, 2, 2, 2, 2, 2],
            "n_conv_per_stage_decoder": [2, 2, 2, 2, 2],
            "conv_bias": True,
            "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
            "norm_op_kwargs": {"eps": 1e-05, "affine": True},
            "dropout_op": None,
            "dropout_op_kwargs": None,
            "nonlin": "torch.nn.LeakyReLU",
            "nonlin_kwargs": {"inplace": True},
            # BraTS-UMamba++ specific options. These are consumed by BraTSUMamba_plus_plus.
            "depths": [2, 2, 2, 2, 2],
            "dims": [16, 32, 64, 128, 256],
            "final_decoder_channels": 8,
            "return_proto_info": False,
        },
        "_kw_requires_import": ["conv_op", "norm_op", "dropout_op", "nonlin"],
    }


def patch_plans_for_bratsumamba(dataset_ids, plans_identifier, configurations=("3d_fullres",), verbose=True):
    from batchgenerators.utilities.file_and_folder_operations import join, load_json, save_json, isfile
    from nnunetv2.paths import nnUNet_preprocessed
    from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

    if plans_identifier is None:
        plans_identifier = "nnUNetPlans"

    for dataset_id in dataset_ids:
        dataset_name = maybe_convert_to_dataset_name(dataset_id)
        plans_file = join(nnUNet_preprocessed, dataset_name, plans_identifier + ".json")
        if not isfile(plans_file):
            raise FileNotFoundError(f"Could not find plans file: {plans_file}")

        plans = load_json(plans_file)
        plans["plans_name"] = plans_identifier

        for cfg in configurations:
            if cfg not in plans.get("configurations", {}):
                if verbose:
                    print(f"[BraTSUMamba++ plans patch] Skip missing configuration: {dataset_name}/{cfg}")
                continue
            plans["configurations"][cfg]["architecture"] = _bratsumamba_architecture_dict()
            if verbose:
                patch_size = plans["configurations"][cfg].get("patch_size", None)
                print(f"[BraTSUMamba++ plans patch] Patched {dataset_name}/{cfg}: patch_size={patch_size}")

        save_json(plans, plans_file, sort_keys=False)
        if verbose:
            print(f"[BraTSUMamba++ plans patch] Saved: {plans_file}")


def extract_fingerprint_entry():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', nargs='+', type=int,
                        help="[REQUIRED] List of dataset IDs. Example: 2 4 5. This will run fingerprint extraction, experiment "
                             "planning and preprocessing for these datasets. Can of course also be just one dataset")
    parser.add_argument('-fpe', type=str, required=False, default='DatasetFingerprintExtractor',
                        help='[OPTIONAL] Name of the Dataset Fingerprint Extractor class that should be used. Default is '
                             '\'DatasetFingerprintExtractor\'.')
    parser.add_argument('-np', type=int, default=default_num_processes, required=False,
                        help=f'[OPTIONAL] Number of processes used for fingerprint extraction. '
                             f'Default: {default_num_processes}')
    parser.add_argument("--verify_dataset_integrity", required=False, default=False, action="store_true",
                        help="[RECOMMENDED] set this flag to check the dataset integrity. This is useful and should be done once for "
                             "each dataset!")
    parser.add_argument("--clean", required=False, default=False, action="store_true",
                        help='[OPTIONAL] Set this flag to overwrite existing fingerprints. If this flag is not set and a '
                             'fingerprint already exists, the fingerprint extractor will not run.')
    parser.add_argument('--verbose', required=False, action='store_true',
                        help='Set this to print a lot of stuff. Useful for debugging. Will disable progress bar! '
                             'Recommended for cluster environments')
    args, unrecognized_args = parser.parse_known_args()
    extract_fingerprints(args.d, args.fpe, args.np, args.verify_dataset_integrity, args.clean, args.verbose)


def plan_experiment_entry():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', nargs='+', type=int,
                        help="[REQUIRED] List of dataset IDs. Example: 2 4 5. This will run fingerprint extraction, experiment "
                             "planning and preprocessing for these datasets. Can of course also be just one dataset")
    parser.add_argument('-pl', type=str, default='ExperimentPlanner', required=False,
                        help='[OPTIONAL] Name of the Experiment Planner class that should be used. Default is '
                             '\'ExperimentPlanner\'.')
    parser.add_argument('-gpu_memory_target', default=None, type=float, required=False)
    parser.add_argument('-preprocessor_name', default='DefaultPreprocessor', type=str, required=False)
    parser.add_argument('-overwrite_target_spacing', default=None, nargs='+', required=False)
    parser.add_argument('-overwrite_plans_name', default=None, required=False)
    parser.add_argument('--bratsumamba_arch', default=False, action='store_true', required=False,
                        help='Patch generated 3d_fullres plans architecture to BraTS-UMamba++.')
    args, unrecognized_args = parser.parse_known_args()
    plans_identifier = plan_experiments(args.d, args.pl, args.gpu_memory_target, args.preprocessor_name,
                                        args.overwrite_target_spacing, args.overwrite_plans_name)
    if args.bratsumamba_arch:
        patch_plans_for_bratsumamba(args.d, plans_identifier, configurations=("3d_fullres",), verbose=True)


def preprocess_entry():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', nargs='+', type=int,
                        help="[REQUIRED] List of dataset IDs. Example: 2 4 5. This will run fingerprint extraction, experiment "
                             "planning and preprocessing for these datasets. Can of course also be just one dataset")
    parser.add_argument('-plans_name', default='nnUNetPlans', required=False,
                        help='[OPTIONAL] You can use this to specify a custom plans file that you may have generated')
    parser.add_argument('-c', required=False, default=['3d_fullres'], nargs='+',
                        help='[OPTIONAL] Configurations for which the preprocessing should be run. '
                             'For BraTS-UMamba++ the recommended default is 3d_fullres only.')
    parser.add_argument('-np', type=int, nargs='+', default=None, required=False)
    parser.add_argument('--verbose', required=False, action='store_true')
    args, unrecognized_args = parser.parse_known_args()
    if args.np is None:
        default_np = {"2d": 8, "3d_fullres": 4, "3d_lowres": 8}
        np = [default_np[c] if c in default_np.keys() else 4 for c in args.c]
    else:
        np = args.np
    preprocess(args.d, args.plans_name, configurations=args.c, num_processes=np, verbose=args.verbose)


def plan_and_preprocess_entry():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', nargs='+', type=int, default=[2020002],
                        help="[REQUIRED] List of dataset IDs. Example: 2 4 5. This will run fingerprint extraction, experiment "
                             "planning and preprocessing for these datasets. Can of course also be just one dataset")
    parser.add_argument('-fpe', type=str, required=False, default='DatasetFingerprintExtractor')
    parser.add_argument('-npfp', type=int, default=8, required=False)
    parser.add_argument("--verify_dataset_integrity", required=False, default=False, action="store_true")
    parser.add_argument('--no_pp', default=False, action='store_true', required=False)
    parser.add_argument("--clean", required=False, default=False, action="store_true")
    parser.add_argument('-pl', type=str, default='ExperimentPlanner', required=False)
    parser.add_argument('-gpu_memory_target', default=None, type=float, required=False)
    parser.add_argument('-preprocessor_name', default='DefaultPreprocessor', type=str, required=False)
    parser.add_argument('-overwrite_target_spacing', default=None, nargs='+', required=False)
    parser.add_argument('-overwrite_plans_name', default='nnUNetPlans_BraTSUMambaPlusPlus', required=False,
                        help='Recommended: write a separate plans file for BraTS-UMamba++ instead of overwriting nnUNetPlans.')
    parser.add_argument('-c', required=False, default=['3d_fullres'], nargs='+',
                        help='For BraTS-UMamba++ the recommended default is 3d_fullres only.')
    parser.add_argument('-np', type=int, nargs='+', default=None, required=False)
    parser.add_argument('--verbose', required=False, action='store_true')
    parser.add_argument('--no_bratsumamba_arch', default=False, action='store_true', required=False,
                        help='Do not patch plans architecture to BraTS-UMamba++.')
    args = parser.parse_args()

    print("Fingerprint extraction...")
    extract_fingerprints(args.d, args.fpe, args.npfp, args.verify_dataset_integrity, args.clean, args.verbose)

    print('Experiment planning...')
    plans_identifier = plan_experiments(args.d, args.pl, args.gpu_memory_target, args.preprocessor_name,
                                        args.overwrite_target_spacing, args.overwrite_plans_name)

    if not args.no_bratsumamba_arch:
        patch_plans_for_bratsumamba(args.d, plans_identifier, configurations=("3d_fullres",), verbose=True)

    if args.np is None:
        default_np = {"2d": 8, "3d_fullres": 4, "3d_lowres": 8}
        np = [default_np[c] if c in default_np.keys() else 4 for c in args.c]
    else:
        np = args.np

    if not args.no_pp:
        print('Preprocessing...')
        preprocess(args.d, plans_identifier, args.c, np, args.verbose)


if __name__ == '__main__':
    plan_and_preprocess_entry()
