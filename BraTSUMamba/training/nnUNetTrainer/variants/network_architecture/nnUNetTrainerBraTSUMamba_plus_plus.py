from copy import deepcopy
from pydoc import locate
from typing import List, Tuple, Union

import numpy as np
import torch
from torch import autocast, nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import dummy_context

from dynamic_network_architectures.architectures.BraTS_UMamba_plus_plus import (
    BraTSUMamba_plus_plus,
    LabelGuidedPrototypeLoss,
)


class nnUNetTrainerBraTSUMamba_plus_plus(nnUNetTrainer):

    def __init__(
            self,
            plans: dict,
            configuration: str,
            fold: int,
            dataset_json: dict,
            device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)

        # Register additional logging keys used by this trainer.
        # nnUNetLogger.log only accepts pre-existing list keys.
        self.logger.my_fantastic_logging.setdefault("train_losses_seg", [])
        self.logger.my_fantastic_logging.setdefault("train_losses_lgpc", [])

        self.lgpc_weight = 0.5

        self.bratsumamba_depths = (2, 2, 2, 2, 2)
        self.bratsumamba_dims = (16, 32, 64, 128, 256)
        self.bratsumamba_final_decoder_channels = 8

    def _do_i_compile(self):
        return False

    def initialize(self):
        super().initialize()

        self.proto_loss_func = LabelGuidedPrototypeLoss(
            num_classes=self.label_manager.num_segmentation_heads,
            ignore_index=0,
            delta_pos=0.9,
            delta_neg=0.1,
            lambda_align=1.0,
            lambda_sep=1.0,
            min_pixels=1,
        )

    @staticmethod
    def _maybe_import(value):
        if isinstance(value, str):
            obj = locate(value)
            if obj is None:
                # Common short-name fallbacks.
                if hasattr(nn, value):
                    return getattr(nn, value)
                raise ImportError(f"Cannot locate class/function from string: {value}")
            return obj
        return value

    @staticmethod
    def build_network_architecture(
            architecture_class_name: str,
            arch_init_kwargs: dict,
            arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
            num_input_channels: int,
            num_output_channels: int,
            enable_deep_supervision: bool = True,
    ) -> nn.Module:
        kwargs = deepcopy(arch_init_kwargs)
        for k in arch_init_kwargs_req_import:
            if k in kwargs:
                kwargs[k] = nnUNetTrainerBraTSUMamba_plus_plus._maybe_import(kwargs[k])

        # Some plans may not explicitly store these after customization, so use safe defaults.
        n_stages = kwargs.get("n_stages", len(kwargs.get("features_per_stage", (16, 32, 64, 128, 256))))
        features_per_stage = kwargs.get("features_per_stage", (16, 32, 64, 128, 256))
        conv_op = kwargs.get("conv_op", nn.Conv3d)
        kernel_sizes = kwargs.get("kernel_sizes", 3)
        strides = kwargs.get("strides", (2, 2, 2, 2, 2))
        n_conv_per_stage = kwargs.get("n_conv_per_stage", (2,) * n_stages)
        n_conv_per_stage_decoder = kwargs.get("n_conv_per_stage_decoder", (2,) * max(1, n_stages - 1))

        network = BraTSUMamba_plus_plus(
            input_channels=num_input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_conv_per_stage=n_conv_per_stage,
            num_classes=num_output_channels,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            conv_bias=kwargs.get("conv_bias", False),
            norm_op=kwargs.get("norm_op", nn.InstanceNorm3d),
            norm_op_kwargs=kwargs.get("norm_op_kwargs", {"eps": 1e-5, "affine": True}),
            dropout_op=kwargs.get("dropout_op", None),
            dropout_op_kwargs=kwargs.get("dropout_op_kwargs", None),
            nonlin=kwargs.get("nonlin", nn.LeakyReLU),
            nonlin_kwargs=kwargs.get("nonlin_kwargs", {"negative_slope": 1e-2, "inplace": True}),
            deep_supervision=enable_deep_supervision,
            nonlin_first=kwargs.get("nonlin_first", False),
            num_modalities=num_input_channels,
            num_regions=num_output_channels,
            depths=(2, 2, 2, 2, 2),
            dims=(16, 32, 64, 128, 256),
            final_decoder_channels=8,
            return_proto_info=True,
        )
        network.apply(network.initialize)
        return network

    def train_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)

            # Training mode + return_proto_info=True: output should be (seg_outputs, proto_info).
            if isinstance(output, tuple) and len(output) == 2:
                output_seg, proto_info = output
            else:
                output_seg, proto_info = output, None

            loss_seg = self.loss(output_seg, target)

            if proto_info is not None:
                loss_lgpc = self.proto_loss_func(proto_info, target)
            else:
                loss_lgpc = loss_seg.new_zeros(())

            loss = loss_seg + self.lgpc_weight * loss_lgpc

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {
            "loss": loss.detach().cpu().numpy(),
            "loss_seg": loss_seg.detach().cpu().numpy(),
            "loss_lgpc": loss_lgpc.detach().cpu().numpy(),
        }

    def on_train_epoch_end(self, train_outputs: List[dict]):
        from nnunetv2.utilities.collate_outputs import collate_outputs
        import torch.distributed as dist

        outputs = collate_outputs(train_outputs)

        if self.is_ddp:
            losses_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_tr, outputs["loss"])
            loss_here = np.vstack(losses_tr).mean()

            losses_seg = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_seg, outputs["loss_seg"])
            loss_seg_here = np.vstack(losses_seg).mean()

            losses_lgpc = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_lgpc, outputs["loss_lgpc"])
            loss_lgpc_here = np.vstack(losses_lgpc).mean()
        else:
            loss_here = np.mean(outputs["loss"])
            loss_seg_here = np.mean(outputs["loss_seg"])
            loss_lgpc_here = np.mean(outputs["loss_lgpc"])

        self.logger.my_fantastic_logging.setdefault("train_losses_seg", [])
        self.logger.my_fantastic_logging.setdefault("train_losses_lgpc", [])

        self.logger.log("train_losses", loss_here, self.current_epoch)
        self.logger.log("train_losses_seg", loss_seg_here, self.current_epoch)
        self.logger.log("train_losses_lgpc", loss_lgpc_here, self.current_epoch)

        self.print_to_log_file(
            f"train_loss_seg {np.round(loss_seg_here, decimals=4)}",
            f"train_loss_lgpc {np.round(loss_lgpc_here, decimals=4)}",
            also_print_to_console=False,
        )
