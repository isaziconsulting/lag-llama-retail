# Copyright 2024 Arjun Ashok
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, Iterable, List, Optional

import pytorch_lightning as pl
import torch

from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import as_stacked_batches
from gluonts.dataset.stat import calculate_dataset_statistics
from gluonts.itertools import Cyclic
from gluonts.time_feature import (
    get_lags_for_frequency,
    time_features_from_frequency_str,
)
from gluonts.torch.distributions import StudentTOutput, NegativeBinomialOutput
from gluonts.torch.model.estimator import PyTorchLightningEstimator
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.torch.modules.loss import DistributionLoss, NegativeLogLikelihood
from gluonts.transform import (
    AddObservedValuesIndicator,
    AddTimeFeatures,
    Chain,
    DummyValueImputation,
    ExpectedNumInstanceSampler,
    RemoveFields,
    InstanceSampler,
    InstanceSplitter,
    TestSplitSampler,
    Transformation,
    ValidationSplitSampler
)

from ...gluon_utils.gluon_ts_distributions.implicit_quantile_network import (
    ImplicitQuantileNetworkOutput,
)
from ..gluon.lightning_module import LagLlamaLightningModule

PREDICTION_INPUT_NAMES = [
    "past_target",
    "past_observed_values",
    "past_time_feat",
    "future_time_feat",
    "past_feat_dynamic_real",
    "future_feat_dynamic_real"
]
TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
    "future_target",
    "future_observed_values",
]


class LagLlamaEstimator(PyTorchLightningEstimator):
    """
    An estimator training a ConvTSMixer model for forecasting.

    This class is uses the model defined in ``ConvTSMixerModel``,
    and wraps it into a ``ConvTSMixerLightningModule`` for training
    purposes: training is performed using PyTorch Lightning's ``pl.Trainer``
    class.

    Parameters
    ----------
    prediction_length
        Length of the prediction horizon.
    context_length
        Number of time steps prior to prediction time that the model
        takes as inputs (default: ``10 * prediction_length``).
    lr
        Learning rate (default: ``1e-3``).
    weight_decay
        Weight decay regularization parameter (default: ``1e-8``).
    distr_output
        Distribution to use to evaluate observations and sample predictions
        (default: StudentTOutput()).
    loss
        Loss to be optimized during training
        (default: ``NegativeLogLikelihood()``).
    batch_norm
        Whether to apply batch normalization.
    batch_size
        The size of the batches to be used for training (default: 32).
    num_batches_per_epoch
        Number of batches to be processed in each training epoch
            (default: 50).
    trainer_kwargs
        Additional arguments to provide to ``pl.Trainer`` for construction.
    train_sampler
        Controls the sampling of windows during training.
    validation_sampler
        Controls the sampling of windows during validation.
    """

    @validated()
    def __init__(
        self,
        prediction_length: int,
        context_length: Optional[int] = None,
        input_size: int = 1,
        n_layer: int = 1,
        n_embd_per_head: int = 32,
        n_head: int = 4,
        max_context_length: int = 2048,
        rope_scaling=None,
        scaling: Optional[str] = "mean",
        lr: float = 1e-3,
        weight_decay: float = 1e-8,
        # Augmentations arguments
        aug_prob: float = 0.1,
        freq_mask_rate: float = 0.1,
        freq_mixing_rate: float = 0.1,
        jitter_prob: float = 0.0,
        jitter_sigma: float = 0.03,
        scaling_prob: float = 0.0,
        scaling_sigma: float = 0.1,
        rotation_prob: float = 0.0,
        permutation_prob: float = 0.0,
        permutation_max_segments: int = 5,
        permutation_seg_mode: str = "equal",
        magnitude_warp_prob: float = 0.0,
        magnitude_warp_sigma: float = 0.2,
        magnitude_warp_knot: int = 4,
        time_warp_prob: float = 0.0,
        time_warp_sigma: float = 0.2,
        time_warp_knot: int = 4,
        window_slice_prob: float = 0.0,
        window_slice_reduce_ratio: float = 0.9,
        window_warp_prob: float = 0.0,
        window_warp_window_ratio: float = 0.1,
        window_warp_scales: list = [0.5, 2.0],
        # Continuning model arguments
        distr_output: str = "studentT",
        loss: DistributionLoss = NegativeLogLikelihood(),
        num_parallel_samples: int = 100,
        batch_size: int = 32,
        num_batches_per_epoch: int = 50,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        train_sampler: Optional[InstanceSampler] = None,
        validation_sampler: Optional[InstanceSampler] = None,
        time_feat: bool = False,
        num_feat_dynamic_real: int = 0,
        dropout: float = 0.0,
        lags_seq: list = ["Q", "M", "W", "D", "H", "T", "S"],
        data_id_to_name_map: dict = {},
        use_cosine_annealing_lr: bool = False,
        cosine_annealing_lr_args: dict = {},
        track_loss_per_series: bool = False,
        ckpt_path: Optional[str] = None,
        partial_weights_ckpt_path: Optional[str] = None,
        freeze_transformer: bool = False,
        nonnegative_pred_samples: bool = False,
        device: torch.device = torch.device("cuda"),
        model_config=None,
        dataset_configs=[]
    ) -> None:
        default_trainer_kwargs = {"max_epochs": 100}
        if trainer_kwargs is not None:
            default_trainer_kwargs.update(trainer_kwargs)
        super().__init__(trainer_kwargs=default_trainer_kwargs)

        self.scaling = scaling
        self.input_size = input_size
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.max_context_length = max_context_length
        self.lags_seq = compute_lag_indices(lags_seq)

        self.n_head = n_head
        self.n_layer = n_layer
        self.n_embd_per_head = n_embd_per_head
        self.rope_scaling = rope_scaling

        self.lr = lr
        self.weight_decay = weight_decay
        if distr_output == "studentT":
            distr_output = StudentTOutput()
        elif distr_output == "neg_bin":
            distr_output = NegativeBinomialOutput()
        elif distr_output == "iqn":
            distr_output = ImplicitQuantileNetworkOutput()
        self.distr_output = distr_output
        self.num_parallel_samples = num_parallel_samples
        self.loss = loss
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.nonnegative_pred_samples = nonnegative_pred_samples

        self.train_sampler = train_sampler or ExpectedNumInstanceSampler(
            num_instances=1.0,
            min_future=prediction_length,
            min_instances=1,
        )
        self.validation_sampler = validation_sampler or ValidationSplitSampler(
            min_future=prediction_length
        )

        self.aug_prob = aug_prob
        self.freq_mask_rate = freq_mask_rate
        self.freq_mixing_rate = freq_mixing_rate
        self.jitter_prob = jitter_prob
        self.jitter_sigma = jitter_sigma
        self.scaling_prob = scaling_prob
        self.scaling_sigma = scaling_sigma
        self.rotation_prob = rotation_prob
        self.permutation_prob = permutation_prob
        self.permutation_max_segments = permutation_max_segments
        self.permutation_seg_mode = permutation_seg_mode
        self.magnitude_warp_prob = magnitude_warp_prob
        self.magnitude_warp_sigma = magnitude_warp_sigma
        self.magnitude_warp_knot = magnitude_warp_knot
        self.time_warp_prob = time_warp_prob
        self.time_warp_sigma = time_warp_sigma
        self.time_warp_knot = time_warp_knot
        self.window_slice_prob = window_slice_prob
        self.window_slice_reduce_ratio = window_slice_reduce_ratio
        self.window_warp_prob = window_warp_prob
        self.window_warp_window_ratio = window_warp_window_ratio
        self.window_warp_scales = window_warp_scales
        self.track_loss_per_series = track_loss_per_series

        self.time_feat = time_feat
        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.dropout = dropout
        self.data_id_to_name_map = data_id_to_name_map
        # Cannot set both a full checkpoint and partial weights checkpoint
        assert ckpt_path is None or partial_weights_ckpt_path is None
        self.ckpt_path = ckpt_path
        self.partial_weights_ckpt_path = partial_weights_ckpt_path
        self.freeze_transformer = freeze_transformer

        self.use_cosine_annealing_lr = use_cosine_annealing_lr
        self.cosine_annealing_lr_args = cosine_annealing_lr_args
        self.device = device
        self.model_config = model_config
        self.dataset_configs = dataset_configs

    @classmethod
    def derive_auto_fields(cls, train_iter):
        stats = calculate_dataset_statistics(train_iter)

        return {
            "num_feat_dynamic_real": stats.num_feat_dynamic_real,
            "num_feat_static_cat": len(stats.feat_static_cat),
            "cardinality": [len(cats) for cats in stats.feat_static_cat],
        }

    def input_names(self, training=True):
        input_names = list(TRAINING_INPUT_NAMES if training else PREDICTION_INPUT_NAMES) 

        if not self.time_feat:
            input_names.remove("past_time_feat")
            input_names.remove("future_time_feat")
        if not self.num_feat_dynamic_real:
            input_names.remove("past_feat_dynamic_real")
            input_names.remove("future_feat_dynamic_real")

        return input_names
    
    def create_transformation(self) -> Transformation:
        remove_field_names = []
        if not self.num_feat_dynamic_real:
            remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)
        transforms = []
        if len(remove_field_names):
            transforms.append(RemoveFields(field_names=remove_field_names))
        if self.time_feat:
            transforms.append(
                AddTimeFeatures(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=time_features_from_frequency_str("1D"),
                    pred_length=self.prediction_length,
                )
            )

        transforms.append(AddObservedValuesIndicator(
            target_field=FieldName.TARGET,
            output_field=FieldName.OBSERVED_VALUES,
            imputation_method=DummyValueImputation(0.0),
        ))

        return Chain(transforms)

    def create_lightning_module(self, use_kv_cache: bool = False) -> pl.LightningModule:
        model_kwargs = {
            "input_size": self.input_size,
            "context_length": self.context_length,
            "max_context_length": self.max_context_length,
            "lags_seq": self.lags_seq,
            "n_layer": self.n_layer,
            "n_embd_per_head": self.n_embd_per_head,
            "n_head": self.n_head,
            "scaling": self.scaling,
            "distr_output": self.distr_output,
            "num_parallel_samples": self.num_parallel_samples,
            "rope_scaling": self.rope_scaling,
            "time_feat": self.time_feat,
            "num_feat_dynamic_real": self.num_feat_dynamic_real,
            "dropout": self.dropout,
        }
        if self.ckpt_path is not None:
            return LagLlamaLightningModule.load_from_checkpoint(
                checkpoint_path=self.ckpt_path,
                map_location=self.device,
                strict=False,
                loss=self.loss,
                lr=self.lr,
                weight_decay=self.weight_decay,
                context_length=self.context_length,
                prediction_length=self.prediction_length,
                model_kwargs=model_kwargs,
                # Augmentations
                aug_prob=self.aug_prob,
                freq_mask_rate=self.freq_mask_rate,
                freq_mixing_rate=self.freq_mixing_rate,
                jitter_prob=self.jitter_prob,
                jitter_sigma=self.jitter_sigma,
                scaling_prob=self.scaling_prob,
                scaling_sigma=self.scaling_sigma,
                rotation_prob=self.rotation_prob,
                permutation_prob=self.permutation_prob,
                permutation_max_segments=self.permutation_max_segments,
                permutation_seg_mode=self.permutation_seg_mode,
                magnitude_warp_prob=self.magnitude_warp_prob,
                magnitude_warp_sigma=self.magnitude_warp_sigma,
                magnitude_warp_knot=self.magnitude_warp_knot,
                time_warp_prob=self.time_warp_prob,
                time_warp_sigma=self.time_warp_sigma,
                time_warp_knot=self.time_warp_knot,
                window_slice_prob=self.window_slice_prob,
                window_slice_reduce_ratio=self.window_slice_reduce_ratio,
                window_warp_prob=self.window_warp_prob,
                window_warp_window_ratio=self.window_warp_window_ratio,
                window_warp_scales=self.window_warp_scales,
                use_kv_cache=use_kv_cache,
                data_id_to_name_map=self.data_id_to_name_map,
                use_cosine_annealing_lr=self.use_cosine_annealing_lr,
                cosine_annealing_lr_args=self.cosine_annealing_lr_args,
                track_loss_per_series=self.track_loss_per_series,
                nonnegative_pred_samples=self.nonnegative_pred_samples,
                model_config = self.model_config,
                dataset_configs = self.dataset_configs
            )
        else:
            lightning_module = LagLlamaLightningModule(
                loss=self.loss,
                lr=self.lr,
                weight_decay=self.weight_decay,
                context_length=self.context_length,
                prediction_length=self.prediction_length,
                model_kwargs=model_kwargs,
                # Augmentations
                aug_prob=self.aug_prob,
                freq_mask_rate=self.freq_mask_rate,
                freq_mixing_rate=self.freq_mixing_rate,
                jitter_prob=self.jitter_prob,
                jitter_sigma=self.jitter_sigma,
                scaling_prob=self.scaling_prob,
                scaling_sigma=self.scaling_sigma,
                rotation_prob=self.rotation_prob,
                permutation_prob=self.permutation_prob,
                permutation_max_segments=self.permutation_max_segments,
                permutation_seg_mode=self.permutation_seg_mode,
                magnitude_warp_prob=self.magnitude_warp_prob,
                magnitude_warp_sigma=self.magnitude_warp_sigma,
                magnitude_warp_knot=self.magnitude_warp_knot,
                time_warp_prob=self.time_warp_prob,
                time_warp_sigma=self.time_warp_sigma,
                time_warp_knot=self.time_warp_knot,
                window_slice_prob=self.window_slice_prob,
                window_slice_reduce_ratio=self.window_slice_reduce_ratio,
                window_warp_prob=self.window_warp_prob,
                window_warp_window_ratio=self.window_warp_window_ratio,
                window_warp_scales=self.window_warp_scales,
                use_kv_cache=use_kv_cache,
                data_id_to_name_map=self.data_id_to_name_map,
                use_cosine_annealing_lr=self.use_cosine_annealing_lr,
                cosine_annealing_lr_args=self.cosine_annealing_lr_args,
                track_loss_per_series=self.track_loss_per_series,
                nonnegative_pred_samples=self.nonnegative_pred_samples,
                model_config = self.model_config,
                dataset_configs = self.dataset_configs
            )

            if self.partial_weights_ckpt_path is not None:
                lightning_module.model.load_partial_weights(self.partial_weights_ckpt_path, self.device, self.freeze_transformer)

            return lightning_module                

    def _create_instance_splitter(self, module: LagLlamaLightningModule, mode: str):
        assert mode in ["training", "validation", "test"]

        instance_sampler = {
            "training": self.train_sampler,
            "validation": self.validation_sampler,
            "test": TestSplitSampler(),
        }[mode]


        ts_fields = [FieldName.OBSERVED_VALUES]
        if self.time_feat:
            ts_fields.append(FieldName.FEAT_TIME)
        if self.num_feat_dynamic_real:
            ts_fields.append(FieldName.FEAT_DYNAMIC_REAL)
        return InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=instance_sampler,
            past_length=self.context_length + max(self.lags_seq),
            future_length=self.prediction_length,
            time_series_fields=ts_fields,
            dummy_value=self.distr_output.value_in_support,
        )

    def create_training_data_loader(
        self,
        data: Dataset,
        module: LagLlamaLightningModule,
        shuffle_buffer_length: Optional[int] = None,
        **kwargs,
    ) -> Iterable:
        data = Cyclic(data).stream()
        instances = self._create_instance_splitter(module, "training").apply(
            data, is_train=True
        )
        return as_stacked_batches(
            instances,
            batch_size=self.batch_size,
            shuffle_buffer_length=shuffle_buffer_length,
            field_names=self.input_names(),
            output_type=torch.tensor,
            num_batches_per_epoch=self.num_batches_per_epoch,
        )

    def create_validation_data_loader(
        self,
        data: Dataset,
        module: LagLlamaLightningModule,
        **kwargs,
    ) -> Iterable:
        instances = self._create_instance_splitter(module, "validation").apply(
            data, is_train=True
        )
        return as_stacked_batches(
            instances,
            batch_size=self.batch_size,
            field_names=self.input_names(),
            output_type=torch.tensor,
        )

    def create_predictor(
        self,
        transformation: Transformation,
        module,
    ) -> PyTorchPredictor:
        prediction_splitter = self._create_instance_splitter(module, "test")
        return PyTorchPredictor(
            input_transform=transformation + prediction_splitter,
            input_names=self.input_names(training=False),
            prediction_net=module,
            batch_size=self.batch_size,
            prediction_length=self.prediction_length,
            device=self.device,
        )

def compute_lag_indices(lags_seq):
    """
    Computes and returns the adjusted lag indices from a sequence of integers or frequency strings.
    If `lags_seq` contains integers, they are treated as direct lag indices which are then adjusted.
    If `lags_seq` contains strings, they are assumed to be frequency identifiers from which lag indices
    are derived using the GluonTS function `get_lags_for_frequency`.

    Parameters:
    lags_seq (list of int or str): A list of integers representing lag indices or strings representing frequencies.

    Returns:
    list of int: A sorted and adjusted list of unique lag indices.
    """
    if all(isinstance(lag, int) for lag in lags_seq):  # Check if all elements are integers (indices)
        return sorted(set(lags_seq))  # Remove duplicate indices and sort

    lag_indices = [] # Otherwise compute lag indices from frequency strings
    for freq in lags_seq:
        lag_indices.extend(
            get_lags_for_frequency(freq_str=freq, num_default_lags=1)
        )

    if lag_indices:
        return [lag_index - 1 for lag_index in sorted(set(lag_indices))]
    else:
        return []