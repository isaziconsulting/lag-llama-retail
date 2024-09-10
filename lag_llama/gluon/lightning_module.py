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
from torch import nn
import os
import random

import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
import shap

from lightning import LightningModule
import torch
import torch.nn.functional as F

from gluonts.core.component import validated
from gluonts.itertools import prod
from gluonts.torch.modules.loss import DistributionLoss, NegativeLogLikelihood
from gluonts.torch.util import repeat_along_dim, take_last

from ...data.augmentations.freq_mask import freq_mask
from ...data.augmentations.freq_mix import freq_mix
from ...data.augmentations.augmentations import (
    ApplyAugmentations,
    Jitter,
    MagnitudeWarp,
    Permutation,
    Rotation,
    Scaling,
    TimeWarp,
    WindowSlice,
    WindowWarp,
)
from ...gluon_utils.gluon_ts_distributions.implicit_quantile_network import (
    ImplicitQuantileNetworkOutput,
)
from ...lag_llama.model.module import LagLlamaModel
from pytorch_lightning.loggers import TensorBoardLogger

class LagLlamaLightningModule(LightningModule):
    """
    A ``pl.LightningModule`` class that can be used to train a
    ``LagLlamaLightningModule`` with PyTorch Lightning.

    This is a thin layer around a (wrapped) ``LagLlamaLightningModule`` object,
    that exposes the methods to evaluate training and validation loss.

    Parameters
    ----------
    model
        ``LagLlamaLightningModule`` to be trained.
    loss
        Loss function to be used for training,
        default: ``NegativeLogLikelihood()``.
    lr
        Learning rate, default: ``1e-3``.
    weight_decay
        Weight decay regularization parameter, default: ``1e-8``.
    """

    @validated()
    def __init__(
        self,
        model_kwargs: dict,
        context_length: int,
        prediction_length: int,
        loss: DistributionLoss = NegativeLogLikelihood(),
        lr: float = 1e-3,
        weight_decay: float = 1e-8,
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
        data_id_to_name_map: dict = {},
        use_cosine_annealing_lr: bool = False,
        cosine_annealing_lr_args: dict = {},
        track_loss_per_series: bool = False,
        nonnegative_pred_samples: bool = False,
        use_kv_cache: bool = True,
        model_config=None,
        dataset_configs=[]
    ):
        super().__init__()
        self.save_hyperparameters()
        self.context_length = self.hparams.context_length
        self.prediction_length = self.hparams.prediction_length
        self.model = LagLlamaModel(**self.hparams.model_kwargs)
        self.loss = self.hparams.loss
        self.lr = self.hparams.lr
        self.weight_decay = self.hparams.weight_decay
        self.aug_prob = self.hparams.aug_prob
        self.freq_mask_rate = self.hparams.freq_mask_rate
        self.freq_mixing_rate = self.hparams.freq_mixing_rate
        self.jitter_prob = self.hparams.jitter_prob
        self.jitter_sigma = self.hparams.jitter_sigma
        self.scaling_prob = self.hparams.scaling_prob
        self.scaling_sigma = self.hparams.scaling_sigma
        self.rotation_prob = self.hparams.rotation_prob
        self.permutation_prob = self.hparams.permutation_prob
        self.permutation_max_segments = self.hparams.permutation_max_segments
        self.permutation_seg_mode = self.hparams.permutation_seg_mode
        self.magnitude_warp_prob = self.hparams.magnitude_warp_prob
        self.magnitude_warp_sigma = self.hparams.magnitude_warp_sigma
        self.magnitude_warp_knot = self.hparams.magnitude_warp_knot
        self.time_warp_prob = self.hparams.time_warp_prob
        self.time_warp_sigma = self.hparams.time_warp_sigma
        self.time_warp_knot = self.hparams.time_warp_knot
        self.window_slice_prob = self.hparams.window_slice_prob
        self.window_slice_reduce_ratio = self.hparams.window_slice_reduce_ratio
        self.window_warp_prob = self.hparams.window_warp_prob
        self.window_warp_window_ratio = self.hparams.window_warp_window_ratio
        self.window_warp_scales = self.hparams.window_warp_scales
        self.data_id_to_name_map = self.hparams.data_id_to_name_map
        self.use_cosine_annealing_lr = self.hparams.use_cosine_annealing_lr
        self.cosine_annealing_lr_args = self.hparams.cosine_annealing_lr_args
        self.track_loss_per_series = self.hparams.track_loss_per_series
        self.nonnegative_pred_samples = self.hparams.nonnegative_pred_samples
        self.model_config = model_config
        self.dataset_configs = dataset_configs

        self.time_feat = self.hparams.model_kwargs["time_feat"]
        self.num_feat_dynamic_real = self.hparams.model_kwargs["num_feat_dynamic_real"]
        self.num_feat_static_cat = self.hparams.model_kwargs["num_feat_static_cat"]
        self.num_feat_static_real = self.hparams.model_kwargs["num_feat_static_real"]
        self.static_cardinalities = self.hparams.model_kwargs["static_cardinalities"]
        # data_id based
        self.train_loss_dict = {}
        self.val_loss_dict = {}
        # item_id based - to be used only in single-dataset mode
        self.train_loss_dict_per_series = {}
        self.val_loss_dict_per_series = {}
        self.use_kv_cache = use_kv_cache
        self.transforms = []
        aug_probs = dict(
            Jitter=dict(prob=self.jitter_prob, sigma=self.jitter_sigma),
            Scaling=dict(prob=self.scaling_prob, sigma=self.scaling_sigma),
            Rotation=dict(prob=self.rotation_prob),
            Permutation=dict(
                prob=self.permutation_prob,
                max_segments=self.permutation_max_segments,
                seg_mode=self.permutation_seg_mode,
            ),
            MagnitudeWarp=dict(
                prob=self.magnitude_warp_prob,
                sigma=self.magnitude_warp_sigma,
                knot=self.magnitude_warp_knot,
            ),
            TimeWarp=dict(
                prob=self.time_warp_prob,
                sigma=self.time_warp_sigma,
                knot=self.time_warp_knot,
            ),
            WindowSlice=dict(
                prob=self.window_slice_prob, reduce_ratio=self.window_slice_reduce_ratio
            ),
            WindowWarp=dict(
                prob=self.window_warp_prob,
                window_ratio=self.window_warp_window_ratio,
                warp_slices=self.window_warp_scales,
            ),
        )
        for aug, params in aug_probs.items():
            if params["prob"] > 0:
                if aug == "Jitter":
                    self.transforms.append(Jitter(params["prob"], params["sigma"]))
                elif aug == "Scaling":
                    self.transforms.append(Scaling(params["prob"], params["sigma"]))
                elif aug == "Rotation":
                    self.transforms.append(Rotation(params["prob"]))
                elif aug == "Permutation":
                    self.transforms.append(
                        Permutation(
                            params["prob"], params["max_segments"], params["seg_mode"]
                        )
                    )
                elif aug == "MagnitudeWarp":
                    self.transforms.append(
                        MagnitudeWarp(params["prob"], params["sigma"], params["knot"])
                    )
                elif aug == "TimeWarp":
                    self.transforms.append(
                        TimeWarp(params["prob"], params["sigma"], params["knot"])
                    )
                elif aug == "WindowSlice":
                    self.transforms.append(
                        WindowSlice(params["prob"], params["reduce_ratio"])
                    )
                elif aug == "WindowWarp":
                    self.transforms.append(
                        WindowWarp(
                            params["prob"],
                            params["window_ratio"],
                            params["warp_slices"],
                        )
                    )

        self.augmentations = ApplyAugmentations(self.transforms)

        # Lags (target and lag indices)
        lag_labels = ['volume'] + [f'lag {lag}' for lag in self.model.lags_seq[1:]]

        # Scaling feature labels
        scaling_labels = ['volume med', 'volume iqr', 'planned promo med', 'planned promo iqr']

        # Time feature labels
        time_labels = ['dow', 'dom', 'doy']

        # Dynamic feature labels (hardcoded)
        dynamic_labels = [
            'rel_promo_price', 'is_promo', 'promo_strength', 
            'is_single_price_promo', 'is_multibuy_promo', 
            'rel_price', 'planned_promo_vol'
        ]

        # Combine all the labels
        self.feature_names = lag_labels + scaling_labels + time_labels + dynamic_labels
        self.iter_index = 1
        self.cumulative_shap_values = None
        self.cumulative_promo_mask = None

    def log_shap_values(self, past_target, past_observed_values, past_time_feat, future_time_feat, past_feat_dynamic_real, future_feat_dynamic_real, feat_static_cat, feat_static_real, future_target):
            output_dir = f"shap_results/batch_{self.iter_index}"
            self.iter_index += 1
            os.makedirs(output_dir, exist_ok=True)

            with torch.set_grad_enabled(True):
                inputs, loc, scale = self.model.prepare_input(past_target,
                                                    past_observed_values,
                                                    past_time_feat,
                                                    future_time_feat,
                                                    past_feat_dynamic_real,
                                                    future_feat_dynamic_real)
            inputs.requires_grad = True
            bsz, seq_len, num_feats = inputs.size()
            inputs = inputs.view(bsz, seq_len * num_feats)
            # Split into two shuffled half size batches
            indices = torch.randperm(bsz)
            bsz = bsz//2
            train_inputs, test_inputs = inputs[indices[:bsz]], inputs[indices[bsz:]]

            # Wrap the model so it's compatble with Shap
            shap_model = ShapModelWrapper(self.model, (bsz, seq_len, num_feats))

            with torch.set_grad_enabled(True):
                explainer = shap.GradientExplainer(shap_model, train_inputs)
                shap_values = explainer.shap_values(test_inputs)
            # We only care about the effects of features at a particular time step
            time_step = -1
            shap_values_sliced = shap_values.reshape(bsz, seq_len, num_feats)[:, time_step, :]

            if self.cumulative_shap_values is None:
                self.cumulative_shap_values = shap_values_sliced
            else:
                self.cumulative_shap_values = np.concatenate([self.cumulative_shap_values, shap_values_sliced])

            text_file_path = os.path.join(output_dir, "shap_aggregated_values.txt")
            with open(text_file_path, 'w') as f:
                for i, label in enumerate(self.feature_names):
                    shap_value_aggregated = abs(self.cumulative_shap_values[:, i]).sum()  # Aggregate over samples
                    f.write(f'{label} Total SHAP magnitude: {shap_value_aggregated}\n')

            max_disp=22

            shap.summary_plot(self.cumulative_shap_values, feature_names=self.feature_names, show=False, max_display = max_disp, use_log_scale=True)
            file_path = f"{output_dir}/shap_summary_plot.png"
            plt.savefig(file_path, format='png')
            plt.close()

            # Get promos & non_promos
            is_promo_dim = self.feature_names.index('is_promo')
            promo_mask = test_inputs.view(bsz, seq_len, num_feats)[:, time_step, is_promo_dim] == 1
            promo_mask = promo_mask.squeeze()
            if self.cumulative_promo_mask is None:
                self.cumulative_promo_mask = promo_mask
            else:
                self.cumulative_promo_mask = np.concatenate([self.cumulative_promo_mask, promo_mask])
            promo_shap_values_reshaped = self.cumulative_shap_values[self.cumulative_promo_mask, :]
            non_promo_shap_values_reshaped = self.cumulative_shap_values[~self.cumulative_promo_mask, :]

            shap.summary_plot(promo_shap_values_reshaped, feature_names=self.feature_names, show=False, max_display = max_disp, use_log_scale=True)
            file_path = f"{output_dir}/shap_summary_plot_promos.png"
            plt.savefig(file_path, format='png')
            plt.close()

            shap.summary_plot(non_promo_shap_values_reshaped, feature_names=self.feature_names, show=False, max_display = max_disp, use_log_scale=True)
            file_path = f"{output_dir}/shap_summary_plot_non_promos.png"
            plt.savefig(file_path, format='png')
            plt.close()

    # greedy prediction
    def forward(self, *args, **kwargs):
        past_target = kwargs[
            "past_target"
        ]  # (bsz, model.context_length+max(model.lags_seq))
        past_observed_values = kwargs[
            "past_observed_values"
        ]  # (bsz, model.context_length+max(model.lags_seq))
        if self.time_feat:
            past_time_feat = kwargs["past_time_feat"]
            future_time_feat = kwargs["future_time_feat"]
        if self.num_feat_dynamic_real:
            past_feat_dynamic_real = kwargs["past_feat_dynamic_real"]
            future_feat_dynamic_real = kwargs["future_feat_dynamic_real"]
        if self.num_feat_static_cat:
            feat_static_cat = kwargs["feat_static_cat"]
        else:
            feat_static_cat = None
        if self.num_feat_static_real:
            feat_static_real = kwargs["feat_static_real"]
        else:
            feat_static_real = None

        future_samples = []
        for t in range(self.prediction_length):
            params, loc, scale = self.model(
                *args,
                past_time_feat=past_time_feat if self.time_feat else None,
                future_time_feat=future_time_feat[..., : t + 1, :] if self.time_feat else None,
                past_feat_dynamic_real=past_feat_dynamic_real if self.num_feat_dynamic_real else None,
                future_feat_dynamic_real=future_feat_dynamic_real[..., : t + 1, :] if self.num_feat_dynamic_real else None,
                feat_static_cat=feat_static_cat,
                feat_static_real=feat_static_real,
                past_target=past_target,
                past_observed_values=past_observed_values,
                use_kv_cache=self.use_kv_cache,
            )

            self.log_shap_values(
                past_time_feat=past_time_feat if self.time_feat else None,
                future_time_feat=future_time_feat[..., : t + 1, :] if self.time_feat else None,
                past_feat_dynamic_real=past_feat_dynamic_real if self.num_feat_dynamic_real else None,
                future_feat_dynamic_real=future_feat_dynamic_real[..., : t + 1, :] if self.num_feat_dynamic_real else None,
                feat_static_cat=feat_static_cat,
                feat_static_real=feat_static_real,
                past_target=past_target,
                past_observed_values=past_observed_values,
                future_target=None
            )

            sliced_params = [
                p[:, -1:] for p in params
            ]  # Take the last timestep predicted. Each tensor is of shape (#bsz*#parallel_samples, 1)
            distr = self.model.distr_output.distribution(sliced_params, loc, scale)
            sample = distr.mean  # (#bsz*#parallel_samples, 1)
            if self.nonnegative_pred_samples:
                sample = F.relu(sample)
            future_samples.append(sample)

            past_target = torch.cat((past_target, sample), dim=1)
            past_observed_values = torch.cat(
                (past_observed_values, torch.ones_like(sample)), dim=1
            )

        self.model.reset_cache()

        concat_future_samples = torch.cat(future_samples, dim=-1)
        return concat_future_samples.reshape(
            (-1, self.model.num_parallel_samples, self.prediction_length)
            + self.model.distr_output.event_shape,
        )
    
    # train
    def _compute_loss(self, batch, do_not_average=False, return_observed_values=False, validating=False):
        past_target = batch[
            "past_target"
        ]  # (bsz, model.context_length+max(model.lags_seq))
        past_observed_values = batch[
            "past_observed_values"
        ]  # (bsz, model.context_length+max(model.lags_seq)) with 0s or 1s indicating available (1s) or missing (0s)
        future_target = batch["future_target"]  # (bsz, model.prediction_length)
        future_observed_values = batch[
            "future_observed_values"
        ]  # (bsz, model.prediction_length) with 0s or 1s indicating available (1s) or missing (0s)
        if self.time_feat:
            past_time_feat = batch["past_time_feat"]
            future_time_feat = batch["future_time_feat"]
        else:
            past_time_feat = None
            future_time_feat = None
        if self.num_feat_dynamic_real:
            past_feat_dynamic_real = batch["past_feat_dynamic_real"]
            future_feat_dynamic_real = batch["future_feat_dynamic_real"]
        else:
            past_feat_dynamic_real = None
            future_feat_dynamic_real = None
        if self.num_feat_static_cat:
            feat_static_cat = batch["feat_static_cat"]
        else:
            feat_static_cat = None
        if self.num_feat_static_real:
            feat_static_real = batch["feat_static_real"]
        else:
            feat_static_real = None

        extra_dims = len(future_target.shape) - len(past_target.shape)  # usually 0
        extra_shape = future_target.shape[:extra_dims]  # shape remains the same

        repeats = prod(extra_shape)  # usually 1
        past_target = repeat_along_dim(
            past_target, 0, repeats
        )  # (bsz, model.context_length+max(model.lags_seq))
        past_observed_values = repeat_along_dim(
            past_observed_values, 0, repeats
        )  # (bsz, model.context_length+max(model.lags_seq))

        future_target_reshaped = future_target.reshape(
            -1,
            *future_target.shape[extra_dims + 1 :],
        )  # (bsz, model.prediction_length)
        future_observed_reshaped = future_observed_values.reshape(
            -1,
            *future_observed_values.shape[extra_dims + 1 :],
        )  # (bsz, model.prediction_length)

        distr_args, loc, scale = self.model(
            past_target=past_target,
            past_observed_values=past_observed_values,
            past_time_feat=past_time_feat,
            future_time_feat=future_time_feat,
            past_feat_dynamic_real=past_feat_dynamic_real,
            future_feat_dynamic_real=future_feat_dynamic_real,
            feat_static_cat=feat_static_cat,
            feat_static_real=feat_static_real,
            future_target=future_target_reshaped,
        )  # distr_args is a tuple with two tensors of shape (bsz, context_length+pred_len-1)
        context_target = take_last(
            past_target, dim=-1, num=self.context_length - 1
        )  # (bsz, context_length-1) # Basically removes the first value since it cannot be predicted
        target = torch.cat(
            (context_target, future_target_reshaped),
            dim=1,
        )  # (bsz, context_length-1+pred_len) # values that can be predicted
        
        context_observed = take_last(
            past_observed_values, dim=-1, num=self.context_length - 1
        )  # same as context_target, but for observed_values tensor

        # If we are validating, ignore the context window when computing loss by marking it not observed
        # As the context window will overlap into the training set
        if validating:
            context_observed = torch.zeros_like(context_observed)

        observed_values = torch.cat(
            (context_observed, future_observed_reshaped), dim=1
        )  # same as target but for observed_values tensor
            
        if type(self.model.distr_output) == ImplicitQuantileNetworkOutput:
            if not do_not_average:
                loss = (
                    self.model.distr_output.loss(target, distr_args, loc, scale)
                    * observed_values
                ).sum() / observed_values.sum().clamp_min(1.0)
            else:
                loss = (
                    self.model.distr_output.loss(target, distr_args, loc, scale)
                    * observed_values
                )
        else:
            distr = self.model.distr_output.distribution(
                distr_args, loc=loc, scale=scale
            )  # an object representing a distribution with the specified parameters. We need this to compute the NLL loss.
            
            if hasattr(self.loss, 'name') and self.loss.name == 'mean-absolute-scaled-error':
                # Absolute scaled error requires a naive forecast
                # We take this as whatever the previous target was
                context_naive_forecast = take_last(
                    past_target, dim=-1, num=self.context_length
                )  # (bsz, context_length)
                naive_forecast = torch.cat(
                    (context_naive_forecast, future_target_reshaped[:, :-1]),
                    dim=1,
                )  # (bsz, context_length+pred_len-1) # values that can be predicted
                loss = self.loss(distr, target, naive_forecast) * observed_values
            else:
                loss = self.loss(distr, target) * observed_values
            
            if not do_not_average:
                loss = loss.sum() / observed_values.sum().clamp_min(1.0)

        if not return_observed_values:
            return loss
        else:
            return loss, observed_values

    def _log_bar_plot(self, values, tag):
        # Log the labels and values as scalars in case
        for i, label in enumerate(self.feature_names):
            self.logger.experiment.add_scalar(f'{tag}-{label}-scalar', values[i], self.current_epoch)

        # Create the bar plot
        plt.figure(figsize=(12, 6))  # Adjusted size for clarity
        plt.bar(range(len(values)), values, tick_label=self.feature_names)
        plt.xlabel('Feature')
        plt.ylabel('Importance')

        # Rotate and align the labels for readability
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Save the plot to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)

        # Open the image from buffer and convert it to a tensor
        image = Image.open(buf)
        image = np.array(image)  # Convert to NumPy array (H, W, C)
        
        # Convert the image from HWC (Height, Width, Channels) to CHW (Channels, Height, Width)
        image_tensor = torch.tensor(image).permute(2, 0, 1)  # Convert to CHW format
        
        # Log the image to TensorBoard
        self.logger.experiment.add_image(tag, image_tensor, self.current_epoch)

    def log_token_embed_layer(self):
        weight_matrix = self.state_dict()['model.transformer.wte.weight']

        # Log the histograms of weights for the embedding layer
        self.logger.experiment.add_histogram("embedding weights", weight_matrix, self.current_epoch)

        # Log feature importance plots using L2 norm & sum of absolute weights
        l2_norms = []
        sum_abs_weights = []

        # Transpose the weight matrix to access features as rows instead of columns
        weight_matrix_t = weight_matrix.T  # Shape is now [features, embeddings]

        # Compute metrics for each input feature (column in the original weight matrix)
        for i in range(weight_matrix_t.size(0)):  # Loop over each input feature
            feature_weights = weight_matrix_t[i]

            # Calculate the sum of absolute weights for each feature
            sum_abs_weights.append(torch.sum(torch.abs(feature_weights)).item())

            # Calculate the L2 norm for each feature
            l2_norms.append(torch.norm(feature_weights, p=2).item())

        # Log bar plot for sum of absolute weights
        self._log_bar_plot(sum_abs_weights, 'sum_abs_weights_bar')

        # Log bar plot for L2 norm
        self._log_bar_plot(l2_norms, 'l2_norm_weights_bar')

        # Log individual feature histograms of weights for the projection layer
        for i in range(weight_matrix_t.size(0)):  # Loop over each input feature
            self.logger.experiment.add_histogram(f'weights/feature_{i}', weight_matrix_t[i], self.current_epoch)

    def training_step(self, batch, batch_idx: int):  # type: ignore
        """
        Execute training step.
        """
        if random.random() < self.aug_prob:
            # Freq mix and Freq mask have separate functions
            if self.freq_mask_rate > 0:
                batch["past_target"], batch["future_target"] = freq_mask(
                    batch["past_target"],
                    batch["future_target"],
                    rate=self.freq_mask_rate,
                )
            if self.freq_mixing_rate:
                batch["past_target"], batch["future_target"] = freq_mix(
                    batch["past_target"],
                    batch["future_target"],
                    rate=self.freq_mixing_rate,
                )
            # Other augmentation
            if len(self.transforms):
                batch["past_target"], batch["future_target"] = self.augmentations(
                    batch["past_target"], batch["future_target"]
                )
        
        if self.nonnegative_pred_samples:
            batch["past_target"] = F.relu(batch["past_target"])
            batch["future_target"] = F.relu(batch["future_target"])

        train_loss_per_sample, observed_values = self._compute_loss(
            batch, do_not_average=True, return_observed_values=True
        )

        train_loss_avg = train_loss_per_sample.sum() / observed_values.sum().clamp_min(
            1.0
        )
        if hasattr(self.loss, 'name') and self.loss.name in ['root-mean-squared-error', 'huber-loss']:
            train_loss_avg = torch.sqrt(train_loss_avg)
        self.log(
            "train_loss", train_loss_avg, on_epoch=True, on_step=False, prog_bar=False
        )
        return train_loss_avg

    def on_train_epoch_end(self):
        self.log_token_embed_layer()
        # Log all losses
        for key, value in self.train_loss_dict.items():
            loss_avg = np.mean(value)
            self.log(
                f"train_loss_avg_per_train_dataset/{self.data_id_to_name_map[key]}",
                loss_avg,
                on_epoch=True,
                on_step=False,
                prog_bar=False,
            )

        if self.track_loss_per_series:
            # Log all losses
            for key, value in self.train_loss_dict_per_series.items():
                loss_avg = np.mean(value)
                self.log(
                    f"train_loss_avg_per_train_series/{key}",
                    loss_avg,
                    on_epoch=True,
                    on_step=False,
                    prog_bar=False,
                )

        # Reset loss_dict
        self.train_loss_dict = {}
        self.train_loss_dict_per_series = {}

    def validation_step(self, batch, batch_idx: int):  # type: ignore
        """
        Execute validation step.
        """
        val_loss_per_sample, observed_values = self._compute_loss(
            batch, do_not_average=True, return_observed_values=True, validating=True
        )

        val_loss_avg = val_loss_per_sample.sum() / observed_values.sum().clamp_min(1.0)

        if hasattr(self.loss, 'name') and self.loss.name in ['root-mean-squared-error', 'huber-loss']:
            val_loss_avg = torch.sqrt(val_loss_avg)
        
        self.log("val_loss", val_loss_avg, on_epoch=True, on_step=False, prog_bar=False)
        return val_loss_avg

    def on_validation_epoch_end(self):
        # Log all losses
        for key, value in self.val_loss_dict.items():
            loss_avg = np.mean(value)
            if key >= 0:
                self.log(
                    f"val_loss_avg_per_train_dataset/{self.data_id_to_name_map[key]}",
                    loss_avg,
                    on_epoch=True,
                    on_step=False,
                    prog_bar=False,
                )
            else:
                self.log(
                    f"val_loss_avg_per_test_dataset/{self.data_id_to_name_map[key]}",
                    loss_avg,
                    on_epoch=True,
                    on_step=False,
                    prog_bar=False,
                )

        if self.track_loss_per_series:
            # Log all losses
            for key, value in self.val_loss_dict_per_series.items():
                loss_avg = np.mean(value)
                self.log(
                    f"val_loss_avg_per_train_series/{key}",
                    loss_avg,
                    on_epoch=True,
                    on_step=False,
                    prog_bar=False,
                )

        # Reset loss_dict
        self.val_loss_dict = {}
        self.val_loss_dict_per_series = {}

    def configure_optimizers(self):
        """
        Returns the optimizer to use.
        """
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        if self.use_cosine_annealing_lr:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, **self.cosine_annealing_lr_args, verbose=True
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return optimizer

class ShapModelWrapper(nn.Module):
    def __init__(self, model, dims):
        super(ShapModelWrapper, self).__init__()
        self.model = model
        self.dims = dims

    def __call__(self, inputs):
        (bsz, seq_len, num_feats) = self.dims
        inputs = inputs.reshape(-1, seq_len, num_feats)
        params = self.model.forward_shap(inputs)
        sliced_params = [
            p[:, -1:] for p in params
        ]  # Take the last timestep predicted. Each tensor is of shape (#bsz*#parallel_samples, 1)
        distr = self.model.distr_output.distribution(sliced_params)
        return distr.mean
    
# # Define a custom masker function
# def custom_masker(mask, X):
#     """
#     Custom masker that masks a feature across all timesteps for a sample.
    
#     Args:
#     - X (numpy.array): Input data of shape (num_samples, seq_len, num_features).
#     - mask (numpy.array or None): Masking array of shape (num_features,). 
#                                 A 0 value means the feature is masked, 1 means it's not masked.
    
#     Returns:
#     - Masked data (numpy.array): Masked version of X.
#     """
#     (bsz, seq_len, num_feats) = dims
#     mask = mask.reshape(seq_len, num_feats)
#     if mask is None:
#         # If no mask is provided, return the input unchanged
#         return X
    
#     # Create a copy of X to apply the mask
#     masked_X = X.view(seq_len, num_feats).detach().clone()
    
#     # Iterate over each sample
#     for feature_idx in range(num_feats):
#         if np.any(mask[:, feature_idx] == 0):
#             # Mask the feature across all timesteps for this sample
#             masked_X[:, feature_idx] = 0  # Set masked feature to 0 or another value
    
#     return masked_X.flatten()
