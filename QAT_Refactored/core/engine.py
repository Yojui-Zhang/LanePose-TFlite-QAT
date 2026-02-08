# Source File: QAT_Refactored/core/engine.py

import os
import tensorflow as tf
import time
import csv
import logging
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Tuple, Any

from QAT_Refactored.utils.tensor_layout import assert_layout_tf
# Project Imports
from QAT_Refactored.config.config import AppConfig
# [REF] Import get_anchors to fix DRY violation
from QAT_Refactored.losses.pose_loss import (
    PoseLabelLoss, 
    get_anchors, 
    build_batch_dict_from_padded_labels
)
from QAT_Refactored.losses.distill_loss import DistillLossPose
from QAT_Refactored.utils.visualization import save_gt_and_plot, save_pred_and_plot

class ModelEMA:
    """Exponential Moving Average for Model Weights"""
    def __init__(self, model: tf.keras.Model, decay: float = 0.9998):
        self.decay = decay
        # Create shadow variables for all model weights
        self.shadow = [tf.Variable(w, trainable=False, name=f"ema_{i}") for i, w in enumerate(model.weights)]
        self.backup: Optional[list] = None

    def update(self, model: tf.keras.Model):
        d = self.decay
        for s, w in zip(self.shadow, model.weights):
            # Only apply EMA to floating point weights (ignore integer steps/counters)
            if w.dtype.is_floating:
                s.assign(d * s + (1.0 - d) * w)
            else:
                s.assign(w)

    def apply_to(self, model: tf.keras.Model):
        """Backup current weights and apply EMA weights for validation/export."""
        self.backup = [tf.identity(w) for w in model.weights]
        for w, s in zip(model.weights, self.shadow):
            w.assign(s)

    def restore(self, model: tf.keras.Model):
        """Restore original training weights."""
        if self.backup is None:
            return
        for w, b in zip(model.weights, self.backup):
            w.assign(b)
        self.backup = None

class Trainer:
    def __init__(
        self, 
        cfg: AppConfig, 
        student_model: tf.keras.Model, 
        teacher_model: Optional[tf.keras.Model] = None
    ):
        self.cfg = cfg
        self.student = student_model
        self.teacher = teacher_model
        
        # Initialize Paths
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = self.cfg.OUTPUT_DIR / self.run_id / "logs"
        self.models_dir = self.cfg.OUTPUT_DIR / self.run_id / "models"
        self.plots_dir = self.cfg.OUTPUT_DIR / self.run_id / "plots"
        
        for p in [self.log_dir, self.models_dir, self.plots_dir]:
            p.mkdir(parents=True, exist_ok=True)
            
        self.log_csv = self.log_dir / "training_log.csv"
        self.best_val_loss = float("inf")
        self.global_step = 0
        self.stop_requested = False

        # Initialize EMA
        self.ema = ModelEMA(self.student, decay=0.9998)
        
        # Initialize Loss Functions (Now Layers)
        # [NOTE] These are instantiated as Layers, so they maintain state if needed
        self.loss_fn_label = PoseLabelLoss(self.cfg)
        self.loss_fn_distill = DistillLossPose(self.cfg) if self.teacher else None
        
        # Initialize Anchors
        # [DRY FIX] Use the utility from pose_loss instead of re-implementing logic
        # Strides 8, 16, 32 are standard for YOLOv8-S
        self.anchors = get_anchors(
            imgsz=self.cfg.IMGSZ, 
            strides=[8, 16, 32], 
            grid_cell_offset=0.5
        )
        
        logging.info(f"[Trainer] Initialized. Run ID: {self.run_id}")

    def _setup_optimizer(self, steps_per_epoch: int) -> Tuple[tf.keras.optimizers.Optimizer, Any]:
        total_steps = int(self.cfg.EPOCHS * steps_per_epoch)
        
        # Cosine Decay
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=self.cfg.BASE_LR,
            decay_steps=max(1, total_steps),
            alpha=self.cfg.END_LR / self.cfg.BASE_LR
        )
        
        # Custom Warmup Wrapper
        class WarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
            def __init__(self, warmup_steps, base_schedule):
                super().__init__()
                self.warmup_steps = tf.cast(warmup_steps, tf.float32)
                self.base_schedule = base_schedule
            
            def __call__(self, step):
                step_f = tf.cast(step, tf.float32)
                lr = self.base_schedule(step_f)
                warmup_factor = tf.clip_by_value(step_f / tf.maximum(self.warmup_steps, 1.0), 0.0, 1.0)
                return lr * warmup_factor
            
            def get_config(self):
                return {"warmup_steps": self.warmup_steps, "base_schedule": self.base_schedule}

        final_schedule = WarmupSchedule(self.cfg.WARMUP_STEPS, lr_schedule)
        
        opt = tf.keras.optimizers.SGD(
            learning_rate=final_schedule,
            momentum=self.cfg.MOMENTUM,
            nesterov=self.cfg.NESTEROV,
            clipnorm=self.cfg.CLIPNORM
        )
        
        # AMP Wrapper
        if self.cfg.USE_AMP:
            opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)
            
        return opt, final_schedule

    @tf.function
    def train_step(self, batch_imgs: tf.Tensor, batch_labels: tf.Tensor,
                optimizer: tf.keras.optimizers.Optimizer, class_weights: Optional[tf.Tensor], 
                freeze_bn: bool = False) -> Dict[str, tf.Tensor]:

        batch_dict = build_batch_dict_from_padded_labels(
            batch_labels, num_kpt=self.cfg.NUM_KPT, kpt_vals=self.cfg.KPT_VALS
        )

        with tf.GradientTape() as tape:
            y_s_out = self.student(batch_imgs, training=True)

            if isinstance(y_s_out, (list, tuple)):
                deploy_raw = y_s_out[0]
                kd_raw = y_s_out[1]
            else:
                deploy_raw = y_s_out
                kd_raw = y_s_out

            # Why: Trainer 邊界 fail-fast，避免下游對錯誤 layout 的 silent slicing
            total_c = self.cfg.total_output_channels
            deploy_raw = assert_layout_tf(deploy_raw, total_c, name="deploy_raw")
            kd_raw     = assert_layout_tf(kd_raw,     total_c, name="kd_raw")

            loss_dep, loss_box, loss_cls, loss_kpt = self.loss_fn_label(
                deploy_raw, batch_dict, self.anchors, class_weights
            )

            loss_kd = tf.constant(0.0, dtype=tf.float32)

            if self.cfg.TRAIN_SUPERVISION == 'distill' and self.teacher is not None:
                y_teacher = self.teacher(batch_imgs, training=False)
                if isinstance(y_teacher, (list, tuple)):
                    y_teacher = y_teacher[0]

                y_teacher = assert_layout_tf(y_teacher, total_c, name="y_teacher")
                loss_kd = self.loss_fn_distill(y_teacher, kd_raw)
            else:
                loss_kd_val, _, _, _ = self.loss_fn_label(
                    kd_raw, batch_dict, self.anchors, class_weights
                )
                loss_kd = loss_kd_val

            total_loss = (self.cfg.KD_LOSS_WEIGHT * loss_kd) + \
                        (self.cfg.DEPLOY_LOSS_WEIGHT * loss_dep)

            if self.student.losses:
                total_loss += tf.reduce_sum(self.student.losses)

            if self.cfg.USE_AMP:
                scaled_loss = optimizer.get_scaled_loss(total_loss)
            else:
                scaled_loss = total_loss

        vars = self.student.trainable_variables
        if freeze_bn:
            # Freeze batch normalization layers for stability in later training
            vars = [v for v in vars if not ('batch_normalization' in v.name.lower() or 'bn' in v.name.lower())]
        
        if self.cfg.USE_AMP:
            scaled_grads = tape.gradient(scaled_loss, vars)
            grads = optimizer.get_unscaled_gradients(scaled_grads)
        else:
            grads = tape.gradient(scaled_loss, vars)

        optimizer.apply_gradients(zip(grads, vars))
        self.ema.update(self.student)

        return {
            "total": total_loss,
            "box": loss_box,
            "cls": loss_cls,
            "kpt": loss_kpt,
            "kd": loss_kd
        }


    @tf.function
    def val_step(self, batch_imgs: tf.Tensor, batch_labels: tf.Tensor, class_weights: Optional[tf.Tensor]) -> Dict[str, tf.Tensor]:
        batch_dict = build_batch_dict_from_padded_labels(
            batch_labels, num_kpt=self.cfg.NUM_KPT, kpt_vals=self.cfg.KPT_VALS
        )

        y_s_out = self.student(batch_imgs, training=False)

        if isinstance(y_s_out, (list, tuple)):
            deploy_raw = y_s_out[0]
        else:
            deploy_raw = y_s_out

        total_c = self.cfg.total_output_channels
        deploy_raw = assert_layout_tf(deploy_raw, total_c, name="val_deploy_raw")

        total_loss, loss_box, loss_cls, loss_kpt = self.loss_fn_label(
            deploy_raw, batch_dict, self.anchors, class_weights
        )

        return {
            "total": total_loss,
            "box": loss_box,
            "cls": loss_cls,
            "kpt": loss_kpt
        }


    def run(self, train_ds, val_ds, steps_per_epoch, val_steps, class_weights=None):
        optimizer, lr_schedule = self._setup_optimizer(steps_per_epoch)
        
        # Init CSV
        headers = ["epoch", "train_loss", "train_box", "train_cls", "train_kpt", "train_kd", 
                   "val_loss", "val_box", "val_cls", "val_kpt", "lr"]
        
        # Ensure log file exists/is empty
        with open(self.log_csv, "w", newline="") as f:
            csv.writer(f).writerow(headers)

        logging.info(f"[Trainer] Start training loop. Epochs: {self.cfg.EPOCHS}")
        
        for epoch in range(self.cfg.EPOCHS):
            if self.stop_requested: 
                logging.warning("[Trainer] Stop requested by user.")
                break
            
            print(f"\nEpoch {epoch+1}/{self.cfg.EPOCHS}")
            
            # --- Training Loop ---
            pbar = tqdm(range(steps_per_epoch), unit="step", desc="Train")
            metrics_sum = {"total": 0.0, "box": 0.0, "cls": 0.0, "kpt": 0.0, "kd": 0.0}
            iter_ds = iter(train_ds)
            
            for _ in pbar:
                try:
                    batch = next(iter_ds)
                    imgs, labels = batch[0], batch[1]
                 
                    # Determine if we should freeze BN layers (last 10% of training)
                    freeze_bn = (self.global_step > (self.cfg.EPOCHS * 0.9) * steps_per_epoch)
                    logs = self.train_step(imgs, labels, optimizer, class_weights, freeze_bn)
                 
                    for k, v in logs.items():
                        metrics_sum[k] += float(v)
                 
                    lr_curr = float(lr_schedule(self.global_step))
                    pbar.set_postfix(loss=f"{float(logs['total']):.4f}", lr=f"{lr_curr:.1e}")
                    self.global_step += 1
                except StopIteration:
                    break
            
            avg_train = {k: v / steps_per_epoch for k, v in metrics_sum.items()}
            
            # --- Validation Loop ---
            # Save generic plot occasionally
            if self.global_step % 1000 == 0: 
                save_gt_and_plot(imgs, labels, self.plots_dir, self.global_step)

            avg_val = {"total": 0.0, "box": 0.0, "cls": 0.0, "kpt": 0.0}
            
            if val_ds and val_steps > 0:
                print("Running Validation...")
                val_metrics_sum = {"total": 0.0, "box": 0.0, "cls": 0.0, "kpt": 0.0}
                iter_val = iter(val_ds)
                
                # Use EMA weights for validation
                self.ema.apply_to(self.student)
                
                for _ in tqdm(range(val_steps), unit="step", desc="Val"):
                    try:
                        batch = next(iter_val)
                        imgs, labels = batch[0], batch[1]
                        logs = self.val_step(imgs, labels, class_weights)
                        
                        for k, v in logs.items():
                            val_metrics_sum[k] += float(v)
                    except StopIteration:
                        break
                
                # Restore original weights after validation
                self.ema.restore(self.student)
                
                avg_val = {k: v / val_steps for k, v in val_metrics_sum.items()}
                print(f"Val Loss: {avg_val['total']:.4f}")
                
                # Plot Validation Predictions
                # Use a fresh batch for prediction plotting if needed, or the last one from loop
                # Re-apply EMA momentarily for visualization
                self.ema.apply_to(self.student)
                y_pred_val = self.student(imgs, training=False)

                if isinstance(y_pred_val, (list, tuple)):
                    y_pred_val = y_pred_val[0]

                save_pred_and_plot(imgs, y_pred_val, self.plots_dir, self.global_step,
                   num_cls=self.cfg.NUM_CLS, total_C=self.cfg.total_output_channels)
                
                self.ema.restore(self.student)

            # --- Logging & Saving ---
            with open(self.log_csv, "a", newline="") as f:
                csv.writer(f).writerow([
                    epoch + 1, 
                    avg_train['total'], avg_train['box'], avg_train['cls'], avg_train['kpt'], avg_train['kd'],
                    avg_val['total'], avg_val['box'], avg_val['cls'], avg_val['kpt'],
                    float(lr_schedule(self.global_step))
                ])

            # Save Last Model
            self.student.save_weights(self.models_dir / "last_model.weights.h5")
            
            # Save Best Model
            monitor_metric = avg_val['total'] if (val_ds and avg_val['total'] > 0) else avg_train['total']
            
            if monitor_metric < self.best_val_loss:
                self.best_val_loss = monitor_metric
                self.student.save_weights(self.models_dir / "best_model.weights.h5")
                
                # Save Best EMA
                self.ema.apply_to(self.student)
                self.student.save_weights(self.models_dir / "best_ema.weights.h5")
                self.ema.restore(self.student)
                print(f"Saved Best Model (Loss={monitor_metric:.4f})")