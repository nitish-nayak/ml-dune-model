"""
DINO training script for DUNE sparse UNet backbone.

Usage:
    python dino/train_dino.py --epochs=100 --batch_size=16 --backbone_name=attn_default
    python dino/train_dino.py --epochs=2 --batch_size=4 --test_mode=True --debug=True
"""

import fire
import inspect
import json
import sys
import torch
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader

from loader.apa_sparse_dataset import APASparseDataset
from loader.collate import voxels_collate_fn
from loader.splits import train_val_split, Subset

from .config import DINOConfig
from .transforms import FeatureLogTransform
from .masking import SparseVoxelMasker
from .cropping import CropConfig, SparseCropper
from .loss import PixelDINOLoss
from .scheduler import CosineScheduler
from .model import DINODuneModel, match_and_gather
from .debug import DINODebugger


@torch.no_grad()
def validate_epoch(model, val_loader, cropper, masker, loss_fn, device, normalizer,
                   use_cropping=False, use_masking=True):
    """
    Compute mean DINO loss on the validation set.

    Student runs in eval mode with the same augmentation strategy as training.
    Teacher is always in eval mode. No gradients are computed.

    Returns:
        mean validation loss (0.0 if val_loader is empty)
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for xs in val_loader:
        xs = xs.to(device)
        if normalizer is not None:
            xs = normalizer(xs)

        if use_cropping:
            all_views = cropper(xs)
            n_global  = cropper.cfg.n_global
        else:
            all_views = [xs]
            n_global  = 1
        n_crops = len(all_views)

        teacher_encoded = [model.encode_teacher(all_views[g]) for g in range(n_global)]

        batch_loss = None
        n_pairs = 0
        for k in range(n_crops):
            view_k = masker(all_views[k])[0] if use_masking else all_views[k]
            student_backbone_k, student_out_k = model.encode_student(view_k)
            for g in range(n_global):
                if k == g and n_crops > 1:
                    continue
                _, teacher_out_g = teacher_encoded[g]
                s_feats, s_bb_feats, t_feats, counts = match_and_gather(
                    student_out_k, student_backbone_k, teacher_out_g,
                )
                loss_k, _, _, _, _, _ = loss_fn(s_feats, s_bb_feats, t_feats, counts)
                batch_loss = loss_k if batch_loss is None else batch_loss + loss_k
                n_pairs += 1
        total_loss += (batch_loss / n_pairs).item()
        n_batches += 1

    model.train()  # restores student; teacher stays eval via DINODuneModel.train()
    return total_loss / n_batches if n_batches > 0 else 0.0


def main(
    backbone_name: str = "attn_default",
    encoding_range: float = 125.0,
    epochs: int = 100,
    batch_size: int = 50,
    lr: float = 1e-4,
    use_cropping: bool = True,
    use_masking: bool = True,
    mask_ratio: float = 0.5,
    crop_n_global: int = 2,
    crop_n_local: int = 4,
    crop_global_scale: tuple = (0.4, 1.0),
    crop_local_scale: tuple = (0.05, 0.2),
    crop_aspect_ratio: tuple = (0.75, 1.333),
    crop_blur_sigma_px: float = 10.0,
    crop_heatmap_power: float = 1.0,
    crop_min_active_pixels: int = 10,
    loss_type: str = "dino",
    center_momentum: float = 0.9,
    use_centering: bool = True,
    teacher_temp: float = 0.07,
    student_temp: float = 0.1,
    use_proj_head: bool = True,
    proj_head_hidden_dim: int = 256,
    proj_head_output_dim: int = 128,
    proj_head_n_layers: int = 2,
    use_cov_penalty: bool = True,
    cov_penalty_weight: float = 1.0,
    use_var_penalty: bool = True,
    var_penalty_weight: float = 1.0,
    var_gamma: float = 0.5,
    momentum_start: float = 0.998,
    momentum_end: float = 0.9999,
    weight_decay: float = 0.04,
    weight_decay_end: float = 0.4,
    warmup_epochs: int = 1,
    datadir: str = "/nfs/data/1/yuhw/cffm-data/prod-jay-100k-truth-2026-02-27",
    cache_dir: str = "./data",
    use_log_transform: bool = True,
    feat_min_val: float = 3.75,
    feat_max_val: float = 83861.2,
    output_dir: str = "./dino_checkpoints",
    save_every: int = 10,
    device: str = "cuda",
    debug: bool = True,
    debug_every: int = 100,
    debug_dir: str = "./dino_debug",
    run_name: str = "",
    test_mode: bool = True,
    num_workers: int = 4,
):
    """
    DINO training loop for DUNE detector.

    Args:
        backbone_name: Model architecture ("attn_default", "base", etc.)
        epochs: Number of training epochs
        batch_size: Batch size per GPU
        lr: Base learning rate
        use_cropping: enable activity-aware multi-crop augmentation
        use_masking: enable random pixel dropout on student views
        mask_ratio: Fraction of active pixels to mask (masking mode only)
        crop_n_global: Number of global crops per image (cropping mode)
        crop_n_local: Number of local crops per image (cropping mode)
        crop_global_scale: Global crop area range as fraction of image area
        crop_local_scale: Local crop area range as fraction of image area
        crop_aspect_ratio: Crop width-to-height aspect ratio range
        crop_blur_sigma_px: Gaussian blur sigma for activity heatmap (px)
        crop_heatmap_power: Exponent applied to heatmap before sampling
        crop_min_active_pixels: Minimum active voxels required inside a crop
        loss_type: "cosine", "mse", or "dino"
        center_momentum: EMA decay for the teacher center buffer
        use_centering: subtract running center from teacher features before loss
        teacher_temp: teacher softmax temperature (only used for "dino")
        student_temp: student softmax temperature (only used for "dino")
        use_proj_head: attach DINO MLP projection head between backbone and loss
        proj_head_hidden_dim: inner MLP width of the projection head
        proj_head_output_dim: output dimension of the projection head's final FC layer
        proj_head_n_layers: number of MLP layers before the final FC
        use_cov_penalty: add VICReg covariance decorrelation penalty on student features
        cov_penalty_weight: weight for the covariance penalty term
        use_var_penalty: add VICReg variance penalty (hinge on per-dim std >= var_gamma)
        var_penalty_weight: weight for the variance penalty term
        var_gamma: target minimum std per feature dimension
        momentum_start: Initial EMA momentum
        momentum_end: Final EMA momentum
        weight_decay: L2 regularization
        weight_decay_end: Final weight decay (cosine annealed)
        warmup_epochs: Linear warmup duration
        datadir: Root directory of the sparse dataset on disk
        cache_dir: Where to cache the dataset index .pt file
        output_dir: Where to save checkpoints
        save_every: Save checkpoint every N epochs
        device: "cuda" or "cpu"
        debug: Enable debugging and history logging
        debug_every: Log scalars / stats / grad norms every N batches
        debug_dir: Base directory for debug outputs
        run_name: Optional label; outputs go to debug_dir/run_name/ if set
        test_mode: Use small subset for quick smoke tests
        num_workers: Number of dataloader workers
    """
    # ============ Setup ============
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # If a run name is given, nest outputs under debug_dir/run_name/ and output_dir/run_name/
    if run_name:
        debug_dir = f"{debug_dir}/{run_name}"
        output_dir = f"{output_dir}/{run_name}"

    # Build config
    # normalize_features is the negation of use_proj_head:
    # the head's internal L2 norm handles normalisation when the head is active.
    normalize_features = not use_proj_head

    cfg = DINOConfig(
        backbone_name=backbone_name,
        encoding_range=encoding_range,
        use_cropping=use_cropping,
        use_masking=use_masking,
        mask_ratio=mask_ratio,
        crop_n_global=crop_n_global,
        crop_n_local=crop_n_local,
        crop_global_scale=crop_global_scale,
        crop_local_scale=crop_local_scale,
        crop_aspect_ratio=crop_aspect_ratio,
        crop_blur_sigma_px=crop_blur_sigma_px,
        crop_heatmap_power=crop_heatmap_power,
        crop_min_active_pixels=crop_min_active_pixels,
        use_proj_head=use_proj_head,
        proj_head_hidden_dim=proj_head_hidden_dim,
        proj_head_output_dim=proj_head_output_dim,
        proj_head_n_layers=proj_head_n_layers,
        loss_type=loss_type,
        normalize_features=normalize_features,
        center_momentum=center_momentum,
        use_centering=use_centering,
        teacher_temp=teacher_temp,
        student_temp=student_temp,
        use_cov_penalty=use_cov_penalty,
        cov_penalty_weight=cov_penalty_weight,
        use_var_penalty=use_var_penalty,
        var_penalty_weight=var_penalty_weight,
        var_gamma=var_gamma,
        momentum_start=momentum_start,
        momentum_end=momentum_end,
        lr=lr,
        weight_decay=weight_decay,
        weight_decay_end=weight_decay_end,
        warmup_epochs=warmup_epochs,
        epochs=epochs,
        datadir=datadir,
        use_log_transform=use_log_transform,
        feat_min_val=feat_min_val,
        feat_max_val=feat_max_val,
        batch_size=batch_size,
        cache_dir=cache_dir,
        output_dir=output_dir,
        save_every=save_every,
        debug=debug,
        debug_every=debug_every,
        debug_dir=debug_dir,
        run_name=run_name,
        num_workers=num_workers,
    )

    print(f"Device: {device}")

    print("Model:")
    print(f"  backbone_name        = {cfg.backbone_name}")
    print(f"  encoding_range       = {cfg.encoding_range}")
    print(f"  use_proj_head        = {cfg.use_proj_head}")
    print(f"  proj_head_hidden_dim = {cfg.proj_head_hidden_dim}")
    print(f"  proj_head_output_dim = {cfg.proj_head_output_dim}")
    print(f"  proj_head_n_layers   = {cfg.proj_head_n_layers}")

    print("Training:")
    print(f"  epochs         = {cfg.epochs}")
    print(f"  lr             = {cfg.lr}")
    print(f"  batch_size     = {cfg.batch_size}")
    print(f"  warmup_epochs  = {cfg.warmup_epochs}")
    print(f"  momentum_start = {cfg.momentum_start}")
    print(f"  momentum_end   = {cfg.momentum_end}")

    print("Augmentation:")
    print(f"  use_cropping           = {cfg.use_cropping}")
    print(f"  use_masking            = {cfg.use_masking}")
    print(f"  mask_ratio             = {cfg.mask_ratio}")
    print(f"  crop_n_global          = {cfg.crop_n_global}")
    print(f"  crop_n_local           = {cfg.crop_n_local}")
    print(f"  crop_global_scale      = {cfg.crop_global_scale}")
    print(f"  crop_local_scale       = {cfg.crop_local_scale}")
    print(f"  crop_aspect_ratio      = {cfg.crop_aspect_ratio}")
    print(f"  crop_blur_sigma_px     = {cfg.crop_blur_sigma_px}")
    print(f"  crop_heatmap_power     = {cfg.crop_heatmap_power}")
    print(f"  crop_min_active_pixels = {cfg.crop_min_active_pixels}")

    print("Data normalization:")
    print(f"  use_log_transform = {cfg.use_log_transform}")
    print(f"  feat_min_val      = {cfg.feat_min_val}")
    print(f"  feat_max_val      = {cfg.feat_max_val}")

    print("Loss:")
    print(f"  type                = {cfg.loss_type}")
    print(f"  center_momentum     = {cfg.center_momentum}")
    print(f"  use_centering       = {cfg.use_centering}")
    print(f"  teacher_temp        = {cfg.teacher_temp}")
    print(f"  student_temp        = {cfg.student_temp}")
    print(f"  use_cov_penalty     = {cfg.use_cov_penalty}")
    print(f"  cov_penalty_weight  = {cfg.cov_penalty_weight}")
    print(f"  use_var_penalty     = {cfg.use_var_penalty}")
    print(f"  var_penalty_weight  = {cfg.var_penalty_weight}")
    print(f"  var_gamma           = {cfg.var_gamma}")

    # ============ Data ============
    print("\nLoading dataset:", cfg.datadir)
    dataset = APASparseDataset(
        datadir=cfg.datadir,
        apa=cfg.apa,
        view=cfg.view,
        use_cache=True,
        cache_dir=cfg.cache_dir
    )

    if test_mode:
        n_subset = 100000
        print(f"TEST MODE: using {n_subset} samples")
        subset_indices = torch.randperm(len(dataset))[:n_subset]
        dataset = Subset(dataset, subset_indices)

    train_ds, val_ds, train_idx, val_idx = train_val_split(dataset, val_fraction=0.2, use_cache=False)

    train_set = set(train_idx.tolist())
    val_set   = set(val_idx.tolist())
    overlap = train_set & val_set
    print(f"Train size: {len(train_set)}, Val size: {len(val_set)}")
    print(f"Overlap: {len(overlap)} samples")                    # should be 0
    print(f"Union covers full dataset: {len(train_set | val_set) == len(dataset)}")  # should be True

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=voxels_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=voxels_collate_fn,
    )

    epoch_len = len(train_loader)
    total_iters = epochs * epoch_len
    print(f"Total training iterations: {total_iters} (epochs={epochs}, epoch_len={epoch_len})")

    # ============ Model, optimizer, loss ============
    print("\nBuilding model...")
    model = DINODuneModel(
        backbone_name=backbone_name,
        encoding_range=cfg.encoding_range,
        use_proj_head=use_proj_head,
        proj_head_hidden_dim=proj_head_hidden_dim,
        proj_head_output_dim=proj_head_output_dim,
        proj_head_n_layers=proj_head_n_layers,
    ).to(device)
    # Optimise backbone + head (if present)
    student_params = list(model.student.parameters())
    if model.student_head is not None:
        student_params += list(model.student_head.parameters())
    optimizer = optim.AdamW(student_params, lr=lr, weight_decay=weight_decay)

    normalizer = FeatureLogTransform(cfg.feat_min_val, cfg.feat_max_val) if cfg.use_log_transform else None

    masker = SparseVoxelMasker(mask_ratio=mask_ratio)

    cropper = None
    if use_cropping:
        crop_cfg = CropConfig(
            n_global=cfg.crop_n_global,
            n_local=cfg.crop_n_local,
            global_scale=cfg.crop_global_scale,
            local_scale=cfg.crop_local_scale,
            aspect_ratio=cfg.crop_aspect_ratio,
            blur_sigma_px=cfg.crop_blur_sigma_px,
            heatmap_power=cfg.crop_heatmap_power,
            min_active_pixels=cfg.crop_min_active_pixels,
            image_h=cfg.image_h,
            image_w=cfg.image_w,
        )
        cropper = SparseCropper(crop_cfg)

    loss_fn = PixelDINOLoss(
        loss_type=cfg.loss_type,
        normalize_features=cfg.normalize_features,
        center_momentum=cfg.center_momentum,
        use_centering=cfg.use_centering,
        teacher_temp=cfg.teacher_temp,
        student_temp=cfg.student_temp,
        use_cov_penalty=cfg.use_cov_penalty,
        cov_penalty_weight=cfg.cov_penalty_weight,
        use_var_penalty=cfg.use_var_penalty,
        var_penalty_weight=cfg.var_penalty_weight,
        var_gamma=cfg.var_gamma,
    ).to(device)

    # ============ Schedulers ============
    warmup_iters = min(warmup_epochs * epoch_len, int(0.2 * total_iters))
    print(f"Schedules: total_iters={total_iters}, warmup_iters={warmup_iters}")

    lr_schedule = CosineScheduler(
        base_value=cfg.lr,
        final_value=cfg.min_lr,
        total_iters=total_iters,
        warmup_iters=warmup_iters,
    )
    wd_schedule = CosineScheduler(
        base_value=cfg.weight_decay,
        final_value=cfg.weight_decay_end,
        total_iters=total_iters,
    )
    momentum_schedule = CosineScheduler(
        base_value=cfg.momentum_start,
        final_value=cfg.momentum_end,
        total_iters=total_iters,
    )

    # ============ Checkpointing ============
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ============ Debugging ============
    debugger = DINODebugger(cfg, enabled=True)
    debugger.log_config(cfg)

    # ============ Training loop ============
    print("\nStarting training...")
    first_batch = True

    for epoch in range(1, epochs + 1):
        model.train()

        for batch_idx, xs in enumerate(train_loader):
            iteration = (epoch - 1) * epoch_len + batch_idx
            xs = xs.to(device)
            if normalizer is not None:
                xs = normalizer(xs)

            # Apply schedules
            lr_val = lr_schedule[iteration]
            wd_val = wd_schedule[iteration]
            mom_val = momentum_schedule[iteration]

            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_val
                param_group["weight_decay"] = wd_val

            # Forward + backward
            optimizer.zero_grad()
            (loss_val, teacher_entropy, student_entropy,
             kl, cov_penalty, var_penalty,
             student_backbone_out, teacher_backbone_out,
             student_out, teacher_out) = model.forward_backward(
                xs, cropper, masker, loss_fn, use_cropping, use_masking,
            )
            optimizer.step()

            # EMA teacher update
            model.update_teacher(mom_val)

            # Centering: update teacher center for next iteration
            loss_fn.update_center(teacher_out)
            debugger.log_center_stats(iteration, loss_fn)

            # Scalar logging — the (teacher_entropy, student_entropy, kl) trio already
            # reflects whatever goes into the loss (head output if a head is present,
            # raw backbone otherwise), so no separate backbone-entropy diagnostic is needed.
            n_valid = student_out.feature_tensor.shape[0]
            debugger.log_batch(epoch, batch_idx, iteration, loss_val, n_valid, lr_val, mom_val, teacher_entropy, student_entropy, kl, cov_penalty, var_penalty)
            debugger.log_gpu_memory(iteration)

            # Gradient norms per backbone module (.grad still populated before next zero_grad)
            debugger.log_gradient_norms(iteration, model.student)

            # Representation-quality statistics (variance, covariance, norm)
            # Pass head features separately when a head is present (student_out != student_backbone_out)
            s_head_feats = student_out.feature_tensor if model.student_head is not None else None
            t_head_feats = teacher_out.feature_tensor if model.teacher_head is not None else None
            debugger.log_feature_stats(iteration, student_backbone_out.feature_tensor, teacher_backbone_out.feature_tensor,
                                        s_head_feats, t_head_feats)

            # First batch: log tensor shapes
            if first_batch:
                debugger.log_shapes(xs.feature_tensor, student_backbone_out.feature_tensor, teacher_backbone_out.feature_tensor)
                first_batch = False

            # Periodically persist histories to disk
            debugger.maybe_save_histories(iteration)

            # Free Voxels objects to release GPU memory before the next forward pass
            del student_backbone_out, student_out, teacher_backbone_out, teacher_out

            if (batch_idx + 1) % 50 == 0 or batch_idx == 0:
                cov_str = f", cov={cov_penalty:.4f}" if cov_penalty is not None else ""
                var_str = f", var={var_penalty:.4f}" if var_penalty is not None else ""
                mem_str = f", gpu={debugger.last_peak_alloc_gib:.2f}GiB"
                print(f"[{epoch}/{epochs}] iter {iteration}: loss={loss_val:.6f}, "
                      f"lr={lr_val:.2e}, mom={mom_val:.6f}{cov_str}{var_str}{mem_str}")

        # Validation
        #val_loss = validate_epoch(model, val_loader, cropper, masker, loss_fn, device, normalizer, use_cropping, use_masking)
        #print(f"[{epoch}/{epochs}] val_loss={val_loss:.6f}")
        #debugger.log_val_epoch(epoch, iteration, val_loss)

        # Save checkpoint
        if epoch % save_every == 0 or epoch == epochs:
            ckpt = {
                "epoch": epoch,
                "student": model.student.state_dict(),
                "teacher": model.teacher.state_dict(),
                "optimizer": optimizer.state_dict(),
                "cfg": cfg,
            }
            if model.student_head is not None:
                ckpt["student_head"] = model.student_head.state_dict()
                ckpt["teacher_head"] = model.teacher_head.state_dict()
            ckpt_path = output_dir / f"checkpoint_epoch{epoch}.pt"
            torch.save(ckpt, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    debugger.save_histories()

    print("\nTraining complete!")
    print(f"Checkpoints saved to: {output_dir}")


def from_config(
    config_path: str,
    run_name: str = "",
    device: str = "cuda",
    test_mode: bool = False,
    **overrides,
):
    """
    Start training from a saved run_config.json file.

    Loads training parameters from a previously saved run_config.json (e.g. from
    ./dino_debug/<run_name>/run_config.json).  Any JSON field that does not match
    a parameter of main() is silently ignored, so old configs with stale or missing
    keys work without errors — missing fields fall back to main()'s defaults.

    The `run_name`, `device`, and `test_mode` arguments override the corresponding
    values from the config file.  Any other parameter accepted by main() can also
    be overridden via CLI (e.g. --use_cov_penalty=True --cov_penalty_weight=1e-2).
    Providing a new run_name is recommended when re-running a config so outputs
    don't overwrite the original run.

    Args:
        config_path: Path to the run_config.json file
        run_name: Override run name (determines output sub-directories)
        device: Override device ("cuda" or "cpu")
        test_mode: Override test_mode flag
        **overrides: Any additional main() parameter to override (e.g. use_cov_penalty=True)
    """
    with open(config_path) as f:
        raw = json.load(f)

    # Discover which parameters main() accepts (names + defaults)
    sig = inspect.signature(main)
    valid_params = set(sig.parameters)

    # Build kwargs: only keep JSON keys that main() understands
    kwargs = {k: v for k, v in raw.items() if k in valid_params}

    # The JSON stores debug_dir and output_dir as fully-nested paths (base/run_name).
    # main() will re-append run_name, so we strip the suffix here to avoid
    # double-nesting.  We use the original run_name from the JSON for this,
    # before any CLI override is applied.
    orig_run_name = kwargs.get("run_name", "")
    stored_debug_dir = kwargs.get("debug_dir", "")
    if orig_run_name and stored_debug_dir.endswith("/" + orig_run_name):
        kwargs["debug_dir"] = stored_debug_dir[: -len("/" + orig_run_name)]
    stored_output_dir = kwargs.get("output_dir", "")
    if orig_run_name and stored_output_dir.endswith("/" + orig_run_name):
        kwargs["output_dir"] = stored_output_dir[: -len("/" + orig_run_name)]

    # CLI-level overrides always win
    if run_name:
        kwargs["run_name"] = run_name
    kwargs["device"] = device
    kwargs["test_mode"] = test_mode
    for k, v in overrides.items():
        if k in valid_params:
            kwargs[k] = v

    main(**kwargs)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "from_config":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        fire.Fire(from_config)
    else:
        fire.Fire(main)
