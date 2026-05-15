"""Configuration dataclass for DINO training."""

from dataclasses import dataclass


@dataclass
class DINOConfig:
    """Configuration for DUNE-DINO self-supervised training."""

    # ============ Run ============
    run_name: str = ""                # optional label; outputs go to debug_dir/run_name/ if set

    # ============ Data ============
    datadir: str = "/nfs/data/1/yuhw/cffm-data/prod-jay-100k-truth-2026-02-27"
    apa: int = 0                      # which APA to train on
    view: str = "W"                   # which wire plane view ("U", "V", or "W")
    image_h: int = 1500               # spatial resolution: height (time ticks)
    image_w: int = 1050               # spatial resolution: width (wire channels)
    n_subset: int = -1                # -1 = full dataset; otherwise cap at N samples
    batch_size: int = 16
    num_workers: int = 4
    cache_dir: str = "./data"           # directory for cached dataset index .pt file
    use_log_transform: bool = True     # apply FeatureLogTransform to raw charge before model
    feat_min_val: float = 3.75        # 2nd percentile of raw charge [ADC]; anchors y = -1
    feat_max_val: float = 83861.2     # 99.999th percentile of raw charge [ADC]; anchors y = +1

    # ============ Backbone ============
    backbone_name: str = "attn_default"  # key into BACKBONE_REGISTRY
    encoding_range: float = 125.0        # sinusoidal positional encoding range
    feature_dim: int = 64                # backbone output channels; must match model's output
    use_proj_head: bool = True           # attach MLP projection head between backbone and loss
    proj_head_hidden_dim: int = 256      # inner MLP width
    proj_head_output_dim: int = 128      # output dimension of the final FC layer
    proj_head_n_layers: int = 2          # number of MLP layers before the final FC

    # ============ Augmentation ============
    use_cropping: bool = True             # enable activity-aware multi-crop augmentation
    use_masking: bool = True              # enable random pixel dropout on student views
    mask_ratio: float = 0.5              # fraction of active pixels to mask (masking mode)
    crop_n_global: int = 2               # number of global crops per image
    crop_n_local: int = 4                # number of local crops per image
    crop_global_scale: tuple = (0.4, 1.0)    # global crop area as fraction of image area
    crop_local_scale: tuple = (0.05, 0.2)    # local crop area as fraction of image area
    crop_aspect_ratio: tuple = (0.75, 1.333) # width-to-height aspect ratio range
    crop_blur_sigma_px: float = 10.0     # Gaussian blur sigma for activity heatmap (px)
    crop_heatmap_power: float = 1.0      # exponent applied to heatmap before sampling
    crop_min_active_pixels: int = 10     # minimum active voxels required inside a crop

    # ============ Training ============
    epochs: int = 10
    momentum_start: float = 0.996        # EMA teacher momentum schedule start
    momentum_end: float = 0.9999         # EMA teacher momentum schedule end
    lr: float = 1e-4                     # base learning rate
    min_lr: float = 1e-6                 # minimum learning rate (cosine annealing floor)
    weight_decay: float = 0.04
    weight_decay_end: float = 0.4        # weight decay at end of training (cosine annealed)
    warmup_epochs: int = 1               # linear LR warmup duration

    # ============ Loss ============
    loss_type: str = "dino"              # "cosine", "mse", or "dino"
    use_centering: bool = True           # subtract running center from teacher before loss
    center_momentum: float = 0.996       # EMA decay for teacher centering
    teacher_temp: float = 0.04           # teacher softmax temperature
    student_temp: float = 0.1            # student softmax temperature
    normalize_features: bool = False     # L2-normalise features before loss; False when use_proj_head=True
    use_cov_penalty: bool = True         # VICReg covariance decorrelation penalty
    cov_penalty_weight: float = 10.0
    use_var_penalty: bool = False        # VICReg variance penalty (hinge on per-dim std)
    var_penalty_weight: float = 1.0
    var_gamma: float = 1.0               # target minimum std per feature dimension

    # ============ Checkpointing & debug ============
    output_dir: str = "./dino_checkpoints"
    save_every: int = 10
    debug: bool = False                  # enable detailed logging and history tracking
    debug_every: int = 100               # log scalars / stats / grad norms every N batches
    debug_dir: str = "./dino_debug"
