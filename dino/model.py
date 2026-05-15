"""DINO training model: student + teacher with EMA update."""

import inspect
import torch
import torch.nn as nn
from torch import Tensor

from warpconvnet.geometry.types.voxels import Voxels

from models import BACKBONE_REGISTRY
from .projhead import DINOProjectionHead


def match_and_gather(
    s_out: Voxels,
    s_backbone: Voxels,
    t_out: Voxels,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Match student and teacher output voxels by spatial coordinates and
    return pre-aligned feature tensors ready for the loss.

    Uses flat coordinate keys (y * W + x) with searchsorted for efficient
    intersection.  No LUTs or index bookkeeping from the cropper are needed —
    coordinates are intrinsic to the backbone output.

    Args:
        s_out:      Student head output (Voxels).
        s_backbone: Student backbone output (Voxels), same coordinates as s_out.
        t_out:      Teacher head output (Voxels).

    Returns:
        s_feats:    [N_matched, D_head]  student head features at intersection
        s_bb_feats: [N_matched, D_bb]    student backbone features at intersection
        t_feats:    [N_matched, D_head]  teacher head features at intersection
        counts:     [B] int64 per-image matched voxel counts
    """
    B = len(s_out.offsets) - 1
    device = s_out.feature_tensor.device

    # Flat key = y * W + x maps each 2D coordinate to a unique integer.
    # W must exceed the max x-coordinate across both inputs.
    W = 1
    if s_out.coordinate_tensor.shape[0] > 0:
        W = max(W, int(s_out.coordinate_tensor[:, 0].max().item()) + 1)
    if t_out.coordinate_tensor.shape[0] > 0:
        W = max(W, int(t_out.coordinate_tensor[:, 0].max().item()) + 1)

    # Collect global indices (into the flat batched feature tensors) for all
    # matched voxels across all batch items.
    s_global_idx = []
    t_global_idx = []
    counts_list = []

    for b in range(B):
        s_start, s_end = int(s_out.offsets[b]), int(s_out.offsets[b + 1])
        t_start, t_end = int(t_out.offsets[b]), int(t_out.offsets[b + 1])

        s_coords = s_out.coordinate_tensor[s_start:s_end]
        t_coords = t_out.coordinate_tensor[t_start:t_end]

        if s_coords.shape[0] == 0 or t_coords.shape[0] == 0:
            counts_list.append(0)
            continue

        s_keys = s_coords[:, 1].long() * W + s_coords[:, 0].long()
        t_keys = t_coords[:, 1].long() * W + t_coords[:, 0].long()

        # Sort teacher keys so we can binary-search student keys into them
        t_sorted, t_order = t_keys.sort()

        # For each student key, find its insertion point in the sorted teacher keys
        pos = torch.searchsorted(t_sorted, s_keys)
        # Clamp so we can safely index t_sorted; out-of-range means no match
        pos = pos.clamp(max=t_sorted.shape[0] - 1)
        # A student key is matched iff t_sorted[pos] equals it exactly
        valid = t_sorted[pos] == s_keys

        # s_local: which student voxels (local to this batch item) have a match
        s_local = valid.nonzero(as_tuple=False).squeeze(1)
        if s_local.numel() == 0:
            counts_list.append(0)
            continue

        # t_order maps sorted positions back to original teacher ordering;
        # pos[valid] gives the sorted positions of matched keys
        t_local = t_order[pos[valid]]

        # Shift local indices to global positions in the flat feature tensors
        s_global_idx.append(s_local + s_start)
        t_global_idx.append(t_local + t_start)
        counts_list.append(s_local.shape[0])

    counts = torch.tensor(counts_list, dtype=torch.int64, device=device)

    # Single gather over the concatenated feature tensors — no intermediate
    # Voxels objects needed.
    if s_global_idx:
        s_idx = torch.cat(s_global_idx)
        t_idx = torch.cat(t_global_idx)
        s_feats    = s_out.feature_tensor[s_idx]
        s_bb_feats = s_backbone.feature_tensor[s_idx]  # same coords as s_out
        t_feats    = t_out.feature_tensor[t_idx]
    else:
        D_head = s_out.feature_tensor.shape[1]
        D_bb   = s_backbone.feature_tensor.shape[1]
        s_feats    = s_out.feature_tensor.new_zeros(0, D_head)
        s_bb_feats = s_backbone.feature_tensor.new_zeros(0, D_bb)
        t_feats    = t_out.feature_tensor.new_zeros(0, D_head)

    return s_feats, s_bb_feats, t_feats, counts


class DINODuneModel(nn.Module):
    """
    DINO-style teacher/student framework for DUNE backbone.

    - Student: trainable backbone (+ optional projection head), receives masked input
    - Teacher: frozen backbone (+ optional projection head), receives full input, EMA-updated from student
    - Both share the same architecture; training updates only student

    Key: teacher parameters are NOT updated by gradients, only by explicit EMA update.
    """

    def __init__(
        self,
        backbone_name: str = "attn_default",
        use_proj_head: bool = False,
        proj_head_hidden_dim: int = 256,
        proj_head_output_dim: int = 256,
        proj_head_n_layers: int = 4,
        encoding_range: float = 125.0,
    ):
        """
        Args:
            backbone_name:        Key into BACKBONE_REGISTRY (e.g., "attn_default", "base")
            use_proj_head:        Attach a DINO MLP projection head after the backbone
            proj_head_hidden_dim: Inner MLP width (DINO paper uses 2048)
            proj_head_output_dim: Output dimension of the final FC layer
            proj_head_n_layers:   Number of MLP layers before the final FC
            encoding_range:       Sinusoidal positional encoding range passed to the backbone
        """
        super().__init__()

        # Instantiate both backbones (sparse: Voxels → Voxels)
        # Only pass encoding_range to backbones that accept it (attn_* variants).
        backbone_cls = BACKBONE_REGISTRY[backbone_name]
        backbone_kwargs = {}
        if "encoding_range" in inspect.signature(backbone_cls.__init__).parameters:
            backbone_kwargs["encoding_range"] = encoding_range
        print("Initializing STUDENT backbone:")
        self.student = backbone_cls(**backbone_kwargs)
        print("Initializing TEACHER backbone:")
        self.teacher = backbone_cls(**backbone_kwargs)

        # Initialize teacher with student weights
        self.teacher.load_state_dict(self.student.state_dict())

        # Optional projection heads — teacher head is the EMA of the student head,
        # exactly as in the original DINO (head is part of the full student network).
        if use_proj_head:
            in_dim = 64  # backbone output channels — must match architecture
            self.student_head = DINOProjectionHead(in_dim, proj_head_hidden_dim, proj_head_output_dim, proj_head_n_layers)
            self.teacher_head = DINOProjectionHead(in_dim, proj_head_hidden_dim, proj_head_output_dim, proj_head_n_layers)
            self.teacher_head.load_state_dict(self.student_head.state_dict())
        else:
            self.student_head = None
            self.teacher_head = None

        # Freeze teacher backbone and head: no gradients
        for p in self.teacher.parameters():
            p.requires_grad = False
        if self.teacher_head is not None:
            for p in self.teacher_head.parameters():
                p.requires_grad = False

        # Teacher always in eval mode (batchnorm, dropout, etc.)
        self.teacher.eval()
        if self.teacher_head is not None:
            self.teacher_head.eval()

    def train(self, mode: bool = True):
        """Override train() to keep teacher (and teacher head) in eval mode."""
        super().train(mode)
        self.teacher.eval()
        if self.teacher_head is not None:
            self.teacher_head.eval()
        return self

    @torch.no_grad()
    def update_teacher(self, momentum: float):
        """
        EMA update: teacher = momentum * teacher + (1 - momentum) * student

        Covers both backbone and projection head (if present), mirroring the
        original DINO where the full student (backbone + head) is momentum-encoded.

        Args:
            momentum: EMA momentum (typically 0.996 → 0.9999 over training)
        """
        pairs = list(zip(self.student.parameters(), self.teacher.parameters()))
        if self.student_head is not None:
            pairs += list(zip(self.student_head.parameters(), self.teacher_head.parameters()))
        for s_param, t_param in pairs:
            t_param.data.mul_(momentum).add_((1.0 - momentum) * s_param.data)

    def encode_teacher(self, xs: Voxels) -> Voxels:
        """Run the teacher backbone (+ head if present). Always called in no_grad context."""
        backbone_out = self.teacher(xs)
        if self.teacher_head is not None:
            return backbone_out, self.teacher_head(backbone_out)
        return backbone_out, backbone_out

    def encode_student(self, xs: Voxels) -> tuple[Voxels, Voxels]:
        """
        Run the student backbone (+ head if present).

        Returns:
            backbone_out: raw 64-dim backbone output (before head)
            final_out:    head output if head present, else same as backbone_out
        """
        backbone_out = self.student(xs)
        if self.student_head is not None:
            return backbone_out, self.student_head(backbone_out)
        return backbone_out, backbone_out

    def forward_backward(
        self,
        xs: Voxels,
        cropper,
        masker,
        loss_fn,
        use_cropping: bool = False,
        use_masking: bool = True,
    ):
        """
        Unified forward pass and backward update.

        Views are always treated as a list:
          - Cropping enabled:  SparseCropper produces n_global + n_local views.
          - Cropping disabled: the original full-image batch is the single view.
        Masking (random pixel dropout) is applied independently to each student
        view when enabled; teacher views are never masked.
        Loss is summed over all (student_k, teacher_g) pairs.  Same-index pairs
        (k == g) are skipped only when multiple views exist; with a single view
        the masking pair (k=0, g=0) is the only valid pair and must be kept.

        Args:
            xs:           batched Voxels (from the sparse dataloader, on device)
            cropper:      SparseCropper instance, or None when use_cropping=False
            masker:       SparseVoxelMasker instance (used only when use_masking=True)
            loss_fn:      PixelDINOLoss instance
            use_cropping: enable activity-aware multi-crop augmentation
            use_masking:  enable random pixel dropout on student views

        Returns:
            loss_value:           mean scalar loss across all (student, teacher) pairs
            teacher_entropy, student_entropy, kl, cov_penalty, var_penalty: averaged diagnostics
            student_backbone_out: backbone output of the last student view (for logging)
            teacher_backbone_out: backbone output of the first teacher global view (for logging)
            student_out:          head output of the last student view (for logging)
            teacher_out:          head output of the first teacher global view (for logging)
        """
        # ── 1. Generate views ──────────────────────────────────────────────
        if use_cropping:
            all_views = cropper(xs)
            n_global  = cropper.cfg.n_global
        else:
            all_views = [xs]
            n_global  = 1
        n_crops = len(all_views)

        # ── 2. Teacher: encode global views, frozen, no gradient ───────────
        with torch.no_grad():
            teacher_encoded = [self.encode_teacher(all_views[g]) for g in range(n_global)]

        # ── 3. Student: encode all views (optionally masked), compute loss ─
        total_loss = None
        sum_t_ent = sum_s_ent = sum_kl = sum_cov = sum_var = 0.0
        n_metric = n_pairs = 0

        for k in range(n_crops):
            view_k = masker(all_views[k])[0] if use_masking else all_views[k]
            student_backbone_k, student_out_k = self.encode_student(view_k)

            for g in range(n_global):
                # Skip same-index pairs only when multiple views exist
                # (with one view, the single masking pair must not be skipped)
                if k == g and n_crops > 1:
                    continue

                teacher_backbone_g, teacher_out_g = teacher_encoded[g]

                s_feats, s_bb_feats, t_feats, counts = match_and_gather(
                    student_out_k, student_backbone_k, teacher_out_g,
                )
                loss_kg, t_ent, s_ent, kl, cov, var = loss_fn(
                    s_feats, s_bb_feats, t_feats, counts,
                )

                total_loss = loss_kg if total_loss is None else total_loss + loss_kg
                n_pairs += 1

                if t_ent is not None:
                    sum_t_ent += t_ent
                    sum_s_ent += s_ent
                    sum_kl    += kl
                    n_metric  += 1
                if cov is not None:
                    sum_cov += cov
                if var is not None:
                    sum_var += var

        total_loss = total_loss / n_pairs
        total_loss.backward()

        avg_t_ent = sum_t_ent / n_metric if n_metric > 0 else None
        avg_s_ent = sum_s_ent / n_metric if n_metric > 0 else None
        avg_kl    = sum_kl    / n_metric if n_metric > 0 else None
        avg_cov   = sum_cov   / n_pairs  if sum_cov != 0.0 else None
        avg_var   = sum_var   / n_pairs  if sum_var != 0.0 else None

        # Logging: last student view, first teacher global view
        teacher_backbone_log, teacher_out_log = teacher_encoded[0]
        return (
            total_loss.item(),
            avg_t_ent, avg_s_ent, avg_kl, avg_cov, avg_var,
            student_backbone_k, teacher_backbone_log,
            student_out_k, teacher_out_log,
        )
