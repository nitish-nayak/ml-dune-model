# loader/apa_sparse_meta_dataset.py

import warnings
import h5py
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from warpconvnet.geometry.types.voxels import Voxels

from loader.apa_sparse_dataset import APASparseDataset

# Maps (nu_pdg, nu_ccnc) → class index.
# nu_ccnc == 1 means neutral-current (any flavour), checked first.
#   0 → numuCC  (nu_pdg=14, nu_ccnc=0)
#   1 → nueCC   (nu_pdg=12, nu_ccnc=0)
#   2 → NC      (nu_ccnc=1)
#  -1 → skip    (anything else, e.g. nu_tau CC)
CLASS_NAMES = ["numuCC", "nueCC", "NC"]


def _classify(nu_pdg: int, nu_ccnc: int) -> int:
    if nu_ccnc == 1:
        return 2          # NC — any flavour
    if nu_pdg == 14 and nu_ccnc == 0:
        return 0          # numuCC
    if nu_pdg == 12 and nu_ccnc == 0:
        return 1          # nueCC
    return -1             # skip (e.g. nu_tau CC)


class APASparseMetaDataset(APASparseDataset):
    """
    Extends APASparseDataset to also load per-event truth information from the
    co-located metadata HDF5 file.

    Two return modes, selected at construction time:

    - return_full_metadata=False (default, backward-compatible):
        __getitem__ returns (voxels: Voxels, label: int) where label is the
        class index produced by `_classify` (0=numuCC, 1=nueCC, 2=NC, -1=skip).

    - return_full_metadata=True:
        __getitem__ returns (voxels: Voxels, meta: dict) with keys
            label:      int            class index (as above)
            nu_pdg:     int
            nu_ccnc:    int
            nu_intType: int
            nu_energy:  float
            vertex_xyz: Tensor[3]      float32, detector coords
            event_key:  str            f"{pixeldata_path.name}:{group}" for traceability
        All numeric fields default to -1 / 0.0 when metadata is missing; the
        caller can detect this via label == -1.

    - return_pixel_truth=True (requires return_full_metadata=True):
        Also adds to the meta dict:
            pid_labels: np.ndarray[N_pixels] int32
                Raw PDG code from frame_pid_1st for each reco pixel, aligned to
                the order of pixels in the returned Voxels.  Value 0 means the
                pixel has no pid1 truth hit.

    Metadata file co-location: given
        {dir}/{basename}_pixeldata-anode{N}.h5
    the metadata file is
        {dir}/{basename}_metadata.h5
    and carries the HDF5 structure:
        /{group}/metadata  — numpy structured array (1,) with fields
                             nu_pdg, nu_ccnc, nu_intType, nu_energy,
                             nu_vertex_{x,y,z}
    """

    def __init__(
        self,
        datadir: Union[str, Path],
        apa: int,
        view: str,
        use_cache: bool = True,
        cache_dir: Union[str, Path] = "./data",
        view_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
        frame_name: str = "frame_rebinned_reco",
        return_full_metadata: bool = False,
        return_pixel_truth: bool = False,
    ):
        super().__init__(
            datadir=datadir,
            apa=apa,
            view=view,
            use_cache=use_cache,
            cache_dir=cache_dir,
            view_ranges=view_ranges,
            frame_name=frame_name,
        )
        self.return_full_metadata = return_full_metadata
        self.return_pixel_truth = return_pixel_truth
        self._warned_missing: set = set()

    def __getitem__(self, idx: int):
        voxels = super().__getitem__(idx)

        s = self.samples[idx]
        if self.return_full_metadata:
            meta = self._read_full_metadata(s.path, s.group)
            if self.return_pixel_truth:
                meta["pid_labels"] = self._read_pixel_truth(s.path, s.group)
            return voxels, meta

        label = self._read_label(s.path, s.group)
        return voxels, label

    # ------------------------------------------------------------------
    # Metadata path helpers
    # ------------------------------------------------------------------

    def _metadata_path(self, pixeldata_path: Path) -> Path:
        suffix = f"_pixeldata-anode{self.apa}.h5"
        if not pixeldata_path.name.endswith(suffix):
            # Older naming: "..._anode3.h5" without "_pixeldata" prefix.
            basename = pixeldata_path.stem   # drop .h5
        else:
            basename = pixeldata_path.name[: -len(suffix)]
        return pixeldata_path.parent / f"{basename}_metadata.h5"

    def _warn_once(self, metadata_path: Path, msg: str) -> None:
        key = str(metadata_path)
        if key not in self._warned_missing:
            warnings.warn(msg)
            self._warned_missing.add(key)

    # ------------------------------------------------------------------
    # Label-only reader (backward compatible)
    # ------------------------------------------------------------------

    def _read_label(self, pixeldata_path: Path, group: str) -> int:
        metadata_path = self._metadata_path(pixeldata_path)

        if not metadata_path.exists():
            self._warn_once(metadata_path, f"Metadata file not found: {metadata_path}")
            return -1

        try:
            with h5py.File(metadata_path, "r") as f:
                row = f[group]["metadata"][0]
                nu_pdg  = int(row["nu_pdg"])
                nu_ccnc = int(row["nu_ccnc"])
            return _classify(nu_pdg, nu_ccnc)
        except Exception as e:
            self._warn_once(metadata_path, f"Could not read metadata from {metadata_path}[{group}]: {e}")
            return -1

    # ------------------------------------------------------------------
    # Full metadata reader
    # ------------------------------------------------------------------

    def _unknown_metadata(self, event_key: str) -> dict:
        """Sentinel metadata dict used when the metadata file is missing/unreadable."""
        return {
            "label":      -1,
            "nu_pdg":     0,
            "nu_ccnc":    -1,
            "nu_intType": -1,
            "nu_energy":  0.0,
            "vertex_xyz": torch.zeros(3, dtype=torch.float32),
            "event_key":  event_key,
        }

    def _read_full_metadata(self, pixeldata_path: Path, group: str) -> dict:
        event_key = f"{pixeldata_path.name}:{group}"
        metadata_path = self._metadata_path(pixeldata_path)

        if not metadata_path.exists():
            self._warn_once(metadata_path, f"Metadata file not found: {metadata_path}")
            return self._unknown_metadata(event_key)

        try:
            with h5py.File(metadata_path, "r") as f:
                row = f[group]["metadata"][0]
                nu_pdg     = int(row["nu_pdg"])
                nu_ccnc    = int(row["nu_ccnc"])
                nu_intType = int(row["nu_intType"])
                nu_energy  = float(row["nu_energy"])
                vx = float(row["nu_vertex_x"])
                vy = float(row["nu_vertex_y"])
                vz = float(row["nu_vertex_z"])
        except Exception as e:
            self._warn_once(metadata_path, f"Could not read metadata from {metadata_path}[{group}]: {e}")
            return self._unknown_metadata(event_key)

        return {
            "label":      _classify(nu_pdg, nu_ccnc),
            "nu_pdg":     nu_pdg,
            "nu_ccnc":    nu_ccnc,
            "nu_intType": nu_intType,
            "nu_energy":  nu_energy,
            "vertex_xyz": torch.tensor([vx, vy, vz], dtype=torch.float32),
            "event_key":  event_key,
        }

    # ------------------------------------------------------------------
    # Pixel-level truth reader
    # ------------------------------------------------------------------

    def _read_pixel_truth(self, pixeldata_path: Path, group: str) -> np.ndarray:
        """
        Read frame_pid_1st from the pixeldata file and return a per-reco-pixel
        array of raw PDG codes, aligned to the pixel order produced by
        APASparseDataset.__getitem__ for the same (path, group).

        Pixels with no pid1 hit carry value 0.
        """
        try:
            with h5py.File(pixeldata_path, "r") as f:
                g = f[group]
                reco_coords = g[self.frame_name]["coords"][()]   # (N, 2) int32

                if "frame_pid_1st" not in g:
                    mask = ((reco_coords[:, 0] >= self.ch_start) &
                            (reco_coords[:, 0] < self.ch_end))
                    return np.zeros(int(mask.sum()), dtype=np.int32)

                pid1_coords = g["frame_pid_1st"]["coords"][()]   # (M, 2) int32
                pid1_feats  = g["frame_pid_1st"]["features"][()]  # (M,) float32
        except Exception as e:
            self._warn_once(
                pixeldata_path,
                f"Could not read pixel truth from {pixeldata_path}[{group}]: {e}",
            )
            return np.zeros(0, dtype=np.int32)

        # Filter to this view's channel range (same logic as APASparseDataset)
        mask_reco = ((reco_coords[:, 0] >= self.ch_start) &
                     (reco_coords[:, 0] < self.ch_end))
        mask_pid1 = ((pid1_coords[:, 0] >= self.ch_start) &
                     (pid1_coords[:, 0] < self.ch_end))

        reco_view       = reco_coords[mask_reco]
        pid1_view_coords = pid1_coords[mask_pid1]
        pid1_view_feats  = pid1_feats[mask_pid1]

        # Build lookup: (rebased_ch, tick) -> PDG code
        pid1_lookup = {
            (int(c[0]) - self.ch_start, int(c[1])): int(v)
            for c, v in zip(pid1_view_coords, pid1_view_feats)
        }

        # Align to reco pixel order (APASparseDataset subtracts ch_start from coords)
        return np.array(
            [pid1_lookup.get((int(c[0]) - self.ch_start, int(c[1])), 0)
             for c in reco_view],
            dtype=np.int32,
        )
