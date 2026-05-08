# loader/apa_sparse_dataset.py

import hashlib
import h5py
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset

from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.geometry.coords.integer import IntCoords
from warpconvnet.geometry.features.cat import CatFeatures

from loader.apa_dataset import APASampleIndex

# Default channel ranges for each wire-plane view.
# Channels 0-799 → U plane, 800-1599 → V plane, 1600-2649 → W plane.
DEFAULT_VIEW_RANGES: Dict[str, Tuple[int, int]] = {
    "U": (0, 800),
    "V": (800, 1600),
    "W": (1600, 2650),
}


class APASparseDataset(Dataset):
    """
    Dataset that:
      - scans recursively for sparse HDF5 files matching *anode<APA>.h5
        (matches both old "…anode3.h5" and new "…pixeldata-anode3.h5" naming)
      - treats each /<group>/<frame_name> as an independent sparse sample
      - selects only one view (U, V, or W) by filtering on the channel coordinate
      - returns a single-item Voxels object per sample
      - caches the index to disk; cache filename encodes rootdir so different
        datasets never share the same cache file

    Expected sparse HDF5 structure per group:
        /<group>/<frame_name>/coords    (N, 2) int32  — (channel, tick)
        /<group>/<frame_name>/features  (N,)   float32

    Structured-folder layout (new datasets):
        root_dir/{run}/{subrun}/{event}/out_{basename}/
            {basename}_pixeldata-anode{N}.h5
            {basename}_metadata.h5
    The recursive glob handles arbitrary nesting depth automatically.
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
    ):
        """
        Args:
            datadir:     Root directory to scan recursively for HDF5 files.
            apa:         APA number (matched against the filename suffix).
            view:        Wire-plane view to load — one of the keys in view_ranges
                         (default: "U", "V", or "W").
            use_cache:   Load/save the file index from/to a .pt cache file.
            cache_dir:   Directory for cache files.
            view_ranges: Mapping from view name → (ch_start, ch_end) channel range.
                         Defaults to DEFAULT_VIEW_RANGES.  Override this for
                         detector geometries with different channel assignments.
            frame_name:  HDF5 group key that holds the sparse frame data
                         (coords + features).  Defaults to "frame_rebinned_reco".
                         Other options in new files: "frame_pid_1st",
                         "frame_trackid_1st", etc.
        """
        self.datadir   = Path(datadir)
        self.apa       = int(apa)
        self.frame_name = frame_name

        self.view_ranges: Dict[str, Tuple[int, int]] = (
            view_ranges if view_ranges is not None else DEFAULT_VIEW_RANGES
        )

        self.view = view.upper()
        if self.view not in self.view_ranges:
            raise ValueError(
                f"view must be one of {list(self.view_ranges)}, got {view!r}"
            )

        self.ch_start, self.ch_end = self.view_ranges[self.view]

        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Include a short hash of the resolved datadir so that two datasets
        # with the same APA/view but different paths never share a cache file.
        root_hash = hashlib.md5(
            str(self.datadir.resolve()).encode()
        ).hexdigest()[:8]
        self.cache_file = (
            self.cache_dir
            / f"APASparseDataset_APA{self.apa}_view{self.view}_{root_hash}_cache.pt"
        )

        self.samples: List[APASampleIndex] = self._scan()
        if not self.samples:
            raise RuntimeError(
                f"No sparse samples found under {self.datadir} for APA {self.apa}"
            )

    # -------------------------
    # cache helpers
    # -------------------------

    def _save_index_pt(self, cache_file: Path, samples: List[APASampleIndex]):
        data = [(str(s.path), s.group) for s in samples]
        torch.save(data, cache_file)

    def _load_index_pt(self, cache_file: Path) -> List[APASampleIndex]:
        data = torch.load(cache_file, map_location="cpu")
        return [APASampleIndex(path=Path(p), group=g) for p, g in data]

    # -------------------------
    # scanning
    # -------------------------

    def _scan(self) -> List[APASampleIndex]:
        samples: List[APASampleIndex] = []

        if self.use_cache and self.cache_file.exists():
            print(f"Loading dataset index from cache: {self.cache_file}")
            return self._load_index_pt(self.cache_file)

        print(f"Cache does not exist: {self.cache_file} -- generating new one!")
        pattern = f"*anode{self.apa}.h5"

        for fp in self.datadir.rglob(pattern):
            if not fp.is_file():
                continue

            try:
                with h5py.File(fp, "r") as f:
                    for group in f.keys():
                        grp = f[group]
                        if not isinstance(grp, h5py.Group):
                            continue
                        if self.frame_name not in grp:
                            continue
                        frame_entry = grp[self.frame_name]
                        # Accept only sparse format: subgroup with coords and features
                        if (
                            isinstance(frame_entry, h5py.Group)
                            and "coords" in frame_entry
                            and "features" in frame_entry
                        ):
                            samples.append(APASampleIndex(path=fp, group=group))
            except OSError as e:
                print(f"Warning: could not open {fp}: {e}")

        # stable ordering
        samples.sort(key=lambda s: (str(s.path), int(s.group)))

        if self.use_cache:
            print(f"Saving dataset index to cache: {self.cache_file}")
            self._save_index_pt(self.cache_file, samples)

        return samples

    # -------------------------
    # Dataset API
    # -------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Voxels:
        s = self.samples[idx]

        with h5py.File(s.path, "r") as f:
            frame = f[s.group][self.frame_name]
            coords = torch.from_numpy(frame["coords"][()]).to(torch.int32)   # (N, 2)
            feats  = torch.from_numpy(frame["features"][()]).to(torch.float32)  # (N,)

        # select view: keep only active pixels in [ch_start, ch_end)
        mask = (coords[:, 0] >= self.ch_start) & (coords[:, 0] < self.ch_end)
        coords = coords[mask].clone()
        feats  = feats[mask]

        # rebase channel coordinate to 0 for this view
        coords[:, 0] -= self.ch_start

        # features must be (N, C) for Voxels
        feats = feats.unsqueeze(1)

        # single-item batch offsets
        n = coords.shape[0]
        offsets = torch.tensor([0, n], dtype=torch.int64)

        return Voxels(
            batched_coordinates=IntCoords(coords, offsets=offsets),
            batched_features=CatFeatures(feats, offsets=offsets),
            offsets=offsets,
        )
