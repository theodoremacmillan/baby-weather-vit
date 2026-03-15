import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import gcsfs
import numpy as np
import pandas as pd

NORMALIZATION_STATS_PATH = Path(__file__).parent / "normalization_stats.json"


class ERA5Dataset(Dataset):
    def __init__(self, variables, times, store, normalizations):
        fs = gcsfs.GCSFileSystem(token="anon")
        # open zarr once — this is lazy, no data loaded yet
        self.ds = xr.open_zarr(fs.get_mapper(store), consolidated=True)
        self.ds = self.ds[variables]
        self.times = times
        self.variables = variables
        self.normalizations = normalizations

    def __len__(self):
        return len(self.times) - 1
    
    def __getitem__(self, idx):
        x = self.ds.sel(time=self.times[idx]).compute()
        y = self.ds.sel(time=self.times[idx + 1]).compute()

        x = torch.tensor(
            np.stack([(x[v].values - self.normalizations[v][0]) / self.normalizations[v][1] for v in self.variables]),
            dtype=torch.float32
        )
        y = torch.tensor(
            np.stack([(y[v].values - self.normalizations[v][0]) / self.normalizations[v][1] for v in self.variables]),
            dtype=torch.float32
        )

        return x, y

def make_times(start_date, end_date, skip):
    return pd.date_range(start=start_date, end=end_date, freq=skip).strftime("%Y-%m-%dT%H:%M").tolist()

def make_dataloaders(
        variables,
        start_date,
        end_date,
        store,
        train_frac=0.8,
        batch_size=32,
        num_workers=0
):
    times = make_times(start_date, end_date, "6h")
    split = int(len(times)*train_frac)

    normalizations = get_normalization_statistics(variables)

    train_times = times[:split]
    train_dataset = ERA5Dataset(variables, train_times, store, normalizations)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_times = times[split:]
    test_dataset = ERA5Dataset(variables, test_times, store, normalizations)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

def load_normalization_stats():
    """Load precomputed normalization stats from JSON file."""
    if NORMALIZATION_STATS_PATH.exists():
        with open(NORMALIZATION_STATS_PATH) as f:
            stats = json.load(f)
        return {v: (s["mean"], s["std"]) for v, s in stats.items()}
    return None


def save_normalization_stats(normalizations):
    """Save normalization stats to JSON file."""
    stats = {v: {"mean": m, "std": s} for v, (m, s) in normalizations.items()}
    with open(NORMALIZATION_STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)


def compute_normalization_statistics(
        variables,
        store="gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_64x32_equiangular_conservative.zarr"
):
    """Compute normalization stats from climatology data (slow, requires network)."""
    fs = gcsfs.GCSFileSystem(token="anon")
    ds = xr.open_zarr(fs.get_mapper(store), consolidated=True)

    means = ds[variables].mean(
        dim=['hour', 'dayofyear', 'longitude', 'latitude']
    ).compute()

    stds = ds[variables].std(
        dim=['hour', 'dayofyear', 'longitude', 'latitude']
    ).compute()

    return {
        v: (float(means[v].values), float(stds[v].values))
        for v in variables
    }


def get_normalization_statistics(variables, store=None):
    """
    Get normalization statistics for variables.

    Loads from cached JSON file if available, otherwise computes from
    climatology data and caches the result.
    """
    cached_stats = load_normalization_stats()

    # Check if we have all required variables cached
    if cached_stats and all(v in cached_stats for v in variables):
        return {v: cached_stats[v] for v in variables}

    # Compute missing stats
    if store is None:
        store = "gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_64x32_equiangular_conservative.zarr"

    computed = compute_normalization_statistics(variables, store)

    # Merge with existing cached stats and save
    merged = cached_stats or {}
    merged.update(computed)
    save_normalization_stats(merged)

    return {v: computed[v] for v in variables}