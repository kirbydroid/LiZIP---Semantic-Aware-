import argparse
import json
import os
import sys
import zlib

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.python.model import PointPredictorMLP
from src.python.encoder import encode_segment, RESID_SCALE, BLOCK_SIZE 
from src.python.voxel_sort import voxel_quantize_and_sort
from src.utils.data_loader import load_nuscenes_lidar_all


DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "nuscenes")
VERSION = "v1.0-mini"
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "grid_search", "mlp_c5_h512.pth")
CONTEXT_SIZE = 5
HIDDEN_DIM = 512

HEADER_BYTES = 24      # existing .lizip header
TIER_HEADER_BYTES = 8  # per tier: num_points + num_blocks

TIERS = ["ground", "structure", "vehicle", "small", "other"]

# lidarseg class index (0-31) -> tier index
TIER_OF_CLASS = np.zeros(32, dtype=np.int64)
TIER_OF_CLASS[24:28] = 0  # flat.*
TIER_OF_CLASS[28:31] = 1  # static.manmade / other / vegetation
TIER_OF_CLASS[13] = 1     # static_object.bicycle_rack
TIER_OF_CLASS[14:24] = 2  # vehicle.*
TIER_OF_CLASS[1:13] = 3   # animal, human.*, movable_object.*
TIER_OF_CLASS[[0, 31]] = 4  # noise, vehicle.ego


def list_frames():
    """Returns (lidar_path, label_path) for every labelled keyframe."""
    meta = os.path.join(DATA_ROOT, VERSION)
    with open(os.path.join(meta, "sample_data.json")) as f:
        lidar_file = {d["token"]: d["filename"] for d in json.load(f)}
    with open(os.path.join(meta, "lidarseg.json")) as f:
        records = json.load(f)
    return [(os.path.join(DATA_ROOT, lidar_file[r["sample_data_token"]]),
             os.path.join(DATA_ROOT, r["filename"])) for r in records]


def compressed_size(points_xyz, model):
    """Bytes needed for one group of points, sorted and encoded on its own."""
    sorted_points, _ = voxel_quantize_and_sort(points_xyz)
    heads, residuals, _, _ = encode_segment(
        sorted_points.astype(np.float32), RESID_SCALE, model, CONTEXT_SIZE)
    return len(zlib.compress(heads + residuals, level=9))

def label_bytes(tiers, points_xyz):
        _,idx = voxel_quantize_and_sort(points_xyz)
        return len(zlib.compress(tiers[idx].astype(np.uint8).tobytes(), level=9))

def block_tiers(tiers, points_xyz, rng = None):
        _,idx = voxel_quantize_and_sort(points_xyz)
        tiers_sorted = tiers[idx]
        n = len(tiers_sorted)
        n_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
        padded = np.full(n_blocks * BLOCK_SIZE, -1, dtype = np.int64)
        padded[:n] = tiers_sorted
        counts = np.stack([(blocks == t).sum(1) for t in range(len(TIERS))], axis = 1)
        majority = counts.argmax(1)

        if rng is not None:
            majority = rng.permutation(majority)

        group = np.empty(n, dtype = np.int64)
        group[idx] = np.repeat(majority, BLOCK_SIZE)[:n]
        return group, majority.astype(np.uint8)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only use the first N frames")
    parser.add_argument("--scheme", choices = ["point", "block", default = "point"])
    parser.add_argument("--control", choices = ["none", "random", default = "none"])
    args = parser.parse_args()
    rng = np.random.default_rng(0) if args.control == "random" else None

    model = PointPredictorMLP(context_size=CONTEXT_SIZE, hidden_dim=HIDDEN_DIM)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    frames = list_frames()[:args.limit]

    total_points = 0
    baseline_bytes = 0
    oracle_bytes = 0
    tier_points = np.zeros(len(TIERS), dtype=np.int64)
    tier_bytes = np.zeros(len(TIERS), dtype=np.int64)

    for i, (lidar_path, label_path) in enumerate(frames):
        points_xyz = load_nuscenes_lidar_all(lidar_path)[:, :3]
        labels = np.fromfile(label_path, dtype=np.uint8)
        assert len(labels) == len(points_xyz), "labels and points do not line up"
        tiers = TIER_OF_CLASS[labels]

        #for block size

        if args.scheme == "block":
            group, sent = block_tiers(tiers, points_xyz, rng)
        label_cost = len(zlib.compress(sent.tobytes(), level=9))
        else:
            group = rng.permutation(tiers) if rng is not None else tiers
            label_cost = label_bytes(group, points_xyz)

        total_points += len(points_xyz)
        baseline_bytes += HEADER_BYTES + compressed_size(points_xyz, model)

        points_only_bytes += HEADER_BYTES
        for t in range(len(TIERS)):
            g_xyz = points_xyz[group == t]
            if len(g_xyz) == 0:
                continue
            size = TIER_HEADER_BYTES + compressed_size(g_xyz, model)
            tier_points[t] += len(g_xyz)
            tier_bytes[t] += size
            points_only_bytes += size   
        oracle_bytes += label_cost

        total_points += len(points_xyz)
        baseline_bytes += HEADER_BYTES + compressed_size(points_xyz, model)

        oracle_bytes += HEADER_BYTES
        for t in range(len(TIERS)):
            tier_xyz = points_xyz[tiers == t]
            if len(tier_xyz) == 0:
                continue
            size = TIER_HEADER_BYTES + compressed_size(tier_xyz, model)
            tier_points[t] += len(tier_xyz)
            tier_bytes[t] += size
            oracle_bytes += size
            #oracle_bytes += label_bytes(tiers, points_xyz)

        print(f"\rframe {i + 1}/{len(frames)}", end="", flush=True)

    print(f"\n\n{'tier':<10} {'points':>10} {'share':>7} {'bytes':>10} {'bits/pt':>8}")
    for t, name in enumerate(TIERS):
        if tier_points[t] == 0:
            continue
        print(f"{name:<10} {tier_points[t]:>10} {tier_points[t] / total_points:>7.1%} "
              f"{tier_bytes[t]:>10} {8 * tier_bytes[t] / tier_points[t]:>8.2f}")

    baseline_bpp = 8 * baseline_bytes / total_points
    oracle_bpp = 8 * oracle_bytes / total_points
    print(f"\nframes: {len(frames)}  points: {total_points}")
    '''print(f"baseline: {baseline_bytes} bytes  {baseline_bpp:.2f} bits/pt")
    print(f"oracle:   {oracle_bytes} bytes  {oracle_bpp:.2f} bits/pt")
    print(f"change:   {oracle_bytes / baseline_bytes - 1:+.2%}")'''

    total = points_only_bytes + oracle_bytes
    print(f"labels:      {oracle_bytes} bytes  ({8 * oracle_bytes / total_points:.2f} bits/pt)")
    print(f"free labels: {points_only_bytes / baseline_bytes - 1:+.2%}")
    print(f"with labels: {total / baseline_bytes - 1:+.2%}")


if __name__ == "__main__":
    main()