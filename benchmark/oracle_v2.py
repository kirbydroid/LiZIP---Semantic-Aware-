''' Oracle experiement to check if compressing each semnatic group seperately
    is smaller than compressing the whole frame? 
    Baseline: THe whole frame, sorted and encoded as one unit.
    Experiment: [Tiered]: points split into teirs, each teri sorted and encoded
    seperately. Then we add the cost of telling the decoder which tier each
    point belongs to. 
'''

import json 
import zlib
import os
import sys
import time
import argparse 
import time

# for computing
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# importing all the functions and constants needed for this exp from the previous files.
from src.python.model import PointPredictorMLP
from src.python.encoder import encode_segment, RESID_SCALE, BLOCK_SIZE
from src.python.voxel_sort import voxel_quantize_and_sort
from src.utils.data_loader import load_nuscenes_lidar_all

DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "nuscenes")
VERSION = "v1.0-mini"
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "grid_search", "mlp_c5_h512.pth")
CONTEXT_SIZE = 5
HIDDEN_DIM = 512

HEADER_BYTES = 24 # already exisiting .lizip header
TIER_HEADER_BYTES = 8 # per tier = num_points + num_blocks

TIERS = ["ground", "structure", "vehicle", "small", "other"]

# lidarseg class labels into different tiers -- tier index [lookup table of sorts]
TIER_OF_CLASS = np.zeros(32, dtype=np.int64)
TIER_OF_CLASS[[0 , 31]] = 4 # Other / noise
TIER_OF_CLASS[1:13] = 3 # ANimals, pedestrains, wheelchair, barrier, cone, similar
TIER_OF_CLASS[14:24] = 2 # Vehicles
TIER_OF_CLASS[24:28] = 0 # Ground, sidewalk, similar
TIER_OF_CLASS[28:31] = 1 # BUildings, static, vegetation
TIER_OF_CLASS[13] = 1 # Bicycle rack

# Giving precision
PRECISE = [2, 3] # giving more importance to vehicles and small 
GROUND_PCT = 5 # height rule: ground height = percentile of z
GROUND_MARGIN = 0.2 # meters above the ground that still count as ground

# scale / steps
S_001MM, S_01MM, S_1MM, S_1CM = 100000, 10000, 1000, 100

# Dihhhferent experiments we will be checking our results with 

CONFIGS = [
    # if we change the precision but no use of semnatic info
    ("uni 0.01mm", "uniform", S_001MM, None),
    ("uni 0.1mm", "uniform", S_01MM, None),
    ("uni 1mm", "uniform", S_1MM, None),
    ("uni 1cm", "uniform", S_1CM, None),

    # if we split frame into semantic classes without changing precision (recheck)
    ("class sanity (all 0.01mm)", "class", S_001MM, S_001MM),
    ("class 0.01mm/1mm", "class", S_001MM, S_1MM),
    ("class 0.01mm/1cm", "class", S_001MM, S_1CM),
    ("class 1mm/1cm", "class", S_1MM, S_1CM),

    # works with consecutive blocks, if it has even one protected point, fine precision to the whole block
    ("block 0.01mm/1mm", "block", S_001MM, S_1MM),
    ("block 0.01mm/1cm", "block", S_001MM, S_1CM),
    ("block 1mm/1cm", "block", S_1MM, S_1CM),

    # using height based heuristics instead of semantic classes.
    ("rule 0.01mm/1mm", "rule", S_001MM, S_1MM),
    ("rule 0.01mm/1cm", "rule", S_001MM, S_1CM),
    ("rule 1mm/1cm", "rule", S_1MM, S_1CM),
]


def list_frames():
    ''' returns (lidar_path, label_path, scene_token) for every labeled keyframe '''

    meta = os.path.join(DATA_ROOT, VERSION)

    # path to the LiDAR point clound
    with open(os.path.join(meta, "sample_data.json")) as f:
        sample_data = {d["token"]: d for d in json.load(f)}

    # which scene the retrieved frame belongs to
    with open(os.path.join(meta, "sample.json")) as f:
        scene = {
            s["token"]: s["scene_token"]
            for s in json.load(f)
        }

    # path to the semantic labels
    with open(os.path.join(meta, "lidarseg.json")) as f:
        records = json.load(f)

    frames = []
    for r in records:
        d = sample_data[r["sample_data_token"]]
        frames.append((os.path.join(DATA_ROOT, d["filename"]),
                       os.path.join(DATA_ROOT, r["filename"]),
                       scene[d["sample_token"]]))
    return frames


def encode(sorted_pts, scale, model):
    heads, residuals, _, _ = encode_segment(sorted_pts.astype(np.float32), scale, model, CONTEXT_SIZE)
    return len(zlib.compress(heads + residuals, level = 9))


def total_bytes(groups, model):
    groups = [g for g in groups if len(g[0]) > 0]
    total = HEADER_BYTES

    for pts, scale in groups:
        total += encode(pts, scale, model)

        if len(groups) > 1:
            total += TIER_HEADER_BYTES

    return total


def error_stats(points, scale_of_point, tiers): 
    ''' Measuring the error introduced by quantization,
    not the decoder's reconstruction error'''
    p = points.astype(np.float32)
    s = scale_of_point.astype(np.float32)[:, None]

    recon = np.round(p * s) / s # multiply the quantization scale, round to integer and divide by scale 
    err = np.abs(recon.astype(np.float64) - p)
    # calculating euclidian error 3d
    d3 = np.linalg.norm(err, axis = 1)

    maxaxis = np.zeros((len(TIERS), 3))
    maxd3 = np.zeros(len(TIERS))

    for t in range(len(TIERS)):
        m = tiers == t

        if m.any():
            maxaxis[t] = err[m].max(axis = 0)
            maxd3[t] = d3[m].max()    

    return maxaxis, maxd3 


def run_config(mode, fine, coarse, points, tiers, model):
    '''encode one frame in one run_config run
    returns (point_bytes, label_bytes, per point scale in original order)'''

    n = len(points)
    sorted_pts, idx = voxel_quantize_and_sort(points)
    label_bytes = 0

    if mode == "uniform":
        groups = [(sorted_pts, fine)]
        scale_pt = np.full(n, fine, dtype = np.float64)

    elif mode == "class":
        tier_scale = np.array([fine if t in PRECISE else coarse for t in range(len(TIERS))],
                              dtype = np.float64)
        groups = []
        for t in range(len(TIERS)):
            if (tiers == t).any():
                tier_sorted, _ = voxel_quantize_and_sort(points[tiers == t])
            else:
                tier_sorted = points[:0]
            groups.append((tier_sorted, tier_scale[t]))

        scale_pt = tier_scale[tiers]
        label_bytes = len(zlib.compress(tiers[idx].astype(np.uint8).tobytes(), level = 9))

    else:
        if mode == "block":
            imp = np.isin(tiers[idx], PRECISE)
        else:
            groundz = np.percentile(points[:, 2], GROUND_PCT)
            imp = sorted_pts[:, 2] >= groundz + GROUND_MARGIN

        blocks_num = (n + BLOCK_SIZE - 1) // BLOCK_SIZE

        padded = np.zeros(blocks_num * BLOCK_SIZE, dtype = bool)
        padded[:n] = imp
        fine_block = padded.reshape(blocks_num, BLOCK_SIZE).any(axis = 1)
        fine_points = np.repeat(fine_block, BLOCK_SIZE)[:n]
        # splitting sorted points into two groups
        groups = [(sorted_pts[fine_points], fine), (sorted_pts[~fine_points], coarse)]

        scale_sorted = np.where(fine_points, fine, coarse).astype(np.float64)
        scale_pt = np.empty(n, dtype = np.float64)
        scale_pt[idx] = scale_sorted
        label_bytes = len(np.packbits(fine_block))

    return total_bytes(groups, model), label_bytes, scale_pt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type = int, default = None, help = "ONLY USE FIRST N FRAMES")
    parser.add_argument("--mode", default = "uniform,class,block,rule", help = "COMMA LIST FROM: uniform, class, block, rule")
    parser.add_argument("--out", default = os.path.join(PROJECT_ROOT, "benchmark", "oracle_v2_results.json"))
    args = parser.parse_args()
    # rng = np.random.default_rng(0) if args.control == "random" else None

    modes = [m.strip() for m in args.mode.split(",")]
    configs = [c for c in CONFIGS if c[1] in modes]

    frames = list_frames()[:args.limit]
    scenes = {f[2] for f in frames}

    model = PointPredictorMLP(context_size = CONTEXT_SIZE, hidden_dim = HIDDEN_DIM)
    model.load_state_dict(torch.load(MODEL_PATH, map_location = "cpu"))
    model.eval()

    res = {c[0]: dict(point_bytes = 0, label_bytes = 0, time = 0.0, fine_pts = 0,
                      max_axis = np.zeros((len(TIERS), 3)), max_d3 = np.zeros(len(TIERS)))
           for c in configs}
    total_points = 0

    # loading each frame and its labels
    for i, (lidar_path, label_path, _) in enumerate(frames):
        points = load_nuscenes_lidar_all(lidar_path)[:, :3]
        labels = np.fromfile(label_path, dtype = np.uint8)
        assert len(labels) == len(points), "labels and points do not lineup"
        tiers = TIER_OF_CLASS[labels]
        total_points += len(points)

        for name, mode, fine, coarse in configs:
            t0 = time.perf_counter()

            pb, lb, scale_pt = run_config(mode, fine, coarse, points, tiers, model)
            r = res[name]

            r["time"] += time.perf_counter() - t0
            r["point_bytes"] += pb
            r["label_bytes"] += lb
            r["fine_pts"] += int((scale_pt == fine).sum())
            a, d = error_stats(points, scale_pt, tiers)

            r["max_axis"] = np.maximum(r["max_axis"], a)
            r["max_d3"] = np.maximum(r["max_d3"], d)

        print(f"frame {i + 1}/{len(frames)} done", flush = True)

    nf = len(frames)
    print(f"\n\nframes: {nf}  scenes: {len(scenes)}  points: {total_points}  ({VERSION})\n")

    def savings(name, ref):
        if ref not in res:
            return " n/a "
        a = res[name]["point_bytes"] + res[name]["label_bytes"]
        b = res[ref]["point_bytes"] + res[ref]["label_bytes"]

        return f"{1 - a / b:+7.1%}"

    print(f"{'config':<27}{'KB/frame':>9}{'bits/pt':>8}{'lbl B/fr':>9}"
          f"{'vs 1mm':>9}{'vs 0.01mm':>11}{'fine%':>7}{'ms/frame':>10}")
    for name, *_ in configs:
        r = res[name]
        tot = r["point_bytes"] + r["label_bytes"]
        print(f"{name:<27}{tot / nf / 1000:>9.1f}{8 * tot / total_points:>8.2f}"
              f"{r['label_bytes'] / nf:>9.0f}"
              f"{savings(name, 'uni 1mm'):>9}{savings(name, 'uni 0.01mm'):>11}"
              f"{100 * r['fine_pts'] / total_points:>7.0f}"
              f"{1000 * r['time'] / nf:>10.0f}")

    print("\nmax 3D error in mm per tier: " + "  ".join(f"{t:>9}" for t in TIERS))
    for name, *_ in configs:
        print(f"{name:<27}" + "  ".join(f"{1000 * v:>9.3f}" for v in res[name]["max_d3"]))

    out = {"frames": nf, "scenes": len(scenes), "points": int(total_points),
           "tiers": TIERS,
           "results": {k: dict(point_bytes=int(v["point_bytes"]),
                               label_bytes=int(v["label_bytes"]),
                               sec_per_frame=v["time"] / nf,
                               fine_point_fraction=v["fine_pts"] / total_points,
                               max_err_axis_m=v["max_axis"].tolist(),
                               max_err_3d_m=v["max_d3"].tolist())
                       for k, v in res.items()}}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
