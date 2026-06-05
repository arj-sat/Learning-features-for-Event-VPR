"""
batch_runner.py
===============
Batches multiple runs by importing worker functions from process.py directly.
process.py is NEVER modified — only LOG_FILE (already a global there) is set.

Place this file in the same folder as process.py and sampler.py.

USAGE:
    python batch_runner.py              # run all jobs
    python batch_runner.py --jobs 0 3  # run only jobs 0 and 3
    python batch_runner.py --dry-run   # preview without running
    python batch_runner.py --resume    # skip jobs that already have output CSVs
"""

import argparse
import os
import time
import traceback
from pathlib import Path
from datetime import datetime

# Import worker functions from process.py — file is untouched
import process as _proc
from sampler import FilterDataRecursive


# =============================================================================
# DATASET PATHS  <-- only thing to update when you get the morning path
# =============================================================================

INPUT_FOLDERS = {
    "sunset1": r"C:\Arjun\Thesis\data\20200421_170039-sunset1\filtered chunks",
    "sunset2": r"C:\Arjun\Thesis\data\20200422_172431-sunset2\filtered",
    "night":   r"C:\Arjun\Thesis\data\20200427_181204-night\filtered",
    "daytime": r"C:\Arjun\Thesis\data\20200424_daytime\filter",
    "morning": r"C:\Arjun\Thesis\data\2020-04-28-09-14-11-morning\filter",  # <-- update
}

RESULTS_DIR = r"C:\Arjun\Thesis\data\Results"

# =============================================================================
# JOB LIST  —  (name, dataset, break_point, sampling_threshold, tau_ms, norm)
# =============================================================================

JOBS = [
    # name                                   dataset    bp     thr    tau   norm
    #("sunset1_bp45_thr010_tau30_nonorm",   "sunset1", 0.15,  0.100, 30.0, None),  completed
    #("sunset2_bp45_thr010_tau30_nonorm",   "sunset2", 0.15,  0.100, 30.0, None),  completed
    #("sunset1_bp45_thr005_tau30_nonorm",   "sunset1", 0.25,  0.005, 30.0, None),
    ("sunset1_bp1_thr005_tau30_nonorm",   "sunset1", 1.01,  0.005, 30.0, None),
    
    #("sunset2_bp45_thr005_tau30_nonorm",   "sunset2", 0.3,  0.005, 30.0, None),
    #("night_bp45_thr005_tau30_nonorm",     "night",   0.25,  0.005, 30.0, None),
    #("morning_bp45_thr005_tau30_nonorm",   "morning", 0.25,  0.005, 30.0, None),
    #("daytime_bp45_thr005_tau30_nonorm",   "daytime", 0.45,  0.005, 30.0, None),
    #("sunset1_bp25_thr005_tau30_norm200",  "sunset1", 0.25,  0.005, 30.0,  200),
    #("sunset2_bp25_thr005_tau30_norm200",  "sunset2", 0.25,  0.005, 30.0,  200),
    #("sunset1_bp20_thr005_tau60_nonorm",   "sunset1", 0.20,  0.005, 60.0, None),
    #("sunset2_bp20_thr005_tau60_nonorm",   "sunset2", 0.20,  0.005, 60.0, None),
]

# Fixed constants — same for every job
IMAGE_SIZE     = (346, 260)
FILTER_SIZE    = 7
FIXED_SAMPLING = True
SEED_STR       = "fixed_subsampling"


# =============================================================================
# RUNNER
# =============================================================================

def run_job(job_tuple, job_idx, total):
    name, dataset, break_point, threshold, tau_ms, norm_len = job_tuple

    input_folder  = Path(INPUT_FOLDERS[dataset])
    #output_folder = Path(DATASET_DIRS[dataset]) / name
    output_folder = Path(RESULTS_DIR) / name
    output_folder.mkdir(parents=True, exist_ok=True)
    log_file = str(output_folder / "process_log.txt")

    # process.py declares LOG_FILE as a global — redirect it to this job's file
    _proc.LOG_FILE = log_file

    header = "\n".join([
        "=" * 60,
        f"Job [{job_idx+1}/{total}]: {name}",
        f"Started  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Input    : {input_folder}",
        f"Output   : {output_folder}",
        f"tau={tau_ms}ms  filter={FILTER_SIZE}  threshold={threshold}"
        f"  norm={norm_len}  break={break_point}",
        "=" * 60, "",
    ])
    print(header, flush=True)
    with open(log_file, "w") as f:
        f.write(header)

    csv_files = sorted(
        input_folder.glob("*.csv"),
        key=lambda p: int(p.stem.split("_")[-1]) if p.stem.split("_")[-1].isdigit() else 0
    )
    if not csv_files:
        msg = f"  WARNING: no CSV files found in {input_folder}"
        print(msg)
        with open(log_file, "a") as f:
            f.write(msg + "\n")
        return False

    max_files = max(1, int(len(csv_files) * break_point))
    csv_files = csv_files[:max_files]
    print(f"  {len(csv_files)} files in total\n")
    print(f"  {len(csv_files)} files to process (break_point={break_point})\n")
    with open(log_file, "a") as f:
        f.write(f"Found {len(csv_files)} CSV files to process\n")

    # Fresh filter instance for every job (clears temporal state)
    filter_instance = FilterDataRecursive(tau_ms, FILTER_SIZE, IMAGE_SIZE)

    job_start = time.time()
    for count, csv_file in enumerate(csv_files, 1):
        print(f"  [{count}/{len(csv_files)}] {csv_file.name}", flush=True)
        with open(log_file, "a") as f:
            f.write(f"Loading {csv_file.name}\n")
        try:
            filtered_df = _proc.process_csv_file(
                str(csv_file),
                filter_instance,
                str(output_folder),
                threshold,
                fixed_sampling=FIXED_SAMPLING,
                seed_str=SEED_STR,
                normalization_length=norm_len,
            )
            out_path = output_folder / f"filtered_{csv_file.name}"
            filtered_df.to_csv(out_path, index=False)
            print(f"    Saved: {out_path.name}", flush=True)
        except Exception as e:
            msg = f"    ERROR on {csv_file.name}: {e}"
            print(msg)
            with open(log_file, "a") as f:
                f.write(msg + "\n" + traceback.format_exc())

    elapsed = (time.time() - job_start) / 60.0
    print(f"\n  Done — {elapsed:.1f} min  |  output: {output_folder}\n")
    return True


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Batch runner for process.py")
    parser.add_argument("--jobs",    nargs="+", type=int,
                        help="0-based indices of jobs to run (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the job table without running anything")
    parser.add_argument("--resume",  action="store_true",
                        help="Skip any job whose output folder already has filtered CSVs")
    args = parser.parse_args()

    selected = args.jobs if args.jobs else list(range(len(JOBS)))

    print(f"\n{'#'*60}")
    print(f"  BATCH RUNNER  —  {len(selected)} job(s) queued")
    print(f"{'#'*60}\n")

    if args.dry_run:
        print(f"  {'#':>3}  {'name':<45} {'bp':>5} {'thr':>6} {'tau':>5} {'norm':>6}")
        print(f"  {'-'*70}")
        for i in selected:
            n, ds, bp, thr, tau, norm = JOBS[i]
            print(f"  {i:>3}  {n:<45} {bp:>5.2f} {thr:>6.3f} {tau:>5.0f} {str(norm):>6}")
        print("\n  (dry-run — nothing executed)\n")
        return

    batch_start = time.time()
    results = {}

    for i in selected:
        if i >= len(JOBS):
            print(f"  WARNING: job index {i} out of range — skipping")
            continue

        if args.resume:
            #out_dir = Path(DATASET_DIRS[JOBS[i][1]]) / JOBS[i][0]
            out_dir = Path(RESULTS_DIR) / JOBS[i][0]
            if list(out_dir.glob("filtered_*.csv")):
                print(f"  SKIP (--resume): job {i}  '{JOBS[i][0]}'")
                results[i] = "skipped"
                continue

        ok = run_job(JOBS[i], i, len(selected))
        results[i] = "ok" if ok else "failed"

    total_min = (time.time() - batch_start) / 60.0
    print(f"\n{'#'*60}")
    print(f"  BATCH COMPLETE — {total_min:.1f} min total")
    print(f"{'#'*60}")
    for i, status in results.items():
        icon = "OK" if status == "ok" else ("--" if status == "skipped" else "!!")
        print(f"  [{icon}]  {i:>2}: {JOBS[i][0]}  ({status})")
    print()


if __name__ == "__main__":
    main()
