import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dclm_paths import (
    REPO_ROOT,
    INDEX_PATH,
    TOKENIZE_OUT_DIR,
    SHUFFLED_DIR,
)


TOKENIZE_SCRIPT = REPO_ROOT / "tokenize_split_buckets.py"
MERGE_SCRIPT = REPO_ROOT / "merge_buckets_shuffle.py"
UPLOAD_SCRIPT = REPO_ROOT / "upload.py"
CREATE_INDEX_SCRIPT = REPO_ROOT / "create_index.py"

NUM_BUCKETS = 128
TOKENIZE_NUM_BUCKET_FILES = 128
DEFAULT_WORKERS = min(8, max(1, (os.cpu_count() or 1) // 2))


def huggingface_cli():
    hf_bin = Path(sys.executable).with_name("hf")
    if hf_bin.exists():
        return str(hf_bin)
    return "hf"


def build_env():
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    env.setdefault("TF_NUM_INTEROP_THREADS", "1")
    return env


def ensure_index_exists():
    if INDEX_PATH.exists():
        return

    subprocess.run([sys.executable, str(CREATE_INDEX_SCRIPT)], check=True, cwd=REPO_ROOT)


def load_index():
    ensure_index_exists()
    with INDEX_PATH.open("r") as fh:
        return json.load(fh)


def shard_ids_from_index():
    index = load_index()
    return sorted(int(key) for key in index.keys())


def unique_global_shards():
    index = load_index()
    prefixes = set()
    for paths in index.values():
        for path in paths:
            match = re.search(r"(global-shard_\d+_of_\d+)", path)
            if match:
                prefixes.add(match.group(1))
    return sorted(prefixes)


def tokenize_output_dir(i):
    return TOKENIZE_OUT_DIR / f"array_record_{i:04d}"


def merge_output_path(i):
    return SHUFFLED_DIR / f"bucket_{i:04d}.array_record"


def tokenize_complete(i):
    out_dir = tokenize_output_dir(i)
    if not out_dir.exists():
        return False

    files = list(out_dir.glob("bucket_*.array_record"))
    return len(files) == TOKENIZE_NUM_BUCKET_FILES and all(path.stat().st_size > 0 for path in files)


def merge_complete(i):
    path = merge_output_path(i)
    return path.exists() and path.stat().st_size > 0


def run_logged_command(cmd, log_path, env):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_file:
        log_file.write(f"Command: {' '.join(cmd)}\n")
        log_file.flush()
        subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=True,
        )


def run_stage(stage_name, jobs, workers, log_root, env):
    if not jobs:
        print(f"[{stage_name}] nothing to do", flush=True)
        return

    print(f"[{stage_name}] launching {len(jobs)} job(s) with {workers} worker(s)", flush=True)
    failures = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {}
        for job_name, cmd, log_path in jobs:
            future = executor.submit(run_logged_command, cmd, log_path, env)
            future_map[future] = (job_name, log_path)
            print(f"[{stage_name}] queued {job_name} -> {log_path.relative_to(REPO_ROOT)}", flush=True)

        for future in as_completed(future_map):
            job_name, log_path = future_map[future]
            try:
                future.result()
                print(f"[{stage_name}] finished {job_name}", flush=True)
            except Exception as exc:
                failures.append((job_name, log_path, exc))
                print(f"[{stage_name}] failed {job_name} -> {log_path.relative_to(REPO_ROOT)}", flush=True)
                print(f"[{stage_name}] error: {exc}", flush=True)

    if failures:
        summary = "\n".join(
            f"  - {job_name}: {log_path.relative_to(REPO_ROOT)}" for job_name, log_path, _ in failures
        )
        raise RuntimeError(f"{stage_name} stage failed:\n{summary}")


def download_jobs(log_root):
    jobs = []
    for prefix in unique_global_shards():
        cmd = [
            huggingface_cli(),
            "download",
            "mlfoundations/dclm-baseline-1.0",
            "--repo-type",
            "dataset",
            "--include",
            f"{prefix}/**",
            "--local-dir",
            "dclm_subset",
        ]
        jobs.append((prefix, cmd, log_root / "download" / f"{prefix}.log"))
    return jobs


def tokenize_jobs(log_root):
    jobs = []
    for i in shard_ids_from_index():
        if tokenize_complete(i):
            print(f"[tokenize] skipping {i:04d} (output already present)", flush=True)
            continue
        cmd = [sys.executable, str(TOKENIZE_SCRIPT), "--i", str(i)]
        jobs.append((f"{i:04d}", cmd, log_root / "tokenize" / f"{i:04d}.log"))
    return jobs


def merge_jobs(log_root):
    jobs = []
    for i in range(NUM_BUCKETS):
        if merge_complete(i):
            print(f"[merge] skipping {i:04d} (output already present)", flush=True)
            continue
        cmd = [
            sys.executable,
            str(MERGE_SCRIPT),
            "--i",
            str(i),
            "--delete",
        ]
        jobs.append((f"{i:04d}", cmd, log_root / "merge" / f"{i:04d}.log"))
    return jobs


def upload_jobs(log_root):
    jobs = []
    for i in range(NUM_BUCKETS):
        cmd = [sys.executable, str(UPLOAD_SCRIPT), "--i", str(i)]
        jobs.append((f"{i:04d}", cmd, log_root / "upload" / f"{i:04d}.log"))
    return jobs


def parse_args():
    parser = argparse.ArgumentParser(description="Local DCLM pipeline runner")
    parser.add_argument(
        "stage",
        nargs="?",
        default="all",
        choices=["download", "tokenize", "merge", "upload", "all"],
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Number of concurrent shard jobs to launch",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=REPO_ROOT / "logs" / "dclm_pipeline",
        help="Directory for per-job logs",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    env = build_env()
    log_root = args.log_dir
    log_root.mkdir(parents=True, exist_ok=True)

    stages = []
    if args.stage in {"download", "all"}:
        stages.append(("download", download_jobs))
    if args.stage in {"tokenize", "all"}:
        stages.append(("tokenize", tokenize_jobs))
    if args.stage in {"merge", "all"}:
        stages.append(("merge", merge_jobs))
    if args.stage in {"upload", "all"}:
        stages.append(("upload", upload_jobs))

    for stage_name, job_builder in stages:
        jobs = job_builder(log_root)
        run_stage(stage_name, jobs, args.workers, log_root, env)

    print("Pipeline complete.", flush=True)


if __name__ == "__main__":
    main()
