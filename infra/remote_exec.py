"""Remote execution over SSH with a GPU-enabled docker run.

Usage as a library:

    from infra.remote_exec import Job, load_cluster, run_pool, run_one

    workers, image = load_cluster(role="collect")
    jobs = [Job(name="job1", inner_cmd="uv run ..."), ...]
    results = run_pool(workers, image, gpu=True, jobs=jobs, logs_dir=Path("logs"))

Usage as a CLI (single command on first matching worker):

    uv run python -m infra.remote_exec \\
        --role train \\
        --log-path logs/train.log \\
        -- uv run experiments/foo/train.py --args

Assumes the cluster nodes share /workspace and /data/eric via NFS, have docker
installed, and have the image already pulled.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import threading
import time
from typing import IO
try:
    import tomllib  # type: ignore[import-not-found]
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]
from dataclasses import dataclass, replace
from pathlib import Path

from infra.gpu_lease import LeaseManager

CLUSTER_TOML = Path("/workspace/cluster.toml")
SSH_KEY = str(Path.home() / ".ssh" / "id_ed25519")
DEFAULT_IMAGE = "ghcr.io/ericjang/alphago-worker:latest"


@dataclass(frozen=True)
class Worker:
    ip: str
    user: str = "root"
    ssh_port: int | None = None
    shares_nfs: bool = True
    num_gpu: int = 1
    # Free-form label for the GPU family on this host (e.g. "rtx6000_ada",
    # "blackwell", "h100"). Used by run_pool for Job.avoid/require filtering.
    # None means unknown — avoid-lists pass, require-lists reject.
    gpu_type: str | None = None
    # If set, this sub-worker is pinned to a single GPU index via
    # CUDA_VISIBLE_DEVICES. Populated by expand_gpu_workers().
    gpu_index: int | None = None

    @property
    def target(self) -> str:
        return f"{self.user}@{self.ip}"


@dataclass(frozen=True)
class Job:
    name: str
    inner_cmd: str
    # Host paths (under /data/eric/...) to rsync to the remote before the job
    # runs and to rsync back after a successful (rc==0) run. Ignored on
    # workers that share NFS. See run_one() for the transfer protocol.
    push_files: tuple[str, ...] = ()
    pull_dirs: tuple[str, ...] = ()
    # Hardware affinity. run_pool skips this job on workers whose gpu_type
    # appears in avoid_gpu_types, or (if require_gpu_types is non-empty) on
    # workers whose gpu_type is not in require_gpu_types. Workers with
    # gpu_type=None pass avoid checks but fail require checks.
    avoid_gpu_types: frozenset[str] = frozenset()
    require_gpu_types: frozenset[str] = frozenset()


def load_cluster(role: str, toml_path: Path = CLUSTER_TOML) -> tuple[list[Worker], str]:
    """Return (workers matching role, image name) from cluster.toml."""
    with open(toml_path, "rb") as f:
        cfg = tomllib.load(f)
    image = cfg.get("image", DEFAULT_IMAGE)
    workers: list[Worker] = []
    for ip, entry in cfg.get("nodes", {}).items():
        if role not in entry.get("roles", []):
            continue
        workers.append(Worker(
            ip=ip,
            user=entry.get("user", "root"),
            ssh_port=entry.get("ssh_port"),
            shares_nfs=entry.get("shares_nfs", True),
            num_gpu=int(entry.get("num_gpu", 1)),
            gpu_type=entry.get("gpu_type"),
        ))
    return workers, image


def expand_gpu_workers(workers: list[Worker]) -> list[Worker]:
    """Explode each multi-GPU worker into num_gpu sub-workers pinned to a
    single GPU via CUDA_VISIBLE_DEVICES. Single-GPU workers pass through.

    Use when callers want to fan out one job per GPU on the same host so
    each GPU can process its own stream of jobs independently.
    """
    out: list[Worker] = []
    for w in workers:
        if w.num_gpu <= 1:
            out.append(w)
            continue
        for i in range(w.num_gpu):
            out.append(replace(w, gpu_index=i))
    return out


def build_ssh_argv(worker: Worker, image: str, gpu: bool, inner_cmd: str,
                   exp_name: str | None = None) -> list[str]:
    """Wrap `inner_cmd` in `docker run` and then `ssh` to the worker.

    The image ships a pre-synced venv at /.venv and the full /workspace tree
    (src, third_party, compiled binaries). We do NOT bind-mount /workspace —
    instead we overlay only the single experiment subdir so updated experiment
    code propagates, while everything else stays pinned to the image.
    """
    env_flags = (
        "-e GAME_DATA_DIR=/nfs/game_data_root "
        "-e PYTHONUNBUFFERED=1 "
        "-e ALPHAGO_ROOT=/workspace "
        "-e ALPHAGO_BASE_DIR=/workspace"
    )
    if worker.gpu_index is not None:
        # --gpus all grants visibility; CUDA_VISIBLE_DEVICES restricts what
        # CUDA sees inside the container so each sub-worker only touches
        # one GPU. Subprocesses (e.g. KataGo) inherit the env var.
        env_flags += f" -e CUDA_VISIBLE_DEVICES={worker.gpu_index}"
    gpu_flag = "--gpus all " if gpu else ""
    docker = "docker" if worker.user == "root" else "sudo docker"
    exp_mount = ""
    if exp_name:
        exp_mount = (
            f"-v /data/eric/LearnAlphaGo/experiments/{exp_name}"
            f":/workspace/experiments/{exp_name} "
        )
    remote = (
        f"{docker} run --rm --network=host {gpu_flag}--shm-size=14g "
        f"{exp_mount}-v /data/eric:/nfs "
        f"-v {'/root' if worker.user == 'root' else f'/home/{worker.user}'}/.secrets:/workspace/.secrets:ro "
        f"-w /workspace {env_flags} {image} "
        f"bash -lc {shlex.quote(inner_cmd)}"
    )
    ssh = ["ssh", "-o", "StrictHostKeyChecking=no", "-i", SSH_KEY]
    if worker.ssh_port:
        ssh += ["-p", str(worker.ssh_port)]
    ssh += [worker.target, remote]
    return ssh


def _ssh_argv(worker: Worker) -> list[str]:
    argv = ["ssh", "-o", "StrictHostKeyChecking=no", "-i", SSH_KEY]
    if worker.ssh_port:
        argv += ["-p", str(worker.ssh_port)]
    return argv


def _rsync_argv(worker: Worker) -> list[str]:
    """rsync -az over the same ssh key/port the rest of the module uses."""
    ssh_cmd = "ssh -o StrictHostKeyChecking=no -i " + shlex.quote(SSH_KEY)
    if worker.ssh_port:
        ssh_cmd += f" -p {worker.ssh_port}"
    return ["rsync", "-az", "--mkpath", "-e", ssh_cmd]


def _ssh(worker: Worker, remote_cmd: str, log_fp: IO[bytes]) -> int:
    argv = _ssh_argv(worker) + [worker.target, remote_cmd]
    log_fp.write(f"# ssh: {argv}\n".encode())
    log_fp.flush()
    return subprocess.run(argv, stdout=log_fp, stderr=subprocess.STDOUT).returncode


def _ensure_remote_src(worker: Worker, exp_name: str, log_fp: IO[bytes]) -> int:
    """Mkdir the expected subtree + rsync experiments/<exp>/ to the remote."""
    # mkdir runs as root over ssh, but the container process is uid 1000 (dev),
    # so we chown the game_data_root tree to 1000 so the container can write
    # into save-dirs it creates inside it.
    exp_q = shlex.quote(exp_name)
    sudo = "" if worker.user == "root" else "sudo "
    # /nfs/game_data_root is where the container (uid 1000 = dev) writes
    # save-dirs, so always chown it to 1000 regardless of ssh user.
    # /nfs/LearnAlphaGo + /nfs/checkpoints are written over rsync by the ssh
    # user — only need chowning when that user isn't root.
    mkdir_cmd = (
        f"{sudo}mkdir -p /nfs/LearnAlphaGo/experiments/{exp_q}"
        f" /nfs/checkpoints /nfs/game_data_root"
        f" /nfs/game_data_root/experiments/{exp_q}"
        f" && {sudo}chown -R 1000:1000 /nfs/game_data_root"
    )
    if worker.user != "root":
        mkdir_cmd += (
            f" && {sudo}chown -R $(id -u):$(id -g)"
            f" /nfs/LearnAlphaGo /nfs/checkpoints"
        )
    rc = _ssh(worker, mkdir_cmd, log_fp)
    if rc != 0:
        return rc
    src = f"/nfs/LearnAlphaGo/experiments/{exp_name}/"
    dst = f"{worker.target}:/nfs/LearnAlphaGo/experiments/{exp_name}/"
    argv = _rsync_argv(worker) + [
        "--exclude=__pycache__", "--exclude=logs/", src, dst,
    ]
    log_fp.write(f"# rsync src: {argv}\n".encode())
    log_fp.flush()
    return subprocess.run(argv, stdout=log_fp, stderr=subprocess.STDOUT).returncode


def _push(worker: Worker, host_path: str, log_fp: IO[bytes]) -> int:
    argv = _rsync_argv(worker) + [host_path, f"{worker.target}:{host_path}"]
    log_fp.write(f"# push: {argv}\n".encode())
    log_fp.flush()
    return subprocess.run(argv, stdout=log_fp, stderr=subprocess.STDOUT).returncode


def _pull(worker: Worker, host_dir: str, log_fp: IO[bytes]) -> int:
    # Trailing slash: copy contents, preserving structure on the receiver.
    src = f"{worker.target}:{host_dir.rstrip('/')}/"
    dst = f"{host_dir.rstrip('/')}/"
    argv = _rsync_argv(worker) + [src, dst]
    log_fp.write(f"# pull: {argv}\n".encode())
    log_fp.flush()
    return subprocess.run(argv, stdout=log_fp, stderr=subprocess.STDOUT).returncode


def _periodic_pull_loop(
    worker: Worker,
    pull_dirs: tuple[str, ...],
    log_fp: IO[bytes],
    stop_event: threading.Event,
    interval_s: float,
) -> None:
    """Rsync-pull `pull_dirs` every `interval_s` seconds until `stop_event` fires.

    Runs in a background thread while a remote job is executing, so partial
    outputs survive a worker crash / SSH drop. Remote scratch is NOT removed
    here — cleanup stays attached to the success-path final pull in run_one.
    Early errors (e.g. the save-dir doesn't exist yet because the job hasn't
    written anything) are logged but otherwise ignored.
    """
    # `wait` returns True when the event is set (stop), False on timeout (tick).
    while not stop_event.wait(interval_s):
        log_fp.write(f"# periodic pull ({int(interval_s)}s tick)\n".encode())
        log_fp.flush()
        for d in pull_dirs:
            rc = _pull(worker, d, log_fp)
            if rc != 0:
                log_fp.write(f"# periodic pull rc={rc} for {d} (non-fatal)\n".encode())
                log_fp.flush()


def _read_priorities(path: Path | str | None) -> dict[str, int]:
    """Parse a priorities file into a `{name: priority}` dict.

    Format: one job per line, ``"<int_priority> <name>"`` separated by
    whitespace. ``#`` comments and blank lines are skipped. Lines whose first
    token isn't an integer, or that have fewer than two tokens, are skipped
    silently — the file is meant to be operator-edited so partial garbage
    shouldn't kill an in-flight run.

    Returns ``{}`` for ``None`` or missing paths so callers can drop the
    feature mid-run by deleting the file.
    """
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    out: dict[str, int] = {}
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            prio = int(parts[0])
        except ValueError:
            continue
        out[parts[1]] = prio
    return out


def _write_priorities_stub(path: Path | str, jobs: list[Job]) -> None:
    """Seed a priorities file with every job at priority 0.

    `run_pool` writes this once at startup if a priorities_file is given and
    the file doesn't already exist, so an operator can edit it mid-run to
    bump specific jobs without restarting.
    """
    lines = [
        "# Edit to re-prioritize pending jobs (higher = sooner). Format:",
        "#   <int> <job_name>",
        "# Blank lines and '#' comments are ignored. Re-read on every claim.",
        "",
    ]
    for j in jobs:
        lines.append(f"0 {j.name}")
    Path(path).write_text("\n".join(lines) + "\n")


def run_one(
    worker: Worker,
    image: str,
    gpu: bool,
    inner_cmd: str,
    log_path: Path,
    max_retries: int = 3,
    retry_backoff_s: float = 5.0,
    job: Job | None = None,
    exp_name: str | None = None,
    prepared_hosts: set[str] | None = None,
    periodic_pull_interval_s: float = 300.0,
) -> int:
    """Run `inner_cmd` on `worker`, tee stdout+stderr to `log_path`. Returns the
    final rc. On non-zero rc, retry up to `max_retries` times with a fixed
    `retry_backoff_s` pause between attempts. Each attempt appends to the same
    log with an attempt header so the history is visible.

    When `periodic_pull_interval_s > 0` and the worker doesn't share NFS, a
    background thread rsyncs `job.pull_dirs` back every interval while the
    job runs, so partial results survive an SSH disconnect or worker crash.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    argv = build_ssh_argv(worker, image, gpu, inner_cmd, exp_name=exp_name)
    total_attempts = max_retries + 1
    rc = -1
    need_transfers = (not worker.shares_nfs) and job is not None
    for attempt in range(1, total_attempts + 1):
        with open(log_path, "ab") as lf:
            header = f"# attempt {attempt}/{total_attempts}  ssh argv: {argv}\n"
            lf.write(header.encode())
            lf.flush()
            if need_transfers:
                assert exp_name, "exp_name required when dispatching to a non-NFS worker"
                assert job is not None  # implied by need_transfers
                # One-shot src overlay per (ip, exp).
                if prepared_hosts is None or worker.ip not in prepared_hosts:
                    prep_rc = _ensure_remote_src(worker, exp_name, lf)
                    if prep_rc != 0:
                        lf.write(f"# _ensure_remote_src rc={prep_rc}\n".encode())
                        rc = prep_rc
                        if attempt == total_attempts:
                            return rc
                        time.sleep(retry_backoff_s)
                        continue
                    if prepared_hosts is not None:
                        prepared_hosts.add(worker.ip)
                # Push input files.
                push_rc = 0
                for p in job.push_files:
                    push_rc = _push(worker, p, lf)
                    if push_rc != 0:
                        break
                if push_rc != 0:
                    lf.write(f"# push rc={push_rc}\n".encode())
                    rc = push_rc
                    if attempt == total_attempts:
                        return rc
                    time.sleep(retry_backoff_s)
                    continue
            # Start periodic partial-result pulls while the job runs. Only
            # enabled for non-NFS workers with pull_dirs, since NFS workers
            # already write straight to shared storage.
            pull_stop = threading.Event()
            pull_thread: threading.Thread | None = None
            if (need_transfers and job is not None and job.pull_dirs
                    and periodic_pull_interval_s > 0):
                pull_thread = threading.Thread(
                    target=_periodic_pull_loop,
                    args=(worker, job.pull_dirs, lf, pull_stop, periodic_pull_interval_s),
                    name=f"periodic-pull-{worker.ip}",
                    daemon=True,
                )
                pull_thread.start()
            try:
                rc = subprocess.run(argv, stdout=lf, stderr=subprocess.STDOUT).returncode
            finally:
                if pull_thread is not None:
                    pull_stop.set()
                    pull_thread.join(timeout=max(30.0, periodic_pull_interval_s))
            lf.write(f"# attempt {attempt} rc={rc}\n".encode())
            if rc == 0 and need_transfers:
                assert job is not None  # implied by need_transfers
                pull_rc = 0
                for d in job.pull_dirs:
                    # Pull, then clear remote scratch on success so the host
                    # doesn't accumulate NPZs across iters.
                    for pull_attempt in range(1, 3):
                        pull_rc = _pull(worker, d, lf)
                        if pull_rc == 0:
                            break
                        lf.write(f"# pull attempt {pull_attempt} rc={pull_rc}\n".encode())
                        time.sleep(retry_backoff_s)
                    if pull_rc != 0:
                        break
                    # Files were written by the container as uid 1000 (dev);
                    # if the ssh user isn't root, it can't delete them
                    # directly — use sudo to clean up.
                    sudo = "" if worker.user == "root" else "sudo "
                    _ssh(worker, f"{sudo}rm -rf {shlex.quote(d)}", lf)
                if pull_rc != 0:
                    lf.write(f"# final pull rc={pull_rc}\n".encode())
                    rc = pull_rc
        if rc == 0 or attempt == total_attempts:
            return rc
        print(f"[{worker.ip}] attempt {attempt} failed rc={rc}, retrying in {retry_backoff_s}s",
              flush=True)
        time.sleep(retry_backoff_s)
    return rc


def run_pool(
    workers: list[Worker],
    image: str,
    gpu: bool,
    jobs: list[Job],
    logs_dir: Path,
    max_retries: int = 3,
    max_reschedules: int = 3,
    role: str | None = None,
    refresh_interval_s: float = 10.0,
    exp_name: str | None = None,
    per_gpu: bool = False,
    periodic_pull_interval_s: float = 300.0,
    share_cluster: bool = False,
    poll_interval_s: float = 5.0,
    priorities_file: Path | None = None,
) -> dict[str, int]:
    """Round-robin dispatch `jobs` across `workers`.

    - Each worker thread pulls from a shared FIFO queue; each job's log lives
      at `logs_dir/<name>.log`.
    - Each failed attempt is retried up to `max_retries` times on the same
      worker (inside `run_one`).
    - If a job exits with rc=255 (SSH-layer failure: host unreachable, key
      mismatch, connection dropped), the worker is treated as dead: the job is
      re-enqueued (up to `max_reschedules` times) and the worker thread exits.
    - If `role` is set, cluster.toml is reloaded every `refresh_interval_s`
      and new workers (IPs not currently active and not already marked dead)
      get fresh worker threads — so an operator can add a replacement node to
      cluster.toml and have it absorb the re-enqueued work.
    - When `per_gpu=True`, multi-GPU workers are expanded into one sub-worker
      per GPU (each pinned via CUDA_VISIBLE_DEVICES); one job at a time per
      sub-worker thread. Applies to both the initial dispatch and cluster
      reloads.

    Returns {job_name: rc}; jobs that never ran are absent from the dict.
    """
    logs_dir.mkdir(parents=True, exist_ok=True)
    if per_gpu:
        workers = expand_gpu_workers(workers)
    lease_mgr = LeaseManager(driver=exp_name or "") if share_cluster else None
    if priorities_file is not None:
        priorities_file = Path(priorities_file)
        if not priorities_file.exists():
            _write_priorities_stub(priorities_file, jobs)

    # Jobs are held in an index-addressable list rather than a Queue so each
    # worker can skip jobs whose hardware constraints it doesn't satisfy and
    # claim the next compatible one instead.
    pending: list[Job] = list(jobs)
    claimed: set[int] = set()          # index in pending that a worker owns
    results: dict[str, int] = {}
    reschedule_counts: dict[str, int] = {j.name: 0 for j in jobs}
    active_keys: set[tuple[str, int | None]] = set()
    dead_ips: set[str] = set()
    prepared_hosts: set[str] = set()
    in_flight: list[int] = [0]
    threads: list[threading.Thread] = []
    lock = threading.Lock()

    def _key(w: Worker) -> tuple[str, int | None]:
        return (w.ip, w.gpu_index)

    def _tag(w: Worker) -> str:
        return f"{w.ip}" + (f":gpu{w.gpu_index}" if w.gpu_index is not None else "")

    def _compatible(w: Worker, j: Job) -> bool:
        if j.require_gpu_types and w.gpu_type not in j.require_gpu_types:
            return False
        if j.avoid_gpu_types and w.gpu_type is not None and w.gpu_type in j.avoid_gpu_types:
            return False
        return True

    def _claim_next(w: Worker) -> tuple[int, Job] | None:
        """Claim the next unclaimed compatible job. With priorities_file set,
        higher-priority jobs win; ties broken by FIFO order. Caller holds lock."""
        prio = _read_priorities(priorities_file)
        candidates = [
            (i, j) for i, j in enumerate(pending)
            if i not in claimed and _compatible(w, j)
        ]
        if not candidates:
            return None
        # Higher priority first; ties → original index (FIFO).
        candidates.sort(key=lambda ij: (-prio.get(ij[1].name, 0), ij[0]))
        idx, job = candidates[0]
        claimed.add(idx)
        return idx, job

    def worker_loop(w: Worker) -> None:
        with lock:
            active_keys.add(_key(w))
        try:
            while True:
                with lock:
                    picked = _claim_next(w)
                if picked is None:
                    return  # nothing left that this worker can run
                idx, j = picked
                handle = lease_mgr.try_acquire(_key(w), job=j.name) if lease_mgr else None
                if lease_mgr and handle is None:
                    # A peer driver owns this GPU; un-claim and back off so
                    # another sub-worker (or a later tick) can pick up j.
                    with lock:
                        claimed.discard(idx)
                    time.sleep(poll_interval_s)
                    continue
                with lock:
                    in_flight[0] += 1
                log = logs_dir / f"{j.name}.log"
                print(f"[{_tag(w)}] START {j.name}  -> {log.name}", flush=True)
                try:
                    rc = run_one(w, image, gpu, j.inner_cmd, log, max_retries=max_retries,
                                 job=j, exp_name=exp_name, prepared_hosts=prepared_hosts,
                                 periodic_pull_interval_s=periodic_pull_interval_s)
                finally:
                    if handle is not None:
                        lease_mgr.release(handle)  # type: ignore[union-attr]
                if rc == 255:
                    # SSH failure: mark host dead, release the claim so another
                    # worker can pick it up (budget permitting), then exit.
                    with lock:
                        in_flight[0] -= 1
                        attempts = reschedule_counts[j.name] + 1
                        reschedule_counts[j.name] = attempts
                        dead_ips.add(w.ip)
                        if attempts <= max_reschedules:
                            claimed.discard(idx)
                            print(f"[{_tag(w)}] SSH failure (rc=255); re-enqueuing "
                                  f"{j.name} (reschedule {attempts}/{max_reschedules})",
                                  flush=True)
                        else:
                            results[j.name] = rc
                            print(f"[{_tag(w)}] SSH failure (rc=255); {j.name} "
                                  f"exhausted {max_reschedules} reschedules, giving up",
                                  flush=True)
                    return
                with lock:
                    results[j.name] = rc
                    in_flight[0] -= 1
                status = "OK" if rc == 0 else f"FAIL rc={rc}"
                print(f"[{_tag(w)}] END   {j.name}  {status}", flush=True)
        finally:
            with lock:
                active_keys.discard(_key(w))

    def start_worker(w: Worker) -> None:
        t = threading.Thread(target=worker_loop, args=(w,), name=f"worker-{_tag(w)}", daemon=False)
        threads.append(t)
        t.start()

    for w in workers:
        start_worker(w)

    # Supervisor: wait until all jobs resolve, periodically refreshing
    # cluster.toml (if role is set) so replacement nodes or nodes with new
    # gpu_type tags can absorb still-pending work.
    while True:
        with lock:
            unclaimed = len(pending) - len(claimed)
            alive = len(active_keys)
            inflight = in_flight[0]
        if unclaimed == 0 and inflight == 0:
            break
        if alive == 0 and inflight == 0 and role is None:
            # No workers left and no way to pull in new ones.
            break
        if role is not None and unclaimed > 0:
            try:
                latest, _img = load_cluster(role)
            except Exception as e:
                print(f"[scheduler] cluster reload failed: {e}", flush=True)
                latest = []
            if per_gpu:
                latest = expand_gpu_workers(latest)
            for w in latest:
                with lock:
                    if _key(w) in active_keys or w.ip in dead_ips:
                        continue
                print(f"[scheduler] new worker from cluster.toml refresh: {_tag(w)}",
                      flush=True)
                start_worker(w)
        if alive == 0 and unclaimed > 0:
            print(f"[scheduler] {unclaimed} jobs pending but no live workers; "
                  f"waiting for cluster.toml refresh", flush=True)
        time.sleep(refresh_interval_s)

    for t in threads:
        t.join()
    if lease_mgr is not None:
        lease_mgr.shutdown()
    return results


def _cli() -> int:
    p = argparse.ArgumentParser(description="Run one command on the first worker matching --role.")
    p.add_argument("--role", required=True, help="cluster.toml role to match")
    p.add_argument("--log-path", type=Path, required=True, help="where to tee remote stdout+stderr")
    p.add_argument("--no-gpu", action="store_true", help="drop --gpus all")
    p.add_argument("--retries", type=int, default=3,
                   help="retry a failed command this many times (default: 3)")
    p.add_argument("--exp-name", default=None,
                   help="experiment subdir to overlay at /workspace/experiments/<exp-name>")
    p.add_argument("--cluster-toml", type=Path, default=CLUSTER_TOML)
    p.add_argument("--share-cluster", action="store_true",
                   help="acquire a per-GPU NFS lease so peers can co-occupy the cluster")
    p.add_argument("cmd", nargs=argparse.REMAINDER,
                   help="command to run remotely (use `--` to separate from launcher args)")
    args = p.parse_args()

    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    if not cmd:
        print("ERROR: no command to run (use `--` followed by the command)", file=sys.stderr)
        return 2
    workers, image = load_cluster(args.role, args.cluster_toml)
    if not workers:
        print(f"ERROR: no node with role '{args.role}' in {args.cluster_toml}", file=sys.stderr)
        return 1

    inner_cmd = " ".join(shlex.quote(c) for c in cmd)
    if args.share_cluster:
        # Route through run_pool so the job waits for any free GPU across
        # the role's hosts instead of contending for workers[0] specifically.
        job = Job(name=args.log_path.stem, inner_cmd=inner_cmd)
        print(f"[share] exec: {inner_cmd}", flush=True)
        print(f"[share] log:  {args.log_path}", flush=True)
        results = run_pool(
            workers, image, not args.no_gpu, [job],
            logs_dir=args.log_path.parent,
            max_retries=args.retries, role=args.role,
            exp_name=args.exp_name, per_gpu=True, share_cluster=True,
        )
        # run_pool names its log `<job.name>.log` in logs_dir; rename if needed.
        produced = args.log_path.parent / f"{job.name}.log"
        if produced != args.log_path and produced.exists():
            produced.replace(args.log_path)
        rc = results.get(job.name, 1)
        print(f"[share] done rc={rc}", flush=True)
        return rc

    worker = workers[0]
    print(f"[{worker.ip}] exec: {inner_cmd}", flush=True)
    print(f"[{worker.ip}] log:  {args.log_path}", flush=True)
    rc = run_one(worker, image, not args.no_gpu, inner_cmd, args.log_path,
                 max_retries=args.retries, exp_name=args.exp_name)
    print(f"[{worker.ip}] done rc={rc}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(_cli())
