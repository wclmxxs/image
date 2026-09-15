#!/usr/bin/env python3
"""Host-only GPU takeover for the dedicated AWS node; no third-party Python dependencies."""

import argparse
import csv
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

PROTECTED_UNITS = {
    "docker.service",
    "containerd.service",
    "kubelet.service",
    "ssh.service",
    "sshd.service",
    "systemd-logind.service",
    "nvidia-persistenced.service",
    "nvidia-fabricmanager.service",
    "amazon-ssm-agent.service",
    "cron.service",
}


@dataclass(frozen=True)
class Process:
    pid: int
    parent: int
    started: int
    name: str
    cgroup: str


@dataclass
class Container:
    id: str
    name: str
    pid: int
    restart: dict
    swarm_service: str | None = None
    kubernetes_pod: str | None = None


class Host:
    def run(self, args, timeout=40):
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
        if result.returncode:
            raise RuntimeError(f"{args[0]} failed ({result.returncode}): {result.stderr.strip()[:1000]}")
        return result.stdout

    def snapshot(self):
        gpu_rows = csv.reader(
            self.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,uuid,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ]
            ).splitlines()
        )
        gpus = []
        for row in gpu_rows:
            index, gpu_uuid, used, total = (value.strip() for value in row)
            gpus.append(
                {"index": int(index), "uuid": gpu_uuid, "used_mib": int(used), "total_mib": int(total)}
            )
        if len(gpus) != 8 or {gpu["index"] for gpu in gpus} != set(range(8)):
            raise RuntimeError("GPU cleanup requires the verified dedicated 8-GPU host")
        rows = csv.reader(
            self.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,gpu_uuid",
                    "--format=csv,noheader,nounits",
                ]
            ).splitlines()
        )
        applications = []
        for row in rows:
            if not row:
                continue
            pid, gpu_uuid = (value.strip() for value in row)
            applications.append({"pid": int(pid), "gpu_uuid": gpu_uuid})
        return {"gpus": gpus, "applications": applications}

    def process(self, pid):
        try:
            root = Path("/proc") / str(pid)
            stat = (root / "stat").read_text()
            fields = stat[stat.rfind(")") + 2 :].split()
            return Process(
                pid,
                int(fields[1]),
                int(fields[19]),
                (root / "comm").read_text().strip(),
                (root / "cgroup").read_text().strip(),
            )
        except (FileNotFoundError, ProcessLookupError):
            return None

    def containers(self):
        ids = self.run(["docker", "ps", "--quiet", "--no-trunc"]).split()
        if not ids:
            return []
        # Select metadata explicitly: never collect container environments or command arguments.
        template = (
            "[{{json .Id}},{{json .Name}},{{json .State.Pid}},{{json .HostConfig.RestartPolicy}},"
            '{{json (index .Config.Labels "com.docker.swarm.service.id")}},'
            '{{json (index .Config.Labels "io.kubernetes.pod.name")}}]'
        )
        result = []
        for container_id in ids:
            try:
                rows = self.run(["docker", "inspect", "--format", template, container_id]).splitlines()
            except RuntimeError:
                if container_id not in self.run(["docker", "ps", "--quiet", "--no-trunc"]).split():
                    continue
                raise
            for row in rows:
                result.append(Container(*json.loads(row)))
        return result

    def ancestors(self, pid):
        result = set()
        while pid > 1 and pid not in result:
            result.add(pid)
            process = self.process(pid)
            if process is None:
                break
            pid = process.parent
        return result

    def stop_unit(self, unit):
        self.run(["systemctl", "stop", "--no-block", "--", unit])

    def stop_container(self, container, grace):
        current = {item.id for item in self.containers()}
        if container.id not in current:
            return
        self.run(["docker", "update", "--restart=no", container.id])
        self.run(["docker", "stop", "--time", str(grace), container.id], timeout=grace + 30)

    def signal_process(self, process, signum):
        if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
            raise RuntimeError("Cleanup requires Linux pidfd support (modern Ubuntu GPU AMI / Python >=3.9)")
        try:
            fd = os.pidfd_open(process.pid)
        except ProcessLookupError:
            return
        try:
            current = self.process(process.pid)
            if current is not None and current.started == process.started:
                signal.pidfd_send_signal(fd, signum)
        except ProcessLookupError:
            pass
        finally:
            os.close(fd)


class Journal:
    def __init__(self, path):
        self.path = Path(path) if path else None
        self.events = []

    def add(self, action, **fields):
        event = {"time": time.time(), "action": action, **fields}
        self.events.append(event)
        print(json.dumps(event, ensure_ascii=False), flush=True)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=".gpu-cleanup-", dir=self.path.parent)
            try:
                with os.fdopen(fd, "w") as stream:
                    json.dump(self.events, stream, ensure_ascii=False, indent=2)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, self.path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)


def plan(host, snapshot):
    containers = host.containers()
    targets = []
    units = set()
    selected_containers = {}
    protected_pids = host.ancestors(os.getpid()) | {1}
    for pid in sorted({app["pid"] for app in snapshot["applications"]}):
        process = host.process(pid)
        if process is None:
            continue
        if pid in protected_pids:
            raise RuntimeError(f"Refusing to terminate the deployment command or its ancestor (PID {pid})")
        targets.append(process)
        ancestors = host.ancestors(pid)
        container = next(
            (item for item in containers if item.id in process.cgroup or item.pid in ancestors), None
        )
        if container:
            if container.swarm_service or container.kubernetes_pod:
                raise RuntimeError(
                    f"GPU container {container.name} is managed by Swarm/Kubernetes; stop its workload controller first"
                )
            selected_containers[container.id] = container
            continue
        if "kubepods" in process.cgroup:
            raise RuntimeError(f"GPU PID {pid} belongs to Kubernetes; stop its workload controller first")
        services = re.findall(r"(?:^|/)([^/\n]+\.service)(?=/|$)", process.cgroup, re.MULTILINE)
        unit = services[-1] if services else None
        if unit and unit.startswith("user@"):
            unit = None
        if unit in PROTECTED_UNITS:
            raise RuntimeError(
                f"GPU PID {pid} belongs to infrastructure unit {unit}; refusing to stop that unit"
            )
        if unit:
            if "/user.slice/" in process.cgroup:
                raise RuntimeError(
                    f"GPU PID {pid} belongs to user service {unit}; stop it with that user's systemctl --user"
                )
            units.add(unit)
    return targets, list(selected_containers.values()), sorted(units)


def idle(snapshot, max_idle_mib):
    return not snapshot["applications"] and all(gpu["used_mib"] <= max_idle_mib for gpu in snapshot["gpus"])


def cleanup(
    host, journal, grace=20, settle=30, rounds=3, max_idle_mib=1024, clock=time.monotonic, sleep=time.sleep
):
    for attempt in range(1, rounds + 1):
        snapshot = host.snapshot()
        journal.add("inventory", round=attempt, **snapshot)
        targets, containers, units = plan(host, snapshot)
        # Record the whole plan (including original restart policies) before any mutation.
        journal.add(
            "plan",
            processes=[asdict(item) for item in targets],
            containers=[asdict(item) for item in containers],
            units=units,
        )
        for unit in units:
            journal.add("stop_systemd", unit=unit)
            host.stop_unit(unit)
        for container in containers:
            journal.add(
                "stop_container", id=container.id, name=container.name, previous_restart=container.restart
            )
            host.stop_container(container, grace)
        for process in targets:
            current = host.process(process.pid)
            if current and current.started == process.started:
                journal.add("signal", pid=process.pid, name=process.name, signal="TERM")
                host.signal_process(process, signal.SIGTERM)
        deadline = clock() + grace
        while any(
            (current := host.process(item.pid)) and current.started == item.started for item in targets
        ):
            if clock() >= deadline:
                break
            sleep(0.5)
        for process in targets:
            current = host.process(process.pid)
            if current and current.started == process.started:
                journal.add("signal", pid=process.pid, name=process.name, signal="KILL")
                host.signal_process(process, signal.SIGKILL)
        deadline = clock() + settle
        consecutive_idle = 0
        while clock() < deadline:
            snapshot = host.snapshot()
            consecutive_idle = consecutive_idle + 1 if idle(snapshot, max_idle_mib) else 0
            # An external supervisor may have respawned the job. Rescan ownership in a bounded new round.
            known = {(item.pid, item.started) for item in targets}
            if any(
                (process := host.process(app["pid"])) and (process.pid, process.started) not in known
                for app in snapshot["applications"]
            ):
                break
            sleep(1)
        if clock() >= deadline and consecutive_idle >= 3:
            snapshot = host.snapshot()
            if idle(snapshot, max_idle_mib):
                journal.add("idle", **snapshot)
                return
    raise RuntimeError(
        "GPUs remain occupied or tasks keep restarting. See cleanup journal; stop their launcher before retrying. No GPU reset was performed."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--report", action="store_true", help="Inventory only, never stop tasks")
    mode.add_argument("--check", action="store_true", help="Inventory only; fail if any GPU is busy")
    mode.add_argument(
        "--clean", action="store_true", help="Stop GPU workloads on all 8 GPUs and verify they are idle"
    )
    parser.add_argument("--journal")
    parser.add_argument("--grace", type=int, default=20)
    parser.add_argument("--settle", type=int, default=30)
    parser.add_argument("--max-idle-mib", type=int, default=1024)
    args = parser.parse_args()
    if sys.platform != "linux":
        raise RuntimeError("GPU cleanup must run on the Linux AWS host")
    if args.grace < 1 or args.settle < 3 or not 0 <= args.max_idle_mib <= 4096:
        raise ValueError("Invalid cleanup timeouts or idle-memory threshold")
    host, journal = Host(), Journal(args.journal)
    if args.clean:
        if os.geteuid() != 0:
            raise PermissionError("Run GPU cleanup as root so other users' GPU tasks can be stopped")
        try:
            cleanup(host, journal, args.grace, args.settle, max_idle_mib=args.max_idle_mib)
        except Exception as error:
            journal.add("failed", error=str(error))
            raise
    else:
        snapshot = host.snapshot()
        journal.add("inventory", **snapshot)
        if args.check and not idle(snapshot, args.max_idle_mib):
            raise RuntimeError(
                "GPUs are occupied. Normal start.sh cleans them; --no-gpu-cleanup refuses to start while busy."
            )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
