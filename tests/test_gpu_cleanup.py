import signal

import pytest

from scripts.gpu_cleanup import Container, Host, Journal, Process, cleanup


class Clock:
    def __init__(self):
        self.now = 0

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeHost(Host):
    def __init__(self, processes=(), containers=(), stubborn=False, residual=False, respawn=False):
        self.processes = {process.pid: process for process in processes}
        self.gpu_pids = set(self.processes)
        self.container_list = containers
        self.actions = []
        self.stubborn, self.residual, self.respawn = stubborn, residual, respawn
        self.counter = 100

    def snapshot(self):
        if self.respawn and not self.gpu_pids:
            self.counter += 1
            process = Process(self.counter, 1, self.counter, "restarted-python", "0::/session.scope")
            self.processes[process.pid] = process
            self.gpu_pids.add(process.pid)
        return {
            "gpus": [
                {
                    "index": i,
                    "uuid": f"GPU-{i}",
                    "used_mib": 4096 if self.residual else 0,
                    "total_mib": 140000,
                }
                for i in range(8)
            ],
            "applications": [{"pid": pid, "gpu_uuid": "GPU-0"} for pid in sorted(self.gpu_pids)],
        }

    def process(self, pid):
        return self.processes.get(pid)

    def containers(self):
        return self.container_list

    def stop_unit(self, unit):
        self.actions.append(("unit", unit))

    def stop_container(self, container, grace):
        self.actions.append(("container", container.id))
        for process in list(self.processes.values()):
            if container.id in process.cgroup or container.pid in self.ancestors(process.pid):
                self.processes.pop(process.pid)
                self.gpu_pids.discard(process.pid)

    def signal_process(self, process, signum):
        current = self.process(process.pid)
        if not current or current.started != process.started:
            return
        self.actions.append(("signal", process.pid, signum))
        if signum == signal.SIGKILL or not self.stubborn:
            self.processes.pop(process.pid)
            self.gpu_pids.discard(process.pid)


def run_cleanup(host, tmp_path, **kwargs):
    clock = Clock()
    journal = Journal(tmp_path / "cleanup.json")
    cleanup(host, journal, grace=2, settle=4, clock=clock.time, sleep=clock.sleep, **kwargs)
    return journal


def test_idle_host_is_not_mutated(tmp_path):
    host = FakeHost()
    journal = run_cleanup(host, tmp_path)
    assert host.actions == []
    assert journal.events[-1]["action"] == "idle"


def test_only_gpu_container_stopped_and_original_policy_journaled(tmp_path):
    gpu = Container("a" * 64, "/gpu-task", 20, {"Name": "always", "MaximumRetryCount": 0})
    cpu = Container("b" * 64, "/cpu-service", 30, {"Name": "unless-stopped"})
    host = FakeHost([Process(21, 20, 10, "python", f"0::/docker/{gpu.id}")], [gpu, cpu])
    journal = run_cleanup(host, tmp_path)
    assert host.actions == [("container", gpu.id)]
    before = next(event for event in journal.events if event["action"] == "plan")
    assert before["containers"][0]["restart"]["Name"] == "always"


def test_systemd_stopped_before_terminating_gpu_child(tmp_path):
    host = FakeHost([Process(40, 1, 10, "python", "0::/system.slice/minimax.service")])
    run_cleanup(host, tmp_path)
    assert host.actions == [("unit", "minimax.service"), ("signal", 40, signal.SIGTERM)]


def test_stubborn_native_task_gets_term_then_kill(tmp_path):
    host = FakeHost([Process(40, 1, 10, "python", "0::/session.scope")], stubborn=True)
    run_cleanup(host, tmp_path)
    assert host.actions == [("signal", 40, signal.SIGTERM), ("signal", 40, signal.SIGKILL)]


@pytest.mark.parametrize(
    "cgroup",
    [
        "0::/kubepods/pod1",
        "0::/system.slice/docker.service",
        "0::/user.slice/user-1000.slice/user@1000.service/app.slice/gpu.service",
    ],
)
def test_unsupported_controller_or_infrastructure_fails_before_mutation(tmp_path, cgroup):
    host = FakeHost([Process(40, 1, 10, "python", "0::/session.scope"), Process(41, 1, 11, "python", cgroup)])
    with pytest.raises(RuntimeError):
        run_cleanup(host, tmp_path)
    assert host.actions == []


def test_swarm_container_is_not_blindly_killed(tmp_path):
    container = Container("a" * 64, "/swarm-task", 20, {"Name": "always"}, swarm_service="service-1")
    host = FakeHost([Process(21, 20, 10, "python", f"0::/docker/{container.id}")], [container])
    with pytest.raises(RuntimeError, match="Swarm/Kubernetes"):
        run_cleanup(host, tmp_path)
    assert host.actions == []


def test_residual_memory_does_not_pass_as_idle(tmp_path):
    with pytest.raises(RuntimeError, match="GPUs remain occupied"):
        run_cleanup(FakeHost(residual=True), tmp_path)


def test_restart_loop_is_bounded(tmp_path):
    host = FakeHost(respawn=True)
    with pytest.raises(RuntimeError, match="keep restarting"):
        run_cleanup(host, tmp_path)
    assert len(host.actions) == 3


def test_delayed_respawn_is_caught_during_settle_window(tmp_path):
    clock = Clock()
    host = FakeHost()
    original_snapshot = host.snapshot

    def snapshot():
        if clock.now == 4:
            process = Process(77, 1, 100, "delayed-python", "0::/session.scope")
            host.processes[77] = process
            host.gpu_pids.add(77)
        return original_snapshot()

    host.snapshot = snapshot
    cleanup(host, Journal(tmp_path / "cleanup.json"), grace=2, settle=6, clock=clock.time, sleep=clock.sleep)
    assert ("signal", 77, signal.SIGTERM) in host.actions
    assert clock.now >= 10


def test_pidfd_does_not_signal_reused_pid(monkeypatch):
    import os

    sent = []
    closed = []
    monkeypatch.setattr(os, "pidfd_open", lambda pid: 12345, raising=False)
    monkeypatch.setattr(signal, "pidfd_send_signal", lambda fd, sig: sent.append((fd, sig)), raising=False)
    monkeypatch.setattr(os, "close", closed.append)
    host = Host()
    original = Process(40, 1, 10, "python", "0::/session.scope")
    monkeypatch.setattr(host, "process", lambda pid: Process(40, 1, 11, "unrelated", "0::/session.scope"))
    host.signal_process(original, signal.SIGKILL)
    assert sent == []
    assert closed == [12345]
    monkeypatch.setattr(host, "process", lambda pid: original)
    host.signal_process(original, signal.SIGTERM)
    assert sent == [(12345, signal.SIGTERM)]


def test_docker_restart_policy_changed_before_stop(monkeypatch):
    host = Host()
    container = Container("a" * 64, "/gpu-task", 20, {"Name": "always"})
    monkeypatch.setattr(host, "containers", lambda: [container])
    commands = []
    monkeypatch.setattr(host, "run", lambda command, **kwargs: commands.append(command))
    host.stop_container(container, 20)
    assert commands == [
        ["docker", "update", "--restart=no", container.id],
        ["docker", "stop", "--time", "20", container.id],
    ]


def test_unreadable_gpu_memory_is_not_assumed_free(monkeypatch):
    host = Host()
    monkeypatch.setattr(host, "run", lambda args: "0, GPU-0, [Insufficient Permissions], 140000\n")
    with pytest.raises(ValueError):
        host.snapshot()
