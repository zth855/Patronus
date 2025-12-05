import logging
import os
import subprocess
import threading
import time


class GPUMonitor:
    """Simple GPU monitor that polls `nvidia-smi` and writes CSV lines to a file.

    Each line is timestamp + nvidia-smi CSV output (no header).
    Uses a background daemon thread so it won't block program exit.
    """

    def __init__(self, interval=5, out_file=None, logger=None):
        self.interval = interval
        self.out_file = out_file or os.path.join(".", "gpu_monitor.log")
        self._stop_event = threading.Event()
        self._thread = None
        self.logger = logger or logging.getLogger(__name__)

    def _sample_lines(self):
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
            text = out.decode("utf-8", errors="ignore").strip()
            if not text:
                return []
            lines = text.splitlines()
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            return [f"{ts},{line.strip()}" for line in lines]
        except Exception as e:
            # If nvidia-smi is not present or fails, log at debug to avoid noisy output
            self.logger.debug("nvidia-smi query failed: %s", e)
            return []

    def _run(self):
        # Ensure directory exists
        try:
            dirpath = os.path.dirname(self.out_file)
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)
        except Exception:
            pass

        with open(self.out_file, "a", buffering=1, encoding="utf-8") as f:
            while not self._stop_event.is_set():
                try:
                    entries = self._sample_lines()
                    for e in entries:
                        f.write(e + "\n")
                        # also debug-log each entry
                        self.logger.debug("GPU: %s", e)
                except Exception as e:
                    self.logger.debug("GPU monitor write failed: %s", e)
                # sleep respects the stop event
                for _ in range(int(self.interval * 10)):
                    if self._stop_event.is_set():
                        break
                    time.sleep(0.1)

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="GPUMonitor"
        )
        self._thread.start()
        self.logger.info("GPUMonitor started, logging to %s", self.out_file)

    def stop(self, timeout=5.0):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        self.logger.info("GPUMonitor stopped")


def start_gpu_monitor(save_dir, task_name=None, interval=5, logger=None, out_file=None):
    """Start a GPUMonitor.

    Parameters:
    - save_dir: base directory (kept for backward compatibility)
    - task_name: if provided and out_file is None, will put monitor file under save_dir/task_name/
    - interval: sampling interval in seconds
    - logger: logger instance
    - out_file: optional explicit path to the output file. If provided, it overrides the default path.
    """
    if out_file:
        out_file_path = out_file
    else:
        task = str(task_name) if task_name is not None else ""
        log_dir = os.path.join(save_dir, task) if task else save_dir
        out_file_path = os.path.join(log_dir, "gpu_monitor.log")
    monitor = GPUMonitor(interval=interval, out_file=out_file_path, logger=logger)
    monitor.start()
    return monitor
