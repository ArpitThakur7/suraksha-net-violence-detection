"""Option B queue runner that survives Ctrl+C / console close events.

Launched with pythonw (no console), it ignores SIGINT itself and starts every step in a new
process group with CTRL+C disabled and a hidden console, so a stray console control event
cannot kill training (the queue died to a ^C at 00:37 on Nov 25). The grid is retried up to
5 times; each retry skips finished runs and resumes the interrupted one from its checkpoint.
"""
import ctypes, datetime, os, signal, subprocess, sys, time

ROOT = r"E:\Projects\major"
LOG = os.path.join(ROOT, "work", "logs", "seeds.log")
FLAGS = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
PY = os.path.join(os.path.dirname(sys.executable), "python.exe")  # console python for children


def note(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"=== {datetime.datetime.now():%Y-%m-%d %H:%M:%S} {msg} ===\n")


def step(args):
    with open(LOG, "a", encoding="utf-8") as f:
        return subprocess.call([PY, "-u", *args], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL, creationflags=FLAGS)


def main():
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, signal.SIG_IGN)
    os.environ["HF_HUB_OFFLINE"] = "1"
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)  # keep PC awake
    note(f"laptop: option B runner (pid {os.getpid()}) - seed 1 MotionLite +/- distillation")
    if len(sys.argv) > 2 and sys.argv[1] == "--wait":  # adopt a grid already running
        wait_pid = int(sys.argv[2])
        note(f"waiting for running grid pid {wait_pid} before continuing")
        h = ctypes.windll.kernel32.OpenProcess(0x00100000, False, wait_pid)  # SYNCHRONIZE
        if h:
            ctypes.windll.kernel32.WaitForSingleObject(h, 0xFFFFFFFF)
            ctypes.windll.kernel32.CloseHandle(h)
    for attempt in range(1, 6):
        rc = step(["-m", "pipeline", "grid", "--grid", "pipeline/configs/grid_seeds.yaml",
                   "--only", "motionlite", "--seeds", "1"])
        if rc == 0:
            break
        note(f"grid exited with code {rc} (attempt {attempt} of 5), restarting in 30 s")
        time.sleep(30)
    # Results table + figures only; slides and reports wait for the user's format.
    for args in (["-m", "pipeline", "probe", "table"], ["-m", "pipeline", "figures"]):
        step(args)
    note("OPTION B DONE")
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    main()
