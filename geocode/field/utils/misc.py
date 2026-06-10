"""Miscellaneous utils."""
import os
import shutil
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import psutil
from tqdm import tqdm


@contextmanager
def _dummy_with():
    """Dummy statement."""
    yield

def kill(proc_pid):
    """Kill proc and its childs."""
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()

def signal_handler(signum, frame):
    """Timeout handler."""
    _ = signum, frame
    raise TimeoutError("Timed out!")

def execute_tnav_models(models, license_url,
                        tnav_path, base_script_path=None, logfile=None,
                        global_timeout=None, process_timeout=None,
                        dump_rsm=True, dump_egrid=True, dump_unsmry=True, dump_unrst=True):
    """Execute a bash script for each model in a set of models.

    Parameters
    ----------
    models : str, list of str
        A path to model or list of pathes.
    license_url : str
        A license server url.
    tnav_path : str
        A path to tNavigator executable.
    base_script_path : str
        Path to script to execute.
    logfile : str
        A path to file where to point stdout and stderr.
    global_timeout : int
        Global timeout in seconds.
    process_timeout : int
        Process timeout. Kill process that exceeds the timeout and go to the next model.
    dump_rsm: bool
        Dump *.RSM file, by default True.
    dump_egrid: bool
        Dump *.EGRID file, by default False.
    dump_unsmry: bool
        Dump *.SMSPEC and *.UNSMRY files, by default False.
    dump_unrst: bool
        Dump *.UNRST file, by default True.
    """
    if base_script_path is None:
        base_script_path = Path(__file__).parents[2] / 'bin/tnav_run.sh'
    if license_url is None:
        raise ValueError('License url is not defined.')
    models = np.atleast_1d(models)
    keys = ''
    if dump_egrid:
        keys += 'e'
    if dump_unrst:
        keys += 'r'
    if dump_unsmry:
        keys += 'um'
    if len(keys) > 0:
        keys = '-' + keys
    if dump_rsm:
        keys += ' --ecl-rsm'

    base_args = ['bash', base_script_path, tnav_path, license_url, keys,]
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(-1 if global_timeout is None else global_timeout)
    with (open(logfile, 'w') if logfile is not None else _dummy_with()) as f:#pylint:disable=consider-using-with
        for model in tqdm(models):
            try:
                p = subprocess.Popen(base_args + [model], stdout=f, stderr=f)#pylint:disable=consider-using-with
                try:
                    p.wait(timeout=process_timeout)
                except subprocess.TimeoutExpired:
                    kill(p.pid)
            except Exception as err:
                kill(p.pid)
                raise err


def execute_julia_models(runid, case_path, out_root, *,
                         restart="none", timeout_s=None):
    """Run the JutulDarcy driver (jutul_run.jl) as a Julia subprocess.

    The driver writes states.h5, wells.h5, cell_indices.h5 and manifest.json
    under <out_root>/<runid>/; consumers read them via georead.jutul.load.

    Parameters
    ----------
    runid : str
        Name of the run subdirectory.
    case_path : str | Path
        Path to the .DATA file.
    out_root : str | Path
        Directory that will contain the runid subdirectory.
    restart : "none" | "latest" | int
        JutulDarcy restart mode. "none" removes previous results; other
        modes reuse the JLD2 cache under <runid>/jutul_state/.
    timeout_s : int | None
        Subprocess timeout in seconds.

    Returns
    -------
    Path
        The runid subdirectory.
    """
    runid_dir = Path(out_root) / runid
    if restart == "none" and runid_dir.exists():
        shutil.rmtree(runid_dir)
    runid_dir.mkdir(parents=True, exist_ok=True)

    script = Path(__file__).parents[2] / "bin" / "jutul_run.jl"
    if isinstance(restart, int):
        restart = f"step:{restart}"
    argv = [os.environ.get("JULIA", "julia"),
            f"--project={script.parent}",
            str(script),
            f"--case={case_path}",
            f"--out={runid_dir}",
            f"--restart={restart}"]

    logpath = runid_dir / "julia.log"
    deadline = None if timeout_s is None else time.monotonic() + timeout_s
    with open(logpath, "wb") as log:
        p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)#pylint:disable=consider-using-with
        # Stream driver output (including the JutulDarcy progress bar) to the
        # console while keeping a copy in julia.log.
        while True:
            chunk = p.stdout.read1(4096)
            if not chunk:
                break
            log.write(chunk)
            sys.stdout.buffer.write(chunk)
            sys.stdout.flush()
            if deadline is not None and time.monotonic() > deadline:
                kill(p.pid)
                raise TimeoutError(f"Julia simulation exceeded {timeout_s}s, see {logpath}")
        rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"jutul_run.jl exited with code {rc}, see {logpath}")
    return runid_dir
