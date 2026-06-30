"""Miscellaneous utils."""
import codecs
import os
import re
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


def execute_julia_simulate(case_path, *, result_dir=None, timeout_s=None):
    """Run the JutulDarcy driver (jutul_run.jl) as a Julia subprocess.

    The driver writes states.h5, wells.h5, cell_indices.h5 and manifest.json
    under <case directory>/result/. Pass result_dir to override that location,
    e.g. when several decks share a directory.

    Parameters
    ----------
    case_path : str | Path
        Path to the .DATA file.
    result_dir : str | Path | None
        Output directory. Defaults to <case directory>/result/.
    timeout_s : int | None
        Subprocess timeout in seconds.

    Returns
    -------
    Path
        The model result directory.
    """
    case_path = Path(case_path)
    result_dir = Path(result_dir) if result_dir else case_path.parent / "result"
    if result_dir.exists():
        shutil.rmtree(result_dir)
    result_dir.mkdir(parents=True)

    script = Path(__file__).parents[2] / "bin" / "jutul_run.jl"
    argv = [os.environ.get("JULIA", "julia"),
            "--threads=auto",
            f"--project={script.parent}",
            str(script),
            f"--case={case_path}",
            f"--out={result_dir}",
            "--restart=none"]

    _stream_julia(argv, result_dir / "julia.log", timeout_s, script.name)
    return result_dir


def _stream_julia(argv, logpath, timeout_s, script_name):
    "Run a Julia subprocess, streaming stdout to the console and julia.log."
    deadline = None if timeout_s is None else time.monotonic() + timeout_s
    notebook = not hasattr(sys.stdout, "buffer")
    if notebook:
        from IPython.display import display
        output = display({"text/plain": ""}, raw=True, display_id=True)
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        stable, buf = [], ""
    with open(logpath, "wb") as log:
        p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)#pylint:disable=consider-using-with
        # Stream driver output (including the JutulDarcy progress bar) to the
        # console while keeping a copy in julia.log.
        while True:
            chunk = p.stdout.read1(4096)
            if not chunk:
                break
            log.write(chunk)
            if notebook:
                # Keep the driver's phase labels ([1/5] ..., "base NPV ...")
                # persistently, and collapse the repeating progress bars into a
                # single live trailing line. \r is an in-place redraw, so the
                # text after the last \r is a line's final state.
                buf = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", buf + decoder.decode(chunk))
                *complete, buf = buf.split("\n")
                for line in complete:
                    line = line.split("\r")[-1].strip()
                    if not line or line.startswith(("Progress ", "Reading ")):
                        continue
                    if not stable or stable[-1] != line:
                        stable.append(line)
                stable[:] = stable[-400:]
                live = buf.split("\r")[-1].strip()
                output.update(
                    {"text/plain": "\n".join(stable + ([live] if live else []))},
                    raw=True)
            else:
                sys.stdout.buffer.write(chunk)
                sys.stdout.flush()
            if deadline is not None and time.monotonic() > deadline:
                kill(p.pid)
                raise TimeoutError(f"{script_name} exceeded {timeout_s}s, see {logpath}")
        rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"{script_name} exited with code {rc}, see {logpath}")
    if notebook:
        output.update(
            {"text/plain": "\n".join(
                stable + [f"{script_name} completed; full output: {logpath}"])},
            raw=True)


def execute_julia_optimize(case_path, out_dir=None, *, params, timeout_s=None):
    """Run the forecast BHP optimization driver (jutul_optimize.jl).

    The driver writes optimal_bhp.csv, production.csv and summary.json under
    out_dir; consumers read them directly.

    Parameters
    ----------
    case_path : str | Path
        Path to the .DATA file.
    out_dir : str | Path | None
        Directory that receives the driver output (created if missing).
        Defaults to <case directory>/optimization/.
    params : dict
        Required CLI options: months, oil-price, gas-price, water-price,
        water-cost, gas-cost, discount-rate, bhp-prod-min/max and
        bhp-inj-min/max. history-cache is optional.

        Two are given in user-friendly units and converted here to the
        driver's raw units:
          - water-price: positive water-handling cost in $/m3 (negated, since
            produced water is a cost in npv_objective);
          - discount-rate: percent per year (divided by 100).
    timeout_s : int | None
        Subprocess timeout in seconds.

    Returns
    -------
    Path
        The out_dir.
    """
    required = ("months", "oil-price", "gas-price", "water-price", "water-cost",
                "gas-cost", "discount-rate", "bhp-prod-min", "bhp-prod-max",
                "bhp-inj-min", "bhp-inj-max")
    missing = [key for key in required if key not in params]
    if missing:
        raise ValueError(f"Missing required params: {', '.join(missing)}")

    params = dict(params)
    params["water-price"] = -abs(float(params["water-price"]))
    params["discount-rate"] = float(params["discount-rate"]) / 100.0

    case_path = Path(case_path)
    out_dir = Path(out_dir) if out_dir else case_path.parent / "optimization"
    out_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).parents[2] / "bin" / "jutul_optimize.jl"
    argv = [os.environ.get("JULIA", "julia"),
            "--threads=auto",
            f"--project={script.parent}",
            str(script),
            f"--case={case_path}",
            f"--out={out_dir}"]
    for key, value in params.items():
        argv.append(f"--{key}={value}")

    _stream_julia(argv, out_dir / "julia.log", timeout_s, script.name)
    return out_dir
