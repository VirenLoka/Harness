#!/usr/bin/env python3
"""Hold a GPU's memory so nothing else takes it while you set something up.

On a shared box the card is claimed by whoever allocates first, and a vLLM
engine spends a minute or two loading weights before it asks for its KV cache.
That minute is a window: another job starts, takes what it wants, and the
engine fails at the end of a long load rather than at the start of a short
one. This parks a process on the card and holds the memory through that
window.

Usage:
    ./hold_gpu.py                      # take everything free on device 0
    ./hold_gpu.py --device 0,1         # both cards
    ./hold_gpu.py --size 60GiB         # leave the rest for a debug process
    ./hold_gpu.py --leave 48GiB        # the same thing from the other end
    ./hold_gpu.py --release-file /tmp/go
    ./hold_gpu.py --release-file /tmp/go --then python -m protactrain.grpo check

Ctrl-C (or SIGTERM) frees it and exits. With `--release-file`, creating that
file does the same -- which is how another shell, or a script, hands the card
over without needing this one's pid. With `--then`, the command runs in this
process the instant the memory is freed, which is the smallest handover gap
available.

THE HANDOVER IS NOT ATOMIC. Nothing can make it so: CUDA has no way to pass a
reservation from one process to another, so between this process freeing and
the next one allocating there is a window where the card is open. `--then`
makes that window milliseconds instead of however long it takes you to type
the next command. It does not close it.

Deliberately importing neither torch nor anything from this repository at
startup. `import torch` alone is several seconds, and the whole point is to be
holding memory before the thing you are racing gets there. The CUDA runtime is
called through ctypes, which is a library load and a context -- about a second
on a warm box. torch is the fallback for when libcudart cannot be found, and
is imported only then.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import glob
import os
import signal
import sys
import time
from typing import Any

KIB = 1024
MIB = 1024 * KIB
GIB = 1024 * MIB

# Allocated in pieces rather than one block. A single cudaMalloc for everything
# free fails outright when the driver has taken a little more since it was
# measured, and failing means holding nothing; in pieces, a late failure still
# leaves everything before it held.
DEFAULT_CHUNK = 512 * MIB
# Below this a chunk is not worth the call: the remainder is rounding, and a
# loop that keeps halving towards zero takes longer than the memory is worth.
MIN_CHUNK = 32 * MIB

# cudaMalloc returning anything else means the memory was not taken.
CUDA_SUCCESS = 0

# Suffixes. The *iB forms and the bare letters are binary, as the drivers and
# nvidia-smi report; the *B forms are decimal, as disks and vendors quote.
UNITS = {
    "": 1,
    "b": 1,
    "k": KIB, "kib": KIB, "kb": 1000,
    "m": MIB, "mib": MIB, "mb": 1000**2,
    "g": GIB, "gib": GIB, "gb": 1000**3,
    "t": 1024 * GIB, "tib": 1024 * GIB, "tb": 1000**4,
}


class HoldError(Exception):
    """Something the user needs to read, not a traceback."""


# ------------------------------------------------------------------ parsing


def parse_size(text: str, whole: int | None = None) -> int:
    """`80GiB`, `80G`, `80gb`, `75%` or a bare byte count, as bytes.

    A percentage needs something to be a percentage OF, so `whole` is required
    for it and refused rather than guessed at.
    """
    cleaned = str(text).strip().lower().replace("_", "").replace(" ", "")
    if not cleaned:
        raise HoldError("empty size")

    if cleaned.endswith("%"):
        if whole is None:
            raise HoldError(f"{text!r} is a percentage, but of what is not known here")
        try:
            fraction = float(cleaned[:-1])
        except ValueError:
            raise HoldError(f"{text!r} is not a number of percent") from None
        if not 0 <= fraction <= 100:
            raise HoldError(f"{text!r} is not between 0% and 100%")
        return int(whole * fraction / 100)

    digits = cleaned
    suffix = ""
    while digits and not (digits[-1].isdigit() or digits[-1] == "."):
        suffix = digits[-1] + suffix
        digits = digits[:-1]
    if suffix not in UNITS:
        raise HoldError(
            f"{text!r} has no unit I know; use bytes, KiB/MiB/GiB/TiB, "
            "KB/MB/GB/TB, or a percentage"
        )
    try:
        amount = float(digits)
    except ValueError:
        raise HoldError(f"{text!r} is not a size") from None
    if amount < 0:
        raise HoldError(f"{text!r} is negative")
    return int(amount * UNITS[suffix])


def parse_devices(text: str, count: int) -> list[int]:
    """`0`, `0,1` or `all`, checked against the devices that exist.

    Checked here because asking for a card that is not there should say so,
    rather than being a CUDA error code from four frames down.
    """
    cleaned = str(text).strip().lower()
    if cleaned in ("all", "*"):
        return list(range(count))
    devices: list[int] = []
    for piece in cleaned.split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            index = int(piece)
        except ValueError:
            raise HoldError(f"{piece!r} is not a device number") from None
        if not 0 <= index < count:
            raise HoldError(
                f"device {index} does not exist; this host has {count} "
                f"({', '.join(str(i) for i in range(count)) or 'none'}). "
                "CUDA_VISIBLE_DEVICES renumbers them, so check it too."
            )
        if index not in devices:
            devices.append(index)
    if not devices:
        raise HoldError("no devices named")
    return devices


def human(nbytes: int) -> str:
    """Bytes as the size a person would say, matching nvidia-smi's units."""
    if nbytes >= GIB:
        return f"{nbytes / GIB:.2f} GiB"
    if nbytes >= MIB:
        return f"{nbytes / MIB:.0f} MiB"
    return f"{nbytes} B"


# ------------------------------------------------------------- the allocator


class CudaRuntime:
    """The four calls this needs, through ctypes rather than torch."""

    # Where libcudart turns up when it is not on the loader path. A conda
    # environment keeps its own, and torch ships one regardless.
    # Filled in by find() with every candidate that loaded and then failed,
    # so the refusal can say which libraries were tried and why each lost.
    rejected: list[str] = []

    NAMES = (
        "libcudart.so",
        "libcudart.so.13",
        "libcudart.so.12",
        "libcudart.so.11.0",
    )

    def __init__(self, lib: Any):
        self.lib = lib
        self.source = "?"
        self.blocks: list[tuple[int, ctypes.c_void_p]] = []
        lib.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        lib.cudaSetDevice.argtypes = [ctypes.c_int]
        lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        lib.cudaFree.argtypes = [ctypes.c_void_p]
        lib.cudaMemGetInfo.argtypes = [
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        ]

    def probe(self) -> tuple[int, int]:
        """(status, device count) from cudaGetDeviceCount, keeping the status.

        The status is the whole point. A libcudart newer than the installed
        driver loads perfectly and then answers every call with
        cudaErrorInsufficientDriver (35), and a runtime that cannot reach the
        driver at all answers cudaErrorInitializationError (3). Collapsing
        either to "0 devices" reports a CUDA_VISIBLE_DEVICES problem on a box
        whose cards are idle and whose next candidate would have worked.
        """
        count = ctypes.c_int(0)
        status = self.lib.cudaGetDeviceCount(ctypes.byref(count))
        return int(status), int(count.value)

    @classmethod
    def find(cls) -> CudaRuntime | None:
        candidates: list[str] = []
        located = ctypes.util.find_library("cudart")
        if located:
            candidates.append(located)
        candidates.extend(cls.NAMES)
        prefix = os.environ.get("CONDA_PREFIX")
        if prefix:
            # Newest last-sorted first: a box with two CUDA majors installed
            # should get the one the rest of the environment was built for,
            # and that is almost always the higher number.
            for pattern in (
                f"{prefix}/lib/libcudart.so*",
                f"{prefix}/lib/python*/site-packages/torch/lib/libcudart*.so*",
                f"{prefix}/lib/python*/site-packages/nvidia/*/lib/libcudart*.so*",
            ):
                candidates.extend(sorted(glob.glob(pattern), reverse=True))
        # Each candidate is tried until one WORKS, not until one loads. There
        # is no portable dlclose through ctypes, so a rejected library stays
        # mapped -- which costs address space and nothing else, and is a far
        # better trade than committing to the first .so with the right name.
        cls.rejected = []
        for name in candidates:
            try:
                runtime = cls(ctypes.CDLL(name))
            except OSError:
                continue
            except AttributeError as exc:
                # It loaded and is not libcudart: the argtypes above look the
                # symbols up by name. Defensive rather than observed.
                cls.rejected.append(f"{name}: not a CUDA runtime ({exc})")
                continue
            status, _count = runtime.probe()
            if status == CUDA_SUCCESS:
                runtime.source = name
                return runtime
            cls.rejected.append(f"{name}: cudaGetDeviceCount returned {status}")
        return None

    def device_count(self) -> int:
        count = ctypes.c_int(0)
        if self.lib.cudaGetDeviceCount(ctypes.byref(count)) != CUDA_SUCCESS:
            return 0
        return int(count.value)

    def select(self, device: int) -> None:
        if self.lib.cudaSetDevice(ctypes.c_int(device)) != CUDA_SUCCESS:
            raise HoldError(f"could not select device {device}")

    def free_total(self, device: int) -> tuple[int, int]:
        self.select(device)
        free = ctypes.c_size_t(0)
        total = ctypes.c_size_t(0)
        if self.lib.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)) != CUDA_SUCCESS:
            raise HoldError(f"could not read memory on device {device}")
        return int(free.value), int(total.value)

    def take(self, device: int, nbytes: int) -> bool:
        self.select(device)
        pointer = ctypes.c_void_p()
        status = self.lib.cudaMalloc(ctypes.byref(pointer), ctypes.c_size_t(nbytes))
        if status != CUDA_SUCCESS:
            return False
        self.blocks.append((device, pointer))
        return True

    def release(self) -> None:
        while self.blocks:
            device, pointer = self.blocks.pop()
            try:
                self.select(device)
                self.lib.cudaFree(pointer)
            except HoldError:
                # Exiting frees it either way; a failure here must not stop
                # the rest of the blocks being freed.
                continue


class TorchRuntime:
    """The fallback, for a box where libcudart cannot be found.

    Several seconds slower to start, which on a card someone else is about to
    take is the difference this script exists to avoid -- so it is the
    fallback and never the first choice.
    """

    def __init__(self, torch: Any):
        self.torch = torch
        self.blocks: list[Any] = []

    @classmethod
    def find(cls) -> TorchRuntime | None:
        try:
            import torch
        except ImportError:
            return None
        if not torch.cuda.is_available():
            return None
        return cls(torch)

    def device_count(self) -> int:
        return int(self.torch.cuda.device_count())

    def free_total(self, device: int) -> tuple[int, int]:
        free, total = self.torch.cuda.mem_get_info(device)
        return int(free), int(total)

    def take(self, device: int, nbytes: int) -> bool:
        try:
            self.blocks.append(
                self.torch.empty(
                    nbytes, dtype=self.torch.uint8, device=f"cuda:{device}"
                )
            )
        except RuntimeError:  # torch reports an OOM as one of these
            return False
        return True

    def release(self) -> None:
        self.blocks.clear()
        # torch's caching allocator keeps freed blocks for itself, so dropping
        # the tensors is not enough: without this the memory is still held as
        # far as every other process on the card is concerned.
        self.torch.cuda.empty_cache()


def load_runtime(prefer_torch: bool = False) -> Any:
    runtime = None if prefer_torch else CudaRuntime.find()
    if runtime is None:
        runtime = TorchRuntime.find()
    if runtime is None:
        tried = "; ".join(CudaRuntime.rejected)
        raise HoldError(
            "no CUDA here: no usable libcudart was found and torch reports no "
            "device. This script only does something on a machine with a GPU."
            + (f"\n  libraries tried: {tried}" if tried else "")
        )
    if runtime.device_count() < 1:
        # Reached only once a runtime has answered cudaGetDeviceCount
        # successfully, so this really is an empty device list rather than a
        # driver that could not be spoken to.
        raise HoldError(
            "the CUDA runtime answered but lists no devices. Check "
            "CUDA_VISIBLE_DEVICES, and that nvidia-smi sees the card."
        )
    return runtime


# ---------------------------------------------------------------- the taking


def take_memory(runtime: Any, device: int, target: int, chunk: int = DEFAULT_CHUNK) -> int:
    """Take up to `target` bytes on `device`, in pieces. Returns what it got.

    Pieces, and then smaller pieces, because free memory is a moving number:
    between measuring it and asking for it the driver or another process may
    have taken some. One large request would simply fail. This holds what it
    can and reports the difference rather than pretending it got everything.
    """
    # A target below MIN_CHUNK is still worth one allocation. The floor exists
    # to stop the halving crawling towards zero, not to refuse a small ask --
    # without this, `--size 16MiB` holds nothing and is then reported as a
    # card that is already full.
    floor = min(MIN_CHUNK, target) if target else MIN_CHUNK
    taken = 0
    size = min(chunk, target) if target else 0
    while taken < target and size >= floor:
        size = min(size, target - taken)
        if runtime.take(device, size):
            taken += size
            continue
        # Whatever is left is smaller than this piece, or gone. Halving finds
        # the edge in a few calls instead of one per MIN_CHUNK.
        size //= 2
    return taken


def plan(free: int, size: str | None, leave: str | None, total: int) -> int:
    """How many bytes to ask for, from whichever end the user described it.

    `--size` is what to take, `--leave` is what not to take, and a percentage
    in either is of the card's total -- not of what happens to be free, which
    would mean the same command asking for a different amount every run.
    """
    if size is not None and leave is not None:
        raise HoldError("--size and --leave say the same thing twice; pick one")
    if size is not None:
        wanted = parse_size(size, total)
        if wanted > free:
            raise HoldError(
                f"asked for {human(wanted)} but only {human(free)} of "
                f"{human(total)} is free. A card never has quite all of its "
                "memory free -- the driver and any context on it hold some -- "
                "and another process may be holding the rest. Lower --size, "
                "or use --leave to say what to keep back instead."
            )
        return wanted
    spare = parse_size(leave, total) if leave is not None else 0
    return max(0, free - spare)


# ------------------------------------------------------------------ the wait


def wait_for_release(release_file: str | None, poll_s: float = 0.25) -> str:
    """Block until told to let go. Returns why, for the closing message."""
    reason: dict[str, str] = {}

    def stop(signum: int, _frame: Any) -> None:
        reason["why"] = signal.Signals(signum).name
        raise KeyboardInterrupt

    previous = {
        number: signal.signal(number, stop)
        for number in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        while True:
            if release_file and os.path.exists(release_file):
                return f"{release_file} appeared"
            time.sleep(poll_s)
    except KeyboardInterrupt:
        return reason.get("why", "SIGINT")
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


# ---------------------------------------------------------------------- main


def split_then(argv: list[str]) -> tuple[list[str], list[str] | None]:
    """Everything after `--then` is the command, taken verbatim.

    argparse cannot do this. With `nargs=REMAINDER` it consumes the first
    `--` itself and reports what follows as unrecognised, so
    `--then uv run -- python x.py` is refused outright with a complaint about
    arguments the user believes belong to their own command. Splitting argv
    first means the command is passed through exactly as typed, `--` and all.

    Returns (what argparse should see, the command). The command is None when
    `--then` was not given, and `[]` when it was given with nothing after it
    -- which is an error, not a silent no-op.
    """
    if "--then" not in argv:
        return list(argv), None
    cut = argv.index("--then")
    return list(argv[:cut]), list(argv[cut + 1 :])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="hold_gpu.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--device", default="0", help="device number, a comma list, or `all` (default: 0)"
    )
    parser.add_argument("--size", help="how much to take, e.g. 60GiB or 40%% of the card")
    parser.add_argument("--leave", help="how much NOT to take; the opposite of --size")
    parser.add_argument(
        "--release-file",
        help="also let go when this path appears, so another shell can hand "
        "the card over without knowing this process's pid",
    )
    # Declared only so it appears in --help: `split_then` has already taken
    # it and everything after it out of argv before this parser runs.
    parser.add_argument(
        "--then",
        action="store_true",
        help="run a command in place of this process once the memory is "
        "freed. Everything after this flag is that command, verbatim, so its "
        "own options and a `--` separator are passed through untouched",
    )
    parser.add_argument(
        "--chunk", default=None, help=f"allocation piece size (default: {human(DEFAULT_CHUNK)})"
    )
    parser.add_argument(
        "--use-torch",
        action="store_true",
        help="allocate through torch instead of libcudart (slower to start)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    head, command = split_then(raw)
    args = parse_args(head)
    try:
        if command is not None and not command:
            raise HoldError("--then was given no command to run")
        # Before the runtime loads, because it costs nothing and because a
        # leftover file is the difference between holding the card and
        # holding nothing. Nothing here ever unlinks the path: a release file
        # is operator state, and removing it would race the shell that is
        # about to create it.
        if args.release_file and os.path.exists(args.release_file):
            raise HoldError(
                f"{args.release_file} already exists, so this would release "
                "on its first poll and hold nothing at all. Remove it first, "
                "or name a path that does not exist yet."
            )
        # Checked before the runtime loads: it needs nothing from the card,
        # and a typo should not cost a CUDA context first.
        if args.size is not None and args.leave is not None:
            raise HoldError("--size and --leave say the same thing twice; pick one")
        runtime = load_runtime(prefer_torch=args.use_torch)
        devices = parse_devices(args.device, runtime.device_count())
        chunk = parse_size(args.chunk) if args.chunk else DEFAULT_CHUNK
        if chunk < MIN_CHUNK:
            raise HoldError(f"--chunk below {human(MIN_CHUNK)} is not worth the calls")
    except HoldError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    held = 0
    try:
        for device in devices:
            free, total = runtime.free_total(device)
            target = plan(free, args.size, args.leave, total)
            got = take_memory(runtime, device, target, chunk)
            held += got
            short = target - got
            print(
                f"device {device}: holding {human(got)} of {human(total)} "
                f"({human(free)} was free)"
                + (f" -- {human(short)} short, taken since it was measured" if short >= MIN_CHUNK else "")
            )
    except HoldError as exc:
        runtime.release()
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not held:
        runtime.release()
        print(
            "error: nothing was held. Either the card is already full, or "
            "--leave covered everything that was free.",
            file=sys.stderr,
        )
        return 1

    print(
        f"\nholding {human(held)} across {len(devices)} device(s), "
        f"pid {os.getpid()}"
    )
    print("release with Ctrl-C, or: kill %d" % os.getpid())
    if args.release_file:
        print(f"        or: touch {args.release_file}")
    print(
        "\nNote: freeing and the next process allocating are not one step, so "
        "the card is open in between."
    )

    why = wait_for_release(args.release_file)
    runtime.release()
    print(f"\nreleased ({why}).")

    if command:
        # No filtering: a `--` inside the command belongs to the command.
        # exec, not a subprocess: this process's CUDA context goes away with
        # its address space, and the command starts in the same pid with the
        # smallest gap this can manage.
        print(f"running: {' '.join(command)}\n", flush=True)
        try:
            os.execvp(command[0], command)
        except OSError as exc:
            print(f"error: could not run {command[0]}: {exc}", file=sys.stderr)
            return 127
    return 0


if __name__ == "__main__":
    sys.exit(main())
