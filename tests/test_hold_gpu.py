"""The parts of the GPU holder that can be checked without a GPU.

Which is most of it: what size was asked for, which devices, how much to take
and how it copes when the card has less free than it was told a moment ago.
The allocator itself is four ctypes calls and is exercised by running it.

Nothing here imports torch. `hold_gpu` only reaches for torch when libcudart
cannot be loaded, and a test that pulled it into the interpreter would break
protactrain's no-torch import invariant when the suites run together.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hold_gpu  # noqa: E402
from hold_gpu import (  # noqa: E402
    GIB,
    KIB,
    MIB,
    MIN_CHUNK,
    HoldError,
    human,
    parse_devices,
    parse_size,
    plan,
    take_memory,
)

CARD = 140 * GIB  # an H200 NVL, near enough


# ------------------------------------------------------------------- sizes


def test_the_units_people_actually_type():
    assert parse_size("80GiB") == 80 * GIB
    assert parse_size("80G") == 80 * GIB
    assert parse_size("80gib") == 80 * GIB
    assert parse_size("512MiB") == 512 * MIB
    assert parse_size("512m") == 512 * MIB
    assert parse_size("4KiB") == 4 * KIB
    assert parse_size("4096") == 4096


def test_the_b_suffixes_are_decimal_and_the_rest_are_binary():
    """nvidia-smi counts in binary; vendors quote decimal. Both get typed."""
    assert parse_size("1GB") == 1000**3
    assert parse_size("1GiB") == GIB
    assert parse_size("1G") == GIB


def test_a_fraction_is_of_the_whole_card():
    """Not of what is free, or the same command asks for a different amount
    on every run depending on who else is on the card."""
    assert plan(CARD, "50%", None, CARD) == CARD // 2
    assert plan(CARD, None, "10%", CARD) == CARD - CARD // 10


def test_a_percentage_with_nothing_to_be_a_percentage_of_is_refused():
    with pytest.raises(HoldError):
        parse_size("50%")


def test_sizes_that_are_not_sizes():
    for bad in ("", "   ", "lots", "80 parsecs", "-5GiB", "GiB", "110%"):
        with pytest.raises(HoldError):
            parse_size(bad, CARD)


def test_decimals_and_underscores_survive():
    assert parse_size("1.5GiB") == int(1.5 * GIB)
    assert parse_size("60_000") == 60000


def test_sizes_read_back_the_way_nvidia_smi_prints_them():
    assert human(80 * GIB) == "80.00 GiB"
    assert human(512 * MIB) == "512 MiB"
    assert human(10) == "10 B"


# ----------------------------------------------------------------- devices


def test_which_cards():
    assert parse_devices("0", 2) == [0]
    assert parse_devices("0,1", 2) == [0, 1]
    assert parse_devices("1, 0", 2) == [1, 0]
    assert parse_devices("all", 2) == [0, 1]


def test_a_card_named_twice_is_held_once():
    """Taking everything free twice on one device would hold nothing new."""
    assert parse_devices("0,0", 2) == [0]


def test_a_card_that_is_not_there_says_so(capsys):
    """Rather than a CUDA error code from four frames down."""
    with pytest.raises(HoldError) as exc:
        parse_devices("3", 2)
    assert "this host has 2" in str(exc.value)
    assert "CUDA_VISIBLE_DEVICES" in str(exc.value)


def test_nonsense_devices():
    for bad in ("", "x", "0,x"):
        with pytest.raises(HoldError):
            parse_devices(bad, 2)


# -------------------------------------------------------------- how much


def test_everything_free_by_default():
    assert plan(100 * GIB, None, None, CARD) == 100 * GIB


def test_leaving_room_for_a_debug_process():
    assert plan(100 * GIB, None, "40GiB", CARD) == 60 * GIB


def test_asking_for_more_than_is_free_is_refused():
    """The card is already someone else's; holding half of it silently would
    look like success right up until the real job failed."""
    with pytest.raises(HoldError) as exc:
        plan(10 * GIB, "60GiB", None, CARD)
    assert "only 10.00 GiB of" in str(exc.value)


def test_saying_it_both_ways_is_refused():
    with pytest.raises(HoldError):
        plan(100 * GIB, "60GiB", "40GiB", CARD)


def test_leaving_more_than_is_free_takes_nothing_rather_than_going_negative():
    assert plan(10 * GIB, None, "40GiB", CARD) == 0


# ------------------------------------------------------------- the taking


class FakeCard:
    """An allocator that runs out, so the chunking can be watched."""

    def __init__(self, capacity: int, max_block: int | None = None):
        self.capacity = capacity
        self.max_block = max_block
        self.taken = 0
        self.asked: list[int] = []

    def take(self, _device: int, nbytes: int) -> bool:
        self.asked.append(nbytes)
        if self.max_block is not None and nbytes > self.max_block:
            return False
        if self.taken + nbytes > self.capacity:
            return False
        self.taken += nbytes
        return True


def test_an_empty_card_gives_up_everything_asked_for():
    card = FakeCard(capacity=8 * GIB)
    assert take_memory(card, 0, 8 * GIB, chunk=GIB) == 8 * GIB
    assert card.asked == [GIB] * 8


def test_a_card_with_less_than_advertised_still_holds_what_it_can():
    """Free memory moves between being measured and being asked for.

    One request for everything would fail outright and hold nothing. In
    pieces, a late failure keeps everything taken before it.
    """
    card = FakeCard(capacity=5 * GIB + 512 * MIB)
    got = take_memory(card, 0, 8 * GIB, chunk=GIB)
    assert got == 5 * GIB + 512 * MIB
    assert card.taken == got


def test_the_tail_is_found_by_halving_not_by_crawling():
    """A loop stepping down by MIN_CHUNK would make hundreds of calls."""
    card = FakeCard(capacity=GIB + 96 * MIB)
    take_memory(card, 0, 8 * GIB, chunk=GIB)
    assert len(card.asked) < 16


def test_a_full_card_holds_nothing_rather_than_looping():
    card = FakeCard(capacity=0)
    assert take_memory(card, 0, 8 * GIB, chunk=GIB) == 0


def test_nothing_asked_for_is_not_an_allocation():
    card = FakeCard(capacity=8 * GIB)
    assert take_memory(card, 0, 0, chunk=GIB) == 0
    assert card.asked == []


def test_a_target_smaller_than_a_chunk_is_not_rounded_up():
    """Asking for 60GiB must not take 64."""
    card = FakeCard(capacity=8 * GIB)
    assert take_memory(card, 0, 100 * MIB, chunk=GIB) == 100 * MIB


def test_a_card_that_refuses_large_blocks_is_still_held():
    """Fragmentation: lots free, none of it contiguous in big pieces."""
    card = FakeCard(capacity=4 * GIB, max_block=64 * MIB)
    got = take_memory(card, 0, 4 * GIB, chunk=GIB)
    assert got > 0
    assert max(card.asked) == GIB  # it did try big first


def test_the_remainder_never_gets_smaller_than_it_is_worth():
    card = FakeCard(capacity=0)
    take_memory(card, 0, 8 * GIB, chunk=GIB)
    assert min(card.asked) >= MIN_CHUNK


# ------------------------------------------------------- no GPU, no script


def test_without_a_gpu_it_says_so_and_stops(monkeypatch, capsys):
    """This machine is one; so is anywhere the drivers are missing."""
    monkeypatch.setattr(hold_gpu.CudaRuntime, "find", classmethod(lambda cls: None))
    monkeypatch.setattr(hold_gpu.TorchRuntime, "find", classmethod(lambda cls: None))

    assert hold_gpu.main([]) == 2
    message = capsys.readouterr().err
    assert "no CUDA here" in message


def test_a_runtime_reporting_no_devices_is_refused(monkeypatch, capsys):
    """A loaded libcudart and an empty CUDA_VISIBLE_DEVICES looks like this."""

    class Empty:
        def device_count(self):
            return 0

    monkeypatch.setattr(hold_gpu.CudaRuntime, "find", classmethod(lambda cls: Empty()))
    assert hold_gpu.main([]) == 2
    assert "lists no devices" in capsys.readouterr().err


def test_a_bad_size_is_reported_before_anything_is_allocated(monkeypatch, capsys):
    taken: list[int] = []

    class Card:
        def device_count(self):
            return 1

        def free_total(self, _device):
            return 100 * GIB, CARD

        def take(self, _device, nbytes):
            taken.append(nbytes)
            return True

        def release(self):
            pass

    monkeypatch.setattr(hold_gpu.CudaRuntime, "find", classmethod(lambda cls: Card()))
    assert hold_gpu.main(["--size", "banana"]) == 2
    assert not taken


# ------------------------------------- a library that loads is not one that works
#
# A libcudart newer than the installed driver loads perfectly and then answers
# every call with cudaErrorInsufficientDriver (35). Committing to the first
# .so with the right name meant exiting 2 on a box whose cards were idle,
# blaming CUDA_VISIBLE_DEVICES, and never reaching the torch fallback that
# would have worked -- torch carries its own libcudart of its own major.


class FakeLib:
    """Enough of libcudart to be loaded and probed."""

    def __init__(self, status: int = 0, count: int = 2):
        self.status = status
        self.count = count
        for name in (
            "cudaGetDeviceCount",
            "cudaSetDevice",
            "cudaMalloc",
            "cudaFree",
            "cudaMemGetInfo",
        ):
            setattr(self, name, _Call())

    def __getattr__(self, name):  # argtypes assignment on an unknown symbol
        raise AttributeError(name)


class _Call:
    argtypes = None
    restype = None

    def __call__(self, *_args):
        return 0


def _probe_returns(monkeypatch, statuses):
    """Load a sequence of libraries whose probes return these statuses."""
    loaded: list[str] = []
    queue = list(statuses)

    def fake_cdll(name):
        loaded.append(name)
        if not queue:
            raise OSError("no such library")
        return FakeLib()

    def fake_probe(self):
        return queue.pop(0), 2

    monkeypatch.setattr(hold_gpu.ctypes, "CDLL", fake_cdll)
    monkeypatch.setattr(hold_gpu.CudaRuntime, "probe", fake_probe)
    monkeypatch.setattr(hold_gpu.ctypes.util, "find_library", lambda _n: None)
    monkeypatch.setattr(hold_gpu.os.environ, "get", lambda *a: None)
    return loaded


def test_a_working_library_is_taken(monkeypatch):
    _probe_returns(monkeypatch, [0])
    assert hold_gpu.CudaRuntime.find() is not None


def test_a_library_that_loads_but_cannot_talk_to_the_driver_is_skipped(monkeypatch):
    """cudaErrorInsufficientDriver is 35, not an empty device list."""
    loaded = _probe_returns(monkeypatch, [35, 0])
    found = hold_gpu.CudaRuntime.find()
    assert found is not None
    assert found.source == loaded[1]
    assert "returned 35" in hold_gpu.CudaRuntime.rejected[0]


def test_when_no_library_works_torch_is_still_reached(monkeypatch):
    """The fallback the docstring promises, in the case it was written for."""
    _probe_returns(monkeypatch, [35, 3])

    class Torch:
        def device_count(self):
            return 2

    monkeypatch.setattr(
        hold_gpu.TorchRuntime, "find", classmethod(lambda cls: Torch())
    )
    assert isinstance(hold_gpu.load_runtime(), Torch)


def test_the_refusal_names_the_libraries_it_tried(monkeypatch, capsys):
    _probe_returns(monkeypatch, [35])
    monkeypatch.setattr(hold_gpu.TorchRuntime, "find", classmethod(lambda cls: None))
    assert hold_gpu.main([]) == 2
    assert "returned 35" in capsys.readouterr().err


# ------------------------------------------------------- the leftover release file


def test_a_release_file_that_already_exists_is_refused(tmp_path, capsys):
    """One successful use leaves the file behind, and the next identical run
    would allocate, release on its first poll and hold nothing -- while
    looking exactly like a run that worked."""
    stale = tmp_path / "go"
    stale.write_text("")
    assert hold_gpu.main(["--release-file", str(stale)]) == 2
    message = capsys.readouterr().err
    assert str(stale) in message
    assert "hold nothing" in message


def test_the_stale_check_happens_before_any_cuda(tmp_path, monkeypatch, capsys):
    """It costs nothing, and a CUDA context is not free."""
    loaded: list[str] = []
    monkeypatch.setattr(
        hold_gpu.CudaRuntime,
        "find",
        classmethod(lambda cls: loaded.append("cuda") or None),
    )
    stale = tmp_path / "go"
    stale.write_text("")
    assert hold_gpu.main(["--release-file", str(stale)]) == 2
    assert not loaded


# ------------------------------------------------------------- --then and its --


def test_everything_after_then_is_the_command():
    head, command = hold_gpu.split_then(["--device", "0", "--then", "python", "x.py"])
    assert head == ["--device", "0"]
    assert command == ["python", "x.py"]


def test_a_separator_inside_the_command_survives():
    """argparse with REMAINDER ate the `--` and then refused the invocation,
    blaming arguments the user believes belong to their own command."""
    _, command = hold_gpu.split_then(
        ["--then", "uv", "run", "--", "python", "-m", "foo", "--lr", "1e-5"]
    )
    assert command == ["uv", "run", "--", "python", "-m", "foo", "--lr", "1e-5"]


def test_the_commands_own_options_are_not_this_scripts():
    _, command = hold_gpu.split_then(["--then", "cmd", "--device", "7"])
    assert command == ["cmd", "--device", "7"]


def test_no_then_is_not_an_empty_command():
    """None and [] mean different things: not asked for, versus asked for and
    left empty."""
    head, command = hold_gpu.split_then(["--device", "0"])
    assert command is None
    assert head == ["--device", "0"]


def test_then_with_nothing_after_it_is_refused(capsys):
    assert hold_gpu.split_then(["--then"])[1] == []
    assert hold_gpu.main(["--then"]) == 2
    assert "no command" in capsys.readouterr().err


# ----------------------------------------------------------- small asks


def test_a_target_below_the_chunk_floor_is_still_held():
    """`--size 16MiB` held nothing and was reported as a full card."""
    card = FakeCard(capacity=8 * GIB)
    assert take_memory(card, 0, 16 * MIB, chunk=GIB) == 16 * MIB


def test_the_floor_still_stops_the_halving_crawling_to_zero():
    card = FakeCard(capacity=0)
    take_memory(card, 0, 8 * GIB, chunk=GIB)
    assert min(card.asked) >= MIN_CHUNK
