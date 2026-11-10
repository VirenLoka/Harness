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
    assert "only 10.00 GiB is free" in str(exc.value)


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
    assert "reports no devices" in capsys.readouterr().err


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
