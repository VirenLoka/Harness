"""One batch in the engine at a time, and every answer to its own prompt.

A GRPO step wedged on the H200 box: VRAM full, GPU at nothing, and

    error: engine returned 5 completions for 1 prompts
    RuntimeWarning: The executor did not finishing joining its threads
    within 300 seconds

`_flush_soon` cleared `_flushing` before draining the queue, so a turn that
arrived while the engine was busy started a SECOND flusher -- and two flushers
call `engine.generate` from two worker threads at once. vLLM's offline `LLM`
is not thread-safe, so the calls crossed and the threads wedged inside it.

It needed multi-turn episodes to show up. While every episode ended at turn
one, all eight of a group's turns arrived inside the same 20ms window, there
was one batch to flush, and nothing ever overlapped.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from moltrain.policy import BatchingGenerator  # noqa: E402


class CountingEngine:
    """Records the most calls that were ever inside it at once."""

    def __init__(self, delay: float = 0.05):
        self.delay = delay
        self.inside = 0
        self.most = 0
        self.calls: list[list[str]] = []
        self.guard = threading.Lock()

    def generate(self, prompts, _params):
        with self.guard:
            self.inside += 1
            self.most = max(self.most, self.inside)
            self.calls.append(list(prompts))
        time.sleep(self.delay)
        with self.guard:
            self.inside -= 1
        return [f"answer to {prompt}" for prompt in prompts]


class FailingOnce:
    """Fails the first batch, then works. The engine does raise sometimes."""

    def __init__(self):
        self.seen = 0

    def generate(self, prompts, _params):
        self.seen += 1
        if self.seen == 1:
            raise RuntimeError("engine went away")
        return [f"answer to {prompt}" for prompt in prompts]


def staggered(engine, first: int, later: int, gap: float, **kwargs):
    """`first` turns, then `later` more once the engine is already busy."""
    generator = BatchingGenerator(engine, params=None, window_s=0.01, **kwargs)

    async def run():
        early = [
            asyncio.create_task(generator.generate(f"p{i}")) for i in range(first)
        ]
        await asyncio.sleep(gap)
        late = [asyncio.create_task(generator.generate(f"q{i}")) for i in range(later)]
        return await asyncio.gather(*early, *late)

    return generator, asyncio.run(run())


def test_a_turn_arriving_mid_generate_does_not_enter_the_engine_twice():
    """The bug, directly. `most` was 2 before the drain was serialised."""
    engine = CountingEngine(delay=0.05)
    _, results = staggered(engine, first=3, later=3, gap=0.03, max_batch=8)
    assert engine.most == 1
    assert len(results) == 6


def test_every_answer_goes_to_its_own_prompt():
    """The crossing is what "5 completions for 1 prompts" was."""
    engine = CountingEngine(delay=0.04)
    _, results = staggered(engine, first=3, later=3, gap=0.02, max_batch=8)
    assert results == [f"answer to p{i}" for i in range(3)] + [
        f"answer to q{i}" for i in range(3)
    ]


def test_the_late_turns_are_not_stranded():
    """Clearing the flag at the wrong moment can also lose a request.

    If `_flushing` were cleared only after the drain but a turn could arrive
    between the final queue check and that clear, it would append, see the
    flag still set, start nobody, and wait forever.
    """
    engine = CountingEngine(delay=0.03)
    generator, results = staggered(engine, first=2, later=2, gap=0.02, max_batch=8)
    assert len(results) == 4
    assert sum(len(call) for call in engine.calls) == 4


def test_a_later_request_after_everything_settled_still_flushes():
    engine = CountingEngine(delay=0.01)
    generator = BatchingGenerator(engine, params=None, window_s=0.01, max_batch=8)

    async def run():
        first = await generator.generate("one")
        await asyncio.sleep(0.05)  # the flusher has exited by now
        second = await generator.generate("two")
        return first, second

    assert asyncio.run(run()) == ("answer to one", "answer to two")


def test_an_engine_failure_does_not_wedge_the_queue_behind_it():
    """Every waiter is told, and the next batch still goes out.

    Without the `finally`, an exception escaping the drain would leave
    `_flushing` set forever and every later turn would hang -- which looks
    exactly like the thing this file is about.
    """
    engine = FailingOnce()
    generator = BatchingGenerator(engine, params=None, window_s=0.01, max_batch=1)

    async def run():
        with pytest.raises(RuntimeError):
            await generator.generate("first")
        return await generator.generate("second")

    assert asyncio.run(run()) == "answer to second"


def test_a_group_that_arrives_together_is_still_one_batch():
    """The batching this class exists for has to survive the fix.

    Eight concurrent episodes asking for a turn inside the window cost one
    decode, not eight.
    """
    engine = CountingEngine(delay=0.01)
    generator = BatchingGenerator(engine, params=None, window_s=0.02, max_batch=8)

    async def run():
        return await asyncio.gather(
            *(generator.generate(f"p{i}") for i in range(8))
        )

    asyncio.run(run())
    assert engine.calls == [[f"p{i}" for i in range(8)]]
    assert generator.batches == [8]


def test_more_turns_than_the_batch_holds_are_split_not_dropped():
    engine = CountingEngine(delay=0.01)
    generator = BatchingGenerator(engine, params=None, window_s=0.02, max_batch=3)

    async def run():
        return await asyncio.gather(
            *(generator.generate(f"p{i}") for i in range(7))
        )

    results = asyncio.run(run())
    assert results == [f"answer to p{i}" for i in range(7)]
    assert generator.batches == [3, 3, 1]
    assert engine.most == 1


# ----------------------------------------------- the rule belongs to the engine


def test_the_engine_itself_cannot_be_entered_twice():
    """`BatchingGenerator` is not the only thing that could call it.

    vLLM's `LLM` is shared by every episode in a group, and entering it twice
    does not raise -- it returns one call's completions to the other. The
    invariant is the engine's, so it is enforced there and not left to
    whichever caller currently happens to respect it.
    """
    import types

    from moltrain.policy import VLLMEngine

    class FakeLLM:
        def __init__(self):
            self.inside = 0
            self.most = 0
            self.guard = threading.Lock()

        def generate(self, prompts, _params, **_kwargs):
            with self.guard:
                self.inside += 1
                self.most = max(self.most, self.inside)
            time.sleep(0.04)
            with self.guard:
                self.inside -= 1
            return [
                types.SimpleNamespace(
                    outputs=[types.SimpleNamespace(text=f"answer to {prompt}")]
                )
                for prompt in prompts
            ]

    llm = FakeLLM()
    engine = VLLMEngine(llm)
    threads = [
        threading.Thread(target=engine.generate, args=([f"p{i}"], None))
        for i in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert llm.most == 1


def test_the_engine_still_returns_the_text_for_each_prompt():
    """The lock must not change what comes back."""
    import types

    from moltrain.policy import VLLMEngine

    class FakeLLM:
        def generate(self, prompts, _params, **_kwargs):
            return [
                types.SimpleNamespace(
                    outputs=[types.SimpleNamespace(text=f"answer to {prompt}")]
                )
                for prompt in prompts
            ]

    engine = VLLMEngine(FakeLLM())
    assert engine.generate(["a", "b"], None) == ["answer to a", "answer to b"]
