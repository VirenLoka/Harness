# Paste-ready system prompts

`dsh web` sends no system prompt. Mounting molbench puts the tools in the
agent's list; it does not tell the agent they exist, what they are for, or
that an unmeasured number is treated as false. A task pasted into the web UI
on its own reaches a model that answers from memory — which looks exactly
like a model that chose not to measure.

So when you drive a run by hand, the session needs two messages:

1. the system prompt from this directory, pasted first, on its own
2. the task, pasted second

| file | source | use for |
| --- | --- | --- |
| `protac_system.txt` | `protacbench.prompts.SYSTEM_PROMPT` | PROTAC linker repair and optimisation |
| `adc_system.txt` | `adclinker.prompts.SYSTEM_PROMPT` | ADC linker, payload and release tasks |

Both declare the `mcp__molbench__*` tools they need, the measurement
discipline rule, and the `REPAIRED:` contract line the graders parse.

## Regenerating

These are copies, so they drift. Rebuild both from source after any change to
either `prompts.py`:

```bash
python -c "
from pathlib import Path
from protacbench import prompts as p
from adclinker import prompts as a
Path('molbench/prompts/protac_system.txt').write_text(p.SYSTEM_PROMPT)
Path('molbench/prompts/adc_system.txt').write_text(a.SYSTEM_PROMPT)
"
```

`adc_system.txt` is generated: `adclinker.prompts` substitutes a measured
limits block into the template at import, so copying the raw triple-quoted
string out of the file by hand gives you one with `__LIMITS__` still in it.

## Not by hand

`python -m protacbench repair` sends the system prompt itself, one process per
task, and verifies each answer against the original fragments. Prefer it for
anything bigger than a one-off. These files are for the case where you want to
watch a single task run in the web UI.

## Checking the wiring

With the system prompt loaded, ask the agent:

> list every tool you have whose name starts with `mcp__molbench__`

Eleven is correct: `apply_smarts_reaction`, `check_adc_linker`,
`check_payload_release`, `check_protac_linker`, `generate_3d_conformer`,
`get_molecule_metrics`, `get_sbdd_metrics`, `prepare_target_receptor`,
`render_molecule_2d`, `run_docking_simulation`, `validate_smiles` — plus the
plugin's own `benchmark_transformation`, which is not namespaced.

Fewer than eleven means the server is up but stale. None at all should now be
impossible: `molbench.cordis.yml` sets `failOnStartupError: true`, so a server
that cannot start fails the launch instead of leaving the agent to improvise.
