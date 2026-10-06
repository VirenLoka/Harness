# MolBench

A local pipeline for molecule design. An agent can apply a SMARTS
transformation to a benchmark SMILES string and see what it does to RDKit's
Synthetic Accessibility (SA) score, dock a molecule into a protein target and
get the binding affinity with medicinal-chemistry efficiency metrics, and look
at both the 2D change (inline in the conversation) and the 3D pose (in a
browser).

```
DeepSeek Harness
├── molbench-harness-plugin  (TypeScript, Cordis)
│     ├── mounts @deepseek-ai/dsh-mcp-client ──► Python MCP server (stdio)
│     └── registers benchmark_transformation on ctx.tools
└── tools the model sees
      ├── benchmark_transformation          (composite, from the plugin)
      ├── mcp__molbench__get_molecule_metrics
      ├── mcp__molbench__validate_smiles
      ├── mcp__molbench__apply_smarts_reaction
      └── mcp__molbench__render_molecule_2d
```

| Path | What it is |
| --- | --- |
| `server.py` | FastMCP server: the eight tools |
| `docking.py` | Receptor prep, conformers, engine wrappers, metrics, 3D viewer |
| `pyproject.toml` | Python dependencies (RDKit, FastMCP, Meeko) |
| `tests/test_server.py` | 2D tool tests through an in-process MCP client |
| `tests/test_docking.py` | Unit tests plus real docking against 3PTB |
| `plugin/src/index.ts` | Cordis plugin: connection plus two composite tools |
| `plugin/tests/integration.test.mjs` | Plugin tests against the real server over stdio |
| `molbench.cordis.yml` | Harness overlay that loads the plugin |
| `visualizations/` | Saved SVG depictions (created on first render) |
| `receptors/`, `docking/` | Prepared receptors, docked poses, complexes, 3D viewers |
| `.conda-dock/` | Docking engine binaries (vina, smina, Open Babel) |

## Setup

Python side:

```bash
uv sync --extra dev
```

Plugin side:

```bash
cd plugin && npm install && npm run build
```

Docking engines (skip if you only want the 2D tools):

```bash
conda create -y -p .conda-dock -c conda-forge smina vina
```

That brings AutoDock Vina, smina and Open Babel into the project without
touching your global environment, and works on Apple Silicon, where the pip
`vina` package has no wheel. The server finds them automatically; override with
`MOLBENCH_VINA_BIN`, `MOLBENCH_SMINA_BIN`, `MOLBENCH_OBABEL_BIN` or
`MOLBENCH_HADDOCK3_BIN` if they live elsewhere.

## Tools

The four Python tools are the primitives:

| Tool | Returns |
| --- | --- |
| `get_molecule_metrics(smiles)` | SA score (1 easy to 10 hard), exact molecular weight, QED, canonical SMILES, heavy-atom count, formula |
| `validate_smiles(smiles)` | `valid`, canonical SMILES, and `errors`: the lines RDKit itself logged, so a bad string explains itself |
| `apply_smarts_reaction(smiles, reaction_smarts)` | Valid products, deduplicated by canonical SMILES, plus anything that failed sanitization in `rejected` |
| `render_molecule_2d(smiles, filename_prefix, highlight_smarts?, width?, height?)` | An inline PNG depiction, and the absolute path of an SVG written to `visualizations/` |

### Structure-based (3D)

| Tool | Returns |
| --- | --- |
| `prepare_target_receptor(pdb_id? , local_pdb_path?, binding_site_center?, box_size?, ligand_resname?, keep_hetatms?, refresh?)` | A rigid `.pdbqt` with waters and ligands stripped, hydrogens added at pH 7.4 and Gasteiger charges assigned, plus the confirmed box. Pass `ligand_resname` to centre the box on a co-crystal ligand instead of looking up coordinates. Cached, since protonation is the slow step |
| `generate_3d_conformer(smiles, seed?, num_conformers?)` | The lowest-energy ETKDGv3 conformer as SDF after MMFF94 optimization, with `local_strain` and `ensemble_strain` in kcal/mol |
| `run_docking_simulation(smiles, receptor_file, engine?, binding_site_center?, box_size?, exhaustiveness?, num_modes?, seed?, label?, timeout_s?)` | Best-pose binding affinity in kcal/mol, every pose's score, the pose SDF, the docked complex PDB, an interactive 3D viewer, and the pose's strain above its relaxed minimum |
| `get_sbdd_metrics(smiles, docking_score)` | `Binding_Affinity`, `Ligand_Efficiency`, `Lipophilic_Ligand_Efficiency`, `cLogP`, `TPSA`, plus `estimated_pIC50` and heavy-atom count |

The box is recorded next to the receptor, so `run_docking_simulation` reuses it
without being told twice. Engines run as subprocesses through
`asyncio.create_subprocess_exec`, and RDKit's CPU-bound work goes to a worker
thread, so a slow dock never blocks the server. Intermediate SDF, PDB and PDBQT
files live in a `tempfile.TemporaryDirectory`; only the pose, complex and
viewer are kept.

### Composite tools

`benchmark_transformation(smiles, reaction_smarts, render?, label?)` from the
plugin runs the whole loop in one call: validate, score the starting material,
apply the reaction, score each product, and report `sa_delta`, `qed_delta` and
`mw_delta` per product, sorted with the most negative `sa_delta` (predicted
easiest to make) first. It also attaches depictions of the starting material —
with the matched motif highlighted — and of each product.

Real output, methylating aspirin's carboxylic acid:

```
[C:1](=[O:2])[OH]>>[C:1](=[O:2])OC applied to CC(=O)Oc1ccccc1C(=O)O
start: SA 1.58 | QED 0.5501 | MW 180.0423 | C9H8O4
product COC(=O)c1ccccc1OC(C)=O: SA 1.6009 (+0.0209, harder) | QED 0.5271 (-0.023) | MW 194.0579 (+14.0156)
```

A reminder of how coarse the heuristic is: fluorinating benzene
(`[cH:1]>>[c:1]F`) leaves the SA score at 1, an unchanged delta, because both
molecules sit at the bottom of the scale.

`dock_and_score(smiles, receptor_file?, engine?, exhaustiveness?, render?, label?)`
does the same for structure-based work: validate, dock, then report the 3D and
2D pictures together.

```
N=C(N)c1ccccc1 docked with vina: -6.081 kcal/mol
efficiency: LE -0.6757 (target < -0.3) | LLE 3.4875 | estimated pIC50 4.4582
properties: cLogP 0.9707 | TPSA 49.87 | SA 1.4737 | QED 0.4208 | MW 120.0687 | 9 heavy atoms
pose strain: 0.3057 kcal/mol above its relaxed minimum
3D viewer: .../docking/node_bzd_..._pose.html
```

## Wiring it into the harness

Build the plugin, then start the harness with the overlay:

```bash
npx @deepseek-ai/dsh@0.2.0-rc.2 web --patch /Users/virenloka/Boot/harness/molbench/molbench.cordis.yml
```

`molbench.cordis.yml` loads the plugin by the absolute path of its built
`lib/index.js`, so nothing needs installing into the harness. (Installing it
there as a package and using `name: molbench-harness-plugin` also works, and
dedupes the `@deepseek-ai/*` copies.) It points `command` at this project's
`.venv/bin/python` so RDKit is importable, and sets `server: molbench`, which
is what makes the Python tools appear as `mcp__molbench__<tool>`. The file also
contains a commented alternative that splits the connection into its own
`@deepseek-ai/dsh-mcp-client` entry (with `connect: false` on the plugin) when
you want to set reconnect or timeout policy yourself.

### Seeing structures without opening files

`render_molecule_2d` returns the depiction as an **inline PNG**, so it shows up
in the conversation through the harness attachment system. PNG rather than SVG
is deliberate: the MCP bridge admits PNG, JPEG, WebP and GIF as real images and
turns anything else into a text diagnostic. The SVG is still written to
`visualizations/` for a crisp copy to keep or edit.

Two conditions apply, both from the harness side: the model must accept image
input, and the attachment feature must be enabled. Otherwise the model gets a
diagnostic instead of the picture, while the structured result (scores, deltas,
SVG path) is unaffected.

Images produced by the composite tool's nested render calls are ferried into
the next model request with `exec.deferContext`, under this plugin's own
`molbench` message source.

### Seeing the 3D pose

Every docking run writes a self-contained HTML viewer next to the pose. Open it
in a browser: the protein is drawn as a cartoon, the pocket residues that line
the pose as sticks, and the ligand in green.

```bash
open "$(ls -t docking/*_pose.html | head -1)"
```

Pocket residues are resolved in Python and written into the page as explicit
residue ids, because 3Dmol's cross-model `within` selector does not filter
reliably. The viewer uses the deposited PDB numbering (Asp189, Ser195 for
trypsin), not the sequential numbering Open Babel writes into the PDBQT.

## Using it without the harness

The server is an ordinary MCP server, so any client can drive it:

```bash
.venv/bin/python server.py                            # stdio
.venv/bin/python server.py --transport http --port 8077   # long-lived service
```

Set `MOLBENCH_VIZ_DIR` to put SVGs somewhere other than `visualizations/`.

## Tests

```bash
.venv/bin/python -m pytest -q
```

The docking tests skip themselves when no engine is installed.

```bash
cd plugin && npm test
```

The Python tests drive all four tools through an in-process MCP client. The
plugin test spawns the real server over stdio and stubs only the harness
surface the plugin touches (`ctx.tools.register` and `ctx.tools.execute`),
mirroring the documented bridge: a canonical value of
`{ content, structuredContent }`, text blocks, and attachment-backed image
blocks. It checks the delta arithmetic, the image ferrying, the saved SVGs, and
the invalid-input, no-match and server-error paths.

## Notes and limits

- **Single-reactant templates.** One molecule goes in, so a template needing
  two reactants is rejected with a clear message rather than silently returning
  nothing.
- **The SA score is a heuristic**, not a synthesis plan: a fragment-frequency
  score from RDKit's `Contrib/SA_Score`. Treat deltas as a ranking signal.
- **Products are capped.** The server returns at most 32 valid products and
  flags `truncated`; the plugin scores `maxProducts` of them (3 by default).
- **Filename prefixes are sanitized** to a leaf name, so a model-supplied
  prefix cannot write outside `visualizations/`.
- **Docking scores are not binding affinities.** Vina-family scoring functions
  rank poses; they do not measure potency. `estimated_pIC50` just rescales the
  score with 2.303RT at 298 K so LLE has a sensible magnitude, and
  `Ligand_Efficiency` uses the signed convention (score / heavy atoms, target
  below -0.3), which is the negative of the conventional positive LE.
- **The receptor is rigid and the pose is one conformer's answer.** Side chains
  do not move, there is no explicit water, and no protonation-state search
  beyond Open Babel's pH model. Treat results as a filter, not a prediction.
- **Strain has two meanings here.** `generate_3d_conformer` reports strain of
  the raw embedding against its own minimum and against the ensemble minimum.
  Only `run_docking_simulation` can report the bound-conformation strain your
  workflow actually wants, as `pose_strain`.
- **Very strained cages fail to embed.** RDKit raises either a `ValueError` or
  a `RuntimeError` invariant violation for these (prismane does the latter);
  both surface as "Molecule too strained to generate 3D conformer". Cubane
  embeds fine.
- **HADDOCK3 is a hook, not a runner.** The engine enum accepts it and returns
  a clear message; driving it needs a workflow configuration and a CNS
  installation, which is licensed separately.
- **Harness package versions.** The plugin builds against
  `@deepseek-ai/dsh-tools`, `dsh-mcp-client`, `dsh-llm` and `dsh-util-values`
  at `0.2.0-rc.2`. These are release candidates whose `latest` npm tag still
  points at the older `0.0.1-rc.1`, which has a different API (`CallId` instead
  of `ToolCallId`, and no `dsh-util-values`), so keep the versions pinned
  together when upgrading.
