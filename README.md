# pyprotalign

Protein structure superposition using sequence alignment and iterative refinement.

## Features

- **Sequence-based alignment**: Automatically identifies corresponding atoms via sequence alignment
- **Kabsch algorithm**: Optimal least-squares superposition
- **Iterative refinement**: Outlier rejection for improved accuracy
- **Multi-chain support**:
  - Single-chain alignment with specified or default chains
  - Global alignment of all matching chains
  - Quaternary alignment with smart chain matching by proximity
- **Batch processing**: Align multiple mobile structures to a single reference

## Installation

### Using uv/pip
```bash
uv pip install pyvolgrid
```

### From source
```bash
git clone https://github.com/ugSUBMARINE/pyprotalign.git
cd pyprotalign
uv venv
uv sync
```

## Quick Start

### CLI Tool

```bash
# Basic superposition (uses first protein chain from each structure)
uv run protalign fixed.cif mobile.cif -o superposed.cif

# Specify chains to align
uv run protalign fixed.cif mobile.cif --fixed-chain A --mobile-chain B

# Global alignment (align all matching chains: A-A, B-B, etc.)
uv run protalign fixed.cif mobile.cif --global

# Quaternary alignment (smart chain matching by proximity)
uv run protalign fixed.cif mobile.cif --quaternary --distance-threshold 8.0

# Quaternary alignment with chain renaming
uv run protalign fixed.cif mobile.cif --quaternary --rename-chains

# With iterative refinement (reject outliers)
uv run protalign fixed.cif mobile.cif --refine --cutoff 2.0 --cycles 5

# Output as PDB
uv run protalign fixed.cif mobile.cif -o superposed.pdb

# Batch alignment: multiple mobile files (outputs <stem>_superposed.cif)
uv run protalign reference.cif mobile1.cif mobile2.cif mobile3.cif

# Custom output suffix (e.g., <stem>_aligned.cif)
uv run protalign reference.cif *.cif --output aligned

# Batch with quaternary mode (e.g., for AlphaFold/Boltz multi-chain models)
uv run protalign reference.cif *.cif --quaternary --output aligned
```

Batch mode:
- Activated when multiple mobile files provided
- Outputs `<stem>_<suffix>.cif` for each mobile file
- Reports progress and summary with RMSD values
- Continues on errors

## Usage

```
usage: protalign [-h] [--version] [-o OUTPUT] [--fixed-chain FIXED_CHAIN] [--mobile-chain MOBILE_CHAIN] [--refine] [--cycles CYCLES] [--cutoff CUTOFF] [--global] [--quaternary] [--distance-threshold DISTANCE_THRESHOLD] [--rename-chains] [--verbose]
                 fixed mobile [mobile ...]

Protein structure superposition tool

positional arguments:
  fixed                 Fixed structure file (PDB or mmCIF)
  mobile                Mobile structure file(s) (PDB or mmCIF). If multiple files provided, batch mode is activated.

options:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
  -o, --output OUTPUT   Output file (single mode) or suffix (batch mode) (default: superposed.cif)
  --fixed-chain FIXED_CHAIN
                        Chain ID for fixed structure (e.g., A). Also used as 'seed' chain in quaternary mode. If not specified, uses first protein chain.
  --mobile-chain MOBILE_CHAIN
                        Chain ID for mobile structure (e.g., A). Also used as 'seed' chain in quaternary mode. If not specified, uses first protein chain.
  --refine              Use iterative refinement to reject outliers
  --cycles CYCLES       Maximum refinement cycles (default: 5)
  --cutoff CUTOFF       Outlier rejection cutoff (distance > cutoff * RMSD) (default: 2.0)
  --global              Align all protein chains by matching chain IDs (A-A, B-B, etc.) and pooling coordinates
  --quaternary          Quaternary alignment: match chains by proximity, rename to match fixed
  --distance-threshold DISTANCE_THRESHOLD
                        Distance threshold (Å) for chain matching in quaternary mode (default: 8.0)
  --rename-chains       Rename mobile chains to match fixed (only with --quaternary)
  --verbose             Enable verbose output (show refinement cycles, chain matching details)
```

### Output

The tool reports:
- Chain(s) and number of residues (single-chain mode)
- Chains aligned and total pairs (global mode)
- Number of aligned CA atom pairs
- Final RMSD in Ångströms
- If using `--refine`: number of pairs retained/rejected

### Examples

**Single-chain alignment:**
```bash
$ uv run protalign 9jn4.cif 9ebk.cif --refine
```

```
Fixed:  chain B, 213 residues
Mobile: chain B, 219 residues
Aligned: 207 CA atom pairs
Refinement: 167 pairs retained, 40 rejected
RMSD: 0.637 Å
Superposed structure written to: superposed.cif
```

**Chain selection:**
```bash
$ uv run protalign 9jn4.cif 9ebk.cif --fixed-chain A --mobile-chain B
```

```
Fixed:  chain A, 213 residues
Mobile: chain B, 219 residues
Aligned: 207 CA atom pairs
RMSD: 1.807 Å
Superposed structure written to: superposed.cif
```

**Global multi-chain alignment:**
```bash
$ uv run protalign 9jn4.cif 9jn6.cif --global
````

```
Chains: A, B, C, D
Aligned: 850 CA atom pairs across 4 chains
RMSD: 33.550 Å
Superposed structure written to: superposed.cif
```

**Quaternary alignment (chain labels differ):**
```bash
$ uv run protalign 9jn4.cif 9jn6.cif --quaternary
```

```
Quaternary alignment:
  B → B (matched)
  D → C (matched)
  A → A (matched)
  C → D (matched)
Aligned: 850 CA pairs across 4 chain pairs
RMSD: 0.180 Å
Superposed structure written to: superposed.cif
```

**Verbose output (detailed progress):**
```bash
$ uv run protalign 9jn4.cif 9jn6.cif --quaternary --refine --verbose
```

```
=== Quaternary Alignment ===
Seed alignment: B → B
  Refinement cycles:
    Cycle 1: 213 pairs, RMSD = 0.110 Å
    Cycle 2: 205 pairs, RMSD = 0.101 Å
    Cycle 3: 197 pairs, RMSD = 0.093 Å
    Cycle 4: 195 pairs, RMSD = 0.092 Å
    Converged (no more outliers)
Chain center distances after seed alignment:
  D ↔ C: 0.05 Å ✓
  D ↔ D: 33.91 Å ✗
  D ↔ A: 40.36 Å ✗
  A ↔ D: 17.00 Å ✗
  A ↔ A: 0.19 Å ✓
  C ↔ D: 0.23 Å ✓
Quaternary alignment:
  B → B (matched)
  D → C (matched)
  A → A (matched)
  C → D (matched)
Aligned: 850 CA pairs across 4 chain pairs
=== Final Refinement ===
  Refinement cycles:
    Cycle 1: 850 pairs, RMSD = 0.180 Å
    Cycle 2: 831 pairs, RMSD = 0.161 Å
    Cycle 3: 813 pairs, RMSD = 0.155 Å
    Cycle 4: 803 pairs, RMSD = 0.152 Å
    Cycle 5: 801 pairs, RMSD = 0.151 Å
    Converged (no more outliers)
RMSD: 0.151 Å
Superposed structure written to: superposed.cif
```

**Batch alignment (multiple mobile structures):**
```bash
$ uv run protalign 9jn4.cif 9jn5.cif 9jn6.cif 9ebk.cif --fixed-chain D --mobile-chain A --output aligned
```

```
Processing 1/3: 9jn5.cif
Fixed:  chain D, 212 residues
Mobile: chain A, 211 residues
Aligned: 211 CA atom pairs
RMSD: 0.142 Å
Output: 9jn5_aligned.cif

Processing 2/3: 9jn6.cif
Fixed:  chain D, 212 residues
Mobile: chain A, 214 residues
Aligned: 212 CA atom pairs
RMSD: 0.302 Å
Output: 9jn6_aligned.cif

Processing 3/3: 9ebk.cif
Fixed:  chain D, 212 residues
Mobile: chain A, 219 residues
Aligned: 207 CA atom pairs
RMSD: 1.754 Å
Output: 9ebk_aligned.cif

================================================================================
SUMMARY
================================================================================
Total: 3 | Successful: 3 | Failed: 0

Successful alignments:
  9jn5.cif                                 RMSD: 0.142 Å → 9jn5_aligned.cif
  9jn6.cif                                 RMSD: 0.302 Å → 9jn6_aligned.cif
  9ebk.cif                                 RMSD: 1.754 Å → 9ebk_aligned.cif
```

## Algorithm

### Single-chain mode (default)
1. **Load structures**: Reads PDB or mmCIF files
2. **Extract chains**: Selects specified chain or first protein chain
3. **Sequence alignment**: Aligns sequences using gemmi's implementation
4. **Extract CA atoms**: Gets Cα coordinates from aligned residues
5. **Superposition**: Applies Kabsch algorithm for optimal transformation
6. **Refinement** (optional): Iteratively rejects outliers beyond `cutoff × RMSD`
7. **Transform**: Applies transformation to entire mobile structure
8. **Output**: Writes superposed structure in requested format

### Global mode (`--global`)
1. **Load structures**: Reads PDB or mmCIF files
2. **Match chains**: Identifies common chain IDs (A-A, B-B, etc.)
3. **Align per chain**: Sequence alignment for each chain pair
4. **Pool coordinates**: Combines CA atoms from all matched chains
5. **Single transformation**: Computes one transformation for all pooled coordinates
6. **Refinement** (optional): Iteratively rejects outliers across all chains
7. **Transform**: Applies transformation to entire mobile structure
8. **Output**: Writes superposed structure in requested format

### Quaternary mode (`--quaternary`)
1. **Load structures**: Reads PDB or mmCIF files
2. **Seed alignment**: Aligns specified or first chain pair with optional refinement
3. **Proximity matching**: Transforms mobile copy, matches remaining chains by distance between chain centers
4. **Pool coordinates**: Sequence aligns all matched chain pairs, pools CA atoms
5. **Final transformation**: Computes transformation on pooled coords with optional refinement
6. **Transform**: Applies transformation to mobile structure
7. **Rename** (optional with `--rename-chains`): Renames mobile chains to match fixed
8. **Output**: Writes superposed structure

## Development

### Setup
```bash
uv venv                    # Create virtual environment
uv sync --group dev        # Install with dev dependencies
```

### Testing
```bash
uv run pytest              # Run all tests
uv run pytest --cov        # With coverage report
```

### Code Quality
```bash
uv run mypy src tests      # Type checking (strict mode)
uv run ruff check .        # Linting
uv run ruff format .       # Auto-formatting
```

## Dependencies

- **numpy** (≥1.26): Numerical operations
- **gemmi** (≥0.7.4): Structure I/O and sequence alignment

## Requirements

- Python ≥3.12

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on GitHub.

## Acknowledgements

Thanks to the developers of `gemmi` for their excellent library. Coding was supported by `warp.dev`.
