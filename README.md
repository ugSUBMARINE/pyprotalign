# pyprotalign

Protein structure superposition using sequence alignment and iterative refinement.

## Features

- **Sequence-based alignment**: Automatically identifies corresponding atoms via sequence alignment
- **Kabsch algorithm**: Optimal least-squares superposition
- **Iterative refinement**: Outlier rejection for improved accuracy
- **Quality filtering**: Filter CA atoms by B-factor and occupancy
- **Multi-chain support**:
  - Single-chain alignment with specified or default chains
  - Global alignment of all matching chains (by ID or by order)
  - Quaternary alignment with smart chain matching by proximity
- **Batch processing**: Align multiple mobile structures to a single reference

## Installation

### Using uv/pip
```bash
uv pip install pyprotalign
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

# Global alignment by chain order (ignore chain IDs)
uv run protalign fixed.cif mobile.cif --global --by-order

# Quaternary alignment (smart chain matching by proximity)
uv run protalign fixed.cif mobile.cif --quaternary --distance-threshold 8.0

# Quaternary alignment with chain renaming
uv run protalign fixed.cif mobile.cif --quaternary --rename-chains

# With iterative refinement (reject outliers)
uv run protalign fixed.cif mobile.cif --refine --cutoff 2.0 --cycles 5

# Filter by B-factor / occupancy (for well-defined regions)
uv run protalign fixed.cif mobile.cif --filter --max-bfac 30 --min-occ 0.9

# Output as PDB
uv run protalign fixed.cif mobile.cif -o superposed.pdb

# Batch alignment: multiple mobile files (outputs <stem>_<suffix>.<ext>)
uv run protalign reference.cif mobile1.cif mobile2.cif mobile3.cif

# Custom output suffix (e.g., <stem>_aligned.cif)
uv run protalign reference.cif *.cif --output aligned

# Batch with quaternary mode (e.g., for AlphaFold/Boltz multi-chain models)
uv run protalign reference.cif *.cif --quaternary --output aligned
```

### All-to-All Chain Alignment

The `protalign-all2all` CLI aligns every extracted protein chain against all others, across one or more structures.
With multiple input files, chain IDs are prefixed with the file stem (e.g., `9ebk_A`).

```bash
# Table output to screen (default)
uv run protalign-all2all 9ebk.cif 9jn4.cif

# CSV output to a file (default name: all2all.csv for multiple inputs)
uv run protalign-all2all 9ebk.cif 9jn4.cif --format csv

# Custom CSV filename
uv run protalign-all2all 9ebk.cif 9jn4.cif --format csv --output pairs.csv

# Tighten filters
uv run protalign-all2all 9ebk.cif 9jn4.cif --min-aligned 50 --max-rmsd 5.0
```

Notes:
- Table output prints to stdout; CSV output always writes to a file.
- Pairs below `--min-aligned` or above `--max-rmsd` are still reported with `NaN` and a `status` in CSV.

Batch mode:
- Activated when multiple mobile files provided
- Outputs `<stem>_<suffix>.<ext>` for each mobile file (extension comes from `--output`, defaults to `.cif`)
- Reports progress per structure with RMSD values
- Continues on errors

## Usage

```
usage: protalign [-h] [--version] [-o OUTPUT] [--fixed-chain FIXED_CHAIN] [--mobile-chain MOBILE_CHAIN]
                 [--filter] [--min-bfac MIN_BFAC] [--max-bfac MAX_BFAC] [--min-occ MIN_OCC]
                 [--refine] [--cycles CYCLES] [--cutoff CUTOFF]
                 [--global] [--by-order]
                 [--quaternary] [--distance-threshold DISTANCE_THRESHOLD] [--rename-chains]
                 [--verbose]
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
  --filter              Apply filtering of CA atoms based on B-factors and occupancies
  --min-bfac MIN_BFAC   Minimum B-factor threshold - filter CA atoms with B-factors < threshold (default: -inf)
  --max-bfac MAX_BFAC   Maximum B-factor threshold - filter CA atoms with B-factors > threshold (default: inf)
  --min-occ MIN_OCC     Minimum occupancy threshold - filter CA atoms with occupancies < threshold (default: -inf)
  --refine              Use iterative refinement to reject outliers
  --cycles CYCLES       Maximum refinement cycles (default: 5)
  --cutoff CUTOFF       Outlier rejection cutoff (distance > cutoff * RMSD) (default: 2.0)
  --global              Align all protein chains by matching chain IDs or their order in the structure and pooling coordinates
  --by-order            Use the order of the chains in the structure for chain matching
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
- Quality filtering stats (if using `--filter`)
- Final RMSD in Ångströms
- If using `--refine`: number of pairs retained/rejected
- Output file path (single mode defaults to `.cif` when no extension is provided)

### Examples

**Single-chain alignment:**
```bash
$ uv run protalign 9jn4.cif 9ebk.cif --refine
```

```
=== Single-Chain Alignment ===

## Fixed structure loaded: 9jn4.cif
Using chain 'B' as target for superposition.

## Processing: 9ebk.cif
Fixed:  chain B, 213 residues
Mobile: chain B, 219 residues
Aligned 180 CA pairs with RMSD 0.649 Å
Superposed structure written to: superposed.cif
```

**Chain selection:**
```bash
$ uv run protalign 9jn4.cif 9ebk.cif --fixed-chain A --mobile-chain B
```

```
=== Single-Chain Alignment ===

## Fixed structure loaded: 9jn4.cif
Using chain 'A' as target for superposition.

## Processing: 9ebk.cif
Fixed:  chain A, 213 residues
Mobile: chain B, 219 residues
Aligned 211 CA pairs with RMSD 1.421 Å
Superposed structure written to: superposed.cif
```

**Global multi-chain alignment:**
```bash
$ uv run protalign 9jn4.cif 9jn6.cif --global
````

```
=== Global Multi-Chain Alignment ===

## Fixed structure loaded: 9jn4.cif

## Processing: 9jn6.cif
Aligned CA pairs across 4 chains: A, B, C, D
Aligned 850 CA pairs with RMSD 33.550 Å
Superposed structure written to: superposed.cif
```

**Global alignment by order (chain IDs differ):**
```bash
$ uv run protalign 9jn4.cif 9jn6.cif --global --by-order
```

```
=== Global Multi-Chain Alignment ===

## Fixed structure loaded: 9jn4.cif

## Processing: 9jn6.cif
Aligned CA pairs across 4 chain pairings (fixed → mobile): B → B, D → C, A → D, C → A
Aligned 850 CA pairs with RMSD 26.013 Å
Superposed structure written to: superposed.cif
```

**Quaternary alignment (chain labels differ):**
```bash
$ uv run protalign 9jn4.cif 9jn6.cif --quaternary
```

```
=== Quaternary Alignment ===

## Fixed structure loaded: 9jn4.cif
Using chain 'B' as seed for quaternary alignment.

## Processing: 9jn6.cif
Chain mapping after seed alignment: B → B, D → C, A → A, C → D
Aligned CA pairs across 4 chain pairings (fixed → mobile): B → B, D → C, A → A, C → D
Aligned 850 CA pairs with RMSD 0.180 Å
Superposed structure written to: superposed.cif
```

**Verbose output (detailed progress):**
```bash
$ uv run protalign 9jn4.cif 9jn6.cif --quaternary --refine --verbose
```

```
=== Quaternary Alignment ===

## Fixed structure loaded: 9jn4.cif
Using chain 'B' as seed for quaternary alignment.

## Processing: 9jn6.cif
-- Seed alignment: B → B --

Sequence alignment:

 Fixed: GAMFVPGPYHAPEDRWLVDLVRGHPLAQLASNGAGGAAPHITHVPIIVDPELDGPVDRLV
        ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
Mobile: GAMFVPGPYHAPEDRWLVDLVRGHPLAQLASNGAGGAAPHITHVPIIVDPELDGPVDRLV

 Fixed: GITLWGHMNRANPHWAALGGAANVVATFAGPNAYVSPAVYRTAPAAPTWNFTSVQVRGEL
        ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
Mobile: GITLWGHMNRANPHWAALGGAANVVATFAGPNAYVSPAVYRTAPAAPTWNFTSVQVRGEL

 Fixed: RKVESADDTLATVRATVAALESRFGAGWDMTGSLDYFRRILPGVGAFRLRVAEADGMFKL
        ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
Mobile: RKVESADDTLATVRATVAALESRFGAGWDMTGSLDYFRRILPGVGAFRLRVAEADGMFKL

 Fixed: SQEQQPAIRRRVRHSFGGAEATRAVAGLMDRLP
        |||||||||||||||||||||||||||||||||
Mobile: SQEQQPAIRRRVRHSFGGAEATRAVAGLMDRLP

Score: 1119
Indentity: 100.0%

213 CA pairs after sequence alignment.

  Refinement cycles:
    Cycle 1: 213 pairs, RMSD = 0.110 Å
    Cycle 2: 205 pairs, RMSD = 0.101 Å
    Cycle 3: 197 pairs, RMSD = 0.093 Å
    Cycle 4: 195 pairs, RMSD = 0.092 Å
    Converged (no more outliers)
Aligned 195 CA pairs with RMSD 0.092 Å

-- Chain center distances after seed alignment. --
  B ↔ B: 0.01 Å ✓
  B ↔ C: 17.08 Å ✗
  B ↔ D: 40.32 Å ✗
  B ↔ A: 49.80 Å ✗
  D ↔ B: 17.09 Å ✗
  D ↔ C: 0.05 Å ✓
  D ↔ D: 33.91 Å ✗
  D ↔ A: 40.36 Å ✗
  A ↔ B: 49.76 Å ✗
  A ↔ C: 40.28 Å ✗
  A ↔ D: 17.00 Å ✗
  A ↔ A: 0.19 Å ✓
  C ↔ B: 40.26 Å ✗
  C ↔ C: 33.90 Å ✗
  C ↔ D: 0.23 Å ✓
  C ↔ A: 17.22 Å ✗

Chain mapping after seed alignment: B → B, D → C, A → A, C → D

-- Chain mapping B → B --

Sequence alignment:

 Fixed: GAMFVPGPYHAPEDRWLVDLVRGHPLAQLASNGAGGAAPHITHVPIIVDPELDGPVDRLV
        ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
Mobile: GAMFVPGPYHAPEDRWLVDLVRGHPLAQLASNGAGGAAPHITHVPIIVDPELDGPVDRLV

 Fixed: GITLWGHMNRANPHWAALGGAANVVATFAGPNAYVSPAVYRTAPAAPTWNFTSVQVRGEL
        ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
Mobile: GITLWGHMNRANPHWAALGGAANVVATFAGPNAYVSPAVYRTAPAAPTWNFTSVQVRGEL

 Fixed: RKVESADDTLATVRATVAALESRFGAGWDMTGSLDYFRRILPGVGAFRLRVAEADGMFKL
        ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
Mobile: RKVESADDTLATVRATVAALESRFGAGWDMTGSLDYFRRILPGVGAFRLRVAEADGMFKL

 Fixed: SQEQQPAIRRRVRHSFGGAEATRAVAGLMDRLP
        |||||||||||||||||||||||||||||||||
Mobile: SQEQQPAIRRRVRHSFGGAEATRAVAGLMDRLP

Score: 1119
Indentity: 100.0%

213 CA pairs after sequence alignment.

-- Chain mapping D → C --

Sequence alignment:

 Fixed: AMFVPGPYHAPEDRWLVDLVRGHPLAQLASNGAGGAAPHITHVPIIVDPELDGPVDRLVG
        ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
Mobile: AMFVPGPYHAPEDRWLVDLVRGHPLAQLASNGAGGAAPHITHVPIIVDPELDGPVDRLVG

 Fixed: ITLWGHMNRANPHWAALGGAANVVATFAGPNAYVSPAVYRTAPAAPTWNFTSVQVRGELR
        ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
Mobile: ITLWGHMNRANPHWAALGGAANVVATFAGPNAYVSPAVYRTAPAAPTWNFTSVQVRGELR

 Fixed: KVESADDTLATVRATVAALESRFGAGWDMTGSLDYFRRILPGVGAFRLRVAEADGMFKLS
        ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
Mobile: KVESADDTLATVRATVAALESRFGAGWDMTGSLDYFRRILPGVGAFRLRVAEADGMFKLS

 Fixed: QEQQPAIRRRVRHSFGGAEATRAVAGLMDRLP
        ||||||||||||||||||||||||||||||||
Mobile: QEQQPAIRRRVRHSFGGAEATRAVAGLMDRLP

Score: 1113
Indentity: 100.0%

212 CA pairs after sequence alignment.

-- Chain mapping A → A --

Sequence alignment:

 Fixed: GAMFVPGPYHAPEDRWLVDLVRGHPLAQLASNGAGGAAPHITHVPIIVDPELDGPVDRLV
        ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
Mobile: GAMFVPGPYHAPEDRWLVDLVRGHPLAQLASNGAGGAAPHITHVPIIVDPELDGPVDRLV

 Fixed: GITLWGHMNRANPHWAALGGAANVVATFAGPNAYVSPAVYRTAPAAPTWNFTSVQVRGEL
        ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
Mobile: GITLWGHMNRANPHWAALGGAANVVATFAGPNAYVSPAVYRTAPAAPTWNFTSVQVRGEL

 Fixed: RKVESADDTLATVRATVAALESRFGAGWDMTGSLDYFRRILPGVGAFRLRVAEADGMFKL
        ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
Mobile: RKVESADDTLATVRATVAALESRFGAGWDMTGSLDYFRRILPGVGAFRLRVAEADGMFKL

 Fixed: SQEQQPAIRRRVRHSFGGAEATRAVAGLMDRLP-
        |||||||||||||||||||||||||||||||||
Mobile: SQEQQPAIRRRVRHSFGGAEATRAVAGLMDRLPT

Score: 1108
Indentity: 100.0%

213 CA pairs after sequence alignment.

-- Chain mapping C → D --

Sequence alignment:

 Fixed: -AMFVPGPYHAPEDRWLVDLVRGHPLAQLASNGAGGAAPHITHVPIIVDPELDGPVDRLV
         |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
Mobile: GAMFVPGPYHAPEDRWLVDLVRGHPLAQLASNGAGGAAPHITHVPIIVDPELDGPVDRLV

 Fixed: GITLWGHMNRANPHWAALGGAANVVATFAGPNAYVSPAVYRTAPAAPTWNFTSVQVRGEL
        ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
Mobile: GITLWGHMNRANPHWAALGGAANVVATFAGPNAYVSPAVYRTAPAAPTWNFTSVQVRGEL

 Fixed: RKVESADDTLATVRATVAALESRFGAGWDMTGSLDYFRRILPGVGAFRLRVAEADGMFKL
        ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
Mobile: RKVESADDTLATVRATVAALESRFGAGWDMTGSLDYFRRILPGVGAFRLRVAEADGMFKL

 Fixed: SQEQQPAIRRRVRHSFGGAEATRAVAGLMDRLP
        |||||||||||||||||||||||||||||||||
Mobile: SQEQQPAIRRRVRHSFGGAEATRAVAGLMDRLP

Score: 1102
Indentity: 100.0%

212 CA pairs after sequence alignment.

-- Total number of aligned CA pairs across 4 chains: 850 --

  Refinement cycles:
    Cycle 1: 850 pairs, RMSD = 0.180 Å
    Cycle 2: 831 pairs, RMSD = 0.161 Å
    Cycle 3: 813 pairs, RMSD = 0.155 Å
    Cycle 4: 803 pairs, RMSD = 0.152 Å
    Cycle 5: 801 pairs, RMSD = 0.151 Å
    Converged (no more outliers)

-- Summary --

Aligned CA pairs across 4 chain pairings (fixed → mobile): B → B, D → C, A → A, C → D
Aligned 801 CA pairs with RMSD 0.151 Å
Superposed structure written to: superposed.cif
```

**Batch alignment (multiple mobile structures):**
```bash
$ uv run protalign 9jn4.cif 9jn5.cif 9jn6.cif 9ebk.cif --fixed-chain D --mobile-chain A --output aligned
```

```
=== Single-Chain Alignment ===

## Fixed structure loaded: 9jn4.cif
Using chain 'D' as target for superposition.

## Processing 1/3: 9jn5.cif
Fixed:  chain D, 212 residues
Mobile: chain A, 211 residues
Aligned 211 CA pairs with RMSD 0.142 Å
Superposed structure written to: 9jn5_aligned.cif

## Processing 2/3: 9jn6.cif
Fixed:  chain D, 212 residues
Mobile: chain A, 214 residues
Aligned 212 CA pairs with RMSD 0.302 Å
Superposed structure written to: 9jn6_aligned.cif

## Processing 3/3: 9ebk.cif
Fixed:  chain D, 212 residues
Mobile: chain A, 219 residues
Aligned 211 CA pairs with RMSD 1.359 Å
Superposed structure written to: 9ebk_aligned.cif
```

**All-to-all chains alignment:**
```bash
$ uv run protalign-all2all 9jn4.cif
```

```
=== All-to-All Chain Alignment ===

Loaded structure: 9jn4.cif
Found 4 protein chain(s) total.


Chain 1    Chain 2    Aligned    RMSD (Å)   Status          
----------------------------------------------------------
A          B          213        0.251                      
A          C          212        0.305                      
A          D          212        0.298                      
B          C          212        0.350                      
B          D          212        0.336                      
C          D          212        0.139                      

Total pairs aligned: 6 (of 6 total)
```

## Algorithm

### Single-chain mode (default)
1. **Load structures**: Reads PDB or mmCIF files
2. **Extract chains**: Selects specified chain or first protein chain
3. **Sequence alignment**: Aligns sequences using gemmi's implementation
4. **Extract CA atoms**: Gets Cα coordinates from aligned residues
5. **Quality filtering** (optional): Filters CA atom pairs by B-factor and occupancy
6. **Superposition**: Applies Kabsch algorithm for optimal transformation
7. **Refinement** (optional): Iteratively rejects outliers beyond `cutoff × RMSD`
8. **Transform**: Applies transformation to entire mobile structure
9. **Output**: Writes superposed structure in requested format

### Global mode (`--global`)
1. **Load structures**: Reads PDB or mmCIF files
2. **Match chains**: Identifies common chain IDs (A-A, B-B, etc.). Use `--by-order` to ignore IDs and match by chain order.
3. **Align per chain**: Sequence alignment for each chain pair
4. **Quality filtering** (optional): Filters CA atom pairs per chain by B-factor and occupancy
5. **Pool coordinates**: Combines CA atoms from all matched chains
6. **Single transformation**: Computes one transformation for all pooled coordinates
7. **Refinement** (optional): Iteratively rejects outliers across all chains
8. **Transform**: Applies transformation to entire mobile structure
9. **Output**: Writes superposed structure in requested format

### Quaternary mode (`--quaternary`)
1. **Load structures**: Reads PDB or mmCIF files
2. **Seed alignment**: Aligns specified or first chain pair with optional quality filtering and refinement
3. **Proximity matching**: Transforms mobile copy, matches remaining chains by distance between chain centers
4. **Pool coordinates**: Sequence aligns all matched chain pairs, pools CA atoms
5. **Quality filtering** (optional): Filters CA atom pairs per chain by B-factor and occupancy
6. **Final transformation**: Computes transformation on pooled coords with optional refinement
7. **Transform**: Applies transformation to mobile structure
8. **Rename** (optional with `--rename-chains`): Renames mobile chains to match fixed
9. **Output**: Writes superposed structure

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

Thanks to the developers of `gemmi` for their excellent library. Coding was supported by `warp.dev` and `codex`.
