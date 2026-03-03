# FlexAID

Compile FlexAID using CMake:
```
git clone --branch flexaid-cpp https://github.com/NRGlab/FlexAID
cd FlexAID
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --target FlexAID -j 4
```

FlexAID requires a config and ga file to run. These can be generated using `ProcessLigand` (installed with pypi: `pip install processligand-py`).
When using ProcessLigand make sure `atom_index=90000` on the ligand.

# Required Config file codes


| Code     | Description              | Value                                                             | 
|:---------|:-------------------------|:------------------------------------------------------------------|
| `INPLIG` | Ligand input file        | Absolute path to ligand .inp file                                 |
| `METOPT` | Optimization method      | `GA`                                                              |
| `OPTIMZ` | Ligand Flexible residues | One line for each flexible bond in the ligand                     |
| `PDBNAM` | Target input file        | Absolute path to target .inp.pdb file                             |
| `RNGOPT` | Binding site file        | `GLOBAL` or `LOCCLF` + Absolute path to binding site `_sph_` file |

## More details for OPTMIZ:
This line appears at least once for the rigid docking search. Each line contains the ID of the residue to be optimized (AAA – NNNN), followed by an integer number.
This number is the number of the rotatable bond to be optimized or a zero for the ligand to be docked. For example,
`OPTIMZ 132 – 0` defines that residue 132 chain “ “ is the ligand to be docked.

Adding the following lines, you would be setting flexible the first rotatable bond of the ligand and the second flexible bond of the residue whose number is 76, chain A:

`OPTIMZ 132 – 1`

`OPTIMZ 76 A 2`

When using ProcessLigand the residue number is typically `9999` and at least 2 lines of `OPTIMZ` are required:

`OPTIMZ 9999 – -1`

`OPTIMZ 9999 – 0`


Additionally, one line is required for each line with `FLEDIH` in the ProcessLigand output.

---

## Optional Config file codes


| Code     | Description                                                              | Value                                                                                       | 
|:---------|:-------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------|
| `ACSWEI` | Weight factor for accessible contact surface normalization (requires `USEACS`) | float                                                                                       |
| `BPKENM` | Binding pocket enumeration method                                        | `XS` or `PB`                                                                                |
| `CLRMSD` | RMSD threshold between poses for clustering                              | float (e.g., 2.0)                                                                           |
| `CLUSTA` | Clustering algorithm (requires `TEMPER` to be set)                       | `FO` or `DP` or `CF` (Typically not set)                                                    |
| `COMPLF` | Complementarity function to use                                          | `SPH` or `VCT`                                                                              |
| `CONSTR` | Constraints file path                                                    | Absolute path to constraints file                                                           |
| `DEECLA` | Clash threshold for dead-end elimination of side-chains                  | float                                                                                       |
| `DEEFLX` | Enable dead-end elimination for flexible ligand bonds                    | N/A                                                                                         |
| `DEFTYP` | Force a specific atom type definition                                    | Atom type string                                                                            |
| `DEPSPA` | Path to dependencies folder                                              | Absolute path to dependencies folder                                                        |
| `EXCHET` | Exclude HET groups when calculating the complementarity function         | N/A                                                                                         |
| `FLEXSC` | Target flexibility                                                       | One line per flexible residue (Residue number, chain, Residue name). Example: ` 196  A HIS` |
| `HTPMOD` | Makes printing and file writing minimal for use in a high throughput way | N/A                                                                                         |
| `IMATRX` | Matrix file to be loaded                                                 | Absolute path to matrix file                                                                |
| `INCHOH` | Include water molecules (overrides default behaviour of removing waters)  | N/A                                                                                         |
| `INTRAF` | Fraction of intramolecular interactions to include in scoring            | float (0.0–1.0)                                                                             |
| `MAXRES` | Maximum number of results to output                                      | 10                                                                                          |
| `NMAAMP` | Path to normal modes amplitude file                                      | Absolute path to amplitude file                                                             |
| `NMAEIG` | Path to normal modes eigenvectors file                                   | Absolute path to eigenvectors file                                                          |
| `NMAMOD` | Number of normal modes to combine                                        | (int)                                                                                       |
| `NOINTR` | Disable intramolecular forces for the ligand                             | N/A                                                                                         |
| `NORMAR` | Normalize contact areas as a fraction of total surface area              | N/A                                                                                         |
| `NRGOUT` | Time FlexAID waits before aborting when `NRGSUI` option is specified     | 60 (seconds)                                                                                |
| `NRGSUI` | Writes a .update file and waits for it to be deleted before continuing   | N/A                                                                                         |
| `OMITBU` | Skip buried atoms in the Vcontacts procedure                             | N/A                                                                                         |
| `OUTRNG` | Output sphere or grid file(s) for the binding range                      | N/A                                                                                         |
| `PERMEA` | Permeability                                                             | 0.9                                                                                         |
| `RMSDST` | Reference for calculating RMSD                                           | Absolute path to ligand _ref.pdb file                                                       |
| `ROTOBS` | Use rotamer observations file instead of default Lovell's library        | N/A                                                                                         |
| `ROTOUT` | Output rotamers as PDB models in rotamers.pdb                            | N/A                                                                                         |
| `ROTPER` | VDW permeability threshold for rotamer acceptance                        | float (0.0–1.0)                                                                             |
| `SCOLIG` | Score ligand only even when flexible side-chains are enabled             | N/A                                                                                         |
| `SCOOUT` | Output only ligand coordinates in results file                           | N/A                                                                                         |
| `SLVTYP` | User specified atom type for solvent                                     | 40                                                                                          |
| `SLVPEN` | Solvent penalty term applied to scoring                                  | float                                                                                       |
| `SPACER` | Spacer length                                                            | 0.375                                                                                       |
| `STATEP` | Path to folder where Pause and Abort files can be written.               | Absolute path                                                                               |
| `TEMPER` | Temperature parameter for Metropolis criterion during clustering         | (unsigned int)                                                                              |
| `TEMPOP` | Temp folder path                                                         | Absolute path to temp folder (typically inside the `STATEP` folder)                         |
| `USEACS` | Normalize interactions by accessible contact surface                     | N/A                                                                                         |
| `USEDEE` | Enable dead-end elimination for flexible side-chains                     | N/A                                                                                         |
| `VARANG` | Delta angle                                                              | 5.0                                                                                         |
| `VARDIS` | Delta in angstroms for translational optimization                        | float                                                                                       |
| `VARDIH` | Delta dihedral                                                           | 5.0                                                                                         |
| `VARFLX` | Delta flexibility                                                        | 10.0                                                                                        |
| `VCTPLA` | Plane definition character for the Vcontacts procedure                   | character                                                                                   |
| `VCTSCO` | Vcontacts self-consistency mode (A→B and B→A contacts)                   | string                                                                                      |
| `VINDEX` | Use indexed boxes and atoms in Vcontacts for faster computation          | N/A                                                                                         |

---

## GA Codes

| Code       | Description                                                   | Value                | 
|:-----------|:--------------------------------------------------------------|:---------------------|
| `NUMCHROM` | Number of chromosomes                                         | (int)                |
| `NUMGENER` | Number of generations                                         | (int)                |
| `ADAPTVGA` | Enable adaptive GA (adjusts crossover/mutation rates dynamically) | (int flag)           |
| `ADAPTKCO` | Adaptive GA response parameters k1–k4 (each in range 0.0–1.0)    | (list) with 4 floats |
| `CROSRATE` | Crossover rate                                                    | float (0.0–1.0)      |
| `MUTARATE` | Mutation rate                                                     | float (0.0–1.0)      |
| `POPINIMT` | Population initialization method                                  | `RANDOM` or `IPFILE` |
| `FITMODEL` | Fitness model                                                     | `PSHARE` or `LINEAR` |
| `SHAREALF` | Sharing parameter α (sigma share)                                 | float                |
| `SHAREPEK` | Expected number of sharing peaks in the search space              | float                |
| `SHARESCL` | Fitness scaling factor for sharing                                | float                |
| `STRTSEED` | Set a custom starting seed                                        | (int)                |
| `REPMODEL` | Reproduction technique code                                       | `STEADY`, `BOOM`     |
| `BOOMFRAC` | Population boom size  (fraction of the number of chromosomes)     | 0 to 1 (float)       |
| `PRINTCHR` | Number of best chromosome to print each generation                | (int)                |
| `PRINTINT` | Print generation progress as well as current best cf              | 0 or 1               |
| `OUTGENER` | Output results for each generation                                | N/A                  |