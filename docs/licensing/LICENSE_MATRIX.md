# License Matrix

Overview of third-party components used in FlexAID∆S and their compatibility with Apache-2.0.

| Component       | License       | Role                          | Apache-2.0 Compatible? |
|----------------|--------------|-------------------------------|------------------------|
| FlexAID Core   | Apache-2.0   | Docking engine foundation     | Yes                    |
| RDKit          | BSD-3-Clause | Cheminformatics toolkit       | Yes                    |
| Eigen          | MPL-2.0      | Linear algebra (header-only)  | Yes                    |
| PyMOL          | PSF          | Visualization (optional)      | Yes                    |
| OpenMP runtime | Various + exe| Parallelization               | Yes (with exceptions)  |
| CUDA Toolkit   | NVIDIA EULA  | GPU backend (optional)        | Yes (not OSS)          |
| Metal          | Apple SDK    | GPU backend (optional)        | Yes (not OSS)          |
| NRGRank        | GPL-3.0      | **Inspiration only**          | Not used as dependency |

For full details and legal text, see `THIRD_PARTY_LICENSES.md` and `docs/licensing/clean-room-policy.md`.
