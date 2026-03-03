// cuda_eval.cuh — CUDA kernel interface for batched chromosome evaluation
//
// When FLEXAIDS_USE_CUDA is defined (and nvcc compiles this TU),
// FlexAIDdS offloads the per-chromosome complementarity-function (CF)
// evaluation to the GPU.  Each CUDA thread handles one chromosome:
//   – Decode gene → IC → Cartesian coordinates
//   – Pairwise distance + energy-matrix lookup
//   – Reduce per-atom CF contributions → chromosome CF value
//
// Host-side API (called from gaboom.cpp):
//   cuda_eval_init()       – allocate device memory, upload constant data
//   cuda_eval_batch()      – evaluate pop_size chromosomes in one launch
//   cuda_eval_shutdown()   – free device memory
#pragma once

#ifdef FLEXAIDS_USE_CUDA

#include <cstddef>

// Opaque handle to all device-resident data
struct CudaEvalCtx;

// Allocate device memory.
//   n_atoms       – total atom count (atoms array size)
//   n_types       – number of atom types (energy_matrix dimension)
//   max_pop       – maximum population size (upper bound on batch)
//   lig_first     – 0-based index of first ligand atom in atoms array
//   lig_last      – 0-based index of last ligand atom in atoms array
//   perm          – van-der-Waals permeability (FA->permeability)
//   h_atom_xyz    – host atom coordinates   [n_atoms × 3, float]
//   h_atom_type   – host atom type array    [n_atoms, int]
//   h_atom_radius – host atom radii         [n_atoms, float]
//   h_emat        – flattened energy matrix  [n_types × n_types, float]
CudaEvalCtx* cuda_eval_init(int   n_atoms,
                             int   n_types,
                             int   max_pop,
                             int   lig_first,
                             int   lig_last,
                             float perm,
                             const float* h_atom_xyz,
                             const int*   h_atom_type,
                             const float* h_atom_radius,
                             const float* h_emat);

// Evaluate a batch of chromosomes on GPU.
//   ctx        – context from cuda_eval_init
//   pop_size   – number of chromosomes to evaluate this call
//   n_genes    – genes per chromosome
//   h_genes    – host gene array [pop_size × n_genes, double]
//   h_cf_out   – host output array [pop_size, double] – CF values written here
void cuda_eval_batch(CudaEvalCtx* ctx,
                     int pop_size,
                     int n_genes,
                     const double* h_genes,
                     double*       h_cf_out);

// Free all device memory.
void cuda_eval_shutdown(CudaEvalCtx* ctx);

#endif  // FLEXAIDS_USE_CUDA
