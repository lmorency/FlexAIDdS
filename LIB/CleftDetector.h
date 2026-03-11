#ifndef CLEFT_DETECTOR_H
#define CLEFT_DETECTOR_H

/*
 * CleftDetector — automatic binding-site detection for FlexAID
 *
 * Implements a SURFNET-style gap-sphere algorithm (the same geometric
 * principle used by GetCleft):
 *   1. For every pair of protein surface atoms within a distance cutoff,
 *      place a probe sphere midway between them.
 *   2. Shrink the probe until no other protein atom penetrates it
 *      (or discard if radius falls below a minimum).
 *   3. Cluster surviving spheres by spatial proximity (single-linkage)
 *      and keep the largest cluster as the primary cleft.
 *   4. Return a linked list of sphere_struct* ready for generate_grid().
 *
 * The implementation is header + .cpp so it can be compiled as part of
 * the FlexAID executable without any new dependencies beyond what
 * CMakeLists.txt already pulls in (Eigen3 optional, OpenMP optional).
 */

#include "flexaid.h"
#include <vector>
#include <string>

struct CleftDetectorParams {
    float max_pair_dist;     // max distance between atom pair for probe placement (A)
    float probe_radius_max;  // initial probe sphere radius (A)
    float probe_radius_min;  // minimum acceptable probe radius (A)
    float probe_shrink_step; // radius decrement per iteration (A)
    float cluster_cutoff;    // single-linkage clustering distance (A)
    int   min_cluster_size;  // discard clusters smaller than this
};

// Default parameters matching typical GetCleft behaviour
inline CleftDetectorParams default_cleft_params() {
    return {
        /* max_pair_dist    */ 12.0f,
        /* probe_radius_max */  5.0f,
        /* probe_radius_min */  1.5f,
        /* probe_shrink_step*/  0.1f,
        /* cluster_cutoff   */  4.0f,
        /* min_cluster_size */  10
    };
}

/*  detect_cleft
 *
 *  atoms    – protein atom array (already read by read_pdb)
 *  residue  – residue array
 *  atm_cnt  – total number of atoms
 *  res_cnt  – total number of residues
 *  params   – tuning knobs (use default_cleft_params() for sane defaults)
 *
 *  Returns a linked list of sphere_struct* identical to what
 *  read_spheres() produces, so it plugs straight into generate_grid().
 *  Caller owns the memory (free with free_sphere_list).
 */
sphere* detect_cleft(const atom* atoms, const resid* residue,
                     int atm_cnt, int res_cnt,
                     const CleftDetectorParams& params = default_cleft_params());

/*  write_cleft_spheres
 *
 *  Writes detected spheres to a PDB-format sphere file
 *  (same format read_spheres expects), useful for caching / inspection.
 */
void write_cleft_spheres(const sphere* spheres, const char* filename);

/*  free_sphere_list – frees the linked list returned by detect_cleft */
void free_sphere_list(sphere* head);

#endif // CLEFT_DETECTOR_H
