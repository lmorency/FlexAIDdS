#ifndef SDF_READER_H
#define SDF_READER_H

/*
 * SdfReader — reads MDL SDF/MOL (V2000) files into FlexAID atom/resid arrays.
 *
 * SDF (Structure-Data File) is widely used in cheminformatics.  This reader
 * parses the V2000 atom/bond blocks from the first molecule in the file
 * and populates atom coordinates, types, and covalent bonds in the same
 * layout that read_lig() produces from .inp files.
 *
 * Usage:
 *   int ok = read_sdf_ligand(FA, &atoms, &residue, "ligand.sdf");
 */

#include "flexaid.h"

/* Returns 1 on success, 0 on failure. */
int read_sdf_ligand(FA_Global* FA, atom** atoms, resid** residue,
                    const char* sdf_file);

#endif // SDF_READER_H
