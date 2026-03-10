#ifndef MOL2_READER_H
#define MOL2_READER_H

/*
 * Mol2Reader — reads Tripos MOL2 files directly into FlexAID atom/resid arrays.
 *
 * MOL2 is a common molecular format that stores:
 *   @<TRIPOS>MOLECULE   – molecule name, counts
 *   @<TRIPOS>ATOM       – atom_id name x y z type [res_id res_name charge]
 *   @<TRIPOS>BOND       – bond_id origin target type
 *
 * This reader populates the FlexAID atom array (coordinates, names, types,
 * covalent bonds) and residue array for the ligand, matching the layout
 * that read_lig() produces from .inp files.
 *
 * Usage:
 *   int ok = read_mol2_ligand(FA, &atoms, &residue, "ligand.mol2");
 *   if (!ok) { error handling }
 *
 * After calling, FA->atm_cnt, FA->res_cnt, residue[].fatm/latm,
 * atom[].coor/bond/type etc. are set up the same way read_lig() does.
 */

#include "flexaid.h"

/* Returns 1 on success, 0 on failure. */
int read_mol2_ligand(FA_Global* FA, atom** atoms, resid** residue,
                     const char* mol2_file);

#endif // MOL2_READER_H
