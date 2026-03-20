#include "flexaid.h"

/******************************************
 THIS FUNCTION COMES FROM VCONTACTS
 AND IS USED TO ASSIGN RADIUS TO ATOMS
 ******************************************/

void assign_radii(atom* atoms,resid* residue,int atm_cnt)
{
    //const char Atoms_typename[11][5] = {"C3H0","C3H1","C4  ","N3H0","N3H1","N3H2","N4  ","O1H0","O2H1","S   ","DEF "};
    const float radius[11] = { 1.61f, 1.76f, 1.88f, 1.64f, 1.64f, 1.64f, 1.64f, 1.42f, 1.46f, 1.77f, 1.80f };
    
    for(int atomi=1; atomi<=atm_cnt; ++atomi) {
        
        // ATOM records
        if(residue[atoms[atomi].ofres].type == 0){
            
            /*****************************************************************/
            /********************       DNA OR RNA      **********************/
            /*****************************************************************/
            
            if(!strncmp(residue[atoms[atomi].ofres].name, "  C", 3) || !strncmp(residue[atoms[atomi].ofres].name, "C  ", 3) ||
               !strncmp(residue[atoms[atomi].ofres].name, "  G", 3) || !strncmp(residue[atoms[atomi].ofres].name, "G  ", 3) ||
               !strncmp(residue[atoms[atomi].ofres].name, "  A", 3) || !strncmp(residue[atoms[atomi].ofres].name, "A  ", 3) ||
               !strncmp(residue[atoms[atomi].ofres].name, "  T", 3) || !strncmp(residue[atoms[atomi].ofres].name, "T  ", 3) ||
               !strncmp(residue[atoms[atomi].ofres].name, "  U", 3) || !strncmp(residue[atoms[atomi].ofres].name, "U  ", 3)) {
                
                // residue is DNA or RNA
                if(atoms[atomi].name[1] == 'P') {
                    // PHOSPHORUS atom
                    atoms[atomi].radius = radius[10];  // radius is same as default
                } else if(atoms[atomi].name[1] == 'O') {
                    // OXYGEN atom
                    // O2H0 is same radius as O2H1
                    if(atoms[atomi].name[2] == 'P') { // part of phosphate
                        atoms[atomi].radius = radius[7];  // O1H0
                    } else if(atoms[atomi].name[3] == '\'') { // ribose
                        atoms[atomi].radius = radius[8];  // O2H1 ( also O2H0, same radius)
                    } else if(atoms[atomi].name[3] == ' ') { // base
                        atoms[atomi].radius = radius[7];  // O1H0
                    } else {
                        atoms[atomi].radius = radius[7];  // O1H0 dummy value
                    }
                } else if(atoms[atomi].name[1] == 'N') {
                    // NITROGEN ATOM
                    if((atoms[atomi].name[2] == '2') || (atoms[atomi].name[2] == '4') || (atoms[atomi].name[2] == '6')) {
                        atoms[atomi].radius = radius[5];  // N3H2
                    } else { // part of ring
                        atoms[atomi].radius = radius[3];  // N3H0 (== N2H0, same radius)
                    }
                } else if(atoms[atomi].name[1] == 'C') {
                    // CARBON ATOM
                    if(atoms[atomi].name[3] == '\'') { // ribose carbon
                        atoms[atomi].radius = radius[2];  // C4
                    } else if(atoms[atomi].name[2] == '2') { 
                        if(residue[atoms[atomi].ofres].name[2] == 'A') {
                            atoms[atomi].radius = radius[1];  // C3H1
                        } else {
                            atoms[atomi].radius = radius[0];  // C3H0						
                        }
                    } else if(atoms[atomi].name[2] == '4') { 
                        atoms[atomi].radius = radius[0];  // C3H0
                    } else if(atoms[atomi].name[2] == '5') { 
                        if((residue[atoms[atomi].ofres].name[2] == 'C') || (residue[atoms[atomi].ofres].name[2] == 'U')) {
                            atoms[atomi].radius = radius[1];  // C3H1
                        } else {
                            atoms[atomi].radius = radius[0];  // C3H0
                        }
                    } else if(atoms[atomi].name[2] == '6') { 
                        if((residue[atoms[atomi].ofres].name[2] == 'A') || (residue[atoms[atomi].ofres].name[2] == 'G')) {
                            atoms[atomi].radius = radius[0];  // C3H0
                        } else {
                            atoms[atomi].radius = radius[1];  // C3H1
                        }
                    } else if(atoms[atomi].name[2] == '8') { 
                        atoms[atomi].radius = radius[1];  // C3H1
                    } else { 
                        atoms[atomi].radius = radius[1];  // C3H1, dummy value					
                    }
                    
                }else{
                    printf("Unknown atom name '%s' from residue '%s': radius set to default (2.0)\n",
                           atoms[atomi].name, residue[atoms[atomi].ofres].name);
                    
                    atoms[atomi].radius = radius[10];  // unknown atom                    
                }
                
                /*****************************************************************/
                /********************        PROTEIN        **********************/
                /*****************************************************************/
                
            } else {
                
                if(atoms[atomi].name[1] == 'O') {
                    //   OXYGEN
                    if((atoms[atomi].name[2] == 'G')||(atoms[atomi].name[2] == 'H')) {
                        atoms[atomi].radius = radius[8];  // O2H1
                    } else {
                        atoms[atomi].radius = radius[7];  // O2H0
                    }
                } else if(atoms[atomi].name[1] == 'S') {
                    // SULFUR
                    atoms[atomi].radius = radius[9]; // S
                } else if(atoms[atomi].name[1] == 'N') {
                    // NITROGEN
                    if(!strncmp(residue[atoms[atomi].ofres].name, "PRO", 3)) {
                        atoms[atomi].radius = radius[3];  // N3H0
                    } else if(atoms[atomi].name[2] == ' ') {
                        atoms[atomi].radius = radius[4];  // N3H1
                    } else if(atoms[atomi].name[2] == 'E') {
                        atoms[atomi].radius = radius[4];  // N3H1
                    } else if((atoms[atomi].name[2] == 'D') && (!strncmp(residue[atoms[atomi].ofres].name, "HIS", 3))) {
                        atoms[atomi].radius = radius[4];  // N3H1
                    } else if((atoms[atomi].name[2] == 'D') && (!strncmp(residue[atoms[atomi].ofres].name, "ASN", 3))) {
                        atoms[atomi].radius = radius[5];  // N3H2
                    } else if((atoms[atomi].name[2] == 'E') && (!strncmp(residue[atoms[atomi].ofres].name, "GLN", 3))) {
                        atoms[atomi].radius = radius[5];  // N3H2
                    } else if((atoms[atomi].name[2] == 'E') && (strncmp(residue[atoms[atomi].ofres].name, "GLN", 3))) {
                        atoms[atomi].radius = radius[4];  // N3H1
                    } else if(atoms[atomi].name[2] == 'H') {
                        atoms[atomi].radius = radius[5];  // N3H2
                    } else if(atoms[atomi].name[2] == 'Z') {
                        atoms[atomi].radius = radius[6];  // N4
                    } else {
                        atoms[atomi].radius = radius[5];  // N3H2,  N default
                    }
                } else if(atoms[atomi].name[1] == 'C') {
                    // CARBON
                    if(atoms[atomi].name[2] == ' ') {
                        atoms[atomi].radius = radius[0];  // C3H0 backbone carbon
                    } else if((!strncmp(residue[atoms[atomi].ofres].name, "ASP", 3)) && (atoms[atomi].name[2] == 'G')) {
                        atoms[atomi].radius = radius[0];  // C3H0
                    } else if((!strncmp(residue[atoms[atomi].ofres].name, "GLU", 3)) && (atoms[atomi].name[2] == 'D')) {
                        atoms[atomi].radius = radius[0];  // C3H0
                    } else if((!strncmp(residue[atoms[atomi].ofres].name, "ASN", 3)) && (atoms[atomi].name[2] == 'G')) {
                        atoms[atomi].radius = radius[0];  // C3H0
                    } else if((!strncmp(residue[atoms[atomi].ofres].name, "GLN", 3)) && (atoms[atomi].name[2] == 'D')) {
                        atoms[atomi].radius = radius[0];  // C3H0
                    } else if((!strncmp(residue[atoms[atomi].ofres].name, "ARG", 3)) && (atoms[atomi].name[2] == 'Z')) {
                        atoms[atomi].radius = radius[0];  // C3H0
                    } else if((!strncmp(residue[atoms[atomi].ofres].name, "PHE", 3)) || (!strncmp(residue[atoms[atomi].ofres].name, "HIS", 3))
                              || (!strncmp(residue[atoms[atomi].ofres].name, "TYR", 3)) || (!strncmp(residue[atoms[atomi].ofres].name, "TRP", 3))) {
                        if((atoms[atomi].name[2] == 'A') || (atoms[atomi].name[2] == 'B')) {
                            atoms[atomi].radius = radius[2];  // C4
                        } else {
                            atoms[atomi].radius = radius[1];  // C3H1
                        }
                    } else { // carbon is C4, aliphatic
                        atoms[atomi].radius = radius[2];    // aliphatic carbon
                    }
                } else {
                    // default radius
                    atoms[atomi].radius = radius[10];     // default for unknown atom;
                }
            }
         
            
        } else {
            // HETATM records: ions (residue-name lookup) then organic cofactors (element)
            static const struct { const char* rnam; float r; } ions[] = {
                {"MG ", 1.73f}, {"ZN ", 1.39f}, {"CA ", 1.74f}, {"NA ", 2.27f},
                {"K  ", 2.75f}, {"FE ", 1.47f}, {"FE2", 1.47f}, {"FE3", 1.47f},
                {"CU ", 1.40f}, {"CU1", 1.40f}, {"CU2", 1.40f}, {"MN ", 1.61f},
                {"CO ", 1.52f}, {"NI ", 1.63f}, {"CL ", 1.75f}, {"BR ", 1.85f},
                {"IOD", 1.98f}, {"LI ", 1.82f}, {"CD ", 1.58f}, {"HG ", 1.55f},
                {"PB ", 2.02f}, {nullptr, 0.0f}
            };
            const char* rname = residue[atoms[atomi].ofres].name;
            bool assigned = false;
            for (int n = 0; ions[n].rnam; ++n) {
                if (!strncmp(rname, ions[n].rnam, 3)) {
                    atoms[atomi].radius = ions[n].r;
                    assigned = true;
                    break;
                }
            }
            if (!assigned) {
                // Organic cofactor or modified residue: infer from element in atom name [1]
                switch (atoms[atomi].name[1]) {
                    case 'C': atoms[atomi].radius = 1.88f; break;  // C4
                    case 'N': atoms[atomi].radius = 1.64f; break;  // N3H0
                    case 'O': atoms[atomi].radius = 1.42f; break;  // O1H0
                    case 'S': atoms[atomi].radius = 1.77f; break;
                    case 'P': atoms[atomi].radius = 1.80f; break;
                    default:  atoms[atomi].radius = 1.50f; break;  // generic
                }
            }
        }
        
        //    printf("Atom[%d]=%d\t%s(%s) Rad=%1.2f\n",atomi,atoms[atomi].number,atoms[atomi].name,residue[atoms[atomi].ofres].name,atoms[atomi].radius);
    }
    return;
}
