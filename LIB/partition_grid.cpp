#include "gaboom.h"
#include "maps.hpp"
#include "fileio.h"

/**************************************************************/
/******** THIS FUNCTION IS USED TO SELECT AREAS OF ************/
/******** THE GRID (INTERSECTIONS) THAT LEAD TO    ************/
/******** INDIVIDUALS WITH HIGHER CF VALUES        ************/
/******** AMONG THE TOP<POP_SELECTION> INDIVIDUALS ************/
/******** THIS FUNCTION ALSO EXPANDS THE GRID BY   ************/
/******** EXPANSION_FACTOR*SPACER THE CHOSEN INT.  ************/
/**************************************************************/

void partition_grid(FA_Global* FA,chromosome* chrom,genlim* gene_lim,atom* atoms,resid* residue,gridpoint** cleftgrid,int pop_selection,int expfac){

	std::map<GridKey, std::vector<int> > cleftgrid_map;
	std::map<GridKey, std::vector<int> >::iterator it;
	std::map<GridKey, int > cleftgrid_struct_map;
	std::map<GridKey, int >::iterator it2;
	std::vector<int>::iterator itv;

	//Add grid intersection indexes for the <pop_selection> first chromosomes (based on evalue)
	//Store the indexes of selected chromosomes (key is the coordinate of the grid point)
	for(int i=0;i<pop_selection;i++){
		//Only consider the first gene
		//Find the integer value from 1 to num_grd
		int grd_idx = (int)chrom[i].genes->to_ic;

		GridKey key((*cleftgrid)[grd_idx].coor);

		it = cleftgrid_map.find(key);
		if(it == cleftgrid_map.end()){
			std::vector<int> newvec;
			newvec.push_back(i);

			// key(coor) -> index individuals (ranked according to evalue)
			cleftgrid_map.insert(std::pair<GridKey, std::vector<int> >(key, newvec));
		}else{
			it->second.push_back(i);
		}
	}

	// Collect expansion keys into a separate container first to avoid
	// inserting into cleftgrid_map while iterating over it.
	std::vector<GridKey> expansion_keys;
	for(it=cleftgrid_map.begin(); it!=cleftgrid_map.end(); ++it){

		if(it->second.size() == 0) continue;

		float coor[3];
		it->first.to_coor(coor);

		for(int x=-expfac;x<=expfac;x++){
			for(int y=-expfac;y<=expfac;y++){
				for(int z=-expfac;z<=expfac;z++){

					float nc[3];
					nc[0] = coor[0] + FA->spacer_length * (float)x;
					nc[1] = coor[1] + FA->spacer_length * (float)y;
					nc[2] = coor[2] + FA->spacer_length * (float)z;

					GridKey nkey(nc);
					if(cleftgrid_map.find(nkey) == cleftgrid_map.end()){
						expansion_keys.push_back(nkey);
					}
				}
			}
		}
	}

	// Now insert expansion keys
	for(size_t i = 0; i < expansion_keys.size(); i++){
		if(cleftgrid_map.find(expansion_keys[i]) == cleftgrid_map.end()){
			std::vector<int> emptyvec;
			cleftgrid_map.insert(std::pair<GridKey, std::vector<int> >(expansion_keys[i], emptyvec));
		}
	}

	FA->num_grd = 1;

	// all keys represent a unique grid point in the new partitionned grid map
	// reinsert each key into cleftgrid structure (start at index = 1, 0 is reference)
	// increase size of cleftgrid structure if necessary
	for(it=cleftgrid_map.begin(); it!=cleftgrid_map.end(); ++it){
		float coor[3];
		it->first.to_coor(coor);

		if (FA->num_grd==FA->MIN_CLEFTGRID_POINTS){
			FA->MIN_CLEFTGRID_POINTS *= 2;

			(*cleftgrid) = (gridpoint*)realloc((*cleftgrid),FA->MIN_CLEFTGRID_POINTS*sizeof(gridpoint));
			if ((*cleftgrid) == NULL){
				fprintf(stderr,"ERROR: memory reallocation error for cleftgrid (partition)\n");
				Terminate(2);
			}
		}

		memset(&(*cleftgrid)[FA->num_grd], 0, sizeof(gridpoint));
		(*cleftgrid)[FA->num_grd].coor[0] = coor[0];
		(*cleftgrid)[FA->num_grd].coor[1] = coor[1];
		(*cleftgrid)[FA->num_grd].coor[2] = coor[2];

		// key(coor) -> grid index in structure
		cleftgrid_struct_map.insert(std::pair<GridKey,int>(it->first, FA->num_grd));

		FA->num_grd++;
	}

	ic_bounds(FA,FA->rngopt);

	//Reset gene limit and gene length
	//Minimum is always 1.0 (not affected)
	FA->max_opt_par[0] = FA->index_max;

	gene_lim->max = FA->max_opt_par[0];
	set_bins(gene_lim);

	// adjust population of chromosomes
	// loop through grid points in original grid
	for(it=cleftgrid_map.begin(); it!=cleftgrid_map.end(); ++it){
		// loop through individuals of the given grid point
		for(itv=it->second.begin(); itv!=it->second.end(); itv++){
			// what is the index of the given grid point in the new cleftgrid structure
			if((it2 = cleftgrid_struct_map.find(it->first)) != cleftgrid_struct_map.end()){

				chrom[*itv].genes->to_ic = (double)it2->second;
				chrom[*itv].genes->to_int32 = ictogene(gene_lim,it2->second);

			}else{
				fprintf(stderr, "ERROR: Could not find key in cleftgrid_struct_map\n");
				Terminate(22);
			}
		}
	}

	// Note: calc_cleftic is NOT called here because partition_grid is always
	// followed by slice_grid (gaboom.cpp), which recalculates all internal
	// coordinates after adding midpoints.

}
