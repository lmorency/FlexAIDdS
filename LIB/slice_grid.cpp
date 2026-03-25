#include "gaboom.h"
#include "maps.hpp"
#include "fileio.h"

/*****************************************************************/
/******* THIS FUNCTION IS USED TO SPLIT IN HALF THE   ************/
/******* SPACING BETWEEN 2 INTERSECTIONS OF THE GRID  ************/
/******* LEADING TO A MUCH MORE DENSE GRID THAT       ************/
/******* COVERS LESS VOLUME OF THE BINDING SITE       ************/
/*****************************************************************/

void slice_grid(FA_Global* FA,genlim* gene_lim,atom* atoms,resid* residue,gridpoint** cleftgrid) {

    std::map<GridKey,int> cleftgrid_map;

    // Build lookup of existing grid points
    for(int i=1; i<FA->num_grd; i++){
        GridKey key((*cleftgrid)[i].coor);
        cleftgrid_map.insert(std::pair<GridKey,int>(key, i));
    }

    // Neighbor offsets: 3 axis-aligned + 6 face-diagonals = 9 directions.
    // Only "positive half" to avoid inserting midpoints twice (A->B and B->A).
    // Axis-aligned neighbors at distance spacer_length:
    static const int axis_offsets[][3] = {
        {1,0,0}, {0,1,0}, {0,0,1}
    };
    // Face-diagonal neighbors at distance spacer_length*sqrt(2):
    static const int diag_offsets[][3] = {
        {1,1,0}, {1,-1,0}, {1,0,1}, {1,0,-1},
        {0,1,1}, {0,1,-1}
    };

    // Snap spacer to integer milliangstroms for exact neighbor probing
    int ispacer = static_cast<int>(std::round(FA->spacer_length * 1000.0f));

    // Collect new midpoints to add
    std::vector<GridKey> new_points;

    // For each existing point, probe known neighbor offsets
    for(auto& kv : cleftgrid_map){
        const GridKey& gk = kv.first;

        // Check axis-aligned neighbors (distance = spacer_length)
        for(int d = 0; d < 3; d++){
            GridKey neighbor;
            neighbor.ix = gk.ix + axis_offsets[d][0] * ispacer;
            neighbor.iy = gk.iy + axis_offsets[d][1] * ispacer;
            neighbor.iz = gk.iz + axis_offsets[d][2] * ispacer;

            if(cleftgrid_map.find(neighbor) != cleftgrid_map.end()){
                // Midpoint in integer coords
                GridKey mid;
                mid.ix = (gk.ix + neighbor.ix) / 2;
                mid.iy = (gk.iy + neighbor.iy) / 2;
                mid.iz = (gk.iz + neighbor.iz) / 2;
                if(cleftgrid_map.find(mid) == cleftgrid_map.end()){
                    new_points.push_back(mid);
                }
            }
        }

        // Check face-diagonal neighbors (distance = spacer_length*sqrt(2))
        for(int d = 0; d < 6; d++){
            GridKey neighbor;
            neighbor.ix = gk.ix + diag_offsets[d][0] * ispacer;
            neighbor.iy = gk.iy + diag_offsets[d][1] * ispacer;
            neighbor.iz = gk.iz + diag_offsets[d][2] * ispacer;

            if(cleftgrid_map.find(neighbor) != cleftgrid_map.end()){
                GridKey mid;
                mid.ix = (gk.ix + neighbor.ix) / 2;
                mid.iy = (gk.iy + neighbor.iy) / 2;
                mid.iz = (gk.iz + neighbor.iz) / 2;
                if(cleftgrid_map.find(mid) == cleftgrid_map.end()){
                    new_points.push_back(mid);
                }
            }
        }
    }

    // Deduplicate and insert new midpoints
    std::map<GridKey,int> new_map;
    for(size_t i = 0; i < new_points.size(); i++){
        if(new_map.find(new_points[i]) == new_map.end() &&
           cleftgrid_map.find(new_points[i]) == cleftgrid_map.end()){
            new_map.insert(std::pair<GridKey,int>(new_points[i], 0));
        }
    }

    // Add new points to cleftgrid
    for(auto& kv : new_map){
        if (FA->num_grd==FA->MIN_CLEFTGRID_POINTS){
            FA->MIN_CLEFTGRID_POINTS *= 2;

            (*cleftgrid) = (gridpoint*)realloc((*cleftgrid),FA->MIN_CLEFTGRID_POINTS*sizeof(gridpoint));
            if ((*cleftgrid) == NULL){
                fprintf(stderr,"ERROR: memory reallocation error for cleftgrid (slice)\n");
                Terminate(2);
            }
        }

        memset(&(*cleftgrid)[FA->num_grd], 0, sizeof(gridpoint));
        float coor[3];
        kv.first.to_coor(coor);
        (*cleftgrid)[FA->num_grd].coor[0] = coor[0];
        (*cleftgrid)[FA->num_grd].coor[1] = coor[1];
        (*cleftgrid)[FA->num_grd].coor[2] = coor[2];

        FA->num_grd++;
    }


    ic_bounds(FA,FA->rngopt);

    //Reset gene limit and gene length
    //Minimum is always 1.0 (not affected)
    FA->max_opt_par[0] = FA->index_max;

    gene_lim->max = FA->max_opt_par[0];
    set_bins(gene_lim);

    calc_cleftic(FA,(*cleftgrid));

    // Set new spacer_length
    FA->spacer_length /= 2.0f;

}
