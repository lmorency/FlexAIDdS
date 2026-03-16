#include "gaboom.h"
#include "Vcontacts.h"
#include "fileio.h"
#include "hardware_dispatch.h"

#include <random>
#include <functional>
#include <cstdint>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef FLEXAIDS_HAS_EIGEN
#include <Eigen/Dense>
#endif

#ifdef FLEXAIDS_USE_CUDA
#include "cuda_eval.cuh"
#endif

#ifdef FLEXAIDS_USE_METAL
#include "metal_eval.h"
#endif

#include "statmech.h"
#include "tencm.h"
#include "ShannonThermoStack/ShannonThermoStack.h"
#include "NATURaL/NATURaLDualAssembly.h"

// in milliseconds
# define SLEEP 25

#ifdef _WIN32
# include <windows.h>
#else
# include <unistd.h>
#endif


/***********************************************************************/
/* 1         2         3         4         5         6          */
/*234567890123456789012345678901234567890123456789012345678901234567890*/
/* 1         2         3         4         5         6         7*/
/***********************************************************************/
int GA(FA_Global* FA, GB_Global* GB,VC_Global* VC,chromosome** chrom,chromosome** chrom_snapshot,
       genlim** gene_lim,atom* atoms,resid* residue,gridpoint** cleftgrid,char gainpfile[],
       int* memchrom, cfstr (*target)(FA_Global*,VC_Global*,atom*,resid*,gridpoint*,int,double*)){

	int i;
	int print=0;

	//char tmp_rrgfile[MAX_PATH__];
	//int rrg_flag;
	//int rrg_skip=100;

	char outfile[MAX_PATH__];
	int n_chrom_snapshot=0;
	char gridfile[MAX_PATH__];
	char gridfilename[MAX_PATH__];

	int geninterval=50;
	int popszpartition=100;

	int  state=0;
	char PAUSEFILE[MAX_PATH__];
	char ABORTFILE[MAX_PATH__];
	char STOPFILE[MAX_PATH__];

	const int INTERVAL = 1; // sleep interval between checking file state

	*memchrom=0; //num chrom allocated in memory

	// for generation random doubles from [0,1[ (mutation crossover operators)
	strcpy(PAUSEFILE,FA->state_path);
#ifdef _WIN32
	strcat(PAUSEFILE,"\\.pause");
#else
	strcat(PAUSEFILE,"/.pause");
#endif


	strcpy(ABORTFILE,FA->state_path);
#ifdef _WIN32
	strcat(ABORTFILE,"\\.abort");
#else
	strcat(ABORTFILE,"/.abort");
#endif

	strcpy(STOPFILE,FA->state_path);
#ifdef _WIN32
	strcat(STOPFILE,"\\.stop");
#else
	strcat(STOPFILE,"/.stop");
#endif

	GB->num_genes=FA->npar;
	if(GB->num_genes == 0){
		fprintf(stderr,"ERROR: no parameters to optimize.\n");
		Terminate(1);
	}

	printf("num_genes=%d\n",GB->num_genes);

	//GB->rrg_skip=0;
	GB->adaptive_ga=0;
	GB->num_print=10;
	GB->print_int=1;
	GB->seed = 0;

	GB->ssnum = 1000;
	GB->pbfrac = 1.0;
	GB->duplicates = 0;
	GB->intragenes = 0;

	// Entropy convergence defaults (opt-in)
	GB->entropy_convergence    = 0;
	GB->entropy_check_interval = 10;
	GB->entropy_window         = 5;
	GB->entropy_rel_threshold  = 0.01;

	printf("file in GA is <%s>\n",gainpfile);

	read_gainputs(FA,GB,&geninterval,&popszpartition,gainpfile);
	unsigned int tt;
	if (GB->seed==0)
	{
		tt = static_cast<unsigned int>(time(0));
	}
	else
	{
		tt = GB->seed;
	}
	//tt = (unsigned)1;
	printf("srand=%u\n", tt);
	srand(tt);
	std::mt19937 rng(tt);

	std::uniform_int_distribution<int32_t> one_to_max_int32( 0, MAX_RANDOM_VALUE );
	std::function<int32_t()> dice = [&]() { return one_to_max_int32(rng); };

	(*gene_lim) = (genlim*)malloc(GB->num_genes*sizeof(genlim));
	if(!(*gene_lim)){
		fprintf(stderr,"ERROR: memory allocation error for gene_lim.\n");
		Terminate(2);
	}

	long int at = 0;

	if(strcmp(GB->pop_init_method,"RANDOM") == 0){
		set_gene_lim(FA, GB, (*gene_lim));
		set_bins((*gene_lim),GB->num_genes);

	}else if(strcmp(GB->pop_init_method,"IPFILE") == 0){
		at = read_pop_init_file(FA, GB, (*gene_lim), GB->pop_init_file);
		if(!at){
			fprintf(stderr,"ERROR: Unknown format for pop init file.\n");
			Terminate(10);
		}
	}

	if(GB->print_int < 0){ GB->print_int = 1; }

	//if(GB->rrg_skip > 0){ rrg_skip = GB->rrg_skip; }

	if(GB->num_print > GB->num_chrom){ GB->num_print = GB->num_chrom; }

	if(popszpartition > GB->num_chrom){ popszpartition = GB->num_chrom; }

	if(FA->opt_grid){
		printf("will partition grid every %d generations considering %d individuals\n",
		       geninterval, popszpartition);
	}

	validate_dups(GB, (*gene_lim), GB->num_genes);

	(*memchrom) = GB->num_chrom;
	if(strcmp(GB->rep_model,"STEADY")==0){
		(*memchrom) += GB->ssnum;
	}else if(strcmp(GB->rep_model,"BOOM")==0){
		(*memchrom) += (int)(GB->pbfrac*(double)GB->num_chrom);
	}

	//printf("memchrom=%d\n",(*memchrom));
	//printf("num_genes=%d\n",GB->num_genes);

	// *** chrom
	(*chrom) = (chromosome*)malloc((*memchrom)*sizeof(chromosome));
	if(!(*chrom)){
		fprintf(stderr,"ERROR: memory allocation error for chrom.\n");
		Terminate(2);
	}

	for(i=0;i<(*memchrom);++i)
	{
		(*chrom)[i].genes = (gene*)malloc(GB->num_genes*sizeof(gene));

		if(!(*chrom)[i].genes){
			fprintf(stderr,"ERROR: memory allocation error for chrom[%d].genes.\n",i);
			Terminate(2);
		}

		(*chrom)[i].app_evalue = 0.0;
		(*chrom)[i].evalue = 0.0;
		(*chrom)[i].fitnes = 0.0;
		(*chrom)[i].status = ' ';
	}

	// *** chrom_snapshot
	(*chrom_snapshot) = (chromosome*)malloc((GB->num_chrom*GB->max_generations)*sizeof(chromosome));
	if(!(*chrom_snapshot))
	{
		fprintf(stderr,"ERROR: memory allocation error for chrom_snapshot.\n");
		Terminate(2);
	}

	for(i=0;i<(GB->num_chrom*GB->max_generations);++i)
	{
		(*chrom_snapshot)[i].genes = (gene*)malloc(GB->num_genes*sizeof(gene));

		if(!(*chrom_snapshot)[i].genes){
			fprintf(stderr,"ERROR: memory allocation error for chrom_snapshot[%d].genes.\n",i);
			Terminate(2);
		}

		(*chrom_snapshot)[i].app_evalue = 0.0;
		(*chrom_snapshot)[i].evalue = 0.0;
		(*chrom_snapshot)[i].fitnes = 0.0;
		(*chrom_snapshot)[i].status = ' ';
		//printf("chrom_snapshot[%d] allocated at address %p!\n", i, &(*chrom_snapshot)[i]);
	}

	printf("alpha %lf peaks %lf scale %lf\n",GB->alpha,GB->peaks,GB->scale);
	GB->sig_share=0.0;

	for(i=0;i<GB->num_genes;i++)
	{
		//printf("max=%6.3f min=%6.3f del=%6.3f\n",(*gene_lim)[i].max,(*gene_lim)[i].min,(*gene_lim)[i].del);
		//PAUSE;
		GB->sig_share += ((*gene_lim)[i].max-(*gene_lim)[i].min)*((*gene_lim)[i].max-(*gene_lim)[i].min);
	}
	GB->sig_share = sqrt(GB->sig_share/(double)GB->num_genes)/(2.0*pow(GB->peaks,(1.0/(double)GB->num_genes)));
	GB->sig_share /= GB->scale;
	printf("SIGMA_SHARE=%f\n",GB->sig_share);
	fflush(stdout);

	// for(i=0;i<GB->num_genes;i++) {
	//printf("GENE(%d)=[%8.3f,%8.3f,%8.3f,%d]\n",
	//	   i,(*gene_lim)[i].min,(*gene_lim)[i].max,(*gene_lim)[i].del);
	//PAUSE;

	std::map<std::string, int> duplicates;

	populate_chromosomes(FA,GB,VC,(*chrom),(*gene_lim),atoms,residue,(*cleftgrid),
			     GB->pop_init_method,target,GB->pop_init_file,at,0,print,dice,duplicates);
	//}

	//print_pop((*chrom),(*gene_lim),GB->num_chrom,GB->num_genes);

	/*
	  for(i=0;i<GB->num_genes;i++){
	  printf("%d %f %f %f\n",i,GB->min_opt_par[i],GB->max_opt_par[i],GB->del_opt_par[i]);
	  PAUSE;
	  }
	*/

	int save_num_chrom = (int)(GB->num_chrom*SAVE_CHROM_FRACTION);
	int nrejected = 0;

	// Entropy convergence tracking
	std::vector<double> entropy_history;
	bool entropy_converged = false;
	if (GB->entropy_convergence) {
		entropy_history.reserve(GB->max_generations / GB->entropy_check_interval + 1);
	}

	////////////////////////////////
	////// Genetic Algorithm ///////
	////////////////////////////////
	for(i=0;i<GB->max_generations;i++)
	{
		///////////////////////////////////////////////////

		state=check_state(PAUSEFILE,ABORTFILE,STOPFILE,INTERVAL);

		if(state == -1){
			return(state);
		}else if(state == 1){
			break;
		}

		////////////////////////////////

		////////////////////////////////

		////////////////////////////////

		//printf("chrom_snapshot[%d] at address %p\n", i*GB->num_chrom, chrom_snapshot[i*GB->num_chrom]);
		if (	FA->opt_grid                    &&     // if a OPTGRD line was specified
		    	((i+1) % geninterval) == 0      &&     // is factor of
		    	(i+1) != GB->max_generations 	)      // discard the last generation
		{

			//need to sort in decreasing order of energy
			QuickSort((*chrom),0,GB->num_chrom-1,true);

			//printf("Partionning grid...(%d)\n",FA->popszpartition);
			partition_grid(FA,(*chrom),(*gene_lim),atoms,residue,cleftgrid,popszpartition,1);

			if(FA->output_range){
#ifdef _WIN32
				sprintf(gridfilename,"\\grid.%d.prt.pdb",i+1);
#else
				sprintf(gridfilename,"/grid.%d.prt.pdb",i+1);
#endif
				strcpy(gridfile,FA->temp_path);
				strcat(gridfile,gridfilename);

				write_grid(FA,(*cleftgrid),gridfile);
			}

			slice_grid(FA,(*gene_lim),atoms,residue,cleftgrid);

			if(FA->output_range){

#ifdef _WIN32
				sprintf(gridfilename,"\\grid.%d.slc.pdb",i+1);
#else
				sprintf(gridfilename,"/grid.%d.slc.pdb",i+1);
#endif

				strcpy(gridfile,FA->temp_path);
				strcat(gridfile,gridfilename);

				write_grid(FA,(*cleftgrid),gridfile);
			}

			validate_dups(GB, (*gene_lim), GB->num_genes);

			//repopulate unselected individuals
			populate_chromosomes(FA,GB,VC,(*chrom),(*gene_lim),atoms,residue,(*cleftgrid),
					     GB->pop_init_method,target,GB->pop_init_file,at,popszpartition,print,dice,duplicates);
		}

		print = ( (i+1) % GB->print_int == 0 ) ? 1 : 0;
		//if(print) { printf("Generation: %5d\n",i+1); }

		//print_par(chrom,gene_lim,20,GB->num_genes);
		//PAUSE;

		/*
		  rrg_flag=0;
		  if((i/rrg_skip)*rrg_skip == i) rrg_flag=1;
		  if((rrg_flag==1) && (GB->outgen==1)){
		  if(FA->refstructure == 1){
		  sprintf(tmp_rrgfile,"%s_%d.rrg",FA->rrgfile,i);
		  //printf("%s\n",tmp_rrgfile);
		  //PAUSE;
		  write_rrg(FA,GB,(*chrom),(*gene_lim),atoms,residue,(*cleftgrid),tmp_rrgfile);
		  }
		  }
		*/


		//before reproducing for an extra generation, evaluate if population has converged.
		//before calculating get avg and max fitness of the whole pop.
		fitness_stats(GB,(*chrom),GB->num_chrom);

		//printf("------fitness stats-------\navg=%8.3f\tmax=%8.3f\n",GB->fit_avg,GB->fit_max);
        //getchar();

		// Entropy convergence check (opt-in via ENTRCNVG config keyword)
		if (GB->entropy_convergence &&
		    ((i + 1) % GB->entropy_check_interval == 0)) {
			std::vector<double> pop_energies(GB->num_chrom);
			for (int c = 0; c < GB->num_chrom; ++c)
				pop_energies[c] = (*chrom)[c].evalue;
			double H = shannon_thermo::compute_shannon_entropy(
				pop_energies, shannon_thermo::DEFAULT_HIST_BINS);
			entropy_history.push_back(H);

			if (shannon_thermo::detect_entropy_plateau(
			        entropy_history, GB->entropy_window,
			        GB->entropy_rel_threshold)) {
				printf("Entropy convergence at generation %d "
				       "(H=%.4f nats, stable for %d checks)\n",
				       i + 1, H, GB->entropy_window);
				entropy_converged = true;
				break;
			}
		}

		nrejected = reproduce(FA,GB,VC,(*chrom),(*gene_lim),atoms,residue,(*cleftgrid),
				      GB->rep_model,GB->mut_rate,GB->cross_rate,print,dice,duplicates,target);

		save_snapshot(&(*chrom_snapshot)[i*GB->num_chrom],(*chrom),save_num_chrom,GB->num_genes);
		n_chrom_snapshot += save_num_chrom;


		if(strcmp(GB->fitness_model,"PSHARE")==0){
			QuickSort((*chrom),0,GB->num_chrom-1,false);

			if(print){
				printf("best by fitnes\n");
				print_par((*chrom),(*gene_lim),GB->num_print,GB->num_genes, stdout);
			}
		}

	}

	printf("%d ligand conformers rejected\n", nrejected);
	if (entropy_converged)
		printf("GA terminated early by entropy convergence\n");

	QuickSort((*chrom),0,GB->num_chrom-1,true);

	strcpy(outfile,FA->rrgfile);
	strcat(outfile,"_par.res");
	if (FA->htpmode == false) {write_par((*chrom),(*gene_lim),i+1,outfile,GB->num_chrom,GB->num_genes);}

	printf("sorting chrom_snapshot\n");
	//quicksort_app_evalue((*chrom_snapshot),0,n_chrom_snapshot-1);
	QuickSort((*chrom_snapshot),0,n_chrom_snapshot-1,true);

	/*
	  printf("Save snapshot == END ==\n");
	  print_par((*chrom_snapshot),(*gene_lim),n_chrom_snapshot,GB->num_genes);
	*/

	printf("removing duplicates\n");
	n_chrom_snapshot = remove_dups((*chrom_snapshot),n_chrom_snapshot,GB->num_genes);

	/*
		printf("Save snapshot == END ==\n");
		print_par((*chrom_snapshot),(*gene_lim),n_chrom_snapshot,GB->num_genes);
	*/

	// Thermodynamic analysis of the final conformational ensemble
	if(n_chrom_snapshot > 0) {
		double T_K = (FA->temperature > 0) ? static_cast<double>(FA->temperature) : 300.0;
		statmech::StatMechEngine sme(T_K);
		for(int s = 0; s < n_chrom_snapshot; ++s)
			sme.add_sample((*chrom_snapshot)[s].evalue);
		statmech::Thermodynamics td = sme.compute();
		printf("--- Thermodynamics (T = %.1f K, N = %d conformers) ---\n",
		       td.temperature, n_chrom_snapshot);
		printf("  Helmholtz free energy  F  = %10.4f kcal/mol\n", td.free_energy);
		printf("  Mean energy          <E>  = %10.4f kcal/mol\n", td.mean_energy);
		printf("  Energy std dev        σ_E = %10.4f kcal/mol\n", td.std_energy);
		printf("  Heat capacity         C_v = %10.4f kcal/(mol·K)\n", td.heat_capacity);
		printf("  Entropy (conf)        S   = %10.6f kcal/(mol·K)\n", td.entropy);

		// ── Phase 3: TorsionalENM vibrational entropy ────────────────
		tencm::TorsionalENM tencm_model;
		if (FA->is_protein && FA->res_cnt > 6) {
			tencm_model.build(atoms, residue, FA->res_cnt);
			if (tencm_model.is_built()) {
				// Store mode count on FA for BindingMode vibrational correction
				FA->normal_modes = static_cast<int>(tencm_model.modes().size());

				// Run full ShannonThermoStack: Shannon conf entropy + torsional vib entropy
				shannon_thermo::FullThermoResult ftr =
					shannon_thermo::run_shannon_thermo_stack(
						sme, tencm_model, td.free_energy, T_K);

				printf("--- ShannonThermoStack (vibrational entropy integration) ---\n");
				printf("  Shannon conf entropy    = %10.4f nats\n", ftr.shannonEntropy);
				printf("  Torsional vib entropy   = %10.6f kcal/(mol·K)\n", ftr.torsionalVibEntropy);
				printf("  Entropy contribution    = %10.4f kcal/mol (-TΔS)\n", ftr.entropyContribution);
				printf("  Total ΔG (F + vib corr) = %10.4f kcal/mol\n", ftr.deltaG);
			}
		}
	}

	// NATURaL co-translational / co-transcriptional DualAssembly analysis
	// Skipped when --folded flag or advanced.assume_folded=true (receptor is fully folded)
	if (!FA->assume_folded && FA->resligand && FA->resligand->fatm && FA->resligand->latm) {
		int lig_start   = FA->resligand->fatm[0];
		int lig_end     = FA->resligand->latm[0];
		int n_lig_atoms = lig_end - lig_start + 1;
		if (n_lig_atoms > 0 && FA->MIN_NUM_RESIDUE > 0) {
			natural::NATURaLConfig ncfg = natural::auto_configure(
				&atoms[lig_start], n_lig_atoms,
				residue, FA->MIN_NUM_RESIDUE);
			if (ncfg.enabled) {
				ncfg.temperature_K = (FA->temperature > 0)
				                     ? static_cast<double>(FA->temperature)
				                     : 310.0;
				natural::DualAssemblyEngine engine(
					ncfg, FA, VC, atoms, residue, FA->MIN_NUM_RESIDUE);
				auto trajectory = engine.run();
				printf("--- NATURaL Co-translational DualAssembly (%zu growth steps) ---\n",
				       trajectory.size());
				if (!trajectory.empty()) {
					printf("  Final ΔG (co-translational) = %10.4f kcal/mol\n",
					       engine.final_deltaG());
					int n_pause = 0, n_tm = 0;
					for (const auto& step : trajectory) {
						if (step.is_pause_site) ++n_pause;
						if (step.tm_inserted)   ++n_tm;
					}
					printf("  Pause sites detected        = %d\n", n_pause);
					printf("  TM insertions               = %d\n", n_tm);
				}
			}
		}
	}

	return n_chrom_snapshot;
}

void copy_chrom(chromosome* dest, const chromosome* src, int num_genes){

	dest->cf = src->cf;
	dest->evalue = src->evalue;
	dest->app_evalue = src->app_evalue;
	dest->fitnes = src->fitnes;
	dest->status = src->status;

	for(int j=0; j<num_genes; j++){
	        dest->genes[j].to_ic = src->genes[j].to_ic;
		dest->genes[j].to_int32 = src->genes[j].to_int32;
	}
}

/***********************************************************************/
/* 1         2         3         4         5         6          */
/*234567890123456789012345678901234567890123456789012345678901234567890*/
/* 1         2         3         4         5         6         7*/
/***********************************************************************/
void save_snapshot(chromosome* chrom_snapshot, const chromosome* chrom, int num_chrom, int num_genes){

	for(int i=0; i<num_chrom; i++)
		copy_chrom(&chrom_snapshot[i],&chrom[i],num_genes);

}

/***********************************************************************/
/* 1         2         3         4         5         6          */
/*234567890123456789012345678901234567890123456789012345678901234567890*/
/* 1         2         3         4         5         6         7*/
/***********************************************************************/
int check_state(char* pausefile, char* abortfile, char* stopfile, int interval){
	FILE* STATE;

	STATE = NULL;

	// try and open pause/stop file
	// (works with PyMOL interface)

	STATE = fopen(pausefile,"r");
	if(STATE != NULL) {
		do {
			fclose(STATE);

# ifdef _WIN32
			Sleep(SLEEP);
# else
			usleep(SLEEP*1000);
# endif
			STATE = fopen(pausefile,"r");

		}while(STATE != NULL);
	}

	STATE = fopen(abortfile,"r");
	if(STATE != NULL) {
		fclose(STATE);
		printf("manual aborting\n");
		return -1;
	}

	STATE = fopen(stopfile,"r");
	if(STATE != NULL) {
		fclose(STATE);
		printf("simulation stopped prematurely\n");
		return 1;
	}

	return 0;
}
/***********************************************************************/
/* 1         2         3         4         5         6          */
/*234567890123456789012345678901234567890123456789012345678901234567890*/
/* 1         2         3         4         5         6         7*/
/***********************************************************************/
void fitness_stats(GB_Global* GB, const chromosome* chrom,int pop_size){
	int i;
	int flag;

	//calculate fitness max and and average of the whole pop
	GB->fit_max=0.0;
	GB->fit_avg=0.0;

	flag=1;
	for(i=0;i<pop_size-i;i++){
		if (flag){
			GB->fit_max=chrom[i].fitnes;
			flag=0;
		}

		if(chrom[i].fitnes > GB->fit_max)
			GB->fit_max=chrom[i].fitnes;

		GB->fit_avg+=chrom[i].fitnes;
	}

	GB->fit_avg /= (double)pop_size;

	return;
}
/***********************************************************************/
/* 1         2         3         4         5         6          */
/*234567890123456789012345678901234567890123456789012345678901234567890*/
/* 1         2         3         4         5         6         7*/
/***********************************************************************/
void adapt_prob(GB_Global* GB,double fit1, double fit2, double* mutp, double* crossp){
	//printf("crossing fit1[%8.3f] with fit2[%8.3f]\n",fit1,fit2);

	//find which crossed individual has higher fitness
	if(fit1 > fit2){
		GB->fit_high=fit1;
		GB->fit_low=fit2;
	}else{
		GB->fit_high=fit2;
		GB->fit_low=fit1;
	}

	//crossp=k1 when high=avg
	//mutp=k2 when high=avg
	//crossp/mutp=0 when high=max

	//calculate new probabilities (pc/pm)
	if (GB->fit_high > GB->fit_avg) *crossp = GB->k1*(GB->fit_max-GB->fit_high)/(GB->fit_max-GB->fit_avg);
	else *crossp = GB->k3;

	if (GB->fit_low > GB->fit_avg) *mutp = GB->k2*(GB->fit_max-GB->fit_low)/(GB->fit_max-GB->fit_avg);
	else *mutp = GB->k4;

	/*
	  printf("f'=%.1f\tf=%.1f\tfmax=%.1f\tfavg=%.1f\t\tPc=%5.3f\tPm=%5.3f\n",
	  GB->fit_high,GB->fit_low,GB->fit_max,GB->fit_avg,*crossp,*mutp);
	*/

	return;
}
/***********************************************************************/
/* 1         2         3         4         5         6          */
/*234567890123456789012345678901234567890123456789012345678901234567890*/
/* 1         2         3         4         5         6         7*/
/***********************************************************************/
int reproduce(FA_Global* FA,GB_Global* GB,VC_Global* VC, chromosome* chrom, const genlim* gene_lim,
               atom* atoms,resid* residue,gridpoint* cleftgrid,char* repmodel,
               double mutprob, double crossprob, int print,
	       std::function<int32_t()> & dice,
	       std::map<std::string, int> & duplicates,
               cfstr (*target)(FA_Global*,VC_Global*,atom*,resid*,gridpoint*,int,double*)){

	static int nrejected = 0;

	int i,j,k;
	int nnew,p1,p2;

	gene chrop1_gen[MAX_NUM_GENES];
	gene chrop2_gen[MAX_NUM_GENES];

	int num_genes_wo_sc=0;

	/*
	  std::mt19937 rng;
	  std::uniform_int_distribution<int32_t> one_to_max_int32( 1, MAX_RANDOM_VALUE );
	  std::function<int32_t()> dice = [&](){ return one_to_max_int32(rng); };
	*/

	if(strcmp(repmodel,"STEADY")==0){
		nnew = GB->ssnum;
	}else if(strcmp(repmodel,"BOOM")==0){
		nnew = (int)(GB->pbfrac*(double)GB->num_chrom);
	}else{
		nnew = 0;
	}

	i=0;
	while(i<nnew){

		/************************************/
		/****** SELECTION OF PARENTS ********/
		/************************************/
		p1=roullete_wheel(chrom,GB->num_chrom);
		p2=roullete_wheel(chrom,GB->num_chrom);
		if (GB->adaptive_ga) adapt_prob(GB,chrom[p1].fitnes,chrom[p2].fitnes,&mutprob,&crossprob);

		/************************************/
		/****** CROSSOVER OPERATOR  ********/
		/************************************/
		// create temporary genes
		memcpy(chrop1_gen,chrom[p1].genes,GB->num_genes*sizeof(gene));
		memcpy(chrop2_gen,chrom[p2].genes,GB->num_genes*sizeof(gene));

		if(RandomDouble() < crossprob){
			crossover(chrop1_gen,chrop2_gen,GB->num_genes,GB->intragenes);
		}

		/************************************/
		/****** MUTATION OPERATOR  ********/
		/************************************/
		num_genes_wo_sc = GB->num_genes-FA->nflxsc_real;

		mutate(chrop1_gen,GB->num_genes-FA->nflxsc_real,mutprob);
		k=0;
		for(j=0;j<FA->nflxsc;j++){
			if(residue[FA->flex_res[j].inum].trot != 0){
				if(RandomDouble() < FA->flex_res[j].prob){
					mutate(&chrop1_gen[num_genes_wo_sc+k],1,mutprob);
				}
				k++;
			}
		}

		mutate(chrop2_gen,GB->num_genes-FA->nflxsc_real,mutprob);
		k=0;
		for(j=0;j<FA->nflxsc;j++){
			if(residue[FA->flex_res[j].inum].trot != 0){
				if(RandomDouble() < FA->flex_res[j].prob){
					mutate(&chrop2_gen[num_genes_wo_sc+k],1,mutprob);
				}
				k++;
			}
		}

		for(j=0; j<GB->num_genes; j++){
			chrop1_gen[j].to_ic = genetoic(&gene_lim[j],chrop1_gen[j].to_int32);
			chrop2_gen[j].to_ic = genetoic(&gene_lim[j],chrop2_gen[j].to_int32);
		}

		std::string sig1 = generate_sig(chrop1_gen,GB->num_genes);
		std::string sig2 = generate_sig(chrop2_gen,GB->num_genes);

		/************************************/
		/****** CHECK DUPLICATION  ********/
		/************************************/
		if(GB->duplicates || duplicates.find(sig1) == duplicates.end()){

			/*
			  if(!FA->useflexdee ||
			  cmp_chrom2rotlist(FA->psFlexDEENode,chrom,gene_lim,num_genes_wo_sc,
			  FA->nflxsc_real,GB->num_chrom,FA->FlexDEE_Nodes)==0){
			*/

			//nrejected += filter_deelig(FA,GB,chrom,chrop1_gen,GB->num_chrom+i,atoms,gene_lim,dice);
			memcpy(chrom[GB->num_chrom+i].genes,chrop1_gen,GB->num_genes*sizeof(gene));

			chrom[GB->num_chrom+i].cf=eval_chromosome(FA,GB,VC,gene_lim,atoms,residue,cleftgrid,
								  chrom[GB->num_chrom+i].genes,target);
			chrom[GB->num_chrom+i].evalue=get_cf_evalue(&chrom[GB->num_chrom+i].cf);
			chrom[GB->num_chrom+i].app_evalue=get_apparent_cf_evalue(&chrom[GB->num_chrom+i].cf);
			chrom[GB->num_chrom+i].status='n';

			duplicates[sig1] = 1;
			i++;
		}

		if(i==nnew) break;

		if(GB->duplicates || duplicates.find(sig2) == duplicates.end()){

			/*
			  if(!FA->useflexdee ||
			  cmp_chrom2rotlist(FA->psFlexDEENode,chrom,gene_lim,num_genes_wo_sc,
			  FA->nflxsc_real,GB->num_chrom,FA->FlexDEE_Nodes)==0){
			*/
			//nrejected += filter_deelig(FA,GB,chrom,chrop2_gen,GB->num_chrom+i,atoms,gene_lim,dice);
			memcpy(chrom[GB->num_chrom+i].genes,chrop2_gen,GB->num_genes*sizeof(gene));

			chrom[GB->num_chrom+i].cf=eval_chromosome(FA,GB,VC,gene_lim,atoms,residue,cleftgrid,
								  chrom[GB->num_chrom+i].genes,target);
			chrom[GB->num_chrom+i].evalue=get_cf_evalue(&chrom[GB->num_chrom+i].cf);
			chrom[GB->num_chrom+i].app_evalue=get_apparent_cf_evalue(&chrom[GB->num_chrom+i].cf);
			chrom[GB->num_chrom+i].status='n';

			duplicates[sig2] = 1;
			i++;
		}
	}

	if(strcmp(repmodel,"STEADY")==0){
		// replace the n individuals from the old population with the new one (elitism)
		QuickSort(chrom,0,GB->num_chrom-1,true);
		for(i=0;i<nnew;i++) chrom[GB->num_chrom-1-i]=chrom[GB->num_chrom+i];
		calculate_fitness(FA,GB,VC,chrom,gene_lim,atoms,residue,cleftgrid,
				  GB->fitness_model,GB->num_chrom,print,target);
	}else if(strcmp(repmodel,"BOOM")==0){
		// merge and sort both merged populations
		calculate_fitness(FA,GB,VC,chrom,gene_lim,atoms,residue,cleftgrid,
				  GB->fitness_model,GB->num_chrom+nnew,print,target);
	}

	//printf("number of conformers rejected: %d\n", nrejected);

	return nrejected;
}

std::string generate_sig(gene genes[], int num_genes){
	std::stringstream ss;
	for(int i=0;i<num_genes;i++)
		ss << (int)(genes[i].to_ic+0.5) << "/";
	return ss.str();
}

/***********************************************************************/
/* 1         2         3         4         5         6          */
/*234567890123456789012345678901234567890123456789012345678901234567890*/
/* 1         2         3         4         5         6         7*/
/***********************************************************************/
int filter_deelig(FA_Global* FA, GB_Global* GB, chromosome* chrom, gene* genes, int ci, atom* atoms, const genlim* gene_lim,
		   std::function<int32_t()> & dice)
{
	int nrejected = 0;

	if(FA->deelig_flex && FA->nflexbonds){

		int j,deelig_list[100];

		for(j=1; j<=FA->resligand->fdih; j++)
			deelig_list[j] = -1000;

		for(j=0; j<GB->num_genes; j++)
			if(FA->map_par[j].typ == 2 && FA->map_par[j].bnd != -1)
				deelig_list[FA->map_par[j].bnd] = (int)(genes[j].to_ic+0.5);

		/*
		printf("searched deelig list = [");
		for(int k=1; k<=FA->resligand->fdih; k++){
			printf("%d,", deelig_list[k]);
		}
		printf("]\n");
		*/

		if(deelig_search(FA->deelig_root_node, deelig_list, FA->resligand->fdih)){
			/*
			printf("conformer rejected:");
			for(j=1; j<=FA->resligand->fdih; j++)
				printf("%d ", deelig_list[j]);
			printf("\n");
			getchar();
			*/

			// generate a new conformer until the conformer
			// has not already been assigned as 'clashing conformer'
			// and is also not a duplicate
			do{
				nrejected++;

				// only generate a new conformer, do not modify other variables
				generate_random_individual(FA,GB,atoms,genes,gene_lim,dice,
							   FA->map_par_flexbond_first_index,
							   FA->map_par_flexbond_first_index+FA->nflexbonds);

				for(j=1; j<=FA->resligand->fdih; j++)
					deelig_list[j] = -1000;

				for(j=0; j<GB->num_genes; j++)
					if(FA->map_par[j].typ == 2 && FA->map_par[j].bnd != -1)
						deelig_list[FA->map_par[j].bnd] = (int)(genes[j].to_ic+0.5);
				/*
				printf("searched do-while deelig list = [");
				for(int k=1; k<=FA->resligand->fdih; k++){
					printf("%d,", deelig_list[k]);
				}
				printf("]\n");
				*/

				/*
				if(deelig_search(FA->deelig_root_node, deelig_list, FA->resligand->fdih)){
					printf("do-while conformer rejected:");
					for(j=1; j<=FA->resligand->fdih; j++)
						printf("%d ", deelig_list[j]);
					printf("\n");
					getchar();
				}
				*/
			}while((!GB->duplicates && cmp_chrom2pop(chrom,genes,GB->num_genes,0,ci)) ||
			       (deelig_search(FA->deelig_root_node, deelig_list, FA->resligand->fdih)));
		}

	}

	return nrejected;
}

/***********************************************************************/
/* 1         2         3         4         5         6          */
/*234567890123456789012345678901234567890123456789012345678901234567890*/
/* 1         2         3         4         5         6         7*/
/***********************************************************************/
int deelig_search(struct deelig_node_struct* root_node, int* deelig_list, int fdih)
{
	std::map<int, struct deelig_node_struct*>::iterator it;
	struct deelig_node_struct* node = root_node;

	for(int i=1; i<=fdih; i++){
		//printf("[%d]: searching %d\n", i, deelig_list[i]);
		if((it=node->childs.find(deelig_list[i])) != node->childs.end() ||
		   (deelig_list[i] != -1000 && (it=node->childs.find(-1000)) != node->childs.end())){
			//printf("found %d\n", it->first);
			node = it->second;
		}else{
			//printf("not found %d\n", deelig_list[i]);
			return(0);
		}
	}

	return(1);
}

/***********************************************************************/
/* 1         2         3         4         5         6          */
/*234567890123456789012345678901234567890123456789012345678901234567890*/
/* 1         2         3         4         5         6         7*/
/***********************************************************************/
int roullete_wheel(const chromosome* chrom,int n){
	double r;
	double tot=0.0;
	int i;

	for(i=0;i<n;i++){tot += chrom[i].fitnes;}
	//printf("tot=%f\n",tot);
	//PAUSE;

	r=RandomDouble()*tot;

	i=0;
	tot=0.0;
	while(tot <= r){
		tot += chrom[i].fitnes;
		i++;
	}
	//printf("r=%f tot=%f i=%d\n",r,tot,i);
	i--;

	//PAUSE;

	return i;
}
/***********************************************************************/
/* 1         2         3         4         5         6          */
/*234567890123456789012345678901234567890123456789012345678901234567890*/
/* 1         2         3         4         5         6         7*/
/***********************************************************************/
void calculate_fitness(FA_Global* FA,GB_Global* GB,VC_Global* VC,chromosome* chrom, const genlim* gene_lim,
                       atom* atoms,resid* residue,gridpoint* cleftgrid,char method[], int pop_size, int print,
                       cfstr (*target)(FA_Global*,VC_Global*,atom*,resid*,gridpoint*,int,double*)){

	static int gen_id = 0;
	int i;

	// ── Chromosome evaluation ────────────────────────────────────────────────
	// Runtime dispatch: CUDA GPU → Metal GPU → OpenMP CPU (thread-safe).
	// All compiled-in backends are available simultaneously; select_backend()
	// picks the best one at runtime based on detected hardware.

#if defined(FLEXAIDS_USE_CUDA) || defined(FLEXAIDS_USE_METAL)
	// Helper lambda: sample each energy-matrix density function at n_samples
	// evenly-spaced x values in [0, 1] and pack into a flat float array
	// [n_types × n_types × n_samples] for GPU upload.
	// When Eigen is available, the x-value linspace is built via Eigen::ArrayXd
	// for vectorised construction; the get_yval evaluation loop is then
	// auto-vectorisable because it operates on a contiguous double buffer.
	auto build_emat_sampled = [&](int n_types, int n_samples) -> std::vector<float> {
		const size_t total = static_cast<size_t>(n_types) * n_types * n_samples;
		std::vector<float> out(total, 0.0f);

#ifdef FLEXAIDS_HAS_EIGEN
		// Build the x-sample linspace via Eigen (vectorised).
		Eigen::ArrayXd xs = Eigen::ArrayXd::LinSpaced(n_samples, 0.0, 1.0);
		for (int t1 = 0; t1 < n_types; ++t1) {
			for (int t2 = 0; t2 < n_types; ++t2) {
				struct energy_matrix* em = &FA->energy_matrix[t1 * n_types + t2];
				if (em->energy_values == NULL) continue;
				float* dst = &out[(t1 * n_types + t2) * n_samples];
				for (int k = 0; k < n_samples; ++k)
					dst[k] = static_cast<float>(get_yval(em, xs[k]));
			}
		}
#else
		for (int t1 = 0; t1 < n_types; ++t1) {
			for (int t2 = 0; t2 < n_types; ++t2) {
				struct energy_matrix* em = &FA->energy_matrix[t1 * n_types + t2];
				if (em->energy_values == NULL) continue;
				for (int k = 0; k < n_samples; ++k) {
					double x = static_cast<double>(k) / (n_samples - 1);
					out[(t1 * n_types + t2) * n_samples + k] =
						static_cast<float>(get_yval(em, x));
				}
			}
		}
#endif
		return out;
	};

	// Helper lambda: pack gene internal coordinates into a flat array for GPU.
	auto pack_genes_batch = [&](int n_genes) -> std::vector<double> {
		std::vector<double> h_genes(pop_size * n_genes, 0.0);
		for (int c = 0; c < pop_size; ++c)
			for (int g = 0; g < n_genes; ++g)
				h_genes[c * n_genes + g] = chrom[c].genes[g].to_ic;
		return h_genes;
	};

	// Helper lambda: unpack GPU batch results into chromosome CF structures.
	auto unpack_gpu_results = [&](const std::vector<double>& h_com,
	                              const std::vector<double>& h_wal,
	                              const std::vector<double>& h_sas) {
		for (int c = 0; c < pop_size; ++c) {
			if (chrom[c].status != 'n') {
				chrom[c].cf.com    = h_com[c];
				chrom[c].cf.wal    = h_wal[c];
				chrom[c].cf.sas    = h_sas[c];
				chrom[c].cf.con    = 0.0;
				chrom[c].cf.totsas = 0.0;
				chrom[c].cf.rclash = (h_wal[c] > 1e4) ? 1 : 0;
				chrom[c].evalue     = get_cf_evalue(&chrom[c].cf);
				chrom[c].app_evalue = get_apparent_cf_evalue(&chrom[c].cf);
				chrom[c].status    = 'n';
			}
		}
	};

	// Helper lambda: prepare GPU atom data arrays from the atoms array.
	struct GPUAtomData {
		std::vector<float> xyz;
		std::vector<int>   type;
		std::vector<float> radius;
		int lig_first;
		int lig_last;
	};
	auto prepare_gpu_atoms = [&]() -> GPUAtomData {
		const int n_atoms = FA->atm_cnt_real;
		GPUAtomData d;
		d.xyz.resize(n_atoms * 3);
		d.type.resize(n_atoms);
		d.radius.resize(n_atoms);
		for (int a = 0; a < n_atoms; ++a) {
			d.xyz[a*3+0] = atoms[a].coor[0];
			d.xyz[a*3+1] = atoms[a].coor[1];
			d.xyz[a*3+2] = atoms[a].coor[2];
			d.type[a]    = atoms[a].type - 1;  // 1-based → 0-based
			d.radius[a]  = atoms[a].radius;
		}
		d.lig_first = (FA->resligand && FA->resligand->fatm)
		            ? FA->resligand->fatm[0] : 0;
		d.lig_last  = (FA->resligand && FA->resligand->latm)
		            ? FA->resligand->latm[0] : 0;
		return d;
	};
#endif  // FLEXAIDS_USE_CUDA || FLEXAIDS_USE_METAL

	// Log dispatch decision on first call.
	static bool dispatch_logged = false;
	[[maybe_unused]] const auto backend = flexaids::select_backend();
	if (!dispatch_logged) {
		auto report = flexaids::get_dispatch_report();
		fprintf(stderr, "[FlexAIDdS] Hardware dispatch: %s (%s)\n",
		        flexaids::backend_name(report.selected), report.reason.c_str());
		dispatch_logged = true;
	}

	[[maybe_unused]] bool gpu_handled = false;

#ifdef FLEXAIDS_USE_CUDA
	if (backend == flexaids::HardwareBackend::CUDA) {
		// Persistent CUDA context: atom data uploaded once; re-init only when
		// the system geometry changes (different run or atom count changes).
		static CudaEvalCtx* s_cuda_ctx    = nullptr;
		static int           s_cuda_natom = 0;
		static int           s_cuda_ntype = 0;

		const int n_atoms = FA->atm_cnt_real;
		const int n_types = FA->ntypes;
		const int n_genes = GB->num_genes;
		const int ns      = CUDA_EMAT_SAMPLES;

		if (!s_cuda_ctx || s_cuda_natom != n_atoms || s_cuda_ntype != n_types) {
			if (s_cuda_ctx) cuda_eval_shutdown(s_cuda_ctx);

			auto ad = prepare_gpu_atoms();
			std::vector<float> h_emat = build_emat_sampled(n_types, ns);

			s_cuda_ctx    = cuda_eval_init(n_atoms, n_types, MAX_NUM_CHROM,
			                               n_genes, ad.lig_first, ad.lig_last,
			                               FA->permeability,
			                               ad.xyz.data(), ad.type.data(),
			                               ad.radius.data(), h_emat.data());
			s_cuda_natom  = n_atoms;
			s_cuda_ntype  = n_types;
		}

		std::vector<double> h_genes = pack_genes_batch(n_genes);
		std::vector<double> h_com(pop_size), h_wal(pop_size), h_sas(pop_size);
		cuda_eval_batch(s_cuda_ctx, pop_size, n_genes, h_genes.data(),
		                h_com.data(), h_wal.data(), h_sas.data());
		unpack_gpu_results(h_com, h_wal, h_sas);
		gpu_handled = true;
	}
#endif

#ifdef FLEXAIDS_USE_METAL
	if (!gpu_handled && backend == flexaids::HardwareBackend::METAL) {
		// Persistent Metal context (same caching strategy as CUDA).
		static MetalEvalCtx* s_metal_ctx   = nullptr;
		static int            s_metal_natom = 0;
		static int            s_metal_ntype = 0;

		const int n_atoms = FA->atm_cnt_real;
		const int n_types = FA->ntypes;
		const int n_genes = GB->num_genes;
		const int ns      = METAL_EMAT_SAMPLES;

		if (!s_metal_ctx || s_metal_natom != n_atoms || s_metal_ntype != n_types) {
			if (s_metal_ctx) metal_eval_shutdown(s_metal_ctx);

			auto ad = prepare_gpu_atoms();
			std::vector<float> h_emat = build_emat_sampled(n_types, ns);

			s_metal_ctx   = metal_eval_init(n_atoms, n_types, MAX_NUM_CHROM,
			                                ad.lig_first, ad.lig_last,
			                                FA->permeability,
			                                ad.xyz.data(), ad.type.data(),
			                                ad.radius.data(), h_emat.data(), ns);
			s_metal_natom = n_atoms;
			s_metal_ntype = n_types;
		}

		std::vector<double> h_genes = pack_genes_batch(n_genes);
		std::vector<double> h_com(pop_size), h_wal(pop_size), h_sas(pop_size);
		metal_eval_batch(s_metal_ctx, pop_size, n_genes, h_genes.data(),
		                 h_com.data(), h_wal.data(), h_sas.data());
		unpack_gpu_results(h_com, h_wal, h_sas);
		gpu_handled = true;
	}
#endif

	if (!gpu_handled) {
	// ── Thread-safe CPU path (AVX-512 / AVX2 / OpenMP / scalar) ─────────
	// Each OpenMP thread receives its own private copies of every data
	// structure that Vcontacts/vcfunction/ic2cf writes to:
	//   • atoms[]        – internal coords (dis/ang/dih) and Cartesian (coor)
	//   • residue[]      – rotamer index (.rot)
	//   • FA scratch     – contacts[], contributions[], optres[].cf
	//   • VC workspace   – Calc[], Calclist[], ca_index[], ca_rec[],
	//                      seed[], contlist[], ptorder[], centerpt[],
	//                      poly[], cont[], vedge[]
	// Read-only fields (energy_matrix, map_par, …) are shared.
	// The DEE linked-list update in ic2cf is skipped in parallel mode
	// (guarded by omp_in_parallel() in ic2cf.cpp) to avoid concurrent
	// linked-list corruption; DEE pruning still operates in serial calls.
	{
#ifdef _OPENMP
		const int n_thr = omp_get_max_threads();
#else
		const int n_thr = 1;
#endif
		const int natm  = FA->atm_cnt;
		const int natmr = FA->atm_cnt_real;
		const int nres  = FA->res_cnt;
		const int nopt  = FA->num_optres;
		const int nctb  = FA->ntypes * FA->ntypes;

		// Per-thread mutable atom arrays.
		std::vector<std::vector<atom>>  tl_atoms(n_thr,
		    std::vector<atom>(atoms, atoms + natm));
		// Per-thread residue arrays (pointer fields shared read-only; .rot private).
		std::vector<std::vector<resid>> tl_res(n_thr,
		    std::vector<resid>(residue, residue + nres));
		// Per-thread FA copies with redirected mutable scratch buffers.
		std::vector<FA_Global>           tl_fa(n_thr, *FA);
		std::vector<std::vector<int>>    tl_contacts(n_thr, std::vector<int>(100000, 0));
		std::vector<std::vector<float>>  tl_contrib(n_thr, std::vector<float>(nctb, 0.0f));
		std::vector<std::vector<OptRes>> tl_optres(n_thr,
		    std::vector<OptRes>(FA->optres, FA->optres + nopt));
		// Per-thread VC workspace (Vcontacts writes all these each call).
		std::vector<VC_Global>               tl_vc(n_thr, *VC);
		std::vector<std::vector<atomsas>>    tl_calc(n_thr, std::vector<atomsas>(natmr));
		std::vector<std::vector<int>>        tl_calclist(n_thr, std::vector<int>(natmr));
		std::vector<std::vector<int>>        tl_caidx(n_thr, std::vector<int>(natmr, -1));
		std::vector<std::vector<ca_struct>>  tl_carec(n_thr,
		    std::vector<ca_struct>(VC->ca_recsize));
		std::vector<std::vector<int>>        tl_seed(n_thr,
		    std::vector<int>(3 * natmr));
		std::vector<std::vector<contactlist>> tl_contlist(n_thr,
		    std::vector<contactlist>(10000));
		std::vector<std::vector<ptindex>>    tl_ptorder(n_thr,
		    std::vector<ptindex>(MAX_PT));
		std::vector<std::vector<vertex>>     tl_centerpt(n_thr,
		    std::vector<vertex>(MAX_PT));
		std::vector<std::vector<vertex>>     tl_poly(n_thr,
		    std::vector<vertex>(MAX_POLY));
		std::vector<std::vector<plane>>      tl_cont(n_thr,
		    std::vector<plane>(MAX_PT));
		std::vector<std::vector<edgevector>> tl_vedge(n_thr,
		    std::vector<edgevector>(MAX_POLY));

		for (int t = 0; t < n_thr; ++t) {
			// Redirect FA mutable scratch to per-thread buffers.
			tl_fa[t].contacts      = tl_contacts[t].data();
			tl_fa[t].contributions = tl_contrib[t].data();
			tl_fa[t].optres        = tl_optres[t].data();
			// Redirect VC mutable workspace to per-thread buffers.
			tl_vc[t].Calc      = tl_calc[t].data();
			tl_vc[t].Calclist  = tl_calclist[t].data();
			tl_vc[t].ca_index  = tl_caidx[t].data();
			tl_vc[t].ca_rec    = tl_carec[t].data();
			tl_vc[t].seed      = tl_seed[t].data();
			tl_vc[t].contlist  = tl_contlist[t].data();
			tl_vc[t].ptorder   = tl_ptorder[t].data();
			tl_vc[t].centerpt  = tl_centerpt[t].data();
			tl_vc[t].poly      = tl_poly[t].data();
			tl_vc[t].cont      = tl_cont[t].data();
			tl_vc[t].vedge     = tl_vedge[t].data();
			// box is shared: if vindex==1 it's pre-built read-only;
			// if vindex==0 Vcontacts will malloc/vcfunction will free per call.
		}

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) default(none) \
	shared(chrom, pop_size, GB, gene_lim, cleftgrid, target, \
	       atoms, residue, FA, VC, \
	       tl_atoms, tl_res, tl_fa, tl_optres, tl_vc, \
	       natm, nres, nopt)
#endif
		for (int ii = 0; ii < pop_size; ++ii) {
			if (chrom[ii].status == 'n') continue;
#ifdef _OPENMP
			const int tid = omp_get_thread_num();
#else
			const int tid = 0;
#endif
			// Reset per-thread state to the reference protein configuration.
			std::copy(atoms,   atoms + natm,   tl_atoms[tid].begin());
			std::copy(residue, residue + nres, tl_res[tid].begin());
			// optres cf fields are cleared by vcfunction itself; pre-clear for safety.
			for (int o = 0; o < nopt; ++o) {
				tl_optres[tid][o].cf.com    = 0.0;
				tl_optres[tid][o].cf.wal    = 0.0;
				tl_optres[tid][o].cf.sas    = 0.0;
				tl_optres[tid][o].cf.totsas = 0.0;
				tl_optres[tid][o].cf.con    = 0.0;
				tl_optres[tid][o].cf.rclash = 0;
			}
			tl_vc[tid].numcarec = 0;

			chrom[ii].cf = eval_chromosome(
			    &tl_fa[tid], GB, &tl_vc[tid], gene_lim,
			    tl_atoms[tid].data(), tl_res[tid].data(),
			    cleftgrid, chrom[ii].genes, target);
			chrom[ii].evalue     = get_cf_evalue(&chrom[ii].cf);
			chrom[ii].app_evalue = get_apparent_cf_evalue(&chrom[ii].cf);
			chrom[ii].status     = 'n';
		}
	}
	}  // !gpu_handled

	QuickSort(chrom,0,pop_size-1,true);

	//print_par(chrom,gene_lim,5,GB->num_genes);
	//PAUSE;
	//chrom_hpsort(pop_size,0,chrom);

	if(strcmp(method,"LINEAR")==0){
		/* the fitness value is a number between 0 and num_chrom.
		   each chromosome is assigned an integer value that
		   corresponds to its position in index_map.
		*/
		for(i=0;i<GB->num_chrom;i++){
			chrom[i].fitnes=(double)(GB->num_chrom-i);
		}
	}

	if(strcmp(method,"PSHARE")==0){
		/* the fitness value is a number between 0 and num_chrom.
		   each chromosome is assigned an integer value that
		   corresponds to its position in index_map. Moreover,
		   each chromosome's fitness is lowered by sharing.
		   The niche count (share) must be accumulated over ALL j before
		   dividing — fixed from the previous per-j assignment bug.
		   The outer loop is data-race free (each i writes only chrom[i].fitnes)
		   and is parallelised with OpenMP.
		*/
#ifdef _OPENMP
#pragma omp parallel for schedule(static) default(none) \
	shared(chrom, GB, FA, cleftgrid)
#endif
		for(int pi=0; pi<GB->num_chrom; pi++){
			double pshare = 0.0;
			for(int pj=0; pj<GB->num_chrom; pj++){
				double prmsp = calc_rmsp(GB->num_genes,
				                         chrom[pi].genes, chrom[pj].genes,
				                         FA->map_par, cleftgrid);
				if(prmsp <= GB->sig_share){
					pshare += (1.0 - pow((prmsp/GB->sig_share), GB->alpha));
				}
			}
			// Assign fitness AFTER accumulating the full niche count.
			chrom[pi].fitnes = (double)(GB->num_chrom - pi) / pshare;
		}
	}

	if(print){

		FILE* outfile_ptr = get_update_file_ptr(FA);

		if(outfile_ptr == NULL){
			fprintf(stderr,"ERROR: The NRGsuite failed to update within the timeout.\n");
			Terminate(10);
		}

		fprintf(outfile_ptr, "Generation: %5d\n", gen_id);
		fprintf(outfile_ptr, "best by energy\n");

		print_par(chrom,gene_lim,GB->num_print,GB->num_genes, outfile_ptr);

		fflush(outfile_ptr);

		close_update_file_ptr(FA, outfile_ptr);

	}

	gen_id++;

	return;
}

/***********************************************************************/
/* 1         2         3         4         5         6          */
/*234567890123456789012345678901234567890123456789012345678901234567890*/
/* 1         2         3         4         5         6         7*/
/***********************************************************************/
void close_update_file_ptr(FA_Global* FA, FILE* outfile_ptr)
{

	if(FA->nrg_suite){
		fclose(outfile_ptr);
	}

}
/***********************************************************************/
/* 1         2         3         4         5         6          */
/*234567890123456789012345678901234567890123456789012345678901234567890*/
/* 1         2         3         4         5         6         7*/
/***********************************************************************/
FILE* get_update_file_ptr(FA_Global* FA)
{

	if(!FA->nrg_suite){
		return stdout;
	}

	FILE* outfile_ptr = NULL;
	char UPDATEFILE[MAX_PATH__];
	long long timeout = 0;

	strcpy(UPDATEFILE,FA->state_path);
#ifdef _WIN32
	strcat(UPDATEFILE,"\\.update");
#else
	strcat(UPDATEFILE,"/.update");
#endif

	outfile_ptr = fopen(UPDATEFILE,"r");
	if(outfile_ptr != NULL) {
		do {
			fclose(outfile_ptr);

# ifdef _WIN32
			Sleep(SLEEP);
# else
			usleep(SLEEP*1000);
# endif

			timeout += SLEEP;
			if(timeout >= FA->nrg_suite_timeout*1000){
				return NULL;
			}

			outfile_ptr = fopen(UPDATEFILE,"r");

		}while(outfile_ptr != NULL);
	}

	outfile_ptr = fopen(UPDATEFILE,"w");
	if(outfile_ptr == NULL){
		fprintf(stderr,"ERROR: Cannot open update file '%s' for reading.\n", UPDATEFILE);
		Terminate(10);
	}

	return outfile_ptr;

}

/***********************************************************************/
/* 1         2         3         4         5         6          */
/*234567890123456789012345678901234567890123456789012345678901234567890*/
/* 1         2         3         4         5         6         7*/
/***********************************************************************/
cfstr eval_chromosome(FA_Global* FA,GB_Global* GB,VC_Global* VC,const genlim* gene_lim,
		      atom* atoms,resid* residue,gridpoint* cleftgrid,gene* john,
		      cfstr (*function)(FA_Global*,VC_Global*,atom*,resid*,gridpoint*,int,double*)){

	double icv[MAX_NUM_GENES] = {0};

	for(int i=0;i<GB->num_genes;i++){
		if(john[i].to_ic > gene_lim[i].max) {
			fprintf(stderr, "Exceptional out of bounds error at: max: %.5lf when ic: %.5lf\n", gene_lim[i].max, john[i].to_ic);
			john[i].to_ic = gene_lim[i].max;
		}else if(john[i].to_ic < gene_lim[i].min) {
			fprintf(stderr, "Exceptional out of bounds error at: min: %.5lf when ic: %.5lf\n", gene_lim[i].max, john[i].to_ic);
			john[i].to_ic = gene_lim[i].min;
		}

		icv[i] = john[i].to_ic;
	}

	return (*function)(FA,VC,atoms,residue,cleftgrid,GB->num_genes,icv);
}

/***********************************************************************/
/* 1         2         3         4         5         6          */
/*234567890123456789012345678901234567890123456789012345678901234567890*/
/* 1         2         3         4         5         6         7*/
/***********************************************************************/
void generate_random_individual(FA_Global* FA, GB_Global* GB, atom* atoms, gene* genes, const genlim* gene_lim,
				std::function<int32_t()> & dice,
				int from_gene, int to_gene)
{
	for(int j=from_gene;j<to_gene;j++)
	{
		// side-chain optimization
		if(FA->map_par[j].typ == 4)
		{
			int l=0;
			while(FA->flex_res[l].inum != atoms[FA->map_par[j].atm].ofres){
				l++;
			};

			//printf("probability of atom[%d].ofres[%d]\t flex_res[%d](%s).inum[%d]= %.3f\n", FA->map_par[j].atm, atoms[FA->map_par[j].atm].ofres, l, FA->flex_res[l].name, FA->flex_res[l].inum, FA->flex_res[l].prob);

			if(RandomDouble() < FA->flex_res[l].prob)
			{
				genes[j].to_int32 = dice();
			}else{
				genes[j].to_int32 = 0;
			}
		}else{
			genes[j].to_int32 = dice();
		}

		genes[j].to_ic = genetoic(&gene_lim[j],genes[j].to_int32);
	}

	return;
}

/***********************************************************************/
/* 1         2         3         4         5         6          */
/*234567890123456789012345678901234567890123456789012345678901234567890*/
/* 1         2         3         4         5         6         7*/
/***********************************************************************/
void populate_chromosomes(FA_Global* FA,GB_Global* GB,VC_Global* VC,chromosome* chrom, const genlim* gene_lim,
                          atom* atoms,resid* residue,gridpoint* cleftgrid,char method[],
                          cfstr (*target)(FA_Global*,VC_Global*,atom*,resid*,gridpoint*,int,double*),
                          char file[], long int at, int popoffset, int print,
                          std::function<int32_t()> & dice,
                          std::map<std::string, int> & duplicates){

	int i,j;

	/*
	  std::mt19937 rng;
	  std::uniform_int_distribution<int32_t> one_to_max_int32( 1, MAX_RANDOM_VALUE );
	  std::function<int32_t()> dice = [&](){ return one_to_max_int32(rng); };
	*/

	FILE* infile_ptr = NULL;

	// initialise genes to zero
	for(i=popoffset;i<GB->num_chrom;i++){
		for(j=0;j<GB->num_genes;j++){
			chrom[i].genes[j].to_int32=0;
			chrom[i].genes[j].to_ic=0.0;
		}
	}

	//------------------------------------------------------------------------------
	// use method to create new genes
	if(strcmp(method,"RANDOM")==0){
		printf("generating random population...\n");
		//printf("num_chrom=%d num_genes=%d\n",GB->num_chrom,GB->num_genes);

		int gener=0;
		std::string sig;

		i=popoffset;
		while(i<GB->num_chrom){
			while(1){
				generate_random_individual(FA,GB,atoms,chrom[i].genes,gene_lim,dice,0,GB->num_genes);
				sig = generate_sig(chrom[i].genes,GB->num_genes);
				if(GB->duplicates || duplicates.find(sig) == duplicates.end()){
					break;
				}
			}

			gener++;
			i++;
			duplicates[sig] = 1;
		}

		printf("generated %d randomized individuals\n", gener);

	}

	//------------------------------------------------------------------------------

	if(strcmp(method,"IPFILE")==0){
		printf("generating population from file...\n");

		if(!OpenFile_B(file,"rb",&infile_ptr)){
			fprintf(stderr,"ERROR: Cannot open file '%s' for reading.\n", file);
			Terminate(8);
		}

		fseek(infile_ptr, at, SEEK_SET);

		i=0;
		j=0;
		while(i<GB->num_chrom && fread(&chrom[i].genes[j].to_int32, 1, sizeof(int32_t), infile_ptr))
		{
			chrom[i].genes[j].to_ic = genetoic(&gene_lim[j],chrom[i].genes[j].to_int32);

			j++;
			if(j==GB->num_genes){
				i++;
				j=0;
			}
		}

		CloseFile_B(&infile_ptr,"r");

		printf("generated %d individuals from file\n", i);

		// reset to RANDOM afterwards
		strcpy(method,"RANDOM");

		// complete remaining population when necessary
		return populate_chromosomes(FA,GB,VC,chrom,gene_lim,atoms,residue,
					    cleftgrid,GB->pop_init_method,target,
					    GB->pop_init_file,at,i,print,dice,duplicates);
	}

	//------------------------------------------------------------------------------

	// calculate evalue for each chromosome — thread-safe OpenMP parallel eval.
	// Uses the same per-thread VC/atoms/FA workspace strategy as calculate_fitness.
	{
#ifdef _OPENMP
		const int n_thr = omp_get_max_threads();
#else
		const int n_thr = 1;
#endif
		const int natm  = FA->atm_cnt;
		const int natmr = FA->atm_cnt_real;
		const int nres  = FA->res_cnt;
		const int nopt  = FA->num_optres;
		const int nctb  = FA->ntypes * FA->ntypes;
		const int range = GB->num_chrom - popoffset;

		std::vector<std::vector<atom>>   p_atoms(n_thr, std::vector<atom>(atoms, atoms + natm));
		std::vector<std::vector<resid>>  p_res(n_thr, std::vector<resid>(residue, residue + nres));
		std::vector<FA_Global>           p_fa(n_thr, *FA);
		std::vector<std::vector<int>>    p_contacts(n_thr, std::vector<int>(100000, 0));
		std::vector<std::vector<float>>  p_contrib(n_thr, std::vector<float>(nctb, 0.0f));
		std::vector<std::vector<OptRes>> p_optres(n_thr,
		    std::vector<OptRes>(FA->optres, FA->optres + nopt));
		std::vector<VC_Global>               p_vc(n_thr, *VC);
		std::vector<std::vector<atomsas>>    p_calc(n_thr, std::vector<atomsas>(natmr));
		std::vector<std::vector<int>>        p_calclist(n_thr, std::vector<int>(natmr));
		std::vector<std::vector<int>>        p_caidx(n_thr, std::vector<int>(natmr, -1));
		std::vector<std::vector<ca_struct>>  p_carec(n_thr,
		    std::vector<ca_struct>(VC->ca_recsize));
		std::vector<std::vector<int>>        p_seed(n_thr, std::vector<int>(3 * natmr));
		std::vector<std::vector<contactlist>> p_contlist(n_thr, std::vector<contactlist>(10000));
		std::vector<std::vector<ptindex>>    p_ptorder(n_thr, std::vector<ptindex>(MAX_PT));
		std::vector<std::vector<vertex>>     p_centerpt(n_thr, std::vector<vertex>(MAX_PT));
		std::vector<std::vector<vertex>>     p_poly(n_thr, std::vector<vertex>(MAX_POLY));
		std::vector<std::vector<plane>>      p_cont(n_thr, std::vector<plane>(MAX_PT));
		std::vector<std::vector<edgevector>> p_vedge(n_thr, std::vector<edgevector>(MAX_POLY));

		for (int t = 0; t < n_thr; ++t) {
			p_fa[t].contacts      = p_contacts[t].data();
			p_fa[t].contributions = p_contrib[t].data();
			p_fa[t].optres        = p_optres[t].data();
			p_vc[t].Calc      = p_calc[t].data();
			p_vc[t].Calclist  = p_calclist[t].data();
			p_vc[t].ca_index  = p_caidx[t].data();
			p_vc[t].ca_rec    = p_carec[t].data();
			p_vc[t].seed      = p_seed[t].data();
			p_vc[t].contlist  = p_contlist[t].data();
			p_vc[t].ptorder   = p_ptorder[t].data();
			p_vc[t].centerpt  = p_centerpt[t].data();
			p_vc[t].poly      = p_poly[t].data();
			p_vc[t].cont      = p_cont[t].data();
			p_vc[t].vedge     = p_vedge[t].data();
		}

		(void)range;  // suppress unused warning when _OPENMP not defined

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) default(none) \
	shared(chrom, FA, GB, VC, gene_lim, atoms, residue, cleftgrid, target, \
	       popoffset, p_atoms, p_res, p_fa, p_optres, p_vc, natm, nres, nopt)
#endif
		for(i=popoffset;i<GB->num_chrom;i++){
#ifdef _OPENMP
			const int tid = omp_get_thread_num();
#else
			const int tid = 0;
#endif
			std::copy(atoms,   atoms + natm,   p_atoms[tid].begin());
			std::copy(residue, residue + nres, p_res[tid].begin());
			for (int o = 0; o < nopt; ++o) {
				p_optres[tid][o].cf.com    = 0.0;
				p_optres[tid][o].cf.wal    = 0.0;
				p_optres[tid][o].cf.sas    = 0.0;
				p_optres[tid][o].cf.totsas = 0.0;
				p_optres[tid][o].cf.con    = 0.0;
				p_optres[tid][o].cf.rclash = 0;
			}
			p_vc[tid].numcarec = 0;

			chrom[i].cf = eval_chromosome(
			    &p_fa[tid], GB, &p_vc[tid], gene_lim,
			    p_atoms[tid].data(), p_res[tid].data(),
			    cleftgrid, chrom[i].genes, target);
			chrom[i].evalue     = get_cf_evalue(&chrom[i].cf);
			chrom[i].app_evalue = get_apparent_cf_evalue(&chrom[i].cf);
			chrom[i].status     = 'n';
		}
	}

	// sort and calculate fitness
	calculate_fitness(FA,GB,VC,chrom,gene_lim,atoms,residue,cleftgrid,GB->fitness_model,GB->num_chrom,print,target);

	return;
}
/***********************************************************************/
/* 1         2         3         4         5         6          */
/*234567890123456789012345678901234567890123456789012345678901234567890*/
/* 1         2         3         4         5         6         7*/
/***********************************************************************/
int cmp_chrom2rotlist(psFlexDEE_Node psFlexDEE_INI_Node, const chromosome* chrom, const genlim* gene_lim,
                      int gene_offset, int num_genes, int tot, int num_nodes){

	int   par[100];
	//int* genes = NULL;
	sFlexDEE_Node sFlexDEENode;

	memset(&par,0,sizeof(par));

	if ( psFlexDEE_INI_Node == NULL ) { return 0; }


	sFlexDEENode.rotlist = par;

	for(int i=0;i<tot;i++){
		//genes = &chrom[i].genesic[gene_offset];

		psFlexDEE_INI_Node = psFlexDEE_INI_Node->last;

		if ( dee_pivot(&sFlexDEENode,&psFlexDEE_INI_Node,1,num_nodes,(num_nodes+1)/2,num_nodes,num_genes) == 0 ) { return 1; }

	}

	return 0;
}
/***********************************************************************/
/* 1         2         3         4         5         6          */
/*234567890123456789012345678901234567890123456789012345678901234567890*/
/* 1         2         3         4         5         6         7*/
/***********************************************************************/
int cmp_chrom2pop(const chromosome* chrom,const gene* genes, int num_genes,int start, int last){
	int i,j,flag;

	for(i=start;i<last;i++){
		flag=0;
		for(j=0;j<num_genes;j++){
			//printf("individuals[%d][%d].gene[%d]=%.3f\t%.3f\n", start-1, i, j,
			//       genes[j].to_ic, chrom[i].genes[j].to_ic);
			flag += abs(genes[j].to_ic - chrom[i].genes[j].to_ic) < 0.1;
		}

		//printf("flag=%d\n",flag);
		if(flag == num_genes){return 1;}
	}

	return 0;
}
/***********************************************************************/
/* 1         2         3         4         5         6          */
/*234567890123456789012345678901234567890123456789012345678901234567890*/
/* 1         2         3         4         5         6         7*/
/***********************************************************************/
int cmp_chrom2pop_int(const chromosome* chrom,const gene* genes, int num_genes,int start, int last){
	int i,j,flag;

	for(i=start;i<last;i++){
		flag=0;
		for(j=0;j<num_genes;j++){
			//printf("comparing %u to %u\n",c->genes[j],chrom[i].genes[j]);
			flag += ( genes[j].to_int32 == chrom[i].genes[j].to_int32 );
		}

		//printf("flag=%d\n",flag);
		if(flag == num_genes){return 1;}
	}

	return 0;
}
/***********************************************************************/
/* 1         2         3         4         5         6          */
/*234567890123456789012345678901234567890123456789012345678901234567890*/
/* 1         2         3         4         5         6         7*/
/***********************************************************************/
void validate_dups(GB_Global* GB, genlim* gene_lim, int num_genes){

	double n_poss = calc_poss(gene_lim, num_genes);

	if(n_poss < (double)GB->num_chrom && !GB->duplicates){
		fprintf(stderr,"Too many chromosomes for the number of possibilites (%.1lf) when no duplicates allowed.\n", n_poss);
		fprintf(stderr,"Duplicates are then allowed.\n");
		GB->duplicates = 1;
	}

	return;
}

double calc_poss(genlim* gene_lim, int num_genes){

	double n_poss = 0.0;

	for(int i=0; i<num_genes; i++){
		if(n_poss > 0.0){
			n_poss *= gene_lim[i].nbin;
		}else{
			n_poss = gene_lim[i].nbin;
		}
	}

	return n_poss;
}

void set_bins(genlim* gene_lim, int num_genes){

	for(int i=0; i<num_genes; i++){
		double nbin = (gene_lim[i].max - gene_lim[i].min) / gene_lim[i].del;
		if(nbin - (int)nbin > 0.0){ nbin += 1.0; }
		if(gene_lim[i].map){ nbin += 1.0; }

		gene_lim[i].bin = 1.0/nbin;
		gene_lim[i].nbin = nbin;
	}

	return;
}

/***********************************************************************/
/* 1         2         3         4         5         6          */
/*234567890123456789012345678901234567890123456789012345678901234567890*/
/* 1         2         3         4         5         6         7*/
/***********************************************************************/
void set_bins(genlim* gene_lim){

	double nbin = (gene_lim->max - gene_lim->min) / gene_lim->del;
	if(nbin - (int)nbin > 0.0){ nbin += 1.0; }
	if(gene_lim->map){ nbin += 1.0; }

	gene_lim->bin = 1.0/nbin;

	return;
}

/***********************************************************************/
/* 1         2         3         4         5         6          */
/*234567890123456789012345678901234567890123456789012345678901234567890*/
/* 1         2         3         4         5         6         7*/
/***********************************************************************/
void read_gainputs(FA_Global* FA,GB_Global* GB,int* gen_int,int* sz_part,char file[]){

	FILE *infile_ptr;        /* pointer to input file */
	char buffer[MAX_PATH__];         /* a line from the INPUT file */
	char field[9];           /* field names on INPUT file */

	//printf("file here is <%s>\n",file);
	infile_ptr=NULL;
	if(!OpenFile_B(file,"r",&infile_ptr)){
		fprintf(stderr,"ERROR: Cannot find file '%s'.\n", file);
		Terminate(8);
	}

	while (fgets(buffer, sizeof(buffer),infile_ptr)){

		if (buffer[strlen(buffer)-1] == '\n')
			buffer[strlen(buffer)-1] = '\0';
		if (buffer[strlen(buffer)-1] == '\r')
			buffer[strlen(buffer)-1] = '\0';


		if(strncmp(buffer,"NUMCHROM",8) == 0){
			sscanf(buffer,"%s %d",field,&GB->num_chrom);
		}else if(strncmp(buffer,"OPTIGRID",8) == 0){
			sscanf(buffer,"%s %d %d %d",field,&FA->opt_grid,gen_int,sz_part);
		}else if(strncmp(buffer,"NUMGENER",8) == 0){
			sscanf(buffer,"%s %d",field,&GB->max_generations);
		}else if(strncmp(buffer,"ADAPTVGA",8) == 0){
			sscanf(buffer,"%s %d",field,&GB->adaptive_ga);
		}else if(strncmp(buffer,"ADAPTKCO",8) == 0){
			//adaptive response parameters
			//k1-k4 are values ranging from 0.0-1.0 inclusively
			sscanf(buffer,"%s %lf %lf %lf %lf",field,&GB->k1,&GB->k2,&GB->k3,&GB->k4);
		}else if(strncmp(buffer,"CROSRATE",8) == 0){
			sscanf(buffer,"%s %lf",field,&GB->cross_rate);
		}else if(strncmp(buffer,"MUTARATE",8) == 0){
			sscanf(buffer,"%s %lf",field,&GB->mut_rate);
		}else if(strncmp(buffer,"INTRAGEN",8) == 0){
			GB->intragenes = 1;
		}else if(strncmp(buffer,"INIMPROB",8) == 0){
			sscanf(buffer,"%s %lf",field,&GB->ini_mut_prob);
		}else if(strncmp(buffer,"ENDMPROB",8) == 0){
			sscanf(buffer,"%s %lf",field,&GB->end_mut_prob);
		}else if(strncmp(buffer,"POPINIMT",8) == 0){
			sscanf(buffer,"%s %s",field,GB->pop_init_method);
			//0         1         2
			//012345678901234567890123456789
			//POPINIMT IPFILE file.dat
			if(strcmp(GB->pop_init_method,"IPFILE") == 0){
				strcpy(GB->pop_init_file,&buffer[16]);
			}
		}else if(strncmp(buffer,"FITMODEL",8) == 0){
			sscanf(buffer,"%s %s",field,GB->fitness_model);
		}else if(strncmp(buffer,"REPMODEL",8) == 0){
			sscanf(buffer,"%s %s",field,GB->rep_model);
		}else if(strncmp(buffer,"DUPLICAT",8) == 0){
			GB->duplicates = 1;
		}else if(strncmp(buffer,"BOOMFRAC",8) == 0){
			sscanf(buffer,"%s %lf",field,&GB->pbfrac);
		}else if(strncmp(buffer,"STEADNUM",8) == 0){
			sscanf(buffer,"%s %d",field,&GB->ssnum);
		}else if(strncmp(buffer,"SHAREALF",8) == 0){
			sscanf(buffer,"%s %lf",field,&GB->alpha);
		}else if(strncmp(buffer,"SHAREPEK",8) == 0){
			sscanf(buffer,"%s %lf",field,&GB->peaks);
		}else if(strncmp(buffer,"SHARESCL",8) == 0){
			sscanf(buffer,"%s %lf",field,&GB->scale);
		}else if(strncmp(buffer,"OUTGENER",8) == 0){
			GB->outgen = 1;
		}else if(strncmp(buffer,"STRTSEED",8) == 0){
			sscanf(buffer,"%s %d",field,&GB->seed);
		}else if(strncmp(buffer,"PRINTCHR",8) == 0){
			sscanf(buffer,"%s %d",field,&GB->num_print);
		}else if(strncmp(buffer,"PRINTINT",8) == 0){
			sscanf(buffer,"%s %d",field,&GB->print_int);
		}else if(strncmp(buffer,"PRINTRRG",8) == 0){
			sscanf(buffer,"%s %d",field,&GB->rrg_skip);
		}else if(strncmp(buffer,"ENTRCNVG",8) == 0){
			sscanf(buffer,"%s %d",field,&GB->entropy_convergence);
		}else if(strncmp(buffer,"ENTRCHKI",8) == 0){
			sscanf(buffer,"%s %d",field,&GB->entropy_check_interval);
		}else if(strncmp(buffer,"ENTRWIND",8) == 0){
			sscanf(buffer,"%s %d",field,&GB->entropy_window);
		}else if(strncmp(buffer,"ENTRTHRS",8) == 0){
			sscanf(buffer,"%s %lf",field,&GB->entropy_rel_threshold);
		}else{
			// ...
		}

	}

	CloseFile_B(&infile_ptr,"r");

}

long int read_pop_init_file(FA_Global* FA, GB_Global* GB, genlim* gene_lim, char* pop_init_file)
{

	long int at = 0;
	FILE* infile_ptr = NULL;

	if(!OpenFile_B(pop_init_file,"rb",&infile_ptr)){
		fprintf(stderr,"ERROR: Cannot open file '%s' for reading.\n", pop_init_file);
		Terminate(8);
	}

	char genes_tag[6];
	fread(&genes_tag[0], 1, sizeof(genes_tag)-1, infile_ptr);
	genes_tag[5] = '\0';
	//printf("genes_tag=%s\n", genes_tag);

	if(strcmp(genes_tag,"genes") == 0){

		int i=0;
		while(i < GB->num_genes){
			fread(&gene_lim[i], 1, sizeof(genlim), infile_ptr);
			i++;
		}

		char chrom_tag[6];
		fread(&chrom_tag[0], 1, sizeof(chrom_tag)-1, infile_ptr);
		chrom_tag[5] = '\0';
		//printf("chrom_tag=%s\n", chrom_tag);

		if(strcmp(chrom_tag,"chrom") == 0){
			at = ftell(infile_ptr);
		}

	}

	CloseFile_B(&infile_ptr, "r");

	return at;
}

void set_gene_lim(FA_Global* FA, GB_Global* GB, genlim* gene_lim)
{

	for(int ngenes=0;ngenes<GB->num_genes;ngenes++){
		gene_lim[ngenes].min=FA->min_opt_par[ngenes];
		gene_lim[ngenes].max=FA->max_opt_par[ngenes];
		gene_lim[ngenes].del=FA->del_opt_par[ngenes];
		gene_lim[ngenes].map=FA->map_opt_par[ngenes];

		printf("gene %d: min: %10.2f max: %10.2f delta: %10.2f map: %d\n", ngenes,
		       gene_lim[ngenes].min,
		       gene_lim[ngenes].max,
		       gene_lim[ngenes].del,
		       gene_lim[ngenes].map);

	}

	return;
}

/***********************************************************************/
/* 1         2         3         4         5         6          */
/*234567890123456789012345678901234567890123456789012345678901234567890*/
/* 1         2         3         4         5         6         7*/
/***********************************************************************/
void crossover(gene *john,gene *mary,int num_genes, int intragenes){

	/* john and mary are two chromosomes to be crossover at points a and b
	 */

	/*
	printf("john before:\n");
	print_chrom(john,num_genes,0);
	printf("mary before:\n");
	print_chrom(mary,num_genes,0);
	*/

	int i,j;
	int optr,temp;
	int gen_a,gen_b,aux_gen;
	int pnt_a,pnt_b,aux_pnt;

	gen_a=(int)(RandomDouble()*(double)num_genes);
	gen_b=(int)(RandomDouble()*(double)num_genes);
	//printf("gen_a=%d\tgen_b=%d\n",gen_a,gen_b);



	if(intragenes){
		pnt_a=(int)(RandomDouble()*(double)(MAX_GEN_LENGTH));
		pnt_b=(int)(RandomDouble()*(double)(MAX_GEN_LENGTH));

		if(gen_a > gen_b){
			aux_gen=gen_a;
			aux_pnt=pnt_a;
			gen_a=gen_b;
			pnt_a=pnt_b;
			gen_b=aux_gen;
			pnt_b=aux_pnt;
		}

		if(gen_a == gen_b && pnt_a < pnt_b){
			aux_pnt=pnt_b;
			pnt_b=pnt_a;
			pnt_a=aux_pnt;
		}
	}else{
		if(gen_a > gen_b){
			aux_gen=gen_a;
			gen_a=gen_b;
			gen_b=aux_gen;
		}

		if(gen_a != gen_b){
			// find left of right bound of gene a
			if(RandomDouble() < 0.5){
				pnt_a=MAX_GEN_LENGTH;
			}else{
				pnt_a=0;
			}

			// find left of right bound of gene b
			if(RandomDouble() < 0.5){
				pnt_b=MAX_GEN_LENGTH;
			}else{
				pnt_b=0;
			}

			if((gen_b - gen_a) == 1 && pnt_a == 0 && pnt_b == MAX_GEN_LENGTH){
				pnt_b=pnt_a;
			}
		}else{
			pnt_a=MAX_GEN_LENGTH; pnt_b=0;
		}
	}

	//printf("gen_a=%d\tpnt_a=%d\tgen_b=%d\tpnt_b=%d\n",gen_a,pnt_a,gen_b,pnt_b);

	for(j=gen_a;j<=gen_b;j++){
		optr=1;
		aux_pnt = (j==gen_a)?pnt_a:(MAX_GEN_LENGTH);
		for(i=0;i<aux_pnt;i++) optr |= (optr << 1);
		temp = (john[j].to_int32 & ~optr) | (mary[j].to_int32 &  optr);
		mary[j].to_int32 = (john[j].to_int32 &  optr) | (mary[j].to_int32 & ~optr);
		john[j].to_int32 = temp;
	}

	if(pnt_b > 0){
		optr=1;
		for(i=0;i<pnt_b-1;i++) optr |= (optr << 1);
		temp = (john[gen_b].to_int32 & ~optr) | (mary[gen_b].to_int32 &  optr);
		mary[gen_b].to_int32 = (john[gen_b].to_int32 &  optr) | (mary[gen_b].to_int32 & ~optr);
		john[gen_b].to_int32 = temp;
	}

	/*
	  printf("john after:\n");
	  print_chrom(john,num_genes,0);
	  printf("mary after:\n");
	  print_chrom(mary,num_genes,0);
	*/

	return;
}

/***********************************************************************/
/* 1         2         3         4         5         6          */
/*234567890123456789012345678901234567890123456789012345678901234567890*/
/* 1         2         3         4         5         6         7*/
/***********************************************************************/
void mutate(gene *john,int num_genes,double mut_rate){
	/* creates an operator with 1's with rate= mut_rate
	   uses it to mutate john.
	*/
	int i,j;
	int optr;
	int test;

	for(j=0;j<num_genes;j++){
		optr=0;
		test=1;
		for(i=0;i<32;i++){
			if(RandomDouble() < mut_rate){
				optr |= test;
			}
			test <<= 1;
		}
		john[j].to_int32 ^= optr;
	}

	return;
}
/***********************************************************************/
/* 1         2         3         4         5         6          */
/*234567890123456789012345678901234567890123456789012345678901234567890*/
/* 1         2         3         4         5         6         7*/
/***********************************************************************/
void bin_print(int dec, int len){
	int i,val;
	int test=0;
	int op=1;
	op <<= len-1;
	//printf("op=%u\n",op);
	//printf("dec=%u len=%d\n",dec,len);
	for(i=len-1;i>=0;i--){
		test = (int)pow(2.0f,i);
		//printf("\n[%u]&[%u]=%u test=%u: ",dec,op,dec&op,test);
		val=0;
		if((dec&op) == test) val=1;
		printf("%1d",val);
		op >>= 1;
	}
	return;
}

/***********************************************************************/
/* 1         2         3         4         5         6          */
/*234567890123456789012345678901234567890123456789012345678901234567890*/
/* 1         2         3         4         5         6         7*/
/***********************************************************************/
void swap_chrom(chromosome *x, chromosome *y){
	chromosome t=*x;*x=*y;*y=t;
}

void QuickSort(chromosome* list, int beg, int end, bool energy)
{
    QS_TYPE piv;

    int  l,r,p;

    while (beg<end)    // This while loop will avoid the second recursive call
    {
        l = beg; p = beg + (end-beg)/2; r = end;

		if(energy)
			piv = list[p].evalue;
		else
			piv = list[p].fitnes;

        while (1)
        {
            while ( (l<=r) && ( ( energy && QS_ASC(list[l].evalue,piv) <= 0 ) ||
								( !energy && QS_DSC(list[l].fitnes,piv) <= 0 ) ) ) l++;
            while ( (l<=r) && ( ( energy && QS_ASC(list[r].evalue,piv) > 0 ) ||
								( !energy && QS_DSC(list[r].fitnes,piv) ) ) ) r--;

            if (l>r) break;

			swap_chrom(&list[l],&list[r]);

            if (p==r) p=l;

            l++; r--;
        }

		swap_chrom(&list[p],&list[r]);
        //list[p]=list[r]; list[r].evalue=piv;
        r--;

        // Recursion on the shorter side & loop (with new indexes) on the longer
        if ((r-beg)<(end-l))
        {
            QuickSort(list, beg, r, energy);
            beg=l;
        }
        else
        {
            QuickSort(list, l, end, energy);
            end=r;
        }
    }
}

/***********************************************************************/
/* 1         2         3         4         5         6          */
/*234567890123456789012345678901234567890123456789012345678901234567890*/
/* 1         2         3         4         5         6         7*/
/***********************************************************************/
int remove_dups(chromosome* chrom, int num_chrom, int num_genes){

	int i=0;
	int j;
	if (num_chrom<=1) return num_chrom;

	for (j=1;j<num_chrom;j++)
	{
		int flag = 0;
		for(int l=0;l<num_genes;l++){
			flag += abs(chrom[j].genes[l].to_ic - chrom[i].genes[l].to_ic) < 0.1;
		}
		if(flag != num_genes)
		{
			copy_chrom(&chrom[++i],&chrom[j],num_genes);
		}
	}

	return i+1;
}

/***********************************************************************/
/* 1         2         3         4         5         6          */
/*234567890123456789012345678901234567890123456789012345678901234567890*/
/* 1         2         3         4         5         6         7*/
/***********************************************************************/
void print_par(const chromosome* chrom,const genlim* gene_lim,int num_chrom,int num_genes, FILE* outfile_ptr){
	for(int i=0;i<num_chrom;i++){
		fprintf(outfile_ptr, "%4d (",i);
		for(int j=0;j<num_genes;j++) fprintf(outfile_ptr, "%10.2f ", chrom[i].genes[j].to_ic);
		fprintf(outfile_ptr, ") ");
		fprintf(outfile_ptr, " cf=%9.3f cf.app=%9.3f fitnes=%9.3f\n",
			chrom[i].evalue, chrom[i].app_evalue, chrom[i].fitnes);
	}

	return;
}

/***********************************************************************/
/* 1         2         3         4         5         6          */
/*234567890123456789012345678901234567890123456789012345678901234567890*/
/* 1         2         3         4         5         6         7*/
/***********************************************************************/
void write_par(const chromosome* chrom,const genlim* gene_lim,int ger,char* outfile,int num_chrom,int num_genes){
	int i,j;
	FILE *outfile_ptr;

	outfile_ptr=NULL;
	if(!OpenFile_B(outfile,"wb",&outfile_ptr)){
		Terminate(6);
	}else{

		char genes_tag[5] = { 'g' , 'e' , 'n' , 'e' , 's' };

		fwrite(&genes_tag[0], 1, sizeof(genes_tag), outfile_ptr);
		for(j=0;j<num_genes;j++){
			fwrite(&gene_lim[j], 1, sizeof(genlim), outfile_ptr);
		}

		char chrom_tag[5] = { 'c' , 'h' , 'r' , 'o' , 'm' };

		fwrite(&chrom_tag[0], 1, sizeof(chrom_tag), outfile_ptr);
		for(i=0;i<num_chrom;i++)
		{
			for(j=0;j<num_genes;j++)
			{
				fwrite(&chrom[i].genes[j].to_int32, 1, sizeof(int32_t), outfile_ptr);
			}
		}

	}

	CloseFile_B(&outfile_ptr,"w");

	return;
}

/***********************************************************************/
/* 1         2         3         4         5         6          */
/*234567890123456789012345678901234567890123456789012345678901234567890*/
/* 1         2         3         4         5         6         7*/
/***********************************************************************/
void print_pop(const chromosome* chrom,const genlim* gene_lim,int numc, int numg){
	int i,j;

	for(i=0;i<numc;i++){
		printf("%2d (",i);
		for(j=0;j<numg;j++){printf(" %10d",chrom[i].genes[j].to_int32);}
		printf(") ");
		for(j=0;j<numg;j++){printf(" "),bin_print(chrom[i].genes[j].to_int32,(MAX_GEN_LENGTH));}
		printf("\n");
	}
	return;
}
/***********************************************************************/
/* 1         2         3         4         5         6          */
/*234567890123456789012345678901234567890123456789012345678901234567890*/
/* 1         2         3         4         5         6         7*/
/***********************************************************************/
void print_chrom(const chromosome* chrom, int num_genes, int real_flag){
	int j;
	//int i;

	printf("(");
	for(j=0;j<num_genes;j++){
		if(real_flag){
			printf(" %10.5f",chrom->genes[j].to_ic);
		}else{
			printf(" %10d",chrom->genes[j].to_int32);
		}
	}
	printf(") ");
	printf("\n");

	return;
}

/***********************************************************************/
/* 1         2         3         4         5         6          */
/*234567890123456789012345678901234567890123456789012345678901234567890*/
/* 1         2         3         4         5         6         7*/
/***********************************************************************/
void print_chrom(const gene* genes, int num_genes, int real_flag){
	int j;
	//int i;

	printf("(");
	for(j=0;j<num_genes;j++){
		if(real_flag){
			printf(" %10.5f",genes[j].to_ic);
		}else{
			printf(" %10d",genes[j].to_int32);
		}
	}
	printf(") ");
	printf("\n");

	return;
}

/********************************************************************************
 * This function calculates the RSMD between atomic coordinates of the atoms in *
 * the register ori_ligatm and those for the atoms of the ligand in             *
 * residue[opt_res[0]] after reconstructing the coordinates using opt_par       *
 ********************************************************************************/

double calc_rmsp(int npar, const gene* g1, const gene* g2, const optmap* map_par, gridpoint* cleftgrid){
#ifdef FLEXAIDS_HAS_EIGEN
	// Vectorised RMSP using Eigen strided Map over the to_ic field.
	// gene_struct lays out {int32_t to_int32; double to_ic}, so stride = sizeof(gene).
	using EMap = Eigen::Map<const Eigen::VectorXd,
	                        Eigen::Unaligned,
	                        Eigen::InnerStride<sizeof(gene)/sizeof(double)>>;
	// to_ic is the second field; offset by one int32_t (4 bytes / 8 bytes per double = 0.5 — not aligned).
	// Fall back to a plain gather instead to avoid alignment issues.
	Eigen::VectorXd diff(npar);
	for (int ii = 0; ii < npar; ++ii) diff[ii] = g1[ii].to_ic - g2[ii].to_ic;
	return std::sqrt(diff.squaredNorm() / (double)npar);
#else
	double rmsp = 0.0;
	for(int i = 0; i < npar; ++i)
		rmsp += (g1[i].to_ic - g2[i].to_ic) * (g1[i].to_ic - g2[i].to_ic);
	return sqrt(rmsp / (double)npar);
#endif
}

double genetoic(const genlim* gene_lim, int32_t gene){

	int i=0;
	double tot=gene_lim->bin;

	while(tot < RandomDouble(gene))
	{
		tot += gene_lim->bin;
		i++;
	}

	double ic = gene_lim->min + gene_lim->del * (double)i;

	/* printf("ic=%.1f gene=%d randdouble=%.8f min=%.3f del=%.3f bin=%.8f\n",
		  ic, gene, RandomDouble(gene),
		  gene_lim->min, gene_lim->del, gene_lim->bin);
	*/

	return(ic);
}

int ictogene(const genlim* gene_lim, double ic){

	int i = (int)((ic - gene_lim->min) / gene_lim->del);

	double tot = 1.0;

	while(i > 0){
		tot -= gene_lim->bin;
		i--;
	}

	int gene = RandomInt(tot);

        /*
	  printf("ic=%.3f gene=%d randdouble=%.5f min=%.3f del=%.3f bin=%.3f\n",
	  ic, gene, RandomDouble(gene),
	  gene_lim->min, gene_lim->del, gene_lim->bin);
	*/

	return(gene);
}


int RandomInt(double frac){
	return (int)(frac*((double)RAND_MAX+1.0));
}

double RandomDouble(int32_t gene){
	return gene/((double)MAX_RANDOM_VALUE+1.0);
}

double RandomDouble(){
	return rand()/((double)RAND_MAX+1.0);
}