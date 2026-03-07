#include "BindingMode.h"

/*****************************************\
			BindingPopulation  
\*****************************************/
// public (explicitely requires unsigned int temperature) *non-overloadable*
// BindingPopulation::BindingPopulation(unsigned int temp) : Temperature(temp)
BindingPopulation::BindingPopulation(FA_Global* pFA, GB_Global* pGB, VC_Global* pVC, chromosome* pchrom, genlim* pgene_lim, atom* patoms, resid* presidue, gridpoint* pcleftgrid, int num_chrom) : Temperature(pFA->temperature), PartitionFunction(0.0), nChroms(num_chrom), FA(pFA), GB(pGB), VC(pVC), chroms(pchrom), gene_lim(pgene_lim), atoms(patoms), residue(presidue), cleftgrid(pcleftgrid)
{
}


void BindingPopulation::add_BindingMode(BindingMode& mode)
{
	for(std::vector<Pose>::iterator pose = mode.Poses.begin(); pose != mode.Poses.end(); ++pose)
	{
		this->PartitionFunction += pose->boltzmann_weight;
	}
    mode.set_energy();
	this->BindingModes.push_back(mode);
	this->Entropize();
}


void BindingPopulation::Entropize()
{
	for(std::vector<BindingMode>::iterator it = this->BindingModes.begin(); it != this->BindingModes.end(); ++it)
	{
		it->set_energy();
	}
	std::sort(this->BindingModes.begin(), this->BindingModes.end(), BindingPopulation::EnergyComparator());
}


int BindingPopulation::get_Population_size() { return this->BindingModes.size(); }


// output BindingMode up to nResults results
void BindingPopulation::output_Population(int nResults, char* end_strfile, char* tmp_end_strfile, char* dockinp, char* gainp, int minPoints)
{
    // Output Population information ~= output clusters informations (*.cad)
    
    
    // Looping through BindingModes
    int num_result = 0;
    if(!nResults) nResults = this->get_Population_size() - 1; // if 0 is sent to this function, output all
	for(std::vector<BindingMode>::iterator mode = this->BindingModes.begin(); mode != this->BindingModes.end() && nResults > 0; ++mode, --nResults, ++num_result)
	{
		mode->output_BindingMode(num_result, end_strfile, tmp_end_strfile, dockinp, gainp, minPoints);
        // mode->output_dynamic_BindingMode(num_result,end_strfile, tmp_end_strfile, dockinp, gainp, minPoints);
	}
}

/// === NEW: Binding Population thermodynamic APIs ===
double BindingPopulation::compute_delta_G(const BindingMode& mode1, const BindingMode& mode2) const
{
	// Get free energies from each mode
	double F1 = mode1.get_free_energy();
	double F2 = mode2.get_free_energy();
	return F1 - F2;  // relative free energy
}

statmech::StatMechEngine BindingPopulation::get_global_ensemble() const
{
	statmech::StatMechEngine global_engine(this->Temperature);
	
	// Aggregate all modes into global ensemble
	for (const auto& mode : this->BindingModes) {
		const auto& weights = mode.get_boltzmann_weights();
		const auto& poses = mode.Poses;
		
		if (weights.size() != poses.size()) {
			continue;  // Safety check
		}
		
		for (size_t i = 0; i < poses.size(); ++i) {
			global_engine.add_sample(poses[i].CF, weights[i]);
		}
	}
	
	return global_engine;
}


/*****************************************\
			  BindingMode
\*****************************************/

// public constructor *non-overloadable*
BindingMode::BindingMode(BindingPopulation* pop) 
	: Population(pop), 
	  energy(0.0),
	  engine_(pop ? pop->Temperature : 298.15),
	  thermo_cache_valid_(false)
{
}


// public method for pose addition
void BindingMode::add_Pose(Pose& pose)
{
	this->Poses.push_back(pose);
	this->thermo_cache_valid_ = false;  // Invalidate cache on modification
}


/// === NEW: Cache rebuild infrastructure (Phase 1) ===
void BindingMode::rebuild_engine() const
{
	// Only rebuild if cache is dirty
	if (thermo_cache_valid_) {
		return;
	}
	
	// Clear previous state
	const_cast<statmech::StatMechEngine&>(engine_).clear();
	
	// Populate engine from all poses in this mode
	for (const auto& pose : Poses) {
		// Add each pose's CF energy with unit multiplicity
		// (CF already computed via ic2cf in GA pipeline)
		const_cast<statmech::StatMechEngine&>(engine_).add_sample(pose.CF, 1.0);
	}
	
	// Mark cache as valid until next pose modification
	const_cast<bool&>(thermo_cache_valid_) = true;
}


double BindingMode::compute_enthalpy() const
{
	// Delegate to StatMechEngine for numerically stable computation
	rebuild_engine();
	return engine_.compute().mean_energy;  // <E> from canonical ensemble
}


double BindingMode::compute_entropy() const
{ 
	// Delegate to StatMechEngine for proper Shannon/configurational entropy
	rebuild_engine();
	return engine_.compute().entropy;  // S = (E - F) / T
}


double BindingMode::compute_energy() const
{ 
	// Delegate to StatMechEngine for free energy via log-sum-exp
	rebuild_engine();
	return engine_.compute().free_energy;  // F = -kT ln(Z)
}


/// === NEW: Thermodynamic APIs (Phase 1) ===
statmech::Thermodynamics BindingMode::get_thermodynamics() const
{
	rebuild_engine();
	return engine_.compute();
	// Returns struct with: {free_energy, mean_energy, entropy, heat_capacity, energy_std}
}


double BindingMode::get_free_energy() const
{
	return get_thermodynamics().free_energy;
}


double BindingMode::get_heat_capacity() const
{
	return get_thermodynamics().heat_capacity;
}


std::vector<double> BindingMode::get_boltzmann_weights() const
{
	rebuild_engine();
	return engine_.boltzmann_weights();  // Returns normalized weights (sum = 1.0)
}


double BindingMode::delta_G_relative_to(const BindingMode& reference) const
{
	return this->get_free_energy() - reference.get_free_energy();
}


std::vector<statmech::WHAMBin> BindingMode::free_energy_profile(
	const std::vector<double>& coordinates,
	int nbins
) const
{
	// Validate input
	if (coordinates.size() != Poses.size()) {
		throw std::invalid_argument(
			"free_energy_profile: coordinate vector size (" +
			std::to_string(coordinates.size()) +
			") must match number of poses (" +
			std::to_string(Poses.size()) + ")"
		);
	}
	
	if (nbins < 2 || nbins > 1000) {
		throw std::invalid_argument(
			"free_energy_profile: nbins must be in [2, 1000]"
		);
	}
	
	// Extract CF energies from poses
	std::vector<double> energies;
	energies.reserve(Poses.size());
	for (const auto& pose : Poses) {
		energies.push_back(pose.CF);
	}
	
	// Call StatMechEngine::wham() for WHAM analysis
	// Returns free energy profile along input coordinate
	return statmech::StatMechEngine::wham(
		energies,
		coordinates,
		Population->Temperature,
		nbins
	);
}


int BindingMode::get_BindingMode_size() const { return this->Poses.size(); }


void BindingMode::clear_Poses() 
{ 
	this->Poses.clear();
	const_cast<statmech::StatMechEngine&>(engine_).clear();
	this->thermo_cache_valid_ = false;
}


void BindingMode::set_energy()
{
	this->energy = this->compute_energy();
}


void BindingMode::output_BindingMode(int num_result, char* end_strfile, char* tmp_end_strfile, char* dockinp, char* gainp, int minPoints)
{
    // File and Output variables declarations
    cfstr CF; /* complementarity function value */
    resid *pRes = NULL;
    cfstr* pCF = NULL;

    char sufix[25];
    char remark[MAX_REMARK];
    char tmpremark[MAX_REMARK];
	
    // 0. elect a Pose representative (Rep) of the current BindingMode
	std::vector<Pose>::const_iterator Rep_lowCF = this->elect_Representative(false);
	std::vector<Pose>::const_iterator Rep_lowOPTICS = this->elect_Representative(true);
	
    // 1. build FA->opt_par[GB->num_genes]
	 for(int k = 0; k < this->Population->GB->num_genes; ++k) this->Population->FA->opt_par[k] = Rep_lowCF->chrom->genes[k].to_ic;
	 // for(int k = 0; k < this->Population->GB->num_genes; ++k) this->Population->FA->opt_par[k] = Rep_lowOPTICS->chrom->genes[k].to_ic;

	// 2. get CF with ic2cf() 
	CF = ic2cf(this->Population->FA, this->Population->VC, this->Population->atoms, this->Population->residue, this->Population->cleftgrid, this->Population->GB->num_genes, this->Population->FA->opt_par);
	
    // 3. print REMARKS for FA->optres (res_ptr && cf_ptr for each optimizable residue)
	strcpy(remark,"REMARK optimized structure\n");
	
	 sprintf(tmpremark,"REMARK Fast OPTICS clustering algorithm used to output the lowest CF as Binding Mode representative\n");
	 // sprintf(tmpremark,"REMARK Fast OPTICS clustering algorithm used to output the lowest OPTICS ordering as Binding Mode representative\n");
	strcat(remark,tmpremark);
	
	sprintf(tmpremark,"REMARK CF=%8.5f\n",get_cf_evalue(&CF));
	strcat(remark,tmpremark);
	sprintf(tmpremark,"REMARK CF.app=%8.5f\n",get_apparent_cf_evalue(&CF));
	strcat(remark,tmpremark);
    
	for(int j = 0; j < this->Population->FA->num_optres; ++j)
	{
		pRes = &this->Population->residue[this->Population->FA->optres[j].rnum];
		pCF  = &this->Population->FA->optres[j].cf;
        
        sprintf(tmpremark,"REMARK optimizable residue %s %c %d\n", pRes->name, pRes->chn, pRes->number);
        strcat(remark,tmpremark);
        
        sprintf(tmpremark ,"REMARK CF.com=%8.5f\n", pCF->com);
        strcat(remark, tmpremark);
        sprintf(tmpremark ,"REMARK CF.sas=%8.5f\n", pCF->sas);
        strcat(remark, tmpremark);
        sprintf(tmpremark ,"REMARK CF.wal=%8.5f\n", pCF->wal);
        strcat(remark, tmpremark);
        sprintf(tmpremark ,"REMARK CF.con=%8.5f\n", pCF->con);
        strcat(remark, tmpremark);
        sprintf(tmpremark, "REMARK Residue has an overall SAS of %.3f\n", pCF->totsas);
        strcat(remark, tmpremark);
	}
    
    sprintf(tmpremark,"REMARK Binding Mode:%d Best CF in Binding Mode:%8.5f OPTICS Center (CF):%8.5f Binding Mode Total CF:%8.5f Binding Mode Frequency:%d\n",
            num_result, Rep_lowCF->CF, Rep_lowOPTICS->CF, this->compute_energy(), this->get_BindingMode_size());
    strcat(remark,tmpremark);
    for(int j=0; j < this->Population->FA->npar; ++j)
	{
		sprintf(tmpremark, "REMARK [%8.3f]\n",this->Population->FA->opt_par[j]);
		strcat(remark,tmpremark);
	}

	// 4. if(REF) prints RMSD to REF
	if(this->Population->FA->refstructure == 1)
	{
		bool Hungarian = false;
		sprintf(tmpremark,"REMARK %8.5f RMSD to ref. structure (no symmetry correction)\n",
		calc_rmsd(this->Population->FA,this->Population->atoms,this->Population->residue,this->Population->cleftgrid,this->Population->FA->npar,this->Population->FA->opt_par, Hungarian));
		strcat(remark,tmpremark);
		Hungarian = true;
		sprintf(tmpremark,"REMARK %8.5f RMSD to ref. structure     (symmetry corrected)\n",
		calc_rmsd(this->Population->FA,this->Population->atoms,this->Population->residue,this->Population->cleftgrid,this->Population->FA->npar,this->Population->FA->opt_par, Hungarian));
		strcat(remark,tmpremark);
	}
	sprintf(tmpremark,"REMARK inputs: %s & %s\n",dockinp,gainp);
	strcat(remark,tmpremark);
	sprintf(sufix,"_%d_%d.pdb", minPoints, num_result);
	strcpy(tmp_end_strfile,end_strfile);
	strcat(tmp_end_strfile,sufix);
	// 5. write_pdb(FA,atoms,residue,tmp_end_strfile,remark)
	write_pdb(this->Population->FA,this->Population->atoms,this->Population->residue,tmp_end_strfile,remark);
}


void BindingMode::output_dynamic_BindingMode(int num_result, char* end_strfile, char* tmp_end_strfile, char* dockinp, char* gainp, int minPoints)
{
    // File and Output variables declarations
    cfstr CF; /* complementarity function value */
    resid *pRes = NULL;
    cfstr* pCF = NULL;

    char sufix[25];
    char remark[MAX_REMARK];
    char tmpremark[MAX_REMARK];
    int nModel = 1;
    for(std::vector<Pose>::iterator Pose = this->Poses.begin(); Pose != this->Poses.end(); ++Pose, ++nModel)
    {
    	// 1. build FA->opt_par[GB->num_genes]
		for(int k = 0; k < this->Population->GB->num_genes; ++k) this->Population->FA->opt_par[k] = Pose->chrom->genes[k].to_ic;

		// 2. get CF with ic2cf() 
		CF = ic2cf(this->Population->FA, this->Population->VC, this->Population->atoms, this->Population->residue, this->Population->cleftgrid, this->Population->GB->num_genes, this->Population->FA->opt_par);
		
	    // 3. print REMARKS for FA->optres (res_ptr && cf_ptr for each optimizable residue)
		strcpy(remark,"REMARK optimized structure\n");
		
	//	 sprintf(tmpremark,"REMARK Fast OPTICS clustering algorithm used to output the lowest CF as Binding Mode representative\n");
		 sprintf(tmpremark,"REMARK Fast OPTICS clustering algorithm used to output the lowest OPTICS ordering as Binding Mode representative\n");
		strcat(remark,tmpremark);
		
		sprintf(tmpremark,"REMARK CF=%8.5f\n",get_cf_evalue(&CF));
		strcat(remark,tmpremark);
		sprintf(tmpremark,"REMARK CF.app=%8.5f\n",get_apparent_cf_evalue(&CF));
		strcat(remark,tmpremark);
	    
		for(int j = 0; j < this->Population->FA->num_optres; ++j)
		{
			pRes = &this->Population->residue[this->Population->FA->optres[j].rnum];
			pCF  = &this->Population->FA->optres[j].cf;
	        
	        sprintf(tmpremark,"REMARK optimizable residue %s %c %d\n", pRes->name, pRes->chn, pRes->number);
	        strcat(remark,tmpremark);
	        
	        sprintf(tmpremark ,"REMARK CF.com=%8.5f\n", pCF->com);
	        strcat(remark, tmpremark);
	        sprintf(tmpremark ,"REMARK CF.sas=%8.5f\n", pCF->sas);
	        strcat(remark, tmpremark);
	        sprintf(tmpremark ,"REMARK CF.wal=%8.5f\n", pCF->wal);
	        strcat(remark, tmpremark);
	        sprintf(tmpremark ,"REMARK CF.con=%8.5f\n", pCF->con);
	        strcat(remark, tmpremark);
	        sprintf(tmpremark, "REMARK Residue has an overall SAS of %.3f\n", pCF->totsas);
	        strcat(remark, tmpremark);
		}
	    
	    for(int j=0; j < this->Population->FA->npar; ++j)
		{
			sprintf(tmpremark, "REMARK [%8.3f]\n",this->Population->FA->opt_par[j]);
			strcat(remark,tmpremark);
		}

		// 4. if(REF) prints RMSD to REF
		if(this->Population->FA->refstructure == 1)
		{
			bool Hungarian = false;
			sprintf(tmpremark,"REMARK %8.5f RMSD to ref. structure (no symmetry correction)\n",
			calc_rmsd(this->Population->FA,this->Population->atoms,this->Population->residue,this->Population->cleftgrid,this->Population->FA->npar,this->Population->FA->opt_par, Hungarian));
			strcat(remark,tmpremark);
			Hungarian = true;
			sprintf(tmpremark,"REMARK %8.5f RMSD to ref. structure     (symmetry corrected)\n",
			calc_rmsd(this->Population->FA,this->Population->atoms,this->Population->residue,this->Population->cleftgrid,this->Population->FA->npar,this->Population->FA->opt_par, Hungarian));
			strcat(remark,tmpremark);
		}
		sprintf(tmpremark,"REMARK inputs: %s & %s\n",dockinp,gainp);
		strcat(remark,tmpremark);
        
		sprintf(sufix,"_%d_MODEL_%d.pdb", minPoints, num_result);
		strcpy(tmp_end_strfile,end_strfile);
		strcat(tmp_end_strfile,sufix);
		// 5. write_pdb(FA,atoms,residue,tmp_end_strfile,remark)
		if(Pose == this->Poses.begin() && Pose+1 == this->Poses.end())
		{
			write_MODEL_pdb(true, true, nModel, this->Population->FA,this->Population->atoms,this->Population->residue,tmp_end_strfile,remark);
		}
		else if(Pose == this->Poses.begin())
		{
			write_MODEL_pdb(true, false, nModel, this->Population->FA,this->Population->atoms,this->Population->residue,tmp_end_strfile,remark);
		}
		else if(Pose+1 == this->Poses.end())
		{
			write_MODEL_pdb(false, true, nModel, this->Population->FA,this->Population->atoms,this->Population->residue,tmp_end_strfile,remark);
		}
		else
		{
			write_MODEL_pdb(false, false, nModel, this->Population->FA,this->Population->atoms,this->Population->residue,tmp_end_strfile,remark);
		}
    }
}

std::vector<Pose>::const_iterator BindingMode::elect_Representative(bool useOPTICSorder) const
{
	std::vector<Pose>::const_iterator Rep = this->Poses.begin();
	for(std::vector<Pose>::const_iterator it = this->Poses.begin(); it != this->Poses.end(); ++it)
	{
		if(!useOPTICSorder && (Rep->CF - it->CF) > DBL_EPSILON ) Rep = it;
		if(useOPTICSorder && it->reachDist < Rep->reachDist && !isUndefinedDist(it->reachDist)) Rep = it;
	}
	return Rep;
}


inline bool const BindingMode::operator< (const BindingMode& rhs) { return (this->compute_energy() < rhs.compute_energy()); }


/*****************************************\
				  Pose
\*****************************************/
// public constructor for Pose *non-overloadable*
Pose::Pose(chromosome* chrom, int index, int iorder, float dist, uint temperature, std::vector<float> vec) : chrom(chrom), order(iorder), chrom_index(index), reachDist(dist), CF(chrom->app_evalue), vPose(vec)
{
	this->boltzmann_weight = pow( E, ((-1.0) * (1/static_cast<double>(temperature)) * chrom->app_evalue) );
}

Pose::~Pose(){}

inline bool const Pose::operator< (const Pose& rhs)
{
	if(this->order < rhs.order) return true;
   	else if(this->order > rhs.order) return false;
	
	if(this->reachDist < rhs.reachDist) return true;
	else if(this->reachDist > rhs.reachDist) return false;
	
	if(this->chrom_index < rhs.chrom_index) return true;
	else if(this->chrom_index > rhs.chrom_index) return false;
	
	return false;
}
