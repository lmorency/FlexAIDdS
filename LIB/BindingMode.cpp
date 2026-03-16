#include "BindingMode.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

/*****************************************\\
			BindingPopulation  
\\*****************************************/
// public (explicitely requires unsigned int temperature) *non-overloadable*
// BindingPopulation::BindingPopulation(unsigned int temp) : Temperature(temp)
BindingPopulation::BindingPopulation(FA_Global* pFA, GB_Global* pGB, VC_Global* pVC, chromosome* pchrom, genlim* pgene_lim, atom* patoms, resid* presidue, gridpoint* pcleftgrid, int num_chrom)
	: Temperature(pFA->temperature),
	  PartitionFunction(0.0),
	  nChroms(num_chrom),
	  FA(pFA),
	  GB(pGB),
	  VC(pVC),
	  chroms(pchrom),
	  gene_lim(pgene_lim),
	  atoms(patoms),
	  residue(presidue),
	  cleftgrid(pcleftgrid),
	  shannonS_population_(0.0),
	  shannon_cache_valid_(false)
{
}


void BindingPopulation::add_BindingMode(BindingMode& mode)
{
	for (std::vector<Pose>::iterator pose = mode.Poses.begin(); pose != mode.Poses.end(); ++pose)
	{
		this->PartitionFunction += pose->boltzmann_weight;
	}
	mode.set_energy();
	this->BindingModes.push_back(mode);
	this->shannon_cache_valid_ = false;  // Invalidate Shannon cache
	this->Entropize();
}


void BindingPopulation::Entropize()
{
	for (std::vector<BindingMode>::iterator it = this->BindingModes.begin(); it != this->BindingModes.end(); ++it)
	{
		it->set_energy();
	}
	std::sort(this->BindingModes.begin(), this->BindingModes.end(), BindingPopulation::EnergyComparator());
}


int BindingPopulation::get_Population_size() { return this->BindingModes.size(); }

const BindingMode& BindingPopulation::get_binding_mode(int index) const { return this->BindingModes.at(index); }
BindingMode& BindingPopulation::get_binding_mode(int index) { return this->BindingModes.at(index); }


// output BindingMode up to nResults results
void BindingPopulation::output_Population(int nResults, char* end_strfile, char* tmp_end_strfile, char* dockinp, char* gainp, int minPoints)
{
	// Output Population information ~= output clusters informations (*.cad)

	// Looping through BindingModes
	int num_result = 0;
	if (!nResults) nResults = this->get_Population_size() - 1; // if 0 is sent to this function, output all
	for (std::vector<BindingMode>::iterator mode = this->BindingModes.begin(); mode != this->BindingModes.end() && nResults > 0; ++mode, --nResults, ++num_result)
	{
		mode->output_BindingMode(num_result, end_strfile, tmp_end_strfile, dockinp, gainp, minPoints);
		// mode->output_dynamic_BindingMode(num_result,end_strfile, tmp_end_strfile, dockinp, gainp, minPoints);
	}
}


/// === NEW: Binding Population thermodynamic APIs ===
double BindingPopulation::compute_delta_G(const BindingMode& mode1, const BindingMode& mode2) const
{
	return mode1.get_free_energy() - mode2.get_free_energy();
}


statmech::StatMechEngine BindingPopulation::get_global_ensemble() const
{
	statmech::StatMechEngine global_engine(static_cast<double>(this->Temperature));

	// Phase 1: count total samples for pre-allocation
	std::size_t total_poses = 0;
	for (const auto& mode : this->BindingModes)
		total_poses += mode.Poses.size();

	// Phase 2: collect energy/weight pairs
	// Pre-collect to enable potential future parallelisation without
	// thread-safety issues on add_sample().
	std::vector<double> all_energies;
	std::vector<double> all_weights;
	all_energies.reserve(total_poses);
	all_weights.reserve(total_poses);

	for (const auto& mode : this->BindingModes)
	{
		const std::vector<double> weights = mode.get_boltzmann_weights();
		const std::vector<Pose>& poses = mode.Poses;

		if (weights.size() != poses.size())
			continue;

		for (std::size_t i = 0; i < poses.size(); ++i)
		{
			all_energies.push_back(poses[i].CF);
			all_weights.push_back(weights[i]);
		}
	}

	// Phase 3: batch add to engine (sequential, but faster due to contiguous access)
	for (std::size_t i = 0; i < all_energies.size(); ++i)
		global_engine.add_sample(all_energies[i], all_weights[i]);

	return global_engine;
}


/// === Population-level Shannon entropy ===
double BindingPopulation::get_shannon_entropy() const
{
	if (shannon_cache_valid_)
	{
		return shannonS_population_;
	}

	// Collect all pose energies across all binding modes
	std::vector<double> all_energies;
	for (const auto& mode : this->BindingModes)
	{
		for (const auto& pose : mode.Poses)
		{
			all_energies.push_back(pose.CF);
		}
	}

	if (all_energies.empty())
	{
		shannonS_population_ = 0.0;
		shannon_cache_valid_ = true;
		return 0.0;
	}

	// Compute Shannon entropy via ShannonThermoStack (energy histogram binning)
	double shannon_bits = shannon_thermo::compute_shannon_entropy(all_energies);

	// Convert from dimensionless bits to thermodynamic units: S = kB * H
	shannonS_population_ = statmech::kB_kcal * shannon_bits;
	shannon_cache_valid_ = true;
	return shannonS_population_;
}


/// === ΔG matrix between all pairs of binding modes ===
std::vector<std::vector<double>> BindingPopulation::get_deltaG_matrix() const
{
	int n = static_cast<int>(this->BindingModes.size());
	std::vector<std::vector<double>> matrix(n, std::vector<double>(n, 0.0));

	for (int i = 0; i < n; ++i)
	{
		for (int j = i + 1; j < n; ++j)
		{
			double dg = compute_delta_G(this->BindingModes[i], this->BindingModes[j]);
			matrix[i][j] = dg;
			matrix[j][i] = -dg;
		}
	}

	return matrix;
}


/*****************************************\\
			  BindingMode
\\*****************************************/

// public constructor *non-overloadable*
BindingMode::BindingMode(BindingPopulation* pop)
	: Population(pop),
	  engine_(pop ? static_cast<double>(pop->Temperature) : 298.15),
	  thermo_cache_valid_(false),
	  vib_correction_cache_(0.0),
	  vib_cache_valid_(false),
	  energy(0.0)
{
}


// public method for pose addition
void BindingMode::add_Pose(Pose& pose)
{
	this->Poses.push_back(pose);
	this->thermo_cache_valid_ = false;
	this->vib_cache_valid_ = false;
}


/// === NEW: Cache rebuild infrastructure (Phase 1) ===
void BindingMode::rebuild_engine() const
{
	if (thermo_cache_valid_)
	{
		return;
	}

	engine_.clear();
	for (const auto& pose : Poses)
	{
		engine_.add_sample(pose.CF, 1.0);
	}
	thermo_cache_valid_ = true;
}


double BindingMode::compute_enthalpy() const
{
	rebuild_engine();
	return engine_.compute().mean_energy;
}


double BindingMode::compute_entropy() const
{
	rebuild_engine();
	return engine_.compute().entropy;
}


double BindingMode::compute_energy() const
{
	rebuild_engine();
	return engine_.compute().free_energy + compute_vibrational_correction();
}


/// === Thermodynamic APIs ===
statmech::Thermodynamics BindingMode::get_thermodynamics() const
{
	rebuild_engine();
	statmech::Thermodynamics td = engine_.compute();
	// Phase 3: include vibrational free energy correction in reported free energy
	td.free_energy += compute_vibrational_correction();
	return td;
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
	return engine_.boltzmann_weights();
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
	if (coordinates.size() != Poses.size())
	{
		throw std::invalid_argument(
			"free_energy_profile: coordinate vector size (" +
			std::to_string(coordinates.size()) +
			") must match number of poses (" +
			std::to_string(Poses.size()) + ")"
		);
	}

	if (nbins < 2 || nbins > 1000)
	{
		throw std::invalid_argument(
			"free_energy_profile: nbins must be in [2, 1000]"
		);
	}

	std::vector<double> energies;
	energies.reserve(Poses.size());
	for (const auto& pose : Poses)
	{
		energies.push_back(pose.CF);
	}

	return statmech::StatMechEngine::wham(
		energies,
		coordinates,
		static_cast<double>(Population->Temperature),
		nbins
	);
}


int BindingMode::get_BindingMode_size() const { return this->Poses.size(); }

const Pose& BindingMode::get_pose(int index) const { return this->Poses.at(index); }


void BindingMode::clear_Poses()
{
	this->Poses.clear();
	this->engine_.clear();
	this->thermo_cache_valid_ = false;
	this->vib_cache_valid_ = false;
}


void BindingMode::set_energy()
{
	this->energy = this->compute_energy();
}


void BindingMode::output_BindingMode(int num_result, char* end_strfile, char* tmp_end_strfile, char* dockinp, char* gainp, int minPoints)
{
	cfstr CF;
	resid* pRes = NULL;
	cfstr* pCF = NULL;

	char sufix[25];
	char remark[MAX_REMARK];
	char tmpremark[MAX_REMARK];

	std::vector<Pose>::const_iterator Rep_lowCF = this->elect_Representative(false);
	std::vector<Pose>::const_iterator Rep_lowOPTICS = this->elect_Representative(true);

	for (int k = 0; k < this->Population->GB->num_genes; ++k) this->Population->FA->opt_par[k] = Rep_lowCF->chrom->genes[k].to_ic;

	CF = ic2cf(this->Population->FA, this->Population->VC, this->Population->atoms, this->Population->residue, this->Population->cleftgrid, this->Population->GB->num_genes, this->Population->FA->opt_par);

	strcpy(remark, "REMARK optimized structure\n");

	sprintf(tmpremark, "REMARK Fast OPTICS clustering algorithm used to output the lowest CF as Binding Mode representative\n");
	strcat(remark, tmpremark);

	sprintf(tmpremark, "REMARK CF=%8.5f\n", get_cf_evalue(&CF));
	strcat(remark, tmpremark);
	sprintf(tmpremark, "REMARK CF.app=%8.5f\n", get_apparent_cf_evalue(&CF));
	strcat(remark, tmpremark);

	for (int j = 0; j < this->Population->FA->num_optres; ++j)
	{
		pRes = &this->Population->residue[this->Population->FA->optres[j].rnum];
		pCF = &this->Population->FA->optres[j].cf;

		sprintf(tmpremark, "REMARK optimizable residue %s %c %d\n", pRes->name, pRes->chn, pRes->number);
		strcat(remark, tmpremark);

		sprintf(tmpremark, "REMARK CF.com=%8.5f\n", pCF->com);
		strcat(remark, tmpremark);
		sprintf(tmpremark, "REMARK CF.sas=%8.5f\n", pCF->sas);
		strcat(remark, tmpremark);
		sprintf(tmpremark, "REMARK CF.wal=%8.5f\n", pCF->wal);
		strcat(remark, tmpremark);
		sprintf(tmpremark, "REMARK CF.con=%8.5f\n", pCF->con);
		strcat(remark, tmpremark);
		sprintf(tmpremark, "REMARK Residue has an overall SAS of %.3f\n", pCF->totsas);
		strcat(remark, tmpremark);
	}

	sprintf(tmpremark, "REMARK Binding Mode:%d Best CF in Binding Mode:%8.5f OPTICS Center (CF):%8.5f Binding Mode Total CF:%8.5f Binding Mode Frequency:%d\n",
		num_result, Rep_lowCF->CF, Rep_lowOPTICS->CF, this->compute_energy(), this->get_BindingMode_size());
	strcat(remark, tmpremark);
	{
		double vib_corr = this->compute_vibrational_correction();
		if (std::abs(vib_corr) > 1e-12) {
			sprintf(tmpremark, "REMARK Vibrational correction (ENCoM): %10.4f kcal/mol\n", vib_corr);
			strcat(remark, tmpremark);
		}
	}
	for (int j = 0; j < this->Population->FA->npar; ++j)
	{
		sprintf(tmpremark, "REMARK [%8.3f]\n", this->Population->FA->opt_par[j]);
		strcat(remark, tmpremark);
	}

	if (this->Population->FA->refstructure == 1)
	{
		bool Hungarian = false;
		sprintf(tmpremark, "REMARK %8.5f RMSD to ref. structure (no symmetry correction)\n",
			calc_rmsd(this->Population->FA, this->Population->atoms, this->Population->residue, this->Population->cleftgrid, this->Population->FA->npar, this->Population->FA->opt_par, Hungarian));
		strcat(remark, tmpremark);
		Hungarian = true;
		sprintf(tmpremark, "REMARK %8.5f RMSD to ref. structure     (symmetry corrected)\n",
			calc_rmsd(this->Population->FA, this->Population->atoms, this->Population->residue, this->Population->cleftgrid, this->Population->FA->npar, this->Population->FA->opt_par, Hungarian));
		strcat(remark, tmpremark);
	}
	sprintf(tmpremark, "REMARK inputs: %s & %s\n", dockinp, gainp);
	strcat(remark, tmpremark);
	sprintf(sufix, "_%d_%d.pdb", minPoints, num_result);
	strcpy(tmp_end_strfile, end_strfile);
	strcat(tmp_end_strfile, sufix);
	write_pdb(this->Population->FA, this->Population->atoms, this->Population->residue, tmp_end_strfile, remark);
}


void BindingMode::output_dynamic_BindingMode(int num_result, char* end_strfile, char* tmp_end_strfile, char* dockinp, char* gainp, int minPoints)
{
	cfstr CF;
	resid* pRes = NULL;
	cfstr* pCF = NULL;

	char sufix[25];
	char remark[MAX_REMARK];
	char tmpremark[MAX_REMARK];
	int nModel = 1;
	for (std::vector<Pose>::iterator Pose = this->Poses.begin(); Pose != this->Poses.end(); ++Pose, ++nModel)
	{
		for (int k = 0; k < this->Population->GB->num_genes; ++k) this->Population->FA->opt_par[k] = Pose->chrom->genes[k].to_ic;

		CF = ic2cf(this->Population->FA, this->Population->VC, this->Population->atoms, this->Population->residue, this->Population->cleftgrid, this->Population->GB->num_genes, this->Population->FA->opt_par);

		strcpy(remark, "REMARK optimized structure\n");
		sprintf(tmpremark, "REMARK Fast OPTICS clustering algorithm used to output the lowest OPTICS ordering as Binding Mode representative\n");
		strcat(remark, tmpremark);

		sprintf(tmpremark, "REMARK CF=%8.5f\n", get_cf_evalue(&CF));
		strcat(remark, tmpremark);
		sprintf(tmpremark, "REMARK CF.app=%8.5f\n", get_apparent_cf_evalue(&CF));
		strcat(remark, tmpremark);

		for (int j = 0; j < this->Population->FA->num_optres; ++j)
		{
			pRes = &this->Population->residue[this->Population->FA->optres[j].rnum];
			pCF = &this->Population->FA->optres[j].cf;

			sprintf(tmpremark, "REMARK optimizable residue %s %c %d\n", pRes->name, pRes->chn, pRes->number);
			strcat(remark, tmpremark);

			sprintf(tmpremark, "REMARK CF.com=%8.5f\n", pCF->com);
			strcat(remark, tmpremark);
			sprintf(tmpremark, "REMARK CF.sas=%8.5f\n", pCF->sas);
			strcat(remark, tmpremark);
			sprintf(tmpremark, "REMARK CF.wal=%8.5f\n", pCF->wal);
			strcat(remark, tmpremark);
			sprintf(tmpremark, "REMARK CF.con=%8.5f\n", pCF->con);
			strcat(remark, tmpremark);
			sprintf(tmpremark, "REMARK Residue has an overall SAS of %.3f\n", pCF->totsas);
			strcat(remark, tmpremark);
		}

		for (int j = 0; j < this->Population->FA->npar; ++j)
		{
			sprintf(tmpremark, "REMARK [%8.3f]\n", this->Population->FA->opt_par[j]);
			strcat(remark, tmpremark);
		}

		if (this->Population->FA->refstructure == 1)
		{
			bool Hungarian = false;
			sprintf(tmpremark, "REMARK %8.5f RMSD to ref. structure (no symmetry correction)\n",
				calc_rmsd(this->Population->FA, this->Population->atoms, this->Population->residue, this->Population->cleftgrid, this->Population->FA->npar, this->Population->FA->opt_par, Hungarian));
			strcat(remark, tmpremark);
			Hungarian = true;
			sprintf(tmpremark, "REMARK %8.5f RMSD to ref. structure     (symmetry corrected)\n",
				calc_rmsd(this->Population->FA, this->Population->atoms, this->Population->residue, this->Population->cleftgrid, this->Population->FA->npar, this->Population->FA->opt_par, Hungarian));
			strcat(remark, tmpremark);
		}
		sprintf(tmpremark, "REMARK inputs: %s & %s\n", dockinp, gainp);
		strcat(remark, tmpremark);

		sprintf(sufix, "_%d_MODEL_%d.pdb", minPoints, num_result);
		strcpy(tmp_end_strfile, end_strfile);
		strcat(tmp_end_strfile, sufix);
		if (Pose == this->Poses.begin() && Pose + 1 == this->Poses.end())
		{
			write_MODEL_pdb(true, true, nModel, this->Population->FA, this->Population->atoms, this->Population->residue, tmp_end_strfile, remark);
		}
		else if (Pose == this->Poses.begin())
		{
			write_MODEL_pdb(true, false, nModel, this->Population->FA, this->Population->atoms, this->Population->residue, tmp_end_strfile, remark);
		}
		else if (Pose + 1 == this->Poses.end())
		{
			write_MODEL_pdb(false, true, nModel, this->Population->FA, this->Population->atoms, this->Population->residue, tmp_end_strfile, remark);
		}
		else
		{
			write_MODEL_pdb(false, false, nModel, this->Population->FA, this->Population->atoms, this->Population->residue, tmp_end_strfile, remark);
		}
	}
}


std::vector<Pose>::const_iterator BindingMode::elect_Representative(bool useOPTICSorder) const
{
	std::vector<Pose>::const_iterator Rep = this->Poses.begin();
	for (std::vector<Pose>::const_iterator it = this->Poses.begin(); it != this->Poses.end(); ++it)
	{
		if (!useOPTICSorder && (Rep->CF - it->CF) > DBL_EPSILON) Rep = it;
		if (useOPTICSorder && it->reachDist < Rep->reachDist && !isUndefinedDist(it->reachDist)) Rep = it;
	}
	return Rep;
}


inline bool const BindingMode::operator<(const BindingMode& rhs) { return (this->compute_energy() < rhs.compute_energy()); }


/*****************************************\\
				  Pose
\\*****************************************/
Pose::Pose(chromosome* chrom, int index, int iorder, float dist, uint temperature, std::vector<float> vec)
	: chrom_index(index),
	  order(iorder),
	  reachDist(dist),
	  chrom(chrom),
	  CF(chrom->app_evalue),
	  boltzmann_weight(0.0),
	  vPose(vec)
{
	this->boltzmann_weight = pow(E, ((-1.0) * (1 / static_cast<double>(temperature)) * chrom->app_evalue));
}


Pose::~Pose() {}


inline bool const Pose::operator<(const Pose& rhs)
{
	if (this->order < rhs.order) return true;
	else if (this->order > rhs.order) return false;

	if (this->reachDist < rhs.reachDist) return true;
	else if (this->reachDist > rhs.reachDist) return false;

	if (this->chrom_index < rhs.chrom_index) return true;
	else if (this->chrom_index > rhs.chrom_index) return false;

	return false;
}


/*****************************************\\
     ENCoM vibrational correction (Phase 3)
\\*****************************************/

double BindingMode::compute_vibrational_correction() const
{
	if (!this->Population->FA->normal_modes) return 0.0;

	// Return cached value if still valid
	if (this->vib_cache_valid_) return this->vib_correction_cache_;

	std::vector<encom::NormalMode> modes;
	int mode_count = this->Population->FA->normal_modes;
	const atom* atoms = this->Population->atoms;

	if (atoms && atoms[0].eigen)
	{
		for (int m = 0; m < mode_count; ++m)
		{
			if (!atoms[0].eigen[m]) continue;
			encom::NormalMode mode;
			mode.index = m + 1;
			mode.eigenvalue = static_cast<double>(atoms[0].eigen[m][0]);
			mode.frequency = std::sqrt(std::abs(mode.eigenvalue));
			modes.push_back(mode);
		}
	}

	if (modes.empty()) {
		this->vib_correction_cache_ = 0.0;
		this->vib_cache_valid_ = true;
		return 0.0;
	}

	double T = static_cast<double>(this->Population->Temperature);
	encom::VibrationalEntropy vs = encom::ENCoMEngine::compute_vibrational_entropy(modes, T);

	this->vib_correction_cache_ = -T * vs.S_vib_kcal_mol_K;
	this->vib_cache_valid_ = true;
	return this->vib_correction_cache_;
}
