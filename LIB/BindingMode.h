#ifndef BINDINGMODE_H
#define BINDINGMODE_H

#include "gaboom.h"
#include "fileio.h"
#include "statmech.h"  // statistical mechanics engine
#include "encom.h"     // ENCoM vibrational entropy (Phase 3)
#include "ShannonThermoStack/ShannonThermoStack.h"  // Shannon configurational entropy

//#define UNDEFINED_DIST FLT_MAX // Defined in FOPTICS as > than +INF
#define UNDEFINED_DIST -0.1f // Defined in FOPTICS as > than +INF
#define isUndefinedDist(a) ((a - UNDEFINED_DIST) <= FLT_EPSILON)

class BindingPopulation; // forward-declaration in order to access BindingPopulation* Population pointer
/*****************************************\
			  Pose
\*****************************************/
struct Pose
{
	// friend class BindingPopulation;
	
	// public constructor :
	Pose(chromosome* chrom, int chrom_index, int order, float dist, uint temperature, std::vector<float>);
	~Pose();
	// public (default behavior when struct is used instead of class)
	int chrom_index;
	int order;
	float reachDist;
	chromosome* chrom;
	double CF;
	double boltzmann_weight;  // ← DEPRECATED: now computed by StatMechEngine
	std::vector<float> vPose;
	inline bool const operator< (const Pose& rhs);
};

struct PoseClassifier
{
   inline bool operator() ( const Pose& Pose1, const Pose& Pose2 )
   {
       	if(Pose1.order < Pose2.order) return true;
       	else if(Pose1.order > Pose2.order) return false;
		if(Pose1.reachDist < Pose2.reachDist) return true;
		else if(Pose1.reachDist > Pose2.reachDist) return false;
		if(Pose1.chrom_index < Pose2.chrom_index) return true;
		else if(Pose1.chrom_index > Pose2.chrom_index) return false;
		
		return false;

		
   }
};
/*****************************************\
			  BindingMode
\*****************************************/
class BindingMode // aggregation of poses (Cluster)
{
	friend class BindingPopulation;
	
	public:
		explicit 	 					BindingMode(BindingPopulation*); // public constructor (explicitely needs a pointer to a BindingPopulation of type BindingPopulation*)

			void	 					add_Pose(Pose&);
			void	 					clear_Poses();
			int	  						get_BindingMode_size() const;
			
			// ═══ LEGACY INTERFACE (backward compatibility) ═══
			double	 					compute_energy() const;      // returns Helmholtz free energy F = H - TS
			double	 					compute_entropy() const;     // returns configurational entropy S
			double	 					compute_enthalpy() const;    // returns Boltzmann-weighted ⟨E⟩
			
			// ═══ NEW STATMECH API ═══
			statmech::Thermodynamics	get_thermodynamics() const;  // full thermo struct (F, S, H, Cv, σ_E)
			double	 					get_free_energy() const;     // alias for compute_energy()
			double	 					get_heat_capacity() const;   // heat capacity C_v
			std::vector<double>	 		get_boltzmann_weights() const; // weights for all poses
			double	 					delta_G_relative_to(const BindingMode& reference) const;  // ΔG between modes
			
			// ═══ ADVANCED: WHAM FREE ENERGY PROFILES ═══
			std::vector<statmech::WHAMBin> free_energy_profile(
				const std::vector<double>& coordinates,
				int nbins = 20
			) const;  // 1D FE profile along arbitrary coordinate
			
			std::vector<Pose>::const_iterator elect_Representative(bool useOPTICSordering) const;
			inline bool const 			operator<(const BindingMode&);

 	protected:
		std::vector<Pose> Poses;
		BindingPopulation* Population; // used to access the BindingPopulation
		
		// ═══ STATMECH ENGINE (replaces manual Boltzmann summation) ═══
		mutable statmech::StatMechEngine engine_;  // mutable: allows lazy evaluation in const methods
		mutable bool thermo_cache_valid_;          // track if engine_ matches current Poses

		void	set_energy();                         // updates cached energy value
		void	rebuild_engine() const;               // populates engine_ from Poses (called on-demand)
		double	compute_vibrational_correction() const; // Phase 3: -T*S_vib from ENCoM modes

	private:
		void 	output_BindingMode(int num_result, char* end_strfile, char* tmp_end_strfile, char* dockinp, char* gainp, int minPoints);
		void	output_dynamic_BindingMode(int nBindingMode, char* end_strfile, char* tmp_end_strfile, char* dockinp, char* gainp, int minPoints);
		double energy;  // cached free energy value
};

/*****************************************\
			BindingPopulation  
\*****************************************/
class BindingPopulation
{
	friend class BindingMode;
	public:
		// Temperature is used for energy calculations of BindingModes
		unsigned int Temperature;
		
		// explicit 	BindingPopulation(unsigned int);// public constructor (explicitely needs an int representative of Temperature)
		explicit 	BindingPopulation(FA_Global* FA, GB_Global* GB, VC_Global* VC, chromosome* chrom, genlim* gene_lim, atom* atoms, resid* residue, gridpoint* cleftgrid, int nChrom);
		 	// add new binding mode to population
		 	void	add_BindingMode(BindingMode&);
		 	// return the number of BindinMonde (size getter)
		 	int	 	get_Population_size();
		 	// output BindingMode up to nResults results
		 	void	output_Population(int nResults, char* end_strfile, char* tmp_end_strfile, char* dockinp, char* gainp, int minPoints);
		 	
		 	// ═══ NEW STATMECH API ═══
		 	/// Compute ΔG between two binding modes (relative binding free energy)
		 	double	compute_delta_G(const BindingMode& mode1, const BindingMode& mode2) const;
		 	/// Get global ensemble StatMechEngine aggregating all binding modes
		 	statmech::StatMechEngine get_global_ensemble() const;

		 	// ═══ POPULATION-LEVEL SHANNON ENTROPY ═══
		 	/// Shannon configurational entropy across all binding modes: S = -kB * sum(p_i * ln(p_i))
		 	double	get_shannon_entropy() const;
		 	/// ΔG matrix between all pairs of binding modes (upper triangle, row-major)
		 	std::vector<std::vector<double>> get_deltaG_matrix() const;

	protected:
		double PartitionFunction;	// sum of all Boltzmann_weight (DEPRECATED: use StatMechEngine)
		int nChroms;	 			// n_chrom_snapshot input to clustergin function

		// FlexAID pointer
		FA_Global* 	FA;	 			// pointer to FA_Global struct
		GB_Global* 	GB;	 			// pointer to GB_Global struct
		VC_Global* 	VC;	 			// pointer to VC_Global struct
		chromosome* chroms;	 		// pointer to chromosomes' array
		genlim* gene_lim;	 		// pointer to gene_lim genlim array (useful for bondaries defined for each gene)
		atom* atoms;	 			// pointer to atoms' array
		resid* residue;	 			// pointer to residues' array
		gridpoint* cleftgrid;	// pointer to gridpoints' array (defining the total search space of the simulation)
	
	private:

		std::vector< BindingMode > 	BindingModes;	// BindingMode container
		mutable double				shannonS_population_;	// cached Shannon entropy
		mutable bool				shannon_cache_valid_;	// cache validity flag

		void 	 					Entropize(); 	// Sort BindinModes according to their observation frequency
		
		struct EnergyComparator
		{
			inline bool operator() ( const BindingMode& BindingMode1, const BindingMode& BindingMode2 )
			{
				return (BindingMode1.compute_energy() < BindingMode2.compute_energy());
			}
		};
};
#endif
