// gaboom.cpp — Genetic Algorithm for molecular docking (FlexAID derivative)
// Gaudreault & Najmanovich 2015 + VoronoiCF scoring (Task 2.2)
// Apache-2.0 © 2026 Le Bonhomme Pharma

#include "flexaid.h"
#include "Scoring/VoronoiCF.h"  // NEW: VoronoiCF integration
#include <omp.h>
#include <chrono>               // NEW: Profiling instrumentation
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>

namespace {
    // GA hyperparameters
    const int DEFAULT_POPULATION_SIZE = 50;
    const int DEFAULT_NUM_GENERATIONS = 100;
    const float MUTATION_RATE = 0.1f;
    const float CROSSOVER_RATE = 0.8f;
}

// Global GA state (simplified for this demonstration)
struct GeneticAlgorithm {
    int population_size;
    int num_generations;
    std::vector<chromosome> population;
    std::vector<double> fitness;
    
    GeneticAlgorithm(int pop_size = DEFAULT_POPULATION_SIZE, 
                     int num_gen = DEFAULT_NUM_GENERATIONS)
        : population_size(pop_size), 
          num_generations(num_gen),
          population(pop_size),
          fitness(pop_size, 0.0) {}
    
    // NEW: VoronoiCF-driven fitness evaluation
    void evaluate_fitness(const atom* ligand_template, int lig_n,
                         const atom* receptor_atoms, int rec_n) {
        // Initialize scorer once (reuse across generations)
        static scoring::VoronoiCF scorer;
        
        // Batch scoring with OpenMP parallelization
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < population_size; ++i) {
            // Convert chromosome to Cartesian (existing FlexAID pipeline)
            atom* lig_atoms = ic2cf(population[i], ligand_template);
            
            // Score via VoronoiCF (real McConkey contact area + interaction matrix)
            scoring::PoseScore score = scorer.score_pose(lig_atoms, lig_n, 
                                                         receptor_atoms, rec_n);
            
            // Composite fitness: contact favorability - clash penalty + solvation bonus
            // (Negative CF is favorable; we maximize, so negate for GA selection)
            fitness[i] = score.total_cf - score.clash_penalty + 0.5f * score.solvation;
            
            // Cleanup
            free(lig_atoms);
        }
    }
    
    // NEW: GA main loop with profiling
    chromosome run(const atom* ligand_template, int lig_n,
                   const atom* receptor_atoms, int rec_n) {
        auto t_start = std::chrono::high_resolution_clock::now();
        
        for (int gen = 0; gen < num_generations; ++gen) {
            auto t_gen_start = std::chrono::high_resolution_clock::now();
            
            // Fitness evaluation (parallelized)
            evaluate_fitness(ligand_template, lig_n, receptor_atoms, rec_n);
            
            // Selection (keep best)
            selection();
            
            // Crossover + Mutation
            variation();
            
            auto t_gen_end = std::chrono::high_resolution_clock::now();
            double gen_ms = std::chrono::duration<double, std::milli>(
                t_gen_end - t_gen_start).count();
            
            if (gen % 10 == 0) {
                double best_fitness = *std::max_element(fitness.begin(), fitness.end());
                std::cerr << "[Gen " << gen << "] best_fitness: " << best_fitness 
                          << ", eval_time: " << gen_ms << " ms\n";
            }
        }
        
        auto t_end = std::chrono::high_resolution_clock::now();
        double total_s = std::chrono::duration<double>(t_end - t_start).count();
        std::cerr << "[GA TOTAL] " << total_s << " s (" << num_generations 
                  << " gens, " << population_size << " pop)\n";
        
        // Return best chromosome
        int best_idx = std::max_element(fitness.begin(), fitness.end()) 
                       - fitness.begin();
        return population[best_idx];
    }
    
private:
    void selection() {
        // Tournament selection: keep fittest 50%
        std::vector<int> indices(population_size);
        std::iota(indices.begin(), indices.end(), 0);
        
        std::sort(indices.begin(), indices.end(),
                 [this](int a, int b) { return fitness[a] > fitness[b]; });
        
        std::vector<chromosome> new_pop;
        for (int i = 0; i < population_size / 2; ++i) {
            new_pop.push_back(population[indices[i]]);
        }
        
        // Keep fittest half
        for (int i = 0; i < population_size / 2; ++i) {
            population[i] = new_pop[i];
        }
    }
    
    void variation() {
        // Simple crossover + mutation for remaining half
        for (int i = population_size / 2; i < population_size; ++i) {
            int parent1 = rand() % (population_size / 2);
            int parent2 = rand() % (population_size / 2);
            
            population[i] = crossover(population[parent1], population[parent2]);
            
            if ((rand() % 100) < (MUTATION_RATE * 100)) {
                mutate(population[i]);
            }
        }
    }
    
    chromosome crossover(const chromosome& p1, const chromosome& p2) {
        // Single-point crossover
        chromosome offspring = p1;
        int crossover_point = rand() % p1.size();
        for (int i = crossover_point; i < p1.size(); ++i) {
            offspring[i] = p2[i];
        }
        return offspring;
    }
    
    void mutate(chromosome& chromo) {
        // Gaussian mutation on real-valued genes
        int mutation_point = rand() % chromo.size();
        chromo[mutation_point] += (rand() % 100 - 50) * 0.01f;  // ±0.5 perturbation
    }
};

// Public API for docking
extern "C" {
    
chromosome* flexaid_dock(const atom* ligand, int lig_n,
                         const atom* receptor, int rec_n,
                         int pop_size, int num_gen) {
    GeneticAlgorithm ga(pop_size, num_gen);
    
    // Run docking
    chromosome best = ga.run(ligand, lig_n, receptor, rec_n);
    
    // Return best chromosome (caller frees)
    chromosome* result = (chromosome*)malloc(sizeof(chromosome));
    *result = best;
    return result;
}

}  // extern "C"
