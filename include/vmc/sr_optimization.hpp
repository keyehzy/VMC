#pragma once

#include <Eigen/Dense>
#include <cstddef>
#include <random>
#include <vector>

#include "vmc/boson_state.hpp"
#include "vmc/hamiltonian.hpp"
#include "vmc/jastrow_parameterization.hpp"
#include "vmc/lattice.hpp"
#include "vmc/sr_iteration.hpp"
#include "vmc/wave_function.hpp"

namespace vmc {

struct SrOptimizationConfig {
  std::size_t max_iterations;
  double energy_tolerance;
  SrIterationConfig iteration;
};

struct SrOptimizationHistoryEntry {
  std::size_t iteration;
  double energy_mean;
  Eigen::VectorXd parameters_before;
  Eigen::VectorXd parameters_after;
  Eigen::VectorXd delta;
};

struct SrOptimizationResult {
  std::vector<SrOptimizationHistoryEntry> history;
  bool converged;
};

SrOptimizationResult run_sr_optimization(const Lattice& lattice,
                                         const CondensateWaveFunction& condensate,
                                         JastrowParameterization& parameterization,
                                         const BoseHubbardHamiltonian& hamiltonian,
                                         BosonState& state, const SrOptimizationConfig& config,
                                         std::mt19937_64& rng);

}  // namespace vmc
