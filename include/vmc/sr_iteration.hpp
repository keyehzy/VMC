#pragma once

#include <Eigen/Dense>
#include <cstddef>
#include <random>

#include "vmc/boson_state.hpp"
#include "vmc/hamiltonian.hpp"
#include "vmc/jastrow_parameterization.hpp"
#include "vmc/lattice.hpp"
#include "vmc/sr_optimizer.hpp"
#include "vmc/sr_statistics.hpp"
#include "vmc/wave_function.hpp"

namespace vmc {

struct SrIterationConfig {
  std::size_t thermalization_sweeps;
  std::size_t sample_count;
  std::size_t sweeps_between_samples;
  SrUpdateConfig update;
};

struct SrIterationResult {
  SrMeasurementResult measurement;
  SrUpdateResult update;
  Eigen::VectorXd parameters_before;
  Eigen::VectorXd parameters_after;
};

SrIterationResult run_sr_iteration(const Lattice& lattice, const CondensateWaveFunction& condensate,
                                   JastrowParameterization& parameterization,
                                   const BoseHubbardHamiltonian& hamiltonian, BosonState& state,
                                   const SrIterationConfig& config, std::mt19937_64& rng);

}  // namespace vmc
