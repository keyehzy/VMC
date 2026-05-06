#include "vmc/sr_iteration.hpp"

#include <utility>

namespace vmc {

SrIterationResult run_sr_iteration(const Lattice& lattice, const CondensateWaveFunction& condensate,
                                   JastrowParameterization& parameterization,
                                   const BoseHubbardHamiltonian& hamiltonian, BosonState& state,
                                   const SrIterationConfig& config, std::mt19937_64& rng) {
  const Eigen::VectorXd parameters_before = parameterization.parameters();
  const JastrowWaveFunction jastrow{parameterization.matrix()};
  const ProductWaveFunction wave_function{{condensate, jastrow}};

  SrMeasurementResult measurement = measure_sr_statistics(
      lattice, wave_function, parameterization, hamiltonian, state, config.thermalization_sweeps,
      config.sample_count, config.sweeps_between_samples, rng);
  SrUpdateResult update = compute_sr_update(measurement.statistics, config.update);

  parameterization.update_parameters(update.delta);
  const Eigen::VectorXd parameters_after = parameterization.parameters();

  return SrIterationResult{
      .measurement = std::move(measurement),
      .update = std::move(update),
      .parameters_before = parameters_before,
      .parameters_after = parameters_after,
  };
}

}  // namespace vmc
