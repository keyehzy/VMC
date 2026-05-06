#include "vmc/sr_optimization.hpp"

#include <cmath>
#include <stdexcept>

namespace vmc {
namespace {

void require_config(const SrOptimizationConfig& config) {
  if (config.max_iterations == 0) {
    throw std::invalid_argument("SR optimization requires at least one iteration");
  }

  if (!std::isfinite(config.energy_tolerance) || config.energy_tolerance < 0.0) {
    throw std::invalid_argument("SR optimization energy tolerance must be finite and nonnegative");
  }
}

bool has_energy_converged(double previous_energy, double current_energy, double tolerance) {
  return std::abs(current_energy - previous_energy) <= tolerance;
}

}  // namespace

SrOptimizationResult run_sr_optimization(const Lattice& lattice,
                                         const CondensateWaveFunction& condensate,
                                         JastrowParameterization& parameterization,
                                         const BoseHubbardHamiltonian& hamiltonian,
                                         BosonState& state, const SrOptimizationConfig& config,
                                         std::mt19937_64& rng) {
  require_config(config);

  SrOptimizationResult result{
      .history = {},
      .converged = false,
  };
  result.history.reserve(config.max_iterations);

  for (std::size_t iteration = 0; iteration < config.max_iterations; ++iteration) {
    const SrIterationResult iteration_result = run_sr_iteration(
        lattice, condensate, parameterization, hamiltonian, state, config.iteration, rng);
    const double energy_mean = iteration_result.measurement.statistics.energy_mean();

    result.history.push_back(SrOptimizationHistoryEntry{
        .iteration = iteration,
        .energy_mean = energy_mean,
        .parameters_before = iteration_result.parameters_before,
        .parameters_after = iteration_result.parameters_after,
        .delta = iteration_result.update.delta,
    });

    if (result.history.size() >= 2) {
      const double previous_energy = result.history[result.history.size() - 2].energy_mean;
      if (has_energy_converged(previous_energy, energy_mean, config.energy_tolerance)) {
        result.converged = true;
        break;
      }
    }
  }

  return result;
}

}  // namespace vmc
