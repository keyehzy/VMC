#include <Eigen/Dense>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>

#include "vmc/hamiltonian.hpp"
#include "vmc/initialization.hpp"
#include "vmc/jastrow_parameterization.hpp"
#include "vmc/lattice.hpp"
#include "vmc/sr_optimization.hpp"
#include "vmc/version.hpp"
#include "vmc/wave_function.hpp"

namespace {

void print_vector(const char* label, const Eigen::VectorXd& values) {
  std::cout << label << " = [";
  for (Eigen::Index i = 0; i < values.size(); ++i) {
    if (i != 0) {
      std::cout << ", ";
    }
    std::cout << values[i];
  }
  std::cout << "]\n";
}

}  // namespace

int main() {
  constexpr std::uint64_t seed = 1234;
  constexpr std::size_t chain_length = 8;
  constexpr std::size_t boson_count = 4;
  constexpr std::size_t max_iterations = 6;
  constexpr std::size_t thermalization_sweeps = 50;
  constexpr std::size_t sample_count = 400;
  constexpr std::size_t sweeps_between_samples = 1;

  std::mt19937_64 rng{seed};

  const vmc::Lattice lattice = vmc::Lattice::chain(chain_length, vmc::BoundaryCondition::Periodic);
  vmc::BosonState state =
      vmc::initialize_boson_state(lattice, boson_count, vmc::OccupancyConstraint::HardCore, rng);

  const vmc::CondensateWaveFunction condensate =
      vmc::CondensateWaveFunction::uniform(lattice.site_count());
  vmc::DistanceJastrowParameterization jastrow_parameters =
      vmc::DistanceJastrowParameterization::periodic_chain(lattice.site_count());

  const vmc::BoseHubbardHamiltonian hamiltonian{
      .hopping_t = 1.0,
      .interaction_u = 0.0,
  };

  const vmc::SrOptimizationConfig optimization_config{
      .max_iterations = max_iterations,
      .energy_tolerance = 1.0e-4,
      .iteration =
          vmc::SrIterationConfig{
              .thermalization_sweeps = thermalization_sweeps,
              .sample_count = sample_count,
              .sweeps_between_samples = sweeps_between_samples,
              .update =
                  vmc::SrUpdateConfig{
                      .step_size = 0.05,
                      .diagonal_shift = 0.1,
                  },
          },
  };

  const vmc::SrOptimizationResult result = vmc::run_sr_optimization(
      lattice, condensate, jastrow_parameters, hamiltonian, state, optimization_config, rng);

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "VMC " << vmc::version() << '\n';
  std::cout << "lattice: periodic chain, L=" << chain_length << '\n';
  std::cout << "bosons: Nb=" << boson_count << ", hard-core\n";
  std::cout << "optimization: iterations=" << result.history.size()
            << ", converged=" << std::boolalpha << result.converged
            << ", energy_tolerance=" << optimization_config.energy_tolerance << '\n';
  std::cout << "sampling: samples_per_iteration=" << sample_count
            << ", thermalization_sweeps=" << thermalization_sweeps
            << ", sweeps_between_samples=" << sweeps_between_samples << '\n';
  std::cout << "sr_update: step_size=" << optimization_config.iteration.update.step_size
            << ", diagonal_shift=" << optimization_config.iteration.update.diagonal_shift << '\n';

  std::cout << "history:\n";
  for (const vmc::SrOptimizationHistoryEntry& entry : result.history) {
    std::cout << "  iteration " << entry.iteration << ": energy_mean=" << entry.energy_mean
              << ", max_abs_delta=" << entry.delta.cwiseAbs().maxCoeff() << '\n';
  }

  print_vector("final_jastrow_by_distance", jastrow_parameters.parameters());

  return 0;
}
