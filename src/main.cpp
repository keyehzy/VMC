#include <Eigen/Dense>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <random>

#include "vmc/hamiltonian.hpp"
#include "vmc/initialization.hpp"
#include "vmc/jastrow_parameterization.hpp"
#include "vmc/lattice.hpp"
#include "vmc/run_config.hpp"
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

const char* occupancy_label(vmc::OccupancyConstraint constraint) {
  switch (constraint) {
    case vmc::OccupancyConstraint::HardCore:
      return "hard-core";
    case vmc::OccupancyConstraint::SoftCore:
      return "soft-core";
  }
  return "unknown";
}

}  // namespace

int main(int argc, char* argv[]) {
  vmc::ParsedRunConfig parsed_config;
  try {
    parsed_config = vmc::parse_run_config(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n\n" << vmc::usage(argv[0]);
    return 2;
  }

  if (parsed_config.help_requested) {
    std::cout << vmc::usage(argv[0]);
    return 0;
  }

  const vmc::RunConfig& config = parsed_config.config;

  std::mt19937_64 rng{config.seed};

  const vmc::Lattice lattice =
      vmc::Lattice::chain(config.chain_length, vmc::BoundaryCondition::Periodic);
  vmc::BosonState state =
      vmc::initialize_boson_state(lattice, config.boson_count, config.occupancy_constraint, rng);

  const vmc::CondensateWaveFunction condensate =
      vmc::CondensateWaveFunction::uniform(lattice.site_count());
  vmc::DistanceJastrowParameterization jastrow_parameters =
      vmc::DistanceJastrowParameterization::periodic_chain(lattice.site_count());

  const vmc::SrOptimizationResult result = vmc::run_sr_optimization(
      lattice, condensate, jastrow_parameters, config.hamiltonian, state, config.optimization, rng);

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "VMC " << vmc::version() << '\n';
  std::cout << "seed: " << config.seed << '\n';
  std::cout << "lattice: periodic chain, L=" << config.chain_length << '\n';
  std::cout << "bosons: Nb=" << config.boson_count << ", "
            << occupancy_label(config.occupancy_constraint) << '\n';
  std::cout << "hamiltonian: t=" << config.hamiltonian.hopping_t
            << ", U=" << config.hamiltonian.interaction_u << '\n';
  std::cout << "optimization: iterations=" << result.history.size()
            << ", converged=" << std::boolalpha << result.converged
            << ", max_iterations=" << config.optimization.max_iterations
            << ", energy_tolerance=" << config.optimization.energy_tolerance << '\n';
  std::cout << "sampling: samples_per_iteration=" << config.optimization.iteration.sample_count
            << ", thermalization_sweeps=" << config.optimization.iteration.thermalization_sweeps
            << ", sweeps_between_samples=" << config.optimization.iteration.sweeps_between_samples
            << '\n';
  std::cout << "sr_update: step_size=" << config.optimization.iteration.update.step_size
            << ", diagonal_shift=" << config.optimization.iteration.update.diagonal_shift << '\n';

  std::cout << "history:\n";
  for (const vmc::SrOptimizationHistoryEntry& entry : result.history) {
    std::cout << "  iteration " << entry.iteration << ": energy_mean=" << entry.energy_mean
              << ", max_abs_delta=" << entry.delta.cwiseAbs().maxCoeff() << '\n';
  }

  print_vector("final_jastrow_by_distance", jastrow_parameters.parameters());

  return 0;
}
