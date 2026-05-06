#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>

#include "vmc/hamiltonian.hpp"
#include "vmc/initialization.hpp"
#include "vmc/lattice.hpp"
#include "vmc/measurement.hpp"
#include "vmc/version.hpp"
#include "vmc/wave_function.hpp"

int main() {
  constexpr std::uint64_t seed = 1234;
  constexpr std::size_t chain_length = 8;
  constexpr std::size_t boson_count = 4;
  constexpr std::size_t thermalization_sweeps = 100;
  constexpr std::size_t sample_count = 1000;
  constexpr std::size_t sweeps_between_samples = 1;
  constexpr std::size_t bin_size = 20;

  std::mt19937_64 rng{seed};

  const vmc::Lattice lattice = vmc::Lattice::chain(chain_length, vmc::BoundaryCondition::Periodic);
  vmc::BosonState state =
      vmc::initialize_boson_state(lattice, boson_count, vmc::OccupancyConstraint::HardCore, rng);

  const vmc::CondensateWaveFunction condensate =
      vmc::CondensateWaveFunction::uniform(lattice.site_count());
  const vmc::JastrowWaveFunction jastrow = vmc::JastrowWaveFunction::zero(lattice.site_count());
  const vmc::ProductWaveFunction wave_function{{condensate, jastrow}};

  const vmc::BoseHubbardHamiltonian hamiltonian{
      .hopping_t = 1.0,
      .interaction_u = 0.0,
  };

  const vmc::EnergyRunResult result =
      vmc::measure_local_energy(lattice, wave_function, hamiltonian, state, thermalization_sweeps,
                                sample_count, sweeps_between_samples, bin_size, rng);

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "VMC " << vmc::version() << '\n';
  std::cout << "lattice: periodic chain, L=" << chain_length << '\n';
  std::cout << "bosons: Nb=" << boson_count << ", hard-core\n";
  std::cout << "samples: " << sample_count << ", thermalization_sweeps=" << thermalization_sweeps
            << ", sweeps_between_samples=" << sweeps_between_samples << ", bin_size=" << bin_size
            << '\n';
  std::cout << "thermalization: proposed=" << result.thermalization_stats.proposed_steps
            << " accepted=" << result.thermalization_stats.accepted_steps
            << " acceptance_rate=" << result.thermalization_stats.acceptance_rate() << '\n';
  std::cout << "sampling: proposed=" << result.sampling_stats.proposed_steps
            << " accepted=" << result.sampling_stats.accepted_steps
            << " acceptance_rate=" << result.sampling_stats.acceptance_rate() << '\n';
  std::cout << "energy:\n";
  std::cout << "  mean = " << result.energy.mean() << '\n';
  std::cout << "  standard_error = " << result.energy.standard_error() << '\n';
  std::cout << "  binned_mean = " << result.binned_energy.bin_stats().mean() << '\n';
  std::cout << "  binned_standard_error = " << result.binned_energy.bin_stats().standard_error()
            << '\n';
  std::cout << "  completed_bins = " << result.binned_energy.completed_bin_count() << '\n';

  return 0;
}
