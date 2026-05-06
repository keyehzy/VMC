#include "vmc/measurement.hpp"

#include <stdexcept>

namespace vmc {

EnergyRunResult measure_local_energy(const Lattice& lattice, const WaveFunction& wave_function,
                                     const BoseHubbardHamiltonian& hamiltonian, BosonState& state,
                                     std::size_t thermalization_sweeps, std::size_t sample_count,
                                     std::size_t sweeps_between_samples, std::mt19937_64& rng) {
  return measure_local_energy(lattice, wave_function, hamiltonian, state, thermalization_sweeps,
                              sample_count, sweeps_between_samples, 1, rng);
}

EnergyRunResult measure_local_energy(const Lattice& lattice, const WaveFunction& wave_function,
                                     const BoseHubbardHamiltonian& hamiltonian, BosonState& state,
                                     std::size_t thermalization_sweeps, std::size_t sample_count,
                                     std::size_t sweeps_between_samples, std::size_t bin_size,
                                     std::mt19937_64& rng) {
  if (sample_count > 0 && sweeps_between_samples == 0) {
    throw std::invalid_argument("energy measurement requires at least one sweep between samples");
  }

  EnergyRunResult result{
      .thermalization_stats =
          run_metropolis_sweeps(lattice, wave_function, state, thermalization_sweeps, rng),
      .sampling_stats =
          MetropolisRunStats{
              .attempted_steps = 0,
              .proposed_steps = 0,
              .accepted_steps = 0,
          },
      .energy = RunningStats{},
      .binned_energy = BinningStats{bin_size},
  };

  for (std::size_t sample = 0; sample < sample_count; ++sample) {
    result.sampling_stats +=
        run_metropolis_sweeps(lattice, wave_function, state, sweeps_between_samples, rng);
    const double energy = local_energy(lattice, state, wave_function, hamiltonian);
    result.energy.add(energy);
    result.binned_energy.add(energy);
  }

  return result;
}

}  // namespace vmc
