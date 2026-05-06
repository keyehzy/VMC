#pragma once

#include <cstddef>
#include <random>

#include "vmc/boson_state.hpp"
#include "vmc/hamiltonian.hpp"
#include "vmc/lattice.hpp"
#include "vmc/metropolis.hpp"
#include "vmc/statistics.hpp"
#include "vmc/wave_function.hpp"

namespace vmc {

struct EnergyRunResult {
  MetropolisRunStats thermalization_stats;
  MetropolisRunStats sampling_stats;
  RunningStats energy;
  BinningStats binned_energy;
};

EnergyRunResult measure_local_energy(const Lattice& lattice, const WaveFunction& wave_function,
                                     const BoseHubbardHamiltonian& hamiltonian, BosonState& state,
                                     std::size_t thermalization_sweeps, std::size_t sample_count,
                                     std::size_t sweeps_between_samples, std::mt19937_64& rng);

EnergyRunResult measure_local_energy(const Lattice& lattice, const WaveFunction& wave_function,
                                     const BoseHubbardHamiltonian& hamiltonian, BosonState& state,
                                     std::size_t thermalization_sweeps, std::size_t sample_count,
                                     std::size_t sweeps_between_samples, std::size_t bin_size,
                                     std::mt19937_64& rng);

}  // namespace vmc
