#pragma once

#include "vmc/boson_state.hpp"
#include "vmc/lattice.hpp"
#include "vmc/wave_function.hpp"

namespace vmc {

struct BoseHubbardHamiltonian {
  double hopping_t;
  double interaction_u;
};

[[nodiscard]] double local_energy(const Lattice& lattice, const BosonState& state,
                                  const WaveFunction& wave_function,
                                  const BoseHubbardHamiltonian& hamiltonian);

}  // namespace vmc
