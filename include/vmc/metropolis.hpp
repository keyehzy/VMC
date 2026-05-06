#pragma once

#include <optional>
#include <random>

#include "vmc/boson_state.hpp"
#include "vmc/lattice.hpp"
#include "vmc/proposal.hpp"
#include "vmc/wave_function.hpp"

namespace vmc {

struct MetropolisStepResult {
  bool proposed;
  bool accepted;
  std::optional<BosonHop> hop;
  double acceptance_probability;
};

MetropolisStepResult metropolis_step(const Lattice& lattice, const WaveFunction& wave_function,
                                     BosonState& state, std::mt19937_64& rng);

}  // namespace vmc
