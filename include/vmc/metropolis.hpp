#pragma once

#include <cstddef>
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

struct MetropolisRunStats {
  std::size_t attempted_steps;
  std::size_t proposed_steps;
  std::size_t accepted_steps;

  [[nodiscard]] double proposal_rate() const;
  [[nodiscard]] double acceptance_rate() const;
};

MetropolisStepResult metropolis_step(const Lattice& lattice, const WaveFunction& wave_function,
                                     BosonState& state, std::mt19937_64& rng);

MetropolisRunStats run_metropolis_steps(const Lattice& lattice, const WaveFunction& wave_function,
                                        BosonState& state, std::size_t step_count,
                                        std::mt19937_64& rng);

MetropolisRunStats run_metropolis_sweeps(const Lattice& lattice, const WaveFunction& wave_function,
                                         BosonState& state, std::size_t sweep_count,
                                         std::mt19937_64& rng);

}  // namespace vmc
