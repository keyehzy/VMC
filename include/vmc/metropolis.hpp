#pragma once

#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>

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

MetropolisRunStats& operator+=(MetropolisRunStats& lhs, const MetropolisRunStats& rhs);
[[nodiscard]] MetropolisRunStats operator+(MetropolisRunStats lhs, const MetropolisRunStats& rhs);

namespace detail {

inline double acceptance_probability_from_log_ratio(double log_ratio) {
  const double log_acceptance = 2.0 * log_ratio;
  if (log_acceptance >= 0.0) {
    return 1.0;
  }
  return std::exp(log_acceptance);
}

}  // namespace detail

template <typename WaveFunctionLike>
MetropolisStepResult metropolis_step(const Lattice& lattice, WaveFunctionLike& wave_function,
                                     BosonState& state, std::mt19937_64& rng) {
  const std::optional<BosonHop> hop = propose_nearest_neighbor_hop(lattice, state, rng);
  if (!hop.has_value()) {
    return MetropolisStepResult{
        .proposed = false,
        .accepted = false,
        .hop = std::nullopt,
        .acceptance_probability = 0.0,
    };
  }

  if (!state.can_move(hop->boson, hop->destination)) {
    return MetropolisStepResult{
        .proposed = true,
        .accepted = false,
        .hop = hop,
        .acceptance_probability = 0.0,
    };
  }

  const double acceptance_probability =
      detail::acceptance_probability_from_log_ratio(wave_function.log_ratio(state, *hop));
  std::uniform_real_distribution<double> uniform{0.0, 1.0};
  const bool accepted = uniform(rng) < acceptance_probability;

  if (accepted) {
    state.move_boson(hop->boson, hop->destination);
    if constexpr (requires { wave_function.apply_accepted_hop(*hop); }) {
      wave_function.apply_accepted_hop(*hop);
    }
  }

  return MetropolisStepResult{
      .proposed = true,
      .accepted = accepted,
      .hop = hop,
      .acceptance_probability = acceptance_probability,
  };
}

template <typename WaveFunctionLike>
MetropolisRunStats run_metropolis_steps(const Lattice& lattice, WaveFunctionLike& wave_function,
                                        BosonState& state, std::size_t step_count,
                                        std::mt19937_64& rng) {
  MetropolisRunStats stats{
      .attempted_steps = 0,
      .proposed_steps = 0,
      .accepted_steps = 0,
  };

  for (std::size_t step = 0; step < step_count; ++step) {
    const MetropolisStepResult result = metropolis_step(lattice, wave_function, state, rng);

    ++stats.attempted_steps;
    if (result.proposed) {
      ++stats.proposed_steps;
    }
    if (result.accepted) {
      ++stats.accepted_steps;
    }
  }

  return stats;
}

template <typename WaveFunctionLike>
MetropolisRunStats run_metropolis_sweeps(const Lattice& lattice, WaveFunctionLike& wave_function,
                                         BosonState& state, std::size_t sweep_count,
                                         std::mt19937_64& rng) {
  const std::size_t boson_count = state.boson_count();
  if (boson_count != 0 && sweep_count > std::numeric_limits<std::size_t>::max() / boson_count) {
    throw std::overflow_error("metropolis sweep count overflows step count");
  }

  return run_metropolis_steps(lattice, wave_function, state, sweep_count * boson_count, rng);
}

}  // namespace vmc
