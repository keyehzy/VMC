#include "vmc/metropolis.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace vmc {
namespace {

double acceptance_probability_from_log_ratio(double log_ratio) {
  const double log_acceptance = 2.0 * log_ratio;
  if (log_acceptance >= 0.0) {
    return 1.0;
  }
  return std::exp(log_acceptance);
}

}  // namespace

double MetropolisRunStats::proposal_rate() const {
  if (attempted_steps == 0) {
    return 0.0;
  }
  return static_cast<double>(proposed_steps) / static_cast<double>(attempted_steps);
}

double MetropolisRunStats::acceptance_rate() const {
  if (proposed_steps == 0) {
    return 0.0;
  }
  return static_cast<double>(accepted_steps) / static_cast<double>(proposed_steps);
}

MetropolisRunStats& operator+=(MetropolisRunStats& lhs, const MetropolisRunStats& rhs) {
  lhs.attempted_steps += rhs.attempted_steps;
  lhs.proposed_steps += rhs.proposed_steps;
  lhs.accepted_steps += rhs.accepted_steps;
  return lhs;
}

MetropolisRunStats operator+(MetropolisRunStats lhs, const MetropolisRunStats& rhs) {
  lhs += rhs;
  return lhs;
}

MetropolisStepResult metropolis_step(const Lattice& lattice, const WaveFunction& wave_function,
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
      acceptance_probability_from_log_ratio(wave_function.log_ratio(state, *hop));
  std::uniform_real_distribution<double> uniform{0.0, 1.0};
  const bool accepted = uniform(rng) < acceptance_probability;

  if (accepted) {
    state.move_boson(hop->boson, hop->destination);
  }

  return MetropolisStepResult{
      .proposed = true,
      .accepted = accepted,
      .hop = hop,
      .acceptance_probability = acceptance_probability,
  };
}

MetropolisRunStats run_metropolis_steps(const Lattice& lattice, const WaveFunction& wave_function,
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

MetropolisRunStats run_metropolis_sweeps(const Lattice& lattice, const WaveFunction& wave_function,
                                         BosonState& state, std::size_t sweep_count,
                                         std::mt19937_64& rng) {
  const std::size_t boson_count = state.boson_count();
  if (boson_count != 0 && sweep_count > std::numeric_limits<std::size_t>::max() / boson_count) {
    throw std::overflow_error("metropolis sweep count overflows step count");
  }

  return run_metropolis_steps(lattice, wave_function, state, sweep_count * boson_count, rng);
}

}  // namespace vmc
