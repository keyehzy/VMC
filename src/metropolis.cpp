#include "vmc/metropolis.hpp"

#include <cmath>

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

}  // namespace vmc
