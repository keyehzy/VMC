#include "vmc/metropolis.hpp"

namespace vmc {

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

}  // namespace vmc
