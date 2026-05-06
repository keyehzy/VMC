#include "vmc/wave_function.hpp"

#include <cmath>
#include <stdexcept>

namespace vmc {
namespace {

void require_state_size(const BosonState& state, std::size_t site_count) {
  if (state.site_count() != site_count) {
    throw std::invalid_argument("boson state and wave function must have the same site count");
  }
}

void require_valid_orbital(const Eigen::VectorXd& orbital) {
  if (orbital.size() == 0) {
    throw std::invalid_argument("condensate orbital must contain at least one site");
  }

  for (Eigen::Index i = 0; i < orbital.size(); ++i) {
    if (!std::isfinite(orbital[i]) || orbital[i] <= 0.0) {
      throw std::invalid_argument("condensate orbital entries must be finite and positive");
    }
  }
}

void require_valid_hop(const BosonState& state, const BosonHop& hop) {
  if (hop.source == hop.destination) {
    throw std::invalid_argument("wave-function ratios require a nontrivial hop");
  }

  if (state.boson_position(hop.boson) != hop.source) {
    throw std::invalid_argument("hop source must match the selected boson's current position");
  }

  if (state.occupation(hop.source) == 0) {
    throw std::invalid_argument("hop source must contain at least one boson");
  }

  if (state.is_hardcore() && state.occupation(hop.destination) > 0) {
    throw std::invalid_argument("hard-core hop destination is occupied");
  }
}

}  // namespace

CondensateWaveFunction::CondensateWaveFunction(Eigen::VectorXd orbital)
    : orbital_{std::move(orbital)} {
  require_valid_orbital(orbital_);
}

CondensateWaveFunction CondensateWaveFunction::uniform(std::size_t site_count) {
  if (site_count == 0) {
    throw std::invalid_argument("uniform condensate orbital must contain at least one site");
  }

  const double amplitude = 1.0 / std::sqrt(static_cast<double>(site_count));
  return CondensateWaveFunction{Eigen::VectorXd::Constant(site_count, amplitude)};
}

std::size_t CondensateWaveFunction::site_count() const {
  return static_cast<std::size_t>(orbital_.size());
}

double CondensateWaveFunction::log_amplitude(const BosonState& state) const {
  require_state_size(state, site_count());

  double value = 0.0;
  for (BosonState::Site site = 0; site < state.site_count(); ++site) {
    const std::size_t occupation = state.occupation(site);
    value += static_cast<double>(occupation) * std::log(orbital_[site]);
    value -= 0.5 * std::lgamma(static_cast<double>(occupation) + 1.0);
  }

  return value;
}

double CondensateWaveFunction::log_ratio(const BosonState& state, const BosonHop& hop) const {
  require_state_size(state, site_count());
  require_valid_hop(state, hop);

  const std::size_t source_occupation = state.occupation(hop.source);
  const std::size_t destination_occupation = state.occupation(hop.destination);

  return std::log(orbital_[hop.destination]) - std::log(orbital_[hop.source]) +
         0.5 * (std::log(static_cast<double>(source_occupation)) -
                std::log(static_cast<double>(destination_occupation + 1)));
}

double CondensateWaveFunction::ratio(const BosonState& state, const BosonHop& hop) const {
  return std::exp(log_ratio(state, hop));
}

}  // namespace vmc
