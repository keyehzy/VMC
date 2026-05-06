#include "vmc/wave_function.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

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

void require_valid_jastrow_parameters(const Eigen::MatrixXd& parameters) {
  if (parameters.rows() == 0 || parameters.cols() == 0) {
    throw std::invalid_argument("Jastrow parameters must contain at least one site");
  }

  if (parameters.rows() != parameters.cols()) {
    throw std::invalid_argument("Jastrow parameters must be square");
  }

  for (Eigen::Index row = 0; row < parameters.rows(); ++row) {
    for (Eigen::Index col = 0; col < parameters.cols(); ++col) {
      if (!std::isfinite(parameters(row, col))) {
        throw std::invalid_argument("Jastrow parameters must be finite");
      }
    }
  }

  if (!parameters.isApprox(parameters.transpose())) {
    throw std::invalid_argument("Jastrow parameters must be symmetric");
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

std::vector<std::size_t> occupations_from_state(const BosonState& state) {
  const auto occupations = state.occupations();
  return {occupations.begin(), occupations.end()};
}

double jastrow_log_amplitude(const Eigen::MatrixXd& parameters,
                             std::span<const std::size_t> occupations) {
  double quadratic_form = 0.0;
  for (std::size_t row = 0; row < occupations.size(); ++row) {
    for (std::size_t col = 0; col < occupations.size(); ++col) {
      quadratic_form += static_cast<double>(occupations[row]) * parameters(row, col) *
                        static_cast<double>(occupations[col]);
    }
  }

  return -0.5 * quadratic_form;
}

std::vector<double> jastrow_cache_values(const JastrowWaveFunction& wave_function,
                                         const BosonState& state) {
  require_state_size(state, wave_function.site_count());

  const Eigen::MatrixXd& parameters = wave_function.parameters();
  std::vector<double> values(state.site_count(), 0.0);

  for (BosonState::Site col = 0; col < state.site_count(); ++col) {
    for (BosonState::Site row = 0; row < state.site_count(); ++row) {
      values[col] += parameters(row, col) * static_cast<double>(state.occupation(row));
    }
  }

  return values;
}

void require_site_in_cache_range(BosonState::Site site, std::size_t site_count) {
  if (site >= site_count) {
    throw std::out_of_range("Jastrow cache hop site is outside the wave function");
  }
}

}  // namespace

double WaveFunction::ratio(const BosonState& state, const BosonHop& hop) const {
  return std::exp(log_ratio(state, hop));
}

ProductWaveFunction::ProductWaveFunction(
    std::vector<std::reference_wrapper<const WaveFunction>> components)
    : components_{std::move(components)}, site_count_{0} {
  if (components_.empty()) {
    throw std::invalid_argument("product wave function requires at least one component");
  }

  site_count_ = components_.front().get().site_count();
  for (const WaveFunction& component : components_) {
    if (component.site_count() != site_count_) {
      throw std::invalid_argument("product wave function components must have the same site count");
    }
  }
}

std::size_t ProductWaveFunction::site_count() const {
  return site_count_;
}

double ProductWaveFunction::log_amplitude(const BosonState& state) const {
  double value = 0.0;
  for (const WaveFunction& component : components_) {
    value += component.log_amplitude(state);
  }
  return value;
}

double ProductWaveFunction::log_ratio(const BosonState& state, const BosonHop& hop) const {
  double value = 0.0;
  for (const WaveFunction& component : components_) {
    value += component.log_ratio(state, hop);
  }
  return value;
}

std::size_t ProductWaveFunction::component_count() const {
  return components_.size();
}

CachedProductWaveFunction::CachedProductWaveFunction(const CondensateWaveFunction& condensate,
                                                     const JastrowWaveFunction& jastrow,
                                                     const BosonState& state)
    : condensate_{condensate}, jastrow_{jastrow}, jastrow_cache_{jastrow, state} {
  if (condensate.site_count() != jastrow.site_count()) {
    throw std::invalid_argument("cached product components must have the same site count");
  }
  require_state_size(state, condensate.site_count());
}

std::size_t CachedProductWaveFunction::site_count() const {
  return condensate_.get().site_count();
}

double CachedProductWaveFunction::log_ratio(const BosonState& state, const BosonHop& hop) const {
  return condensate_.get().log_ratio(state, hop) + jastrow_cache_.log_ratio(state, hop);
}

double CachedProductWaveFunction::ratio(const BosonState& state, const BosonHop& hop) const {
  return std::exp(log_ratio(state, hop));
}

void CachedProductWaveFunction::apply_accepted_hop(const BosonHop& hop) {
  jastrow_cache_.apply_accepted_hop(hop);
}

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

JastrowWaveFunction::JastrowWaveFunction(Eigen::MatrixXd parameters)
    : parameters_{std::move(parameters)} {
  require_valid_jastrow_parameters(parameters_);
}

JastrowWaveFunction JastrowWaveFunction::zero(std::size_t site_count) {
  if (site_count == 0) {
    throw std::invalid_argument("zero Jastrow parameters must contain at least one site");
  }

  return JastrowWaveFunction{Eigen::MatrixXd::Zero(site_count, site_count)};
}

std::size_t JastrowWaveFunction::site_count() const {
  return static_cast<std::size_t>(parameters_.rows());
}

double JastrowWaveFunction::log_amplitude(const BosonState& state) const {
  require_state_size(state, site_count());
  return jastrow_log_amplitude(parameters_, state.occupations());
}

double JastrowWaveFunction::log_ratio(const BosonState& state, const BosonHop& hop) const {
  require_state_size(state, site_count());
  require_valid_hop(state, hop);

  auto after_occupations = occupations_from_state(state);
  const double before = jastrow_log_amplitude(parameters_, after_occupations);

  --after_occupations[hop.source];
  ++after_occupations[hop.destination];
  const double after = jastrow_log_amplitude(parameters_, after_occupations);

  return after - before;
}

const Eigen::MatrixXd& JastrowWaveFunction::parameters() const {
  return parameters_;
}

JastrowCache::JastrowCache(const JastrowWaveFunction& wave_function, const BosonState& state)
    : wave_function_{wave_function}, values_{jastrow_cache_values(wave_function, state)} {}

std::size_t JastrowCache::site_count() const {
  return values_.size();
}

std::span<const double> JastrowCache::values() const {
  return values_;
}

double JastrowCache::log_ratio(const BosonState& state, const BosonHop& hop) const {
  const JastrowWaveFunction& wave_function = wave_function_.get();
  require_state_size(state, site_count());
  require_state_size(state, wave_function.site_count());
  require_valid_hop(state, hop);

  const Eigen::MatrixXd& parameters = wave_function.parameters();
  const double quadratic_delta = parameters(hop.destination, hop.destination) +
                                 parameters(hop.source, hop.source) -
                                 2.0 * parameters(hop.destination, hop.source);

  return values_[hop.source] - values_[hop.destination] - 0.5 * quadratic_delta;
}

double JastrowCache::ratio(const BosonState& state, const BosonHop& hop) const {
  return std::exp(log_ratio(state, hop));
}

void JastrowCache::apply_accepted_hop(const BosonHop& hop) {
  require_site_in_cache_range(hop.source, site_count());
  require_site_in_cache_range(hop.destination, site_count());
  if (hop.source == hop.destination) {
    throw std::invalid_argument("Jastrow cache updates require a nontrivial hop");
  }

  const Eigen::MatrixXd& parameters = wave_function_.get().parameters();
  for (BosonState::Site site = 0; site < site_count(); ++site) {
    values_[site] += parameters(hop.destination, site) - parameters(hop.source, site);
  }
}

}  // namespace vmc
