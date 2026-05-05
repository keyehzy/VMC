#include "vmc/initialization.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace vmc {
namespace {

void validate_config(const InitialBosonConfig& config) {
  if (config.site_count == 0) {
    throw std::invalid_argument("initial boson config must contain at least one site");
  }

  if (config.constraint == OccupancyConstraint::HardCore &&
      config.boson_count > config.site_count) {
    throw std::invalid_argument("hard-core initialization requires boson_count <= site_count");
  }
}

std::vector<BosonState::Site> initialize_softcore_positions(const InitialBosonConfig& config,
                                                            std::mt19937_64& rng) {
  std::vector<BosonState::Site> positions;
  positions.reserve(config.boson_count);

  std::uniform_int_distribution<BosonState::Site> site_distribution{0, config.site_count - 1};
  for (std::size_t boson = 0; boson < config.boson_count; ++boson) {
    positions.push_back(site_distribution(rng));
  }

  return positions;
}

std::vector<BosonState::Site> initialize_hardcore_positions(const InitialBosonConfig& config,
                                                            std::mt19937_64& rng) {
  std::vector<BosonState::Site> sites(config.site_count);
  std::iota(sites.begin(), sites.end(), BosonState::Site{0});
  std::shuffle(sites.begin(), sites.end(), rng);
  sites.resize(config.boson_count);
  return sites;
}

}  // namespace

BosonState initialize_boson_state(const InitialBosonConfig& config, std::mt19937_64& rng) {
  validate_config(config);

  std::vector<BosonState::Site> positions;
  if (config.constraint == OccupancyConstraint::HardCore) {
    positions = initialize_hardcore_positions(config, rng);
  } else {
    positions = initialize_softcore_positions(config, rng);
  }

  return BosonState::from_boson_positions(config.site_count, std::move(positions),
                                          config.constraint);
}

BosonState initialize_boson_state(const Lattice& lattice, std::size_t boson_count,
                                  OccupancyConstraint constraint, std::mt19937_64& rng) {
  return initialize_boson_state(
      InitialBosonConfig{
          .site_count = lattice.site_count(),
          .boson_count = boson_count,
          .constraint = constraint,
      },
      rng);
}

}  // namespace vmc
