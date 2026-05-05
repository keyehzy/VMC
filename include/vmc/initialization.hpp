#pragma once

#include <cstddef>
#include <random>

#include "vmc/boson_state.hpp"
#include "vmc/lattice.hpp"

namespace vmc {

struct InitialBosonConfig {
  std::size_t site_count;
  std::size_t boson_count;
  OccupancyConstraint constraint;
};

BosonState initialize_boson_state(const InitialBosonConfig& config, std::mt19937_64& rng);

BosonState initialize_boson_state(const Lattice& lattice, std::size_t boson_count,
                                  OccupancyConstraint constraint, std::mt19937_64& rng);

}  // namespace vmc
