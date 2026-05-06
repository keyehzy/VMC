#pragma once

#include <optional>
#include <random>

#include "vmc/boson_state.hpp"
#include "vmc/lattice.hpp"

namespace vmc {

struct BosonHop {
  BosonState::Boson boson;
  BosonState::Site source;
  BosonState::Site destination;
};

[[nodiscard]] bool operator==(const BosonHop& lhs, const BosonHop& rhs);

[[nodiscard]] std::optional<BosonHop> propose_nearest_neighbor_hop(const Lattice& lattice,
                                                                   const BosonState& state,
                                                                   std::mt19937_64& rng);

}  // namespace vmc
