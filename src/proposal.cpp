#include "vmc/proposal.hpp"

#include <stdexcept>

namespace vmc {

bool operator==(const BosonHop& lhs, const BosonHop& rhs) {
  return lhs.boson == rhs.boson && lhs.source == rhs.source && lhs.destination == rhs.destination;
}

std::optional<BosonHop> propose_nearest_neighbor_hop(const Lattice& lattice,
                                                     const BosonState& state,
                                                     std::mt19937_64& rng) {
  if (lattice.site_count() != state.site_count()) {
    throw std::invalid_argument("lattice and boson state must have the same site count");
  }

  if (state.boson_count() == 0) {
    return std::nullopt;
  }

  std::uniform_int_distribution<BosonState::Boson> boson_distribution{0, state.boson_count() - 1};
  const BosonState::Boson boson = boson_distribution(rng);
  const BosonState::Site source = state.boson_position(boson);

  const auto neighbors = lattice.neighbors(source);
  if (neighbors.empty()) {
    return std::nullopt;
  }

  std::uniform_int_distribution<std::size_t> neighbor_distribution{0, neighbors.size() - 1};
  const BosonState::Site destination = neighbors[neighbor_distribution(rng)];

  return BosonHop{
      .boson = boson,
      .source = source,
      .destination = destination,
  };
}

}  // namespace vmc
