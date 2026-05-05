#include "vmc/boson_state.hpp"

#include <numeric>
#include <stdexcept>

namespace vmc {
namespace {

void require_site_count(std::size_t site_count) {
  if (site_count == 0) {
    throw std::invalid_argument("boson state must contain at least one site");
  }
}

void require_site_in_range(BosonState::Site site, std::size_t site_count) {
  if (site >= site_count) {
    throw std::out_of_range("site index is outside the boson state");
  }
}

void require_boson_in_range(BosonState::Boson boson, std::size_t boson_count) {
  if (boson >= boson_count) {
    throw std::out_of_range("boson index is outside the boson state");
  }
}

void require_hardcore_occupations(std::span<const std::size_t> occupations) {
  for (const std::size_t occupation : occupations) {
    if (occupation > 1) {
      throw std::invalid_argument("hard-core boson states cannot have occupation greater than one");
    }
  }
}

std::vector<BosonState::Site> positions_from_occupations(std::span<const std::size_t> occupations) {
  const std::size_t boson_count =
      std::accumulate(occupations.begin(), occupations.end(), std::size_t{0});

  std::vector<BosonState::Site> positions;
  positions.reserve(boson_count);

  for (BosonState::Site site = 0; site < occupations.size(); ++site) {
    for (std::size_t count = 0; count < occupations[site]; ++count) {
      positions.push_back(site);
    }
  }

  return positions;
}

std::vector<std::size_t> occupations_from_positions(std::size_t site_count,
                                                    std::span<const BosonState::Site> positions) {
  std::vector<std::size_t> occupations(site_count, 0);

  for (const BosonState::Site site : positions) {
    require_site_in_range(site, site_count);
    ++occupations[site];
  }

  return occupations;
}

}  // namespace

BosonState BosonState::empty(std::size_t site_count, OccupancyConstraint constraint) {
  require_site_count(site_count);
  return BosonState{std::vector<std::size_t>(site_count, 0), {}, constraint};
}

BosonState BosonState::from_occupations(std::vector<std::size_t> occupations,
                                        OccupancyConstraint constraint) {
  require_site_count(occupations.size());
  if (constraint == OccupancyConstraint::HardCore) {
    require_hardcore_occupations(occupations);
  }

  auto positions = positions_from_occupations(occupations);
  return BosonState{std::move(occupations), std::move(positions), constraint};
}

BosonState BosonState::from_boson_positions(std::size_t site_count,
                                            std::vector<Site> boson_positions,
                                            OccupancyConstraint constraint) {
  require_site_count(site_count);

  auto occupations = occupations_from_positions(site_count, boson_positions);
  if (constraint == OccupancyConstraint::HardCore) {
    require_hardcore_occupations(occupations);
  }

  return BosonState{std::move(occupations), std::move(boson_positions), constraint};
}

std::size_t BosonState::site_count() const {
  return occupations_.size();
}

std::size_t BosonState::boson_count() const {
  return boson_positions_.size();
}

OccupancyConstraint BosonState::constraint() const {
  return constraint_;
}

std::span<const std::size_t> BosonState::occupations() const {
  return occupations_;
}

std::span<const BosonState::Site> BosonState::boson_positions() const {
  return boson_positions_;
}

std::size_t BosonState::occupation(Site site) const {
  require_site_in_range(site, site_count());
  return occupations_[site];
}

BosonState::Site BosonState::boson_position(Boson boson) const {
  require_boson_in_range(boson, boson_count());
  return boson_positions_[boson];
}

bool BosonState::is_hardcore() const {
  return constraint_ == OccupancyConstraint::HardCore;
}

bool BosonState::can_move(Boson boson, Site destination) const {
  require_boson_in_range(boson, boson_count());
  require_site_in_range(destination, site_count());

  if (boson_positions_[boson] == destination) {
    return false;
  }

  if (is_hardcore() && occupations_[destination] > 0) {
    return false;
  }

  return true;
}

void BosonState::move_boson(Boson boson, Site destination) {
  if (!can_move(boson, destination)) {
    throw std::invalid_argument("boson move violates occupancy constraints");
  }

  const Site source = boson_positions_[boson];
  --occupations_[source];
  ++occupations_[destination];
  boson_positions_[boson] = destination;
}

BosonState::BosonState(std::vector<std::size_t> occupations, std::vector<Site> boson_positions,
                       OccupancyConstraint constraint)
    : occupations_{std::move(occupations)},
      boson_positions_{std::move(boson_positions)},
      constraint_{constraint} {}

}  // namespace vmc
