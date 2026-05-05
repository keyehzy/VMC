#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "vmc/lattice.hpp"

namespace vmc {

enum class OccupancyConstraint {
  SoftCore,
  HardCore,
};

class BosonState {
 public:
  using Site = Lattice::Site;
  using Boson = std::size_t;

  static BosonState empty(std::size_t site_count, OccupancyConstraint constraint);
  static BosonState from_occupations(std::vector<std::size_t> occupations,
                                     OccupancyConstraint constraint);
  static BosonState from_boson_positions(std::size_t site_count, std::vector<Site> boson_positions,
                                         OccupancyConstraint constraint);

  [[nodiscard]] std::size_t site_count() const;
  [[nodiscard]] std::size_t boson_count() const;
  [[nodiscard]] OccupancyConstraint constraint() const;

  [[nodiscard]] std::span<const std::size_t> occupations() const;
  [[nodiscard]] std::span<const Site> boson_positions() const;

  [[nodiscard]] std::size_t occupation(Site site) const;
  [[nodiscard]] Site boson_position(Boson boson) const;

  [[nodiscard]] bool is_hardcore() const;
  [[nodiscard]] bool can_move(Boson boson, Site destination) const;

  void move_boson(Boson boson, Site destination);

 private:
  BosonState(std::vector<std::size_t> occupations, std::vector<Site> boson_positions,
             OccupancyConstraint constraint);

  std::vector<std::size_t> occupations_;
  std::vector<Site> boson_positions_;
  OccupancyConstraint constraint_;
};

}  // namespace vmc
