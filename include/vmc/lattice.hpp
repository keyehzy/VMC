#pragma once

#include <cstddef>
#include <span>
#include <utility>
#include <vector>

namespace vmc {

enum class BoundaryCondition {
  Open,
  Periodic,
};

class Lattice {
 public:
  using Site = std::size_t;
  using Bond = std::pair<Site, Site>;

  static Lattice chain(std::size_t length, BoundaryCondition boundary);
  static Lattice square(std::size_t width, std::size_t height, BoundaryCondition boundary);

  [[nodiscard]] std::size_t site_count() const;
  [[nodiscard]] std::size_t bond_count() const;

  [[nodiscard]] std::span<const Site> neighbors(Site site) const;
  [[nodiscard]] std::span<const Bond> bonds() const;

  [[nodiscard]] bool are_neighbors(Site a, Site b) const;

 private:
  Lattice(std::size_t site_count, std::vector<Bond> bonds);

  std::size_t site_count_{0};
  std::vector<std::size_t> neighbor_offsets_;
  std::vector<Site> neighbors_;
  std::vector<Bond> bonds_;
};

}  // namespace vmc
