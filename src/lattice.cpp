#include "vmc/lattice.hpp"

#include <algorithm>
#include <stdexcept>

namespace vmc {
namespace {

void require_site_count(std::size_t site_count) {
  if (site_count == 0) {
    throw std::invalid_argument("lattice must contain at least one site");
  }
}

void require_periodic_extent(std::size_t extent) {
  if (extent < 3) {
    throw std::invalid_argument("periodic lattice extents must be at least 3");
  }
}

Lattice::Bond canonical_bond(Lattice::Site a, Lattice::Site b) {
  if (a < b) {
    return {a, b};
  }
  return {b, a};
}

void add_bond(std::vector<Lattice::Bond>& bonds, Lattice::Site a, Lattice::Site b) {
  if (a == b) {
    return;
  }
  bonds.push_back(canonical_bond(a, b));
}

}  // namespace

Lattice Lattice::chain(std::size_t length, BoundaryCondition boundary) {
  require_site_count(length);
  if (boundary == BoundaryCondition::Periodic) {
    require_periodic_extent(length);
  }

  std::vector<Bond> bonds;
  bonds.reserve(boundary == BoundaryCondition::Periodic ? length : length - 1);

  for (Site site = 0; site + 1 < length; ++site) {
    add_bond(bonds, site, site + 1);
  }

  if (boundary == BoundaryCondition::Periodic) {
    add_bond(bonds, length - 1, 0);
  }

  return Lattice{length, std::move(bonds)};
}

Lattice Lattice::square(std::size_t width, std::size_t height, BoundaryCondition boundary) {
  require_site_count(width);
  require_site_count(height);
  if (boundary == BoundaryCondition::Periodic) {
    require_periodic_extent(width);
    require_periodic_extent(height);
  }

  const auto index = [width](std::size_t x, std::size_t y) { return y * width + x; };

  std::vector<Bond> bonds;
  bonds.reserve(boundary == BoundaryCondition::Periodic
                    ? 2 * width * height
                    : width * (height - 1) + height * (width - 1));

  for (std::size_t y = 0; y < height; ++y) {
    for (std::size_t x = 0; x < width; ++x) {
      const Site site = index(x, y);

      if (x + 1 < width) {
        add_bond(bonds, site, index(x + 1, y));
      } else if (boundary == BoundaryCondition::Periodic) {
        add_bond(bonds, site, index(0, y));
      }

      if (y + 1 < height) {
        add_bond(bonds, site, index(x, y + 1));
      } else if (boundary == BoundaryCondition::Periodic) {
        add_bond(bonds, site, index(x, 0));
      }
    }
  }

  return Lattice{width * height, std::move(bonds)};
}

std::size_t Lattice::site_count() const {
  return site_count_;
}

std::size_t Lattice::bond_count() const {
  return bonds_.size();
}

std::span<const Lattice::Site> Lattice::neighbors(Site site) const {
  if (site >= site_count_) {
    throw std::out_of_range("site index is outside the lattice");
  }

  const std::size_t begin = neighbor_offsets_[site];
  const std::size_t end = neighbor_offsets_[site + 1];
  return std::span<const Site>{neighbors_.data() + begin, end - begin};
}

std::span<const Lattice::Bond> Lattice::bonds() const {
  return bonds_;
}

bool Lattice::are_neighbors(Site a, Site b) const {
  if (a >= site_count_ || b >= site_count_) {
    throw std::out_of_range("site index is outside the lattice");
  }

  const auto adjacent = neighbors(a);
  return std::binary_search(adjacent.begin(), adjacent.end(), b);
}

Lattice::Lattice(std::size_t site_count, std::vector<Bond> bonds) : site_count_{site_count} {
  std::sort(bonds.begin(), bonds.end());
  bonds.erase(std::unique(bonds.begin(), bonds.end()), bonds.end());
  bonds_ = std::move(bonds);

  std::vector<std::vector<Site>> adjacency(site_count_);
  for (const auto& [a, b] : bonds_) {
    adjacency[a].push_back(b);
    adjacency[b].push_back(a);
  }

  neighbor_offsets_.reserve(site_count_ + 1);
  neighbor_offsets_.push_back(0);

  for (auto& site_neighbors : adjacency) {
    std::sort(site_neighbors.begin(), site_neighbors.end());
    site_neighbors.erase(std::unique(site_neighbors.begin(), site_neighbors.end()),
                         site_neighbors.end());

    neighbors_.insert(neighbors_.end(), site_neighbors.begin(), site_neighbors.end());
    neighbor_offsets_.push_back(neighbors_.size());
  }
}

}  // namespace vmc
