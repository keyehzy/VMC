#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include "vmc/boson_state.hpp"
#include "vmc/hamiltonian.hpp"
#include "vmc/sr_optimization.hpp"

namespace vmc {

struct RunConfig {
  std::uint64_t seed;
  std::size_t chain_length;
  std::size_t boson_count;
  OccupancyConstraint occupancy_constraint;
  BoseHubbardHamiltonian hamiltonian;
  SrOptimizationConfig optimization;
};

struct ParsedRunConfig {
  RunConfig config;
  bool help_requested;
};

[[nodiscard]] RunConfig default_run_config();
[[nodiscard]] ParsedRunConfig parse_run_config(int argc, char* argv[]);
[[nodiscard]] std::string usage(std::string_view executable_name);

}  // namespace vmc
