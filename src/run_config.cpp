#include "vmc/run_config.hpp"

#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace vmc {
namespace {

std::string require_value(int argc, char* argv[], int& index, std::string_view flag) {
  if (index + 1 >= argc) {
    throw std::invalid_argument("missing value for " + std::string{flag});
  }

  ++index;
  return argv[index];
}

std::uint64_t parse_uint64(std::string_view value, std::string_view flag) {
  std::uint64_t parsed = 0;
  const char* begin = value.data();
  const char* end = value.data() + value.size();
  const auto result = std::from_chars(begin, end, parsed);
  if (result.ec != std::errc{} || result.ptr != end) {
    throw std::invalid_argument("invalid unsigned integer for " + std::string{flag});
  }
  return parsed;
}

std::size_t parse_size(std::string_view value, std::string_view flag) {
  const std::uint64_t parsed = parse_uint64(value, flag);
  if (parsed > std::numeric_limits<std::size_t>::max()) {
    throw std::invalid_argument("value is too large for " + std::string{flag});
  }
  return static_cast<std::size_t>(parsed);
}

double parse_double(std::string_view value, std::string_view flag) {
  std::string owned_value{value};
  char* parse_end = nullptr;
  errno = 0;
  const double parsed = std::strtod(owned_value.c_str(), &parse_end);
  if (parse_end != owned_value.c_str() + owned_value.size() || errno == ERANGE ||
      !std::isfinite(parsed)) {
    throw std::invalid_argument("invalid finite number for " + std::string{flag});
  }
  return parsed;
}

void require_positive(std::size_t value, std::string_view name) {
  if (value == 0) {
    throw std::invalid_argument(std::string{name} + " must be greater than zero");
  }
}

void require_positive(double value, std::string_view name) {
  if (!std::isfinite(value) || value <= 0.0) {
    throw std::invalid_argument(std::string{name} + " must be finite and positive");
  }
}

void require_nonnegative(double value, std::string_view name) {
  if (!std::isfinite(value) || value < 0.0) {
    throw std::invalid_argument(std::string{name} + " must be finite and nonnegative");
  }
}

void validate_config(const RunConfig& config) {
  if (config.chain_length < 3) {
    throw std::invalid_argument("chain length must be at least 3 for periodic chains");
  }

  if (config.occupancy_constraint == OccupancyConstraint::HardCore &&
      config.boson_count > config.chain_length) {
    throw std::invalid_argument("hard-core boson count cannot exceed chain length");
  }

  require_positive(config.optimization.max_iterations, "max iterations");
  require_nonnegative(config.optimization.energy_tolerance, "energy tolerance");
  require_positive(config.optimization.iteration.sample_count, "sample count");
  require_positive(config.optimization.iteration.sweeps_between_samples, "sweeps between samples");
  require_positive(config.optimization.iteration.update.step_size, "SR step size");
  require_nonnegative(config.optimization.iteration.update.diagonal_shift, "SR diagonal shift");
  require_nonnegative(config.optimization.iteration.update.singular_tolerance,
                      "SR singular tolerance");

  if (!std::isfinite(config.hamiltonian.hopping_t) ||
      !std::isfinite(config.hamiltonian.interaction_u)) {
    throw std::invalid_argument("Hamiltonian parameters must be finite");
  }
}

}  // namespace

RunConfig default_run_config() {
  return RunConfig{
      .seed = 1234,
      .chain_length = 8,
      .boson_count = 4,
      .occupancy_constraint = OccupancyConstraint::HardCore,
      .hamiltonian =
          BoseHubbardHamiltonian{
              .hopping_t = 1.0,
              .interaction_u = 0.0,
          },
      .optimization =
          SrOptimizationConfig{
              .max_iterations = 6,
              .energy_tolerance = 1.0e-4,
              .iteration =
                  SrIterationConfig{
                      .thermalization_sweeps = 50,
                      .sample_count = 400,
                      .sweeps_between_samples = 1,
                      .update =
                          SrUpdateConfig{
                              .step_size = 0.05,
                              .diagonal_shift = 0.1,
                          },
                  },
          },
  };
}

ParsedRunConfig parse_run_config(int argc, char* argv[]) {
  RunConfig config = default_run_config();
  bool help_requested = false;

  for (int index = 1; index < argc; ++index) {
    const std::string_view flag{argv[index]};

    if (flag == "--help") {
      help_requested = true;
    } else if (flag == "--hard-core") {
      config.occupancy_constraint = OccupancyConstraint::HardCore;
    } else if (flag == "--soft-core") {
      config.occupancy_constraint = OccupancyConstraint::SoftCore;
    } else if (flag == "--seed") {
      config.seed = parse_uint64(require_value(argc, argv, index, flag), flag);
    } else if (flag == "--chain-length") {
      config.chain_length = parse_size(require_value(argc, argv, index, flag), flag);
    } else if (flag == "--bosons") {
      config.boson_count = parse_size(require_value(argc, argv, index, flag), flag);
    } else if (flag == "--max-iterations") {
      config.optimization.max_iterations = parse_size(require_value(argc, argv, index, flag), flag);
    } else if (flag == "--thermalization-sweeps") {
      config.optimization.iteration.thermalization_sweeps =
          parse_size(require_value(argc, argv, index, flag), flag);
    } else if (flag == "--samples") {
      config.optimization.iteration.sample_count =
          parse_size(require_value(argc, argv, index, flag), flag);
    } else if (flag == "--sweeps-between-samples") {
      config.optimization.iteration.sweeps_between_samples =
          parse_size(require_value(argc, argv, index, flag), flag);
    } else if (flag == "--energy-tolerance") {
      config.optimization.energy_tolerance =
          parse_double(require_value(argc, argv, index, flag), flag);
    } else if (flag == "--step-size") {
      config.optimization.iteration.update.step_size =
          parse_double(require_value(argc, argv, index, flag), flag);
    } else if (flag == "--diagonal-shift") {
      config.optimization.iteration.update.diagonal_shift =
          parse_double(require_value(argc, argv, index, flag), flag);
    } else if (flag == "--hopping-t") {
      config.hamiltonian.hopping_t = parse_double(require_value(argc, argv, index, flag), flag);
    } else if (flag == "--interaction-u") {
      config.hamiltonian.interaction_u = parse_double(require_value(argc, argv, index, flag), flag);
    } else {
      throw std::invalid_argument("unknown option: " + std::string{flag});
    }
  }

  if (!help_requested) {
    validate_config(config);
  }

  return ParsedRunConfig{
      .config = config,
      .help_requested = help_requested,
  };
}

std::string usage(std::string_view executable_name) {
  std::ostringstream output;
  output << "Usage: " << executable_name << " [options]\n\n"
         << "Options:\n"
         << "  --chain-length N              periodic chain length (default: 8)\n"
         << "  --bosons N                    number of bosons (default: 4)\n"
         << "  --hard-core                   enforce n_i in {0,1} (default)\n"
         << "  --soft-core                   allow unrestricted site occupation\n"
         << "  --seed N                      random seed (default: 1234)\n"
         << "  --max-iterations N            SR iteration limit (default: 6)\n"
         << "  --thermalization-sweeps N     sweeps before each SR measurement (default: 50)\n"
         << "  --samples N                   samples per SR iteration (default: 400)\n"
         << "  --sweeps-between-samples N    sweeps between samples (default: 1)\n"
         << "  --energy-tolerance X          convergence tolerance (default: 1e-4)\n"
         << "  --step-size X                 SR update step size (default: 0.05)\n"
         << "  --diagonal-shift X            SR covariance diagonal shift (default: 0.1)\n"
         << "  --hopping-t X                 Bose-Hubbard hopping t (default: 1.0)\n"
         << "  --interaction-u X             Bose-Hubbard interaction U (default: 0.0)\n"
         << "  --help                        show this help text\n";
  return output.str();
}

}  // namespace vmc
