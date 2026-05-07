#include "vmc/run_config.hpp"

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace {

using vmc::OccupancyConstraint;
using vmc::ParsedRunConfig;
using vmc::RunConfig;

ParsedRunConfig parse_args(std::vector<std::string> args) {
  std::vector<char*> argv;
  argv.reserve(args.size());
  for (std::string& arg : args) {
    argv.push_back(arg.data());
  }
  return vmc::parse_run_config(static_cast<int>(argv.size()), argv.data());
}

TEST(RunConfigTest, DefaultsMatchExample) {
  const RunConfig config = vmc::default_run_config();

  EXPECT_EQ(config.seed, 1234);
  EXPECT_EQ(config.chain_length, 8);
  EXPECT_EQ(config.boson_count, 4);
  EXPECT_EQ(config.occupancy_constraint, OccupancyConstraint::HardCore);
  EXPECT_DOUBLE_EQ(config.hamiltonian.hopping_t, 1.0);
  EXPECT_DOUBLE_EQ(config.hamiltonian.interaction_u, 0.0);
  EXPECT_EQ(config.optimization.max_iterations, 6);
  EXPECT_DOUBLE_EQ(config.optimization.energy_tolerance, 1.0e-4);
  EXPECT_EQ(config.optimization.iteration.thermalization_sweeps, 50);
  EXPECT_EQ(config.optimization.iteration.sample_count, 400);
  EXPECT_EQ(config.optimization.iteration.sweeps_between_samples, 1);
  EXPECT_DOUBLE_EQ(config.optimization.iteration.update.step_size, 0.05);
  EXPECT_DOUBLE_EQ(config.optimization.iteration.update.diagonal_shift, 0.1);
}

TEST(RunConfigTest, ParsesSupportedOptions) {
  const ParsedRunConfig parsed = parse_args({"vmc",
                                             "--chain-length",
                                             "12",
                                             "--bosons",
                                             "9",
                                             "--soft-core",
                                             "--seed",
                                             "42",
                                             "--max-iterations",
                                             "11",
                                             "--thermalization-sweeps",
                                             "7",
                                             "--samples",
                                             "13",
                                             "--sweeps-between-samples",
                                             "3",
                                             "--energy-tolerance",
                                             "0.002",
                                             "--step-size",
                                             "0.03",
                                             "--diagonal-shift",
                                             "0.4",
                                             "--hopping-t",
                                             "1.5",
                                             "--interaction-u",
                                             "6.0"});

  EXPECT_FALSE(parsed.help_requested);
  EXPECT_EQ(parsed.config.chain_length, 12);
  EXPECT_EQ(parsed.config.boson_count, 9);
  EXPECT_EQ(parsed.config.occupancy_constraint, OccupancyConstraint::SoftCore);
  EXPECT_EQ(parsed.config.seed, 42);
  EXPECT_EQ(parsed.config.optimization.max_iterations, 11);
  EXPECT_EQ(parsed.config.optimization.iteration.thermalization_sweeps, 7);
  EXPECT_EQ(parsed.config.optimization.iteration.sample_count, 13);
  EXPECT_EQ(parsed.config.optimization.iteration.sweeps_between_samples, 3);
  EXPECT_DOUBLE_EQ(parsed.config.optimization.energy_tolerance, 0.002);
  EXPECT_DOUBLE_EQ(parsed.config.optimization.iteration.update.step_size, 0.03);
  EXPECT_DOUBLE_EQ(parsed.config.optimization.iteration.update.diagonal_shift, 0.4);
  EXPECT_DOUBLE_EQ(parsed.config.hamiltonian.hopping_t, 1.5);
  EXPECT_DOUBLE_EQ(parsed.config.hamiltonian.interaction_u, 6.0);
}

TEST(RunConfigTest, LastOccupancyFlagWins) {
  const ParsedRunConfig hard_core = parse_args({"vmc", "--soft-core", "--hard-core"});
  const ParsedRunConfig soft_core =
      parse_args({"vmc", "--hard-core", "--soft-core", "--bosons", "12"});

  EXPECT_EQ(hard_core.config.occupancy_constraint, OccupancyConstraint::HardCore);
  EXPECT_EQ(soft_core.config.occupancy_constraint, OccupancyConstraint::SoftCore);
}

TEST(RunConfigTest, HelpSkipsValidation) {
  const ParsedRunConfig parsed =
      parse_args({"vmc", "--help", "--chain-length", "1", "--bosons", "99"});

  EXPECT_TRUE(parsed.help_requested);
}

TEST(RunConfigTest, UsageMentionsOptions) {
  const std::string help = vmc::usage("vmc");

  EXPECT_NE(help.find("Usage: vmc [options]"), std::string::npos);
  EXPECT_NE(help.find("--chain-length"), std::string::npos);
  EXPECT_NE(help.find("--soft-core"), std::string::npos);
  EXPECT_NE(help.find("--diagonal-shift"), std::string::npos);
}

TEST(RunConfigTest, RejectsUnknownAndMissingOptions) {
  EXPECT_THROW(parse_args({"vmc", "--unknown"}), std::invalid_argument);
  EXPECT_THROW(parse_args({"vmc", "--samples"}), std::invalid_argument);
}

TEST(RunConfigTest, RejectsInvalidNumbers) {
  EXPECT_THROW(parse_args({"vmc", "--samples", "abc"}), std::invalid_argument);
  EXPECT_THROW(parse_args({"vmc", "--samples", "-1"}), std::invalid_argument);
  EXPECT_THROW(parse_args({"vmc", "--step-size", "nan"}), std::invalid_argument);
  EXPECT_THROW(parse_args({"vmc", "--step-size", "inf"}), std::invalid_argument);
}

TEST(RunConfigTest, RejectsInvalidSimulationConfig) {
  EXPECT_THROW(parse_args({"vmc", "--chain-length", "2"}), std::invalid_argument);
  EXPECT_THROW(parse_args({"vmc", "--hard-core", "--chain-length", "4", "--bosons", "5"}),
               std::invalid_argument);
  EXPECT_THROW(parse_args({"vmc", "--samples", "0"}), std::invalid_argument);
  EXPECT_THROW(parse_args({"vmc", "--max-iterations", "0"}), std::invalid_argument);
  EXPECT_THROW(parse_args({"vmc", "--sweeps-between-samples", "0"}), std::invalid_argument);
  EXPECT_THROW(parse_args({"vmc", "--step-size", "0"}), std::invalid_argument);
  EXPECT_THROW(parse_args({"vmc", "--diagonal-shift", "-0.1"}), std::invalid_argument);
}

TEST(RunConfigTest, SoftCoreAllowsMoreBosonsThanSites) {
  const ParsedRunConfig parsed =
      parse_args({"vmc", "--soft-core", "--chain-length", "4", "--bosons", "8"});

  EXPECT_EQ(parsed.config.occupancy_constraint, OccupancyConstraint::SoftCore);
  EXPECT_EQ(parsed.config.boson_count, 8);
}

}  // namespace
