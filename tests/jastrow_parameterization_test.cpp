#include "vmc/jastrow_parameterization.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using testing::ElementsAre;
using testing::ElementsAreArray;
using vmc::BosonState;
using vmc::DistanceJastrowParameterization;
using vmc::FullMatrixJastrowParameterization;
using vmc::OccupancyConstraint;

std::vector<double> values_of(const Eigen::VectorXd& values) {
  return {values.data(), values.data() + values.size()};
}

std::vector<int> values_of(const Eigen::MatrixXi& values) {
  std::vector<int> flattened;
  flattened.reserve(static_cast<std::size_t>(values.size()));
  for (Eigen::Index row = 0; row < values.rows(); ++row) {
    for (Eigen::Index col = 0; col < values.cols(); ++col) {
      flattened.push_back(values(row, col));
    }
  }
  return flattened;
}

MATCHER_P(DoubleNear, expected, "") {
  return testing::ExplainMatchResult(testing::DoubleNear(expected, 1.0e-12), arg, result_listener);
}

TEST(FullMatrixJastrowParameterizationTest, BuildsZeroMatrix) {
  const FullMatrixJastrowParameterization parameterization{3};

  EXPECT_EQ(parameterization.site_count(), 3);
  EXPECT_EQ(parameterization.parameter_count(), 6);
  EXPECT_TRUE(parameterization.parameters().isZero());
  EXPECT_TRUE(parameterization.matrix().isZero());
}

TEST(FullMatrixJastrowParameterizationTest, PacksUpperTriangleParameters) {
  FullMatrixJastrowParameterization parameterization{3};
  Eigen::VectorXd parameters(6);
  parameters << 0.0, 1.0, 2.0, 3.0, 4.0, 5.0;

  parameterization.set_parameters(parameters);

  Eigen::MatrixXd expected(3, 3);
  expected << 0.0, 1.0, 2.0, 1.0, 3.0, 4.0, 2.0, 4.0, 5.0;
  EXPECT_TRUE(parameterization.matrix().isApprox(expected));
  EXPECT_THAT(values_of(parameterization.parameters()), ElementsAreArray(values_of(parameters)));
}

TEST(FullMatrixJastrowParameterizationTest, BuildsFromSymmetricMatrix) {
  Eigen::MatrixXd matrix(2, 2);
  matrix << 0.25, -0.5, -0.5, 1.25;

  const FullMatrixJastrowParameterization parameterization{matrix};

  EXPECT_THAT(values_of(parameterization.parameters()), ElementsAre(0.25, -0.5, 1.25));
  EXPECT_TRUE(parameterization.matrix().isApprox(matrix));
}

TEST(FullMatrixJastrowParameterizationTest, UpdatesParameters) {
  FullMatrixJastrowParameterization parameterization{2};
  Eigen::VectorXd parameters(3);
  parameters << 1.0, 2.0, 3.0;
  Eigen::VectorXd delta(3);
  delta << 0.5, -1.0, 2.0;

  parameterization.set_parameters(parameters);
  parameterization.update_parameters(delta);

  EXPECT_THAT(values_of(parameterization.parameters()), ElementsAre(1.5, 1.0, 5.0));
}

TEST(FullMatrixJastrowParameterizationTest, ComputesLogDerivativesForIndependentSymmetricPairs) {
  const FullMatrixJastrowParameterization parameterization{3};
  const BosonState state = BosonState::from_occupations({2, 3, 1}, OccupancyConstraint::SoftCore);

  const Eigen::VectorXd derivatives = parameterization.log_derivatives(state);

  EXPECT_THAT(values_of(derivatives),
              ElementsAre(DoubleNear(-2.0), DoubleNear(-6.0), DoubleNear(-2.0), DoubleNear(-4.5),
                          DoubleNear(-3.0), DoubleNear(-0.5)));
}

TEST(FullMatrixJastrowParameterizationTest, RejectsInvalidInputs) {
  EXPECT_THROW(FullMatrixJastrowParameterization{0}, std::invalid_argument);

  Eigen::MatrixXd nonsquare = Eigen::MatrixXd::Zero(2, 3);
  EXPECT_THROW(FullMatrixJastrowParameterization{nonsquare}, std::invalid_argument);

  Eigen::MatrixXd nonsymmetric(2, 2);
  nonsymmetric << 1.0, 2.0, 3.0, 4.0;
  EXPECT_THROW(FullMatrixJastrowParameterization{nonsymmetric}, std::invalid_argument);

  Eigen::MatrixXd nonfinite = Eigen::MatrixXd::Zero(2, 2);
  nonfinite(0, 1) = std::numeric_limits<double>::infinity();
  nonfinite(1, 0) = std::numeric_limits<double>::infinity();
  EXPECT_THROW(FullMatrixJastrowParameterization{nonfinite}, std::invalid_argument);

  FullMatrixJastrowParameterization parameterization{2};
  EXPECT_THROW(parameterization.set_parameters(Eigen::VectorXd::Zero(2)), std::invalid_argument);

  Eigen::VectorXd invalid_parameters(3);
  invalid_parameters << 0.0, std::numeric_limits<double>::quiet_NaN(), 0.0;
  EXPECT_THROW(parameterization.set_parameters(invalid_parameters), std::invalid_argument);

  const BosonState wrong_state =
      BosonState::from_occupations({1, 0, 0}, OccupancyConstraint::SoftCore);
  EXPECT_THROW((void)parameterization.log_derivatives(wrong_state), std::invalid_argument);
}

TEST(DistanceJastrowParameterizationTest, BuildsPeriodicChainShells) {
  const DistanceJastrowParameterization parameterization =
      DistanceJastrowParameterization::periodic_chain(5);

  EXPECT_EQ(parameterization.site_count(), 5);
  EXPECT_EQ(parameterization.parameter_count(), 3);
  EXPECT_TRUE(parameterization.parameters().isZero());
  EXPECT_THAT(
      values_of(parameterization.shell_indices()),
      ElementsAre(0, 1, 2, 2, 1, 1, 0, 1, 2, 2, 2, 1, 0, 1, 2, 2, 2, 1, 0, 1, 1, 2, 2, 1, 0));
}

TEST(DistanceJastrowParameterizationTest, BuildsMatrixFromPeriodicChainParameters) {
  Eigen::VectorXd parameters(3);
  parameters << 0.5, 1.5, 2.5;

  const DistanceJastrowParameterization parameterization =
      DistanceJastrowParameterization::periodic_chain(5, parameters);

  Eigen::MatrixXd expected(5, 5);
  expected << 0.5, 1.5, 2.5, 2.5, 1.5, 1.5, 0.5, 1.5, 2.5, 2.5, 2.5, 1.5, 0.5, 1.5, 2.5, 2.5, 2.5,
      1.5, 0.5, 1.5, 1.5, 2.5, 2.5, 1.5, 0.5;
  EXPECT_TRUE(parameterization.matrix().isApprox(expected));
  EXPECT_THAT(values_of(parameterization.parameters()), ElementsAre(0.5, 1.5, 2.5));
}

TEST(DistanceJastrowParameterizationTest, UpdatesParameters) {
  DistanceJastrowParameterization parameterization =
      DistanceJastrowParameterization::periodic_chain(4);
  Eigen::VectorXd parameters(3);
  parameters << 1.0, 2.0, 3.0;
  Eigen::VectorXd delta(3);
  delta << -0.5, 1.5, 2.0;

  parameterization.set_parameters(parameters);
  parameterization.update_parameters(delta);

  EXPECT_THAT(values_of(parameterization.parameters()), ElementsAre(0.5, 3.5, 5.0));
}

TEST(DistanceJastrowParameterizationTest, ComputesPeriodicChainLogDerivativesByDistanceShell) {
  const DistanceJastrowParameterization parameterization =
      DistanceJastrowParameterization::periodic_chain(4);
  const BosonState state =
      BosonState::from_occupations({1, 2, 0, 3}, OccupancyConstraint::SoftCore);

  const Eigen::VectorXd derivatives = parameterization.log_derivatives(state);

  EXPECT_THAT(values_of(derivatives),
              ElementsAre(DoubleNear(-7.0), DoubleNear(-5.0), DoubleNear(-6.0)));
}

TEST(DistanceJastrowParameterizationTest, SupportsCustomContiguousShells) {
  Eigen::MatrixXi shells(3, 3);
  shells << 0, 1, 1, 1, 0, 2, 1, 2, 0;
  Eigen::VectorXd parameters(3);
  parameters << 0.5, 1.5, -1.0;
  const DistanceJastrowParameterization parameterization{shells, parameters};
  const BosonState state = BosonState::from_occupations({2, 1, 3}, OccupancyConstraint::SoftCore);

  Eigen::MatrixXd expected_matrix(3, 3);
  expected_matrix << 0.5, 1.5, 1.5, 1.5, 0.5, -1.0, 1.5, -1.0, 0.5;

  EXPECT_TRUE(parameterization.matrix().isApprox(expected_matrix));
  EXPECT_THAT(values_of(parameterization.log_derivatives(state)),
              ElementsAre(DoubleNear(-7.0), DoubleNear(-8.0), DoubleNear(-3.0)));
}

TEST(DistanceJastrowParameterizationTest, RejectsInvalidInputs) {
  EXPECT_THROW((void)DistanceJastrowParameterization::periodic_chain(0), std::invalid_argument);
  EXPECT_THROW((void)DistanceJastrowParameterization::periodic_chain(5, Eigen::VectorXd::Zero(2)),
               std::invalid_argument);

  Eigen::MatrixXi nonsquare = Eigen::MatrixXi::Zero(2, 3);
  EXPECT_THROW(DistanceJastrowParameterization(nonsquare, Eigen::VectorXd::Zero(1)),
               std::invalid_argument);

  Eigen::MatrixXi negative(2, 2);
  negative << 0, -1, -1, 0;
  EXPECT_THROW(DistanceJastrowParameterization(negative, Eigen::VectorXd::Zero(1)),
               std::invalid_argument);

  Eigen::MatrixXi nonsymmetric(2, 2);
  nonsymmetric << 0, 1, 2, 0;
  EXPECT_THROW(DistanceJastrowParameterization(nonsymmetric, Eigen::VectorXd::Zero(3)),
               std::invalid_argument);

  Eigen::MatrixXi noncontiguous(2, 2);
  noncontiguous << 0, 2, 2, 0;
  EXPECT_THROW(DistanceJastrowParameterization(noncontiguous, Eigen::VectorXd::Zero(3)),
               std::invalid_argument);

  DistanceJastrowParameterization parameterization =
      DistanceJastrowParameterization::periodic_chain(4);
  Eigen::VectorXd invalid_parameters(3);
  invalid_parameters << 0.0, std::numeric_limits<double>::infinity(), 0.0;
  EXPECT_THROW(parameterization.set_parameters(invalid_parameters), std::invalid_argument);

  const BosonState wrong_state =
      BosonState::from_occupations({1, 0, 0}, OccupancyConstraint::SoftCore);
  EXPECT_THROW((void)parameterization.log_derivatives(wrong_state), std::invalid_argument);
}

}  // namespace
