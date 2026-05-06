#pragma once

#include <Eigen/Dense>
#include <cstddef>

#include "vmc/boson_state.hpp"

namespace vmc {

class JastrowParameterization {
 public:
  virtual ~JastrowParameterization() = default;

  [[nodiscard]] virtual std::size_t site_count() const = 0;
  [[nodiscard]] virtual std::size_t parameter_count() const = 0;

  [[nodiscard]] virtual Eigen::VectorXd parameters() const = 0;
  virtual void set_parameters(const Eigen::VectorXd& parameters) = 0;
  virtual void update_parameters(const Eigen::VectorXd& delta) = 0;

  [[nodiscard]] virtual Eigen::MatrixXd matrix() const = 0;
  [[nodiscard]] virtual Eigen::VectorXd log_derivatives(const BosonState& state) const = 0;
};

class FullMatrixJastrowParameterization final : public JastrowParameterization {
 public:
  explicit FullMatrixJastrowParameterization(std::size_t site_count);
  explicit FullMatrixJastrowParameterization(Eigen::MatrixXd matrix);

  [[nodiscard]] std::size_t site_count() const override;
  [[nodiscard]] std::size_t parameter_count() const override;

  [[nodiscard]] Eigen::VectorXd parameters() const override;
  void set_parameters(const Eigen::VectorXd& parameters) override;
  void update_parameters(const Eigen::VectorXd& delta) override;

  [[nodiscard]] Eigen::MatrixXd matrix() const override;
  [[nodiscard]] Eigen::VectorXd log_derivatives(const BosonState& state) const override;

 private:
  Eigen::MatrixXd matrix_;
};

class DistanceJastrowParameterization final : public JastrowParameterization {
 public:
  DistanceJastrowParameterization(Eigen::MatrixXi shell_indices, Eigen::VectorXd parameters);

  [[nodiscard]] static DistanceJastrowParameterization periodic_chain(std::size_t site_count);
  [[nodiscard]] static DistanceJastrowParameterization periodic_chain(std::size_t site_count,
                                                                      Eigen::VectorXd parameters);

  [[nodiscard]] std::size_t site_count() const override;
  [[nodiscard]] std::size_t parameter_count() const override;

  [[nodiscard]] Eigen::VectorXd parameters() const override;
  void set_parameters(const Eigen::VectorXd& parameters) override;
  void update_parameters(const Eigen::VectorXd& delta) override;

  [[nodiscard]] Eigen::MatrixXd matrix() const override;
  [[nodiscard]] Eigen::VectorXd log_derivatives(const BosonState& state) const override;

  [[nodiscard]] const Eigen::MatrixXi& shell_indices() const;

 private:
  Eigen::MatrixXi shell_indices_;
  Eigen::VectorXd parameters_;
};

}  // namespace vmc
