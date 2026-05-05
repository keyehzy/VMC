#include <Eigen/Dense>
#include <iostream>

#include "vmc/version.hpp"

int main() {
  const Eigen::Vector2d sanity_check{1.0, 0.0};
  std::cout << "VMC " << vmc::version() << " Eigen sanity check: " << sanity_check.norm() << '\n';
  return 0;
}
