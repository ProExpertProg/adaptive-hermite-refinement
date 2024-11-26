#pragma once

#include "constants.hpp"
#include "grid.hpp"
#include "hyper.hpp"

namespace ahr::exp {
class Base {
public:
  explicit Base(Grid const &grid) : grid(grid) {}

  void update(HyperCoefficients hyper, Real dt) {
    this->hyper = hyper;
    this->dt = dt;
  }

protected:
  Grid const &grid;
  HyperCoefficients hyper{};
  Real dt{-1};
};

class Eta : protected Base {
public:
  using Base::Base;
  using Base::update;

  Real operator()(Dim kx, Dim ky) {
    return std::exp(
        -(res * grid.kPerp2(kx, ky) + hyper.eta2 * std::pow(grid.kPerp2(kx, ky), hyper_order)) *
        dt / (1.0 + grid.kPerp2(kx, ky) * de * de));
  }
};

class Nu : protected Base {
public:
  using Base::Base;
  using Base::update;

  Real operator()(Dim kx, Dim ky) {
    return std::exp(
        -(nu * grid.kPerp2(kx, ky) + hyper.nu_2 * std::pow(grid.kPerp2(kx, ky), hyper_order)) * dt);
  }
};

class NuG : protected Base {
public:
  using Base::Base;
  using Base::update;

  Real operator()(Dim kx, Dim ky) {
    return std::exp(
        -(nu * grid.kPerp2(kx, ky) + hyper.nu_g * std::pow(grid.kPerp2(kx, ky), hyper_order)) * dt);
  }
};

class GM : protected Base {
public:
  using Base::Base;
  using Base::update;

  Real operator()(Dim m) {
    return std::exp(-(Real(m) * nu_ei + std::pow(m, 2 * hyper_morder) * hyper.nu_ei) * dt);
  }
};

} // namespace ahr::exp