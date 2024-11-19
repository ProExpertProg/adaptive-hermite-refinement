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
    return std::exp(-(res * grid.kPerp2(kx, ky)) * dt / (1.0 + grid.kPerp2(kx, ky) * de * de));
  }
};

class Nu : protected Base {
public:
  using Base::Base;
  using Base::update;

  Real operator()(Dim kx, Dim ky) { return std::exp(-(nu * grid.kPerp2(kx, ky)) * dt); }
};

class NuG : protected Base {
public:
  using Base::Base;
  using Base::update;

  Real operator()(Dim kx, Dim ky) { return std::exp(-(nu * grid.kPerp2(kx, ky)) * dt); }
};

class GM : protected Base {
public:
  using Base::Base;
  using Base::update;

  Real operator()(Dim m) { return std::exp(-(Real(m) * nu_ei) * dt); }
};

} // namespace ahr::exp