#pragma once

#include "Exponentials.hpp"

namespace ahr::exp {

template <typename Exp>
concept space_like = requires(Exp exp, Dim kx, Dim ky) {
  { exp(kx, ky) } -> std::same_as<Real>;
};

template <typename Exp>
concept moment_like = requires(Exp exp, Dim m) {
  { exp(m) } -> std::same_as<Real>;
};

template <typename Exp>
concept updatable = requires(Exp exp, Real dt) {
  { exp.update(HyperCoefficients{}, dt) };
};

template <updatable Exp>
  requires space_like<Exp>
struct CachedKXKY : Exp {
  explicit CachedKXKY(Grid const &grid) : Exp(grid), factors(grid.KX, grid.KY), dt(-1) {}

  void update(HyperCoefficients hyper, Real dt) {
    // skip update if dt hasn't changed
    if (this->dt == dt) { return; }
    this->dt = dt;

    Exp::update(hyper, dt);
    this->grid.for_each_kxky([&](Dim kx, Dim ky) { factors(kx, ky) = Exp::operator()(kx, ky); });
  }

  Real operator()(Dim kx, Dim ky) { return factors(kx, ky); }

protected:
  Grid::Buf::R_XY factors; ///< Real but of size (KX,KY)
  Real dt;
};

template <updatable Exp>
  requires moment_like<Exp>
struct CachedM : Exp {
  explicit CachedM(Grid const &grid) : Exp(grid), factors(grid.M), dt(-1) {}

  void update(HyperCoefficients hyper, Real dt) {
    // skip update if dt hasn't changed
    if (this->dt == dt) { return; }
    this->dt = dt;

    Exp::update(hyper, dt);
    for (Dim m = 0; m < this->grid.M; ++m) {
      factors[m] = Exp::operator()(m);
    }
  }

  Real operator()(Dim m) { return factors[m]; }

protected:
  std::vector<Real> factors;
  Real dt;
};
} // namespace ahr::exp