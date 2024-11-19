#pragma once
#include "constants.hpp"
#include "grid.hpp"

namespace ahr {
struct HyperCoefficients {
  Real nu_g, nu_2, eta2, nu_ei;

  static HyperCoefficients calculate(Real dt, Grid const &g) {
    return {0.0, 0.0, 0.0, 0.0};
    Real kPerpMax2 = std::pow(g.KX, 2) + std::pow(Real(g.KY) / 2, 2);

    HyperCoefficients ret{};
    ret.nu_g = hyper_coef_g / dt / std::pow(kPerpMax2, hyper_order_g);
    ret.nu_2 = hyper_coef / dt / std::pow(kPerpMax2, hyper_order);
    if (kPerpMax2 * de * de > 1) {
      ret.eta2 = hyper_coef / dt / std::pow(kPerpMax2, hyper_order - 1) * de * de;
    } else {
      ret.eta2 = hyper_coef / dt / std::pow(kPerpMax2, hyper_order);
    }

    ret.nu_ei = hyperm_coef / dt / std::pow(g.M, 2 * hyper_morder);

    return ret;
  }
};
} // namespace ahr