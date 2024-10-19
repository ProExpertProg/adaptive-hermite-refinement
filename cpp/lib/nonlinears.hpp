#pragma once

#include "constants.hpp"

namespace ahr {
    [[nodiscard]] inline Real Gamma0(Real x) {
        Real ax = std::abs(x);

        if (ax < 3.75) {
            Real y = std::pow(x / 3.75, 2);
            return 1.0 + y * (3.5156229 +
                              y * (3.0899424 + y * (1.2067492 + y * (0.2659732 + y * (0.360768e-1 + y * 0.45813e-2)))));
        } else {
            Real y = 3.75 / ax;
            return (std::exp(ax) / std::sqrt(ax)) *
                   (0.39894228 + y * (0.1328592e-1 + y *
                                                     (0.225319e-2 + y * (-0.157565e-2 + y *
                                                                                        (0.916281e-2 +
                                                                                         y * (-0.2057706e-1 + y *
                                                                                                              (0.2635537e-1 +
                                                                                                               y *
                                                                                                               (-0.1647633e-1 +
                                                                                                                y *
                                                                                                                0.392377e-2))))))));
        }
    }

    namespace nonlinear {

        [[nodiscard]] inline Complex phi(Complex ne_K, Real kPerp2) {
            if (rhoI < smallRhoI)
                return -ne_K / kPerp2;
            else
                return rhoI * rhoI * 0.5 / (Gamma0(kPerp2 * rhoI * rhoI * 0.5) - 1.0) * ne_K;
        }

        [[nodiscard]] inline Complex phiInv(Complex phi_K, Real kPerp2) {
            if (rhoI < smallRhoI)
                return - kPerp2 * phi_K;
            else
                return 2.0 / (rhoI * rhoI) * (Gamma0(kPerp2 * rhoI * rhoI * 0.5) - 1.0) * phi_K;
        }

        [[nodiscard]] inline Complex semiImplicitOp(Real dt, Real bPerpMax, Real aa0, Real kPerp2) {
            if (rhoI <= smallRhoI) {
                return aa0 * aa0 * (1 + kPerp2 * (3.0 / 4.0 * rhoI * rhoI + rhoS * rhoS)) *
                       kPerp2 * std::pow( bPerpMax * dt, 2) / (1 + kPerp2 * de * de);
            } else {
                return aa0 * aa0 * (3.0 * rhoS * rhoS - rhoI * rhoI / (Gamma0(0.5 * kPerp2 * rhoI * rhoI) - 1)) *
                       std::pow(kPerp2 * bPerpMax * dt, 2) / (1 + kPerp2 * de * de);
            }
        }

        [[nodiscard]] inline Complex N(Complex bracketPhiNE_K, Complex bracketAParUEKPar_K) {
            return -bracketPhiNE_K + bracketAParUEKPar_K;
        }

        [[nodiscard]] inline Complex A(Complex bracketAParPhiG2Ne_K, Complex bracketPhiDeUEKPar_K,
                                       Real kPerp2) {
            return (bracketAParPhiG2Ne_K - de * de * bracketPhiDeUEKPar_K) / (1 + de * de * kPerp2);
        }

        [[nodiscard]] inline Complex G2(Complex bracketPhiG2_K, Complex bracketAParG3_K,
                                        Complex bracketAParUEKPar_K) {
            return -bracketPhiG2_K +
                   std::sqrt(3.0) * rhoS / de * bracketAParG3_K +
                   std::sqrt(2.0) * bracketAParUEKPar_K;
        }

        [[nodiscard]] inline Complex GM(Dim m, Complex bracketPhiGM_K, Complex bracketAParGMMinusPlus_K) {
            return -bracketPhiGM_K + rhoS / de * bracketAParGMMinusPlus_K;
        }

        [[nodiscard]] inline Complex GLast(Complex bracketPhiGLast_K, Complex bracketTotalGLast_K) {
            return -bracketPhiGLast_K + bracketTotalGLast_K;
        }

        [[nodiscard]] inline Complex GLastBracketFactor(Dim M, Real kPerp2, HyperCoefficients hyper) {
            // Note that this is ngtot + 1 in Viriato. Here, the last moment is M-1, so this should be M.
            auto M1 = double(M);
            return (rhoS * rhoS / de / de * M1) /
                   (M1 * nu_ei + std::pow(M1, 2 * hyper_morder) * hyper.nu_ei +
                    nu * kPerp2 + hyper.nu_2 * std::pow(kPerp2, hyper_order));
        }
    }
}
