#include "Naive.h"

#include <utility>

namespace {
    template<class T, class ...Args, size_t ...I>
    auto forward_to_array_impl(std::index_sequence<I...> indices, Args &&...args) {
        return std::array<T, indices.size()>{
                [&args...](size_t index) {
                    return T{std::forward<Args>(args)...};
                }(I)...
        };
    }


    template<class T, size_t Size, class ...Args>
    std::array<T, Size> forward_to_array(Args &&...args) {
        return forward_to_array_impl<T>(std::make_index_sequence<Size>(), std::forward<Args>(args)...);
    }
}

namespace ahr {
    Naive::Naive(std::ostream &out, Dim M, Dim X, Dim Y) : HermiteRunner(out), M(M), X(X), Y(Y),
                                                           momentsPH(M, X, Y), moments(M, X, Y), momentsPH_X(M, X, Y),
                                                           momentsPH_Y(M, X, Y), moments_X(M, X, Y), moments_Y(M, X, Y),
                                                           phiPH(X, Y),
                                                           temp{forward_to_array<fftw::mdbuffer<2u>, 3u>(X, Y)} {}

    void Naive::init(mdspan<Real, dextents<Dim, 3u>> initialMoments, Dim N_, Real initialDT_) {
        initialDT = initialDT_;
        N = N_;
        assert(M == initialMoments.extent(0));
        assert(X == initialMoments.extent(1));
        assert(Y == initialMoments.extent(2));

        // Currently assuming X==Y for simplicity, but the code is written generally for the most part.
        assert(X == Y);

        // Initialize moments
        for_each_mxy([&](Dim m, Dim x, Dim y) {
            moments(m, x, y) = initialMoments(m, x, y);
        });

        // Plan FFTs both ways
        plan = fftw::plan<2u>::dft(sliceXY(moments, 0), sliceXY(momentsPH, 0), fftw::FORWARD, fftw::MEASURE);
        planInv = fftw::plan<2u>::dft(sliceXY(momentsPH, 0), sliceXY(moments, 0), fftw::BACKWARD, fftw::MEASURE);

        // Transform moments into phase space
        for (int m = 0; m < M; ++m) {
            plan(sliceXY(moments, m), sliceXY(momentsPH, m));
        }
    }

    void Naive::run() {

        for (int t = 0; t < N; ++t) {
            // predictor step

            for (int m = 2; m < M; ++m) {
                // BEGIN(bracket)
                for_each_xy([&](Dim kx, Dim ky) {
                    momentsPH_X(m, kx, ky) = -double(kx) * 1i * momentsPH(m, kx, ky);
                    momentsPH_Y(m, kx, ky) = -double(ky) * 1i * momentsPH(m, kx, ky);
                });

                planInv(sliceXY(momentsPH_X, m), sliceXY(moments_X, m));
                planInv(sliceXY(momentsPH_Y, m), sliceXY(moments_Y, m));

                for_each_xy([&](Dim x, Dim y) {
                    moments_X(m, x, y) =
                            moments_X(m, x, y) * moments_Y(A_PAR, x, y) - moments_Y(m, x, y) * moments_X(A_PAR, x, y);
                });

                plan(sliceXY(moments_X, m), sliceXY(momentsPH_X, m));
                // END(bracket)
                // TODO extract into function
            }

            // corrector step
            for (int m = 0; m < M; ++m) {

            }
        }
    }

    mdarray<Real, dextents<Dim, 3u>> Naive::getFinalValues() {
        for (int m = 0; m < M; ++m) {
            planInv(sliceXY(momentsPH, m), sliceXY(moments, m));
        }

        mdarray<Real, dextents<Dim, 3u>> result{M, X, Y};
        for_each_mxy([&](Dim m, Dim x, Dim y) {
            result(m, x, y) = moments(m, x, y).real();
        });

        return result;
    }
}