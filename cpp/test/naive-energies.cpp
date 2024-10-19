#include "Naive.hpp"
#include "NaiveTester.hpp"
#include "util.hpp"
#include <gtest/gtest.h>
#include <sstream>

namespace ahr {
using namespace ::testing;

using NaiveEnergyParam = WithDiffusion<NaiveParam>;

class NaiveEnergy : public NaiveTester<NaiveEnergyParam> {
protected:
  Naive::Energies expectedEnergies(Real t, Naive::Energies e_init) {
    return {.magnetic = std::exp(-t * res * 2) * e_init.magnetic,
            .kinetic = std::exp(-t * nu * 2) * e_init.kinetic};
  };
};

TEST_P(NaiveEnergy, Gauss) {
  auto const p = GetParam();
  if (p.nu == 0.0 && p.res == 0.0) {
    GTEST_SKIP_("Without diffusion, energy grows uncontrollably (unless apar "
                "is the last moment).");
  }

  if (p.res != 0.0) { GTEST_SKIP_("Magnetic diffusion is currently not working as expected."); }

  nu = p.nu;
  res = p.res;

  Naive naive{out, p.M, p.X, p.X};
  naive.init("gauss");
  auto const e_init = naive.calculateEnergies();

  // Kinetic energy should be zero (absolute tolerance hence needed)
  EXPECT_THAT(e_init.kinetic, AllClose(0.0, 1e-7, 1e-6));

  naive.run(p.N, 0); // no saving
  auto const e_final = naive.calculateEnergies();
  auto const e_expected = expectedEnergies(naive.elapsedTime(), e_init);

  // Kinetic should still be zero
  EXPECT_THAT(e_final.kinetic, AllClose(0.0, 1e-7, 1e-5));
  EXPECT_THAT(e_expected.kinetic, AllClose(0.0, 1e-7, 1e-5));

  // Magnetic energy should not increase
  EXPECT_THAT(e_final.magnetic, LeTolerant(e_init.magnetic, 1e-7, 1e-5));

  // Magnetic energy should roughly diffuse.
  Real diffused = e_init.magnetic - e_final.magnetic;
  Real expected_diffusion = e_init.magnetic - e_expected.magnetic;

  // Add energy tolerance
  auto const rtol = 2e-5 * Real(p.N);
  auto const e_tol = rtol * e_init.magnetic;
  EXPECT_THAT(diffused, AllOf(GeTolerant(expected_diffusion * 0.9, 0.0, e_tol),
                              LeTolerant(expected_diffusion * 1.1, 0.0, e_tol)));

  std::cout << "mag_init: " << e_init.magnetic << " mag_final: " << e_final.magnetic
            << " mag_expected: " << e_expected.magnetic << std::endl;
  std::cout << "diffused: " << diffused << " expected_diffusion: " << expected_diffusion
            << std::endl;
}

TEST_P(NaiveEnergy, OT01) {
  auto const p = GetParam();
  if (p.M > 2 && p.nu == 0.0 && p.res == 0.0) {
    GTEST_SKIP_("Without diffusion, energy grows uncontrollably (unless apar "
                "is the last moment).");
  }

  if (p.res != 0.0) { GTEST_SKIP_("Magnetic diffusion is currently not working as expected."); }

  nu = p.nu;
  res = p.res;

  Naive naive{out, p.M, p.X, p.X};
  naive.init("OT01");
  auto const e_init = naive.calculateEnergies();

  naive.run(p.N, 0); // no saving
  auto const e_final = naive.calculateEnergies();
  auto const e_expected = expectedEnergies(naive.elapsedTime(), e_init);

  // Energy sum should not increase
  EXPECT_THAT(e_final.total(), LeTolerant(e_init.total(), 1e-7, 1e-5));

  // Energy sum should roughly diffuse.
  Real const diffused = e_init.total() - e_final.total();
  Real const expected_diffusion = e_init.total() - e_expected.total();

  // Add energy tolerance -> larger for smaller grids
  auto const grid_rtol = 2e-2 * Real(p.N) / (Real(p.X) * Real(p.X));
  auto const rtol = 1e-6 * Real(p.N) + grid_rtol;
  auto const e_tol = rtol * e_init.total();
  EXPECT_THAT(diffused, AllOf(GeTolerant(expected_diffusion * 0.8, 0.0, e_tol),
                              LeTolerant(expected_diffusion * 1.25, 0.0, e_tol)));

  std::cout << "mag_init: " << e_init.magnetic << " mag_final: " << e_final.magnetic
            << " mag_expected: " << e_expected.magnetic << std::endl;
  std::cout << "kin_init: " << e_init.kinetic << " kin_final: " << e_final.kinetic
            << " kin_expected: " << e_expected.kinetic << std::endl;
  std::cout << "diffused: " << diffused << " expected_diffusion: " << expected_diffusion
            << std::endl;
}

INSTANTIATE_TEST_SUITE_P(NaiveEnergy2TestsSmallM, NaiveEnergy,
                         ConvertGenerator<NaiveEnergyParam::Tuple>(Combine(Values(2, 4),        // M
                                                                           Values(32, 64, 128), // X
                                                                           Values(10, 20, 40),  // N
                                                                           Values(0.0, 0.1,
                                                                                  1.0), // res
                                                                           Values(0.1, 1.0))), // nu
                         NaiveEnergy::Printer{});

INSTANTIATE_TEST_SUITE_P(NaiveEnergy2TestsLargeM, NaiveEnergy,
                         ConvertGenerator<NaiveEnergyParam::Tuple>(Combine(Values(10, 20, 45), // M
                                                                           Values(32),         // X
                                                                           Values(10, 20),     // N
                                                                           Values(0.0, 0.1,
                                                                                  1.0), // res
                                                                           Values(0.1, 1.0))), // nu
                         NaiveEnergy::Printer{});
} // namespace ahr
