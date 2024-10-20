#include "Naive.hpp"
#include "NaiveTester.hpp"
#include "util.hpp"

#include <cnpy.h>

#include <filesystem>
#include <gtest/gtest.h>
#include <utility>

namespace fs = std::filesystem;
namespace ahr {

constexpr Real HIGH_PRECISION_RTOL = 1e-14;
constexpr Real HIGH_PRECISION_ATOL = 1e-15;
constexpr Real LOW_PRECISION_RTOL = 1e-10;
constexpr Real LOW_PRECISION_ATOL = 1e-10;

#ifdef TEST_MOMENTS_HIGH_PRECISION
constexpr Real ATOL = HIGH_PRECISION_ATOL;
constexpr Real RTOL = HIGH_PRECISION_RTOL;
#else
constexpr Real RTOL = LOW_PRECISION_RTOL;
constexpr Real ATOL = LOW_PRECISION_ATOL;
#endif

template <param_like Param> class NaiveMomentsBase : public NaiveTester<Param> {
protected:
  using Base = NaiveTester<Param>;
  std::string getFilename(Dim m) {
    auto const p = Base::GetParam();

    // Tests are run from inside the `test` directory
    auto const filename = p.to_param_str() + "_m" + std::to_string(m) + ".npy";
    return fs::current_path() / "_test_data" / filename;
  }

  // Owning holder of a npy array with convenience to see it as an mdspan.
  // We can use this to avoid needlessly copying into an mdarray.
  class NpyMdspan {
    cnpy::NpyArray array_;

  public:
    explicit NpyMdspan(cnpy::NpyArray array) : array_(std::move(array)) {}

    // TODO(luka) const view
    Naive::ViewXY view() {
      std::span<size_t, 2> const extents{array_.shape.data(), 2};
      return Naive::ViewXY{array_.data<Real>(), extents};
    }

    [[nodiscard]] bool valid() const { return array_.word_size == sizeof(Real); }
  };

  NpyMdspan readMoment(Dim m) {
    auto const filename = getFilename(m);
    return NpyMdspan{cnpy::npy_load(filename)};
  }
};

using MomentParam = WithEquilibrium<WithDiffusion<NaiveParam>>;
using NaiveMoments = NaiveMomentsBase<MomentParam>;

TEST_P(NaiveMoments, CheckMoments) {
  auto const f0 = getFilename(0);
  // When updating values, comment this line
  ASSERT_TRUE(fs::exists(f0)) << "File " << f0 << " does not exist!";

  auto const p = GetParam();

  nu = p.nu;
  res = p.res;

  Naive naive{p.M, p.X, p.X};

  naive.init(p.equilibrium);
  naive.run(p.N + 1, 0); // no saving

  for (Dim m = 0; m < p.M; m++) {
    // To update values, uncomment these 2 lines
    // std::cout << "WARNING!: Overwriting " << getFilename(m) << std::endl;
    // naive.exportToNpy(getFilename(m), naive.getMoment(m));

    auto npy = readMoment(m);
    ASSERT_TRUE(npy.valid());
    auto const max_val = std::ranges::max(std::span(npy.view().data_handle(), npy.view().size()));
    auto const n = static_cast<Real>(p.N);
    auto const rtol = RTOL * n;
    auto const atol = (ATOL + max_val * RTOL) * n;
    EXPECT_THAT(naive.getMoment(m).to_mdspan(), MdspanElementsAllClose(npy.view(), rtol, atol))
        << "Moment " << m << " mismatch! Max value:" << std::setprecision(16) << max_val
        << ", atol: " << atol;
  }
}

using namespace testing;
INSTANTIATE_TEST_SUITE_P(
    NaiveMomentsSmallM, NaiveMoments,
    ConvertGenerator<MomentParam::Tuple>(Combine(Values(2, 4),          // M
                                                 Values(32, 64, 128),   // X
                                                 Values(20),            // N
                                                 Values(0.0, 0.1, 1.0), // res
                                                 Values(0.1, 1.0), // nu - if 0, energy blows up
                                                 Values("OT01", "gauss") // equilibrium
                                                 )),
    NaiveMoments::Printer{});

INSTANTIATE_TEST_SUITE_P(
    NaiveMomentsLargeM, NaiveMoments,
    ConvertGenerator<MomentParam::Tuple>(Combine(Values(10, 20),   // M
                                                 Values(32, 64),   // X
                                                 Values(20),       // N
                                                 Values(0.1, 1.0), // res
                                                 Values(0.1, 1.0), // nu - if 0, energy blows up
                                                 Values("OT01", "gauss") // equilibrium
                                                 )),
    NaiveMoments::Printer{});

using NaiveMomentsMultiRun = NaiveMomentsBase<MomentParam>;

TEST_P(NaiveMomentsMultiRun, RunMultipleTimes) {
  auto const p = GetParam();

  nu = p.nu;
  res = p.res;

  Naive naive{p.M, p.X, p.X};
  Naive naive2{p.M, p.X, p.X};

  naive.init(p.equilibrium);
  naive2.init(p.equilibrium);

  // Run both for N timesteps
  naive.run(p.N, 0);

  // Split the run into two parts for naive2
  naive2.run(1, 0);
  naive2.run(p.N - 1, 0);

  // Check that both produced the same results
  for (int m = 0; m < p.M; m++) {
    auto const moment = naive.getMoment(m).to_mdspan();
    auto const moment2 = naive2.getMoment(m).to_mdspan();
    EXPECT_THAT(moment, MdspanElementsAllClose(moment2, HIGH_PRECISION_RTOL, HIGH_PRECISION_ATOL))
        << "Moment " << m << " mismatch!";
  }
}

INSTANTIATE_TEST_SUITE_P(
    NaiveMomentsMultiRun, NaiveMomentsMultiRun,
    ConvertGenerator<MomentParam::Tuple>(Combine(Values(2, 5),     // M
                                                 Values(128),      // X
                                                 Values(2, 5),     // N
                                                 Values(0.0, 1.0), // res
                                                 Values(0.1, 1.0), // nu - if 0, energy blows up
                                                 Values("OT01", "gauss") // equilibrium
                                                 )),
    NaiveMomentsMultiRun::Printer{});

} // namespace ahr