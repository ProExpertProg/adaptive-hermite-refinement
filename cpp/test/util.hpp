#pragma once

#include <fftw-cpp/fftw-cpp.h>
#include <gmock/gmock.h>
#include <iomanip>
#include <tuple>

// Function to slice the tuple
template <std::size_t Start, std::size_t End, typename Tuple, std::size_t... Indices>
auto slice_impl(const Tuple &t, std::index_sequence<Indices...>) {
  return std::make_tuple(std::get<Start + Indices>(t)...);
}

template <std::size_t Start, std::size_t End, typename Tuple> auto slice(const Tuple &t) {
  static_assert(Start <= End, "Start index must be less than or equal to End index");
  static_assert(End <= std::tuple_size<Tuple>::value, "End index out of bounds");
  return slice_impl<Start, End>(t, std::make_index_sequence<End - Start>{});
}

template <std::size_t Start, typename... Args> auto slice(const std::tuple<Args...> &t) {
  return slice<Start, sizeof...(Args)>(t);
}

template <typename T> T absdiff_no_overflow(T a, T b) { return std::max(a, b) - std::min(a, b); }

template <typename T> T absdiff_no_overflow(std::complex<T> a, std::complex<T> b) {
  auto const real_diff = absdiff_no_overflow(a.real(), b.real());
  auto const imag_diff = absdiff_no_overflow(a.imag(), b.imag());
  return std::abs(std::complex<T>{real_diff, imag_diff});
}

using ::testing::PrintToString;
MATCHER_P3(AllClose, val, rel_tol, abs_tol,
           PrintToString(val) + " ±" + PrintToString(abs_tol) + " (±" +
               PrintToString(double(rel_tol) * std::abs(val)) + ")") {
  auto diff = absdiff_no_overflow(val, arg);
  double tolerance_diff = double(diff) - double(abs_tol) - double(rel_tol) * std::abs(val);
  return tolerance_diff <= 0;
}

// abs is zero by default
template <typename V, typename R> auto AllClose(V &&val, R &&rel_tol) {
  return AllClose(std::forward<V>(val), std::forward<R>(rel_tol), 0.0);
}

MATCHER_P3(LeTolerant, val, rel_tol, abs_tol,
           "less or equal than " + PrintToString(val) + " ±" + PrintToString(abs_tol) + " (±" +
               PrintToString(double(rel_tol) * std::abs(val)) + ")") {
  auto diff = double(arg) - double(val);
  double tolerance_diff = double(diff) - double(abs_tol) - double(rel_tol) * std::abs(val);

  return tolerance_diff <= 0;
}

// abs is zero by default
template <typename V, typename R> auto LeTolerant(V &&val, R &&rel_tol) {
  return LeTolerant(std::forward<V>(val), std::forward<R>(rel_tol), 0.0);
}

MATCHER_P3(GeTolerant, val, rel_tol, abs_tol,
           "greater or equal than " + PrintToString(val) + " ±" + PrintToString(abs_tol) + " (±" +
               PrintToString(double(rel_tol) * std::abs(val)) + ")") {
  auto diff = double(arg) - double(val);
  double tolerance_diff = double(diff) + double(abs_tol) + double(rel_tol) * std::abs(val);

  return tolerance_diff >= 0;
}

// abs is zero by default
template <typename V, typename R> auto GeTolerant(V &&val, R &&rel_tol) {
  return GeTolerant(std::forward<V>(val), std::forward<R>(rel_tol), 0.0);
}

namespace stdex = std::experimental;
template <class T> static constexpr bool is_mdspan_v = false;

template <class... Args> static constexpr bool is_mdspan_v<stdex::mdspan<Args...>> = true;

MATCHER_P3(MdspanElementsAllClose, vals, rel_tol, abs_tol,
           "Elements within " + PrintToString(abs_tol) + " (abs) and " + PrintToString(rel_tol) +
               " (rel) of \n" + PrintToString(vals)) {
  static_assert(is_mdspan_v<std::decay_t<decltype(vals)>> &&
                    is_mdspan_v<std::decay_t<decltype(arg)>>,
                "ElementsAllClose only works with mdspan");

#define _CHECK_EQ(a, b)                                                                            \
  do {                                                                                             \
    bool _result = ::testing::ExplainMatchResult(::testing::Eq(b), a, result_listener);            \
    if (!_result) { return false; }                                                                \
  } while (0)

  _CHECK_EQ(vals.rank(), arg.rank());
  _CHECK_EQ(vals.extents(), arg.extents());

  // recursively index into both mdspans
  auto recurse = [&]<size_t Rank = 0>(auto self, std::array<size_t, Rank> indices = {}) {
    auto constexpr v_rank = decltype(vals)::rank();
    if constexpr (v_rank == Rank) {
      auto val = vals(indices);
      auto arg_val = arg(indices);
      bool close =
          ::testing::ExplainMatchResult(AllClose(val, rel_tol, abs_tol), arg_val, result_listener);
      if (!close) {
        *result_listener << std::setprecision(16) << val << " vs " << arg_val << " at index "
                         << PrintToString(indices);
        return false;
      }
    } else {
      std::array<size_t, Rank + 1> indices_new{};
      std::ranges::copy(indices, indices_new.begin());
      for (size_t idx = 0; idx < vals.extent(Rank); idx++) {
        indices_new[Rank] = idx;
        if (!self(self, indices_new)) { return false; }
      }
    }
    return true;
  };

  return recurse(recurse);
}

template <class V, class R> auto MdspanElementsAllClose(V &&vals, R &&rel_tol) {
  return MdspanElementsAllClose(std::forward<V>(vals), std::forward<R>(rel_tol), 0.0);
}

namespace std::experimental {
/// This function provides an overload to GoogleTest for printing an mdspan.
/// It will always print at the highest precision (16 for double).
/// It must be in the same namespace as mdspan.
/// This overload is only instantiated if an ostream operator<< for mdspan exists.
template <typename T, typename Extents, typename Layout, typename Accessor>
  requires requires(mdspan<T, Extents, Layout, Accessor> const &m, std::ostream &o) { o << m; }
void PrintTo(mdspan<T, Extents, Layout, Accessor> const &m, ::std::ostream *os) {
  *os << std::setprecision(16) << m;
}

} // namespace std::experimental
