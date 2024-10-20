#include <experimental/mdarray>
#include <experimental/mdspan>
#include <ostream>
#include <spdlog/fmt/bundled/format.h>

// #define _ln1(x) #x
// #define _ln2(x) _ln1(x)
// #define debug(name, var) print(__FILE__ ":" _ln2(__LINE__) " " name, var)
// #define debug2(var) debug(#var, (var))
// #d efine debugXY(var) \
//  debug2((var).DX); \ debug2((var).DY)

/// Specialize fmt::formatter for no-parameter functions.
/// This allows us to pass a lambda to a log statement which will only be evaluated if the log
/// statement is active.
template <std::invocable Fn> struct fmt::formatter<Fn> : formatter<std::invoke_result_t<Fn>> {
  template <typename FormatContext>
  auto format(Fn strFunc, FormatContext &ctx) -> decltype(ctx.out()) {
    return formatter<std::invoke_result_t<Fn>>::format(strFunc(), ctx);
  }
};

/// For passing modifiers to ostream
template <typename... Args> struct ostream_tuple {
  std::tuple<Args...> t;

  explicit ostream_tuple(Args &&...args) : t(std::forward<Args>(args)...) {}

  friend std::ostream &operator<<(std::ostream &o, ostream_tuple const &ot) {
    std::apply([&o](auto const &...args) { ((o << args), ...); }, ot.t);
    return o;
  }
};

// ================
// ostream overloads for mdspan/mdarray
// ================
namespace std::experimental { // NOLINT(*-dcl58-cpp)

// 2D-case:
template <typename T, typename Extents, typename Layout, typename Accessor>
  requires(Extents::rank() == 2)
std::ostream &operator<<(std::ostream &o, mdspan<T, Extents, Layout, Accessor> const &m) {
  for (size_t i = 0; i < m.extent(0); ++i) {
    for (size_t j = 0; j < m.extent(1); ++j) {
      o << m(i, j) << " ";
    }
    o << "\n";
  }
  return o;
}

template <typename T, typename Extents, typename Layout, typename Container>
std::ostream &operator<<(std::ostream &o, mdarray<T, Extents, Layout, Container> const &m) {
  return o << m.to_mdspan();
}

} // namespace std::experimental
