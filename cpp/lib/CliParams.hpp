#pragma

#include <argparse/argparse.hpp>

class CliParams {
  using Dim = ahr::Dim;

public:
  CliParams(std::string_view name);

  struct Params {
    Dim X, M, N, saveInterval;
  };

  void parse(int argc, const char *argv[]);
  Params get();

private:
  argparse::ArgumentParser arguments;
};

inline CliParams::CliParams(std::string_view name) : arguments(std::string{name}) {
  arguments.add_argument("X")
      .help("Size of X, Y for domain")
      .scan<'i', Dim>()
      .default_value(Dim{256});

  arguments.add_argument("M")
      .help("Total number of moments")
      .scan<'i', Dim>()
      .default_value(Dim{100});

  arguments.add_argument("N")
      .help("Number of timesteps")
      .scan<'i', Dim>()
      .default_value(Dim{20});

  arguments.add_argument("save-interval")
      .help("How often the code should write out the results")
      .scan<'i', Dim>()
      .default_value(Dim{1000});
}

inline void CliParams::parse(int argc, const char **argv) {
  arguments.parse_args(argc, argv);
  if (arguments.get<Dim>("M") == 3) {
    throw std::invalid_argument("2 or at least 4 moments required");
  }
}

inline CliParams::Params CliParams::get() {
  return {.X = arguments.get<Dim>("X"),
          .M = arguments.get<Dim>("M"),
          .N = arguments.get<Dim>("N"),
          .saveInterval = arguments.get<Dim>("save-interval")};
}
