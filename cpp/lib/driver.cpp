#include "typedefs.hpp"

#include "CliParams.hpp"
#include "HermiteRunner.hpp"
#include "Naive.hpp"
#include <iostream>

// TODO
using namespace ahr;

int main(int argc, const char *argv[]) {
    CliParams cliParams{"naive"};

    try {
        cliParams.parse(argc, argv);
    }
    catch (const std::runtime_error &err) {
        std::cerr << err.what() << std::endl;
        return 1;
    }

    auto const p = cliParams.get();
    Naive naive{std::cout, p.M, p.X, p.X};
    HermiteRunner &runner = naive;

    runner.init("OT01");
    runner.run(p.N, p.saveInterval);
}
