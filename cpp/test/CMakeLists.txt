include(FetchContent)

FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG 1d17ea141d2c11b8917d2c7d029f1c4e2b9769b2
)
FetchContent_MakeAvailable(googletest)

function(add_viriato_test_impl NAME)
    add_executable(${NAME} main.cpp ${ARGN})
    target_link_libraries(${NAME} src-lib GTest::gmock)
    add_test(NAME ${NAME} COMMAND ${NAME} WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endfunction()

function(add_viriato_test FILE)
    string(REGEX REPLACE ".cpp$" "" NAME ${FILE})
    add_viriato_test_impl(${NAME} ${FILE} ${ARGN})
endfunction()

add_viriato_test(naive-energies.cpp)
add_viriato_test(naive-moments.cpp)
