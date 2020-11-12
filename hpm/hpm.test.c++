#include <boost/ut.hpp> //import boost.ut;

#include <hpm/hpm.h++>

auto sum(auto... args) { return (args + ...); }

int main() {
  using namespace boost::ut;

  expect(42l == 42_l and 3_i == 3)
      << "Very important that these are equal, says tobben\n";

  "hello world"_test = [] {
    int i = 43;
    expect(43_i == i) << "Info about comparison inside test\n";
  };

  "sum"_test = [] {
    expect(sum(0) == 0_i);
    expect(sum(1, 2) == 3_i);
    expect(sum(1, 2) > 0_i and 41_i == sum(40, 2));
  };
}

// Real positions of markers
// blue0:  [ 144.896 ,    0    , 22]
// blue1:  [  72.4478,  125.483, 22]
// green0: [ -72.4478,  125.483, 22]
// green1: [-144.896 ,    0    , 22]
// red0:   [ -72.4478, -125.483, 22]
// red1:   [  72.4478, -125.483, 22]
//
// Span an inner hexagon of a circle with radius:
// circle_r = 144.896
//
// Camera position
// translate [0,0,0]
// rotate [45,0,0]
// distance 755
// So... at
// [            0,
//   -755/sqrt(2),
//    755/sqrt(2)]
//
// Found positions relative to the camera/camera plane:
// [ 142.799,  15.4505, 728.94 ]
// [ 71.4615,  103.155, 817.945]
// [-143.007,  15.4491, 728.896]
// [-71.7038,  103.155, 817.945]
// [ 71.5839, -72.3039, 643.429]
// [-71.7745, -72.3039, 643.429]
//
// We see that
//
// cam + rel =
//
