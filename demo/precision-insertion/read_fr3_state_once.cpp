// Read one FR3 FCI state without sending a motion or setup command.
// Build this separate diagnostic against the local libfranka package.
#include <franka/robot.h>

#include <iomanip>
#include <iostream>
#include <string>

template <std::size_t N>
void print_array(const std::array<double, N>& values) {
  std::cout << '[';
  for (std::size_t i = 0; i < N; ++i) {
    if (i) std::cout << ',';
    std::cout << values[i];
  }
  std::cout << ']';
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: read_fr3_state_once <FR3-FCI-IP>\n";
    return 2;
  }
  try {
    franka::Robot robot(argv[1], franka::RealtimeConfig::kIgnore);
    const franka::RobotState state = robot.readOnce();
    std::cout << std::setprecision(17);
    std::cout << "{\"q\":";
    print_array(state.q);
    std::cout << ",\"dq\":";
    print_array(state.dq);
    std::cout << ",\"O_T_EE_column_major\":";
    print_array(state.O_T_EE);
    std::cout << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FR3 state read failed: " << error.what() << '\n';
    return 1;
  }
}
