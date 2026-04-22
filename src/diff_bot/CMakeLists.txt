cmake_minimum_required(VERSION 3.8)
project(diff_bot)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch urdf config worlds meshes models
  DESTINATION share/${PROJECT_NAME}
)
install(
  PROGRAMS
    scripts/nav.py
    scripts/map.py
  DESTINATION lib/${PROJECT_NAME}
)
ament_package()