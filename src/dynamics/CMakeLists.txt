include_directories(
  ${BULLET_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${Eigen_INCLUDE_DIRS}
  ${OpenRAVE_INCLUDE_DIRS}
  ${JSON_INCLUDE_DIRS}
)

set(CMAKE_CXX_FLAGS "-std=c++0x")

add_library(dynamics dynamics_problem.cpp box.cpp contacts.cpp)
target_link_libraries(dynamics sco trajopt ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY} ${OpenRAVE_BOTH_LIBRARIES})

add_executable(test_dynamics test_dynamics.cpp)
target_link_libraries(test_dynamics sco trajopt ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY} ${OpenRAVE_BOTH_LIBRARIES})

add_executable(test_box_dynamics test_box_dynamics.cpp)
target_link_libraries(test_box_dynamics dynamics)

add_executable(test_quat test_quat.cpp)
