
#find_package(PCL 1.6 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


include_directories(
  ${BULLET_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${Eigen_INCLUDE_DIRS}
  ${OpenRAVE_INCLUDE_DIRS}
  ${JSON_INCLUDE_DIRS}
)

add_executable(cloud_convex_decomp_experiment cloud_convex_decomp_experiment.cpp)
target_link_libraries(cloud_convex_decomp_experiment trajopt cloudproc osgviewer ${OpenRAVE_LIBRARIES})

add_executable(polygon_expt polygon_expt.cpp)

add_executable(test_dynamics test_dynamics.cpp)
target_link_libraries(test_dynamics sco trajopt ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY} ${OpenRAVE_BOTH_LIBRARIES})

add_executable(test_box_dynamics test_box_dynamics.cpp)
target_link_libraries(test_box_dynamics sco trajopt ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY} ${OpenRAVE_BOTH_LIBRARIES})
