cmake_minimum_required(VERSION 3.8...3.30)

find_package(Eigen3 REQUIRED)
find_package(Glog REQUIRED)
find_package(OpenCV REQUIRED calib3d)

set(LIB_NAME okvis_common)
add_library(${LIB_NAME}
  include/okvis/FrameTypedefs.hpp
  include/okvis/Measurements.hpp
  include/okvis/Parameters.hpp
  include/okvis/ViFrontendInterface.hpp
  include/okvis/ViInterface.hpp
  include/okvis/ViParametersReader.hpp
  include/okvis/VioDiagnostics.hpp
  src/ViInterface.cpp
  src/ViParametersReader.cpp
  src/VioDiagnostics.cpp
)
target_include_directories(${LIB_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${GLOG_INCLUDE_DIRS}
)
target_link_libraries(${LIB_NAME}
  PUBLIC
    Eigen3::Eigen
    okvis::CV
    okvis::Kinematics
    okvis::Time
    okvis::Util
    ${OpenCV_LIBS}
  PRIVATE
    ${GLOG_LIBRARIES}
)
target_compile_features(${LIB_NAME} PUBLIC cxx_std_${OKVIS_CXX_STANDARD})
target_compile_options(${LIB_NAME}
  PUBLIC
    ${OKVIS_PUBLIC_CXX_FLAGS}
  PRIVATE
    ${OKVIS_PRIVATE_CXX_FLAGS}
)
add_library(okvis::Common ALIAS ${LIB_NAME})

install(TARGETS ${LIB_NAME} EXPORT okvisTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY include/okvis DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} FILES_MATCHING PATTERN "*.hpp")

if(BUILD_TESTS)
  set(TEST_NAME okvis_common_test)
  add_executable(${TEST_NAME}
    test/TestVioDiagnostics.cpp
    test/test_main.cpp
  )
  target_link_libraries(${TEST_NAME}
    ${LIB_NAME}
    ${GLOG_LIBRARIES}
    gtest
  )
  target_compile_options(${TEST_NAME}
    PUBLIC ${OKVIS_PUBLIC_CXX_FLAGS}
    PRIVATE ${OKVIS_PRIVATE_CXX_FLAGS}
  )
  add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endif()
