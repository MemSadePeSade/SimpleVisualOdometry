cmake_minimum_required(VERSION 2.8)
project( mono-vo )

set(CMAKE_CXX_FLAGS   "${CMAKE_CXX_FLAGS} -std=c++14 -pthread") 
set(CMAKE_CXX_FLAGS   "${CMAKE_CXX_FLAGS} -march=native")

option(WITH_CUDA "adds cuda support" OFF)
if(${WITH_CUDA})
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DWITH_CUDA")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DWITH_CPU")
endif()

find_package( OpenCV 3.0 REQUIRED )
find_package(Boost COMPONENTS system filesystem REQUIRED)


include_directories( ${Boost_INCLUDE_DIRS} )
include_directories( ${OpenCV_INCLUDE_DIRS} )

file(GLOB viso
    "src/*.h"
    "src/main.cpp"
	"src/tracker.cpp"
)

add_executable(  vo ${viso} )
target_link_libraries( vo ${OpenCV_LIBS} 
                       ${Boost_FILESYSTEM_LIBRARY}
                       ${Boost_SYSTEM_LIBRARY} )

