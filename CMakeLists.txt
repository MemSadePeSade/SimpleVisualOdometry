cmake_minimum_required(VERSION 2.8)
project( mono-vo )
find_package( OpenCV 3.0 REQUIRED )
find_package( Boost  REQUIRED )


include_directories( ${Boost_INCLUDE_DIRS} )
include_directories( ${OpenCV_INCLUDE_DIRS} )

file(GLOB viso
    "src/*.h"
    "src/main.cpp"
)
#add_executable( vo ${viso} )
add_executable( vo ${viso} )
target_link_libraries( vo ${OpenCV_LIBS} 
                       optimized ${Boost_LIBRARY_DIRS}/boost_filesystem-vc140-mt.lib
						   debug ${Boost_LIBRARY_DIRS}/../debug/lib/boost_filesystem-vc140-gd.lib)

