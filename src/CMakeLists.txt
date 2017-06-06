add_library(dashenfu_lib 
			ImageConsProd.cpp
			RMVideoCapture.cpp)

target_link_libraries(dashenfu_lib ${OpenCV_LIBRARIES})
