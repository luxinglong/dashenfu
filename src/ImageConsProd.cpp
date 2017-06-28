#include "ImageConsProd.hpp"
#include "RMVideoCapture.hpp"
#include "ShenfuDetector.hpp"
#include "LedDebugger.hpp"

#include <unistd.h>

using namespace std;
using namespace cv;

#define BUFFER_SIZE 3

// multi-threads shared value
volatile unsigned int prdIdx;
volatile unsigned int csmIdx;

struct ImageData{
	Mat img;
	unsigned int frame;	
};

ImageData data[BUFFER_SIZE];

//#define USE_VIDEO

void ImageConsProd::ImageProducer(){
#ifdef USE_VIDEO
	string video_path = "/home/luban-master/Desktop/WorkSpace/dashenfu/data/4.MP4";
	VideoCapture cap(video_path);
	if (!cap.isOpened())
			cout << "Can not open the video file!" << endl;
#else
	RMVideoCapture cap("/dev/video0", 3);
	cap.setVideoFormat(640, 480, 0);
	cap.setExposureTime(0,79);
	cap.startStream();
	cap.info();

	//VideoCapture cap(0);
	//if (!cap.isOpened())
			//cout << "Can not open the camera!" << endl;
#endif
	
	while(1){
		while(prdIdx - csmIdx >= BUFFER_SIZE);
		cap >> data[prdIdx % BUFFER_SIZE].img;
		data[prdIdx % BUFFER_SIZE].frame++;

		++prdIdx;
	}
}

void ImageConsProd::ImageConsumer(){
	
	ShenfuDetector shenfu_detector;
	shenfu_detector.loadModel();

	// vars for debug use
	//LedDebugger led("/sys/class/gpio/gpio158/value");
	//led.ledON();

	Mat src;
	int frame_num = 0;
	while(1){
		while(prdIdx-csmIdx == 0);
		data[csmIdx % BUFFER_SIZE].img.copyTo(src);
		frame_num = data[csmIdx % BUFFER_SIZE].frame;
		++csmIdx;
		// sleep(); // int seconds
		//usleep(120000); // int micro-seconds

		if(src.rows != 480)
			continue;
		
		pair<int, int> res = shenfu_detector.getTarget(src);
		if (waitKey(1)>=0)
			break;
	}
}
