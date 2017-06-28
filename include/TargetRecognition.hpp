#pragma once

#include "opencv2/opencv.hpp"
#include <string>

class TargetRecognition{
public:
	TargetRecognition(){}

	bool loadSVMModel(const string & model_path){
		SVM.load(model_path);
		return true;
	}
	
	int recognizeHWDigit(const cv::Mat & image);
	
	int recognizeLEDDigit(const cv::Mat & image);

protected:
	bool match2array(const int* array1, const int* array2, int length);
	
private:
	cv::CvSVM SVM;	
};
