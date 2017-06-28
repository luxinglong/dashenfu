#include "TargetRecognition.hpp"

using namespace cv;
using namespace std;

int TargetRecognition::recognizeHWDigit(const cv::Mat & image){
	if(!image.data){
		cout << "The handwritten digit image isn't valid." << endl;
		return -1;
	}
	
	// preprocessing the input image
	Mat digitNorm;
	if(image.rows != 28 && image.cols != 28)
		resize(image, digitNorm, Size(28, 28));
	else
		image.copyTo(digitNorm);
	
	// TODO: InvMat shoud be cut to reduce the computation cost

	threshold(digitNorm,digitNorm,95,255,THRESH_BINARY);

	Mat resultMat = Mat::zeros(1, 1, CV_32FC1);
	Mat featureMat = Mat::zeros(1, 324, CV_32FC1);

	// extract HOG descriptor
	HOGDescriptor *hog = new HOGDescriptor(cvSize(28,28),cvSize(14,14),cvSize(7,7),cvSize(7,7),9);
	vector<float> descriptors;
	hog->compute(digitNorm, descriptors,Size(1,1), Size(0,0)); //Hog特征计算
	for (int j=0; j<descriptors.size(); j++){
		featureMat.at<float>(0,j) = descriptors[j];
	}

	// get the result
	SVM.predict(featureMat,resultMat);

	int	hw_number = (int)resultMat.at<float>(0,0);

	return hw_number;
}

int TargetRecognition::recognizeLEDDigit(const cv::Mat & image){
	if(!image.data){
		cout << "The handwritten digit image isn't valid." << endl;
		return -1;
	}
	
	// preprocessing the input image
	Mat led_img;
	if (image.rows!=112 || image.cols!=56)
		resize(image,led_img,Size(56,112));
	else
		image.copyTo(led_img);

	int database[10][7]={{2,2,2,2,2,2,0}, // 0
						 {0,2,2,0,0,0,0}, // 1
						 {2,2,0,2,2,0,2}, // 2
						 {2,2,2,2,0,0,2}, // 3
						 {0,2,2,0,0,2,2}, // 4
						 {2,0,2,2,0,2,2}, // 5 
						 {2,0,2,2,2,2,2}, // 6
						 {2,2,2,0,0,0,0}, // 7
						 {2,2,2,2,2,2,2}, // 8
						 {2,2,2,2,0,2,2}};// 9
	int result[7]={0,0,0,0,0,0,0};
	int counter0=0;
	for (int i=0; i<36; i++){ // compute index 0
		if (led_img.at<uchar>(i,30)!=led_img.at<uchar>(i+1,30))
			counter0++;
	}
	result[0]=counter0;

	int counter1=0;
	for (int j=28; j<55; j++){ // compute index 1
		if (led_img.at<uchar>(36,j)!=led_img.at<uchar>(36,j+1))
			counter1++;
	}
	result[1]=counter1;
	
	int counter2=0;
	for (int j=28; j<55; j++){ // compute index 2
		if (led_img.at<uchar>(72,j)!=led_img.at<uchar>(72,j+1))
			counter2++;
	}
	result[2]=counter2;

	int counter3=0;
	for (int i=72; i<111; i++){ // compute index 3
		if (led_img.at<uchar>(i,30)!=led_img.at<uchar>(i+1,30))
			counter3++;
	}
	result[3]=counter3;
	
	int counter4=0;
	for (int j=0; j<28; j++){ // compute index 4
		if (led_img.at<uchar>(72,j)!=led_img.at<uchar>(72,j+1))
			counter4++;
	}
	result[4]=counter4;

	int counter5=0;
	for (int j=0; j<28; j++){ // compute index 5
		if (led_img.at<uchar>(36,j)!=led_img.at<uchar>(36,j+1))
			counter5++;
	}
	result[5]=counter5;

	int counter6=0;
	for (int i=36; i<72; i++){ // compute index 6
		if (led_img.at<uchar>(i,30)!=led_img.at<uchar>(i+1,30))
			counter6++;
	}
	result[6]=counter6;
	//cout << result[0] << result[1] << result[2] << result[3] << result[4] << result[5] << result[6] << endl;
	for (int idx=0; idx<10; idx++)
	{
		if (match2array(result,database[idx],7))
			return idx;
	}
	return 0;	
}

bool TargetRecognition::match2array(const int* array1, const int* array2, int length)
{
	for (int i=0; i<length; i++)
	{
		if (array1[i]!=array2[i])
			return false;
	}
	return true;
}


