#include "TargetRecognition.hpp"

using namespace cv;
using namespace std;

int TargetRecognition::recognizeHWDigit(const cv::Mat & image){
	if(!image.data){
		cout << "The handwritten digit image isn't valid." << endl;
		return -1;
	}
	
	Mat gray;
	if(image.channels() != 1)
		cvtColor(image, gray, CV_RGB2GRAY);
	else
		image.copyTo(gray);	
	
	// preprocessing the input image
	Mat digitNorm;
	if(gray.rows != 28 && gray.cols != 28)
		resize(gray, digitNorm, Size(28, 28));
	else
		gray.copyTo(digitNorm);
	
	// TODO: InvMat shoud be cut to reduce the computation cost

	threshold(digitNorm,digitNorm,160,255,THRESH_BINARY);

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

	Mat gray;
	if(image.channels() != 1)
		cvtColor(image, gray, CV_RGB2GRAY);
	else
		image.copyTo(gray);	

	medianBlur(gray, gray, 3);
	int width, height;
	width = gray.cols;
	height = gray.rows;
	Mat grayEx(Size(width+4, height+4), CV_8UC1, Scalar(0, 0, 0));
	for (int i=2; i<height; i++){
		for (int j=2; j<width+2; j++){
			grayEx.at<uchar>(i,j) = gray.at<uchar>(i-2, j-2);
		}
	}

	//Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
	//dilate(grayEx, grayEx, element);
	resize(grayEx, grayEx, Size(56, 112));
	threshold(grayEx, grayEx, 150, 255, THRESH_BINARY);
	
	Mat gray_temp;
	grayEx.copyTo(gray_temp);
	vector<vector<Point>> contours;
	findContours(gray_temp, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
	vector<Rect> mini_rects;
	for(int i=0; i<contours.size(); i++){
		Rect rc = boundingRect(contours[i]);
		mini_rects.push_back(rc);
	}
	sort(mini_rects.begin(), mini_rects.end(), [](const Rect & r1, const Rect & r2){return r1.width*r1.height > r2.width*r2.height;});
	
	Mat led;
	if(mini_rects.size() > 1){
		if((double)(mini_rects[0].width*mini_rects[0].height)/(mini_rects[1].width*mini_rects[1].height)>2){
			led = grayEx(mini_rects[0]);
		}else{
			/*vector<Point> twoGridPoints;
			twoGridPoints.push_back(Point(mini_rects[0].x,mini_rects[0].y));
			twoGridPoints.push_back(Point(mini_rects[0].x+mini_rects[0].width,mini_rects[0].y+mini_rects[0].height));
			twoGridPoints.push_back(Point(mini_rects[1].x,mini_rects[1].y));
			twoGridPoints.push_back(Point(mini_rects[1].x+mini_rects[1].width,mini_rects[1].y+mini_rects[1].height));
			sort(twoGridPoints.begin(),twoGridPoints.end(),[](const Point & pt1, const Point & pt2){return pt1.y<pt2.y;});
			Rect joint(twoGridPoints[0].x,twoGridPoints[0].y,mini_rects[3].x-twoGridPoints[0].x,mini_rects[3].y-twoGridPoints[0].y);
			led = grayEx(joint);*/
			grayEx.copyTo(led);			
		}
	}else if(mini_rects.size() == 1){
		led = grayEx(mini_rects[0]);
	}

	Mat ledEx(Size(led.cols+10, led.rows+10), CV_8UC1, Scalar(0, 0, 0));
	for(int i=5; i<led.rows+5; i++){
		for(int j=5; j<led.cols+5; j++){
			ledEx.at<uchar>(i, j) = led.at<uchar>(i-5, j-5);
		}
	}	

	Mat led_img;
	if (ledEx.rows!=112 || ledEx.cols!=56)
		resize(ledEx,led_img,Size(56,112));
	else
		ledEx.copyTo(led_img);

	threshold(led_img,led_img,150,255,THRESH_BINARY);

	//imshow("led", led_img);

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
	for (int j=25; j<55; j++){ // compute index 2
		if (led_img.at<uchar>(72,j)!=led_img.at<uchar>(72,j+1))
			counter2++;
	}
	result[2]=counter2;

	int counter3=0;
	for (int i=72; i<111; i++){ // compute index 3
		if (led_img.at<uchar>(i,25)!=led_img.at<uchar>(i+1,25))
			counter3++;
	}
	result[3]=counter3;
	
	int counter4=0;
	for (int j=0; j<25; j++){ // compute index 4
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
		if (led_img.at<uchar>(i,28)!=led_img.at<uchar>(i+1,28))
			counter6++;
	}
	result[6]=counter6;
	//cout << result[0] << result[1] << result[2] << result[3] << result[4] << result[5] << result[6] << endl;
	int idx = 0;
	for (int i=0; i<10; i++)
	{
		if (match2array(result,database[i],7))
			idx = i;
	}
	if(mini_rects[0].height/mini_rects[0].width > 3)
			idx = 1;
	return idx;	
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


