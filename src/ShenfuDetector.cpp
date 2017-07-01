#include "ShenfuDetector.hpp"
#include "TargetRecognition.hpp"

#include <iostream>
#include <sstream>

#ifndef SHOW_IMAGE
#define SHOW_IMAGE
#endif

using namespace cv;
using namespace std;

pair<int, int> ShenfuDetector::getTarget(const cv::Mat & image){
	cvtColor(image, src, CV_BGR2GRAY);
	Mat binary;
	threshold(src, binary, 170, 255, THRESH_BINARY);
#ifdef SHOW_IMAGE
	//imshow("binary", binary);
#endif
	Mat temp;
	binary.copyTo(temp);
	Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
	dilate(temp, temp, element);
	erode(temp, temp, element);
	medianBlur(temp, temp, 3);

	imshow("temp",temp);

	vector<vector<Point2i>> contours;
	vector<Vec4i> hierarchy;
	findContours(temp, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
#ifdef SHOW_IMAGE
	Mat show;
	image.copyTo(show);
	for(int i=0; i<contours.size(); ++i){
		drawContours(show, contours, i, CV_RGB(rand()%255, rand()%255, rand()%255), 3, CV_FILLED);
	}
	//imshow("contours", show);
#endif
	sudoku_rects.clear();
	Mat sudoku_show;
	image.copyTo(sudoku_show);
	if (checkSudoku(contours, sudoku_rects)){ // find the sudoku

		pair<int, int> idx = chooseTarget(src, sudoku_rects);

		for (int i=0; i<sudoku_rects.size(); i++){
			Point2f vertices[4];
			sudoku_rects[i].points(vertices);
			for (int j = 0; j < 4; j++)
				line(sudoku_show, vertices[j], vertices[(j+1)%4], Scalar(0,255,0),2);
		}
		//getLedRects(sudoku_show);
		imshow("sudoku",sudoku_show);

		cout << "target1: " << idx.second << endl;
		cout << "target2: " << idx.first << endl;

		return idx;
	}
	return make_pair(-1, -1);		
}

bool ShenfuDetector::checkSudoku(const vector<vector<Point2i>> & contours, vector<RotatedRect> & sudoku_rects){
	if (contours.size() < 9)
			return false;
	
	float width = sudoku_width;
	float height = sudoku_height;
	float ratio = 28.0 / 16.0;
	int sudoku = 0;

    float low_threshold = 0.6;
    float high_threshold = 1.2;
    vector<Point2f> centers;
	for (size_t i = 0; i < contours.size(); i++) {
		RotatedRect rect = minAreaRect(contours[i]);
		rect = adjustRRect(rect);
		const Size2f & s = rect.size;
		float ratio_cur = s.width / s.height;

		if (ratio_cur > 0.8 * ratio && ratio_cur < 1.2 * ratio &&
			s.width > low_threshold * width && s.width < high_threshold * width &&
			s.height > low_threshold * height && s.height < high_threshold * height &&
			((rect.angle > -10 && rect.angle < 10) || rect.angle < -170 || rect.angle > 170)){

			sudoku_rects.push_back(rect);
            centers.push_back(rect.center);
            //vector<Point2i> poly;
            //approxPolyDP(contours[i], poly, 20, true);
            ++sudoku;
		}
	}

    cout << "sudoku num: " << sudoku << endl;

    if (sudoku > 15)
        return false;

    if(sudoku > 9){
        float dist_map[15][15] = {0};
        // calculate distance of each cell center
        for(int i = 0; i < sudoku; ++i){
            for (int j = i+1; j < sudoku; ++j){
                float d = sqrt((centers[i].x - centers[j].x)*(centers[i].x - centers[j].x) + (centers[i].y - centers[j].y)*(centers[i].y - centers[j].y));
                dist_map[i][j] = d;
                dist_map[j][i] = d;
            }
        }

        // choose the minimun distance cell as center cell
        int center_idx = 0;
        float min_dist = 100000000;
        for(int i = 0; i < sudoku; ++i){
            float cur_d = 0;
            for (int j = 0; j < sudoku; ++j){
                cur_d += dist_map[i][j];
            }
            if(cur_d < min_dist){
                min_dist = cur_d;
                center_idx = i;
            }
        }

        // sort distance between each cell and the center cell
        vector<pair<float, int> > dist_center;
        for (int i = 0; i < sudoku; ++i){
            dist_center.push_back(make_pair(dist_map[center_idx][i], i));
        }
        std::sort(dist_center.begin(), dist_center.end(), [](const pair<float, int> & p1, const pair<float, int> & p2) { return p1.first < p2.first; });

        // choose the nearest 9 cell as suduku
        vector<RotatedRect> sudoku_rects_temp;
        for(int i = 0; i < 9; ++i){
            sudoku_rects_temp.push_back(sudoku_rects[dist_center[i].second]);
        }
        sudoku_rects_temp.swap(sudoku_rects);
    }
    cout << "sudoku n: " << sudoku_rects.size()  << endl;
	return sudoku_rects.size() == 9;
}

pair<int, int> ShenfuDetector::chooseTarget(const Mat & image,const vector<RotatedRect> & sudoku_rects){
	vector<Point2fWithIdx> centers;
	for(size_t i=0; i<sudoku_rects.size(); i++){
		const RotatedRect & rect = sudoku_rects[i];
		centers.push_back(Point2fWithIdx(rect.center, i));
	}

	sort(centers.begin(), centers.end(), [](const Point2fWithIdx & p1, const Point2fWithIdx & p2){return p1.p.y < p2.p.y;});
	sort(centers.begin()+0, centers.begin()+3, [](const Point2fWithIdx & p1, const Point2fWithIdx & p2){return p1.p.x < p2.p.x;});
	sort(centers.begin()+3, centers.begin()+6, [](const Point2fWithIdx & p1, const Point2fWithIdx & p2){return p1.p.x < p2.p.x;});
	sort(centers.begin()+6, centers.begin()+9, [](const Point2fWithIdx & p1, const Point2fWithIdx & p2){return p1.p.x < p2.p.x;});

	Mat cell[9];

	for(size_t i=0; i<3; i++){
		for(size_t j=0; j<3; j++){
			size_t idx = i * 3 + j;
			Rect cell_roi = sudoku_rects[centers[idx].idx].boundingRect();
			int margin_x = 0.05*cell_roi.width;
			int margin_y = 0.1*cell_roi.height;
			Rect scale_roi = Rect(cell_roi.x + margin_x, cell_roi.y + margin_y, cell_roi.width - 2*margin_x, cell_roi.height - 2*margin_y);
			if(scale_roi.x<0 || scale_roi.y<0 || scale_roi.width<0 || scale_roi.height<0 || scale_roi.x + scale_roi.width > image.cols || scale_roi.y + scale_roi.height > image.rows)
				return make_pair(-1, -1);
			image(scale_roi).copyTo(cell[idx]);
		}
	}
	
	Point2f vertices1[4];
	sudoku_rects[centers[0].idx].points(vertices1);
	pt[0] = vertices1[1];
	Point2f vertices2[4];
	sudoku_rects[centers[2].idx].points(vertices2);
	pt[1] = vertices2[2];

	int width, height;
	for (int i=0; i<9; i++){
		width += sudoku_rects[centers[i].idx].size.width;
		height += sudoku_rects[centers[i].idx].size.height;
	}
	sudoku_real_width = (int)width/9;
	sudoku_real_height = (int)height/9;

	int idx = -1;
	Mat led_image;
	image.copyTo(led_image);
	//idx = findTargetCanny(cell);
	idx =findTargetDigit(cell, led_image);

#ifdef SHOW_IMAGE	
	for(int i=0; i<9; i++){
		// preprocessing the input image	cv::RotatedRect adjustRRect(const cv::RotatedRect & rect);
		Mat digitNorm;
		if(image.rows != 28 && image.cols != 28)
			resize(cell[i], digitNorm, Size(28, 28));
		else
			cell[i].copyTo(digitNorm);
	
		// TODO: InvMat shoud be cut to reduce the computation cost

		threshold(digitNorm,digitNorm,160,255,THRESH_BINARY);

		//Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
		//erode(digitNorm, digitNorm, element);
		//dilate(digitNorm, digitNorm, element);

		stringstream stream;
		string str;
		stream << cur_sudoku_results[i];
		stream >> str;
		putText(cell[i],str,Point(10,20),CV_FONT_HERSHEY_COMPLEX, 1,Scalar(0, 0, 0));
		stringstream stream2;
		string str2;
		stream2 << i;
		stream2 >> str2;
		namedWindow(str2,CV_WINDOW_NORMAL);		
		imshow(str2, cell[i]);
	}
#endif

	return idx < 0 ? make_pair(-1, -1) : make_pair((int)centers[idx].idx, idx);
}

int ShenfuDetector::findTargetCanny(cv::Mat * cells){
	int min_count_idx = -1;
	int w_3 = cells[0].cols / 2.8;
	int w_23 = cells[0].cols*2/3.0;
	double mid_ratio = 0.0;

	for(size_t i=0; i<9; i++){
		int mid_area_count = 0;
		int black_count = 0;
		Mat edge;
		Canny(cells[i], edge, 20, 50);
		uchar * ptr = (uchar *) edge.data;

		for(size_t j=0; j<cells[i].rows; ++j){
			for(size_t k=0; k<cells[i].cols; ++k, ++ptr){
				int v = *ptr;
				if(v == 255)
					++black_count;
				
				if(k >= w_3 && k <= w_23)
					++mid_area_count;
			}
		}

		double cur_ratio = (double)mid_area_count/black_count;
		if(mid_ratio < cur_ratio){
			mid_ratio = cur_ratio;
			min_count_idx = i;
		}
	}
	
	return min_count_idx;	
}

int ShenfuDetector::findTargetDigit(cv::Mat * sudoku_cells, cv::Mat & image){
	
	getSudokuResults(sudoku_cells);
	getLedRects(image);

	if(!is_init){
		init();
		return -1;
	}

	int recognition_idx = -1;
	updateHitBit();

	for(size_t i=0; i<9; i++){
		if(cur_sudoku_results[i] == cur_led_results[hit_bit])
			return i;
	}

	return recognition_idx;
}

bool ShenfuDetector::init(){
	if (is_init)
		return true;

	for (size_t i=0; i<9; i++){
		last_sudoku_results[i] = cur_sudoku_results[i];
	}

	for (size_t i=0; i<5; i++){
		last_led_results[i] = cur_led_results[i];
	}

	is_init = true;
	return is_init;
}

bool ShenfuDetector::getSudokuResults(cv::Mat * cells){
	
	for(size_t i=0; i<9; i++){
		cur_sudoku_results[i] = recognition.recognizeHWDigit(cells[i]);
	}

	return true;	
}

bool ShenfuDetector::getLedResults(cv::Mat * cells){
	for(size_t i=0; i<5; i++){
		cur_led_results[i] = recognition.recognizeLEDDigit(cells[i]);
	}

	return true;
}

int ShenfuDetector::updateHitBit(){
	bool sudoku_flag = matchArray(last_sudoku_results, cur_sudoku_results, 9); // sudoku not change
	bool led_flag = matchArray(last_led_results, cur_led_results, 5); // led not change

	for(int i=0; i<9; ++i){
		last_sudoku_results[i] = cur_sudoku_results[i];
	}
	for(int i=0; i<5; ++i){
		last_led_results[i] = cur_led_results[i];
	}

	if(!sudoku_flag && led_flag){  // hit success
		++hit_bit;
		cout << "Hit success!" << endl;
		return 10;
	}else if(!sudoku_flag && !led_flag){  // hit fail
		hit_bit = 0;
		cout << "Hit fail!" << endl;
		return 20;	
	}else if(sudoku_flag && led_flag){  // not finish hit
		return 30;
	}

	return 40;
}

bool ShenfuDetector::matchArray(const int * array1, const int * array2, int length){
	int counter = 0;
	for(size_t i=0; i<length; i++)
		if(array1[i] != array2[i])
			counter++;

	if(counter >= 0.5*length)
		return false;  // change
	else
		return true;   // not change
}

bool ShenfuDetector::getLedRects(cv::Mat & frame){
	Point2f pt1(pt[0].x + 0.85*sudoku_real_width, pt[0].y - 1.25*sudoku_real_height);
	Point2f	pt2(pt[1].x - 0.8*sudoku_real_width, pt[1].y - 1.25*sudoku_real_height);
	Point2f pt3(pt[1].x - 0.8*sudoku_real_width, pt[1].y - 0.45*sudoku_real_height);
	Point2f pt4(pt[0].x + 0.85*sudoku_real_width, pt[0].y - 0.45*sudoku_real_height);

	float singleShuamguanW=(pt2.x-pt1.x)/5;
	std::vector<cv::Rect> led_rects;

#ifdef SHOW_IMAGE	
	// draw the big rectangle cover all shumaguan numbers
	/*line(frame,pt1,pt2,Scalar(0,255,0),2);
	line(frame,pt2,pt3,Scalar(0,255,0),2);
	line(frame,pt3,pt4,Scalar(0,255,0),2);
	line(frame,pt4,pt1,Scalar(0,255,0),2);*/
#endif
	
	if (pt2.y-pt1.y<5 && pt2.y-pt1.y>-5){
		led_rects.push_back(Rect(pt1,Point2f(pt4.x+singleShuamguanW,pt4.y)));//(Rect(pt1.x,pt1.y,singleShuamguanW,pt4.y-pt1.y));
		led_rects.push_back(Rect(Point2f(pt1.x+singleShuamguanW,pt1.y), Point2f(pt4.x+2*singleShuamguanW,pt4.y)));//(Rect(pt1.x+singleShuamguanW,pt1.y,singleShuamguanW,pt4.y-pt1.y));
		led_rects.push_back(Rect(Point2f(pt1.x+2*singleShuamguanW,pt1.y), Point2f(pt4.x+3*singleShuamguanW,pt4.y)));//(Rect(pt1.x+2*singleShuamguanW,pt1.y,singleShuamguanW,pt4.y-pt1.y));
		led_rects.push_back(Rect(Point2f(pt1.x+3*singleShuamguanW,pt1.y), Point2f(pt4.x+4*singleShuamguanW,pt4.y)));//(Rect(pt1.x+3*singleShuamguanW,pt1.y,singleShuamguanW,pt4.y-pt1.y));
		led_rects.push_back(Rect(Point2f(pt1.x+4*singleShuamguanW,pt1.y), pt3));//(Rect(pt1.x+4*singleShuamguanW,pt1.y,singleShuamguanW,pt4.y-pt1.y));
#ifdef SHOW_IMAGE				
		/*line(frame,Point2f(pt1.x+singleShuamguanW,pt1.y),Point2f(pt4.x+singleShuamguanW,pt4.y),Scalar(0,0,255),2);
		line(frame,Point2f(pt1.x+2*singleShuamguanW,pt1.y),Point2f(pt4.x+2*singleShuamguanW,pt4.y),Scalar(0,0,255),2);
		line(frame,Point2f(pt1.x+3*singleShuamguanW,pt1.y),Point2f(pt4.x+3*singleShuamguanW,pt4.y),Scalar(0,0,255),2);
		line(frame,Point2f(pt1.x+4*singleShuamguanW,pt1.y),Point2f(pt4.x+4*singleShuamguanW,pt4.y),Scalar(0,0,255),2);*/
#endif
	}
	else{
		led_rects.push_back(Rect(pt1, Point2f(pt4.x+singleShuamguanW,((pt3.y-pt4.y)/(pt3.x-pt4.x))*(pt4.x+singleShuamguanW-pt3.x)+pt3.y)));
		led_rects.push_back(Rect(Point2f(pt1.x+singleShuamguanW,((pt2.y-pt1.y)/(pt2.x-pt1.x))*(pt1.x+singleShuamguanW-pt2.x)+pt2.y), Point2f(pt4.x+2*singleShuamguanW,((pt3.y-pt4.y)/(pt3.x-pt4.x))*(pt4.x+2*singleShuamguanW-pt3.x)+pt3.y)));
		led_rects.push_back(Rect(Point2f(pt1.x+2*singleShuamguanW,((pt2.y-pt1.y)/(pt2.x-pt1.x))*(pt1.x+2*singleShuamguanW-pt2.x)+pt2.y), Point2f(pt4.x+3*singleShuamguanW,((pt3.y-pt4.y)/(pt3.x-pt4.x))*(pt4.x+3*singleShuamguanW-pt3.x)+pt3.y)));
		led_rects.push_back(Rect(Point2f(pt1.x+3*singleShuamguanW,((pt2.y-pt1.y)/(pt2.x-pt1.x))*(pt1.x+3*singleShuamguanW-pt2.x)+pt2.y), Point2f(pt4.x+4*singleShuamguanW,((pt3.y-pt4.y)/(pt3.x-pt4.x))*(pt4.x+4*singleShuamguanW-pt3.x)+pt3.y)));
		led_rects.push_back(Rect(Point2f(pt1.x+4*singleShuamguanW,((pt2.y-pt1.y)/(pt2.x-pt1.x))*(pt1.x+4*singleShuamguanW-pt2.x)+pt2.y), pt3));

		/*led_rects.push_back(Rect(pt1.x,pt1.y,singleShuamguanW,pt4.y-pt1.y));
		led_rects.push_back(Rect(pt1.x+singleShuamguanW,(int)((pt2.y-pt1.y)/(pt2.x-pt1.x))*(pt1.x+singleShuamguanW-pt2.x)+pt2.y,singleShuamguanW,pt4.y-pt1.y));
		led_rects.push_back(Rect(pt1.x+2*singleShuamguanW,(int)((pt2.y-pt1.y)/(pt2.x-pt1.x))*(pt1.x+2*singleShuamguanW-pt2.x)+pt2.y,singleShuamguanW,pt4.y-pt1.y));
		led_rects.push_back(Rect(pt1.x+3*singleShuamguanW,(int)((pt2.y-pt1.y)/(pt2.x-pt1.x))*(pt1.x+3*singleShuamguanW-pt2.x)+pt2.y,singleShuamguanW,pt4.y-pt1.y));
		led_rects.push_back(Rect(pt1.x+4*singleShuamguanW,(int)((pt2.y-pt1.y)/(pt2.x-pt1.x))*(pt1.x+4*singleShuamguanW-pt2.x)+pt2.y,singleShuamguanW,pt4.y-pt1.y));*/
#ifdef SHOW_IMAGE
		/*line(frame,Point2f(pt1.x+singleShuamguanW,((pt2.y-pt1.y)/(pt2.x-pt1.x))*(pt1.x+singleShuamguanW-pt2.x)+pt2.y),Point2f(pt4.x+singleShuamguanW,((pt3.y-pt4.y)/(pt3.x-pt4.x))*(pt4.x+singleShuamguanW-pt3.x)+pt3.y),Scalar(0,0,255),2);
		line(frame,Point2f(pt1.x+2*singleShuamguanW,((pt2.y-pt1.y)/(pt2.x-pt1.x))*(pt1.x+2*singleShuamguanW-pt2.x)+pt2.y),Point2f(pt4.x+2*singleShuamguanW,((pt3.y-pt4.y)/(pt3.x-pt4.x))*(pt4.x+2*singleShuamguanW-pt3.x)+pt3.y),Scalar(0,0,255),2);
		line(frame,Point2f(pt1.x+3*singleShuamguanW,((pt2.y-pt1.y)/(pt2.x-pt1.x))*(pt1.x+3*singleShuamguanW-pt2.x)+pt2.y),Point2f(pt4.x+3*singleShuamguanW,((pt3.y-pt4.y)/(pt3.x-pt4.x))*(pt4.x+3*singleShuamguanW-pt3.x)+pt3.y),Scalar(0,0,255),2);
		line(frame,Point2f(pt1.x+4*singleShuamguanW,((pt2.y-pt1.y)/(pt2.x-pt1.x))*(pt1.x+4*singleShuamguanW-pt2.x)+pt2.y),Point2f(pt4.x+4*singleShuamguanW,((pt3.y-pt4.y)/(pt3.x-pt4.x))*(pt4.x+4*singleShuamguanW-pt3.x)+pt3.y),Scalar(0,0,255),2);*/
#endif
	}

	sort(led_rects.begin(), led_rects.begin()+5, [](const Rect & r1, const Rect & r2){return r1.x < r2.x;});		
	Mat led_cells[5];

	for(size_t i=0; i<5; i++){
		if(led_rects[i].x<0 || led_rects[i].y<0 || led_rects[i].width<0 || led_rects[i].height<0 || led_rects[i].x + led_rects[i].width > frame.cols || led_rects[i].y + led_rects[i].height > frame.rows)
				return false;
		frame(led_rects[i]).copyTo(led_cells[i]);
	}

	getLedResults(led_cells);
	
#ifdef SHOW_IMAGE	
	for(int i=0; i<5; i++){

		rectangle(frame, led_rects[i], Scalar(0,0,255));

		if(!led_cells[i].data){
			cout << "The handwritten digit image isn't valid." << endl;
			return -1;
		}

		Mat gray;
		if(led_cells[i].channels() != 1)
			cvtColor(led_cells[i], gray, CV_RGB2GRAY);
		else
			led_cells[i].copyTo(gray);	

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
		
		stringstream stream3;
		string str3;
		stream3 << cur_led_results[i];
		stream3 >> str3;
		putText(led_img,str3,Point(10,20),CV_FONT_HERSHEY_COMPLEX, 1,Scalar(255, 255, 255));

		stringstream stream4;
		string str4;
		stream4 << i+10;
		stream4 >> str4;
		namedWindow(str4,CV_WINDOW_NORMAL);		
		imshow(str4, led_img);
	}
#endif
	
}

RotatedRect ShenfuDetector::adjustRRect(const RotatedRect & rect){
	const Size2f & s = rect.size;
	if (s.width > s.height)
		return rect;
	return RotatedRect(rect.center, Size2f(s.height, s.width), rect.angle + 90.0);
}
