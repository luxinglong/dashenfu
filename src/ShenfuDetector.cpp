#include "ShenfuDetector.hpp"

#include <iostream>

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
	imshow("binary", binary);
#endif
	vector<vector<Point2i>> contours;
	vector<Vec4i> hierarchy;
	findContours(binary, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
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
	if (checkSudoku(contours, sudoku_rects)){
		for (int i=0; i<sudoku_rects.size(); i++){
			Point2f vertices[4];
			sudoku_rects[i].points(vertices);
			for (int j = 0; j < 4; j++)
				line(sudoku_show, vertices[j], vertices[(j+1)%4], Scalar(0,255,0),2);
		}
		imshow("sudoku",sudoku_show);
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
    float high_threshold = 1.4;
    vector<Point2f> centers;
	for (size_t i = 0; i < contours.size(); i++) {
		RotatedRect rect = minAreaRect(contours[i]);
		//rect = adjustRRect(rect);
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
