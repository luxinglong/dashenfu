#pragma once
#include "opencv2/opencv.hpp"
#include "TargetRecognition.hpp"

#include <vector>
#include <utility>
#include <string>

class ShenfuDetector{
public:
	struct Point2fWithIdx{
		cv::Point2f p;
		size_t idx;
		Point2fWithIdx(const cv::Point2f _p, size_t _idx):p(_p), idx(_idx){}
	};

	typedef enum {Shenfu_ORB, Shenfu_GRAD, Shenfu_CANNY, Shenfu_CNN, Shenfu_DIGIT} Methed_Type;

public:
	ShenfuDetector(int cell_width = 127, int cell_height = 71, int code_width = 48, int code_height = 24, bool perspective = false, Methed_Type m_type = Shenfu_CANNY, bool is_init = false, int hit_bit = 0):recognition(){
		sudoku_width = cell_width;
		sudoku_height = cell_height;
		led_width = code_width;
		led_height = code_height;
		use_perspective = perspective;
		type = m_type;
	}

	void loadModel(){
		recognition.loadSVMModel("../model/SVM_DATA.xml");
	}

	const cv::RotatedRect & getRect(int idx) const{
		return sudoku_rects[idx];
	}

	/**
	* @brief getTarget
	* @param image
	* @return return.first is the index of unstructured sudoku_rects vector
	*		  return.second is the index of ordered sukoku cell
	*/
	std::pair<int, int> getTarget(const cv::Mat & image);

protected:
	std::pair<int, int> chooseTargetPerspective();
	std::pair<int, int> chooseTarget(const cv::Mat & image, const std::vector<cv::RotatedRect> & sudoku_rects);

	int findTargetORB(cv::Mat * cells);
	int findTargetEdge(cv::Mat * cells);
	int findTargetCanny(cv::Mat * cells);
	int findTargetCNN(cv::Mat * cells);
	int findTargetDigit(cv::Mat * sudoku_cells, cv::Mat & image);

	bool init();
	
	int updateHitBit();
	bool matchArray(const int * array1, const int * array2, int length);
	bool getSudokuResults(cv::Mat * cells);
	bool getLedResults(cv::Mat * led_cells);	

	bool checkSudoku(const std::vector<std::vector<cv::Point2i>> & contours, std::vector<cv::RotatedRect> & sudoku_rects);

	bool getLedRects(cv::Mat & frame);

	cv::RotatedRect adjustRRect(const cv::RotatedRect & rect);


private:
	std::vector<cv::RotatedRect> sudoku_rects;
	std::vector<cv::Rect> ledd_rects;
	cv::Point2f pt[2];
	int last_sudoku_results[9];
	int last_led_results[5];
	int cur_sudoku_results[9];
	int cur_led_results[5];
	int hit_bit;
	bool is_init;
	int sudoku_width; // pixel
	int sudoku_height;
	int led_width;
	int led_height;
	bool use_perspective;
	int sudoku_real_width;
	int sudoku_real_height;
	Methed_Type type;
	TargetRecognition recognition;
	cv::Mat src;
};
