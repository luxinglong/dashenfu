#pragma once
#include "opencv2/opencv.hpp"

#include <vector>
#include <utility>

class ShenfuDetector{
public:
	struct Point2fWithIdx{
		cv::Point2f p;
		size_t idx;
		Point2fWithIdx(const cv::Point2f _p, size_t _idx):p(_p), idx(_idx){}
	};

	typedef enum {Shenfu_ORB, Shenfu_GRAD, Shenfu_CANNY, Shenfu_CNN, Shenfu_DIGIT} Methed_Type;

public:
	ShenfuDetector(int cell_width = 127, int cell_height = 71, int code_width = 48, int code_height = 24, bool perspective = false, Methed_Type m_type = Shenfu_CANNY){
		sudoku_width = cell_width;
		sudoku_height = cell_height;
		password_width = code_width;
		password_height = code_height;
		use_perspective = perspective;
		type = m_type;
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

	int findTargetORB(cv::Mat * cells);
	int findTargetEdge(cv::Mat * cells);
	int findTargetCanny(cv::Mat * cells);
	int findTargetCNN(cv::Mat * cells);
	int findTargetDigit(cv::Mat * cells);

	bool checkSudoku(const std::vector<std::vector<cv::Point2i>> & contours, std::vector<cv::RotatedRect> & sudoku_rects);


private:
	std::vector<cv::RotatedRect> sudoku_rects;
	std::vector<cv::RotatedRect> password_rects;
	int sudoku_width; // pixel
	int sudoku_height;
	int password_width;
	int password_height;
	bool use_perspective;
	Methed_Type type;
	cv::Mat src;
};
