#ifndef CARD_H_
#define CARD_H_

#include<opencv2/opencv.hpp>
#include<iostream>

using namespace cv;
using namespace std;

class Query_card {
public:
	vector<Point> contours;
	double width = 0;
	double height = 0;
	vector<Point> corner_pts;
	Point2f center;
	Mat warp;
	Mat rank_img;
	Mat suit_img;
	string best_rank_match = "Unknown";
	string best_suit_match = "Unknown";
	int rank_diff = 0;
	int suit_diff = 0;
};

class Train_ranks {
public:
	Mat img;
	string name= "unknown";
};

class Train_suit {
public:
	Mat img;
	string name = "Placeholder";
};



bool load_ranks(vector<Train_ranks> &train_ranks);
bool load_suits(vector<Train_suit> &train_suits);
Mat img_pre(Mat src);
bool find(Mat img_threshold, vector<int> &is_card, vector<vector<Point>> &sort_contours);
bool card_pre(vector<Point>contour, Mat src, Query_card &qCard);
bool match(Query_card qCard, vector<Train_ranks> train_ranks, vector<Train_suit> train_suits,
	string &best_rank_match_name, string &best_suit_match_name, int &best_rank_match,
	int &best_suit_match);
Mat drawResult(Mat img, Query_card qCard);

#endif