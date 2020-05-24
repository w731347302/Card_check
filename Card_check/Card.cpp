#include<opencv2/opencv.hpp>
#include<iostream>
#include<Card.h>

using namespace cv;
using namespace std;


double MAX_WIDTH = 600;
double MAX_HEIGHT = 600;


int RANK_WIDTH = 70;
int RANK_HEIGHT = 125;
int SUIT_WIDTH = 70;
int SUIT_HEIGHT = 100;

int RANK_DIFF_MAX = 3000;
int SUIT_DIFF_MAX = 700;

int CORNER_HEIGHT = 160;
int CORNER_WIDTH = 95;

int CARD_MAX_AREA = 120000;
int CARD_MIN_AREA = 25000;

bool load_ranks(vector<Train_ranks> &train_ranks)
{
	string Rank[13] = { "Ace","Two","Three","Four","Five","Six","Seven",
					"Eight","Nine","Ten","Jack","Queen","King" };

	for (int i = 0; i < 13; i++)
	{
		train_ranks[i].name = Rank[i];
		string filename = Rank[i] + ".jpg";
		train_ranks[i].img = imread(filename, 0);
	}
	return true;
}

bool load_suits(vector<Train_suit> &train_suits)
{
	string Suit[4] = { "Spades","Diamonds","Clubs","Hearts" };

	for (int i = 0; i < 4; i++)
	{
		train_suits[i].name = Suit[i];
		string filename = Suit[i] + ".jpg";
		train_suits[i].img = imread(filename, 0);
	}
	return true;
}

Mat img_pre(Mat src)
{
	Mat img_gray, img_blur;	//预处理，变成灰度图然后高斯滤波
	cvtColor(src, img_gray, COLOR_BGR2GRAY);
	GaussianBlur(img_gray, img_blur, Size(5, 5), 0);

	//二值化(阈值问题？)
	Mat img_threshold;

	threshold(img_blur, img_threshold, 100, 255, THRESH_BINARY);


	return img_threshold;
}

static inline bool ContoursSortFun(vector<cv::Point> contour1, vector<cv::Point> contour2)
{
	return (cv::contourArea(contour1) > cv::contourArea(contour2));
}

bool find(Mat img_threshold, vector<int> &is_card, vector<vector<Point>> &sort_contours)
{
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(img_threshold, contours,hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

	sort(contours.begin(), contours.end(), ContoursSortFun);

	if (contours.size() == 0)
	{
		cout << "没有检测到轮廓" << endl;
	}

	//vector<Vec4i> sort_hierarchy;
	for (int m = 0; m < contours.size(); m++)
	{
		sort_contours.push_back(contours[m]);
		//sort_hierarchy.push_back(hierarchy[0][m]);
	}

	double size;
	double peri;
	vector<vector<Point>> approx(sort_contours.size());
	for (int i = 0; i < sort_contours.size(); i++)
	{
		size = contourArea(sort_contours[i]);
		peri = arcLength(sort_contours[i], true);
		approxPolyDP(sort_contours[i], approx[i], (0.01*peri), true);

		if (/*(size < CARD_MAX_AREA)*//* and (size > CARD_MIN_AREA)
			 and */approx[i].size() == 4)
		{
			is_card.push_back(1);
		}

	}
	return true;
}


Mat flattener(Mat src, vector<Point> pts, double w, double h)
{
	int a, b, c, d;
	if (w <= (0.71*h))
	{
		//垂直下
		a = 1; b = 0; c = 3; d = 2;
	}
	if (w >= (1.29*h))
	{
		a = 0; b = 2; c = 1; d = 3;
	}
	if (w > (0.71*h) and w < (1.29*h))
	{
		a = 0; b = 3; c = 2; d = 1;
	}
	Point2f points[] = { Point2f(pts[a]),Point2f(pts[b]),Point2f(pts[c]) ,Point2f(pts[d]) };
	Point2f dst[] = { Point2f(0,0),Point2f(MAX_WIDTH - 1,0),
		Point2f(MAX_WIDTH - 1,MAX_HEIGHT - 1),
		Point2f(0,MAX_HEIGHT - 1) };
	Mat warp;


	Mat M = getPerspectiveTransform(points, dst);
	warpPerspective(src, warp, M, Size(MAX_WIDTH, MAX_HEIGHT));
	cvtColor(warp, warp, COLOR_BGR2GRAY);

	return warp;
}


bool card_pre(vector<Point>contour, Mat src, Query_card &qCard)
{
	qCard.contours = contour;

	double peri;
	vector<Point> approx(contour.size());
	vector<Point> pts;

	peri = arcLength(contour, true);
	approxPolyDP(contour, approx, 0.01*peri, true);
	pts = approx;
	qCard.corner_pts = pts;

	Rect rect;		//高和宽
	rect = boundingRect(contour);
	qCard.height = rect.height;
	qCard.width = rect.width;

	Moments M;		//中心点
	M = moments(contour);
	double cent_x = M.m10 / M.m00;
	double cent_y = M.m01 / M.m00;
	qCard.center = Point2f(cent_x, cent_y);

	qCard.warp = flattener(src, pts, rect.width, rect.height);

	Rect warp_rect(0, 0, CORNER_WIDTH, CORNER_HEIGHT);
	Mat Qcorner = qCard.warp(warp_rect);
	Mat Qcorner_zoom;
	resize(Qcorner, Qcorner_zoom, Size(0, 0), 4, 4);

	Mat query_thresh;
	threshold(Qcorner_zoom, query_thresh, 100, 255, THRESH_BINARY_INV);

	Mat Qrank, Qsuit;
	Rect Qrank_rect(0, 0, 380, 400);
	Rect Qsuit_rect(0, 400, 380, 240);

	Qrank = query_thresh(Qrank_rect);
	Qsuit = query_thresh(Qsuit_rect);

	vector<vector<Point>> Qrank_conts;
	findContours(Qrank, Qrank_conts, RETR_TREE, CHAIN_APPROX_SIMPLE);
	sort(Qrank_conts.begin(), Qrank_conts.end(), ContoursSortFun);

	if (Qrank_conts.size() != 0)
	{
		Rect Qrank_conts_rect = boundingRect(Qrank_conts[0]);
		Mat Qrank_roi = Qrank(Qrank_conts_rect);
		resize(Qrank_roi, Qrank_roi, Size(RANK_WIDTH, RANK_HEIGHT));
		qCard.rank_img = Qrank_roi;
	}

	vector<vector<Point>> Qsuit_conts;
	findContours(Qsuit, Qsuit_conts, RETR_TREE, CHAIN_APPROX_SIMPLE);
	sort(Qsuit_conts.begin(), Qsuit_conts.end(), ContoursSortFun);

	if (Qsuit_conts.size() != 0)
	{
		Rect Qsuit_conts_rect = boundingRect(Qsuit_conts[0]);
		Mat Qsuit_roi = Qsuit(Qsuit_conts_rect);
		resize(Qsuit_roi, Qsuit_roi, Size(SUIT_WIDTH, SUIT_HEIGHT));
		qCard.suit_img = Qsuit_roi;
	}

	return true;
}

bool match(Query_card qCard, vector<Train_ranks> train_ranks, vector<Train_suit> train_suits,
	       string &best_rank_match_name, string &best_suit_match_name, int &best_rank_match,
			int &best_suit_match)
{
	best_rank_match = 10000;
	best_suit_match = 10000;
	best_rank_match_name = "unknown";
	best_suit_match_name = "unknwon";
	string best_rank_name;
	Mat best_rank_diff_img;
	string best_suit_name;
	Mat best_suit_diff_img;

	if ((qCard.rank_img).size != 0 and (qCard.suit_img).size != 0)
	{
		for (int i = 0; i < train_ranks.size(); i++)
		{
			Mat diff_img;
			if (!qCard.rank_img.empty())
			{
				absdiff(qCard.rank_img, train_ranks[i].img, diff_img);
				int rank_diff = (sum(diff_img) / 255)[0];

				if (rank_diff < best_rank_match)
				{
					best_rank_diff_img = diff_img;
					best_rank_match = rank_diff;
					best_rank_name = train_ranks[i].name;
				}
			}
		}
		for (int j = 0; j < train_suits.size(); j++)
		{
			Mat diff_suit_img;
			if (!qCard.suit_img.empty())
			{
				absdiff(qCard.suit_img, train_suits[j].img, diff_suit_img);
				int suit_diff = (sum(diff_suit_img) / 255)[0];

				if (suit_diff < best_suit_match)
				{
					best_suit_diff_img = diff_suit_img;
					best_suit_match = suit_diff;
					best_suit_name = train_suits[j].name;
				}
			}
		}
	}
	if (best_rank_match < RANK_DIFF_MAX)
	{
		best_rank_match_name = best_rank_name;
	}
	if (best_suit_match < SUIT_DIFF_MAX)
	{
		best_suit_match_name = best_suit_name;
	}

	return true;
}

Mat drawResult(Mat img, Query_card qCard)
{
	double x = qCard.center.x;
	double y = qCard.center.y;
	circle(img, qCard.center, 5, Scalar(255, 0, 0), -1);

	string rank_name = qCard.best_rank_match;
	string suit_name = qCard.best_suit_match;

	putText(img, rank_name, Point2f(x - 60, y - 10), FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 0), 3, 8);
	putText(img, rank_name, Point2f(x - 60, y - 10), FONT_HERSHEY_COMPLEX, 1, Scalar(50, 200, 200), 2, 8);

	putText(img, suit_name, Point2f(x - 60, y + 25), FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 0), 3, 8);
	putText(img, suit_name, Point2f(x - 60, y + 25), FONT_HERSHEY_COMPLEX, 1, Scalar(50, 200, 200), 2, 8);

	return img;
}