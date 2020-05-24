#include<opencv2/opencv.hpp>
#include<iostream>
#include<Card.h>

using namespace cv;
using namespace std;

int RanksNum = 13;  //牌的点数
int SuitsNum = 4;		//牌的花色数

int main()
{
	vector<Train_ranks> train_ranks(RanksNum);
	vector<Train_suit> train_suits(SuitsNum);

	load_ranks(train_ranks);		//读取点数
	load_suits(train_suits);		//读取花色

	VideoCapture vc("video.mp4");
	Mat frame;
	int cnt = 0;
	while (1)
	{
		vc >> frame;
		Mat test_img = frame;
		if (frame.empty())
			return 0;
		if (cnt > 100)
		{
			resize(test_img, test_img, Size(1010, 572));
			Mat img_threshold = img_pre(test_img);
			
			vector<int> is_card;
			vector<vector<Point>> sort_contours;

			find(img_threshold, is_card, sort_contours);

			drawContours(test_img, sort_contours, -1, Scalar(0, 255, 0));
			Query_card qCard;
			vector<Query_card> card;
			if (sort_contours.size() != 0)
			{
				for (int i = 0; i < sort_contours.size(); i++)
				{
					if (i < is_card.size())
					{
						if (is_card[i] == 1)
						{
							card_pre(sort_contours[i], test_img, qCard);
							card.push_back(qCard);
							match(card[i], train_ranks, train_suits, card[i].best_rank_match,
								card[i].best_suit_match, card[i].rank_diff, card[i].suit_diff);
							test_img = drawResult(test_img, card[i]);
						}
					}
				}

				if (card.size() != 0)
				{
					vector<vector<Point>> temp_cnts;
					for (int j = 0; j < card.size(); j++)
					{
						temp_cnts.push_back(card[j].contours);
					}
					drawContours(test_img, temp_cnts, -1, Scalar(255, 0, 0), 2);

				}
			}
			namedWindow("res", WINDOW_NORMAL);
			imshow("res", test_img);
			waitKey(30);
		}cnt++;
	}
	

	return 0;
}