#include <iostream>
#include <fstream>
#include <string>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

void getContours(String path);
void getContours(Mat image);
void detectLines(Mat image);

int main(int argc, char* argv[])
{
	// initialization
	//string imageName = "..\\data\\TP-0329\\1\\Main.IMAGE_ID_GREY.bmp";
	string imageName = "..\\data\\real\\1.bmp"; //"..\\data\\test_img\\1.bmp";

	Mat img = imread(imageName, CV_LOAD_IMAGE_GRAYSCALE);
	/*
	int a = img.type();
	int ch = img.channels();
	bool flag = (a == CV_8UC1);
	imshow("input", img);
	system("pause");
	*/
	imwrite("..\\data\\1_0.bmp", img);

	// equalize
	equalizeHist(img, img);
	imwrite("..\\data\\1_1.bmp", img);

	// median filter
	GaussianBlur(img, img, Size(9,9), 5);
	//medianBlur(img, img, 3);
	//blur(img, img, Size(9, 9));
	imwrite("..\\data\\1_2.bmp", img);



	// getContour
	// C interface
	//getContours("..\\data\\1_2.bmp");
	// C++ interface
	getContours(img);

	//IplImage *src = cvLoadImage(path.c_str, CV_LOAD_IMAGE_GRAYSCALE);

	return 0;
}


void getContours(String path)
{
	IplImage *src = cvLoadImage(path.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
	CvMemStorage *storage = cvCreateMemStorage();
	CvSeq *seq = NULL;
	int g_thresh = 128;

	cvThreshold(src, src, g_thresh, 255, CV_THRESH_BINARY);
	cvSaveImage("..\\data\\1_3.bmp", src);

	int cnt = cvFindContours(src, storage, &seq);
	seq = seq->h_next;
	double length = cvArcLength(seq);
	double area = cvContourArea(seq);
	CvRect rect = cvBoundingRect(seq, 1);
	CvBox2D box = cvMinAreaRect2(seq, NULL);

	cout << "Length = " << length << endl;
	cout << "Area = " << area << endl;

	IplImage *dst = cvCreateImage(cvGetSize(src), 8, 3); cvZero(dst);
	cvDrawContours(dst, seq, CV_RGB(255, 0, 0), CV_RGB(255, 0, 0), 0);
	cvRectangleR(dst, rect, CV_RGB(0, 255, 0));
	cvShowImage("dst", dst);
	cvWaitKey();

	CvPoint2D32f center;
	float radius;
	cvMinEnclosingCircle(seq, &center, &radius);
	cvCircle(dst, cvPointFrom32f(center), cvRound(radius), CV_RGB(100, 100, 100));
	cvShowImage("dst", dst);
	cvWaitKey();

	/*
	CvBox2D ellipse = cvFitEllipse2(seq);
	cvEllipseBox(dst, ellipse, CV_RGB(255, 255, 0));
	cvShowImage("dst", dst);
	cvWaitKey();

	//绘制外接最小矩形
	CvPoint2D32f pt[4];
	cvBoxPoints(box, pt);
	for (int i = 0; i<4; ++i){
		cvLine(dst, cvPointFrom32f(pt[i]), cvPointFrom32f(pt[((i + 1) % 4) ? (i + 1) : 0]), CV_RGB(0, 0, 255));
	}
	cvShowImage("dst", dst);
	cvWaitKey();
	*/
	cvSaveImage("..\\data\\1_4.bmp", dst);
	cvReleaseImage(&src);
	cvReleaseImage(&dst);
	cvReleaseMemStorage(&storage);

}

void getContours(Mat image)
{
	if (image.channels() == 3)
	{
		cvtColor(image, image, CV_BGR2GRAY);
	}
	vector<vector<Point>> contours;
	
	int g_thresh = 80;
	// binaralize
	// threshold
	threshold(image, image, g_thresh, 255, CV_THRESH_BINARY);
	bitwise_xor(image, Scalar(255), image);
	imwrite("..\\data\\1_3_2.bmp", image);
	// canny
	//detectLines(image);

	findContours(image, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	
	int max_index = 0;
	double max_area = 0;
	for (int i = 0; i < contours.size(); i++) {
		double length = arcLength(contours[i], true);
		double area = contourArea(contours[i]);
		if (area > max_area) {
			max_area = area;
			max_index = i;
		}
	}
	double length = arcLength(contours[max_index], true);
	double area = contourArea(contours[max_index]);

	CvRect rect = boundingRect(contours[max_index]);
	CvBox2D box = minAreaRect(contours[max_index]);

	cout << "Length = " << length << endl;
	cout << "Area = " << area << endl;

	//Mat dst = Mat(image.size(), CV_8U, Scalar(0));
	//drawContours(dst, contours, -1, CV_RGB(255, 0, 0), 2);

	//rectangle(dst, rect, CV_RGB(0, 255, 0));
	//imshow("dst", dst);
	//cvWaitKey();

	Point2f center;
	float radius;
	minEnclosingCircle(contours[max_index], center, radius);
	circle(image, cvPointFrom32f(center), cvRound(radius), CV_RGB(100, 100, 100));
	//imshow("dst", dst);
	cvWaitKey();

	/*
	CvBox2D ellipse = cvFitEllipse2(seq);
	cvEllipseBox(dst, ellipse, CV_RGB(255, 255, 0));
	cvShowImage("dst", dst);
	cvWaitKey();

	//绘制外接最小矩形
	CvPoint2D32f pt[4];
	cvBoxPoints(box, pt);
	for (int i = 0; i<4; ++i){
	cvLine(dst, cvPointFrom32f(pt[i]), cvPointFrom32f(pt[((i + 1) % 4) ? (i + 1) : 0]), CV_RGB(0, 0, 255));
	}
	cvShowImage("dst", dst);
	cvWaitKey();
	*/
	imwrite("..\\data\\1_4_2.bmp", image);

}

void detectLines(Mat image)
{

	Mat dst, cdst;
	if (image.channels() == 3)
	{
		Canny(image, dst, 50, 200, 3);
	}
	else if (image.channels() == 1)
	{
		dst = image;
	}
	//Canny(image, dst, 50, 200, 3);
	cvtColor(dst, cdst, CV_GRAY2BGR);
	//cdst = dst;
	vector<Vec2f> lines;
	// detect lines
	HoughLines(dst, lines, 1, CV_PI / 180, int(0.9*dst.cols), 0, 0);
	// draw lines
	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0], theta = lines[i][1];
		//if (theta>CV_PI / 180 * 80 && theta < CV_PI / 180 * 100)
		{
			Point pt1, pt2;
			double a = cos(theta), b = sin(theta);
			double x0 = a*rho, y0 = b*rho;
			pt1.x = cvRound(x0 - 1000 * b);
			pt1.y = cvRound(y0 + 1000 * a);
			pt2.x = cvRound(x0 + 1000 * b);
			pt2.y = cvRound(y0 - 1000 * a);
			line(cdst, pt1, pt2, Scalar(0, 0, 255), 1, CV_AA);
			//line(cdst, pt1, pt2, Scalar(255), 1, CV_AA);

		}
	}
	//imshow("source", image);
	//imshow("detected lines", cdst);
	imwrite("..\\data\\111.bmp", dst);
	imwrite("..\\data\\222.bmp", cdst);
}