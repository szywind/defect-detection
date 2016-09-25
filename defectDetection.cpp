#define _USE_MATH_DEFINES

#include <iostream>
#include <fstream>
#include <string>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cmath>
#include "Defect.h"
#include <queue>

using namespace cv;
using namespace std;

vector<vector<Point> > getContours(Mat image, double g_thread = 50);
void displayResult(Mat image, vector<vector<Point>> contours, String outputImageName, double field_width, double defect_size);
float bfs(Mat image, Point2f center);

float g_thresh;

int main(int argc, char* argv[])
{
	// load the image
	string inputImageName, outputImageName;
	double defect_size, field_width;
	if (argc != 5 && argc != 6)
	{
		cout << "Usage: DefectDetection.exe <src image path> <dst image path> <field width(in mm)> <threshold of the defects to be detected(in mm)> [<threshold for binarilization>(default is 50)]" << endl;
		return -1;
	} else {
		inputImageName = argv[1]; //"..\\data\\real\\1.bmp";
		outputImageName = argv[2]; //"..\\result\\1.bmp";
		field_width = atof(argv[3]);
		defect_size = atof(argv[4]);
		g_thresh = (argc == 6) ? atof(argv[5]) : 50;
	}

	Mat image = imread(inputImageName, CV_LOAD_IMAGE_GRAYSCALE);

	// equalization
	Mat temp;
	
	//equalizeHist(image, temp);
	double minVal, maxVal;
	minMaxLoc(image, &minVal, &maxVal);
	image.convertTo(temp, CV_8UC1, 255 / (maxVal-minVal), -minVal * 255 / (maxVal-minVal));
	

	/*
	for (int i = 0; i < temp.rows; i++){
		for (int j = 0; j < temp.cols; j++){
			temp.at<uchar>(i, j) = int(image.at<uchar>(i,j) - minVal) / int(maxVal - minVal);
		}
	}
	*/
	
	
	
	//Scalar mean, stddev;
	//meanStdDev(image, mean, stddev);
	//image.convertTo(temp, CV_32FC1, 1.0 / stddev(0), -mean(0) / stddev(0));

	//imwrite(outputImageName, temp);
	
	// gaussian filtering
	GaussianBlur(temp, temp, Size(11,11), 5);

	// contour detection
	vector<vector<Point> > contours = getContours(temp, g_thresh);

	// display result
	//String outputImageName = inputImageName.substr(0, inputImageName.rfind(".")) + ".bmp";
	displayResult(image, contours, outputImageName, field_width, defect_size);

	return 0;
}


vector<vector<Point> > getContours(Mat image, double g_thresh)
{
	if (image.channels() == 3)
	{
		cvtColor(image, image, CV_BGR2GRAY);
	}
	vector<vector<Point> > contours;

	// binaralize with tuned threshold
	threshold(image, image, g_thresh, 255, CV_THRESH_BINARY_INV);
	//imwrite(name, image);

	//bitwise_xor(image, Scalar(255), image);
	findContours(image, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	return contours;
}

bool compDefect(const Defect& df1, const Defect& df2)
{
	return df2 < df1;
}

bool isOnContour(Mat image, int row, int col) {
	Mat patch(image, Range(row-2,row+3), Range(col-2,col+3));
	Scalar mean, std;
	meanStdDev(patch, mean, std);
	if (mean(0) < 0.8*image.at<uchar>(row, col) && std(0) > 0.1){
		return false;
	}
	else {
		return true;
	}
}

float bfs(Mat image, Point2f center, float diameter){
	queue<Point2f> myqueue;
	myqueue.push(center);
	map<vector<int>, bool> visited;
	float ans = 0;
	while (!myqueue.empty()){
		int n = myqueue.size();
		for (int i = 0; i < n; i++){
			Point2f pt = myqueue.front();
			myqueue.pop();
			int r = round(pt.y), c = round(pt.x);
			if (visited.find(vector<int>{c, r-1}) == visited.end() && isOnContour(image, r-1, c)){
				myqueue.push(Point2f(c, r-1));
				visited[vector<int>{c, r-1}] = true;
				float temp = (c - center.x)*(c - center.x) + (r - 1 - center.y) * (r - 1 - center.y);
				if ( temp > 0.25*ans*ans){
					ans = 2 * sqrt(temp);
				}
			}
			if (visited.find(vector<int>{c, r + 1}) == visited.end() && isOnContour(image, r + 1, c)){
				myqueue.push(Point2f(c, r + 1));
				visited[vector<int>{c, r + 1}] = true;
				float temp = (c - center.x)*(c - center.x) + (r + 1 - center.y) * (r + 1 - center.y);
				if (temp > 0.25*ans*ans){
					ans = 2 * sqrt(temp);
				}
			}
			if (visited.find(vector<int>{c-1, r}) == visited.end() && isOnContour(image, r, c-1)){
				myqueue.push(Point2f(c-1, r));
				visited[vector<int>{c-1, r}] = true;
				float temp = (c-1 - center.x)*(c-1 - center.x) + (r - center.y) * (r - center.y);
				if (temp > 0.25*ans*ans){
					ans = 2 * sqrt(temp);
				}
			}
			if (visited.find(vector<int>{c + 1, r}) == visited.end() && isOnContour(image, r, c + 1)){
				myqueue.push(Point2f(c + 1, r));
				visited[vector<int>{c + 1, r}] = true;
				float temp = (c + 1 - center.x)*(c + 1 - center.x) + (r - center.y) * (r - center.y);
				if (temp > 0.25*ans*ans){
					ans = 2 * sqrt(temp);
				}
			}
		}
	}
	return ans;
}


void displayResult(Mat image, vector<vector<Point>> contours, String outputImageName, double field_width, double defect_size)
{
	Mat result;
	cvtColor(image, result, CV_GRAY2BGR);
	int count = 0;

	// find all detects larger than the threshold specified by the argument defect_size
	cout << "Detecting defects larger than " << defect_size << " mm ..." << endl;
	double scale_factor = field_width / image.cols;
	vector<Defect> defects;
	for (int i = 0; i < int(contours.size()); i++) {
		double length = arcLength(contours[i], true);
		double area = contourArea(contours[i]);

		Point2f center;
		float radius;
		minEnclosingCircle(contours[i], center, radius);

		//float ans = bfs(image, center);
		float diameter = 2 * radius;
		if (diameter * scale_factor < defect_size)
		{
			
			continue;
		}

		//cout << "diameter = " << diameter << endl;
		//float ans = bfs(image, center, diameter);



		/*
		int row = round(center.y), col = round(center.x);
		Mat patch(image, Range(row - 50, row + 51), Range(col - 50, col + 51));
		imwrite(" ..\\data\\real\\tmp.bmp", patch);
		
		resize(patch, patch, Size(1010, 1010), CV_INTER_LINEAR);
		Mat temp;
		equalizeHist(patch, temp);
		GaussianBlur(temp, temp, Size(11, 11), 3);
		vector<vector<Point> > contours = getContours(temp, g_thresh);
		
		// find the largest defect
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
		Point2f c;
		float refined_radius;
		minEnclosingCircle(contours[max_index], c, refined_radius);
		
		double refined_diameter = 2 * refined_radius * 0.1;

		//cout << int(image.at<uchar>(int(center.y), int(center.x))) << endl;
		count++;

		CvRect rect = boundingRect(contours[i]);
		CvBox2D box = minAreaRect(contours[i]);

		//cout << "Length = " << length << endl;
		//cout << "Area = " << area << endl;
		*/
		count++;

		circle(result, cvPointFrom32f(center), cvRound(0.5 * diameter), CV_RGB(255, 0, 0));

		defects.push_back(Defect(center, diameter));
	}
	cout << count << " defects are detected" << endl;
	
	sort(defects.begin(), defects.end(), compDefect);

	for (int i = 0; i < defects.size(); i++)
	{
		float diameter = defects[i].getDiameter();
		Point2f center = defects[i].getCenter();

		ostringstream ss;
		//ss.precision(3);
		//ss.setf(ios::fixed);
		ss << setiosflags(ios::fixed) << setprecision(3) << (diameter * scale_factor);

		bool showTextBottom = center.y < 100;
		bool showTextRight = image.cols - center.x > 300;
		int offset_vert = 2 * showTextBottom - 1;
		if (showTextRight) {
			putText(result, to_string(i + 1) + ": ", Point(center.x + diameter + 5, center.y + offset_vert * diameter + showTextBottom * 25), CV_FONT_HERSHEY_COMPLEX, 1, CV_RGB(0, 0, 255));
			putText(result, string(ss.str()) + "mm", Point(center.x + diameter + 45, center.y + offset_vert * diameter + showTextBottom * 25), CV_FONT_HERSHEY_COMPLEX, 1, CV_RGB(255, 0, 0));
		}
		else {
			putText(result, to_string(i + 1) + ": ", Point(center.x - diameter - 190, center.y + offset_vert * diameter + showTextBottom * 25), CV_FONT_HERSHEY_COMPLEX, 1, CV_RGB(0, 0, 255));
			putText(result, string(ss.str()) + "mm", Point(center.x - diameter - 150, center.y + offset_vert * diameter + showTextBottom * 25), CV_FONT_HERSHEY_COMPLEX, 1, CV_RGB(255, 0, 0));
		}
		cout << "defect " << (i+1) << ": size(diameter) is " << string(ss.str()) << "mm" << endl;
	}

	imwrite(outputImageName, result);
	cout << "SRC:End" << endl;
}
