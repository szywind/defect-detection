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

void processImage(Mat& image, double g_thresh = 50, int mode = -1);
vector<vector<Point> > getContours(Mat image);
void displayResult(Mat image, vector<vector<Point>> contours, String outputImageName, double field_width, double defect_size);
bool isAllWhite(Mat image, int r);
void refineDefectSize(Mat image, Defect& defect, int y0, int x0);

float g_thresh, g_alpha;

int main(int argc, char* argv[])
{
	// load the image
	string inputImageName, outputImageName;
	double defect_size, field_width;
	if (argc != 5 && argc != 6 && argc != 7)
	{
		cout << "Usage: DefectDetection.exe <src image path> <dst image path> <field width(in mm)> <threshold of the defects to be detected(in mm)> [<tolerance>(default is 3)] [<threshold for binarilization>(default is 50)]" << endl;
		return -1;
	} else {
		inputImageName = argv[1]; //"..\\data\\real\\1.bmp";
		outputImageName = argv[2]; //"..\\result\\1.bmp";
		field_width = atof(argv[3]);
		defect_size = atof(argv[4]);
		g_alpha = (argc == 6) ? atof(argv[5]) : 3;
		g_thresh = (argc == 7) ? atof(argv[6]) : 50.0;
	}

	// 0.
	cout << "Loading and processing image "<< inputImageName <<endl;
	Mat image = imread(inputImageName, CV_LOAD_IMAGE_GRAYSCALE);
	
	Mat temp;
	image.copyTo(temp);

	//Scalar mean, stddev;
	//meanStdDev(image, mean, stddev);
	//image.convertTo(temp, CV_32FC1, 1.0 / stddev(0), -mean(0) / stddev(0));

	processImage(temp, g_thresh);
	
	vector<vector<Point> > contours = getContours(temp);

	//String outputImageName = inputImageName.substr(0, inputImageName.rfind(".")) + ".bmp";
	displayResult(image, contours, outputImageName, field_width, defect_size);

	return 0;
}

void sharpen(Mat src, Mat & dst){
	Mat kernel = (Mat_<float>(3,3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
	filter2D(src, dst, -1, kernel);
}

void processImage(Mat& image, double g_thresh, int mode)
{
	string info;

	if (image.channels() == 3)
	{
		cvtColor(image, image, CV_BGR2GRAY);
	}
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(5,5));
	double minVal, maxVal;
	minMaxLoc(image, &minVal, &maxVal);
	image.convertTo(image, CV_8UC1, 255 / (maxVal-minVal), -minVal * 255 / (maxVal-minVal));

	switch (mode){
	case 99:
#ifdef _DEBUG
		imshow("original", image);
		waitKey();
#endif
		//for(int i=0; i<1; i++)
		//morphologyEx(image, image, MORPH_CLOSE, element);//--MORPH_OPEN, --MORPH_BLACKHAT, --MORPH_TOPHAT, -MORPH_GRADIENT, MORPH_CLOSE

#ifdef _DEBUG
		imshow("morph", image);
		waitKey();
#endif
		//equalizeHist(image, image);
#ifdef _DEBUG
		imshow("equalization", image);
		waitKey();
#endif
		//GaussianBlur(image, image, Size(3, 3), 1);
#ifdef _DEBUG
		imshow("after blurred", image);
		waitKey();
#endif
		//sharpen(image, image);
#ifdef _DEBUG
		imshow("after sharpening", image);
		waitKey();
#endif

		/*
		Laplacian(image, image, -1, 3);
		imshow("after Laplace", image);
		waitKey();
		*/

		threshold(image, image, 0, 255, CV_THRESH_BINARY_INV|CV_THRESH_OTSU);
#ifdef _DEBUG
		imshow("after OTSU", image);
		waitKey();
#endif
		info = "Use OTSU binarilization";
		break;
	case 1:
		equalizeHist(image, image);
		GaussianBlur(image, image, Size(5, 5), 3);
		threshold(image, image, 0, 255, CV_THRESH_BINARY_INV|CV_THRESH_OTSU);
		info = "Use OTSU binarilization";
		break;
	case 2:
		Canny(image, image, 40, 160, 3, true);
		info = "Use Canny edge detection";
		break;
	case 3:
		adaptiveThreshold(image, image, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, 5, 5);
		info = "Use adaptive thresholding";
		break;
	case 0:
		equalizeHist(image, image);
		GaussianBlur(image, image, Size(5, 5), 3);
		threshold(image, image, g_thresh, 255, CV_THRESH_BINARY_INV);
		info = "Use fine thresholding";
		break;
	default:
		equalizeHist(image, image);
		GaussianBlur(image, image, Size(9,9), 5);
		sharpen(image, image);
		threshold(image, image, g_thresh, 255, CV_THRESH_BINARY_INV);
		info = "Use coarse thresholding";
		break;
	}
	
	//bitwise_xor(image, Scalar(255), image);
#ifdef _DEBUG
	cout<<info<<endl;
#endif
	
}

vector<vector<Point> > getContours(Mat image)
{
	vector<vector<Point> > contours;
 	findContours(image, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE); //CV_RETR_EXTERNAL
	return contours;
}

bool compDefect(const Defect& df1, const Defect& df2)
{
	return df2 < df1;
}


void blobDetection(Mat image, Defect& defect, int y0, int x0){
	/*
	// Blob detector
	vector<KeyPoint> keypoints;
	SimpleBlobDetector::Params params;
	params.filterByArea = true;
	params.minArea = 10;// 0.5*area;
	params.maxArea = 1.5*area;

	SimpleBlobDetector blobDetector(params);
	blobDetector.create("SimpleBlob");
	blobDetector.detect(image, keypoints);
	drawKeypoints(image, keypoints, image, Scalar(255, 0, 0));


	float radius = diameter/2;
	float minDist = radius;
	int minId = -1;
	for (int i = 0; i < keypoints.size(); i++){
		Point2f dist = Point2f(keypoints[i].pt.y - image.rows / 2, keypoints[i].pt.x - image.cols / 2);
		float temp = norm(dist);
		if (norm(dist) < minDist){
			minDist = norm(dist);
			minId = i;
		}
	}
	imwrite("..\\data\\result\\test_result.png", image);
	namedWindow("result", 1);
	imshow("result", image);
	waitKey();
	return (minId == -1) ? diameter : keypoints[minId].size;
	*/

	// find contours
	//resize(patch, patch, Size(1010, 1010), CV_INTER_LINEAR);
	
	processImage(image, g_thresh, 99);
	Mat temp;
	image.copyTo(temp);
	vector<vector<Point> > contours = getContours(temp);
/////////////////////////////////////////////////////////////////////////////////////////////////////
	
	float r;
	Point2f c;
	Point2f refined_center = Point2f(image.cols/2.0, image.rows/2.0);
	float refined_radius = defect.getDiameter()/2;

	/*
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
	*/
	float minDist = refined_radius;// /2;
	int minId = -1;

	float maxRadius = refined_radius;
	
	for (int i = 0; i < contours.size(); i++){
		minEnclosingCircle(contours[i], c, r);
		Point2f dist = Point2f(c.y - image.rows / 2.0, c.x - image.cols / 2.0);
		//float temp = norm(Point(3,4));
		if (norm(dist) < minDist && r < defect.getDiameter() && r > maxRadius){
			//minDist = norm(dist);
			maxRadius = r;
			minId = i;
			refined_radius = r;
			refined_center = c;
		}
	}
		
	double refined_diameter = 2 * refined_radius;
	//cout << int(image.at<uchar>(int(center.y), int(center.x))) << endl;

	//CvRect rect = boundingRect(contours[i]);
	//CvBox2D box = minAreaRect(contours[i]);
	
	ostringstream ss;
	ss << defect.getId();


	Mat result;
	cvtColor(image, result, CV_GRAY2BGR);
	imwrite(string("..\\data\\result\\") + string(ss.str()) + string(".png"), result);
	cout<<defect.getDiameter() << " -> " << refined_diameter << endl;
	namedWindow("result", 1);;
	circle(result, cvPointFrom32f(refined_center), cvRound(refined_radius), CV_RGB(255, 0, 0));
	imshow("result", result);
	waitKey();
	defect.setDiameter(refined_diameter);
	defect.setCenter(Point2f(refined_center.x + x0, refined_center.y + y0));
}

bool isAllWhite(Mat image, int r){
	int count = 0;
	int cx = int(image.cols/2.0+0.5), cy = int(image.rows/2.0+0.5);
	for(int i=cx-r; i<=cx+r; i++){
		for(int j=cy-r; j<=cy+r; j++){
			if(i<0 || i>=image.rows || j<0 || j>=image.cols) return false;
			if(norm(Point2f(i-cx, j-cy)) >= r) continue;
			if(image.at<uchar>(i,j) == 0) {
				count ++;
				//return false;
			}
			if(count>g_alpha*r) return false; 
		}
	}
	return true;
}

void refineDefectSize(Mat image, Defect& defect, int y0, int x0){
	Mat result;
	image.copyTo(result);
	//------------------------------------------------------------------------------------------------------------------
	processImage(image, g_thresh, 99);
	float r;
	float refined_radius, radius;
	refined_radius = radius = defect.getDiameter()/2;
	for(int iter = 0; iter<31; iter++){
		r = radius*(1.0 + iter*0.1);
		if(!isAllWhite(image, r)) continue;
		refined_radius = r;
	}

	double refined_diameter = 2 * refined_radius;

#ifdef _DEBUG
	cout<<defect.getDiameter() << " -> " << refined_diameter << endl;
	
	ostringstream ss;
	ss << defect.getId();

	cvtColor(result, result, CV_GRAY2BGR);
	
	imwrite(string("..\\data\\result\\") + string(ss.str()) + string(".png"), result);
	
	namedWindow("result", 1);
	circle(result, Point2f(image.cols/2.0, image.rows/2.0), refined_radius, CV_RGB(255, 0, 0));
	imshow("result", result);
	waitKey();
#endif
	
	defect.setDiameter(refined_diameter);
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
		if (diameter * scale_factor < defect_size) continue;
		

		//cout << "diameter = " << diameter << endl;
		//float ans = bfs(image, center, diameter);
		
		defects.push_back(Defect(count, center, diameter, area));

		// refine diameter
		int row = int(center.y+0.5), col = int(center.x+0.5);
		if(row - 50 >= 0 && row + 51 <= image.rows && col - 50 >= 0 && col + 51 <= image.cols){
			Mat patch(image, Range(row - 50, row + 51), Range(col - 50, col + 51));
			//blobDetection(patch, defects.back(), row, col);
			refineDefectSize(patch, defects.back(), row, col);
			diameter = defects.back().getDiameter();
		}
		count++;

		circle(result, cvPointFrom32f(center), 0.5 * diameter, CV_RGB(255, 0, 0));
	}
	cout << count << " defects are detected" << endl;
	
	sort(defects.begin(), defects.end(), compDefect);

	for (int i = 0; i < defects.size(); i++)
	{
		float diameter = defects[i].getDiameter();
		Point2f center = defects[i].getCenter();

		ostringstream ss,ss1;
		//ss.precision(3);
		//ss.setf(ios::fixed);
		ss << setiosflags(ios::fixed) << setprecision(3) << (diameter * scale_factor);
		ss1 << i+1;
		bool showTextBottom = center.y < 100;
		bool showTextRight = image.cols - center.x > 300;
		int offset_vert = 2 * showTextBottom - 1;
		if (showTextRight) {
			putText(result, string(ss1.str()) + ": ", Point(center.x + diameter/2 + 5, center.y + offset_vert * diameter/2 + showTextBottom * 25), CV_FONT_HERSHEY_COMPLEX, 1, CV_RGB(0, 0, 255));
			putText(result, string(ss.str()) + "mm", Point(center.x + diameter/2 + 45, center.y + offset_vert * diameter/2 + showTextBottom * 25), CV_FONT_HERSHEY_COMPLEX, 1, CV_RGB(255, 0, 0));
		}
		else {
			putText(result, string(ss1.str()) + ": ", Point(center.x - diameter/2 - 190, center.y + offset_vert * diameter/2 + showTextBottom * 25), CV_FONT_HERSHEY_COMPLEX, 1, CV_RGB(0, 0, 255));
			putText(result, string(ss.str()) + "mm", Point(center.x - diameter/2 - 150, center.y + offset_vert * diameter/2 + showTextBottom * 25), CV_FONT_HERSHEY_COMPLEX, 1, CV_RGB(255, 0, 0));
		}
		cout << "defect " << (i+1) << ": size(diameter) is " << string(ss.str()) << "mm" << endl;
	}

	imwrite(outputImageName, result);
   	cout << "SRC:End" << endl;
 }
