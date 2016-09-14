#define _USE_MATH_DEFINES

#include <iostream>
#include <fstream>
#include <string>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cmath>

using namespace cv;
using namespace std;

vector<vector<Point> > getContours(Mat image);
void displayResult(Mat image, vector<vector<Point>> contours, String outputImageName, double field_width, double defect_size);


int main(int argc, char* argv[])
{
	// load the image
	string inputImageName, outputImageName;
	double defect_size, field_width;
	if (argc != 5)
	{
		cout << "Usage: DefectDetection.exe <src image path> <dst image path> <field width(in mm)> <threshold of the defects to be detected(in mm)>" << endl;
		return -1;
	} else {
		inputImageName = argv[1]; //"..\\data\\real\\1.bmp";
		outputImageName = argv[2]; //"..\\result\\1.bmp";
		field_width = atof(argv[3]);
		defect_size = atof(argv[4]);
	}

	Mat image = imread(inputImageName, CV_LOAD_IMAGE_GRAYSCALE);

	// equalization
	Mat temp;
	equalizeHist(image, temp);

	// gaussian filtering
	GaussianBlur(temp, temp, Size(11,11), 5);

	// contour detection
	vector<vector<Point> > contours = getContours(temp);

	// display result
	//String outputImageName = inputImageName.substr(0, inputImageName.rfind(".")) + ".bmp";
	displayResult(image, contours, outputImageName, field_width, defect_size);

	return 0;
}


vector<vector<Point> > getContours(Mat image)
{
	if (image.channels() == 3)
	{
		cvtColor(image, image, CV_BGR2GRAY);
	}
	vector<vector<Point> > contours;

	// binaralize with tuned threshold
	int g_thresh = 80; // hard-code which needs tuning
	threshold(image, image, g_thresh, 255, CV_THRESH_BINARY);
	bitwise_xor(image, Scalar(255), image);
	findContours(image, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	return contours;
}

void displayResult(Mat image, vector<vector<Point>> contours, String outputImageName, double field_width, double defect_size)
{
	Mat result;
	cvtColor(image, result, CV_GRAY2BGR);
	int count = 0;

	// find all detects larger than the threshold specified by the argument defect_size
	cout << "Detecting defects larger than " << defect_size << " mm ..." << endl;
	double scale_factor = field_width / image.cols;
	for (int i = 0; i < contours.size(); i++) {
		double length = arcLength(contours[i], true);
		double area = contourArea(contours[i]);

		double eff_radius = sqrt(area / M_PI);
		if (eff_radius * scale_factor < defect_size)
		{
			continue;
		}

		Point2f center;
		float radius;
		minEnclosingCircle(contours[i], center, radius);

		CvRect rect = boundingRect(contours[i]);
		CvBox2D box = minAreaRect(contours[i]);

		cout << "Length = " << length << endl;
		cout << "Area = " << area << endl;

		circle(result, cvPointFrom32f(center), cvRound(eff_radius), CV_RGB(255, 0, 0));

		ostringstream ss;
		ss << (eff_radius * scale_factor);
		putText(result, string("Defect Radius = ") + string(ss.str()) + "mm", Point(center.x + 2 * radius, center.y), CV_FONT_HERSHEY_COMPLEX, 1, CV_RGB(255, 0, 0));
	
		count++;
	}
	
	imwrite(outputImageName, result);
	cout << count << " defects are detected" << endl;
	cout << "SRC:End";
}
