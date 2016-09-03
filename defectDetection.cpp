#include <iostream>
#include <fstream>
#include <string>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

vector<vector<Point> > getContours(Mat image);
void displayResult(Mat image, vector<vector<Point>> contours, String outputImageName, double field_width);


int main(int argc, char* argv[])
{
	// load the image
	string inputImageName, outputImageName;
	double field_width;
	if (argc != 4)
	{
		cout << "Usage: DefectDetection.exe <src_image_path> <dst_image_path> <field width in mm>" << endl;
		return -1;
	} else {
		inputImageName = argv[1]; //"..\\data\\real\\1.bmp";
		outputImageName = argv[2]; //"..\\result\\1.bmp";
		field_width = atoi(argv[3]);
	}

	Mat image = imread(inputImageName, CV_LOAD_IMAGE_GRAYSCALE);

	// equalization
	Mat temp;
	equalizeHist(image, temp);

	// gaussian filtering
	GaussianBlur(temp, temp, Size(9,9), 5);

	// contour detection
	vector<vector<Point> > contours = getContours(temp);

	// display result
	//String outputImageName = inputImageName.substr(0, inputImageName.rfind(".")) + ".bmp";
	displayResult(image, contours, outputImageName, field_width);

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
	int g_thresh = 80;
	threshold(image, image, g_thresh, 255, CV_THRESH_BINARY);
	bitwise_xor(image, Scalar(255), image);
	findContours(image, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	return contours;
}

void displayResult(Mat image, vector<vector<Point>> contours, String outputImageName, double field_width)
{
	// find the largest defec
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

	Point2f center;
	float radius;
	minEnclosingCircle(contours[max_index], center, radius);
	Mat result;
	cvtColor(image, result, CV_GRAY2BGR);
	circle(result, cvPointFrom32f(center), cvRound(radius), CV_RGB(255, 0, 0));
	
	ostringstream ss;
	ss << (radius / image.cols*field_width);
	putText(result, string("Defect Radius = ") + string(ss.str()) + "mm", Point(center.x + 2 * radius, center.y), CV_FONT_HERSHEY_COMPLEX, 1, CV_RGB(255, 0, 0));
	cvWaitKey();
	
	imwrite(outputImageName, result);

}
