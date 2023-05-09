//
// Created by user on 25/04/23.
//
#include<iostream>
#include<opencv2/opencv.hpp>

using namespace cv;
using namespace std;


int main() {
//    Mat img = imread("/home/user/CLionProjects/realsense-opencv/image/box.jpg", IMREAD_GRAYSCALE);
////step1:边缘检测，得到边缘二值图
//    GaussianBlur(img, img, Size(3, 3), 0.5);
//    Mat binaryImg;
//    Canny(img, binaryImg, 50, 200);
////step2:边缘的轮廓
//    vector<vector<Point>> contours;
//    findContours(binaryImg, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
////step3:对每一个轮廓进行拟合
////    for (int i = 0; i < contours.size(); i++)
////    {
////        Rect rect = boundingRect(contours[i]);
////        if (rect.area() > 500)
////        {
////            rectangle(img, rect, Scalar(255));
////        }
////    }
//    Mat outImg2;
//    resize(img, outImg2, cv::Size(img.cols * 0.3,img.rows * 0.3), 0, 0, cv::INTER_LINEAR);
//    imshow("image_gray", outImg2);
//    waitKey(0);
    Mat src, src_gray;
    Mat edges;
    int lowThreshold = 100;
    int const max_lowThreshold = 200;
    int ratio = 6;
    int kernel_size = 3;

    src = imread("/home/user/CLionProjects/realsense-opencv/image/box.jpg");  //读取图像；
// Convert the image to grayscale
    cvtColor(src, src_gray, COLOR_BGR2GRAY);

// Apply Gaussian blur to reduce noise
    GaussianBlur(src_gray, src_gray, Size(3, 3), 0, 0, BORDER_DEFAULT);

// Apply Canny edge detection algorithm
    Canny(src_gray, edges, lowThreshold, max_lowThreshold, kernel_size);

// Find contours of the edges
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(edges, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

// Draw the contours on the original image
    Mat drawing = Mat::zeros(edges.size(), CV_8UC3);
    for (size_t i = 0; i < contours.size(); i++) {
        Scalar color = Scalar(255, 255, 255);
        drawContours(drawing, contours, (int) i, color, 2, LINE_8, hierarchy, 0);
    }

//    for (size_t i = 0; i < squares.size(); i++) {
//        // Draw the red lines
//        const Point* p = &squares[i][0];
//        int n = (int)squares[i].size();
//        polylines(image, &p, &n, 1, Scalar(0, 0, 255), 3, LINE_AA);
//    }


// Show the result
    Mat outImg2;
    resize(drawing, outImg2, cv::Size(drawing.cols * 0.3,drawing.rows * 0.3), 0, 0, cv::INTER_LINEAR);
    imshow("image_gray", outImg2);
    // imshow("Contours", drawing);
    waitKey(0);
}
