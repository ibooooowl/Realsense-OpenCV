#include<iostream>
#include<opencv2/opencv.hpp>

using namespace cv;
using namespace std;



int main()
{
    Mat image, image_gray, image_bw, image_bw2, image_bw3, image_bw4;
    image = imread("/home/user/CLionProjects/realsense-opencv/image/box.jpg");  //读取图像；
    if (image.empty())
    {
        cout << "读取错误" << endl;
        return -1;
    }

    //转换为灰度图像
    cvtColor(image, image_gray, COLOR_BGR2GRAY);
//    Mat outImg2;
//    cv::resize(image, outImg2, cv::Size(image.cols * 0.3,image.rows * 0.3), 0, 0, cv::INTER_LINEAR);
//    cv::imshow("image_gray", outImg2);

//    //水平
//    Mat Laplacian_kernel = (cv::Mat_<float>(3, 3) << -1, -1, -1,
//            2, 2, 2,
//            -1, -1, -1);
//    filter2D(image_gray, image_bw, -1, Laplacian_kernel);
//    Mat outImg3;
//    cv::resize(image_bw, outImg3, cv::Size(image_bw.cols * 0.3,image_bw.rows * 0.3), 0, 0, cv::INTER_LINEAR);
//
//    cv::imshow("image_bw", outImg3);

//    //45°
//    Mat Laplacian_kernel2 = (cv::Mat_<float>(3, 3) << 2, -1, -1,
//            -1, 2, -1,
//            -1, -1, 2);
//    filter2D(image_gray, image_bw2, -1, Laplacian_kernel2);
//    Mat outImg4;
//    cv::resize(image_bw2, outImg4, cv::Size(image_bw2.cols * 0.3,image_bw2.rows * 0.3), 0, 0, cv::INTER_LINEAR);
//
//    cv::imshow("image_bw2", outImg4);

    //垂直
    Mat Laplacian_kernel3 = (cv::Mat_<float>(3, 3) << -1, 2, -1,
            -1, 2, -1,
            -1, 2, -1);
    filter2D(image_gray, image_bw3, -1, Laplacian_kernel3);
    Mat outImg5;
    cv::resize(image_bw3, outImg5, cv::Size(image_bw3.cols * 0.2,image_bw3.rows * 0.2), 0, 0, cv::INTER_LINEAR);

    cv::imshow("image_bw3", image_bw3);

//    //-45°
//    Mat Laplacian_kernel4 = (cv::Mat_<float>(3, 3) << -1, -1, 2,
//            -1, 2, -1,
//            2, -1, -1);
//    filter2D(image_gray, image_bw4, -1, Laplacian_kernel4);
//    Mat outImg6;
//    cv::resize(image_bw4, outImg6, cv::Size(image_bw4.cols * 0.3,image_bw4.rows * 0.3), 0, 0, cv::INTER_LINEAR);
//
//    cv::imshow("image_bw4", outImg6);

    cv::waitKey(0);  //暂停，保持图像显示，等待按键结束
    return 0;
}



////
//// Created by user on 24/04/23.
////
//#include <librealsense2/rs.hpp>
//#include <opencv2/opencv.hpp>
//
//using namespace cv;
//
//double angle(cv::Point pt1, cv::Point pt2, cv::Point pt0)
//{
//    double dx1 = pt1.x - pt0.x;
//    double dy1 = pt1.y - pt0.y;
//    double dx2 = pt2.x - pt0.x;
//    double dy2 = pt2.y - pt0.y;
//    return (dx1 * dx2 + dy1 * dy2) / sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10);
//}
//
//int main()
//{
//
//    // Load image
//    Mat src = imread("/home/user/CLionProjects/realsense-opencv/image/box.jpg");
//    Mat canny;
//    Canny(src, canny, 100, 250);
//    Mat outImg2;
//    cv::resize(canny, outImg2, cv::Size(canny.cols * 0.3,canny.rows * 0.3), 0, 0, cv::INTER_LINEAR);
//
//    imshow("canny", outImg2);
//    std::vector<Vec2f> lines;
//    HoughLines(canny, lines, 1.0, CV_PI / 180, 102, 0, 0, 0,0.1); //垂直直线
//    //依次在图中绘制出每条线段
//    for (size_t i = 0; i < lines.size(); i++)
//    {
//        float rho = lines[i][0], theta = lines[i][1];
//        Point pt1, pt2;
//        double a = cos(theta), b = sin(theta);
//        double x0 = a * rho, y0 = b * rho;
//        pt1.x = cvRound(x0 + 2000 * (-b));  //把浮点数转化成整数
//        pt1.y = cvRound(y0 + 2000 * (a));
//        pt2.x = cvRound(x0 - 2000 * (-b));
//        pt2.y = cvRound(y0 - 2000 * (a));
//        line(src, pt1, pt2, Scalar(255), 1, cv::LINE_AA);
//    }
//    Mat outImg;
//    cv::resize(src, outImg, cv::Size(src.cols * 0.3,src.rows * 0.3), 0, 0, cv::INTER_LINEAR);
//    imshow("src", outImg);
//    waitKey(0);
////    Mat src1;
////    namedWindow("效果图窗口1", 1);//定义窗口
////    Canny(src, src1, 50, 200, 3);//进行一此canny边缘检测
////    std::vector<Vec4i> lines; //定义一个矢量结构lines用于存放得到的线段矢量集合
////    //HoughLines(src1, lines, 1, CV_PI / 180, 150, 0, 0);
////    HoughLinesP(src1, lines, 1, CV_PI / 180, 80, 50, 10);//进行霍夫线变换
////
////    Mat outImg;
////    cv::resize(src1, outImg, cv::Size(src1.cols * 0.3,src1.rows * 0.3), 0, 0, cv::INTER_LINEAR);
////    imshow("效果图窗口1", outImg);
////    waitKey(0);
////    // Convert image to grayscale
////    Mat gray;
////    cvtColor(img, gray, COLOR_BGR2GRAY);
////
////    // Blur image to reduce noise
////    GaussianBlur(gray, gray, Size(5, 5), 0);
////
////    // Detect edges using Canny algorithm
////    Mat edges;
////    Canny(gray, edges, 100, 200);
////
////    // Find contours in the image
////    std::vector<std::vector<Point>> contours;
////    findContours(edges, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
////
////    // Iterate through all detected contours
////    for (size_t i = 0; i < contours.size(); i++)
////    {
////        // Approximate contour with polygon
////        std::vector<Point> approx;
////        approxPolyDP(contours[i], approx, arcLength(contours[i], true) * 0.02, true);
////
////        // Check if contour is a rectangle
////        if (approx.size() == 4 && isContourConvex(approx))
////        {
////            double maxCosine = 0;
////            for (int j = 2; j < 5; j++)
////            {
////                double cosine = fabs(angle(approx[j % 4], approx[j - 2], approx[j - 1]));
////                maxCosine = std::max(maxCosine, cosine);
////            }
////            if (maxCosine < 0.3)
////            {
////                // Draw rectangle on image
////                drawContours(img, std::vector<std::vector<Point>>{approx}, 0, Scalar(0, 0, 255), 2);
////            }
////        }
////    }
////
////    // Display result
////    cv::namedWindow( "Result", cv::WINDOW_AUTOSIZE );
////    Mat outImg;
////    cv::resize(img, outImg, cv::Size(img.cols * 0.2,img.rows * 0.2), 0, 0, cv::INTER_LINEAR);
////    imshow("Result", outImg);
////    waitKey(0);
////    return 0;
//}
//
//
