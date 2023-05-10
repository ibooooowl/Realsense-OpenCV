//
// Created by user on 05/05/23.
//
/**
 * @file houghcircles.cpp
 * @brief This program demonstrates circle finding with the Hough transform
 * https://docs.opencv.org/3.4/d4/d70/tutorial_hough_circle.html
 */
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
bool onLine(Point P_i, Point P_j, Point Q){
    if((Q.x- P_i.x)*(P_j.y-P_i.y) == (P_j.x- P_i.x)*(Q.y-P_i.y) && std::min(P_i.x, P_j.x) <= Q.x && Q.x <= std::max(P_i.x, P_j.x) && std::min(P_i.y, P_j.y) <= Q.y && Q.y <= std::max(P_i.y, P_j.y)){
        return true;
    }
    return false;
}

bool onRectangle(Point x_min_y_min, Point x_max_y_max, Point Q){
    // point Q does not on four lines of a rectangle, then return false.
    if(!onLine(x_min_y_min, Point(x_min_y_min.x, x_max_y_max.y), Q) && !onLine(x_min_y_min, Point(x_max_y_max.x, x_min_y_min.y), Q) && !onLine(x_max_y_max, Point(x_min_y_min.x, x_max_y_max.y), Q) && !onLine(x_max_y_max, Point(x_max_y_max.x, x_min_y_min.y), Q)){
        return false;
    }
    return true;
}

Point2f rotate2d(const Point2f& inPoint, const double& angRad)
{
    Point2f outPoint;
    //CW rotation
    outPoint.x = std::cos(angRad)*inPoint.x - std::sin(angRad)*inPoint.y;
    outPoint.y = std::sin(angRad)*inPoint.x + std::cos(angRad)*inPoint.y;
    return outPoint;
}

Point2f rotatePoint(const Point2f& inPoint, const Point2f& center, const double& angRad)
{
    return rotate2d(inPoint - center, angRad) + center;
}

int main(int argc, char** argv)
{
    //![load]
    const char* filename = argc >=2 ? argv[1] : "/home/user/CLionProjects/realsense-opencv/image/box3.jpg";

    // Loads an ima
    Mat src = imread( samples::findFile( filename ), IMREAD_COLOR );

    // Check if image is loaded fine
    if(src.empty()){
        printf(" Error opening image\n");
        printf(" Program Arguments: [image_name -- default %s] \n", filename);
        return EXIT_FAILURE;
    }
    //![load]

    double angle = 45;

    // get the center coordinates of the image to create the 2D rotation matrix
    Point2f center_img((src.cols - 1) / 2.0, (src.rows - 1) / 2.0);
    // using getRotationMatrix2D() to get the rotation matrix
    Mat rotation_matix = getRotationMatrix2D(center_img, angle, 1.0);

    // we will save the resulting image in rotated_image matrix
    Mat rotated_image;
    // rotate the image using warpAffine
    warpAffine(src, rotated_image, rotation_matix, src.size());

    //![convert_to_gray]
    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    //![convert_to_gray]

    //![reduce_noise]
    medianBlur(gray, gray, 5);
    //![reduce_noise]

    //![houghcircles]
    vector<Vec3f> circles;
    HoughCircles(gray, circles, HOUGH_GRADIENT, 1,
                 gray.rows/64,  // change this value to detect circles with different distances to each other
                 100, 30, 0, 30 // change the last two parameters
            // (min_radius & max_radius) to detect larger circles
    );
    //![houghcircles]

    //![ Find the four circles with the largest and smallest x and y coordinates]
    int x_min = int(circles[0][0]);
    int x_max = int(circles[0][0]);
    int y_min = int(circles[0][1]);
    int y_max = int(circles[0][1]);


    //![draw the detected circles]
    std::map<int,int> radius_map;
    for(const auto & i : circles) {
        Vec3i c = i;
        if(radius_map.count(c[2])<1){
            radius_map.emplace(c[2], 1);
        }else{
            radius_map[c[2]] = radius_map[c[2]]+1;
        }
    }
    std::vector<std::pair<int, int>> radius_list(radius_map.begin(), radius_map.end());
    // sort the vector by the non-decreasing order of frequency of radius
    std::sort(radius_list.begin(), radius_list.end(), [] (const std::pair<char,int>& a, const std::pair<char,int>& b)->bool{ return a.second > b.second; });

    for(const auto & i : circles)
    {
        Vec3i c = i;
        Point center = Point(c[0], c[1]);
        // circle center
        circle( src, center, 1, Scalar(0,100,100), 3, LINE_AA);
        // circle outline
        int radius = c[2];
        circle( src, center, radius, Scalar(0,0,255), 3, LINE_AA);
        // if radius is one of the three top radius then, record it.
        if(radius == radius_list[0].first || radius == radius_list[1].first || radius == radius_list[2].first) {
            // update x_min, x_max
            if (x_min > c[0]) {
                x_min = c[0];
            }
            if (x_max < c[0]) {
                x_max = c[0];
            }
            // update y_min, y_max
            if (y_min > c[1]) {
                y_min = c[1];
            }
            if (y_max < c[1]) {
                y_max = c[1];
            }

        }
        if(radius == radius_list[0].first) {
            circle(src, center, 50, Scalar(255, 255, 0), 3, LINE_AA);
        }
        if(radius == radius_list[1].first) {
            circle(src, center, 45, Scalar(0, 255, 255), 3, LINE_AA);
        }
        if(radius == radius_list[2].first) {
            circle(src, center, 40, Scalar(255, 0, 255), 3, LINE_AA);
        }
//        if(radius == radius_list[3].first) {
//            circle(src, center, 35, Scalar(255, 255, 255), 3, LINE_AA);
//        }
        //std::cout << circle << std::endl;
    }

//    circle( src, Point(int(x_min), int(y_min)), 3, Scalar(255,0,255), 3, LINE_AA);
//    circle( src, Point(int(x_max), int(y_min)), 3, Scalar(255,0,255), 3, LINE_AA);
//    circle( src, Point(int(x_min), int(y_max)), 3, Scalar(255,0,255), 3, LINE_AA);
//    circle( src, Point(int(x_max), int(y_max)), 3, Scalar(255,0,255), 3, LINE_AA);
    rectangle(src, Point(x_min, y_min), Point(x_max, y_max), Scalar(255,0,0), 3);
    //![draw]

    int counter= 0;
    for(const auto & i : circles) {
        Vec3i c = i;
        Point center = Point(c[0], c[1]);
        if(onRectangle(Point(x_min, y_min), Point(x_max, y_max), center)){
            circle( src, center, 60, Scalar(0,255,0), 3, LINE_AA);
            counter++;
        }
    }
    std::cout << counter << std::endl;

    circle(src, Point(int(circles[0][0]), int(circles[0][1])) , 70, Scalar(255, 255, 255), 6, LINE_AA);
    //![display]

    // Resizes the image to half its size
    Size newSize(src.cols/5, src.rows/5);
    resize(src, src, newSize);
    imshow("detected circles", src);


    //![ Find the four circles with the largest and smallest x and y coordinates]
    Point2f first = rotatePoint(Point2f(circles[0][0], circles[0][1]), center_img, 45);
    int x_min_p = int(first.x);
    int x_max_p = int(first.x);
    int y_min_p = int(first.y);
    int y_max_p = int(first.y);
    circle(rotated_image, first, 70, Scalar(255, 255, 255), 6, LINE_AA);
    for(const auto & i : circles)
    {
        Point2f center(i[0], i[1]);
        Point2f center_p = rotatePoint(center, center_img, 45);
        // circle center
        circle( rotated_image, center_p, 1, Scalar(0,100,100), 3, LINE_AA);
        // circle outline
        int radius = int(i[2]);
        circle( rotated_image, center_p, radius, Scalar(0,0,255), 3, LINE_AA);
        // if radius is one of the three top radius then, record it.
        if(radius == radius_list[0].first || radius == radius_list[1].first || radius == radius_list[2].first) {
            // update x_min, x_max
            if (x_min_p > int(center_p.x)) {
                x_min_p = int(center_p.x);
            }
            if (x_max_p < int(center_p.x)) {
                x_max_p = int(center_p.x);
            }
            // update y_min, y_max
            if (y_min_p > int(center_p.y)) {
                y_min_p= int(center_p.y);
            }
            if (y_max_p < int(center_p.y)) {
                y_max_p = int(center_p.y);
            }

        }
        if(radius == radius_list[0].first) {
            circle(rotated_image, center_p, 50, Scalar(255, 255, 0), 3, LINE_AA);
        }
        if(radius == radius_list[1].first) {
            circle(rotated_image, center_p, 45, Scalar(0, 255, 255), 3, LINE_AA);
        }
        if(radius == radius_list[2].first) {
            circle(rotated_image, center_p, 40, Scalar(255, 0, 255), 3, LINE_AA);
        }
//        if(radius == radius_list[3].first) {
//            circle(src, center, 35, Scalar(255, 255, 255), 3, LINE_AA);
//        }
        //std::cout << circle << std::endl;
    }

//    circle( src, Point(int(x_min), int(y_min)), 3, Scalar(255,0,255), 3, LINE_AA);
//    circle( src, Point(int(x_max), int(y_min)), 3, Scalar(255,0,255), 3, LINE_AA);
//    circle( src, Point(int(x_min), int(y_max)), 3, Scalar(255,0,255), 3, LINE_AA);
//    circle( src, Point(int(x_max), int(y_max)), 3, Scalar(255,0,255), 3, LINE_AA);
    rectangle(rotated_image, Point(x_min_p, y_min_p), Point(x_max_p, y_max_p), Scalar(255,0,0), 3);

    resize(rotated_image, rotated_image, newSize);

    imshow("Rotated image", rotated_image);
    //![display]
    waitKey();
    return EXIT_SUCCESS;
}