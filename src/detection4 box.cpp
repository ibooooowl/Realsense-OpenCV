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
// detect whether a point q is located on the seg line (p_i, p_j).
bool onLine(Point P_i, Point P_j, Point Q){
    if((Q.x- P_i.x)*(P_j.y-P_i.y) == (P_j.x- P_i.x)*(Q.y-P_i.y) && std::min(P_i.x, P_j.x) <= Q.x && Q.x <= std::max(P_i.x, P_j.x) && std::min(P_i.y, P_j.y) <= Q.y && Q.y <= std::max(P_i.y, P_j.y)){
        return true;
    }
    return false;
}


// calculate the distance of a point q to the seg line (p_i, p_j).
double PointToSegDist(Point P_i, Point P_j, Point Q)
{
    double x = Q.x;
    double y = Q.y;
    double x1 = P_i.x;
    double y1 = P_i.y;
    double x2 = P_j.x;
    double y2 = P_j.y;

    double cross = (x2 - x1) * (x - x1) + (y2 - y1) * (y - y1);
    if (cross <= 0) return std::sqrt((x - x1) * (x - x1) + (y - y1) * (y - y1));

    double d2 = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1);
    if (cross >= d2) return  std::sqrt((x - x2) * (x - x2) + (y - y2) * (y - y2));

    double r = cross / d2;
    double px = x1 + (x2 - x1) * r;
    double py = y1 + (y2 - y1) * r;
    return  std::sqrt((x - px) * (x - px) + (py - y) * (py - y));
}
// detect whether a point q is located exactly on the edge of the rectangle (x_min-y_min, x_max-y_max)
bool onRectangle(Point x_min_y_min, Point x_max_y_max, Point Q){
    // point Q does not on four lines of a rectangle, then return false.
    if(!onLine(x_min_y_min, Point(x_min_y_min.x, x_max_y_max.y), Q) && !onLine(x_min_y_min, Point(x_max_y_max.x, x_min_y_min.y), Q) && !onLine(x_max_y_max, Point(x_min_y_min.x, x_max_y_max.y), Q) && !onLine(x_max_y_max, Point(x_max_y_max.x, x_min_y_min.y), Q)){
        return false;
    }
    return true;
}

// detect whether a point q is located around the edge of the rectangle (x_min-y_min, x_max-y_max) with a distance of distance_threshold.
// the first bit 0 or 1 indicates the whether point q is saied on line a.
// the second bit 0 or 1 indicates the whether point q is saied on line b.
// the third bit 0 or 1 indicates the whether point q is saied on line c.
// the fourth bit 0 or 1 indicates the whether point q is saied on line d.
// for example 0001 (integer value is 1) means that point q is on line a.
int onRectangleW(Point x_min_y_min, Point x_max_y_max, Point Q, double distance_threshold){
    // point Q does not on four lines of a rectangle, then return false.
    // (x_min, y_min)--(x_min, y_max)
    bool not_find_a = PointToSegDist(x_min_y_min, Point(x_min_y_min.x, x_max_y_max.y), Q)> distance_threshold;
    // (x_min, y_min)--(x_max, y_min)
    bool not_find_b = PointToSegDist(x_min_y_min, Point(x_max_y_max.x, x_min_y_min.y), Q)>distance_threshold;
    // (x_max, y_max)--(x_min, y_max)
    bool not_find_c = PointToSegDist(x_max_y_max, Point(x_min_y_min.x, x_max_y_max.y), Q)>distance_threshold;
    // (x_max, y_max)--(x_max, y_min)
    bool not_find_d = PointToSegDist(x_max_y_max, Point(x_max_y_max.x, x_min_y_min.y), Q)>distance_threshold;
    return (not_find_a?0:1) + (not_find_b?0:2) +  (not_find_c?0:4) +(not_find_d?0:8) ;
}

// suppose that rotated matrix is 2x3
// get the rotated point of the point inPoint, by the rotated matrix
Point rotateBy(const Point2d& inPoint, const Mat& rotated_matrix)
{
    Point outPoint;
    vector<double> point{inPoint.x, inPoint.y, 1};
    Mat my_m = Mat(point);
    Mat res = rotated_matrix* my_m;
    outPoint.x= int(res.at<double>(0, 0));
    outPoint.y= int(res.at<double>(0, 1));
    return outPoint;
}



int main(int argc, char** argv)
{
    //![load]
    const char* filename = argc >=2 ? argv[1] : "/home/user/CLionProjects/realsense-opencv/image/box3.jpg";

    // Loads an image
    Mat src = imread( samples::findFile( filename ), IMREAD_COLOR );

    // Check if image is loaded fine
    if(src.empty()){
        printf(" Error opening image\n");
        printf(" Program Arguments: [image_name -- default %s] \n", filename);
        return EXIT_FAILURE;
    }
    //![load]



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



    // get the radius of the detected circles and their occurrences.
    std::map<int,int> radius_map;
    for(auto && i : circles) {
        int radius = int(i[2]);
        if(radius_map.count(radius)<1){
            radius_map.emplace(radius, 1);
        }else{
            radius_map[radius] = radius_map[radius]+1;
        }
    }
    // transform the map into the corresponding vector to sort the radius of circles by their occurrences.
    std::vector<std::pair<int, int>> radius_list(radius_map.begin(), radius_map.end());
    // sort the vector by the non-decreasing order of occurrences of radius
    std::sort(radius_list.begin(), radius_list.end(), [] (const std::pair<int,int>& a, const std::pair<int,int>& b)->bool{ return a.second > b.second; });

    // threshold to adjust the performance of proposed algorithm.
    int threshold_radius = 10;
    double dist_threshold=20;

    // select the radius of hough circles by their occurrences not less than a given threshold.
    std::set<int> select_radius;

    for(auto&&ele: radius_list){
        if(ele.second>threshold_radius){
            select_radius.emplace(ele.first);
        }
    }

    // get the center coordinates of the image to create the 2D rotation matrix
    Point2d center_img((src.cols - 1) / 2.0, (src.rows - 1) / 2.0);


    // the map to record the captured hough circles by created rectangle at different rotated angle.
    std::map<int, int> angle_count_map;

    // we rotate 360 degree, each time 0.5 degree
    for(int angle=0; angle< 360; angle++){
        // using getRotationMatrix2D() to get the rotation matrix
        Mat rotation_matrix = getRotationMatrix2D(center_img, angle/2.0, 1.0);
        // get the first hough circles.
        Point center_first = Point(int(circles[0][0]),int(circles[0][1]));
        // get the corresponding rotated point of the first hough circles.
        Point center_p_first = rotateBy(center_first, rotation_matrix);
        // initialize the value of x_min, y_min, x_max, y_max.
        int x_min = center_p_first.x;
        int x_max = center_p_first.x;
        int y_min = center_p_first.y;
        int y_max = center_p_first.y;
        // iterate all detected hough circles, get their rotated one by the given rotated angle,
        // then update the x_min, y-Min, x_max, y_max
        for(const auto & i : circles)
        {
            // get the radius of the corresponding center point.
            int radius = int(i[2]);
            // if the radius is in the selected radius set, then update x_min, y_min, x_max, y_max.
            // otherwise, these hough circles could be the noise.
            if((select_radius.count(radius)>0)) {
                // get the center point of detected hough circle.
                Point center = Point(int(i[0]), int(i[1]));
                // get the rotated center point.
                Point center_p = rotateBy(center, rotation_matrix);
                // update x_min, x_max
                if (x_min > center_p.x) {
                    x_min = center_p.x;
                }
                if (x_max < center_p.x) {
                    x_max = center_p.x;
                }
                // update y_min, y_max
                if (y_min > center_p.y) {
                    y_min = center_p.y;
                }
                if (y_max < center_p.y) {
                    y_max = center_p.y;
                }
            }
        }

        int counter= 0;
        // iterate the hough circles, count how many rotated hough circles can be captured by the constructed rectangle.
        for(const auto & i : circles) {
            Vec3i c = i;
            Point center = Point(c[0], c[1]);
            Point center_p = rotateBy(center, rotation_matrix);
            if(select_radius.count(c[2])>0 && onRectangleW(Point(x_min, y_min), Point(x_max, y_max), center_p, dist_threshold)>0){
                counter++;
            }
        }
        // update the map of rotated angle and their captured hough circles.
        angle_count_map.emplace(angle, counter);
    }

    // transform the involved map into the vector in order to sort it.
    std::vector<std::pair<int, int>> angle_count_list(angle_count_map.begin(), angle_count_map.end());
    // sort the vector non-decreasing by the number of captured hough circles.
    std::sort(angle_count_list.begin(), angle_count_list.end(), [] (const std::pair<int,int>& a, const std::pair<int,int>& b)->bool{ return a.second > b.second; });

    // display the rotated angle, and the number of captured hough circles.
    for (auto&& ele: angle_count_list) {
        std::cout << ele.first/2.0 << "->" <<ele.second << std::endl;
    }
    // the optimal rotated angle is the one has the largest number of captured hough circles by the constructed rectangle.
    int angle_opt = angle_count_list[0].first;

    // we then rotate the image by the associate angle to get the correct view of image.

    // using getRotationMatrix2D() to get the rotation matrix
    Mat rotation_matrix_opt = getRotationMatrix2D(center_img, angle_opt/2.0, 1.0);
    // we will save the resulting image in rotated_image matrix
    Mat rotated_image_opt;
    // rotate the image using warpAffine
    warpAffine(src, rotated_image_opt, rotation_matrix_opt, src.size());



    // draw the detected hough circles in rotated image and also the constructed rectangle.
    // initialize the value of x_min_p, y_min_p, x_max_p, y_max_p.
    Point first = rotateBy(Point(int(circles[0][0]), int(circles[0][1])), rotation_matrix_opt);
    int x_min_p = int(first.x);
    int x_max_p = int(first.x);
    int y_min_p = int(first.y);
    int y_max_p = int(first.y);
    // circle(rotated_image, first, 70, Scalar(255, 255, 255), 6, LINE_AA);
    // iterate all the detected hough circles.
    for(const auto & i : circles)
    {
        // get the center point of hough circle.
        Point center = Point(int(i[0]), int(i[1]));
        // get the rotated center point in the rotated image.
        Point center_p = rotateBy(center, rotation_matrix_opt);
        // circle outline
        int radius = int(i[2]);
        // draw the detected hough circles in the rotated image.
        circle( rotated_image_opt, center_p, radius, Scalar(0,0,255), 3, LINE_AA);
        // if radius is in the selected radius set, then draw it and update the x_min_p,y_min_p,x_max_p,y_max_p.
        if((select_radius.count(radius)>0)) {
            // update x_min, x_max
            if (x_min_p > int(center_p.x)) {
                x_min_p = int(center_p.x);
            }
            if (x_max_p < int(center_p.x)) {
                x_max_p = int(center_p.x);
            }
            // update y_min, y_max
            if (y_min_p > int(center_p.y)) {
                y_min_p = int(center_p.y);
            }
            if (y_max_p < int(center_p.y)) {
                y_max_p = int(center_p.y);
            }
            circle(rotated_image_opt, center_p, 45, Scalar(255, 255, 255), 5, LINE_AA);
        }
    }
    // draw the corresponding rectangle.
    rectangle(rotated_image_opt, Point(x_min_p, y_min_p), Point(x_max_p, y_max_p), Scalar(255,0,0), 3);

    // count up the hough circles can be captured by each edge of the rectangle.
    int counter_p= 0;
    std::vector<Point> edge_a;
    std::vector<Point> edge_b;
    std::vector<Point> edge_c;
    std::vector<Point> edge_d;
    // iterate the detected hough circles.
    for(const auto & i : circles) {
        // get the center point of the hough circle.
        Point center = Point(int(i[0]), int(i[1]));
        // get the rotated center point.
        Point center_p = rotateBy(center, rotation_matrix_opt);
        // detect whether capture by the constructed rectangle.
        int code = onRectangleW(Point(x_min_p, y_min_p), Point(x_max_p, y_max_p), center_p, dist_threshold);
        //if(select_radius.count(int(c[2]))>0 && code >0){
        // extract the code by the capture function.
        if(code >0){
            circle( rotated_image_opt, center_p, 60, Scalar(0,255,0), 10, LINE_AA);
            counter_p++;
            // get the value of each bit of corresponding edge.
            int a_code = (code & ( 1 << 0 )) >> 0;
            int b_code = (code & ( 1 << 1 )) >> 1;
            int c_code = (code & ( 1 << 2 )) >> 2;
            int d_code = (code & ( 1 << 3 )) >> 3;

            if(a_code>0){
                edge_a.emplace_back(center_p);
            }
            if(b_code>0){
                edge_b.emplace_back(center_p);
            }
            if(c_code>0){
                edge_c.emplace_back(center_p);
            }
            if(d_code>0){
                edge_d.emplace_back(center_p);
            }
        }
    }
    // draw the hough circle captured by the four edges.
    for(auto&& point: edge_a){
        circle( rotated_image_opt, point, 70, Scalar(255,0,0), 3, LINE_AA);
    }
    for(auto&& point: edge_b){
        circle( rotated_image_opt, point, 80, Scalar(0,255,120), 3, LINE_AA);
    }
    for(auto&& point: edge_c){
        circle( rotated_image_opt, point, 90, Scalar(225,210,0), 3, LINE_AA);
    }
    for(auto&& point: edge_d){
        circle( rotated_image_opt, point, 100, Scalar(210,240,0), 3, LINE_AA);
    }

    if((edge_a.size()<2 && edge_d.size()<2) || (edge_b.size()<2 && edge_c.size()<2)){
        std::cerr << "fatal error: can not know which direction is okay" << std::endl;
    }

    // calculate the distances between each two hough circles captured by each edge.
    // record the distance into horizontal edge and vertical edge.
    std::vector<double> distance_a;
    std::vector<double> distance_b;
    for(auto&& point_i: edge_a){
        for(auto&& point_j: edge_a){
            if(point_i.x != point_j.x && point_i.y!=point_j.y){
                distance_a.emplace_back(norm(point_i-point_j));
            }
        }
    }
    for(auto&& point_i: edge_b){
        for(auto&& point_j: edge_b){
            if(point_i.x != point_j.x && point_i.y!=point_j.y){
                distance_b.emplace_back(norm(point_i-point_j));
            }
        }
    }
    for(auto&& point_i: edge_c){
        for(auto&& point_j: edge_c){
            if(point_i.x != point_j.x && point_i.y!=point_j.y){
                distance_b.emplace_back(norm(point_i-point_j));
            }
        }
    }
    for(auto&& point_i: edge_d){
        for(auto&& point_j: edge_d){
            if(point_i.x != point_j.x && point_i.y!=point_j.y){
                distance_a.emplace_back(norm(point_i-point_j));
            }
        }
    }
    // find the minimum distance on vertical edge (ad) and horizontal edge (bc).
    double min_distance_height = *(std::min_element(distance_a.begin(), distance_a.end()));
    double min_distance_width =*(std::min_element(distance_b.begin(), distance_b.end()));
    std::cout << "min_distance_height: " << min_distance_height << std::endl;
    std::cout << "min_distance_width: " << min_distance_width<< std::endl;
    std::cout << "edge_a:" << edge_a.size() << ", edge_b:" << edge_b.size() <<", edge_c:" << edge_c.size() <<", edge_d:" << edge_d.size() <<std::endl;

    std::cout << "rotated: "<< counter_p << std::endl;
    std::cout << "angle: "<< angle_opt/2.0 << std::endl;
    Mat rotated_image_opt_p;
    // if the vertical minimum distance is less than the horizontal one, the rotated the image by 90 degree.
    // otherwise, it is in correct postion.
    if(min_distance_height < min_distance_width){
        rotation_matrix_opt = getRotationMatrix2D(center_img, 90, 1.0);

        // rotate the image using warpAffine
        warpAffine(rotated_image_opt, rotated_image_opt_p, rotation_matrix_opt, src.size());
    }else{
        rotated_image_opt_p = rotated_image_opt;
    }
    // display the instruction of rotation.
    double real_rotation =  min_distance_height < min_distance_width ? angle_opt/2.0+90: angle_opt/2.0;
    if(real_rotation > 360){
        real_rotation -= 360;
    }

    std::cout << "real rotate angle: clock_wise "<< (real_rotation> 180? 360-real_rotation: -real_rotation) << "°" << std::endl;

    // display the correct image.
    Size newSize(src.cols/5, src.rows/5);
    resize(rotated_image_opt_p, rotated_image_opt_p, newSize);

    imshow("Rotated image by clockwise of "+ std::to_string((real_rotation> 180? 360-real_rotation: -real_rotation) )+ "°" , rotated_image_opt_p);
    //![display]
    waitKey();
    return EXIT_SUCCESS;
}