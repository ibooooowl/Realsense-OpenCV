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
// https://theailearner.com/tag/cv2-getrotationmatrix2d/

//Point rotateBy(const Point2d& inPoint, const Mat& rotated_matrix)
//{
//    Point outPoint;
//    vector<double> point{inPoint.x, inPoint.y, 1};
//    Mat my_m = Mat(point);
//    Mat res = rotated_matrix* my_m;
//    outPoint.x= int(res.at<double>(0, 0));
//    outPoint.y= int(res.at<double>(0, 1));
//    return outPoint;
//}

Point rotateBy(const Point2d& inPoint, const Mat& rotated_matrix)
{
    Point outPoint;
    Mat point = (Mat_<double>(3, 1) << inPoint.x, inPoint.y, 1.0);
    Mat res = rotated_matrix * point;
    outPoint.x = int(res.at<double>(0, 0));
    outPoint.y = int(res.at<double>(1, 0));
    return outPoint;
}

double approximate_area_contour(const vector<Point>& contour){
    // initialize values of the minimum bounding rectangle.
    int x_min = contour[0].x;
    int x_max = contour[0].x;
    int y_min = contour[0].y;
    int y_max = contour[0].y;
    for(auto point: contour){
        // update x_min, x_max
        if (x_min > point.x) {
            x_min = point.x;
        }
        if (x_max < point.x) {
            x_max = point.x;
        }
        // update y_min, y_max
        if (y_min > point.y) {
            y_min = point.y;
        }
        if (y_max < point.y) {
            y_max = point.y;
        }
    }

    return (x_max-x_min)*(y_max-y_min)- pow((x_max-x_min)-(y_max-y_min) , 2);
}

bool less_min_dist(const vector<Point> &contour_1, const vector<Point>&contour_2, double dist_threshold){
    for(auto p1: contour_1){
        for(auto p2:contour_2){
            if(norm(p1-p2) < dist_threshold){
                return true;
            };
        }
    }
    return false;
}



void dfs(int v, vector<bool> &used, vector<int> &comp, const vector<set<int>> &adj) {
    used[v] = true ;
    comp.push_back(v);
    for (int u : adj[v]) {
        if (!used[u]) {
            dfs(u, used, comp, adj);
        }
    }
}

bool rectContains(const Rect&rect, const Point&point){
    return point.x>=rect.x && point.x<=rect.x+rect.width && point.y>=rect.y && point.y<=rect.y-rect.height;
}

cv::Mat GetSquareImage( const cv::Mat& img, int target_width = 500){
    int width = img.cols, height = img.rows;

    cv::Mat square = cv::Mat::zeros( target_width, target_width, img.type() );

    int max_dim = ( width >= height ) ? width : height;
    float scale = ( ( float ) target_width ) / max_dim;
    cv::Rect roi;
    if ( width >= height )
    {
        roi.width = target_width;
        roi.x = 0;
        roi.height = height * scale;
        roi.y = ( target_width - roi.height ) / 2;
    }
    else
    {
        roi.y = 0;
        roi.height = target_width;
        roi.width = width * scale;
        roi.x = ( target_width - roi.width ) / 2;
    }

    cv::resize( img, square( roi ), roi.size() );

    return square;
}



int main(int argc, char** argv)
{
    try {
        //![load]
        const char *filename =
                // argc >= 2 ? argv[1] : "/home/user/CLionProjects/realsense-opencv/image/picture_ampoule15_Color.png";
                argc >= 2 ? argv[1] : "/home/user/Downloads/image.jpg";
        //const char* filename = argc >=2 ? argv[1] : "/home/user/CLionProjects/realsense-opencv/image/box.jpg";

        // threshold to adjust the performance of proposed algorithm.
        // minimum distance between the contours.
        const double dis_thre = 5;
        // minimum distance between the hough circles.
        const double dist_min_hough = 30;
        // minimum frequency of radius of hough circles to construct the bounding rectangle.
        const int threshold_radius = 5;
        // maximum distance of point the bounding rectangle.
        const double dist_threshold = 5;
        const float min_radius_circle = 1;
        const float  max_radius_circle = 30;

        // Loads an image
        Mat src_orig = imread(samples::findFile(filename), IMREAD_COLOR);

        Mat src = GetSquareImage(src_orig, max(src_orig.rows, src_orig.cols));



        // Check if image is loaded fine
        if (src.empty()) {
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



//    Mat img_binary;
//    threshold(gray, img_binary, 127, 255, THRESH_BINARY);
//
//    resize(img_binary, img_binary, newSize);
//    imshow("Detected by binary", img_binary);
//    waitKey();

        // display the correct image.
//    Mat img_gray=gray.clone();
//    resize(img_gray, img_gray, newSize);
//    imshow("Detected by grey", img_gray);
//    waitKey();

        Mat img_binary;
        threshold(gray, img_binary, 127, 255, THRESH_BINARY);


//    imshow("Detected by binary", img_binary);
//    waitKey();

        // Edge detection
        Mat edges;
        Canny(img_binary, edges, 100, 200);    //Canny

        // Contour extraction
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(edges, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        vector<set<int>> adjacent_matrix(contours.size(), set<int>());


        // find the closest contour for each if the distance is less than given threshold.
        for (int i = 0; i < contours.size(); ++i) {
            // adjacent_matrix[i].emplace(i);
            for (int j = i + 1; j < contours.size(); ++j) {
                // calculate the minimum distance between two contours.
                if (less_min_dist(contours[i], contours[j], dis_thre)) {
                    adjacent_matrix[i].emplace(j);
                    adjacent_matrix[j].emplace(i);
                }
            }

        }
        vector<bool> used(contours.size(), false);
        vector<vector<int>> comp;
        for (int v = 0; v < contours.size(); ++v) {
            if (!used[v]) {
                vector<int> comp_i;
                dfs(v, used, comp_i, adjacent_matrix);
                comp.emplace_back(comp_i);
            }
        }

        vector<vector<Point>> contours_merge;
        for (auto &i: comp) {
            if (!i.empty()) {
                vector<Point> contour;
                for (auto &j: i) {
                    for (auto &point: contours[j]) {
                        contour.emplace_back(point);
                    }
                }
                contours_merge.emplace_back(contour);
            }
        }

        // draw orig contours
        Mat linePic_orig;
        linePic_orig = Mat::zeros(edges.rows, edges.cols, CV_8UC3);
        for (int index = 0; index < contours.size(); index++) {
            drawContours(linePic_orig, contours, index, Scalar(rand() & 255, rand() & 255, rand() & 255), 1,
                         8/*, hierarchy*/);
        }

        // draw contours
        Mat linePic;
        linePic = Mat::zeros(edges.rows, edges.cols, CV_8UC3);
        for (int index = 0; index < contours_merge.size(); index++) {
            drawContours(linePic, contours_merge, index, Scalar(rand() & 255, rand() & 255, rand() & 255), 1,
                         8/*, hierarchy*/);
        }


        // Filter square contours
        vector<vector<Point>> polyContours(contours_merge.size());
        std::vector<double> area_contours;
        for (auto &contour: contours_merge) {
            area_contours.emplace_back(approximate_area_contour(contour));
        }
        auto max_area_contour_arg =
                std::max_element(area_contours.begin(), area_contours.end()) - area_contours.begin();

        for (int index = 0; index < contours_merge.size(); index++) {
            approxPolyDP(contours_merge[index], polyContours[index], 10, true);
        }

        Mat polyPic = Mat::zeros(src.size(), CV_8UC3);
        drawContours(polyPic, polyContours, max_area_contour_arg,
                     Scalar(0, 0, 255/*rand() & 255, rand() & 255, rand() & 255*/), 2);
        RotatedRect box = minAreaRect(contours_merge[max_area_contour_arg]);
        Point2f vtx[4];
        box.points(vtx);

        for (int i = 0; i < 4; ++i) {
            line(polyPic, vtx[i], vtx[(i + 1) % 4], Scalar(255, 255, 0), 5, LINE_AA);
            line(src, vtx[i], vtx[(i + 1) % 4], Scalar(255, 255, 0), 5, LINE_AA);
        }
//    imshow("Draw square", polyPic);
//    waitKey();

        vector<Point2f> contour_optimal_approximate = {Point2f(vtx[0].x, vtx[0].y), Point2f(vtx[1].x, vtx[1].y),
                                                       Point2f(vtx[2].x, vtx[2].y), Point2f(vtx[3].x, vtx[3].y)};

        Rect boundRect = boundingRect(Mat(contour_optimal_approximate));


//        boundRect.x = max(boundRect.x - 100, 0);
//        boundRect.y = max(boundRect.y - 100, 0);
//        boundRect.width += min(src.cols - boundRect.x - boundRect.width, 200);
//        boundRect.height += min(src.rows - boundRect.y - boundRect.height, 200);
        Mat src_crop = src ;




        std::cout << "Crop the image" << std::endl;

        vector<Vec3f> circles_orig;


        for (const auto & index : contours) {
            Point2f center;
            float radius;
            minEnclosingCircle(index, center,  radius);
            Point center_p =center;
            if(radius < max_radius_circle &&radius > min_radius_circle && pointPolygonTest(contour_optimal_approximate, center, false)==1){
                circles_orig.emplace_back(center.x, center.y, radius);
            }
        }


//        Mat gray_crop;
//        cvtColor(src_crop, gray_crop, COLOR_BGR2GRAY);
//
//        //![convert_to_gray]
//
//        //![reduce_noise]
//        medianBlur(gray_crop, gray_crop, 5);
//
//        //![houghcircles]
//        vector<Vec3f> circles_orig;
//        HoughCircles(gray_crop, circles_orig, HOUGH_GRADIENT, 1,
//                     100,  // change this value to detect circles with different distances to each other
//                     30, 50, 0, 50 // change the last two parameters
//                // (min_radius & max_radius) to detect larger circles
//        );
//        //![houghcircles]
        std::cout << "get hough circles" << std::endl;
        // the distance to eliminate the closed hough circles with larger radius.


        vector<Vec3f> circles;
        vector<vector<pair<int, int>>> circles_bad;
        for (int i = 0; i < circles_orig.size(); ++i) {
            bool good_circle = true;
            vector<pair<int, int>> temp;
            for (int j = 0; j < circles_orig.size(); ++j) {
                if (norm(Point2f(circles_orig[i][0], circles_orig[i][1]) -
                         Point2f(circles_orig[j][0], circles_orig[j][1])) < dist_min_hough) {
                    good_circle = false;
                    temp.emplace_back(j, int(circles_orig[j][2]));
                }
            }
            if (good_circle) {
                circles.emplace_back(circles_orig[i]);
            } else {
                temp.emplace_back(i, int(circles_orig[i][2]));
                circles_bad.emplace_back(temp);
            }
        }
        for (auto &i: circles_bad) {
            std::sort(i.begin(), i.end(),
                      [](const pair<int, int> &a, const pair<int, int> &b) { return a.second > b.second; });
        }
        set<int> index_set;
        for (auto &i: circles_bad) {
            index_set.emplace(i.front().first);
        }
        for (auto &i: index_set) {
            circles.emplace_back(circles_orig[i]);
        }

        std::cout << "Get proper hough circles" << std::endl;


        Mat img_hough = src_crop.clone();
        for (const auto &point: circles) {
            Vec3i point_p = point;
            // get the radius of the corresponding center point.
            int radius = point_p[2];
            Point center = Point(point_p[0], point_p[1]);
            circle(img_hough, center, radius, Scalar(255, 0, 255), 3, LINE_AA);

        }

//        imshow("Detected by hough", img_hough);
//        waitKey();



        // get the radius of the detected circles and their occurrences.
        std::map<int, int> radius_map;
        for (auto &&point: circles) {
            Vec3i point_p = point;
            int radius = point_p[2];
            if (radius_map.count(radius) < 1) {
                radius_map.emplace(radius, 1);
            } else {
                radius_map[radius] = radius_map[radius] + 1;
            }

        }
        // transform the map into the corresponding vector to sort the radius of circles by their occurrences.
        std::vector<std::pair<int, int>> radius_list(radius_map.begin(), radius_map.end());
        // sort the vector by the non-decreasing order of occurrences of radius
        std::sort(radius_list.begin(), radius_list.end(),
                  [](const std::pair<int, int> &a, const std::pair<int, int> &b) -> bool {
                      return a.second > b.second;
                  });



        // select the radius of hough circles by their occurrences not less than a given threshold.
        std::set<int> select_radius;

        for (auto &&ele: radius_list) {
            if (ele.second > threshold_radius) {
                select_radius.emplace(ele.first);
            }
        }

        // get the center coordinates of the image to create the 2D rotation matrix
        Point2d center_img((src_crop.cols - 1) / 2.0, (src_crop.rows - 1) / 2.0);


        // the map to record the captured hough circles by created rectangle at different rotated angle.
        std::map<int, int> angle_count_map;

        // we rotate 90 degree for a rectangle, each time 0.1 degree
        Vec3i point_first = circles[0];
        Point center_first = Point(point_first[0], point_first[1]);
        for (int i = -150; i < 150; i++) {
            double angle = box.angle+ i / 10.0;
            // using getRotationMatrix2D() to get the rotation matrix
            Mat rotation_matrix = getRotationMatrix2D(center_img, angle, 1.0);
            // get the corresponding rotated point of the first hough circles.
            Point center_p_first = rotateBy(center_first, rotation_matrix);
            // initialize the value of x_min, y_min, x_max, y_max.
            int x_min = center_p_first.x;
            int x_max = center_p_first.x;
            int y_min = center_p_first.y;
            int y_max = center_p_first.y;
            // iterate all detected hough circles, get their rotated one by the given rotated angle,
            // then update the x_min, y-Min, x_max, y_max
            for (const auto &point: circles) {
                Vec3i point_p = point;
                // get the radius of the corresponding center point.
                int radius = point_p[2];
                // if the radius is in the selected radius set, then update x_min, y_min, x_max, y_max.
                // otherwise, these hough circles could be the noise.
                if ((select_radius.count(radius) > 0)) {
                    // get the center point of detected hough circle.
                    Point center = Point(point_p[0], point_p[1]);
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

            int counter = 0;
            int counter_a = 0;
            int counter_b = 0;
            int counter_c = 0;
            int counter_d = 0;
            // iterate the hough circles, count how many rotated hough circles can be captured by the constructed rectangle.
            for (const auto &point: circles) {
                Vec3i point_p = point;
                Point center = Point(point_p[0], point_p[1]);
                Point center_p = rotateBy(center, rotation_matrix);
                //if(select_radius.count(point_p[2])>0 && onRectangleW(Point(x_min, y_min), Point(x_max, y_max), center_p, dist_threshold)>0){
                int code = onRectangleW(Point(x_min, y_min), Point(x_max, y_max), center_p, dist_threshold);
                //if(select_radius.count(point_p[2])>0 && code >0){
                // extract the code by the capture function.
                if (code > 0) {
                    // circle( rotated_image_opt, center_p, 60, Scalar(0,255,0), 10, LINE_AA);
                    counter++;
                    // get the value of each bit of corresponding edge.
                    int a_code = (code & (1 << 0)) >> 0;
                    int b_code = (code & (1 << 1)) >> 1;
                    int c_code = (code & (1 << 2)) >> 2;
                    int d_code = (code & (1 << 3)) >> 3;

                    if (a_code > 0) {
                        counter_a++;
                    }
                    if (b_code > 0) {
                        counter_b++;
                    }
                    if (c_code > 0) {
                        counter_c++;
                    }
                    if (d_code > 0) {
                        counter_d++;
                    }
                }
//                if (onRectangleW(Point(x_min, y_min), Point(x_max, y_max), center_p, dist_threshold) > 0) {
//                    counter++;
//                }
            }
            // update the map of rotated angle and their captured hough circles.
            if((counter_a>1||counter_d>1)&&(counter_b>1||counter_c>1)) {
                angle_count_map.emplace(i, counter);
            }else{
                angle_count_map.emplace(i, -1);
            }
        }

        // transform the involved map into the vector in order to sort it.
        std::vector<std::pair<int, int>> angle_count_list(angle_count_map.begin(), angle_count_map.end());
        // sort the vector non-decreasing first by the number of captured hough circles.
        // secondly by the non-increasing of rotation angle.
        std::sort(angle_count_list.begin(), angle_count_list.end(),
                  [](const std::pair<int, int> &a, const std::pair<int, int> &b) -> bool {
                      if (a.second > b.second) return true;
                      else if (a.second < b.second) return false;
                      else
                          return a.first < b.first;
                  });

        // display the rotated angle, and the number of captured hough circles.
        for (auto &&ele: angle_count_list) {
            std::cout << ele.first / 10.0 << "->" << ele.second << std::endl;
        }
        std::cout << "end" << std::endl;
        // the optimal rotated angle is the one has the largest number of captured hough circles by the constructed rectangle.
        int radius_opt = angle_count_list[0].second;
        bool find = false;
        int i_ter = 0;
        while (!find) {
            if (angle_count_list[i_ter].second != radius_opt) {
                find = true;
            } else {
                i_ter++;
            }
        }
        double angle_opt = box.angle+(angle_count_list[int(i_ter / 2)].first) / 10.0;

        // we then rotate the image by the associate angle to get the correct view of image.

        // using getRotationMatrix2D() to get the rotation matrix
        Mat rotation_matrix_opt = getRotationMatrix2D(center_img, angle_opt, 1.0);
        // we will save the resulting image in rotated_image matrix
        Mat rotated_image_opt;
        // rotate the image using warpAffine
        warpAffine(src_crop, rotated_image_opt, rotation_matrix_opt, src_crop.size());
        std::cout << "rotate the image with optimal angle " << std::endl;



        // draw the detected hough circles in rotated image and also the constructed rectangle.
        // initialize the value of x_min_p, y_min_p, x_max_p, y_max_p.
        Point first = rotateBy(center_first, rotation_matrix_opt);
        int x_min_p = first.x;
        int x_max_p = first.x;
        int y_min_p = first.y;
        int y_max_p = first.y;
        // circle(rotated_image, first, 70, Scalar(255, 255, 255), 6, LINE_AA);
        // iterate all the detected hough circles.
        for (const auto &point: circles) {
            // get the center point of hough circle.
            Vec3i point_p = point;
            Point center = Point(point_p[0], point_p[1]);
            // get the rotated center point in the rotated image.
            Point center_p = rotateBy(center, rotation_matrix_opt);
            // circle outline
            int radius = point_p[2];
            // draw the detected hough circles in the rotated image.
            circle(rotated_image_opt, center_p, radius, Scalar(0, 0, 255), 3, LINE_AA);
            // if radius is in the selected radius set, then draw it and update the x_min_p,y_min_p,x_max_p,y_max_p.
            if ((select_radius.count(radius) > 0)) {
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
                circle(rotated_image_opt, center_p, radius + 5, Scalar(255, 255, 255), 5, LINE_AA);
            }
        }
        // draw the corresponding rectangle.
        rectangle(rotated_image_opt, Point(x_min_p, y_min_p), Point(x_max_p, y_max_p), Scalar(255, 0, 0), 3);
        std::cout << " draw the bounding rectangle" << std::endl;
        // count up the hough circles can be captured by each edge of the rectangle.
        int counter_p = 0;
        std::vector<Point> edge_a;
        std::vector<Point> edge_b;
        std::vector<Point> edge_c;
        std::vector<Point> edge_d;
        // iterate the detected hough circles.
        for (const auto &point: circles) {
            // get the center point of the hough circle.
            Vec3i point_p = point;
            Point center = Point(point_p[0], point_p[1]);
            // get the rotated center point.
            Point center_p = rotateBy(center, rotation_matrix_opt);
            // detect whether capture by the constructed rectangle.
            int code = onRectangleW(Point(x_min_p, y_min_p), Point(x_max_p, y_max_p), center_p, dist_threshold);
            //if(select_radius.count(point_p[2])>0 && code >0){
            // extract the code by the capture function.
            if (code > 0) {
                // circle( rotated_image_opt, center_p, 60, Scalar(0,255,0), 10, LINE_AA);
                counter_p++;
                // get the value of each bit of corresponding edge.
                int a_code = (code & (1 << 0)) >> 0;
                int b_code = (code & (1 << 1)) >> 1;
                int c_code = (code & (1 << 2)) >> 2;
                int d_code = (code & (1 << 3)) >> 3;

                if (a_code > 0) {
                    edge_a.emplace_back(center_p);
                }
                if (b_code > 0) {
                    edge_b.emplace_back(center_p);
                }
                if (c_code > 0) {
                    edge_c.emplace_back(center_p);
                }
                if (d_code > 0) {
                    edge_d.emplace_back(center_p);
                }
            }
        }
        // draw the hough circle captured by the four edges.
        for (auto &&point: edge_a) {
            circle(rotated_image_opt, point, 30, Scalar(255, 0, 0), 3, LINE_AA);
        }
        for (auto &&point: edge_b) {
            circle(rotated_image_opt, point, 35, Scalar(0, 255, 120), 3, LINE_AA);
        }
        for (auto &&point: edge_c) {
            circle(rotated_image_opt, point, 40, Scalar(225, 210, 0), 3, LINE_AA);
        }
        for (auto &&point: edge_d) {
            circle(rotated_image_opt, point, 45, Scalar(210, 240, 0), 3, LINE_AA);
        }

        if ((edge_a.size() < 2 && edge_d.size() < 2) || (edge_b.size() < 2 && edge_c.size() < 2)) {
            std::cerr << "fatal error: can not know which direction is okay" << std::endl;
        }

        std::cout << "draw the captured circles by the bounding rectangle" << std::endl;
        std::cout << "edge_a:" << edge_a.size() << ", edge_b:" << edge_b.size() << ", edge_c:" << edge_c.size()
                  << ", edge_d:" << edge_d.size() << std::endl;
//    std::cout << "on edge a:" << std::endl;
//    for(auto& a : edge_a){
//        std::cout << a.x << "," << a.y << std::endl;
//    }
//    std::cout << "on edge b:" << std::endl;
//    for(auto& a : edge_b){
//        std::cout << a.x << "," << a.y << std::endl;
//    }
//    std::cout << "on edge c:" << std::endl;
//    for(auto& a : edge_c){
//        std::cout << a.x << "," << a.y << std::endl;
//    }
//    std::cout << "on edge d:" << std::endl;
//    for(auto& a : edge_d){
//        std::cout << a.x << "," << a.y << std::endl;
//    }
        // calculate the distances between each two hough circles captured by each edge.
        // record the distance into horizontal edge and vertical edge.
        std::map<int, int> distance_a;
        std::map<int, int> distance_b;

        for (auto &&point_i: edge_a) {
            for (auto &&point_j: edge_a) {
                if ((point_i.x != point_j.x) || (point_i.y != point_j.y)) {
                    int distance = int(round(norm(point_i - point_j)));
                    if (distance_a.count(distance) > 0) {
                        distance_a[distance] = distance_a[distance] + 1;
                    } else {
                        distance_a[distance] = 1;
                    }
                    // std::cout << "dis a:" << "(" <<  point_i.x << "," << point_i.y << ")," << "(" <<  point_j.x << "," << point_j.y<< "):"  << distance<< std::endl;
                }
            }
        }
        for (auto &&point_i: edge_b) {
            for (auto &&point_j: edge_b) {
                if ((point_i.x != point_j.x) || (point_i.y != point_j.y)) {
                    int distance = int(round(norm(point_i - point_j)));
                    if (distance_b.count(distance) > 0) {
                        distance_b[distance] = distance_b[distance] + 1;
                    } else {
                        distance_b[distance] = 1;
                    }
                    // std::cout << "dis b:" << "(" <<  point_i.x << "," << point_i.y << ")," << "(" <<  point_j.x << "," << point_j.y<< "):"  << distance<< std::endl;
                }
            }
        }
        for (auto &&point_i: edge_c) {
            for (auto &&point_j: edge_c) {
                if ((point_i.x != point_j.x) || (point_i.y != point_j.y)) {
                    int distance = int(round(norm(point_i - point_j)));
                    if (distance_b.count(distance) > 0) {
                        distance_b[distance] = distance_b[distance] + 1;
                    } else {
                        distance_b[distance] = 1;
                    }
                    // std::cout << "dis c:" << "(" <<  point_i.x << "," << point_i.y << ")," << "(" <<  point_j.x << "," << point_j.y<< "):"  << distance<< std::endl;
                }
            }
        }
        for (auto &&point_i: edge_d) {
            for (auto &&point_j: edge_d) {
                if ((point_i.x != point_j.x) || (point_i.y != point_j.y)) {
                    int distance = int(round(norm(point_i - point_j)));
                    if (distance_a.count(distance) > 0) {
                        distance_a[distance] = distance_a[distance] + 1;
                    } else {
                        distance_a[distance] = 1;
                    }
                    // std::cout << "dis d:" << "(" <<  point_i.x << "," << point_i.y << ")," << "(" <<  point_j.x << "," << point_j.y<< "):"  << distance<< std::endl;
                }
            }
        }
        std::cout << "count the most common distance" << std::endl;
//    std::cout << "on height:" << std::endl;
//    for(auto& a : distance_a){
//        std::cout << a.first << ":"<< a.second << std::endl;
//    }
//    std::cout << "on width:" << std::endl;
//    for(auto& a : distance_b){
//        std::cout << a.first << ":"<< a.second << std::endl;
//    }
        // find the minimum distance on vertical edge (ad) and horizontal edge (bc).
        std::vector<pair<int, int>> distance_a_v(distance_a.begin(), distance_a.end());
        std::vector<pair<int, int>> distance_b_v(distance_b.begin(), distance_b.end());
        std::sort(distance_a_v.begin(), distance_a_v.end(),
                  [](const std::pair<int, int> &a, const std::pair<int, int> &b) {
                      if (a.second > b.second) return true;
                      if (a.second > b.second) return false;
                      return a.first < b.first;
                  });
        std::sort(distance_b_v.begin(), distance_b_v.end(),
                  [](const std::pair<int, int> &a, const std::pair<int, int> &b) {
                      if (a.second > b.second) return true;
                      if (a.second > b.second) return false;
                      return a.first < b.first;
                  });
        std::cout << "sort distance vector" << std::endl;
//    std::cout << "on height:" << std::endl;
//    for(auto& a : distance_a_v){
//        std::cout << a.first << ":"<< a.second << std::endl;
//    }
//    std::cout << "on width:" << std::endl;
//    for(auto& a : distance_b_v){
//        std::cout << a.first << ":"<< a.second << std::endl;
//    }
        double min_distance_height = distance_a_v[0].first;
        double min_distance_width = distance_b_v[0].first;
        std::cout << "min_distance_height: " << min_distance_height << std::endl;
        std::cout << "min_distance_width: " << min_distance_width << std::endl;
        std::cout << "edge_a:" << edge_a.size() << ", edge_b:" << edge_b.size() << ", edge_c:" << edge_c.size()
                  << ", edge_d:" << edge_d.size() << std::endl;

        std::cout << "nb of captured circles: " << counter_p << std::endl;
        std::cout << "angle: " << angle_opt << std::endl;
        Mat rotated_image_opt_p;
        // if the vertical minimum distance is less than the horizontal one, the rotated the image by 90 degree.
        // otherwise, it is in correct postion.
        if (min_distance_height < min_distance_width) {
            std::cout << "rotated 90" << std::endl;
            rotation_matrix_opt = getRotationMatrix2D(center_img, 90, 1.0);

            // rotate the image using warpAffine
            warpAffine(rotated_image_opt, rotated_image_opt_p, rotation_matrix_opt, src_crop.size());
        } else {
            rotated_image_opt_p = rotated_image_opt;
        }
        // display the instruction of rotation.
        double real_rotation = min_distance_height < min_distance_width ? angle_opt + 90 : angle_opt;

        std::cout << "real rotate angle: clock_wise " << (real_rotation > 90 ? 180 - real_rotation : -real_rotation)
                  << "°" << std::endl;


        Size newSize(src.cols/2, src.rows/2);

//        resize(src, src, newSize);
//        imshow("Src Img", src);


        resize(linePic_orig, linePic_orig, newSize);
        imshow("Draw contours orig", linePic_orig);

        resize(linePic, linePic, newSize);
        imshow("Draw contours", linePic);


        resize(img_hough, img_hough, newSize);

        imshow("Detected by houghcircles", img_hough);


        // display the correct image.
        resize(rotated_image_opt_p, rotated_image_opt_p, newSize);

        imshow("Rotated image by clockwise of " +
               std::to_string((real_rotation > 90 ? 180 - real_rotation : -real_rotation)) + "±" +
               std::to_string(i_ter / 20.0) + "°", rotated_image_opt_p);
        //![display]
        waitKey();
        return EXIT_SUCCESS;
    }catch (Exception&e) {
        return 1;
    }
}