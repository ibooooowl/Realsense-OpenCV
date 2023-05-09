//
// Created by user on 04/05/23.
//
#include<iostream>
#include<opencv2/opencv.hpp>
#include <vector>
#include <algorithm>

using namespace cv;
using namespace std;

/*int main(int argc, char** argv) {
    // [load]
    const char* default_file = "/home/user/CLionProjects/realsense-opencv/image/box8.jpg";
    const char* filename = argc > 1 ? argv[1] : default_file;


    // Loads an image
    Mat src = imread(samples::findFile(filename), IMREAD_COLOR);
    Mat img1 = imread(samples::findFile(filename), IMREAD_COLOR);
    Mat img2 = imread(samples::findFile(filename), IMREAD_COLOR);
    cout << src.size() << endl; // height, width

    // Check if image is loaded fine
    if (src.empty()) {
        cout << "Error opening image!" << endl;
        cout << "Usage: " << argv[0] << " [image_name -- default " << default_file << "]" << endl;
        return -1;
    }
    // [load]

    // [convert_to_gray]
    // Convert it to gray
    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    // [convert_to_gray]

    // [reduce_noise]
    // Reduce the noise to avoid false circle detection
    medianBlur(gray, gray, 5);
    // [reduce_noise]

    // [houghcircles]
    vector<Vec3f> circles;
    int rows = gray.rows;
    HoughCircles(gray, circles, HOUGH_GRADIENT, 1, rows / 64, 100, 30, 0, 30);
    // [houghcircles]
    cout << circles << endl;

    // Display the loaded image
    imshow("Image", src);
    waitKey(0);

    return 0;


}*/

int main(int argc, char** argv) {
    // Load an image
    String default_file = "/home/user/CLionProjects/realsense-opencv/image/box.jpg";
    String filename = (argc > 1) ? argv[1] : default_file;
    Mat src = imread(samples::findFile(filename), IMREAD_COLOR);
    Mat img1 = imread(samples::findFile(filename), IMREAD_COLOR);
    Mat img2 = imread(samples::findFile(filename), IMREAD_COLOR);

    // Check if image is loaded fine
    if (src.empty()) {
        cerr << "Error opening image!" << endl;
        cerr << "Usage: " << argv[0] << " [image_name -- default " << default_file << "] \n";
        return -1;
    }

    // Convert it to gray
    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);

    // Reduce the noise to avoid false circle detection
    medianBlur(gray, gray, 5);

    // Detect circles using HoughCircles
    vector<Vec3f> circles;
    int rows = gray.rows;
    HoughCircles(gray, circles, HOUGH_GRADIENT, 1, rows / 64, 100, 30, 0, 30);


}
