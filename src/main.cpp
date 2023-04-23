#include <iostream>
#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>

int main() {
  // Create a RealSense pipeline object
  rs2::pipeline pipeline;

  // Start the pipeline and configure it to capture color frames
  pipeline.start();

  // Wait for the next frame set
  rs2::frameset frames = pipeline.wait_for_frames();

  // Get the color frame from the frame set
  rs2::frame color_frame = frames.get_color_frame();

  // Convert the color frame to OpenCV format
  cv::Mat color_mat(cv::Size(640, 480), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);

  // Display the color frame on the screen
  cv::imshow("Color Frame", color_mat);
  cv::waitKey(0);

  // Stop the pipeline and release any resources
  pipeline.stop();

  return 0;
}


