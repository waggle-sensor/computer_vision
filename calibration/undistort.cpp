#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"

#include <cctype>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <time.h>

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
    const char* calibration_filename = "calibration170.yml";
    FileStorage fs(calibration_filename, FileStorage::READ);
    if( !fs.isOpened() )
        return false;
    Mat camera_matrix, distortion_coefficients;
    fs["camera_matrix"] >> camera_matrix;
    cout << "Camera Matrix:" << endl << camera_matrix << endl;
    fs["distortion_coefficients"] >> distortion_coefficients;
    cout << "Distortion Matrix:" << endl << distortion_coefficients << endl;

    Mat map1, map2;
    initUndistortRectifyMap(camera_matrix, distortion_coefficients,
                            Mat(), Mat(), Size(640, 480),
                            CV_32FC1, map1, map2);

    int interpolation = INTER_LINEAR;


    Mat input_image;
    /*
    const char* image_filename = "images/chessboard1.jpg";
    input_image = imread(image_filename);
    */

    VideoCapture capture;
    capture.set(cv::CAP_PROP_FRAME_WIDTH, 2592);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, 1944); 
    //capture.open(-1);
    capture.open(1);
    if (! capture.isOpened()) {
      printf("--(!)Error opening video capture\n");
      return -1;
    }
		while (capture.read(input_image)) {
			if(input_image.empty()) {
				printf(" --(!) No captured frame -- Break!");
				break;
			}

			imshow("Distorted Image", input_image);
			Mat output_image;
			remap(input_image, output_image,
						map1, map2,
						interpolation);
			imshow("Undistorted Image", output_image);

      char c = (char)waitKey(10);
      if(c == 27) {
        break;
      } // escape
		}

    //waitKey(0);

    return 0;
}
