To build:
% cmake -DOpenCV_DIR=/home/lane/opencv/build .
% make

Image List Creation Example
===========================
% fswebcam --device /dev/video1 --jpeg -1 --no-banner --resolution 640x480 chessboard1.jpg
...
%./imagelist_creator images170.xml images170/*.jpg

Calibration Example
===================
% ./calibration -w 9 -h 6 -s 1.95 -o calibration170.yml -op -oe images170.xml

Undistort Video Example
=======================
% ./undistort
