.. _find_contours:

Finding contours in your image
******************************

Goal
=====

In this tutorial you will learn how to:

.. container:: enumeratevisibleitemswithsquare

   * Use the OpenCV function :find_contours:`findContours <>`
   * Use the OpenCV function :draw_contours:`drawContours <>`

Theory
======

Code
====

This tutorial code's is shown lines below. You can also download it from `here <https://github.com/Itseez/opencv/tree/master/samples/cpp/tutorial_code/ShapeDescriptors/findContours_demo.cpp>`_

.. code-block:: cpp

   #include "opencv2/highgui/highgui.hpp"
   #include "opencv2/imgproc/imgproc.hpp"
   #include <iostream>
   #include <stdio.h>
   #include <stdlib.h>

   using namespace cv;
   using namespace std;

   Mat src; Mat src_gray;
   int thresh = 100;
   int max_thresh = 255;
   RNG rng(12345);

   /// Function header
   void thresh_callback(int, void* );

   /** @function main */
   int main( int argc, char** argv )
   {
     /// Load source image and convert it to gray
     src = imread( argv[1], 1 );

     /// Convert image to gray and blur it
     cvtColor( src, src_gray, CV_BGR2GRAY );
     blur( src_gray, src_gray, Size(3,3) );

     /// Create Window
     char* source_window = "Source";
     namedWindow( source_window, CV_WINDOW_AUTOSIZE );
     imshow( source_window, src );

     createTrackbar( " Canny thresh:", "Source", &thresh, max_thresh, thresh_callback );
     thresh_callback( 0, 0 );

     waitKey(0);
     return(0);
   }

   /** @function thresh_callback */
   void thresh_callback(int, void* )
   {
     Mat canny_output;
     vector<vector<Point> > contours;
     vector<Vec4i> hierarchy;

     /// Detect edges using canny
     Canny( src_gray, canny_output, thresh, thresh*2, 3 );
     /// Find contours
     findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

     /// Draw contours
     Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
     for( int i = 0; i< contours.size(); i++ )
        {
          Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
          drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
        }

     /// Show in a window
     namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
     imshow( "Contours", drawing );
   }

Explanation
============

Result
======

#. Here it is:

   ============= =============
    |contour_0|   |contour_1|
   ============= =============

   .. |contour_0|  image:: images/Find_Contours_Original_Image.jpg
                     :align: middle

   .. |contour_1|  image:: images/Find_Contours_Result.jpg
                     :align: middle
