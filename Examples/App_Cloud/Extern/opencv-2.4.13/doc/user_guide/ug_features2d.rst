**********
Features2d
**********

.. highlight:: cpp

Detectors
=========

Descriptors
===========

Matching keypoints
==================

The code
--------
We will start with a short sample ``opencv/samples/cpp/matcher_simple.cpp``: ::

    Mat img1 = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    Mat img2 = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
    if(img1.empty() || img2.empty())
    {
        printf("Can't read one of the images\n");
        return -1;
    }

    // detecting keypoints
    SurfFeatureDetector detector(400);
    vector<KeyPoint> keypoints1, keypoints2;
    detector.detect(img1, keypoints1);
    detector.detect(img2, keypoints2);

    // computing descriptors
    SurfDescriptorExtractor extractor;
    Mat descriptors1, descriptors2;
    extractor.compute(img1, keypoints1, descriptors1);
    extractor.compute(img2, keypoints2, descriptors2);

    // matching descriptors
    BruteForceMatcher<L2<float> > matcher;
    vector<DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);

    // drawing the results
    namedWindow("matches", 1);
    Mat img_matches;
    drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches);
    imshow("matches", img_matches);
    waitKey(0);

The code explained
------------------

Let us break the code down. ::

    Mat img1 = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    Mat img2 = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
    if(img1.empty() || img2.empty())
    {
        printf("Can't read one of the images\n");
        return -1;
    }

We load two images and check if they are loaded correctly.::

    // detecting keypoints
    FastFeatureDetector detector(15);
    vector<KeyPoint> keypoints1, keypoints2;
    detector.detect(img1, keypoints1);
    detector.detect(img2, keypoints2);

First, we create an instance of a keypoint detector. All detectors inherit the abstract ``FeatureDetector`` interface, but the constructors are algorithm-dependent. The first argument to each detector usually controls the balance between the amount of keypoints and their stability. The range of values is different for different detectors (For instance, *FAST* threshold has the meaning of pixel intensity difference and usually varies in the region *[0,40]*. *SURF* threshold is applied to a Hessian of an image and usually takes on values larger than *100*), so use defaults in case of doubt. ::

    // computing descriptors
    SurfDescriptorExtractor extractor;
    Mat descriptors1, descriptors2;
    extractor.compute(img1, keypoints1, descriptors1);
    extractor.compute(img2, keypoints2, descriptors2);

We create an instance of descriptor extractor. The most of OpenCV descriptors inherit ``DescriptorExtractor`` abstract interface. Then we compute descriptors for each of the keypoints. The output ``Mat`` of the ``DescriptorExtractor::compute`` method contains a descriptor in a row *i* for each *i*-th keypoint. Note that the method can modify the keypoints vector by removing the keypoints such that a descriptor for them is not defined (usually these are the keypoints near image border). The method makes sure that the ouptut keypoints and descriptors are consistent with each other (so that the number of keypoints is equal to the descriptors row count). ::

    // matching descriptors
    BruteForceMatcher<L2<float> > matcher;
    vector<DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);

Now that we have descriptors for both images, we can match them. First, we create a matcher that for each descriptor from image 2 does exhaustive search for the nearest descriptor in image 1 using Euclidean metric. Manhattan distance is also implemented as well as a Hamming distance for Brief descriptor. The output vector ``matches`` contains pairs of corresponding points indices. ::

    // drawing the results
    namedWindow("matches", 1);
    Mat img_matches;
    drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches);
    imshow("matches", img_matches);
    waitKey(0);

The final part of the sample is about visualizing the matching results.
