Feature Detection and Description
=================================

.. highlight:: cpp

.. note::

   * An example explaining keypoint detection and description can be found at opencv_source_code/samples/cpp/descriptor_extractor_matcher.cpp

FAST
----
Detects corners using the FAST algorithm

.. ocv:function:: void FAST( InputArray image, vector<KeyPoint>& keypoints, int threshold, bool nonmaxSuppression=true )

.. ocv:function:: void FASTX( InputArray image, vector<KeyPoint>& keypoints, int threshold, bool nonmaxSuppression, int type )

    :param image: grayscale image where keypoints (corners) are detected.

    :param keypoints: keypoints detected on the image.

    :param threshold: threshold on difference between intensity of the central pixel and pixels of a circle around this pixel.

    :param nonmaxSuppression: if true, non-maximum suppression is applied to detected corners (keypoints).

    :param type: one of the three neighborhoods as defined in the paper: ``FastFeatureDetector::TYPE_9_16``, ``FastFeatureDetector::TYPE_7_12``, ``FastFeatureDetector::TYPE_5_8``

Detects corners using the FAST algorithm by [Rosten06]_.

.. [Rosten06] E. Rosten. Machine Learning for High-speed Corner Detection, 2006.


MSER
----
.. ocv:class:: MSER : public FeatureDetector

Maximally stable extremal region extractor. ::

    class MSER : public CvMSERParams
    {
    public:
        // default constructor
        MSER();
        // constructor that initializes all the algorithm parameters
        MSER( int _delta, int _min_area, int _max_area,
              float _max_variation, float _min_diversity,
              int _max_evolution, double _area_threshold,
              double _min_margin, int _edge_blur_size );
        // runs the extractor on the specified image; returns the MSERs,
        // each encoded as a contour (vector<Point>, see findContours)
        // the optional mask marks the area where MSERs are searched for
        void operator()( const Mat& image, vector<vector<Point> >& msers, const Mat& mask ) const;
    };

The class encapsulates all the parameters of the MSER extraction algorithm (see [wiki]_ article).

.. note::

    * there are two different implementation of MSER: one for grey image, one for color image the grey image algorithm is taken from: [nister2008linear]_ ; the paper claims to be faster than union-find method; it actually get 1.5~2m/s on my centrino L7200 1.2GHz laptop.

    * the color image algorithm is taken from: [forssen2007maximally]_ ; it should be much slower than grey image method ( 3~4 times ); the chi_table.h file is taken directly from paper's source code which is distributed under GPL.

    * (Python) A complete example showing the use of the MSER detector can be found at opencv_source_code/samples/python2/mser.py

.. [wiki] http://en.wikipedia.org/wiki/Maximally_stable_extremal_regions
.. [nister2008linear] David Nistér and Henrik Stewénius. Linear time maximally stable extremal regions. In Computer Vision–ECCV 2008, pages 183–196. Springer, 2008.
.. [forssen2007maximally] Per-Erik Forssén. Maximally stable colour regions for recognition and matching. In Computer Vision and Pattern Recognition, 2007. CVPR'07. IEEE Conference on, pages 1–8. IEEE, 2007.

MSER::MSER
----------
The MSER constructor

.. ocv:function:: MSER::MSER(int _delta=5, int _min_area=60, int _max_area=14400, double _max_variation=0.25, double _min_diversity=.2, int _max_evolution=200, double _area_threshold=1.01, double _min_margin=0.003, int _edge_blur_size=5)

    :param _delta: Compares (sizei - sizei-delta)/sizei-delta
    :param _min_area: Prune the area which smaller than minArea
    :param _max_area: Prune the area which bigger than maxArea
    :param _max_variation: Prune the area have simliar size to its children
    :param _min_diversity: For color image, trace back to cut off mser with diversity less than min_diversity
    :param _max_evolution: For color image, the evolution steps
    :param _area_threshold: For color image, the area threshold to cause re-initialize
    :param _min_margin: For color image, ignore too small margin
    :param _edge_blur_size: For color image, the aperture size for edge blur

MSER::operator()
----------------

Detect MSER regions

.. ocv:function:: void MSER::operator()(const Mat& image, vector<vector<Point> >& msers, const Mat& mask=Mat() ) const

    :param image: Input image (8UC1, 8UC3 or 8UC4)
    :param msers: Resulting list of point sets
    :param mask: The operation mask

ORB
---
.. ocv:class:: ORB : public Feature2D

Class implementing the ORB (*oriented BRIEF*) keypoint detector and descriptor extractor, described in [RRKB11]_. The algorithm uses FAST in pyramids to detect stable keypoints, selects the strongest features using FAST or Harris response, finds their orientation using first-order moments and computes the descriptors using BRIEF (where the coordinates of random point pairs (or k-tuples) are rotated according to the measured orientation).

.. [RRKB11] Ethan Rublee, Vincent Rabaud, Kurt Konolige, Gary R. Bradski: ORB: An efficient alternative to SIFT or SURF. ICCV 2011: 2564-2571.

ORB::ORB
--------
The ORB constructor

.. ocv:function:: ORB::ORB(int nfeatures = 500, float scaleFactor = 1.2f, int nlevels = 8, int edgeThreshold = 31, int firstLevel = 0, int WTA_K=2, int scoreType=ORB::HARRIS_SCORE, int patchSize=31)

    :param nfeatures: The maximum number of features to retain.

    :param scaleFactor: Pyramid decimation ratio, greater than 1. ``scaleFactor==2`` means the classical pyramid, where each next level has 4x less pixels than the previous, but such a big scale factor will degrade feature matching scores dramatically. On the other hand, too close to 1 scale factor will mean that to cover certain scale range you will need more pyramid levels and so the speed will suffer.

    :param nlevels: The number of pyramid levels. The smallest level will have linear size equal to ``input_image_linear_size/pow(scaleFactor, nlevels)``.

    :param edgeThreshold: This is size of the border where the features are not detected. It should roughly match the ``patchSize`` parameter.

    :param firstLevel: It should be 0 in the current implementation.

    :param WTA_K: The number of points that produce each element of the oriented BRIEF descriptor. The default value 2 means the BRIEF where we take a random point pair and compare their brightnesses, so we get 0/1 response. Other possible values are 3 and 4. For example, 3 means that we take 3 random points (of course, those point coordinates are random, but they are generated from the pre-defined seed, so each element of BRIEF descriptor is computed deterministically from the pixel rectangle), find point of maximum brightness and output index of the winner (0, 1 or 2). Such output will occupy 2 bits, and therefore it will need a special variant of Hamming distance, denoted as ``NORM_HAMMING2`` (2 bits per bin).  When ``WTA_K=4``, we take 4 random points to compute each bin (that will also occupy 2 bits with possible values 0, 1, 2 or 3).

    :param scoreType: The default HARRIS_SCORE means that Harris algorithm is used to rank features (the score is written to ``KeyPoint::score`` and is used to retain best ``nfeatures`` features); FAST_SCORE is alternative value of the parameter that produces slightly less stable keypoints, but it is a little faster to compute.

    :param patchSize: size of the patch used by the oriented BRIEF descriptor. Of course, on smaller pyramid layers the perceived image area covered by a feature will be larger.

ORB::operator()
---------------
Finds keypoints in an image and computes their descriptors

.. ocv:function:: void ORB::operator()(InputArray image, InputArray mask, vector<KeyPoint>& keypoints, OutputArray descriptors, bool useProvidedKeypoints=false ) const

    :param image: The input 8-bit grayscale image.

    :param mask: The operation mask.

    :param keypoints: The output vector of keypoints.

    :param descriptors: The output descriptors. Pass ``cv::noArray()`` if you do not need it.

    :param useProvidedKeypoints: If it is true, then the method will use the provided vector of keypoints instead of detecting them.

BRISK
-----
.. ocv:class:: BRISK : public Feature2D

Class implementing the BRISK keypoint detector and descriptor extractor, described in [LCS11]_.

.. [LCS11] Stefan Leutenegger, Margarita Chli and Roland Siegwart: BRISK: Binary Robust Invariant Scalable Keypoints. ICCV 2011: 2548-2555.

BRISK::BRISK
------------
The BRISK constructor

.. ocv:function:: BRISK::BRISK(int thresh=30, int octaves=3, float patternScale=1.0f)

    :param thresh: FAST/AGAST detection threshold score.

    :param octaves: detection octaves. Use 0 to do single scale.

    :param patternScale: apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

BRISK::BRISK
------------
The BRISK constructor for a custom pattern

.. ocv:function:: BRISK::BRISK(std::vector<float> &radiusList, std::vector<int> &numberList, float dMax=5.85f, float dMin=8.2f, std::vector<int> indexChange=std::vector<int>())

    :param radiusList: defines the radii (in pixels) where the samples around a keypoint are taken (for keypoint scale 1).

    :param numberList: defines the number of sampling points on the sampling circle. Must be the same size as radiusList..

    :param dMax: threshold for the short pairings used for descriptor formation (in pixels for keypoint scale 1).

    :param dMin: threshold for the long pairings used for orientation determination (in pixels for keypoint scale 1).

    :param indexChanges: index remapping of the bits.

BRISK::operator()
-----------------
Finds keypoints in an image and computes their descriptors

.. ocv:function:: void BRISK::operator()(InputArray image, InputArray mask, vector<KeyPoint>& keypoints, OutputArray descriptors, bool useProvidedKeypoints=false ) const

    :param image: The input 8-bit grayscale image.

    :param mask: The operation mask.

    :param keypoints: The output vector of keypoints.

    :param descriptors: The output descriptors. Pass ``cv::noArray()`` if you do not need it.

    :param useProvidedKeypoints: If it is true, then the method will use the provided vector of keypoints instead of detecting them.

FREAK
-----
.. ocv:class:: FREAK : public DescriptorExtractor

Class implementing the FREAK (*Fast Retina Keypoint*) keypoint descriptor, described in [AOV12]_. The algorithm propose a novel keypoint descriptor inspired by the human visual system and more precisely the retina, coined Fast Retina Key- point (FREAK). A cascade of binary strings is computed by efficiently comparing image intensities over a retinal sampling pattern. FREAKs are in general faster to compute with lower memory load and also more robust than SIFT, SURF or BRISK. They are competitive alternatives to existing keypoints in particular for embedded applications.

.. [AOV12] A. Alahi, R. Ortiz, and P. Vandergheynst. FREAK: Fast Retina Keypoint. In IEEE Conference on Computer Vision and Pattern Recognition, 2012. CVPR 2012 Open Source Award Winner.

.. note::

   * An example on how to use the FREAK descriptor can be found at opencv_source_code/samples/cpp/freak_demo.cpp

FREAK::FREAK
------------
The FREAK constructor

.. ocv:function:: FREAK::FREAK( bool orientationNormalized=true, bool scaleNormalized=true, float patternScale=22.0f, int nOctaves=4, const vector<int>& selectedPairs=vector<int>() )

    :param orientationNormalized: Enable orientation normalization.
    :param scaleNormalized: Enable scale normalization.
    :param patternScale: Scaling of the description pattern.
    :param nOctaves: Number of octaves covered by the detected keypoints.
    :param selectedPairs: (Optional) user defined selected pairs indexes,

FREAK::selectPairs
------------------
Select the 512 best description pair indexes from an input (grayscale) image set. FREAK is available with a set of pairs learned off-line. Researchers can run a training process to learn their own set of pair. For more details read section 4.2 in: A. Alahi, R. Ortiz, and P. Vandergheynst. FREAK: Fast Retina Keypoint. In IEEE Conference on Computer Vision and Pattern Recognition, 2012.

We notice that for keypoint matching applications, image content has little effect on the selected pairs unless very specific what does matter is the detector type (blobs, corners,...) and the options used (scale/rotation invariance,...). Reduce corrThresh if not enough pairs are selected (43 points --> 903 possible pairs)

.. ocv:function:: vector<int> FREAK::selectPairs(const vector<Mat>& images, vector<vector<KeyPoint> >& keypoints, const double corrThresh = 0.7, bool verbose = true)

    :param images: Grayscale image input set.
    :param keypoints: Set of detected keypoints
    :param corrThresh: Correlation threshold.
    :param verbose: Prints pair selection informations.
