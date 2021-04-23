#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;
using std::tr1::make_tuple;
using std::tr1::get;

typedef perf::TestBaseWithParam<std::string> surf;

#define SURF_IMAGES \
    "cv/detectors_descriptors_evaluation/images_datasets/leuven/img1.png",\
    "stitching/a3.png"

PERF_TEST_P(surf, detect, testing::Values(SURF_IMAGES))
{
    String filename = getDataPath(GetParam());
    Mat frame = imread(filename, IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame.empty()) << "Unable to load source image " << filename;

    Mat mask;
    declare.in(frame).time(90);
    SURF detector;
    vector<KeyPoint> points;

    TEST_CYCLE() detector(frame, mask, points);

    SANITY_CHECK_KEYPOINTS(points, 1e-3);
}

PERF_TEST_P(surf, extract, testing::Values(SURF_IMAGES))
{
    String filename = getDataPath(GetParam());
    Mat frame = imread(filename, IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame.empty()) << "Unable to load source image " << filename;

    Mat mask;
    declare.in(frame).time(90);

    SURF detector;
    vector<KeyPoint> points;
    vector<float> descriptors;
    detector(frame, mask, points);

    TEST_CYCLE() detector(frame, mask, points, descriptors, true);

    SANITY_CHECK(descriptors, 1e-4);
}

PERF_TEST_P(surf, full, testing::Values(SURF_IMAGES))
{
    String filename = getDataPath(GetParam());
    Mat frame = imread(filename, IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame.empty()) << "Unable to load source image " << filename;

    Mat mask;
    declare.in(frame).time(90);
    SURF detector;
    vector<KeyPoint> points;
    vector<float> descriptors;

    TEST_CYCLE() detector(frame, mask, points, descriptors, false);

    SANITY_CHECK_KEYPOINTS(points, 1e-3);
    SANITY_CHECK(descriptors, 1e-4);
}
