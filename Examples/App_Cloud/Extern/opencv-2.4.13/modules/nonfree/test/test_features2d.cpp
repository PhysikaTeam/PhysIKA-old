/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "test_precomp.hpp"
#include "opencv2/calib3d/calib3d.hpp"

using namespace std;
using namespace cv;

const string FEATURES2D_DIR = "features2d";
const string DETECTOR_DIR = FEATURES2D_DIR + "/feature_detectors";
const string DESCRIPTOR_DIR = FEATURES2D_DIR + "/descriptor_extractors";
const string IMAGE_FILENAME = "tsukuba.png";

#if defined(HAVE_OPENCV_OCL) && 0 // unblock this to see SURF_OCL tests failures
static Ptr<Feature2D> getSURF()
{
    ocl::PlatformsInfo p;
    if(ocl::getOpenCLPlatforms(p) > 0)
        return new ocl::SURF_OCL;
    else
        return new SURF;
}
#else
static Ptr<Feature2D> getSURF()
{
    return new SURF;
}
#endif

/****************************************************************************************\
*            Regression tests for feature detectors comparing keypoints.                 *
\****************************************************************************************/

class CV_FeatureDetectorTest : public cvtest::BaseTest
{
public:
    CV_FeatureDetectorTest( const string& _name, const Ptr<FeatureDetector>& _fdetector ) :
        name(_name), fdetector(_fdetector) {}

protected:
    bool isSimilarKeypoints( const KeyPoint& p1, const KeyPoint& p2 );
    void compareKeypointSets( const vector<KeyPoint>& validKeypoints, const vector<KeyPoint>& calcKeypoints );

    void emptyDataTest();
    void regressionTest(); // TODO test of detect() with mask

    virtual void run( int );

    string name;
    Ptr<FeatureDetector> fdetector;
};

void CV_FeatureDetectorTest::emptyDataTest()
{
    // One image.
    Mat image;
    vector<KeyPoint> keypoints;
    try
    {
        fdetector->detect( image, keypoints );
    }
    catch(...)
    {
        ts->printf( cvtest::TS::LOG, "detect() on empty image must not generate exception (1).\n" );
        ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
    }

    if( !keypoints.empty() )
    {
        ts->printf( cvtest::TS::LOG, "detect() on empty image must return empty keypoints vector (1).\n" );
        ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
        return;
    }

    // Several images.
    vector<Mat> images;
    vector<vector<KeyPoint> > keypointCollection;
    try
    {
        fdetector->detect( images, keypointCollection );
    }
    catch(...)
    {
        ts->printf( cvtest::TS::LOG, "detect() on empty image vector must not generate exception (2).\n" );
        ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
    }
}

bool CV_FeatureDetectorTest::isSimilarKeypoints( const KeyPoint& p1, const KeyPoint& p2 )
{
    const float maxPtDif = 1.f;
    const float maxSizeDif = 1.f;
    const float maxAngleDif = 2.f;
    const float maxResponseDif = 0.1f;

    float dist = (float)norm( p1.pt - p2.pt );
    return (dist < maxPtDif &&
            fabs(p1.size - p2.size) < maxSizeDif &&
            abs(p1.angle - p2.angle) < maxAngleDif &&
            abs(p1.response - p2.response) < maxResponseDif &&
            p1.octave == p2.octave &&
            p1.class_id == p2.class_id );
}

void CV_FeatureDetectorTest::compareKeypointSets( const vector<KeyPoint>& validKeypoints, const vector<KeyPoint>& calcKeypoints )
{
    const float maxCountRatioDif = 0.01f;

    // Compare counts of validation and calculated keypoints.
    float countRatio = (float)validKeypoints.size() / (float)calcKeypoints.size();
    if( countRatio < 1 - maxCountRatioDif || countRatio > 1.f + maxCountRatioDif )
    {
        ts->printf( cvtest::TS::LOG, "Bad keypoints count ratio (validCount = %d, calcCount = %d).\n",
                    validKeypoints.size(), calcKeypoints.size() );
        ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
        return;
    }

    int progress = 0, progressCount = (int)(validKeypoints.size() * calcKeypoints.size());
    int badPointCount = 0, commonPointCount = max((int)validKeypoints.size(), (int)calcKeypoints.size());
    for( size_t v = 0; v < validKeypoints.size(); v++ )
    {
        int nearestIdx = -1;
        float minDist = std::numeric_limits<float>::max();

        for( size_t c = 0; c < calcKeypoints.size(); c++ )
        {
            progress = update_progress( progress, (int)(v*calcKeypoints.size() + c), progressCount, 0 );
            float curDist = (float)norm( calcKeypoints[c].pt - validKeypoints[v].pt );
            if( curDist < minDist )
            {
                minDist = curDist;
                nearestIdx = (int)c;
            }
        }

        assert( minDist >= 0 );
        if( !isSimilarKeypoints( validKeypoints[v], calcKeypoints[nearestIdx] ) )
            badPointCount++;
    }
    ts->printf( cvtest::TS::LOG, "badPointCount = %d; validPointCount = %d; calcPointCount = %d\n",
                badPointCount, validKeypoints.size(), calcKeypoints.size() );
    if( badPointCount > 0.9 * commonPointCount )
    {
        ts->printf( cvtest::TS::LOG, " - Bad accuracy!\n" );
        ts->set_failed_test_info( cvtest::TS::FAIL_BAD_ACCURACY );
        return;
    }
    ts->printf( cvtest::TS::LOG, " - OK\n" );
}

void CV_FeatureDetectorTest::regressionTest()
{
    assert( !fdetector.empty() );
    string imgFilename = string(ts->get_data_path()) + FEATURES2D_DIR + "/" + IMAGE_FILENAME;
    string resFilename = string(ts->get_data_path()) + DETECTOR_DIR + "/" + string(name) + ".xml.gz";

    // Read the test image.
    Mat image = imread( imgFilename );
    if( image.empty() )
    {
        ts->printf( cvtest::TS::LOG, "Image %s can not be read.\n", imgFilename.c_str() );
        ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );
        return;
    }

    FileStorage fs( resFilename, FileStorage::READ );

    // Compute keypoints.
    vector<KeyPoint> calcKeypoints;
    fdetector->detect( image, calcKeypoints );

    if( fs.isOpened() ) // Compare computed and valid keypoints.
    {
        // TODO compare saved feature detector params with current ones

        // Read validation keypoints set.
        vector<KeyPoint> validKeypoints;
        read( fs["keypoints"], validKeypoints );
        if( validKeypoints.empty() )
        {
            ts->printf( cvtest::TS::LOG, "Keypoints can not be read.\n" );
            ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );
            return;
        }

        compareKeypointSets( validKeypoints, calcKeypoints );
    }
    else // Write detector parameters and computed keypoints as validation data.
    {
        fs.open( resFilename, FileStorage::WRITE );
        if( !fs.isOpened() )
        {
            ts->printf( cvtest::TS::LOG, "File %s can not be opened to write.\n", resFilename.c_str() );
            ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );
            return;
        }
        else
        {
            fs << "detector_params" << "{";
            fdetector->write( fs );
            fs << "}";

            write( fs, "keypoints", calcKeypoints );
        }
    }
}

void CV_FeatureDetectorTest::run( int /*start_from*/ )
{
    if( fdetector.empty() )
    {
        ts->printf( cvtest::TS::LOG, "Feature detector is empty.\n" );
        ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );
        return;
    }

    emptyDataTest();
    regressionTest();

    ts->set_failed_test_info( cvtest::TS::OK );
}

/****************************************************************************************\
*                     Regression tests for descriptor extractors.                        *
\****************************************************************************************/
static void writeMatInBin( const Mat& mat, const string& filename )
{
    FILE* f = fopen( filename.c_str(), "wb");
    if( f )
    {
        int type = mat.type();
        fwrite( (void*)&mat.rows, sizeof(int), 1, f );
        fwrite( (void*)&mat.cols, sizeof(int), 1, f );
        fwrite( (void*)&type, sizeof(int), 1, f );
        int dataSize = (int)(mat.step * mat.rows * mat.channels());
        fwrite( (void*)&dataSize, sizeof(int), 1, f );
        fwrite( (void*)mat.data, 1, dataSize, f );
        fclose(f);
    }
}

static Mat readMatFromBin( const string& filename )
{
    FILE* f = fopen( filename.c_str(), "rb" );
    if( f )
    {
        int rows, cols, type, dataSize;
        size_t elements_read1 = fread( (void*)&rows, sizeof(int), 1, f );
        size_t elements_read2 = fread( (void*)&cols, sizeof(int), 1, f );
        size_t elements_read3 = fread( (void*)&type, sizeof(int), 1, f );
        size_t elements_read4 = fread( (void*)&dataSize, sizeof(int), 1, f );
        CV_Assert(elements_read1 == 1 && elements_read2 == 1 && elements_read3 == 1 && elements_read4 == 1);

        uchar* data = (uchar*)cvAlloc(dataSize);
        size_t elements_read = fread( (void*)data, 1, dataSize, f );
        CV_Assert(elements_read == (size_t)(dataSize));
        fclose(f);

        return Mat( rows, cols, type, data );
    }
    return Mat();
}

template<class Distance>
class CV_DescriptorExtractorTest : public cvtest::BaseTest
{
public:
    typedef typename Distance::ValueType ValueType;
    typedef typename Distance::ResultType DistanceType;

    CV_DescriptorExtractorTest( const string _name, DistanceType _maxDist, const Ptr<DescriptorExtractor>& _dextractor,
                                Distance d = Distance() ):
            name(_name), maxDist(_maxDist), dextractor(_dextractor), distance(d) {}
protected:
    virtual void createDescriptorExtractor() {}

    void compareDescriptors( const Mat& validDescriptors, const Mat& calcDescriptors )
    {
        if( validDescriptors.size != calcDescriptors.size || validDescriptors.type() != calcDescriptors.type() )
        {
            ts->printf(cvtest::TS::LOG, "Valid and computed descriptors matrices must have the same size and type.\n");
            ts->printf(cvtest::TS::LOG, "Valid size is (%d x %d) actual size is (%d x %d).\n", validDescriptors.rows, validDescriptors.cols, calcDescriptors.rows, calcDescriptors.cols);
            ts->printf(cvtest::TS::LOG, "Valid type is %d  actual type is %d.\n", validDescriptors.type(), calcDescriptors.type());
            ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );
            return;
        }

        CV_Assert( DataType<ValueType>::type == validDescriptors.type() );

        int dimension = validDescriptors.cols;
        DistanceType curMaxDist = std::numeric_limits<DistanceType>::min();
        for( int y = 0; y < validDescriptors.rows; y++ )
        {
            DistanceType dist = distance( validDescriptors.ptr<ValueType>(y), calcDescriptors.ptr<ValueType>(y), dimension );
            if( dist > curMaxDist )
                curMaxDist = dist;
        }

        stringstream ss;
        ss << "Max distance between valid and computed descriptors " << curMaxDist;
        if( curMaxDist < maxDist )
            ss << "." << endl;
        else
        {
            ss << ">" << maxDist  << " - bad accuracy!"<< endl;
            ts->set_failed_test_info( cvtest::TS::FAIL_BAD_ACCURACY );
        }
        ts->printf(cvtest::TS::LOG,  ss.str().c_str() );
    }

    void emptyDataTest()
    {
        assert( !dextractor.empty() );

        // One image.
        Mat image;
        vector<KeyPoint> keypoints;
        Mat descriptors;

        try
        {
            dextractor->compute( image, keypoints, descriptors );
        }
        catch(...)
        {
            ts->printf( cvtest::TS::LOG, "compute() on empty image and empty keypoints must not generate exception (1).\n");
            ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );
        }

        image.create( 50, 50, CV_8UC3 );
        try
        {
            dextractor->compute( image, keypoints, descriptors );
        }
        catch(...)
        {
            ts->printf( cvtest::TS::LOG, "compute() on nonempty image and empty keypoints must not generate exception (1).\n");
            ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );
        }

        // Several images.
        vector<Mat> images;
        vector<vector<KeyPoint> > keypointsCollection;
        vector<Mat> descriptorsCollection;
        try
        {
            dextractor->compute( images, keypointsCollection, descriptorsCollection );
        }
        catch(...)
        {
            ts->printf( cvtest::TS::LOG, "compute() on empty images and empty keypoints collection must not generate exception (2).\n");
            ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );
        }
    }

    void regressionTest()
    {
        assert( !dextractor.empty() );

        // Read the test image.
        string imgFilename =  string(ts->get_data_path()) + FEATURES2D_DIR + "/" + IMAGE_FILENAME;

        Mat img = imread( imgFilename );
        if( img.empty() )
        {
            ts->printf( cvtest::TS::LOG, "Image %s can not be read.\n", imgFilename.c_str() );
            ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );
            return;
        }

        vector<KeyPoint> keypoints;
        FileStorage fs( string(ts->get_data_path()) + FEATURES2D_DIR + "/keypoints.xml.gz", FileStorage::READ );
        if( fs.isOpened() )
        {
            read( fs.getFirstTopLevelNode(), keypoints );

            Mat calcDescriptors;
            double t = (double)getTickCount();
            dextractor->compute( img, keypoints, calcDescriptors );
            t = getTickCount() - t;
            ts->printf(cvtest::TS::LOG, "\nAverage time of computing one descriptor = %g ms.\n", t/((double)cvGetTickFrequency()*1000.)/calcDescriptors.rows );

            if( calcDescriptors.rows != (int)keypoints.size() )
            {
                ts->printf( cvtest::TS::LOG, "Count of computed descriptors and keypoints count must be equal.\n" );
                ts->printf( cvtest::TS::LOG, "Count of keypoints is            %d.\n", (int)keypoints.size() );
                ts->printf( cvtest::TS::LOG, "Count of computed descriptors is %d.\n", calcDescriptors.rows );
                ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
                return;
            }

            if( calcDescriptors.cols != dextractor->descriptorSize() || calcDescriptors.type() != dextractor->descriptorType() )
            {
                ts->printf( cvtest::TS::LOG, "Incorrect descriptor size or descriptor type.\n" );
                ts->printf( cvtest::TS::LOG, "Expected size is   %d.\n", dextractor->descriptorSize() );
                ts->printf( cvtest::TS::LOG, "Calculated size is %d.\n", calcDescriptors.cols );
                ts->printf( cvtest::TS::LOG, "Expected type is   %d.\n", dextractor->descriptorType() );
                ts->printf( cvtest::TS::LOG, "Calculated type is %d.\n", calcDescriptors.type() );
                ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
                return;
            }

            // TODO read and write descriptor extractor parameters and check them
            Mat validDescriptors = readDescriptors();
            if( !validDescriptors.empty() )
                compareDescriptors( validDescriptors, calcDescriptors );
            else
            {
                if( !writeDescriptors( calcDescriptors ) )
                {
                    ts->printf( cvtest::TS::LOG, "Descriptors can not be written.\n" );
                    ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );
                    return;
                }
            }
        }
        else
        {
            ts->printf( cvtest::TS::LOG, "Compute and write keypoints.\n" );
            fs.open( string(ts->get_data_path()) + FEATURES2D_DIR + "/keypoints.xml.gz", FileStorage::WRITE );
            if( fs.isOpened() )
            {
                SurfFeatureDetector fd;
                fd.detect(img, keypoints);
                write( fs, "keypoints", keypoints );
            }
            else
            {
                ts->printf(cvtest::TS::LOG, "File for writting keypoints can not be opened.\n");
                ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );
                return;
            }
        }
    }

    void run(int)
    {
        createDescriptorExtractor();
        if( dextractor.empty() )
        {
            ts->printf(cvtest::TS::LOG, "Descriptor extractor is empty.\n");
            ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );
            return;
        }

        emptyDataTest();
        regressionTest();

        ts->set_failed_test_info( cvtest::TS::OK );
    }

    virtual Mat readDescriptors()
    {
        Mat res = readMatFromBin( string(ts->get_data_path()) + DESCRIPTOR_DIR + "/" + string(name) );
        return res;
    }

    virtual bool writeDescriptors( Mat& descs )
    {
        writeMatInBin( descs,  string(ts->get_data_path()) + DESCRIPTOR_DIR + "/" + string(name) );
        return true;
    }

    string name;
    const DistanceType maxDist;
    Ptr<DescriptorExtractor> dextractor;
    Distance distance;

private:
    CV_DescriptorExtractorTest& operator=(const CV_DescriptorExtractorTest&) { return *this; }
};

/*template<typename T, typename Distance>
class CV_CalonderDescriptorExtractorTest : public CV_DescriptorExtractorTest<Distance>
{
public:
    CV_CalonderDescriptorExtractorTest( const char* testName, float _normDif, float _prevTime ) :
            CV_DescriptorExtractorTest<Distance>( testName, _normDif, Ptr<DescriptorExtractor>(), _prevTime )
    {}

protected:
    virtual void createDescriptorExtractor()
    {
        CV_DescriptorExtractorTest<Distance>::dextractor =
                new CalonderDescriptorExtractor<T>( string(CV_DescriptorExtractorTest<Distance>::ts->get_data_path()) +
                                                    FEATURES2D_DIR + "/calonder_classifier.rtc");
    }
};*/

/****************************************************************************************\
*                       Algorithmic tests for descriptor matchers                        *
\****************************************************************************************/
class CV_DescriptorMatcherTest : public cvtest::BaseTest
{
public:
    CV_DescriptorMatcherTest( const string& _name, const Ptr<DescriptorMatcher>& _dmatcher, float _badPart ) :
        badPart(_badPart), name(_name), dmatcher(_dmatcher)
        {}
protected:
    static const int dim = 500;
    static const int queryDescCount = 300; // must be even number because we split train data in some cases in two
    static const int countFactor = 4; // do not change it
    const float badPart;

    virtual void run( int );
    void generateData( Mat& query, Mat& train );

    void emptyDataTest();
    void matchTest( const Mat& query, const Mat& train );
    void knnMatchTest( const Mat& query, const Mat& train );
    void radiusMatchTest( const Mat& query, const Mat& train );

    string name;
    Ptr<DescriptorMatcher> dmatcher;

private:
    CV_DescriptorMatcherTest& operator=(const CV_DescriptorMatcherTest&) { return *this; }
};

void CV_DescriptorMatcherTest::emptyDataTest()
{
    assert( !dmatcher.empty() );
    Mat queryDescriptors, trainDescriptors, mask;
    vector<Mat> trainDescriptorCollection, masks;
    vector<DMatch> matches;
    vector<vector<DMatch> > vmatches;

    try
    {
        dmatcher->match( queryDescriptors, trainDescriptors, matches, mask );
    }
    catch(...)
    {
        ts->printf( cvtest::TS::LOG, "match() on empty descriptors must not generate exception (1).\n" );
        ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
    }

    try
    {
        dmatcher->knnMatch( queryDescriptors, trainDescriptors, vmatches, 2, mask );
    }
    catch(...)
    {
        ts->printf( cvtest::TS::LOG, "knnMatch() on empty descriptors must not generate exception (1).\n" );
        ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
    }

    try
    {
        dmatcher->radiusMatch( queryDescriptors, trainDescriptors, vmatches, 10.f, mask );
    }
    catch(...)
    {
        ts->printf( cvtest::TS::LOG, "radiusMatch() on empty descriptors must not generate exception (1).\n" );
        ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
    }

    try
    {
        dmatcher->add( trainDescriptorCollection );
    }
    catch(...)
    {
        ts->printf( cvtest::TS::LOG, "add() on empty descriptors must not generate exception.\n" );
        ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
    }

    try
    {
        dmatcher->match( queryDescriptors, matches, masks );
    }
    catch(...)
    {
        ts->printf( cvtest::TS::LOG, "match() on empty descriptors must not generate exception (2).\n" );
        ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
    }

    try
    {
        dmatcher->knnMatch( queryDescriptors, vmatches, 2, masks );
    }
    catch(...)
    {
        ts->printf( cvtest::TS::LOG, "knnMatch() on empty descriptors must not generate exception (2).\n" );
        ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
    }

    try
    {
        dmatcher->radiusMatch( queryDescriptors, vmatches, 10.f, masks );
    }
    catch(...)
    {
        ts->printf( cvtest::TS::LOG, "radiusMatch() on empty descriptors must not generate exception (2).\n" );
        ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
    }

}

void CV_DescriptorMatcherTest::generateData( Mat& query, Mat& train )
{
    RNG& rng = theRNG();

    // Generate query descriptors randomly.
    // Descriptor vector elements are integer values.
    Mat buf( queryDescCount, dim, CV_32SC1 );
    rng.fill( buf, RNG::UNIFORM, Scalar::all(0), Scalar(3) );
    buf.convertTo( query, CV_32FC1 );

    // Generate train decriptors as follows:
    // copy each query descriptor to train set countFactor times
    // and perturb some one element of the copied descriptors in
    // in ascending order. General boundaries of the perturbation
    // are (0.f, 1.f).
    train.create( query.rows*countFactor, query.cols, CV_32FC1 );
    float step = 1.f / countFactor;
    for( int qIdx = 0; qIdx < query.rows; qIdx++ )
    {
        Mat queryDescriptor = query.row(qIdx);
        for( int c = 0; c < countFactor; c++ )
        {
            int tIdx = qIdx * countFactor + c;
            Mat trainDescriptor = train.row(tIdx);
            queryDescriptor.copyTo( trainDescriptor );
            int elem = rng(dim);
            float diff = rng.uniform( step*c, step*(c+1) );
            trainDescriptor.at<float>(0, elem) += diff;
        }
    }
}

void CV_DescriptorMatcherTest::matchTest( const Mat& query, const Mat& train )
{
    dmatcher->clear();

    // test const version of match()
    {
        vector<DMatch> matches;
        dmatcher->match( query, train, matches );

        if( (int)matches.size() != queryDescCount )
        {
            ts->printf(cvtest::TS::LOG, "Incorrect matches count while test match() function (1).\n");
            ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
        }
        else
        {
            int badCount = 0;
            for( size_t i = 0; i < matches.size(); i++ )
            {
                DMatch match = matches[i];
                if( (match.queryIdx != (int)i) || (match.trainIdx != (int)i*countFactor) || (match.imgIdx != 0) )
                    badCount++;
            }
            if( (float)badCount > (float)queryDescCount*badPart )
            {
                ts->printf( cvtest::TS::LOG, "%f - too large bad matches part while test match() function (1).\n",
                            (float)badCount/(float)queryDescCount );
                ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
            }
        }
    }

    // test version of match() with add()
    {
        vector<DMatch> matches;
        // make add() twice to test such case
        dmatcher->add( vector<Mat>(1,train.rowRange(0, train.rows/2)) );
        dmatcher->add( vector<Mat>(1,train.rowRange(train.rows/2, train.rows)) );
        // prepare masks (make first nearest match illegal)
        vector<Mat> masks(2);
        for(int mi = 0; mi < 2; mi++ )
        {
            masks[mi] = Mat(query.rows, train.rows/2, CV_8UC1, Scalar::all(1));
            for( int di = 0; di < queryDescCount/2; di++ )
                masks[mi].col(di*countFactor).setTo(Scalar::all(0));
        }

        dmatcher->match( query, matches, masks );

        if( (int)matches.size() != queryDescCount )
        {
            ts->printf(cvtest::TS::LOG, "Incorrect matches count while test match() function (2).\n");
            ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
        }
        else
        {
            int badCount = 0;
            for( size_t i = 0; i < matches.size(); i++ )
            {
                DMatch match = matches[i];
                int shift = dmatcher->isMaskSupported() ? 1 : 0;
                {
                    if( i < queryDescCount/2 )
                    {
                        if( (match.queryIdx != (int)i) || (match.trainIdx != (int)i*countFactor + shift) || (match.imgIdx != 0) )
                            badCount++;
                    }
                    else
                    {
                        if( (match.queryIdx != (int)i) || (match.trainIdx != ((int)i-queryDescCount/2)*countFactor + shift) || (match.imgIdx != 1) )
                            badCount++;
                    }
                }
            }
            if( (float)badCount > (float)queryDescCount*badPart )
            {
                ts->printf( cvtest::TS::LOG, "%f - too large bad matches part while test match() function (2).\n",
                            (float)badCount/(float)queryDescCount );
                ts->set_failed_test_info( cvtest::TS::FAIL_BAD_ACCURACY );
            }
        }
    }
}

void CV_DescriptorMatcherTest::knnMatchTest( const Mat& query, const Mat& train )
{
    dmatcher->clear();

    // test const version of knnMatch()
    {
        const int knn = 3;

        vector<vector<DMatch> > matches;
        dmatcher->knnMatch( query, train, matches, knn );

        if( (int)matches.size() != queryDescCount )
        {
            ts->printf(cvtest::TS::LOG, "Incorrect matches count while test knnMatch() function (1).\n");
            ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
        }
        else
        {
            int badCount = 0;
            for( size_t i = 0; i < matches.size(); i++ )
            {
                if( (int)matches[i].size() != knn )
                    badCount++;
                else
                {
                    int localBadCount = 0;
                    for( int k = 0; k < knn; k++ )
                    {
                        DMatch match = matches[i][k];
                        if( (match.queryIdx != (int)i) || (match.trainIdx != (int)i*countFactor+k) || (match.imgIdx != 0) )
                            localBadCount++;
                    }
                    badCount += localBadCount > 0 ? 1 : 0;
                }
            }
            if( (float)badCount > (float)queryDescCount*badPart )
            {
                ts->printf( cvtest::TS::LOG, "%f - too large bad matches part while test knnMatch() function (1).\n",
                            (float)badCount/(float)queryDescCount );
                ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
            }
        }
    }

    // test version of knnMatch() with add()
    {
        const int knn = 2;
        vector<vector<DMatch> > matches;
        // make add() twice to test such case
        dmatcher->add( vector<Mat>(1,train.rowRange(0, train.rows/2)) );
        dmatcher->add( vector<Mat>(1,train.rowRange(train.rows/2, train.rows)) );
        // prepare masks (make first nearest match illegal)
        vector<Mat> masks(2);
        for(int mi = 0; mi < 2; mi++ )
        {
            masks[mi] = Mat(query.rows, train.rows/2, CV_8UC1, Scalar::all(1));
            for( int di = 0; di < queryDescCount/2; di++ )
                masks[mi].col(di*countFactor).setTo(Scalar::all(0));
        }

        dmatcher->knnMatch( query, matches, knn, masks );

        if( (int)matches.size() != queryDescCount )
        {
            ts->printf(cvtest::TS::LOG, "Incorrect matches count while test knnMatch() function (2).\n");
            ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
        }
        else
        {
            int badCount = 0;
            int shift = dmatcher->isMaskSupported() ? 1 : 0;
            for( size_t i = 0; i < matches.size(); i++ )
            {
                if( (int)matches[i].size() != knn )
                    badCount++;
                else
                {
                    int localBadCount = 0;
                    for( int k = 0; k < knn; k++ )
                    {
                        DMatch match = matches[i][k];
                        {
                            if( i < queryDescCount/2 )
                            {
                                if( (match.queryIdx != (int)i) || (match.trainIdx != (int)i*countFactor + k + shift) ||
                                    (match.imgIdx != 0) )
                                    localBadCount++;
                            }
                            else
                            {
                                if( (match.queryIdx != (int)i) || (match.trainIdx != ((int)i-queryDescCount/2)*countFactor + k + shift) ||
                                    (match.imgIdx != 1) )
                                    localBadCount++;
                            }
                        }
                    }
                    badCount += localBadCount > 0 ? 1 : 0;
                }
            }
            if( (float)badCount > (float)queryDescCount*badPart )
            {
                ts->printf( cvtest::TS::LOG, "%f - too large bad matches part while test knnMatch() function (2).\n",
                            (float)badCount/(float)queryDescCount );
                ts->set_failed_test_info( cvtest::TS::FAIL_BAD_ACCURACY );
            }
        }
    }
}

void CV_DescriptorMatcherTest::radiusMatchTest( const Mat& query, const Mat& train )
{
    dmatcher->clear();
    // test const version of match()
    {
        const float radius = 1.f/countFactor;
        vector<vector<DMatch> > matches;
        dmatcher->radiusMatch( query, train, matches, radius );

        if( (int)matches.size() != queryDescCount )
        {
            ts->printf(cvtest::TS::LOG, "Incorrect matches count while test radiusMatch() function (1).\n");
            ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
        }
        else
        {
            int badCount = 0;
            for( size_t i = 0; i < matches.size(); i++ )
            {
                if( (int)matches[i].size() != 1 )
                    badCount++;
                else
                {
                    DMatch match = matches[i][0];
                    if( (match.queryIdx != (int)i) || (match.trainIdx != (int)i*countFactor) || (match.imgIdx != 0) )
                        badCount++;
                }
            }
            if( (float)badCount > (float)queryDescCount*badPart )
            {
                ts->printf( cvtest::TS::LOG, "%f - too large bad matches part while test radiusMatch() function (1).\n",
                            (float)badCount/(float)queryDescCount );
                ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
            }
        }
    }

    // test version of match() with add()
    {
        int n = 3;
        const float radius = 1.f/countFactor * n;
        vector<vector<DMatch> > matches;
        // make add() twice to test such case
        dmatcher->add( vector<Mat>(1,train.rowRange(0, train.rows/2)) );
        dmatcher->add( vector<Mat>(1,train.rowRange(train.rows/2, train.rows)) );
        // prepare masks (make first nearest match illegal)
        vector<Mat> masks(2);
        for(int mi = 0; mi < 2; mi++ )
        {
            masks[mi] = Mat(query.rows, train.rows/2, CV_8UC1, Scalar::all(1));
            for( int di = 0; di < queryDescCount/2; di++ )
                masks[mi].col(di*countFactor).setTo(Scalar::all(0));
        }

        dmatcher->radiusMatch( query, matches, radius, masks );

        //int curRes = cvtest::TS::OK;
        if( (int)matches.size() != queryDescCount )
        {
            ts->printf(cvtest::TS::LOG, "Incorrect matches count while test radiusMatch() function (1).\n");
            ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
        }

        int badCount = 0;
        int shift = dmatcher->isMaskSupported() ? 1 : 0;
        int needMatchCount = dmatcher->isMaskSupported() ? n-1 : n;
        for( size_t i = 0; i < matches.size(); i++ )
        {
            if( (int)matches[i].size() != needMatchCount )
                badCount++;
            else
            {
                int localBadCount = 0;
                for( int k = 0; k < needMatchCount; k++ )
                {
                    DMatch match = matches[i][k];
                    {
                        if( i < queryDescCount/2 )
                        {
                            if( (match.queryIdx != (int)i) || (match.trainIdx != (int)i*countFactor + k + shift) ||
                                (match.imgIdx != 0) )
                                localBadCount++;
                        }
                        else
                        {
                            if( (match.queryIdx != (int)i) || (match.trainIdx != ((int)i-queryDescCount/2)*countFactor + k + shift) ||
                                (match.imgIdx != 1) )
                                localBadCount++;
                        }
                    }
                }
                badCount += localBadCount > 0 ? 1 : 0;
            }
        }
        if( (float)badCount > (float)queryDescCount*badPart )
        {
            ts->printf( cvtest::TS::LOG, "%f - too large bad matches part while test radiusMatch() function (2).\n",
                        (float)badCount/(float)queryDescCount );
            ts->set_failed_test_info( cvtest::TS::FAIL_BAD_ACCURACY );
        }
    }
}

void CV_DescriptorMatcherTest::run( int )
{
    Mat query, train;
    generateData( query, train );

    matchTest( query, train );

    knnMatchTest( query, train );

    radiusMatchTest( query, train );
}

/****************************************************************************************\
*                                Tests registrations                                     *
\****************************************************************************************/

/*
 * Detectors
 */


TEST( Features2d_Detector_SIFT, regression )
{
    CV_FeatureDetectorTest test( "detector-sift", FeatureDetector::create("SIFT") );
    test.safe_run();
}

TEST( Features2d_Detector_SURF, regression )
{
    CV_FeatureDetectorTest test( "detector-surf", Ptr<FeatureDetector>(getSURF()) );
    test.safe_run();
}

/*
 * Descriptors
 */

#ifdef __aarch64__
// The discrepancy is 1, but 0.03 is allowed
TEST( Features2d_DescriptorExtractor_SIFT, DISABLED_regression )
#else
TEST( Features2d_DescriptorExtractor_SIFT, regression )
#endif
{
    CV_DescriptorExtractorTest<L2<float> > test( "descriptor-sift", 0.03f,
                                                  DescriptorExtractor::create("SIFT") );
    test.safe_run();
}

TEST( Features2d_DescriptorExtractor_SURF, regression )
{
    CV_DescriptorExtractorTest<L2<float> > test( "descriptor-surf",  0.05f,
                                                 Ptr<DescriptorExtractor>(getSURF()) );
    test.safe_run();
}

#ifdef __aarch64__
// The discrepancy is 1, but 0.18 is allowed
TEST( Features2d_DescriptorExtractor_OpponentSIFT, DISABLED_regression )
#else
TEST( Features2d_DescriptorExtractor_OpponentSIFT, regression )
#endif
{
    CV_DescriptorExtractorTest<L2<float> > test( "descriptor-opponent-sift", 0.18f,
                                                 DescriptorExtractor::create("OpponentSIFT") );
    test.safe_run();
}

TEST( Features2d_DescriptorExtractor_OpponentSURF, regression )
{
    CV_DescriptorExtractorTest<L2<float> > test( "descriptor-opponent-surf",  0.3f,
                                                 DescriptorExtractor::create("OpponentSURF") );
    test.safe_run();
}

/*#if CV_SSE2
TEST( Features2d_DescriptorExtractor_Calonder_uchar, regression )
{
    CV_CalonderDescriptorExtractorTest<uchar, L2<uchar> > test( "descriptor-calonder-uchar",
                                                                std::numeric_limits<float>::epsilon() + 1,
                                                                0.0132175f );
    test.safe_run();
}

TEST( Features2d_DescriptorExtractor_Calonder_float, regression )
{
    CV_CalonderDescriptorExtractorTest<float, L2<float> > test( "descriptor-calonder-float",
                                                                std::numeric_limits<float>::epsilon(),
                                                                0.0221308f );
    test.safe_run();
}
#endif*/ // CV_SSE2

TEST(Features2d_BruteForceDescriptorMatcher_knnMatch, regression)
{
    const int sz = 100;
    const int k = 3;

    Ptr<DescriptorExtractor> ext = Ptr<DescriptorExtractor>(getSURF());
    ASSERT_TRUE(ext != NULL);

    Ptr<FeatureDetector> det = Ptr<FeatureDetector>(getSURF());
    //"%YAML:1.0\nhessianThreshold: 8000.\noctaves: 3\noctaveLayers: 4\nupright: 0\n"
    ASSERT_TRUE(det != NULL);

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");
    ASSERT_TRUE(matcher != NULL);

    Mat imgT(sz, sz, CV_8U, Scalar(255));
    line(imgT, Point(20, sz/2), Point(sz-21, sz/2), Scalar(100), 2);
    line(imgT, Point(sz/2, 20), Point(sz/2, sz-21), Scalar(100), 2);
    vector<KeyPoint> kpT;
    kpT.push_back( KeyPoint(50, 50, 16, 0, 20000, 1, -1) );
    kpT.push_back( KeyPoint(42, 42, 16, 160, 10000, 1, -1) );
    Mat descT;
    ext->compute(imgT, kpT, descT);

    Mat imgQ(sz, sz, CV_8U, Scalar(255));
    line(imgQ, Point(30, sz/2), Point(sz-31, sz/2), Scalar(100), 3);
    line(imgQ, Point(sz/2, 30), Point(sz/2, sz-31), Scalar(100), 3);
    vector<KeyPoint> kpQ;
    det->detect(imgQ, kpQ);
    Mat descQ;
    ext->compute(imgQ, kpQ, descQ);

    vector<vector<DMatch> > matches;

    matcher->knnMatch(descQ, descT, matches, k);

    //cout << "\nBest " << k << " matches to " << descT.rows << " train desc-s." << endl;
    ASSERT_EQ(descQ.rows, static_cast<int>(matches.size()));
    for(size_t i = 0; i<matches.size(); i++)
    {
        //cout << "\nmatches[" << i << "].size()==" << matches[i].size() << endl;
        ASSERT_GE(min(k, descT.rows), static_cast<int>(matches[i].size()));
        for(size_t j = 0; j<matches[i].size(); j++)
        {
            //cout << "\t" << matches[i][j].queryIdx << " -> " << matches[i][j].trainIdx << endl;
            ASSERT_EQ(matches[i][j].queryIdx, static_cast<int>(i));
        }
    }
}

/*TEST(Features2d_DescriptorExtractorParamTest, regression)
{
    Ptr<DescriptorExtractor> s = DescriptorExtractor::create("SURF");
    ASSERT_STREQ(s->paramHelp("extended").c_str(), "");
}
*/

class CV_DetectPlanarTest : public cvtest::BaseTest
{
public:
    CV_DetectPlanarTest(const string& _fname, int _min_ninliers) : fname(_fname), min_ninliers(_min_ninliers) {}

protected:
    void run(int)
    {
        Ptr<Feature2D> f;
        if(fname == "SURF")
            f = getSURF();
        else
            f = Algorithm::create<Feature2D>("Feature2D." + fname);
        if(f.empty())
            return;
        string path = string(ts->get_data_path()) + "detectors_descriptors_evaluation/planar/";
        string imgname1 = path + "box.png";
        string imgname2 = path + "box_in_scene.png";
        Mat img1 = imread(imgname1, 0);
        Mat img2 = imread(imgname2, 0);
        if( img1.empty() || img2.empty() )
        {
            ts->printf( cvtest::TS::LOG, "missing %s and/or %s\n", imgname1.c_str(), imgname2.c_str());
            ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );
            return;
        }
        vector<KeyPoint> kpt1, kpt2;
        Mat d1, d2;
        f->operator()(img1, Mat(), kpt1, d1);
        f->operator()(img1, Mat(), kpt2, d2);
        for( size_t i = 0; i < kpt1.size(); i++ )
            CV_Assert(kpt1[i].response > 0 );
        for( size_t i = 0; i < kpt2.size(); i++ )
            CV_Assert(kpt2[i].response > 0 );

        vector<DMatch> matches;
        BFMatcher(NORM_L2, true).match(d1, d2, matches);

        vector<Point2f> pt1, pt2;
        for( size_t i = 0; i < matches.size(); i++ ) {
            pt1.push_back(kpt1[matches[i].queryIdx].pt);
            pt2.push_back(kpt2[matches[i].trainIdx].pt);
        }

        Mat inliers, H = findHomography(pt1, pt2, RANSAC, 10, inliers);
        int ninliers = countNonZero(inliers);

        if( ninliers < min_ninliers )
        {
            ts->printf( cvtest::TS::LOG, "too little inliers (%d) vs expected %d\n", ninliers, min_ninliers);
            ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );
            return;
        }
    }

    string fname;
    int min_ninliers;
};

TEST(Features2d_SIFTHomographyTest, regression) { CV_DetectPlanarTest test("SIFT", 80); test.safe_run(); }
TEST(Features2d_SURFHomographyTest, regression) { CV_DetectPlanarTest test("SURF", 80); test.safe_run(); }

class FeatureDetectorUsingMaskTest : public cvtest::BaseTest
{
public:
    FeatureDetectorUsingMaskTest(const Ptr<FeatureDetector>& featureDetector) :
        featureDetector_(featureDetector)
    {
        CV_Assert(!featureDetector_.empty());
    }

protected:

    void run(int)
    {
        const int nStepX = 2;
        const int nStepY = 2;

        const string imageFilename = string(ts->get_data_path()) + "/features2d/tsukuba.png";

        Mat image = imread(imageFilename);
        if(image.empty())
        {
            ts->printf(cvtest::TS::LOG, "Image %s can not be read.\n", imageFilename.c_str());
            ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
            return;
        }

        Mat mask(image.size(), CV_8U);

        const int stepX = image.size().width / nStepX;
        const int stepY = image.size().height / nStepY;

        vector<KeyPoint> keyPoints;
        vector<Point2f> points;
        for(int i=0; i<nStepX; ++i)
            for(int j=0; j<nStepY; ++j)
            {

                mask.setTo(0);
                Rect whiteArea(i * stepX, j * stepY, stepX, stepY);
                mask(whiteArea).setTo(255);

                featureDetector_->detect(image, keyPoints, mask);
                KeyPoint::convert(keyPoints, points);

                for(size_t k=0; k<points.size(); ++k)
                {
                    if ( !whiteArea.contains(points[k]) )
                    {
                        ts->printf(cvtest::TS::LOG, "The feature point is outside of the mask.");
                        ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
                        return;
                    }
                }
            }

        ts->set_failed_test_info( cvtest::TS::OK );
    }

    Ptr<FeatureDetector> featureDetector_;
};

TEST(Features2d_SIFT_using_mask, regression)
{
    FeatureDetectorUsingMaskTest test(Algorithm::create<FeatureDetector>("Feature2D.SIFT"));
    test.safe_run();
}

TEST(DISABLED_Features2d_SURF_using_mask, regression)
{
    FeatureDetectorUsingMaskTest test(Algorithm::create<FeatureDetector>("Feature2D.SURF"));
    test.safe_run();
}
