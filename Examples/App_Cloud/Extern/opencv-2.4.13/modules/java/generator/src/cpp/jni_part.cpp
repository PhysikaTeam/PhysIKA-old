#include "common.h"

#include "opencv2/opencv_modules.hpp"

#ifdef HAVE_OPENCV_NONFREE
#  include "opencv2/nonfree/nonfree.hpp"
#endif

#ifdef HAVE_OPENCV_FEATURES2D
#  include "opencv2/features2d/features2d.hpp"
#endif

#ifdef HAVE_OPENCV_VIDEO
#  include "opencv2/video/video.hpp"
#endif

#ifdef HAVE_OPENCV_ML
#  include "opencv2/ml/ml.hpp"
#endif

#ifdef HAVE_OPENCV_CONTRIB
#  include "opencv2/contrib/contrib.hpp"
#endif

extern "C" {

JNIEXPORT jint JNICALL
JNI_OnLoad(JavaVM* vm, void* )
{
    JNIEnv* env;
    if (vm->GetEnv((void**) &env, JNI_VERSION_1_6) != JNI_OK)
        return -1;

    bool init = true;
#ifdef HAVE_OPENCV_NONFREE
    init &= cv::initModule_nonfree();
#endif
#ifdef HAVE_OPENCV_FEATURES2D
    init &= cv::initModule_features2d();
#endif
#ifdef HAVE_OPENCV_VIDEO
    init &= cv::initModule_video();
#endif
#ifdef HAVE_OPENCV_ML
    init &= cv::initModule_ml();
#endif
    #ifdef HAVE_OPENCV_CONTRIB
    init &= cv::initModule_contrib();
#endif

    if(!init)
        return -1;

    /* get class with (*env)->FindClass */
    /* register methods with (*env)->RegisterNatives */

    return JNI_VERSION_1_6;
}

JNIEXPORT void JNICALL
JNI_OnUnload(JavaVM*, void*)
{
  //do nothing
}

} // extern "C"
