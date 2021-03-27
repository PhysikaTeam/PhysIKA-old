#ifndef TCDSM_UTIL_OSGIMAGEOPERATOR_H
#define TCDSM_UTIL_OSGIMAGEOPERATOR_H

#include <osg/Image>

namespace TCDSM{
    namespace Util {
        class TCDSM_UTIL_EXPORT osgImageOperator
        {
        public:
            //计算一维单浮点数的最大值和最小值
            static void getImageMaxAndMinValue(osg::Image *data, float &max, float &min);

            static bool convertToImage(osg::Image *source, osg::Image *out,const osg::Vec3 &color);

            static void count(osg::Image *data, float gb, unsigned int &gbs, float lb, unsigned int &lbs);

            static bool saveAsImage(osg::Image *source,const char *filePath,const osg::Vec3 &color);

            static bool saveAsImage(osg::Image *source,const char *filePath);

            static osg::Image *ReadImageData(const char *fileName);

            static bool WriteImageData(osg::Image *data, const char *fileName);
        };
    }
}
#endif // TCDSM_UTIL_OSGIMAGEOPERATOR_H
