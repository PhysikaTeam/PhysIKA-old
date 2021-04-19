#ifndef TCDSM_MODEL_CONFIG_H
#define TCDSM_MODEL_CONFIG_H

#include <tcdsmModel/export.h>

#include <float.h>
#include <osg/BoundingBox>
#include <osg/BoundingSphere>

#if defined(MODEL_USE_DOUBLE)
#include <osg/Vec4d>
#include <osg/Vec3d>
#include <osg/Vec2d>
#include <osg/Array>

typedef double real;
#define REAL_MAX DBL_MAX
#define REAL_MIN DBL_MIN

namespace tcdsmModel {
typedef osg::Vec2d           vec2           ;
typedef osg::Vec3d           vec3           ;
typedef osg::Vec4d           vec4           ;
typedef osg::DoubleArray     vec1Array      ;
typedef osg::Vec2dArray      vec2Array      ;
typedef osg::Vec3dArray      vec3Array      ;
typedef osg::Vec4dArray      vec4Array      ;
typedef osg::BoundingBoxd    boundingBox    ;
typedef osg::BoundingSphered boundingSphere ;
}
#else

#include <osg/Vec2f>
#include <osg/Vec3f>
#include <osg/Vec4f>
#include <osg/Array>

typedef float real;
#define REAL_MAX FLT_MAX
#define REAL_MIN FLT_MIN

namespace TCDSM {
    namespace Model {

        typedef osg::Vec2f           vec2;
        typedef osg::Vec3f           vec3;
        typedef osg::Vec4f           vec4;
        typedef osg::FloatArray      realArray;
        typedef osg::UIntArray       uintArray;
        typedef osg::Vec2Array       vec2Array;
        typedef osg::Vec3Array       vec3Array;
        typedef osg::Vec4Array       vec4Array;
        typedef osg::BoundingBoxf    boundingBox;
        typedef osg::BoundingSpheref boundingSphere;
    }
}
#endif



#endif // TCDSM_MODEL_CONFIG_H
