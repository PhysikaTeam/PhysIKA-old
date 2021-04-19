#include <tcdsmUtil/osgimageoperator.h>
#include <tcdsmUtil/ScopedLog.h>

#include <easylogging++.h>
#include <fstream>

#include <osgDB/WriteFile>
#include <osgDB/ReadFile>
#include <osgDB/ReaderWriter>

using namespace TCDSM::Util;

void osgImageOperator::getImageMaxAndMinValue(osg::Image *data, float &max, float &min)
{
    float min2 = FLT_MAX;
    float max2 = FLT_MIN;
    if(data->getDataType()==GL_FLOAT  && data->getPixelSizeInBits()==sizeof(float)*8)
    {
        const unsigned int imagesize = data->r()*data->t()*data->s();
        float *dataPoint = (float*)data->data();

        for(unsigned int i = 0; i < imagesize; ++i)
        {
            max2 = dataPoint[i] > max2 ? dataPoint[i]:max2;
            min2 = dataPoint[i] < min2 ? dataPoint[i]:min2;
        }
    }

    max = max2;
    min = min2;
    LOG(INFO) << "max:"<<max << "\tmin:" << min;
}

bool osgImageOperator::convertToImage(osg::Image *source, osg::Image *out,const osg::Vec3 &color)
{
    if(source == NULL ||!source->valid() ||source->getImageSizeInBytes()==0 ||
                source->getDataType() != GL_FLOAT || source->getPixelSizeInBits()!=sizeof(float)*8)
        {

            LOG(ERROR) << "输入图像内存地址："<< source ;
            LOG(ERROR) << "输入图像是否有效："<< (source->valid() ? "有效": "无效" );
            LOG(ERROR) << "输入图像分配空间："<< (source->getImageSizeInBytes() == 0 ?"为零":"不为零" );
            LOG(ERROR) << "输入图像像素格式："<< (source->getPixelFormat() != GL_FLOAT ? "为GL_FLOAT" :"不为GL_FLOAT" );
            LOG(ERROR) << "输入图像单像素大小："<< source ->getPixelSizeInBits() ;
            return false;
        }

        if(out == NULL || !out->valid() ||out->getImageSizeInBytes() == 0 ||
                out->getPixelFormat() != GL_RGB || out ->getPixelSizeInBits() != 24)
        {
            LOG(ERROR) << "输出图像内存地址："<< out ;
            LOG(ERROR) << "输出图像是否有效："<< (out->valid() ? "有效": "无效" );
            LOG(ERROR) << "输出图像分配空间："<< (out->getImageSizeInBytes() == 0 ?"为零":"不为零" );
            LOG(ERROR) << "输出图像像素格式："<< (out->getPixelFormat() != GL_RGB ? "为RGB" :"不为RGB" );
            LOG(ERROR) << "输出图像单像素大小："<< out ->getPixelSizeInBits() ;
            return false;
        }

        if(out->s() != source->s() || out->t()!= source->t())
        {
            LOG(ERROR) << "输出图像尺寸不一致";
            return false;
        }

        const unsigned int r = source->r();//厚度
        const unsigned int t = source->t();//高度
        const unsigned int s = source->s();//宽度

        osg::ref_ptr<osg::Image> image = new osg::Image;
        image->allocateImage(s,t,1,GL_RED,GL_FLOAT);
        float *tmpdata = (float*)image->getDataPointer();
        memset(tmpdata,0,s*t*sizeof(float));

        float *sourceData = (float*)source->getDataPointer();
        {
            TCDSM::Util::ScopedLog addDepthDataLog("将深度数据叠加");
            unsigned int indexs = 0;
            for(unsigned int k = 0; k < r; ++k)
            {
                unsigned int indexTmp = 0;
                for(unsigned int j = 0;j < t; ++j)
                {
                    for(unsigned int i = 0; i < s; ++i)
                    {

                        tmpdata[indexTmp++] += sourceData[indexs++];
                    }
                }
            }
        }


        float max,min;
        getImageMaxAndMinValue(image,max,min);
        float invLen = 1/(max-min);

        char *outdata = (char *)out->getDataPointer();

        {
            TCDSM::Util::ScopedLog addToColorLog("根据强度增加颜色");
            unsigned int indexTmp = 0;
            for(unsigned int j = 0;j < t; ++j)
            {
                for(unsigned int i = 0; i < s; ++i)
                {

//                    outdata[3*indexTmp + 0] = 255;
//                    outdata[3*indexTmp + 1] = 255;
//                    outdata[3*indexTmp + 2] = 255;
//                    if(tmpdata[indexTmp] > 0.0001)
//                    {
//                        osg::Vec3 tmp = /*outdata[indexTmp] +*/ color*(tmpdata[indexTmp]-min)*invLen * 255;
//                        outdata[3*indexTmp + 0] = (unsigned char)osg::clampTo(tmp.x() > 0 ? tmp.x():255,0.0f,255.0f);
//                        outdata[3*indexTmp + 1] = (unsigned char)osg::clampTo(tmp.y() > 0 ? tmp.y():255,0.0f,255.0f);
//                        outdata[3*indexTmp + 2] = (unsigned char)osg::clampTo(tmp.z() > 0 ? tmp.z():255,0.0f,255.0f);
//                    }

                    osg::Vec3 tmp = /*outdata[indexTmp] +*/ color*(tmpdata[indexTmp]-min)*invLen * 255;
                    outdata[3*indexTmp + 0] = (unsigned char)osg::clampTo(tmp.x() + outdata[3*indexTmp + 0],0.0f,255.0f);
                    outdata[3*indexTmp + 1] = (unsigned char)osg::clampTo(tmp.y() + outdata[3*indexTmp + 1],0.0f,255.0f);
                    outdata[3*indexTmp + 2] = (unsigned char)osg::clampTo(tmp.z() + outdata[3*indexTmp + 2],0.0f,255.0f);
                    indexTmp++;
                }
            }
        }
        return true;
}

void osgImageOperator::count(osg::Image *data, float gb, unsigned int &gbs, float lb, unsigned int &lbs)
{
    unsigned int x = 0;
    unsigned int y = 0;
    if(data->getDataType()==GL_FLOAT  && data->getPixelSizeInBits()==sizeof(float)*8)
    {
        const unsigned int imagesize = data->r()*data->s()*data->t();
        float *dataPoint = (float*)data->data();

        for(unsigned int i = 0; i < imagesize; ++i)
        {
            if(dataPoint[i] > gb) x++;
            if(dataPoint[i] < lb) y++;
        }
    }

    gbs = x;
    lbs = y;
}

bool osgImageOperator::saveAsImage(osg::Image *source,const char *filePath,const osg::Vec3 &color)
{
    osg::ref_ptr<osg::Image> image = new osg::Image;

    image->allocateImage(source->s(),source->t(),1,GL_RGB,GL_UNSIGNED_BYTE);
    memset(image->data(),0,image->getTotalSizeInBytes()/sizeof(char));

    TCDSM::Util::osgImageOperator::convertToImage(source,image,color);

    osgDB::writeImageFile(*image,filePath);
    return true;
}

bool osgImageOperator::saveAsImage(osg::Image *source, const char *filePath)
{
    osg::ref_ptr<osg::Image> image = new osg::Image;

    image->allocateImage(source->s(),source->t(),1,GL_RGB,GL_UNSIGNED_BYTE);
    memset(image->data(),0,image->getTotalSizeInBytes()/sizeof(char));

    if(source == NULL ||!source->valid() ||source->getImageSizeInBytes()==0 ||
                source->getDataType() != GL_FLOAT || source->getPixelSizeInBits()!=sizeof(float)*8*3)
    {

        LOG(ERROR) << "输入图像内存地址："<< source ;
        LOG(ERROR) << "输入图像是否有效："<< (source->valid() ? "有效": "无效" );
        LOG(ERROR) << "输入图像分配空间："<< (source->getImageSizeInBytes() == 0 ?"为零":"不为零" );
        LOG(ERROR) << "输入图像像素格式："<< (source->getPixelFormat() != GL_FLOAT ? "为GL_FLOAT" :"不为GL_FLOAT" );
        LOG(ERROR) << "输入图像单像素大小："<< source ->getPixelSizeInBits();
        return false;
    }

    const unsigned int r = source->r();//厚度
    const unsigned int t = source->t();//高度
    const unsigned int s = source->s();//宽度

    osg::ref_ptr<osg::Image> image2 = new osg::Image;
    image2->allocateImage(s,t,1,GL_RGB,GL_FLOAT);
    float *tmpdata = (float*)image2->getDataPointer();
    memset(tmpdata,0,s*t*sizeof(float)*3);

    osg::Vec3 *sourceData = (osg::Vec3 *)source->getDataPointer();
    osg::Vec3 *tmpdataV = (osg::Vec3 *)image2->getDataPointer();

    {
        TCDSM::Util::ScopedLog addDepthDataLog("将深度数据叠加");
        unsigned int indexs = 0;
        for(unsigned int k = 0; k < r; ++k)
        {
            unsigned int indexTmp = 0;
            for(unsigned int j = 0;j < t; ++j)
            {
                for(unsigned int i = 0; i < s; ++i)
                {

                    tmpdataV[indexTmp++] += sourceData[indexs++];
                }
            }
        }
    }

    float min_= FLT_MAX;
    float max_= FLT_MIN;
    for(unsigned int i = 0;i < s*r; ++i)
    {
        min_ = min_ > tmpdataV[i].x() ?tmpdataV[i].x():min_;
        min_ = min_ > tmpdataV[i].y() ?tmpdataV[i].y():min_;
        min_ = min_ > tmpdataV[i].z() ?tmpdataV[i].z():min_;

        max_ = max_ < tmpdataV[i].x() ?tmpdataV[i].x():max_;
        max_ = max_ < tmpdataV[i].y() ?tmpdataV[i].y():max_;
        max_ = max_ < tmpdataV[i].z() ?tmpdataV[i].z():max_;
    }

    float invLen = 255/(max_-min_);
    char *outdata = (char *)image->getDataPointer();
    {
        TCDSM::Util::ScopedLog addToColorLog("根据强度增加颜色");
        unsigned int indexTmp = 0;
        for(unsigned int j = 0;j < t; ++j)
        {
            for(unsigned int i = 0; i < s; ++i)
            {
                outdata[3*indexTmp + 0] = (unsigned char)osg::clampTo((tmpdataV[indexTmp].x() - min_)*invLen,0.0f,255.0f);
                outdata[3*indexTmp + 1] = (unsigned char)osg::clampTo((tmpdataV[indexTmp].y() - min_)*invLen,0.0f,255.0f);
                outdata[3*indexTmp + 2] = (unsigned char)osg::clampTo((tmpdataV[indexTmp].z() - min_)*invLen,0.0f,255.0f);
                indexTmp++;
            }
        }
    }
    osgDB::writeImageFile(*image,filePath);
    return true;
}

struct DataHead{
    unsigned int s; //Width
    unsigned int t; //Height
    unsigned int r; //Depth
    unsigned int datatype;
    unsigned int prixsize;
};

osg::Image *osgImageOperator::ReadImageData(const char *fileName)
{
    ScopedLog logs(std::string("read Image data from") + fileName);

    std::ifstream in;
    in.open(fileName,std::ios_base::in|std::ios_base::binary);

    if(!in.is_open())
    {
        LOG(ERROR) << fileName << "open error";
        return NULL;
    }

    struct DataHead head;
    in.read((char*)&head,sizeof(head));

    if(head.datatype != GL_FLOAT  &&
       head.datatype != GL_DOUBLE &&
       head.datatype != GL_INT    &&
       head.datatype != GL_UNSIGNED_INT )
    {
        LOG(ERROR)  << "image data data type is not support!";
        in.close();
        return NULL;
    }

    unsigned int prixSize = 0;
    switch (head.datatype) {
    case GL_FLOAT:
    case GL_INT:
    case GL_UNSIGNED_INT:
        prixSize = 32;
        break;
    default:
        prixSize = 64;
        break;
    }

    if ((head.prixsize%prixSize)) {
        LOG(ERROR)  << "image data prix size is not correct!";
        in.close();
        return NULL;
    }

    unsigned int prixFormat = 0;

    switch (head.prixsize/prixSize) {
    case 1:
        prixFormat = GL_RED;
        break;
    case 3:
        prixFormat = GL_RGB;
        break;
    case 4:
        prixFormat = GL_RGBA;
        break;
    default:
        break;
    }

    if(prixFormat == 0 || prixSize == 0)
    {
        LOG(ERROR)  << "image head data is not correct!";
        in.close();
        return NULL;
    }

    osg::Image *data = new osg::Image;
    data->allocateImage(head.s,head.t,head.r,prixFormat,head.datatype);

    in.read((char*)data->data(),data->getTotalSizeInBytes());

    in.close();
    return data;
}

bool osgImageOperator::WriteImageData(osg::Image *data, const char *fileName)
{
    ScopedLog logs(std::string("write Image data to") + fileName);

    std::ofstream out;
    out.open(fileName,std::ios_base::out|std::ios_base::binary);

    if(!out.is_open())
    {
        LOG(ERROR) << fileName << "open error";
        return false;
    }

    //only support GL_FLOAT GL_UNSIGNED_INT GL_INT GL_DOUBLE
    struct DataHead head;
    head.s = data->s();
    head.t = data->t();
    head.r = data->r();
    head.datatype = data->getDataType();
    head.prixsize = data->getPixelSizeInBits();

    if(head.datatype != GL_FLOAT  &&
       head.datatype != GL_DOUBLE &&
       head.datatype != GL_INT    &&
       head.datatype != GL_UNSIGNED_INT )
    {
        LOG(ERROR)  << "image data data type is not support!";
        return false;
    }

    out.write((char*)&head,sizeof(head));
    out.write((char*)data->data(),data->getTotalSizeInBytes());
    out.close();

    return true;
}
