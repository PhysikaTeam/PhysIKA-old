#ifndef TCDSM_UTIL_H
#define TCDSM_UTIL_H


#include <string>
#include <osg/Vec4>
#include <sstream>
#include <tcdsmUtil/export.h>

namespace TCDSM {
    namespace Util {

        template<typename T>
        TCDSM_UTIL_EXPORT const std::string toString(const T& value);

        //    bool mkdir(const std::string &pathName);
        //    bool mkpath(const std::string &pathName);
        TCDSM_UTIL_EXPORT bool save(const std::string& fileName, void* dataPoint, unsigned int size);

        TCDSM_UTIL_EXPORT void Clamp(osg::Vec4& color);


        TCDSM_UTIL_EXPORT const char* TimeToChar(const time_t& t);
        TCDSM_UTIL_EXPORT time_t CharToTime(const char* time_c);

        template<typename T1, typename T2>
        TCDSM_UTIL_EXPORT bool convent(T1* from, T2* to, const unsigned int& size)
        {
            if (NULL == from || NULL == to)
                return false;
            for (unsigned int i = 0; i < size; ++i)
                to[i] = (T2)from[i];
        }

    }
}


template<typename T>
const std::string TCDSM::Util::toString(const T &value)
{
    std::stringstream ss;
    ss << value;
    return ss.str();
}



#endif // UTIL_H
