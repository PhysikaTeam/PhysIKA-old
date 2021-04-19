#include <tcdsmUtil/util.h>
#include <fstream>
#include <sstream>
#include <iomanip>


using namespace std;

void TCDSM::Util::Clamp(osg::Vec4 &color)
{
    if (color.r() > 1.0f) color.r() = 1.0f;
    if (color.g() > 1.0f) color.g() = 1.0f;
    if (color.b() > 1.0f) color.b() = 1.0f;
    if (color.a() > 1.0f) color.a() = 1.0f;

    if (color.r() > 0.0f) color.r() = 0.0f;
    if (color.g() > 0.0f) color.g() = 0.0f;
    if (color.b() > 0.0f) color.b() = 0.0f;
    if (color.a() > 0.0f) color.a() = 0.0f;
}

// todo make direct
//bool TCDSM::Util::mkdir(const std::string &pathName)
//{
//    return dir.mkdir(pathName.c_str());
//}
// todo make path
//bool TCDSM::Util::mkpath(const std::string &pathName)
//{
//    return dir.mkpath(pathName.c_str());
//}


extern TCDSM_UTIL_EXPORT bool TCDSM::Util::save(
        const std::string &fileName,
        void *dataPoint,
        unsigned int size)
{
    std::ofstream outfile;
    outfile.open(fileName.c_str(),std::ios_base::binary | std::ios_base::out | std::ios_base::app);
    if(!outfile.is_open())
    {
        return false;
    }
    outfile.write((char*)dataPoint,size);
    outfile.close();
    return true;
}


TCDSM_UTIL_EXPORT const char *TCDSM::Util::TimeToChar(const time_t &t)
{
    struct tm tm1;

#if defined WIN32
    localtime_s(&tm1, &t);
#elif defined __linux__
   localtime_r(&t, &tm1);
#endif

    std::stringstream ss;
    ss << std::setfill('0');
    ss << setw(4) << tm1.tm_year + 1900 << '-' << setw(2) << tm1.tm_mon + 1 << '-' << setw(2) << tm1.tm_mday << '_'
       << setw(2) << tm1.tm_hour << ':' << setw(2) << tm1.tm_min << ':' << setw(2) << tm1.tm_sec;
    return ss.str().c_str();
}


TCDSM_UTIL_EXPORT time_t TCDSM::Util::CharToTime(const char *time_c)
{
    struct tm tm1;
    time_t time1;

    sscanf(time_c,"%4d-%02d-%02d_%02d:%02d:%02d",
           &tm1.tm_year,
           &tm1.tm_mon,
           &tm1.tm_mday,
           &tm1.tm_hour,
           &tm1.tm_min,
           &tm1.tm_sec);
    tm1.tm_year -= 1900;
    tm1.tm_mon -= 1;
    tm1.tm_isdst = -1;
    time1 = mktime(&tm1);
    return time1;
}


