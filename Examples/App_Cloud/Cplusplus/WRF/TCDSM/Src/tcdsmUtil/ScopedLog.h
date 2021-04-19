#ifndef TCDSM_SCOPEDLOG_H
#define TCDSM_SCOPEDLOG_H

#include <tcdsmUtil/export.h>
#include <string>


//基础的log就是用easylogging++

namespace TCDSM {
    namespace Util {
        //0 = info
        //1 = warning
        //2 = debug
        //3 = fatal
        //4 = ERROR

        class TCDSM_UTIL_EXPORT ScopedLog {
        public:
            ScopedLog(const std::string& info, const int& level = 0);

            virtual ~ScopedLog();
        protected:

            int _level;
            std::string _info;
        };
    }
}
#endif //TCDSM_SCOPEDLOG_H
