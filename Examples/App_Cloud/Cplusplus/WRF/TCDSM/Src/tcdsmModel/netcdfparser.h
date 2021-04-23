/* 名称：netcdfset
 * 作者：李阳
 * 时间：2016年4月21日
 * 描述：对不同的nc数据做不同的操作，传输相同的数据到建模程序
 * 作用：
 * 1、获取nc文件的时间信息
 * 2、维度名称的转换 map<std::string,std::string> _dimMap  //从文件中获取
 * 3、获取nc文件中的维度信息  --主要指经纬海拔信息
 * 4、变量名称的转换 map<std::string,std::string> _varMap  //从文件中获取
 * 5、获取变量的维度信息 --维度信息
 * 6、获取用户所需的均匀网格数据（一二三维数据） --以后还可以对数据进行插值
 * 7、输出转换后的数据
 * 8、需要的获取数据的维度信息根据维度，变量等的多少进行分类的时候使用
 * 使用：
 * 1、创建对象
 * 2、如果需要的话，设置map<>或者从文件中获取
 * 3、使用
 */

#ifndef NETCDFOPERATOR_H
#define NETCDFOPERATOR_H

#include "tcdsmModel/export.h"
#include <netcdfcpp.h>
#include <vector>
#include <osg/Vec3d>
#include <osg/Image>

namespace TCDSM {
    namespace Model {

        typedef enum {
            WIND = 0x00,
            //变量
            QICE = 0x01,
            QVAPOR = 0x02,
            QCLOUD = 0x03,
            QSNOW = 0x04,
            QRAIN = 0x05,
            QGRAUP = 0x0A,
            PRESSURE = 0x06,
            TEMPERATURE = 0x07,
            DENSITY = 0x08,
            HUMIDITY = 0x09
        }Variable;

        class TCDSM_MODEL_EXPORT NetCDFParser
        {
        public:
            NetCDFParser() {}
            virtual ~NetCDFParser() {}
            virtual bool check(NcFile* ncfile) const = 0;
            virtual unsigned int getTimeNum(const NcFile* ncfile) const = 0;
            virtual time_t getTime(const NcFile* ncfile, const unsigned int& time) const = 0;
            virtual osg::Image* getCoordinate(const NcFile* ncfile, const unsigned int& time) const = 0;
            virtual osg::Image* getData(const NcFile* ncfile, const unsigned int& time, const Variable& var) const = 0;
        };

    }//end namespace Model

} //end of namespace TCDSM

#endif // NETCDFOPERATOR_H
