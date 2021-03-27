#ifndef DESCRIBENETCDFOPERATOR_H
#define DESCRIBENETCDFOPERATOR_H

#include <tcdsmModel/netcdfparser.h>


namespace osg{class Image;}
namespace TCDSM {
    namespace Model {

        //只是对变量的映射和数据的转换,不牵涉到数据
        typedef enum {
            GRID = 0x30,
            UGRID = 0x31,
            VGRID = 0x32,
            WGRID = 0x33
        } GridType;

        typedef enum {
            LONGITUDE = 0x21,
            LATITUDE = 0x22,
            ALTITUDE = 0x24,
            TIMEDIM = 0x20
        } DimType;

        class TCDSM_MODEL_EXPORT WRFNetCDFOperator :public NetCDFParser
        {
        public:
            virtual bool check(NcFile* ncfile) const;

            virtual unsigned int getTimeNum(const NcFile* ncfile) const;
            virtual time_t getTime(const NcFile* ncfile, const unsigned int& time) const;
            virtual osg::Image* getCoordinate(const NcFile* ncfile, const unsigned int& time) const;
            virtual osg::Image* getData(const NcFile* ncfile, const unsigned int& time, const Variable& name) const;

            osg::Image* getCoordinate(const NcFile* ncfile, const unsigned int& time, const GridType& gridType) const;
        protected:

            osg::Image* getWind(const NcFile* ncfile, const unsigned int& time) const;
            bool haveDim(const NcFile* ncfile, const unsigned int& name) const;
            bool haveVar(const NcFile* ncfile, const unsigned int& name) const;
        };//end of class
    } //end of namespace Model
} //end of namespace TCDSM
#endif // DESCRIBENETCDFOPERATOR_H

