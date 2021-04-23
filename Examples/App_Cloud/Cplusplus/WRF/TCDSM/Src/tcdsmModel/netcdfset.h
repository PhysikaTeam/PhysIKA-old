#ifndef TCDSM_MODEL_NETCDFSET_H
#define TCDSM_MODEL_NETCDFSET_H

#include <list>
#include <set>
#include <map>
#include <string>
#include <netcdfcpp.h>
#include <osg/BoundingBox>

#include <tcdsmModel/config.h>
#include <tcdsmModel/netcdfparser.h>

namespace TCDSM {
    namespace Model {

        class TCDSM_MODEL_EXPORT NetCDFSet :public osg::Referenced
        {
        public:
            NetCDFSet();
            virtual ~NetCDFSet();

            bool initComputer();

            void setParser(const NetCDFParser* oprtr);

            bool addFile(NcFile* file);
            void clear();

            const osg::BoundingBox& getBoundingBox() { return _box; }

            unsigned int getTimeNum(const time_t& time);
            inline unsigned int getTimeSize();
            time_t getTime(const int& t);

            osg::Image* getData(const int& time, const Variable& variable);
            osg::Image* getPosition(const int& time);
            osg::Image* getCoordinate(const int& time) { return getPosition(time); }

        protected://函数
            osg::Image* getData(const time_t& time, const Variable& variable);

            unsigned int getAttNumber()const { return (unsigned int)(_attNames.size()); }
            unsigned int getDimNumber()const { return (unsigned int)(_dimNames.size()); }
            unsigned int getVarNumber()const { return (unsigned int)(_varNames.size()); }

            void addFirstFile(NcFile* file);

            bool checkDims(NcFile* file);
            bool checkAtts(NcFile* file);
            bool checkVars(NcFile* file);

            void setAttsName(NcFile* ncfile);
            void setDimsName(NcFile* ncfile);
            void setVarsName(NcFile* ncfile);

            const char* getAtt(const std::string& attName);

        protected:
            struct TimePair
            {
                TimePair(time_t time = 0, int fileNumber = 0, int timeNumber = 0) :
                    _time(time),
                    _fileNumber(fileNumber),
                    _timeNumber(timeNumber) {}
                TimePair(const TimePair& pair) :
                    _time(pair._time),
                    _fileNumber(pair._fileNumber),
                    _timeNumber(pair._timeNumber) {}

                friend bool operator < (const NetCDFSet::TimePair& a, const NetCDFSet::TimePair& b)
                {
                    return a._time < b._time;
                }

                time_t _time;
                int _fileNumber;
                int _timeNumber;
            };
            typedef std::map<int, TimePair> TimeMap;

        protected://变量
            bool _isInit;
            unsigned int _fileNumber;
            std::vector<NcFile*> _ncFiles;

            const NetCDFParser* _ncparser;

            osg::BoundingBox _box;
            //时间维度
            TimeMap _timeMap;

            //验证数据的一致性
            std::set<std::string> _attNames;
            std::map<std::string, int> _dimNames;
            std::set<std::string> _varNames;

        };//end class NetCDFSet

        inline void NetCDFSet::setParser(const NetCDFParser* oprtr)
        {
			std::cout << "oprtr : " << oprtr << std::endl;
			if (oprtr != nullptr)
			{
				_ncparser = oprtr;
				std::cout << "oprtr_ncparser : " << _ncparser << std::endl;
			}
                
            initComputer();
        }//end function NetCDFSet::setOperator

        inline unsigned int NetCDFSet::getTimeSize() { return (unsigned)_timeMap.size(); }

    } //end of namespace Model

}//end of namespace TCDSM

#endif // TCDSM_MODEL_NETCDFSET_H
