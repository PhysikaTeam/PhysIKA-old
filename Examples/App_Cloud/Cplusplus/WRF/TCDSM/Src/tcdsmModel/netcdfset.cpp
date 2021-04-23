#include <tcdsmModel/netcdfset.h>
#include <queue>

#include <easylogging++.h>
INITIALIZE_EASYLOGGINGPP

using namespace TCDSM::Model;

NetCDFSet::NetCDFSet()
    :_isInit(false)
    ,_fileNumber(0)
    ,_ncparser(NULL)
{
    _box.init();
}

NetCDFSet::~NetCDFSet()
{
    if(_ncparser != NULL)

    clear();
}

bool NetCDFSet::initComputer()
{	
	std::cout << "_npFIles.size() : " << _ncFiles.size() << std::endl;
	std::cout << "_ncparser : " << _ncparser << std::endl;

    //验证是否设置_operator
    if(!_ncparser || _ncFiles.size() == 0)
        return false;

	std::cout << "initComputer 1" << std::endl;

    osg::ref_ptr<osg::Image> _position = _ncparser->getCoordinate(_ncFiles[0],0);

	std::cout << "initComputer 2" << std::endl;

    if(!_position->data())
    {
        return false;
    }

	std::cout << "initComputer 3" << std::endl;

    _box.init();

	std::cout << "initComputer 4" << std::endl;

    osg::Vec3 *data = (osg::Vec3* )_position->data();
    const int dataSize = _position->s() * _position->t() *_position->r();
    for(int i = 0; i < dataSize ; ++i)
        _box.expandBy(data[i]);

    //获取时间范围
    //将时间排序
    std::priority_queue<NetCDFSet::TimePair> queue;
    for (unsigned int i = 0; i < _ncFiles.size(); ++i)
    {
        const unsigned int timeNum = _ncparser->getTimeNum(_ncFiles[i]);
        for (unsigned int j = 0; j < timeNum; ++j)
        {
            TimePair timepair(_ncparser->getTime(_ncFiles[i], j), i, j);
            //如果时间为0？
            if (timepair._time == 0)
                return false;
            queue.push(timepair);
        }
    }

    //创建TimeMap
    for(unsigned long i = queue.size(); i > 0 ;--i)
    {
        _timeMap.insert(std::pair<int,TimePair>(i-1,queue.top()));
        queue.pop();
    }

    _isInit = true;
    return true;
}

bool NetCDFSet::addFile(NcFile *file)
{
    //文件无效，则返回添加失败
    if(file && !file->is_valid())
    {
        LOG(ERROR) << "not a valid netcdf file" ;
        return false;
    }
    //如果还没有文件，则新增文件并设置属性，维度和变量信息

//    if(!_ncparser->check(file))
//    {
//        //LOG(ERROR) << "无法解析netcdf file";
//        return false;
//    }

    if(_fileNumber == 0)
    {
        addFirstFile(file);
        return true;
    }

    if(checkAtts(file) && checkDims(file) && checkVars(file))
    {
        //TODO
//        //按照时间插入
//        for(std::list< NcFile *>::iterator itr = _ncFiles.begin();
//            itr != _ncFiles.end(); ++itr)
//        {
//            NcFile * itrfile = (*itr);
//            NcVar *timeVar = itrfile->get_var(_timeVarName.c_str());
//            std::string timeString = timeVar->values();
//        }

        //if(file->get_dim(_timeDimName.c_str())->is_valid())
        //{
        //    long timeSize  = file->get_dim(_timeDimName.c_str())->size();
        //    NcVar *timeVar = file->get_var(_timeVarName.c_str());
        //    timeVar->num_dims();
        //    get_dim();
        //    switch (timeVar->type())
        //    {
        //    case ncChar:
        //        break;
        //    default:
        //        break;
        //    }
        //}

        _ncFiles.push_back(file);
        ++_fileNumber;
        return true;
    }
    return false;
}

void NetCDFSet::clear()
{
    std::vector<NcFile *>::iterator tmp;
    std::vector<NcFile *>::iterator itr = _ncFiles.begin();
    for(;itr != _ncFiles.end();)
    {
        tmp = itr;
        delete *tmp;
        itr = _ncFiles.erase(itr);
    }
    _fileNumber = 0;
//    if(_operator != NULL)
//        delete _operator;
//    _operator = NULL;
}

unsigned int NetCDFSet::getTimeNum(const time_t &time)
{
    for(TimeMap::const_iterator itr = _timeMap.begin();
        itr != _timeMap.end(); ++itr)
    {
        if(itr->second._time == time)
            return (unsigned int)(itr->first);
    }
    return 0;
}

time_t NetCDFSet::getTime(const int &t)
{
    return (_timeMap[t])._time;
}

const char *NetCDFSet::getAtt(const std::string &attName)
{
    NcFile * ncfile = *(_ncFiles.begin());

    for(int i = 0; i < ncfile->num_atts(); ++i)
    {
        if( attName == ncfile->get_att(i)->name() )
        {
            return ncfile->get_att(i)->as_string(0);
        }
    }
    return "";
}

osg::Image *NetCDFSet::getData(const int &time, const Variable &variable)
{
    TimeMap::iterator itr = _timeMap.find(time);
    if(itr == _timeMap.end())
        return new osg::Image;

    return _ncparser->getData(_ncFiles[itr->second._fileNumber],(unsigned int)(itr->second._timeNumber),variable);
}

osg::Image *NetCDFSet::getData(const time_t &time, const Variable &variable)
{
    return getData((int)getTimeNum(time),variable);
}

osg::Image *NetCDFSet::getPosition(const int &time)
{
    TimeMap::iterator itr = _timeMap.find(time);
    if(itr == _timeMap.end())
        return new osg::Image;

    return _ncparser->getCoordinate(_ncFiles[itr->second._fileNumber],(unsigned)(itr->second._timeNumber));
}

void NetCDFSet::addFirstFile(NcFile *file)
{
    _ncFiles.push_back(file);
    ++_fileNumber;

    setAttsName(file);
    setDimsName(file);
    setVarsName(file);
}

bool NetCDFSet::checkDims(NcFile *file)
{
    //检查维度变量的多少和名称是否相同，
    int dimNum = file->num_dims();

    //验证维度是否一致
    if(dimNum != (int)_dimNames.size())
        return false;

    //验证每一个维度名称和大小是否一致
    for(int i = 0; i < dimNum; ++i)
    {
        if(_dimNames.find(file->get_dim(i)->name()) == _dimNames.end() ||
           _dimNames.find(file->get_dim(i)->name())->second != file->get_dim(i)->size())
        {
            return false;
        }
    }

    return true;
}

bool NetCDFSet::checkAtts(NcFile *file)
{
    //检查属性的多少和属性名是否一致
    int attNum = file->num_atts();
    if(attNum != (int)_attNames.size())
        return false;
    for(int i = 0; i < attNum; ++i)
    {
        if(_varNames.find(file->get_var(i)->name()) == _varNames.end())
        {
            return false;
        }
    }
    //TODO 检测属性中的特殊属性是否一致
    return true;
}

bool NetCDFSet::checkVars(NcFile *file)
{
    int varNum = file->num_vars();
    if(varNum != (int)_varNames.size())
        return false;
    for(int i = 0; i < varNum; ++i)
    {
        if(_varNames.find(file->get_var(i)->name()) == _varNames.end())
            return false;
    }
    return true;
}

void NetCDFSet::setAttsName(NcFile *ncfile) {
    int attNum = ncfile->num_atts();
    for(int i = 0; i < attNum; ++i)
    {
        _attNames.insert(ncfile->get_att(i)->name());
    }
}

void NetCDFSet::setDimsName(NcFile *ncfile) {
    int dimNum = ncfile->num_dims();
    for(int i = 0; i < dimNum; ++i)
    {
        _dimNames.insert(std::pair<std::string, long >
                         (ncfile->get_dim(i)->name(),ncfile->get_dim(i)->size()));
    }
}

void NetCDFSet::setVarsName(NcFile *ncfile) {
    int varNum = ncfile->num_vars();
    for(int i = 0; i < varNum; ++i)
    {
        _varNames.insert(ncfile->get_var(i)->name());
    }
}
