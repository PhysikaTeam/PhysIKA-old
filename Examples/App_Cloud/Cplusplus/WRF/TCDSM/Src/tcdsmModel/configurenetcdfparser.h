#ifndef TCDSM_MODEL_CONFIGURENETCDFPARSER_H
#define TCDSM_MODEL_CONFIGURENETCDFPARSER_H

#include <tcdsmModel/netcdfparser.h>

namespace TCDSM {namespace Model {

class TCDSM_MODEL_EXPORT configureNetcdfParser:public NetCDFParser
{
public:
    explicit configureNetcdfParser();
    static configureNetcdfParser* create(const char *configureFile );
    virtual ~configureNetcdfParser();
    virtual bool check(NcFile *ncfile) const;
    virtual unsigned int getTimeNum(const NcFile *ncfile) const;
    virtual time_t getTime(const NcFile *ncfile,const unsigned int &time) const;
    virtual osg::Image* getCoordinate(const NcFile *ncfile,const unsigned int &time) const;
    virtual osg::Image* getData(const NcFile *ncfile,const unsigned int &time,const Variable &var) const;

protected:
    bool _isValid;
    std::map<std::string,std::string> _varNameMap;
    std::map<std::string,std::string> _dimNameMap;
};

}

}


#endif // CONFIGURENETCDFPARSER_H

