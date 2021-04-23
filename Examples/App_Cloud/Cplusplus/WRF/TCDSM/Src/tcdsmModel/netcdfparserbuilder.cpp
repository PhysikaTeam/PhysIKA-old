#include <tcdsmModel/netcdfparserbuilder.h>


using namespace TCDSM::Model;

NetCDFParserBuilder NetCDFParserBuilder::_instance;

NetCDFParserBuilder *NetCDFParserBuilder::instance()
{
    return &_instance;
}

const NetCDFParser *NetCDFParserBuilder::getNetCDFParser(NcFile *ncfile, const char *parserName)const
{
    if(NULL != parserName )
    {
        ParserMap::const_iterator citr = _parserMap.find(parserName);
        if(_parserMap.end() != _parserMap.find(parserName))
        {
            const std::pair<std::string, boost::shared_ptr< NetCDFParser > > &p = *citr;
            const boost::shared_ptr< NetCDFParser > &sp = p.second;
            if(!sp.get()->check(ncfile))
                return sp.get();
        }
//        if( !=
//                && _parserMap.find(parserName)->secend.get()->check(ncfile))
//        {
//            return _parserMap[parserName].get();
//        }
    }
    for(ParserMap::const_iterator parserIter = _parserMap.begin();
        parserIter != _parserMap.end(); ++parserIter)
    {
        const std::pair<std::string, boost::shared_ptr< NetCDFParser > > &p = *parserIter;
        const boost::shared_ptr< NetCDFParser > &sp = p.second;
        if(!sp.get()->check(ncfile))
            return sp.get();
    }

    return NULL;
}

void NetCDFParserBuilder::addNetCDFParser(const char *parserName,NetCDFParser *parser)
{
    if(NULL != parser)
    {
        boost::shared_ptr<NetCDFParser> parserSP(parser);

        _parserMap[parserName] = parserSP;
    }
}

NetCDFParserBuilder::~NetCDFParserBuilder()
{
    //简单方法,不用管他
    //
    _parserMap.clear();
}

NetCDFParserBuilder::NetCDFParserBuilder()
{

}
