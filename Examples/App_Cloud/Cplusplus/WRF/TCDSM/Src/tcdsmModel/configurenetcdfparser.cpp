#include <tcdsmModel/configurenetcdfparser.h>
#include <netcdf>
#include <tinyxml.h>
#include <easylogging++.h>

using namespace TCDSM::Model;

const std::string rootNodeName = "netcdfparser";
const std::string dimNodeName = "dimensions";
const std::string varNodeName = "variables";



configureNetcdfParser::configureNetcdfParser()
:_isValid(false)
{
}

configureNetcdfParser* configureNetcdfParser::create(const char *configureFile)
{
    configureNetcdfParser *parser = NULL;
    TiXmlDocument doc(configureFile);
    if(!doc.LoadFile())
    {
        LOG(ERROR) << "------------configureNetcdfParser::loadConfigure-------------";
        LOG(ERROR) << "load file \"" << configureFile << "\" failure";
        return parser;
    }

    TiXmlHandle hDoc(&doc);
    TiXmlElement *rootElement = hDoc.FirstChildElement().ToElement();

    if(!rootElement || rootNodeName != rootElement->Value())
    {
        LOG(ERROR) << "------------configureNetcdfParser::loadConfigure-------------";
        LOG(ERROR) << "file \"" << configureFile << "\" is not a xml file";
        return parser;
    }

    parser = new configureNetcdfParser;
    bool rst = false;
    TiXmlHandle handle(rootElement);
    //TiXmlElement *rootElement =  handle.FirstChild( ).Element();

    if(rootElement && rootNodeName == rootElement->Value())\
    {

        TiXmlElement *dimElem =  handle.FirstChildElement(dimNodeName).Element();
        TiXmlElement *varElem =  handle.FirstChildElement(varNodeName).Element();
        if(dimElem && varElem)
        {
            TiXmlHandle dimhandle(dimElem);
            for(TiXmlElement* dimElement = dimhandle.FirstChild().Element();
                    dimElement;          dimElement = dimElement->NextSiblingElement())
            {
                std::string name = dimElement->Attribute("name");
                parser->_dimNameMap[dimElement->Value()] = name;
            }

            TiXmlHandle varhandle(varElem);
            for(TiXmlElement* varElement = varhandle.FirstChild().Element();
                    varElement;          varElement = varElement->NextSiblingElement())
            {
                std::string name = varElement->Attribute("name");
                parser->_varNameMap[varElement->Value()] = name;
            }
            parser->_isValid = true;
            rst = true;
        }
    }

    if(!rst)
    {
        LOG(ERROR) << "------------configureNetcdfParser::loadConfigure-------------";
        LOG(ERROR) << "file \"" << configureFile << "\" is not a netcdf parser configure xml file!";
        return NULL;
    }
    return parser;
}

configureNetcdfParser::~configureNetcdfParser()
{
    _isValid = false;
    _varNameMap.clear();
    _dimNameMap.clear();
}

bool configureNetcdfParser::check(NcFile *ncfile) const
{
//    bool rst = true;
//    for(unsigned int i = 0; i < _dimNameMap.size(); ++i)
//    {
//        std::map<std::string,std::string>::const_iterator c_itr = _dimNameMap.begin();
//        for(;c_itr;++c_itr)
//        {
//            const std::pair<std::string,std::string> &pair = *c_itr;
//            pair.second
//        }

//    }
    return false;
}

unsigned int configureNetcdfParser::getTimeNum(const NcFile *ncfile) const
{

}

time_t configureNetcdfParser::getTime(const NcFile *ncfile, const unsigned int &time) const
{

}

osg::Image *configureNetcdfParser::getCoordinate(const NcFile *ncfile, const unsigned int &time) const
{

}

osg::Image *configureNetcdfParser::getData(const NcFile *ncfile, const unsigned int &time, const Variable &var) const
{

}
