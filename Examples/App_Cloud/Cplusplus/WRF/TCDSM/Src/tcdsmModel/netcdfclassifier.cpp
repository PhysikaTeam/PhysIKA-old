#include <tcdsmModel/netcdfclassifier.h>
#include <tcdsmModel/netcdfparserbuilder.h>

#include <easylogging++.h>

using namespace TCDSM::Model;
using namespace std;

NetCDFClassifier::NetCDFClassifier()
    : _classNumber(0)
    , _ncclasses(0)
{
    LOG(INFO) << "create a NetCDF classifier!" ;
}

NetCDFClassifier::NetCDFClassifier(const NetCDFClassifier &netcdfClassifier)
{

}

bool NetCDFClassifier::addFile(const std::string &fileName)
{
    //if nc file is not a valid file return
    boost::shared_ptr<NcFile> file (new NcFile(fileName.c_str()));
    if(!file->is_valid())
    {
        LOG(ERROR) << "file \"" << fileName << "\" is not a valid netcdf file!";
        return false;
    }

    //if have a set contain this type nc file ,direct insert
    for(NcSetListsIter itr = _ncclasses.begin();itr != _ncclasses.end(); ++itr)
    {
        boost::shared_ptr<NetCDFSet> ncs = (*itr);
        if(ncs->addFile(file.get()))
        {
            LOG(INFO) << "add file \"" << fileName << "\" to a netcdf set that already exists!" ;
            return false;
        }
    }

    //else find a new parser and create a class netcdf file set
    const NetCDFParser *parser = NetCDFParserBuilder::instance()->getNetCDFParser(file.get());

    if(NULL == parser)
    {
        LOG(ERROR) << "not find current ncfile parser";
        return false;
    }

    boost::shared_ptr<NetCDFSet> ncs(new NetCDFSet);
    ncs->setParser(parser);
    if(ncs->addFile(file.get()))
    {
        LOG(ERROR) << "can not insert netcdf file in new netcdf file set";
        return false;
    }

    //update netcdfset status
    _classNumber  = _ncclasses.size();
    LOG(INFO) << "create a netcdf set and add the file \"" << fileName << "\" to the netcdf set !";
    _ncclasses.push_back(ncs);
    return true;
}

NetCDFSet *NetCDFClassifier::getClass(int num)
{
    NcSetListsIter itr = _ncclasses.begin();
    for(int i = 0;itr != _ncclasses.end() && i < num ; ++itr);
    if(itr == _ncclasses.end())
        return NULL;
    else
        return (*itr).get();
}

void NetCDFClassifier::clear()
{
    _ncclasses.clear();

    if(_ncclasses.size() == 0)
        return;
    NetCDFSetLists::iterator tmp;
    NetCDFSetLists::iterator itr = _ncclasses.begin();
    for(;itr != _ncclasses.end();)
    {
        tmp = itr;
        (*tmp)->clear();
        itr = _ncclasses.erase(itr);
       tmp->reset();
    }
}
