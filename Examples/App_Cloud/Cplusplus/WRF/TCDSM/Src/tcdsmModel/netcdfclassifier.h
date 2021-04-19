#ifndef NETCDFCLASSIFIER_H
#define NETCDFCLASSIFIER_H


#include <tcdsmModel/netcdfset.h>
#include <tcdsmModel/netcdfparser.h>
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include <vector>

namespace TCDSM {
    namespace Model {

        class TCDSM_MODEL_EXPORT NetCDFClassifier
        {
        public:
            NetCDFClassifier();
            virtual ~NetCDFClassifier() {}
            bool addFile(const std::string& fileName);
            unsigned long classCount()const { return _classNumber; }
            NetCDFSet* getClass(int i);
            void clear();

        protected:
            NetCDFClassifier(const NetCDFClassifier& netcdfClassifier);


        protected:
            typedef std::vector<boost::shared_ptr<NetCDFSet>> NetCDFSetLists;
            typedef NetCDFSetLists::iterator                NcSetListsIter;
            unsigned long   _classNumber;
            NetCDFSetLists  _ncclasses;

            //NetCDFParser *                    _operator;
            //void setOperator(NetCDFParser *optr){ _operator = optr;}
        }; //end of class NetCDFClassifier

    }//end of namespace Model
}//end of namespace TCDSM

#endif // NETCDFCLASSIFIER_H
