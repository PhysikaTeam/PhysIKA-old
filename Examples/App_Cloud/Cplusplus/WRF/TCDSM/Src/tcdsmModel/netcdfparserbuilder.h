/****************************************
 * 单例模式，其中保存所注册的netcdfPerser
 * 通过getNetCDFParser(NcFile*)返回能够解析ncfile的perser
 *
 *
 */


#ifndef TCDSM_MODEL_NETCDFOPERATORBUILDER_H
#define TCDSM_MODEL_NETCDFOPERATORBUILDER_H

#include <tcdsmModel/netcdfparser.h>
#include <boost/shared_ptr.hpp>
#include <vector>

namespace TCDSM {
    namespace Model {


        class TCDSM_MODEL_EXPORT NetCDFParserBuilder
        {
        public:
            ////////////////////////////////////////////////////
            /// \brief instance          //单例模式入口
            /// \return
            ///
            static NetCDFParserBuilder* instance();

            /////////////////////////////////////////////////////
            /// \brief getNetCDFParser  //获取能够解析 netcdf parser
            /// \param ncfile
            /// \return if not found return NULL
            ///
            const NetCDFParser* getNetCDFParser(NcFile* ncfile, const char* parserName = NULL)const;

            ///////////////////////////////////////////////////
            /// \brief addNetCDFParser
            /// \param parser
            ///
            void addNetCDFParser(const char* parserName, NetCDFParser* parser);

            ~NetCDFParserBuilder();

        protected: //function
            NetCDFParserBuilder();

        protected://member variable
            typedef std::map< std::string, boost::shared_ptr< NetCDFParser > > ParserMap;
            static NetCDFParserBuilder _instance;
            ParserMap _parserMap;

        };

    }

}

#endif // TCDSM_MODEL_NETCDFOPERATORBUILDER_H
