//
// Created by breezelee on 16-10-31.
//

#include <tcdsmUtil/ScopedLog.h>
#include <easylogging++.h>
INITIALIZE_EASYLOGGINGPP

const std::string beginString = "Begin : ";
const std::string endString = "End : ";

using namespace TCDSM::Util;

static void printLog(const std::string &message,const int &level){
	switch (level){
		case 1:
			LOG(WARNING) << message;
			break;
		case 2:
			LOG(DEBUG) << message;
			break;
		case 3:
			LOG(FATAL) << message;
			break;
		case 4:
			LOG(ERROR) << message;
			break;
		default:
			LOG(INFO) << message;
			break;
	}
}


ScopedLog::ScopedLog(const std::string &info, const int &level)
:_level(level)
,_info(info){
	printLog(beginString+_info,_level);
}
ScopedLog::~ScopedLog() {
	printLog(endString+_info,_level);
}

//void ScopedLog::ExtraLog(const std::string &info, const int &level) {
//	printLog(info,level);
//}

//ScopedLog::ScopedLog(const char *info, const loglevel &levle)
//:_level(levle)
//,_info(info) {
//	printLog(beginString+_info,_level);
//}

