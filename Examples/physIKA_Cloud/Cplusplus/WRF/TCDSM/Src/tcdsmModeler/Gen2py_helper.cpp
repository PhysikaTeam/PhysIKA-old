#include <tcdsmModeler/Gen2py_helper.h>
#include <tcdsmModeler/simplemodeler.h>
#include <string>

void execute(const std::string& NetCDFFile, const std::string& ModelSavePath) {
    TCDSM::Modeler::SimpleModeler modeler;
    modeler.setNetCDFFile(NetCDFFile);
    modeler.setModelSavePath(ModelSavePath);
    modeler.setDefault();
    modeler.execute();
}