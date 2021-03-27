#ifndef TCDSM_SIMPLEMODELER_H
#define TCDSM_SIMPLEMODELER_H

#include <tcdsmModeler/modeler.h>
#include <tcdsmModel/netcdfset.h>
#include <tcdsmModel/particles.h>
#include <string>
#include <map>
#include <osg/Image>


namespace TCDSM{
namespace Modeler{

    class TCDSM_MODELER_EXPORT SimpleModeler:public AbstractModeler{
    public:
        SimpleModeler(const char *ncfile = "",const char *savePath = "");

        virtual ~SimpleModeler();

        bool setNetCDFFile(const std::string &path);
        inline void setModelSavePath(const std::string &path);
        virtual bool execute();

        inline void addVariable(const Model::Variable &name, double weight);
        void setDefault();
    protected:
        void addAMixingRatioVariable(Model::Particles *p, osg::Image *source, osg::Image * density, double mix);
        Model::NetCDFSet* generateNetCDFSet();
        Model::Particles* generateParticles(Model::NetCDFSet *netCDFSet, int time= 0);
        bool computeMass      (Model::Particles *p, Model::NetCDFSet *netCDFSet, int time= 0);
        bool computeExtinction(Model::Particles *p, Model::NetCDFSet *netCDFSet, int time= 0);
        bool computeRadius    (Model::Particles *p, Model::NetCDFSet *netCDFSet, int time= 0);
        bool computeVelocity  (Model::Particles *p, Model::NetCDFSet *netCDFSet, int time= 0);
        void computeOldAndLastPositionDependVelocity(Model::Particles *p, const double &dt = 1.0f);

        // 临时设置的颜色
        bool SetAsDefaultColor(TCDSM::Model::Particles *p);

		// 生成体数据文件
		bool GenerateVolumeFile(TCDSM::Model::NetCDFSet *netCDFSet, int time);
		// 将密度数据保存为.vti文件(ascii形式)(length，width，height：数据场长宽高；data：密度数据；path：文件保存路径)
		// bool WriteVTI(int length, int width, int height, const std::vector<float>& data, std::string& path);

    protected:
        std::string _netcdfFileName; //单一文件
        std::string _modelPath;
        std::map <Model::Variable, double>  _qVariableWeight;
    };

    void SimpleModeler::addVariable(const Model::Variable &name, double weight) {
        _qVariableWeight[name] = weight;
    }
    void SimpleModeler::setModelSavePath(const std::string &path){_modelPath = path;}
}
}

#endif //TCDSM_SIMPLEMODELER_H
