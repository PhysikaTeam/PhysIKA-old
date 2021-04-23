#include <tcdsmModeler/simplemodeler.h>
#include <tcdsmUtil/ScopedLog.h>
#include <tcdsmModel/wrfnetcdfparser.h>

#include <tcdsmUtil/util.h>
#include <easylogging++.h>

#include "WriteVTI.hpp"

INITIALIZE_EASYLOGGINGPP

using namespace TCDSM::Modeler;
using namespace TCDSM::Model;

SimpleModeler::SimpleModeler(const char *ncfile, const char *savePath)
        : _netcdfFileName(ncfile), _modelPath(savePath) {

}

SimpleModeler::~SimpleModeler() {

}

bool SimpleModeler::setNetCDFFile(const std::string &path) {

    if(_netcdfFileName == path)
        return false;
    NcFile ncFile(path.c_str());

    //std::cout << "is_valid: " << ncFile.is_valid() << std::endl;

    if (!ncFile.is_valid())
        return false;
    _netcdfFileName = path;
    return true;
}

bool SimpleModeler::execute() {

    Model::NetCDFSet *netCDFSet = generateNetCDFSet();

	std::cout << "netCDFSet : " << netCDFSet << std::endl;

    if(!netCDFSet) return false;
    const unsigned tSize =netCDFSet->getTimeSize();

	std::cout << "tSize : " << tSize << std::endl;

    for (unsigned int ti = 0; ti < tSize; ++ti) {
        //TODO 时间信息
        //生成粒子的位置信息
        osg::ref_ptr<Model::Particles> particles = generateParticles(netCDFSet,ti);
        if(!particles.valid()) {
            delete netCDFSet;
            return false;
        }

        if(!(computeRadius(particles,netCDFSet,ti)   &&   //计算粒子的半径
             computeMass(particles,netCDFSet,ti)     &&   //计算粒子的质量//如果没有，则全局一个值
             computeVelocity(particles,netCDFSet,ti) &&   //计算粒子的速度
             computeExtinction(particles,netCDFSet,ti)))  //计算粒子的消光系数
        {
            break;
        }

		//my 注释
        //computeOldAndLastPositionDependVelocity(particles); //计算粒子的二阶位置

        SetAsDefaultColor(particles); //设置颜色值

		/*
        {    //输出粒子文件
            Util::ScopedLog log("保存第"+ Util::toString(ti) + "帧粒子数据！" );
			particles->save(_modelPath + "/frame" + Util::toString(ti) + ".dat",
				//Model::Particles::CLOUD_DYNAMIC | Model::Particles::CLOUD_STATIC_EXTINCTION);
				Model::Particles::CLOUD_STATIC_EXTINCTION);
        }
		*/

		GenerateVolumeFile(netCDFSet, ti);
    }

    delete netCDFSet;
    return true;
}

void SimpleModeler::addAMixingRatioVariable(TCDSM::Model::Particles *p, osg::Image *source, osg::Image * density, double mix) {
    TCDSM::Util::ScopedLog log("合并变量");

    const size_t dataSize = p->size();
    if (dataSize != (size_t)source->r() * source->s() * source->t()) {
        LOG(WARNING) << "[SimpleModeler::addAMixingRatioVariable] source image and distance have not same size of dimension!";
        return;
    }

    float *s_data = (float *) (source->data());
    float *densityData = (float*)(density->data());
    for (size_t i = 0; i < dataSize; ++i) {
        p->getExtinction(i) += 3 * (*(densityData+i)) * *(s_data + i) * mix;
    }

}

void SimpleModeler::setDefault() {
    addVariable(Model::QCLOUD, 0.01);
    addVariable(Model::QICE  , 1.0);
    addVariable(Model::QRAIN , 1.0);
    addVariable(Model::QSNOW , 2.0);
    addVariable(Model::QGRAUP, 2.5);
//    addVariable(Model::QVAPOR, 0.001);
}

TCDSM::Model::NetCDFSet *SimpleModeler::generateNetCDFSet() {

    Util::ScopedLog log("创建NetCDF Set");

    NcFile *ncFile = new NcFile(_netcdfFileName.c_str());

    if (!ncFile->is_valid()) {
        LOG(WARNING) << "File \" " << _netcdfFileName << " \" 不是有效的netcdf文件";
        delete ncFile;
        return NULL;
    }

	std::cout << "generateNetCDFSet 1" << std::endl;

    TCDSM::Model::NetCDFSet *netCDFSet = new TCDSM::Model::NetCDFSet;
//    netCDFSet->setOperator(new TCDSM::Model::WRFNetCDFOperator());

	std::cout << "generateNetCDFSet 2" << std::endl;

    //增加进去以后就是有NetCDFSet管理
    netCDFSet->addFile(ncFile);

	std::cout << "generateNetCDFSet 3" << std::endl;

    //设置netcdf解析器
    WRFNetCDFOperator *perser = new WRFNetCDFOperator;
    netCDFSet->setParser(perser);

	std::cout << "generateNetCDFSet 4" << std::endl;

	//setParser()中已经调用过initComputer一次
    //初始化netCDSSet
	//bool tmp = netCDFSet->initComputer();
	//std::cout << "tmp : " << tmp << std::endl;
	//std::cout << "generateNetCDFSet 5" << std::endl;

    return netCDFSet;
}

TCDSM::Model::Particles *SimpleModeler::generateParticles(TCDSM::Model::NetCDFSet *netCDFSet, int time) {
    TCDSM::Util::ScopedLog generatelog("create Particles and set poisition");
    //计算粒子坐标
    osg::ref_ptr<osg::Image> coord;
    osg::Vec3i dimSize;
    {
        TCDSM::Util::ScopedLog("获取坐标信息");
        coord = netCDFSet->getCoordinate(time);
        dimSize = osg::Vec3i(coord->s(), coord->t(), coord->r());
    }

    const size_t particleSize = (size_t) dimSize.x() * dimSize.y() * dimSize.z();
    if (particleSize > 0x7fffffff) {
        LOG(WARNING) << "Size is beyond memory!";
        return NULL;
    }

    TCDSM::Model::Particles * particles = new TCDSM::Model::Particles(particleSize);
    //速度太慢，需要加速
    for(unsigned int i = 0; i < particleSize; ++i) {
        particles->setPosition(i, *(vec3 *) (coord->data(i)));
        particles->setPosition0(i, *(vec3 *) (coord->data(i)));
    }
    return particles;
}

bool SimpleModeler::computeMass(TCDSM::Model::Particles *p, TCDSM::Model::NetCDFSet *netCDFSet, int time) {
    TCDSM::Util::ScopedLog masslog("计算云粒子的质量");

    if(!p || !netCDFSet){
        LOG(WARNING) << "计算粒子质量时：粒子数据为空或者NetCDFSet 为空！";
        return false;
    }
    const unsigned int pSize = (int)p->size();
    osg::ref_ptr<osg::Image> density = netCDFSet->getData(time,TCDSM::Model::DENSITY);

    if(pSize != density->t() * density->s() * density->r() )
    {
        LOG(WARNING) << "数据不匹配或者没有NetCDF中缺少密度数据！";
        return false;
    }

    for (unsigned int i = 0; i < pSize; ++i) {
        /*TODO 计算密度需要修改
         *更具经纬度计算其具体啊的密度
         *这里是只是似计算
        **/
        double V = p->getRadius(i);
        V = V*V*V;
        p->setMass(i,(real)(V * *(float*)(density->data(i))));
    }
    return true;
}

bool SimpleModeler::computeExtinction(TCDSM::Model::Particles *p, TCDSM::Model::NetCDFSet *netCDFSet, int time) {
    TCDSM::Util::ScopedLog log("计算第"+ TCDSM::Util::toString(time) + "帧数据的消光系数！" );

    osg::ref_ptr<osg::Image> density = netCDFSet->getData(time,TCDSM::Model::DENSITY);

    for (auto &a:_qVariableWeight) {
        TCDSM::Util::ScopedLog log2(std::string("提取 ") + TCDSM::Util::toString(a.first) + " 变量并合并变量");
        //必须使用一个智能指针指向这个数据，这样可以保证内存不泄露
        osg::ref_ptr<osg::Image> qVariable = netCDFSet->getData(time, a.first);
        addAMixingRatioVariable(p, qVariable, density,a.second);
    }
    return true;
}

bool SimpleModeler::computeRadius(TCDSM::Model::Particles *p, TCDSM::Model::NetCDFSet *netCDFSet, int time) {
    TCDSM::Util::ScopedLog log("计算第"+ TCDSM::Util::toString(time) + "帧数据的半径");
    /* TODO 计算粒子的半径
     * 这里指的半径为网格的半径
     * 思路1：在netcdf中计算，就需要netcdf支持
     * 思路2：直接计算，使用GDAL等，暂时不考虑
     */
    // 通过ncdump查看的a.nc 为18000
    // 通过ncdump查看的a.nc 为5000
    const real radius = 18000.f / 2.f;
    const size_t pSize = p->size();
    for (unsigned int i = 0; i < pSize; ++i) {
        p->setRadius(i,radius);
    }
    return true;
}

bool SimpleModeler::computeVelocity(TCDSM::Model::Particles *p, TCDSM::Model::NetCDFSet *netCDFSet, int time) {
    TCDSM::Util::ScopedLog log("计算第"+ TCDSM::Util::toString(time) + "帧数据的速度");

    osg::ref_ptr< osg::Image > wind = netCDFSet->getData(time,WIND);
    if(!wind.valid())
    {
        LOG(WARNING) << "计算第" << TCDSM::Util::toString(time) << "帧数据的速度时无法获取风场数据";
        return false;
    }
    const size_t pSize = p->size();
    for (unsigned int i = 0; i < pSize; ++i) {
        p->setVelocity(i,*(vec3*)(wind->data(i)));
    }
    return true;
}

void SimpleModeler::computeOldAndLastPositionDependVelocity(TCDSM::Model::Particles *p,const double &dt) {

    TCDSM::Util::ScopedLog log("计算第"+ TCDSM::Util::toString(time) + "帧数据的一阶速度和二阶速度");
    //TODO 只有知道dt才能计算速度
    const size_t pSize = p->size();

    for (unsigned int i = 0; i < pSize; ++i) {
        //这个胡闹啊。。。
        p->setOldPosition (i,p->getPosition(i) - p->getVelocity(i) * dt);
        p->setLastPosition(i,p->getVelocity(i) - p->getVelocity(i) * dt * 2 );
    }
}

bool SimpleModeler::SetAsDefaultColor(TCDSM::Model::Particles *p) {

    TCDSM::Util::ScopedLog log("设置颜色值为0.5,0.5,0.5,0.5" );

    const vec4 color (0.5,0.5,0.5,0.5);
    const unsigned int pSize = p->size();

    for (unsigned int i = 0; i < pSize; ++i) {
        p->setColor(i,color);
    }

    return true;
}


bool SimpleModeler::GenerateVolumeFile(TCDSM::Model::NetCDFSet *netCDFSet, int time)
{
	osg::ref_ptr<osg::Image> density = netCDFSet->getData(time, TCDSM::Model::DENSITY);

	int length = density->s();
	int width = density->t();
	int height = density->r();
	size_t dataSize = length * height * width;

	std::vector<float> extinction(dataSize, 0);

	float *dd = (float*)(density->data());

	for (auto &tmp : _qVariableWeight) {
		//必须使用一个智能指针指向这个数据，这样可以保证内存不泄露
		osg::ref_ptr<osg::Image> qVariable = netCDFSet->getData(time, tmp.first);

		float* qvd = (float *)(qVariable->data());
		for (size_t i = 0; i < dataSize; i++)
		{
			extinction[i] = 3 * (*(dd + i)) * *(qvd + i) * tmp.second;
		}
	}

	std::cout << "准备写入vti文件……" << std::endl;
	return WriteVTI(length-1, width-1, height-1, extinction, (_modelPath + "/frame" + Util::toString(time) + ".vti"));
}
