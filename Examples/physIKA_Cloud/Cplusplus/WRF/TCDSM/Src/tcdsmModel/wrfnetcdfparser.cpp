#include <tcdsmModel/wrfnetcdfparser.h>
#include <tcdsmUtil/util.h>
#include <osg/Vec4>

using namespace TCDSM::Model;
using namespace std;

typedef enum {
    LONGITUDE_S = 0x25,
     LATITUDE_S = 0x26,
     ALTITUDE_S = 0x27
    /*,DATASTRLEN = 0x28, SOILLAYERS_STAG = 0x29*/
}DimType2;

typedef enum {
    V_TIM   = 0x14,
    V_LON   = 0x0B,
    V_LAT   = 0x0C,
    V_LON_U = 0x0D,
    V_LAT_U = 0x0E,
    V_LON_V = 0x0F,
    V_LAT_V = 0x10,
    X_WIND  = 0x11,
    Y_WIND  = 0x12,
    Z_WIND  = 0x13 //19
}NeedVariable;

//为了计算维度
map<const GridType, const map<const DimType,const unsigned> > initDimName()
{
    map<const GridType, const map<const DimType,const unsigned> > dimMap;
    {
        map<const DimType, const unsigned> dimName;
        dimName.insert(pair<const DimType, const unsigned>(LONGITUDE,LONGITUDE));
        dimName.insert(pair<const DimType, const unsigned>(LATITUDE,LATITUDE));
        dimName.insert(pair<const DimType, const unsigned>(ALTITUDE,ALTITUDE));
        dimMap.insert(pair<const GridType,const map<const DimType,const unsigned> >(GRID,dimName));
    }

    {
        map<const DimType, const unsigned> dimName;
        dimName.insert(pair<const DimType, const unsigned>(LONGITUDE,LONGITUDE_S));
        dimName.insert(pair<const DimType, const unsigned>(LATITUDE,LATITUDE));
        dimName.insert(pair<const DimType, const unsigned>(ALTITUDE,ALTITUDE));
        dimMap.insert(pair<const GridType,const map<const DimType,const unsigned> >(UGRID,dimName));
    }

    {
        map<const DimType, const unsigned> dimName;
        dimName.insert(pair<const DimType, const unsigned>(LONGITUDE,LONGITUDE));
        dimName.insert(pair<const DimType, const unsigned>(LATITUDE,LATITUDE_S));
        dimName.insert(pair<const DimType, const unsigned>(ALTITUDE,ALTITUDE));
        dimMap.insert(pair<const GridType,const map<const DimType,const unsigned> >(VGRID,dimName));
    }

    {
        map<const DimType, const unsigned> dimName;
        dimName.insert(pair<const DimType, const unsigned>(LONGITUDE,LONGITUDE));
        dimName.insert(pair<const DimType, const unsigned>(LATITUDE,LATITUDE));
        dimName.insert(pair<const DimType, const unsigned>(ALTITUDE,ALTITUDE_S));
        dimMap.insert(pair<const GridType,const map<const DimType,const unsigned> >(WGRID,dimName));
    }

    return dimMap;
}

map<const GridType, const map<const DimType,const unsigned> > initCoordVariableMapName()
{
    map<const GridType, const map<const DimType,const unsigned> > coordVariableMap;
    {
        map<const DimType,const unsigned> coordMap;
        coordMap.insert(pair<const DimType,const unsigned>(LONGITUDE,V_LON));
        coordMap.insert(pair<const DimType,const unsigned>(LATITUDE ,V_LAT));
        coordVariableMap.insert(pair<const GridType,const map<const DimType,const unsigned> >(GRID,coordMap));
        coordVariableMap.insert(pair<const GridType,const map<const DimType,const unsigned> >(WGRID,coordMap));
    }

    {
        map<const DimType,const unsigned> coordMap;
        coordMap.insert(pair<const DimType,const unsigned>(LONGITUDE,V_LON_U));
        coordMap.insert(pair<const DimType,const unsigned>(LATITUDE ,V_LAT_U));
        coordVariableMap.insert(pair<const GridType,const map<const DimType,const unsigned> >(UGRID,coordMap));
    }

    {
        map<const DimType,const unsigned> coordMap;
        coordMap.insert(pair<const DimType,const unsigned>(LONGITUDE,V_LON_V));
        coordMap.insert(pair<const DimType,const unsigned>(LATITUDE ,V_LAT_V));
        coordVariableMap.insert(pair<const GridType,const map<const DimType ,const unsigned> >(VGRID,coordMap));
    }
    return coordVariableMap;
}

map<const unsigned int, const string> initNameMap()
{
	map<const unsigned int, const string> varMap;
	//7个需要的维度，另外数据长度维度不需要，因此这里面就不是用
	//下标一次是 32-38
	varMap.insert(pair<const unsigned int, const string>(TIMEDIM, "Time"));
	varMap.insert(pair<const unsigned int, const string>(LONGITUDE, "west_east"));
	varMap.insert(pair<const unsigned int, const string>(LATITUDE, "south_north"));
	varMap.insert(pair<const unsigned int, const string>(ALTITUDE, "bottom_top"));
	//varMap.insert(pair<const unsigned int,const string>(LONGITUDE_S,"west_east_stag"  ));  //大气所项目文件中没有该dim
	//varMap.insert(pair<const unsigned int,const string>(LATITUDE_S ,"south_north_stag"));
	//varMap.insert(pair<const unsigned int,const string>(ALTITUDE_S ,"bottom_top_stag" ));

	//需要19个数据 前标依次是1-19
	//varMap.insert(pair<const unsigned int,const string>(V_TIM      ,"Times"  ));
	//varMap.insert(pair<const unsigned int,const string>(V_LON      ,"XLONG"  ));
	//varMap.insert(pair<const unsigned int,const string>(V_LAT      ,"XLAT"   ));
	//varMap.insert(pair<const unsigned int,const string>(V_LON_U    ,"XLONG_U"));
	//varMap.insert(pair<const unsigned int,const string>(V_LAT_U    ,"XLAT_U" ));
	//varMap.insert(pair<const unsigned int,const string>(V_LON_V    ,"XLONG_V"));
	//varMap.insert(pair<const unsigned int,const string>(V_LAT_V    ,"XLAT_V" ));
	//varMap.insert(pair<const unsigned int,const string>(X_WIND     ,"U"      ));
	//varMap.insert(pair<const unsigned int,const string>(Y_WIND     ,"V"      ));
	//varMap.insert(pair<const unsigned int,const string>(Z_WIND     ,"W"      ));
	//varMap.insert(pair<const unsigned int,const string>(QICE       ,"QICE"   ));
	//varMap.insert(pair<const unsigned int,const string>(QVAPOR     ,"QVAPOR" ));
	//varMap.insert(pair<const unsigned int,const string>(QCLOUD     ,"QCLOUD" ));
	//varMap.insert(pair<const unsigned int,const string>(QSNOW      ,"QSNOW"  ));
	//varMap.insert(pair<const unsigned int,const string>(QRAIN      ,"QRAIN"  ));
	//varMap.insert(pair<const unsigned int,const string>(PRESSURE   ,"PB"     ));
	//varMap.insert(pair<const unsigned int,const string>(TEMPERATURE,"TEMP"   ));//b.nc没有
	//varMap.insert(pair<const unsigned int,const string>(DENSITY    ,"RRB"    ));//b.nc没有
	varMap.insert(pair<const unsigned int, const string>(HUMIDITY, "RELHUM"));//b.nc没有
	//varMap.insert(pair<const unsigned int,const string>(QGRAUP     ,"QGRAUP" ));

	// 尝试对应大气所项目中.nc数据的variable names
	varMap.insert(pair<const unsigned int, const string>(V_TIM, "times"));

	varMap.insert(pair<const unsigned int, const string>(V_LON, "west_east"));
	varMap.insert(pair<const unsigned int, const string>(V_LAT, "south_north"));

	varMap.insert(pair<const unsigned int, const string>(X_WIND, "uu"));
	varMap.insert(pair<const unsigned int, const string>(Y_WIND, "vv"));
	varMap.insert(pair<const unsigned int, const string>(Z_WIND, "ww"));

	varMap.insert(pair<const unsigned int, const string>(QICE, "qice"));
	varMap.insert(pair<const unsigned int, const string>(QVAPOR, "qvapor"));
	varMap.insert(pair<const unsigned int, const string>(QCLOUD, "qcloud"));
	varMap.insert(pair<const unsigned int, const string>(QSNOW, "qsnow"));
	varMap.insert(pair<const unsigned int, const string>(QRAIN, "qrain"));

	varMap.insert(pair<const unsigned int, const string>(PRESSURE, "pp"));
	varMap.insert(pair<const unsigned int, const string>(TEMPERATURE, "tt"));
	varMap.insert(pair<const unsigned int, const string>(DENSITY, "rho"));

	varMap.insert(pair<const unsigned int, const string>(QGRAUP, "qgraupal"));
	return varMap;
}

const map<const GridType, const map<const DimType,const unsigned> > dimensionMap = initDimName();
const map<const GridType, const map<const DimType,const unsigned> > coordinateVariableMap = initCoordVariableMapName();
const map<const unsigned int,const string>  varNameMap = initNameMap();

const char *getName(const unsigned int &key){
    return varNameMap.find(key)->second.c_str();
}

bool WRFNetCDFOperator::check(NcFile *ncfile)const
{
    if(!ncfile)
        return false;

    //check dim
    for(auto a:dimensionMap)
    {
        if(!haveDim(ncfile,a.first))
        {
            return false;
        }
    }
    //check var
    //TODO check var
    for(auto a:varNameMap)
    {
        if(!haveVar(ncfile,a.first))
        {
            return false;
        }
    }
    return true;
}

unsigned int WRFNetCDFOperator::getTimeNum(const NcFile *ncfile)const{
    NcError err(NcError::verbose_nonfatal);
    if(ncfile == NULL && !ncfile->is_valid() && !haveDim(ncfile,TIMEDIM))
        return 0;

    NcDim *dim = ncfile->get_dim(getName(TIMEDIM));
    return dim->size();
}

time_t WRFNetCDFOperator::getTime(const NcFile *ncfile,const unsigned int &time)const{
    NcError err(NcError::verbose_nonfatal);
    if(ncfile == NULL && !ncfile->is_valid() && !haveDim(ncfile,TIMEDIM) && !haveVar(ncfile,V_TIM) &&
            time >= ncfile->get_dim(getName(TIMEDIM))->size())
        return 0;
    NcVar* timesVar = ncfile->get_var(getName(V_TIM));
    string timeStr = timesVar->get_rec(time)->as_string(0);
    return TCDSM::Util::CharToTime(timeStr.c_str());
}

osg::Image* WRFNetCDFOperator::getCoordinate(const NcFile *ncfile,const unsigned int &time)const{
    return getCoordinate(ncfile,time,GRID);
}

osg::Image* WRFNetCDFOperator::getData(const NcFile *ncfile,const unsigned int &time,const Variable &name)const{
    if(name == WIND)
        return getWind(ncfile,time);

    if(ncfile == NULL && !ncfile->is_valid() && !haveDim(ncfile,TIMEDIM) && !haveVar(ncfile,name) && time >= ncfile->get_dim(getName(TIMEDIM))->size())
        return new osg::Image;

    NcVar *var = ncfile->get_var(getName(name));
#ifdef DEBUG
    cout << getName(name)<<"---------------------"<< endl;
    cout << (var->is_valid()?"true":"false") <<endl;
    cout << var->num_dims() << endl;
#endif
    const long x = var->get_dim(3)->size();
    const long y = var->get_dim(2)->size();
    const long z = var->get_dim(1)->size();

    osg::Image *data = new osg::Image;
    data->allocateImage(x,y,z,GL_INTENSITY,GL_FLOAT);
    NcValues *values = var->get_rec(time);
    float *dataPoint = (float *)data->data();
    const long size = values->num();
    for(long k = 0; k < size; ++k)
    {
        dataPoint[k] = values->as_float(k);
		if (dataPoint[k] < 0)
		{
			dataPoint[k] = 0;
		}
		else
		{
			//std::cout << dataPoint[k] << endl;
		}
    }
    return data;
}

osg::Image* WRFNetCDFOperator::getWind(const NcFile *ncfile,const unsigned int &time)const{
    if(ncfile == NULL && !(ncfile->is_valid()) && !haveDim(ncfile,TIMEDIM)&& time >= ncfile->get_dim(getName(TIMEDIM))->size()
       && !haveVar(ncfile,X_WIND) && !haveVar(ncfile,Y_WIND) && !haveVar(ncfile,Z_WIND) )
        return new osg::Image;

    const unsigned x = (unsigned)(ncfile->get_dim(getName(LONGITUDE))->size());
    const unsigned y = (unsigned)(ncfile->get_dim(getName(LATITUDE ))->size());
    const unsigned z = (unsigned)(ncfile->get_dim(getName(ALTITUDE ))->size());

    osg::Image *data = new osg::Image;
    data->allocateImage((int)x,(int)y,(int)z,GL_RGB,GL_FLOAT);

    NcValues *valuesU = ncfile->get_var(getName(X_WIND))->get_rec(time);
    NcValues *valuesV = ncfile->get_var(getName(Y_WIND))->get_rec(time);
    NcValues *valuesW = ncfile->get_var(getName(Z_WIND))->get_rec(time);

    osg::Vec3f *dataPoint = (osg::Vec3f *)data->data();

    const unsigned int xxy = x*y;
    for(unsigned k = 0; k < z; ++k)
    {
        for(unsigned j = 0; j < y; ++j)
        {
            for(unsigned i = 0; i < x; ++i)
            {
                unsigned index   = i + j*x + k*xxy;

                //unsigned indexXb = i     + j*(x+1) + k*(x+1)*y;
                //unsigned indexXe = (i+1) + j*(x+1) + k*(x+1)*y;
                unsigned indexXb = index + j + k*y;
                unsigned indexXe = indexXb+1;

                //unsigned indexYb = i + j    *x + k*x*(y+1);
                //unsigned indexYe = i + (j+1)*x + k*x*(y+1);
                unsigned indexYb = index + k*x;
                unsigned indexYe = indexYb + x;

                //unsigned indexZb = i + j*x +  k   *x*y;
                //unsigned indexZe = i + j*x + (k+1)*x*y;
                unsigned indexZb = index;
                unsigned indexZe = indexZb + xxy;

                //dataPoint[index].x() = (valuesU->as_float(indexXb) + valuesU->as_float(indexXe))/2.0f;
                //dataPoint[index].y() = (valuesV->as_float(indexYb) + valuesV->as_float(indexYe))/2.0f;
                //dataPoint[index].z() = (valuesW->as_float(indexZb) + valuesW->as_float(indexZe))/2.0f;

				//没弄明白
				dataPoint[index].x() = valuesU->as_float(i) < 0 ? 0 : valuesU->as_float(i);
				dataPoint[index].y() = valuesU->as_float(j) < 0 ? 0 : valuesU->as_float(j);
				dataPoint[index].z() = valuesU->as_float(k) < 0 ? 0 : valuesU->as_float(k);

            }
        }
    }

    return data;
}

osg::Image* WRFNetCDFOperator::getCoordinate(const NcFile *ncfile,const unsigned int &time,const GridType &gridType)const{

	std::cout << "getCoordinate 1" << std::endl;

    const map<const DimType,const unsigned> &dimMap =dimensionMap.find(gridType)->second;
    const map<const DimType,const unsigned> &coordinateMap = coordinateVariableMap.find(gridType)->second;

	std::cout << "getCoordinate 2" << std::endl;

	//std::cout << std::string(getName(dimMap.find(LONGITUDE)->second)) << std::endl;

    const long x = ncfile->get_dim(getName(dimMap.find(LONGITUDE)->second))->size();
    const long y = ncfile->get_dim(getName(dimMap.find(LATITUDE )->second))->size();
    const long z = ncfile->get_dim(getName(dimMap.find(ALTITUDE )->second))->size();

	std::cout << "x : " << x << "  y : " << y << "  z : " << z << std::endl;
	std::cout << "getCoordinate 3" << std::endl;

	std::cout << std::string(getName(coordinateMap.find(LONGITUDE)->second)) << std::endl;
	std::cout << "get_rec(time) : " << ncfile->get_var(getName(coordinateMap.find(LONGITUDE)->second))->get_rec(time)->num() << std::endl;

	NcVar* xlong_tmp = ncfile->get_var(getName(coordinateMap.find(LONGITUDE)->second));
	std::cout << "xlong_tmp->num_dims() : " << xlong_tmp->num_dims() << std::endl;
	std::cout << "xlong_tmp->num_vals() : " << xlong_tmp->num_vals() << std::endl;

    NcValues *xlong =  ncfile->get_var(getName(coordinateMap.find(LONGITUDE)->second))->get_rec(time);
    NcValues *xlat  =  ncfile->get_var(getName(coordinateMap.find(LATITUDE )->second))->get_rec(time);

    osg::Image *coord  = new osg::Image;

	std::cout << "getCoordinate 4" << std::endl;

	coord->allocateImage((int)x, (int)y, (int)z, GL_RGB, GL_FLOAT);
    float zoffset = 0;
	if (WGRID == gridType)
	{
		zoffset = -0.5;
	}

	std::cout << "getCoordinate 5" << std::endl;

    unsigned int index = 0;
	for (unsigned int k = 0; k < z; ++k) {
		long xlongindex = 0;
		for (unsigned int j = 0; j < y; j++) {
			for (unsigned int i = 0; i < x; ++i)
			{
				//((osg::Vec3 *)coord->data())[index].set(xlong->as_float(xlongindex), xlat->as_float(xlongindex), zoffset + k);
				((osg::Vec3 *)coord->data())[index].set(xlong->as_float(i), xlat->as_float(j), zoffset + k);
				index++;
				xlongindex++;
				//std::cout << xlong->as_float(i) << "  " << xlat->as_float(j) << " " << xlongindex << std::endl;
			}
		}
	}
	std::cout << "getCoordinate 6" << std::endl;
    return coord;
}

bool WRFNetCDFOperator::haveVar(const NcFile *ncfile,const unsigned int &name)const{
    bool have = false;
    const unsigned int varNum = (unsigned int )ncfile->num_vars();
    for(unsigned int i = 0; i < varNum; ++i)
    {
        if(getName(name) == string(ncfile->get_var(i)->name()))
        { have = true; break;}
    }
    return have;
}

bool WRFNetCDFOperator::haveDim(const NcFile *ncfile,const unsigned int &name)const{
    bool have = false;
    const unsigned int dimNum = (unsigned int )ncfile->num_dims();
    for(unsigned int i = 0; i < dimNum; ++i)
    {
        if(getName(name) == string(ncfile->get_dim(i)->name()))
        { have = true; break;}
    }
    return have;
}
