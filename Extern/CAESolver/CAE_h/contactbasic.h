#pragma once
#include"surface.h"
#include"contactnode.h"
#include"surfacemanager.h"
#include"setmanager.h"
#include "part.h"
#include"contactnodemanager.h"
#include"SurfaceInteract.h"
#include <cublas.h>

#define controlDistance 0.0
#define cds2 controlDistance*controlDistance

struct ContactNodeManager;
struct Part;
struct Surface;
struct SetManager;
struct SurfaceManager;
struct SurfaceInteract;

/**
tol:预估系数, 当等于0, 表示不进行预估, 每次搜索都形成新的接触对, 计算时间最长, 精度最好
当等于1时,进行最乐观的估计,计算时间最短,精度最差
建议取值为0.5至0.7之间
*/
#define SearchPredictCoef 0.2

struct ContactNodeCuda;

typedef struct Contact
{
	enum Type
	{
		nodeToSurface, surfaceToSurface, singleSurface, general, shellEdgeToSurface
	};

	Type contactType;

	int id;

	int slaveId;
	int masterId;
	int slaveType;
	int masterType;
	double fnu;

	//ls:2020-03-17
	/**
	maximum prescribed velocity
	*/
	double vmax;

	/**
	control distance
	*/
	double controlDist;

	/**
	Static coefficient of friction
	*/
	double FS;

	/**
	Dynamic coefficient of friction
	*/
	double FD;

	/**
	Birth time (contact surface becomes active at this time).
	*/
	double BT;

	/**

	*/
	double DT;//Death time (contact surface is deactivated at this time).

			  /**
			  Scale factor on default slave penalty stiffness
			  */
	double SFS;

	/**
	Scale factor on default master penalty stiffness
	*/
	double SFM;

	/**
	*/
	double SST;//Optional thickness for slave surface (overrides true thickness

			   /**
			   Optional thickness for master surface (overrides true thickness)
			   */
	double MST;

	/**
	*/
	double SFST;//Scale factor for slave surface thickness (scales true thickness).

				/**
				Scale factor for master surface thickness (scales true thickness).
				*/
	double SFMT;

	/**
	Coulomb friction scale factor.
	*/
	double FSF;

	/**
	Viscous friction scale factor
	*/
	double VSF;

	//KDH
	int SOFT;
	double SOFSCL;
	int LCIDAB;
	double MAXPAR;
	double SBOPT;
	int DEPTH;
	int BSORT;
	int FRCFRQ;

	double PENMAX;
	int THKOPT;
	int SHLTHK;
	int SNLOG;
	int ISYM;
	int I2D3D;
	double SLDTHK;
	double SLDSTF;

	//ls:2020-04-06
	int IGAP;
	int IGNORE0;
	double DPRFAC;
	double DTSTIF;
	double UNUSED;
	double FLANGL;
	int CID_RCF;
	string title;
	int SBOXID, MBOXID, SPR, MPR;
	//
	//MPP
	int BCKT;
	int LCBCKT;
	int NS2TRK;
	int INITITR;
	double PARMAX;
	int CPARM8;
	//

	/**
	点对面形成的碰撞点
	*/
	int contact_node_num;
	vector<ContactNodeCuda*> slave_contact_node_array_cpu;
	ContactNodeCuda** slave_contact_node_array_gpu;

	/*
	* 多gpu接触点
	*/
	vector<ContactNodeCuda**> slaveContactNodeArrayMultiGpu;
	vector<int> slaveContactNodeNumMultiGpu;

	/**
	surfaceToSurface
	*/
	Surface* slaveSurface;
	Surface* masterSurface;

	bool isNeedJudge;

	/**
	nodeToSurface
	*/
	vector<NodeCuda *> slave_node_array;

	/**
	所有的碰撞点
	*/
	vector<ContactNodeCuda *> AllContactNodeArray;

	/**
	maximum prescribed velocity
	*/
	double velMaxModulus;

	/**
	从接触面最大的位移增量
	*/
	double dispIncre;

	/**
	Static coefficient of friction
	*/
	double static_coeffic_friction;

	/**
	Dynamic coefficient of friction
	*/
	double dynamic_coeffic_friction;

	/**
	Exponential decay coefficien
	*/
	double DC;

	/**
	Coefficient for viscous friction.
	*/
	double VC;

	/**
	Viscous damping coefficient in percent of critical
	*/
	double VDC;

	/**
	Small penetration in contact search option
	*/
	double PENCHK;

	/**
	丛接触面或者从接触点最大速度值
	*/
	double vmax_slave;
	double* velModulusGpu;
	double* velModulusCpu;

	/**
	两个接触单位中最小的预测最早发生接触的迭代次数
	*/
	int minPredicIterNum;
	double* predicIterNumGpu;
	double* predicIterNumCpu;

	Contact();

	/**
	迭代计数
	*/
	int isch;

	/*
	 * 定义产生接触两个面之间的相互作用
	 */
	int surfInterNum_;
	vector<SurfaceInteract*> surfaceInteractCpu_;
	vector<SurfaceInteract**> surfaceInteractGpu_;

	/**
	生成主从接触面
	*/
	void surfaceProudce(SurfaceManager *surfaceManager, vector<Part> &partArray_, SetManager *setManager, NodeManager *nodeManager);

	/**
	生成碰撞点
	*/
	void contactNodeCollect(ContactNodeManager *contactNodeManager);

	/**
	生成gpu上的接触点
	*/
	void produceGpuContactNodeCuda(ContactNodeManager* contact_node_manager);
	void produceMultiGpuContactNode(ContactNodeManager* cnNdMag, vector<int> gpuIdArray);

	/**
	计算从接触元素的最大数据值
	*/
	void calVamxSlaveGpu(cublasHandle_t& cu_handle);
	void CalVmaxSlaveCpu();

	/**
	计算本接触管理器中即将发生接触的的最小迭代次数
	*/
	void calMinCyclesGpu(cublasHandle_t& cu_handle, const double dt);
	void CalMinCyclesCpu(const double dt);

	void imposeSpecificInteractCpu(double dt);

	void imposeSpecificInteractGpu(double dt);

	void imposeSpecificInteractMultiGpu(FEMDynamic *domain, double dt);

	/**
	分配零复制内存
	*/
	void allocZeroMemContact();

	virtual ~Contact();

} Contact;

