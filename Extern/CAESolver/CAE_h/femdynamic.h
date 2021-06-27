#pragma once
#include"rigidmanager.h"
#include"boundarymanager.h"
#include "elmmanager.h"
#include"setmanager.h"
#include"tiemanager.h"
#include"loadmanager.h"
#include"elementcontrol.h"
#include"constrainedmanager.h"
#include"timecontrol.h"
#include"multigpumanager.h"
#include"contactmanager.h"
#include"curvemanager.h"
#include"initialmanager.h"
#include"materialmanager.h"
#include"sectionmanager.h"
#include"hita.h"
#include"unified_contact.h"
#include"nodemanager.h"
#include"part.h"
#include"hourglasscontrolmanager.h"
#include"outputmanager.h"
#include<vector>
#include "computedevice.h"
//ls:2020-03-18
#include "accuracycontrol.h"
#include "contactcontrol.h" 
#include "historynode.h" 
#include "cpucontrol.h"
#include "energycontrol.h"
#include "extentbinary.h"
#include "outputcontrol.h"
#include "shellcontrol.h"
#include "solidcontrol.h"
#include "Airbagfile.h"
#include "AirbagManager.h"
#include "ElementMass.h"
#include "Seatbelt_Accelerometer.h"
#include "Tablemanager.h"
#include "coor_nodesmanager.h"
#include "inc_transformanager.h"
#include "def_transformanager.h"
#include "elementmassmanager.h"
#include "accelermetermanager.h"
//
//ls:2020-04-06
#include "boxmanager.h"
#include "controlmanager.h"
#include "databasemanager.h"
#include "rigidwallmanager.h"
#include "historynodemanager.h"
//#include "rigidwall.h"
//
#include "rigidExtraNodeManager.h"

using std::vector;

struct OutputManager;
struct ContactManager;
struct NodeManager;
struct ContactNodeManager;
struct PairManager;
struct SurfaceManager;
struct MaterialManager;
struct SectionManager;
struct CurveManager;
struct ElementMassManager;
struct RigidManager;
struct BoundaryManager;
struct RigidwallManager;
struct LoadManager;

struct RigidExtraNodeManager;

typedef struct FEMDynamic
{
	double *globalSpeed;
	double *globalDisp;
	double *globalAcceleration;
	double *globalMass;
	double *globalJointForce;
	double *globalDispIncre;
	double *cc;

	/**
	时间安全因子
	*/
	double scl;
	int *dofBounSign;
	int node_amount;
	int elementAmout;

	int cyc_num;
	double dt;
	double previous_dt;
	double current_time;

	string intFilePath_;
	string outputFilePath_;

	ComputeDevice calDevice_;

	vector<Part> partArray_;
	vector<ElmManager> elmentManager_array;
	NodeManager *nodeManager;
	ContactNodeManager *contactNodeManager;
	PairManager *pair_manager;
	SurfaceManager* surfaceManager;
	MaterialManager* materialManager;
	SectionManager *sectionManager;
	CurveManager *curveManager;
	//ls:2020-03-17
	TableManager *tableManager;
	Coor_nodesManager *coor_nodesManager;
	Def_transforManager *def_transforManager;
	Inc_transforManager *inc_transforManager;
	//
	//ls:2020-03-18
	RigidwallManager *rigidwallManager;
	HistorynodeManager *historynodeManager;
	AirbagManager *airbagManager;
	ElementMassManager *elementMassManager;
	AccelermeterManager *accelermeterManager;
	//
	//ls:2020-04-06
	BoxManager *boxManager;
	ControlManager *controlManager;
	DatabaseManager *databaseManager;
	//
	SetManager *setManager;
	BoundaryManager *boundManager;
	LoadManager *loadManager;
	ConstrainedManager *constraManager;
	InitialManager *initialManager;
	TimeControl *timeControl;
	RigidManager *rigidManager;
	ElementControl *elmControl;
	HourglassControlManager *hgControlManager;  //ls:2020-04-10
	TieManager *tieManager;
	IntegratePointManager *integraPointManager;
	SmoothFemManager *smoothFemManager;
	OutputManager* outputManager;

	RigidExtraNodeManager *rigidExtraNodeManager;

	//ls:2020-04-06
	map<int, int>partID_ls; //part
	map<int, int>node_id_ls; //node
	map<int, int>element_id_ls; // element
	map<int, int>surface_id_ls; //surface
	//

	FEMDynamic();

	/**
	多gpu管理器
	*/
	MultiGpuManager *multiGpuManager;

	/**
	接触管理，提供统一的读取，并实现最基本的主从面接触算法
	*/
	ContactManager *contactManager;

	/**
	并行接触算法：一体化法接触算法
	*/
	UnifiedContactManager *unifiedContactManager;

	/**
	串行接触算法：级域法接触算法
	*/
	HitaContactManager *hitaManager;

	/**
	接触碰撞多gpu并行计算
	*/
	void DynamicAnalysisParallelMultiGpu(std::ofstream &fout, double &cnTime, double& fileTime);

	/**
	接触碰撞显式有限元Gpu并行计算
	*/
	void DynamicAnalysisParallel(std::ofstream &fout, double& cnTime, double& fileTime);

	/**
	接触碰撞显式式有限元计算
	*/
	void DynamicAnalysis(std::ofstream &fout, double &cnTime, double& fileTime);

	/**
	cpu端求解接触块的罚因子
	*/
	void cpuCalSegmentPenaFactor(const double pena_scale);

	/**
	多gpu并行计算相关内容
	*/
	/**
	多gpu初始化
	*/
	bool initialComputeMultiGpu();

	/**
	多gpu显存分配
	*/
	void allocateAndCopyMultiGpu();

	/*
	 * 单GPU初始化
	 * 包括单gpu的接触算法的初始化，零拷贝内存的分配
	 */
	bool initialComputeSingleGpu();

	/*
	 * 单GPU显存分配与复制
	 * 各类需要计算的的数据，如单元节点材料截面属性等的复制
	 * 各类约束信息如焊接绑定等数据的复制
	 */
	void allocateAndCopySingleGpu();

	/*
	 * cpu初始化
	 * 包括接触搜寻前的数据准备，质量计算，焊接绑定等关系的初始化等
	 * 包括TL中B矩阵的计算
	 */
	bool initialComputeCpu();

	/**
	内存分配
	*/
	void femAllocateCpu();

	/**
	内存释放
	*/
	void femFreeCpu();

	~FEMDynamic();

} FEMDynamic;
