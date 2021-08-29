#pragma once
#include<string>
#include"materialold.h"
#include"sectionold.h"
#include<vector>
#include"node.h"
#include "Elm2ndManager.h"
#include"elmmanager.h"
#include"materialmanager.h"
#include"sectionmanager.h"

using std::vector;

struct ElmManager;
struct MaterialNew;
struct Material;
struct Section;
struct NodeManager;
struct MaterialManager;
struct SectionManager;
struct FEMDynamic;
struct Surface;
struct SurfaceManager;

typedef struct Part
{
	int id;
	int SECID;
	int MID;
	int EOSID;
	int HGID;
	int GRAV;
	int ADPOPT;
	int TMID;
	std::string name;

	//ls:2020-04-06
	int storeNum;

	//PART_CONTACT
	double FS;
	double FD;
	double DC;
	double VC;
	double OPTT;
	double SFT;
	double SSF;
	//

	//ls:2020-03-17
	/**
	PART_INERTIA 
	*/
	double translationalMass;
	int IRCS;// Flag for inertia tensor reference coordinate system:
	int NODEID;
	int coordinateID;

	//ls:2020-04-06
	double XC;
	double YC;
	double ZC;
	double TM;
	double IXX;//component of inertia tensor
	double IXY;
	double IXZ;
	double IYY;
	double IYZ;
	double IZZ;
	double VTX;
	double VTY;
	double VTZ;
	double VRX;
	double VRY;
	double VRZ;
	double XL;
	double YL;
	double ZL;
	double XLIP;
	double YLIP;
	double ZLIP;
	//

	//

	/**
	四节点壳与三节点壳的数目
	*/
	/*int quad_shell_num;
	int tri_shell_num;*/

	/**
	部件质量
	*/
	double part_mass;

	/**
	part中得单元管理器
	*/
	ElmManager *elmManager;

	/**
	part材料
	*/
	Material *material;
	Material *materialGpu;
	
	MaterialNew *matNew;
	MaterialNew *matNewGpu;

	/**
	part属性
	*/
	Section *section;
	Section *sectionGpu;

	/**
	cpu端part中的节点
	*/
	vector<NodeCuda *> node_array_cpu;

	/**
	gpu端part中的节点
	*/
	NodeCuda **node_array_gpu;

	/*
	 * 从part上生成的surface；
	 */
	int surfaceId;
	Surface *surface;
	

	Part();

	/**
	统计part中的节点
	*/
	void setPartNode(NodeManager *node_manager);

	//ls:2020-04-18 added (为了补全集合)
	void partNodelist(NodeManager *node_manager, std::vector<int> node_list);
	//

	/**
	生成gpu端的节点
	*/
	void gpuNodeCreate(NodeManager *node_manager);

	/**
	part成员设置
	*/
	void memberSet(vector<ElmManager> &elm1stManager_array,	MaterialManager* materialManager, SectionManager* sectionManager);

	/*
	 * part设置新式材料
	 */
	void setMaterialNew(MaterialManager* materialManager);

	/**
	材料及属性复制
	*/
	void matPrpCpyGpu(MaterialManager *mat_manager, SectionManager *section_manager);

	/*
	 * 计算part中所有单元的积分点权重与自然坐标
	 */
	void calElementGaussCoordWeight(SectionManager* secManger);

	/**
	设置部件中节点的厚度
	*/
	void setNodeThick();

	/**
	多gpu分割部件
	*/
	void splitPartMultiGpu(int totNdNum, int num_gpu, vector<int> &elm_num_gpu);

	/*
	 * 计算part中的最小单元特征长度
	 */
	void computeElementCharactLengthCpu(double& min_lg);
	void computeElementCharactLengthGpu(double& min_lg);

	/*
	 * 从part表面生成面
	 */
	void produceSurface(FEMDynamic *domain,Surface *surf,SurfaceManager *surfMag);

	void setSubSurface(Surface *sf,int sfId);

	virtual ~Part();
} Part;