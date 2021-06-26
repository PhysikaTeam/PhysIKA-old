#pragma once
#include"cuda_runtime.h"
#include"shell1st.h"
#include"discrete1st.h"
#include"beam1st.h"
#include"solid1st.h"
#include"contactnode.h"
#include"solid2nd.h"
#include"beam2nd.h"
#include"shell2nd.h"

struct ShellElementCuda;
struct BeamElementCuda;
struct SolidElementCuda;
struct DiscreteCuda;
struct ContactNodeCuda;
struct SegmentCuda;
struct Element;

typedef struct NodeCuda
{
	int file_id;
	int store_id;
	double coord[6];
	double ini_coord[3];
	double thick;

//#define maxShellList 10
//#define maxBeamList 4
//#define maxSolidList 70
//#define maxDiscreteList 3

	/**
	自由度预索引单元
	*/
	/*ShellElementCuda* shellList[maxShellList];
	BeamElementCuda* beamList[maxBeamList];
	SolidElementCuda* solidList[maxSolidList];
	DiscreteCuda* discreteList[maxDiscreteList];*/

	/**
	主节点信息，默认所有节点为从节点
	*/
	NodeCuda *master_node;
	int master_node_id;

	/**
	索引的位置信息
	*/
	/*int shellNdex[maxShellList];
	int solidNdex[maxSolidList];
	int beamNdex[maxBeamList];
	int discreteNdex[maxDiscreteList];*/

	/**
	索引的数目
	*/
	/*int shellNum;
	int solidNum;
	int beamNum;
	int discreteNum;*/

#define MAXELEMENTNUM 72

	int elementNum_;
	int elementIndex_[MAXELEMENTNUM];
	Element* elementList_[MAXELEMENTNUM];
	double elmContriAccumulate_;
	
	/**
	由节点生成的参与接触的碰撞点
	*/
	ContactNodeCuda *contact_node;

	/**
	平移质量
	*/
	double translate_mass;

	/**
	旋转质量
	*/
	double rotate_mass[3];

	/**
	初始质量参数
	*/
	double ini_tran_mass;
	double ini_rota_mass[3];

	/**
	接触力计算质量
	*/
	double effective_mass;

	/**
	参与接触时,该节点的罚因子
	*/
	double interfaceStiff;

	/**
	内力
	*/
	double int_force[6];

	/**
	外力
	*/
	double ext_force[6];

	/**
	节点应力
	*/
	double stress[6];
	double strain[6];

	double joint_force[6];
	double accel[6];
	double vel[6];
	double dfn[3];
	double disp[6];
	double disp_incre[6];

	/**
	节点边缘标志
	*/
	int edge_sign;
	SegmentCuda *parent_segment;

	/**
	焊点标志
	*/
	int weld_sign;
	int spotweld_OneNode;

	/**
	刚体标志
	*/
	int rigid_sign;

	/*
	* 旋转自由度标志
	*/
	int rotate_sign;

	__device__ __host__ NodeCuda();

	/**
	重置节点的<符号
	*/
	bool operator <(const NodeCuda &nodeCuda)const;

	/**
	节点时间增量
	*/
	double dt_node;

	/**
	节点加速度更新，清除节点力，清除接触力
	*/
	__host__ __device__ void accelVelUpdate(const double dt, const double previous_dt);

	/**
	节点速度及位移等变量更新
	*/
	__host__ __device__ void dispUpdate(const double dt, const int ndf);

	/*
	 * 清除节点应力应变与单元累积统计
	 */
	__host__ __device__ void clearStressStrain();

	/**
	建立索引信息
	*/
	__device__ void buildShellIndexGpu(ShellElementCuda *shell, int num);
	__device__ void buildBeamIndexGpu(BeamElementCuda *beam, int num);
	__device__ void buildSolidIndexGpu(SolidElementCuda *solid, int num);
	__device__ void buildDiscreteIndexGpu(DiscreteCuda *discrete, int num);

	__device__ void buildElementIndexGpu(Element *el, int num);

	/**
	根据索引信息计算节点内力
	*/
	__host__ __device__ void calIndexInterFor();

	/**
	根据索引信息计算节点应力应变
	*/
	__host__ __device__ void calNodeStressStrain();

	/**
	保存初始坐标
	*/
	__host__ __device__ void storeIniCoord();

	/**
	质量还原
	*/
	__host__ __device__ void massReduction();

	/**
	从节点质量附加到主节点上
	*/
	__device__ void massAttachedToMainNode(int type);

	/**
	保存等效质量
	*/
	__host__ __device__ void storeEquivalentMass();

	/**
	保存初始质量
	*/
	__host__ __device__ void storeInitialMass();
} NodeCuda;