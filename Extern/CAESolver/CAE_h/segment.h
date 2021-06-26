#pragma once
#include"node.h"
#include"surface.h"
#include"boundarytype.h"

/**
计算扩展域的参数
*/
#define EXTFACTOR 0.6
#define THICKFACTOR 0.5
#define MINSEGMENTTHICK 0.001

struct NodeCuda;
struct Surface;

typedef struct SegmentCuda
{
	//ls:2020-04-06
	double A[4];
	//

	/**
	本接触的节点
	*/
	int nodeID_array[4];
	NodeCuda* node_array[4];

	int parentSurfaceId;

	//记录本segment在所有的面中以及当前面中的序号，从0开始
	int global_id;
	int local_id;

	int node_amount;
	double thick;

	/**
	主面的最大位移增量
	*/
	double *dispIncre;

	/**
	相邻接触块及法向
	*/
	SegmentCuda* adjacent_segment_array[4];
	int nb_sg_id[4];
	int nb_sg_normal_array[4];

	/**
	本接触块法向量
	*/
	double nor_vec[3];
	double ini_nor_vec[3];

	/**
	本接触块的罚参数
	*/
	double interfaceStiff;
	double segment_mass;

	/**
	相邻节点编号
	*/
	int node_of_field[20]; //ls:focus

	/**
	扩展域
	*/
	double hi_te_min[3];
	double hi_te_max[3];
	double expand_territory;

	/**
	预测碰撞循环次数
	*/
	int predictCyc;
	int previousPredictCyc;

	/**
	接触块边界标记
	*/
	int lineOfEdgeSign[4];

	/**
	统计本接触块有多少接触点
	*/
	void nodeCount();

	/**
	cpu端接触块连接节点
	*/
	void cpuNodeLink(vector<NodeCuda> &sys_node_array);

	/*
	* 计算块内任意两点之间最大距离
	*/
	__host__ __device__ void computeMaxShellDiagonal(double &msd);

	/*
	* 计算块内任意两点之间最小距离
	*/
	__host__ __device__ void computeMinShellDiagonal(double& msd);
	
	/**
	最大厚度设置
	*/
	__host__ __device__ void maxThickSet();

	/*
	* 施加面上载荷
	*/
	__host__ __device__ void applyUniformNormLoadToNode(const double loadValue);

	/*
	 * 面上施加边界条件
	 */
	__host__ __device__ void imposeNormBoundaryToNode(const double boundValue, BoundaryType bType, double dt, double previous_dt);
	__host__ __device__ void imposeDofBoundaryToNode(const double boundValue, BoundaryType bType, double dt, double previous_dt,int dof);

	/**
	计算本接触块的扩展域
	*/
	__host__ __device__ void calExtTerritory();

	/**
	gpu接触块复制
	*/
	__device__ void gpuSegmentCudaCpy(NodeCuda *system_node_array, SegmentCuda *segment_array);

	/**
	接触块预测碰撞次数的计算
	*/
	__host__ __device__ int calPreictCyc(const double dispIncre);

	/**
	确定本接触块的本地id
	*/
	__host__ __device__ void defineLocalId(const int temp_local_id, const int temp_global_id);

	/**
	计算初始法向量
	*/
	__host__ __device__ void solveIniNorVec();

	/*
	 * 计算法向量
	 */
	__host__ __device__ void solveNormVec();

	/*
	 * 计算扩展域的大小
	 */
	__host__ __device__ void computeExtTerritorySize();

	/*
	 * 计算面积
	 */
	__host__ __device__ void computeArea(double &ar);
} SegmentCuda;

typedef struct SegmentData
{
	int predict_cycle;
	int position;
	int relative_position;
	double cos_value;
	SegmentCuda *test_segment;

}SegmentData;