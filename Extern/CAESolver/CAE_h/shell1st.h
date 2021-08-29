#pragma once
#include"cuda_runtime.h"
#include"node.h"
#include"elementtypeenum.h"
#include"materialold.h"
#include"sectionold.h"
#include"hourglasscontrol.h"
#include"element.h"

struct NodeCuda;
struct Element;

typedef struct ShellElementCuda:Element
{

	int node_id_array[4];
	NodeCuda* node_array[4];
	
	int gpu_id;

	/**
	自由度预索引所需的内存，二维数组第一个指标表示节点，第二个指标表示自由度
	*/
	double fint[4][6];

	double ini_area;
	double area;
	double ini_thick;
	double thick;

	double strain_energy;
	double total_energy;

	/**
	单元时间增量
	*/
	double dt_ele;

	/**
	单元特征长度
	*/
	double charact_length;

	/**
	质量缩放系数
	*/
	double mass_scale_rate;

	/**
	质量及旋转质量
	*/
	double element_mass;
	double element_rot_mass[3];

	/**
	沙漏控制
	*/
	double qs[5];

public:

	/*
	 * 壳单元求解质量
	 */
	__host__ double computeShellMass(Material *matPtr, Section *sectionPtr);

	/**
	时间步长相关数值求解
	*/
	__host__ __device__ double computeShellCharactLength();

	__host__ __device__ double computeShellTimeStep(MaterialNew *material, double safe_factor);

	/**
	指定当前单元的时间增量
	*/
	__host__ __device__ void assignTimeStep(const double dt);

	/**
	壳单元质量缩放
	*/
	__device__ double computeShellMassScaleGpu(const double min_dt);
	__host__ double computeShellMassScaleCpu(const double min_dt);

	/*
	* 壳单元求解内力
	*/
	__device__ void computeShellInterForceGpu(const MaterialNew *material, Section *section, HourglassControl *hourglass, double dt);
	__host__ void computeShellInterForceCpu(const MaterialNew *material, Section *section, HourglassControl *hourglass, double dt);

	/**
	节点匹配
	*/
	__host__ __device__ void bulidNodeMatch(NodeCuda *node_array_gpu);

	/**
	建立索引信息
	*/
	__device__ void builNodeIndex();

	/**
	罚参数求解
	*/
	__host__ __device__ void computeShellPenaltyFactor(const double penalty_scale_factor, Material *material, Section *section);

	/**
	单元质量初始化
	*/
	__host__ __device__ void shellNodeMassInitia();

	/**
	求解单元的面积
	*/
	__host__ __device__ double calElementArea();

	__host__ __device__ void setElementSectionType();

	/*
	* 离散单元内力到节点上
	*/
	__device__ void discreteShell1stInterForceToNodeGpu();

	/*
	ls:2020-07-30 added
	*/
	__host__ __device__ void applyUniformNormLoadToNode(const double loadValue);

	/*
	* 计算面积
	*/
	__host__ __device__ void computeArea(double &ar);

private:
	/**
	壳单元内力求解
	*/
	template<typename anyType>
	__device__ void interForce_BT_Gpu(const MaterialNew *material, Section *section, const double dt);
	__host__ void interForce_BT_Cpu(const MaterialNew *material, Section *section, const double dt);


	__device__ void interForce_HL_Gpu(const MaterialNew *material, const Section *section, const double dt);

	__device__ void interForce_S3_Gpu(const MaterialNew *material, const Section *section, const double dt);
	__host__ void interForce_S3_Cpu(const MaterialNew *material, Section *section, const double dt);


	/**
	膜单元内力求解
	*/
	__device__ void interForce_M4_gpu(const MaterialNew *material, const Section *section, const double dt);
	__device__ void interForce_M3_gpu(const MaterialNew *material, const Section *section, const double dt);

	/**
	计算本地坐标系
	*/
	__host__ __device__ void calTriLocateAxi(double xn[3], double yn[3], double zn[3]);
	__host__ __device__ void calQuadLocateAxi(double xn[3], double yn[3], double zn[3]);

	/**
	三角形壳单元B矩阵求解
	*/
	__host__ __device__ void computeBMatrixCPDSG(const double xcoord[3], const double ycoord[3],
		double bm[3][18], double bb[3][18], double bs[2][18]);
	__host__ __device__ void computeBMatrixDSG(double bm[3][18], double bb[3][18], double bs[2][18],
		double convert_coord[3][3]);

	/*
	 * 三节点与四节点单元求解特征长度
	 */
	__host__ __device__ double computeQuad1stCharactLength();
	__host__ __device__ double computeTria1stCharactLength();

	/**
	bt壳塑性应变求解
	*/
	/*__host__ __device__ void bmain(const int ipt, const double d1[6], const Material *material);*/
	/**
	vectorized routine to find lambda which satisfies the plane stress
	plastic consistency condition using an exact analytical solution
	(only valid for cases not involving isotropic hardening)
	the procedures used in this routine are quite sensitive to machine
	precision, esp.in intrinisic functions.performance should be
	verified before use on non - cray hardware.
	*/
	/*__host__ __device__ void lametx(double eta[], double r, double scle, double alfaht1, double betaht1, double ym, double &xlamkp1);*/

	/**
	三角形单元转换矩阵求解
	*/
	__host__ __device__ void AreaTransMatrixCal_S3();
	__host__ __device__ void transform_S3(double xn_1[3][3],double cos_matrix[3][3],double convert_coord[3][3]);

	/**
	计算应变能
	*/
	__host__ __device__ void calStrainEnergy();

	/**
	设置初始厚度
	*/
	__host__ __device__ void setThick(Section *section);

} ShellElementCuda;