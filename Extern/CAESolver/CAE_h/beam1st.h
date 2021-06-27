#pragma once
#include"cuda_runtime.h"
#include"elementtypeenum.h"
#include"node.h"
#include "materialold.h"
#include"hourglasscontrol.h"
#include"element.h"

struct NodeCuda;
struct MaterialNew;
struct Section;
struct Material;
struct HourglassControl;
struct Element;

typedef struct BeamElementCuda:Element
{
	NodeCuda* node_array[3];
	int node_id_array[3];

	int gpu_id;

	double ini_length;
	double ini_area;
	double strs[28];

	double strain_energy;
	double total_energy;

	double area;
	double element_mass;
	double element_rot_mass[3];

	/**
	当前长度值
	*/
	double length;

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
	自由度预索引所需的内存,二维数组第一个指标表示节点，第二个指标表示自由度
	*/
	double fint[3][6];

	/**
	两端梁截面的积分点
	*/
	double strain_xx_i[9];
	double strain_xx_j[9];
	double strain_xy_i[9];
	double strain_xy_j[9];
	double stress_xx_i[9];
	double stress_xx_j[9];
	double stress_xy_i[9];
	double stress_xy_j[9];

	/**
	梁横截面上的积分点的应力与等效应变
	*/
	double ys_k_i[9];
	double ys_k_j[9];
	double pstn_i[9];
	double pstn_j[9];

	/**
	两端塑性迭代计数
	*/
	int kount_i[9];
	int kount_j[9];

public:
	__host__ double computeBeamMass(Material *matPtr,Section *sectionPtr);

	/*
	 *计算特征长度
	 */
	__host__ __device__ double computeBeamCharactLength();

	/*
	 *计算时间步长
	 */
	__host__ __device__ double computeBeamTimeStep(MaterialNew *material, double safe_factor);

	/*
	 *指定当前单元的时间增量
	 */
	__host__ __device__ void assignTimeStep(const double dt);

	/*
	 * 梁单元质量缩放
	 */
	__device__ double computeBeamMassScaleGpu(const double min_dt);

	__host__ double computeBeamMassScaleCpu(const double min_dt);

	/*
	 * 梁单元求解内力
	 */
	__device__ void computeBeamInterForceGpu(MaterialNew *material, Section *section, HourglassControl *hourglass, double dt);
	__host__ void computeBeamInterForceCpu(MaterialNew *material, Section *section, HourglassControl *hourglass, double dt);

	/*
	 * 梁单元节点匹配
	 */
	__host__ __device__ void bulidNodeMatch(NodeCuda *nodeArraySys);

	/*
	 * 梁单元自由度预处理
	 */
	__device__ void builNodeIndex();

	/*
	 * 梁单元求罚参数
	 */
	__host__ __device__ void computeBeamPenaltyFactor(double factor,Material *material,Section *section);

	/**
	计算梁单元的应变能
	*/
	__host__ __device__ void calStrainEnergy(double ddln, double rot1[], double rot2[], double f21, double oldf21, double fm[][3], double oldfm[][3]);

	/**
	单元质量初始化
	*/
	__host__ __device__ void beamNodeMassInitia();

	/**
	设置初始屈服应力
	*/
	__host__ __device__ void setInitialYield(const double iniYield);

	__host__ __device__ void setElementSectionType();

	/*
	* 梁单元内力离散到节点上
	*/
	__device__ void discreteBeam1stInterForceToNodeGpu();
	
private:

	/**
	梁单元米塞斯应变应力法则
	*/
	__host__ __device__ void beamMises(const MaterialNew *elmPrp, double dexx, double dexo, double &sxx, double &sxo,
		double beam_sxx[], double beam_sxo[], double beam_ys[], int beam_kount[], int ipt);

	/**
	单元质量矩阵求解
	*/
	__host__  double beamMassCalCpu(const double rho, const double area, const double sm1, const double sm2, const double smt);

	/**
	弹簧元质量矩阵求解
	*/
	__host__  double springMassCalCpu(double rho, double volumeSpring_, double iner);


	/**
	计算本地坐标系
	*/
	__host__ __device__ void calBeamLocateAxi(double e32[], double e31[], double e1[], double e2[], double e3[], double rot1[], double rot2[],
		double &aln, double dt, double e21[], double e22[]);

	/**
	find rigid body rotation and local deformed nodal values
	*/
	__host__ __device__ void brigid(double dt, double &ddln, double dcx[], double dcy[], double dcz[], double rot1[], double rot2[],
		double aln, double e21[], double e22[], double e32[], double e31[]);

	/**
	弹性梁计算
	*/
	__host__ __device__ void bsBeamElasticInterForce(const MaterialNew *material, const Section *section, double &dln, double &ddln, double f[][3], double fm[][3],
		double rot1[], double rot2[], double dt, double length);

	/**
	弹塑性梁计算
	*/
	__host__ __device__ void bsBeamPlasticInterForce(const MaterialNew *material, const Section *section, double &dln, double &ddln, double f[][3], double fm[][3],
		double rot1[], double rot2[], double dt, double length);
	__host__ __device__ void open_cross(double ddln, double rot1[3], double rot2[3], double aln, double fm[3][3], double f[3][3], double dt,
		const MaterialNew *elmPrp, const Section *section);
	__host__ __device__ void close_cross(double ddln, double rot1[3], double rot2[3], double aln, double fm[3][3], double f[3][3], double dt,
		const MaterialNew *elmPrp, const Section *section);

	/**
	梁单元内力求解
	*/
	__device__ void interForce_bsbeam_Gpu(const MaterialNew *material, const Section *section, const double dt);
	__host__ void interForce_bsbeam_Cpu(MaterialNew *material, Section *section, const double dt);

	/**
	杆单元内力求解
	*/
	__device__ void interforce_truss_gpu(const MaterialNew *material, const Section *section, const double dt);

	/**
	弹簧元内力求解
	*/
	__device__ void interForce_spring_Gpu(const MaterialNew *material, const Section *section, const double dt);
	__host__ void interForce_spring_Cpu(MaterialNew *material, Section *section, const double dt);
	__host__ __device__ void cal_spring_disp(double disp[3], double rot_disp[3], double e1[], double e2[], double e3[]);
	__host__ __device__ void cal_spring_for(double dt, double dcx[], double dcy[], double dcz[], double f[3][3],
		double fm[3][3], double rot1[3], double rot2[3], const MaterialNew *elmPrp);


} BeamElementCuda;
