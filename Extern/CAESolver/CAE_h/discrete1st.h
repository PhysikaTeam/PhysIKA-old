#pragma once
#include"cuda_runtime.h"
#include"node.h"
#include"elementtypeenum.h"
#include "materialold.h"
#include"sectionold.h"
#include"element.h"
#include"hourglasscontrol.h"

struct HourglassControl;
struct Section;
struct Material;
struct MaterialNew;
struct NodeCuda;
struct Element;

typedef struct DiscreteCuda:Element
{
	int node_id_array[3];
	NodeCuda* node_array[3];
	
	int gpu_id;

	/**
	单元内力
	*/
	double fint[2][6];

	/**
	轴向矢量
	*/
	double axi_vec[3];

	/**
	当前长度值
	*/
	double length;

	/**
	轴向变型率
	*/
	double axi_deform_rate;

	/**
	伸长量
	*/
	double length_change;

	/**
	轴向力
	*/
	double axi_force;

	/**
	单元时间增量
	*/
	double dt_ele;

	/**
	质量缩放因子
	*/
	double mass_scale_rate;

public:

	__host__ double computeDiscreteMass(Material *material,Section *section);

	/*
	*计算特征长度
	*/
	__host__ __device__ double computeDiscreteCharactLength();

	/*
	*计算时间步长
	*/
	__host__ __device__ double computeDiscreteTimeStep(Material *material, double safe_factor);

	/*
	*指定当前单元的时间增量
	*/
	__host__ __device__ void assignTimeStep(const double dt);

	/*
	* 单元质量缩放
	*/
	__device__ double computeDiscreteMassScaleGpu(const double min_dt);

	__host__ double computeDiscreteMassScaleCpu(const double min_dt);

	/*
	* 单元求解内力
	*/
	__device__ void computeDiscreteInterForceGpu(MaterialNew *material, Section *section, HourglassControl *hourglass, double dt);
	__host__ void computeDiscreteInterForceCpu(MaterialNew *material, Section *section, HourglassControl *hourglass, double dt);

	/*
	* 单元节点匹配
	*/
	__host__ __device__ void bulidNodeMatch(NodeCuda *nodeArraySys);

	/*
	* 单元自由度预处理
	*/
	__device__ void builNodeIndex();

	/*
	* 单元求罚参数
	*/
	__host__ __device__ void computeDiscretePenaltyFactor(double factor, Material *material, Section *section);

	/**
	单元质量初始化
	*/
	__host__ __device__ void DiscreteNodeMassInitia();

	__host__ __device__ void setElementSectionType();

	__device__ void discreteDiscrete1stInterForceToNodeGpu();

private:
	/**
	计算单元内力
	*/
	__device__ void interForce_discrete_Gpu(MaterialNew *material, Section *section, double dt);
	__host__ void interForce_discrete_Cpu(MaterialNew *material, Section *section, double dt);

	/**
	计算单元轴向向量
	*/
	__host__ __device__ void calAxiVec(double &du);

	/**
	单元内力组装
	*/
	__host__ __device__ void assemInterForce();

	/**
	根据位移或者速度计算内力
	*/
	__host__ __device__ void calInterForce(const MaterialNew *material, const double dt, const double du);

	/**
	求解轴向形变率
	*/
	__host__ __device__ void calAxiDeformRate(const double dt);

}DiscreteCuda;