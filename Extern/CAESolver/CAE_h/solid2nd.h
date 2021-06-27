#pragma once
#include"cuda_runtime.h"
#include"node.h"
#include"elementtypeenum.h"
#include"element.h"
#include"materialnew.h"
#include"sectionold.h"

struct Section;
struct NodeCuda;
struct Element;

typedef struct SolidElement2nd:Element
{

	int nodeIdArray_[20];
	NodeCuda* nodeArray_[20];

	int gpuId_;

	double dtEle_;

	double lengthChart_;

	double massScaleRate_;

	double volume_;

	double translateMass_;

	double rotateMass_[3];

	double fint[20][3];

	/*
	 * 设置单元截面属性类型
	 */
	__host__ __device__ void setElementSectionType();

	/*
	 * 计算单元体积
	 */
	__host__ __device__ double calSolid2ndVolume(const int gaussNum);

	/*
	 * 计算积分点形函数相乘的累加
	 */
	__host__ __device__ void computeTetr2ndNNMatrix(double *nn,const int gaussNum);

	/*
	 * 计算积分点的形函数
	 */
	__host__ __device__ void calTetr2ndGaussShape(GaussPoint *gp,double *shp);
	__host__ __device__ void calHexa2ndGaussShape(GaussPoint *gp,double *shp);
	__host__ __device__ void calPent2ndGaussShape(GaussPoint *gp,double *shp);

	/*
	* 计算积分点的雅克比矩阵
	*/
	__host__ __device__ void computeHexa2ndGaussJacobMat(GaussPoint *gp, double jacobMat[20][3])const;
	__host__ __device__ void computePent2ndGaussJacobMat(GaussPoint *gp, double jacobMat[15][3])const;
	__host__ __device__ void computeTetr2ndGaussJacobMat(GaussPoint *gp, double jacobMat[10][3])const;

	/*
	* 计算积分点本地形函数偏导
	*/
	__host__ __device__ void calHexa2ndGaussLocalDerivative(GaussPoint *gp, double dN[20][3])const;
	__host__ __device__ void calPent2ndGaussLocalDerivative(GaussPoint *gp, double dN[15][3])const;
	__host__ __device__ void calTetr2ndGaussLocalDerivative(GaussPoint *gp, double dN[10][3])const;

	/*
	* 计算积分点所属积分域体积
	*/
	__host__ __device__ void computeHexa2ndGaussVolume(GaussPoint *gp, double &vol)const;
	__host__ __device__ void computePent2ndGaussVolume(GaussPoint *gp, double &vol)const;
	__host__ __device__ void computeTetr2ndGaussVolume(GaussPoint *gp, double &vol)const;

	/*
	 * 内力计算
	 */
	__host__ void computeSolid2ndInterForceCpu(MaterialNew* material, Section* section, HourglassControl* hourglass, double dt);
	__device__ void computeSolid2ndInterForceGpu(MaterialNew* material, Section* section, HourglassControl* hourglass, double dt);

	__device__ void interForceTet2ndMGpu(const MaterialNew* material, const Section* section, const double dt);
	__host__ __device__ void calBMatrixTet2ndM(GaussPoint* gp, double bMat[3][10], double ncd[10][3]);
	__host__ __device__ void calInterForTet2ndM(GaussPoint *gp, double bMat[3][10], const MaterialNew *mat, double dt, double ft[10][3], double ncv[10][3]);

	__device__ void interForceTet2ndGpu(const MaterialNew* material, const Section *section, const double dt);
	__host__ __device__ void calBMatrixTet2nd(GaussPoint *gp,double bMat[3][10]);
	__host__ __device__ void calInterForTet2nd(GaussPoint *gp, double bMat[3][10],const MaterialNew *mat, double dt,int ig);

	__device__ void interForceHex2ndGpu(const MaterialNew* material, const Section *section, const double dt);
	__host__ __device__ void calBmaterixHex2nd(GaussPoint *gp,double bMat[3][20]);

	
	__device__ void interForcePenta2ndGpu(const MaterialNew* material, const Section *section, const double dt);
	__host__ __device__ void calBMaterixPenta2nd(GaussPoint *gp,double bMat[3][15]);

	/*
	 * 时间步长计算
	 */

	__host__ __device__ double computeSolidTimeStep(const double wave_num, const double safe_factor);

	/*
	 * 高阶实体单元的特征长度提供两种算法
	 * 一：以单元中任意两节点之间的最短距离为特征长度
	 * 二：求解相应的低阶单元的特征长度，再以此除以2
	 */

	__host__ __device__ double computeSolid2ndCharactLength();

	__host__ __device__ double computeHexa2ndCharactLength();

	__host__ __device__ double computePent2ndCharactLength();

	__host__ __device__ double computeTetr2ndCharactLength();

	__host__ __device__ void assignTimeStep(const double dt);


	/*
	 * 质量缩放
	 */
	__host__ __device__ void solid2ndNodeMassIntia();

	__device__ double computeSolid2ndMassScaleGpu(const double minDt);

	__host__ double computeSolid2ndMassScaleCpu(const double minDt);

	/*
	 * 建立索引
	 */

	__device__ void builNodeIndex();

	__host__ __device__ void bulidNodeMatch(NodeCuda *node_array_gpu);

	/*
	 * 单元质量计算
	 */
	__host__ double computeSolid2ndMass(Material *matPtr, Section *sectionPtr);

	/*
	 * 计算节点的应力应变
	 */
	__device__ void computeSolid2ndNodeStrGpu(const int gaussNum);
	 __device__ void computeTetr2ndNodeStressStrainGpu(const int gaussNum);

	 /*
	 * 离散单元内力到节点上
	 */
	 __device__ void discreteSolid2ndInterForceToNodeGpu();

	 __host__ void discreteSolid2ndInterForceToNodeCpu();

private:

	/*
	 * 计算单元体积，目前只是近似地求解其对应低阶单元的体积
	 */
	__host__ __device__ void calPent2ndVol(double &vol);
	__host__ __device__ void calHexa2ndVol(double &vol);
	__host__ __device__ void calTetr2ndVol(double &vol);

}SolidElement2nd;
