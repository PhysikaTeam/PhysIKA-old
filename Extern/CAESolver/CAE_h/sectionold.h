#pragma once
#include"elementtypeenum.h"
#include"elmmanager.h"
#include"intergrationdomain.h"
#include"gausspoint.h"
#include"formulatemode.h"
#include <string>
using std::string;

struct ElmManager;
struct GaussPoint;

enum IntegrationRule
{
	Gauss,
	Lobatto,
};

typedef struct Section
{
	GeometricNewType gType_;
	ElementNewType eType_;
	SmoothFEMType sType_;
	IntegrationDomain intType_;
	FormulateMode fMode;

	//ls:2020-03-18
	int typeId;
	int llkt;
	double NIP;
	double thickIntegrPoints;
	double rotool;
	double tktool;
	double qhg;
	double QR;
	string title;
	//

	//ls:2020-04-06
	//shell
	int ICOMP;
	int SETYP;
	double NLOC;
	double MAREA;
	double IDOF;
	int EDGSET;
	double PROPT;
	//

	//ls:2020-04-06
	//beam
	double CST; //Cross section type EQ.0.0: rectangular,EQ.1.0: tubular(circular only),EQ.2.0 : arbitrary(user defined integration rule).
	double SCOOR;
	double NSM;

	//type=2,3
	double A; //Cross-sectional area
	//type=2,12
	double SA; //Shear area.

	double IST;

	//type=6 
	//double volumeSpring_;
	double massInertiaSpring;
	double CID;
	double CA;
	double OFFSET;
	double RRCON;
	double SRCON;
	double TRCON;
	//

	//ls:2020-04-06
	//SECTION_DISCRETE
	int DRO;
	double KD;
	double V0;
	double CL;
	double FD;
	double CDL;
	double TDL;
	//

	//ls:2020-04-06
	//solid
	int aet;                                 //P2805 场元素类型
	//

	int id;
	int elmFormulateOptionDYNA_;

	int addMemSizeForElement;

	int nodeNumForElement;

	int integratePointZ_;
	int integratePointXY_;
	int integratePointNum_;

	IntegrationRule integrationRule_;
	
	double thick[4];
	double shearCorrectFactor_;
	
	int thick_update;

	/*
	 * 内部自由度，基于wilson理论
	 */
	int numInterDof_;

	/**
	梁单元截面上的积分点
	*/
	double beam_inter_point_y[16];
	double beam_inter_point_z[16];

	/**
	弹簧元
	*/
	double volumeSpring_;
	double inertiaSpring;

	/**
	Cross section type EQ.0.0: rectangular,EQ.1.0: tubular(circular only),
	EQ.2.0 : arbitrary(user defined integration rule).
	*/
	double crossSectionType_;

	/**
	膜单元拉伸或者压缩截止
	*/
	double compre_cutoff;

	/**
	outer diameter node 1
	*/
	double TS1;

	/**
	outer diameter node 2
	*/
	double TS2;

	/**
	inner diameter node 1
	*/
	double TT1;

	/**inner diameter node 2
	*/
	double TT2;

	/**
	location of reference surface
	*/
	double NSLOC;

	/**
	location of reference surface
	*/
	double NTLOC;

	/**
	cross sectional area
	*/
	double crossSectionArea_;
	double shearArea_;

	/**
	转动惯量
	*/
	double yInertiaMoment_;
	double zInertiaMoment_;
	double polarInertiaMoment_;

	/**
	轴向响应阻尼系数
	*/
	double pcad;

	/**
	弯曲响应阻尼系数
	*/
	double pcbd;

	/**
	moment of inertia
	*/
	double ISS;

	/**
	moment of inertia about local t-axis
	*/
	double ITT;

	/**
	torsional constant
	*/
	double IRR;

	/**
	bt壳的厚度积分点及权重
	*/
	//double zeta[6][6];
	//double gw[6][6];

	/**
	三角形壳单元厚度积分及权重
	*/
	//double tk_weight[5];
	//double tk_point[5];

	/**
	离散元旋转平移标志
	*/
	int disc_disp_rot_flag;

	/**
	膜单元沙漏控制参数
	*/
	double hourg_para_mem;

	/**
	设置壳单元厚度积分点
	*/
	/*void setShellIntegrPoints();*/

	/*
	 * 必要时对单元的额外内存进行一定操作
	 */
	__host__ __device__ void operatorOnElementAddMem(double *addMemptr);

	/*
	 * 计算单元计算所需的额外内存大小
	 */
	void computeAddMemorySize();
	/**
	求解梁单元所需要的变量
	*/
	void beamFactorCal();

	/**
	设置梁横截面上的积分点
	*/
	void setBeamInterPoint();

	/**
	检查沙漏因子
	*/
	void testHorgParam();

	/*
	 * 计算当前单元类型的积分点个数
	 */
	void computeGaussNum();

	/*
	 * 计算当前单元的内变量个数
	 */
	void computeInterDofNum();

	/*
	 * 计算积分点自然坐标与权重
	 */
	void computeGaussPointNaturalCoord();
	void computeGaussPointWeight();

	/*
	 * 根据dyna单元类型编号确定单元类型
	 */
	void judgeElementTypeForDyna(ElmManager* elm1stManager);

	/*
	 * 根据单元类型判断单元节点个数
	 */
	void judgeElementNodeNumBaseType();

	/*
	 * 根据单元类型判断积分格式
	 */
	void judgeFormulateModeBaseEType();

	/*
	 * 判断单元的积分域形状
	 */
	void judgeElementIntegrateDomainType();

	/*
	 * 根据积分域的形状设置积分点
	 */
	void computeGaussCoordWeight(GaussPoint *gpPtr);

	/*
	 * 计算线性积分域积分点
	 */
	void computeGaussOnLine(GaussPoint *gpPtr);

	/*
	 * 计算三角形积分域积分点
	 */
	void computeGaussOnTriangle(GaussPoint *gpPtr);

	/*
	 * 计算四边形积分域积分点
	 */
	void computeGaussOnSquare(GaussPoint *gpPtr);

	/*
	 * 计算六面体积分域积分点
	 */
	void computeGaussOnCube(GaussPoint *gpPtr);

	/*
	 * 计算四面体积分域积分点
	 */
	void computeGaussOnTetra(GaussPoint *gpPtr);

	/*
	 * 计算三菱柱积分域积分点
	 */
	void computeGaussOnWedge(GaussPoint *gpPtr);

private:
	/*
	 * 线性积分点自然坐标以及权重设置
	 */
	void setLineGaussCoordWeight(int nPoint,double coord[],double weight[]);

	/*
	 * 三角形积分点自然坐标以及权重设置
	 */
	void setTriGaussCoordWeight(int nPoint,double coordXi1[],double coordXi2[],double weight[]);

	/*
	 * 四面体积分点自然坐标以及权重设置
	 */
	void setTetrCoordWeight(int nPoint,double coordXi1[],double coordXi2[],double coordX3[],double weight[]);

} Section;