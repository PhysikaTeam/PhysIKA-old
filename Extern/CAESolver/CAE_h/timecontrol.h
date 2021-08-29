#pragma once
#include <vector>
#include"materialmanager.h"
#include"nodemanager.h"
#include"elmmanager.h"
#include"tiemanager.h"
#include"constrained.h"
#include"constrainedmanager.h"
#include"part.h"
#include"hourglasscontrol.h"
#include"elementcontrol.h"
#include"sectionmanager.h"
#include"elementcontrol.h"

struct ConstrainedManager;
struct ElementControl;
struct MultiGpuManager;
struct ElmManager;
struct TieManager;

typedef struct TimeControl
{
	/**
	控制分析终止的参数
	*/
	int endCyc;
	double endTime;
	double DTMIN;
	double ENDENG;
	double ENDMas;
	double outTimeStep;
	//ls:2020-03-18
	vector<double> outTimeStep_array;
	double initialTimeStep;
	double TSSFAC;
	int ISDO;
	double TSLIMT;
	double timeStep;
	int LCTM;
	int ERODE;
	int MS1ST;
	int NOSOL;
	//
	//ls:2020-04-06
	double DTINIT;            //initialTimeStep;
	double DT2MS;                   //timeStep;
	//
	

	/**
	时间步长安全因子
	*/
	double safeFactor;

	/**
	质量缩放频率
	*/
	int massSacleFrequen;

	/**
	初始时间步
	*/
	double iniDt;

	/**
	壳指定最小时间步
	*/
	double min_shell_time_step;

	/**
	壳单元特征长度求解方式
	*/
	int shell_char_leng_type;

	/*
	质量缩放标识，默认为0
	*/
	int isMassScaled;

	/**
	采用质量缩放时的最小时间步
	*/
	double minDtmassScale;

	/**
	进行刚性连接的质量缩放计算
	*/
	void massRedistributeForMassScaledGpu(TieManager *tieManager, ConstrainedManager *constrainManager,
		ElementControl *elementControl, NodeManager *nodeManager, const int cycNum, double &new_dt);

	void massRedistributeForMassScaledMultiGpu(TieManager *tieManager, ConstrainedManager *constrainManager,
		MultiGpuManager* multiGpuMag, NodeManager *nodeManager, const int cycNum, double &new_dt);

	void massRedistributeForMassScaledCpu(TieManager *tieManager, ConstrainedManager *constrainManager,
		vector<ElmManager> &elmManagerArray, NodeManager *nodeManager, const int cycNum, double &new_dt);

	/**
	cpu计算时间增量
	*/
	double timeStepCalCpu(vector<Part> &partArray_,double& dt);

	/**
	计算最小时间步长
	*/
	void calMinTimeStep();

	/**
	gpu更高效率计算时间增量
	*/
	double timeStepCalGpuImprove(SectionManager*section_manager, MaterialManager *mater_manager, ElementControl *element_control);
	double timeStepCalMultiGpuImprove(SectionManager*section_manager, MaterialManager *mater_manager, MultiGpuManager *multiGpuMag);

	/**
	输出初始时间增量
	*/
	void printIniTimeStep(const double dt);

	/**
	预计总的计算时间
	*/
	void evaTotalCalTime(MaterialManager *material_manager, SectionManager *section_mananger,
		ElementControl *elm_control, const double dt, HourglassControl *hourg_control);

	TimeControl()
	{
		isMassScaled = 0;
		safeFactor = 1.0;
	}
} TimeControl;