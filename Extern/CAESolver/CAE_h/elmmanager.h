#pragma once
#include"shell1st.h"
#include"solid1st.h"
#include"beam1st.h"
#include"discrete1st.h"
#include"nodemanager.h"
#include"materialold.h"
#include"sectionold.h"
#include "beam2nd.h"
#include"shell2nd.h"
#include"solid2nd.h"
#include<map>

//ls:2020-03-18
//#include"Airbagfile.h"
//#include"ElementMass.h"
//#include"Seatbelt_Accelerometer.h"
//

using std::map;

struct ShellElementCuda;
struct SolidElementCuda;
struct BeamElementCuda;
struct DiscreteCuda;
struct Material;
struct Section;
struct NodeManager;
struct BeamElement2nd;
struct ShellElement2nd;
struct SolidElement2nd;

typedef struct ElmManager
{
	/**
	gpu与cpu的壳单元
	*/
	vector<ShellElementCuda> shell_element_array_cpu;
	ShellElementCuda *shell_element_array_gpu;

	/*
	gpu与cpu的实体单元
	*/
	vector<SolidElementCuda> solid_element_array_cpu;
	SolidElementCuda *solid_element_array_gpu;

	/**
	gpu与cpu的梁单元
	*/
	vector<BeamElementCuda> beam_element_array_cpu;
	BeamElementCuda *beam_element_array_gpu;

	/**
	gpu与cpu的离散元
	*/
	vector<DiscreteCuda> discrete_element_array_cpu;
	DiscreteCuda *discrete_element_array_gpu;

	vector<BeamElement2nd> beam2ndArrayCpu_;
	BeamElement2nd* beam2ndArrayGpu_;

	vector<ShellElement2nd> shell2ndArrayCpu_;
	ShellElement2nd* shell2ndArrayGpu_;

	vector<SolidElement2nd> solid2ndArrayCpu_;
	SolidElement2nd* solid2ndArrayGpu_;

	int nodePerElement;
	int node_amount;
	int elementAmount;

	/**
	单元长度最小值最大值
	*/
	double *eign_cpu;
	double *eign_gpu;

	/**
	单元管理器中的总质量
	*/
	double tot_mass;

	/*
	 * 单元指针与单元编号的映射关系
	 */
	map<int, Element*> elementCpuMap;
	map<int, Element*> elementGpuMap;

	ElmManager();

	/*
	 * 统计单元数
	 */
	void judgeElementAmount();

	/*
	 * 设置初始质量缩放因子
	 */
	void setInitialMassScaleFactor(const double msf = 1.0);

	/*
	 * 为实体单元设置光滑有限元标记
	 */
	void setSfemForSolid(Section *sec);

	/**
	单元质量矩阵计算
	*/
	void calElementMass(Material *material, Section *section);

	/**
	cpu单元节点数据匹配
	*/
	void elementNodeMatch(NodeManager *node_manager);

	/**
	gpu端单元数据创建
	*/
	void gpuElementCreate(const Section *section);

	/**
	gpu节点数据匹配
	*/
	void gpuNodeMatch(NodeManager *node_manager, const Section *section);

	/**
	gpu端数据清除
	*/
	void gpuDataClear();

	/**
	gpu求解单元的罚因子并分配到节点上
	*/
	void gpuCalPenaFactor(double penalty_scale_factor, Material *material, Section *section);

	/**
	节点厚度分配
	*/
	void shellNodeThick(Section *section);

	/**
	gpu端求解最小特征值
	*/
	void computeCharactLengthGpu(double &min_eigen);
	void computeCharactLengthCpu(double &min_eigen);

	/**
	求解所有单元内力
	*/
	void allElementInterForceCalGpu(MaterialNew **material, Section *section, HourglassControl*hourglass, double dt);
	void allElementInterForceCalCpu(MaterialNew **material, Section *section, HourglassControl*hourglass, double dt, double &new_dt, double safe_factor);

	/**
	TL：在时间迭代之前求解单元的B矩阵，并在时间迭代中保持不变
	*/
	void computeBMatrixTL(Section *section);

	/**
	根据预索引，建立所有单元与节点之间的索引信息
	*/
	void bulidElementIndexNodeGpu();

	/**
	确定所有单元的材料属性编号
	*/
	void verElementMatSecId(const int mater_id, const int section_id);

	/**
	分配零拷贝内存
	*/
	void allocZeroMemEle();

	/*
	 * 单元排序
	 */
	void elementSortBaseType();

	/*
	 * 设置大的单元截面类型
	 */
	void setAllElementSectionType();

	void setElementNodeNumFormulateMode(Section *sec);

	/*
	 * 确定单元计算所需的额外内存大小
	 */
	void setAllElementAddMemSize(Section *sec);

	/*
	 * cpu端分配单元计算所需的额外内存
	 */
	void allocateElementAddMem();

	/*
	 * cpu端释放单元计算所需的额外内存
	 */
	void freeElementAddMem();

	/*
	 * 在单元的额外内存上进行操作
	 */
	void operatorOnElementAddMemCpu(Section *sec);

	/*
	 * cpu上完成质量缩放
	 */
	void massScaleForElementCpu(const double min_dt);

	/*
	 * 建立单元与ID之间的映射关系
	 */
	void bulidElementMapCpu();
	void bulidElementMapGpu();

	/*
	ls:2020-07-21 added
	建立单元截面属性中的单元类型
	*/
	void buildElementSectionEtype(Section *section);

	/*
	ls:2020-07-22 added
	按照单元类型赋予计算格式UL/TL
	*/
	void buildFormulateModeBaseEType();

	/*
	ls:2020-07-22 added
	将附加自由度按单元类型赋予
	*/
	void computeInterDofNum();

	/*
	ls:2020-07-22 added
	addMemSizeForElement
	*/
	void computeAddMemorySize();

	/*
	ls:2020-07-22 added
	判断单元节点数目
	*/
	void judgeElementNodeNumBaseType();

	/*
	ls:2020-07-22 added
	设置单元积分点数目
	*/
	void computeGaussNum(Section *section);

	/*
	ls:2020-07-22 added
	设置单元积分区域类型
	*/
	void judgeElementIntegrateDomainType();

	Element* returnElementPtrCpu(int file_id);
	Element* returnElementPtrGpu(int file_id);

	virtual ~ElmManager();
} ElmManager;