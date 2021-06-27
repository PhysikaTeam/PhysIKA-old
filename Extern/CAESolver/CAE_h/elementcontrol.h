#pragma once
#include <cuda_runtime.h>
#include <cublas.h>
#include "beam1st.h"
#include"shell1st.h"
#include"solid1st.h"
#include"discrete1st.h"
#include"node.h"
#include"timecontrol.h"
#include"materialold.h"
#include"sectionold.h"
#include"smoothFEM.h"
#include"gausspoint.h"
#include"timecontrol.h"
#include"smoothFemManager.h"
#include"integrapointmanager.h"
#include "hourglasscontrolmanager.h"

struct TimeControl;
struct IntegratePointManager;
struct GaussPoint;
struct SmoothFEM;
struct NodeCuda;
struct BeamElementCuda;
struct SolidElementCuda;
struct ShellElementCuda;
struct DiscreteCuda;
struct BeamElement2nd;
struct ShellElement2nd;
struct SolidElement2nd;
struct Section;
struct Material;
struct MaterialNew;
struct NodeManager;
struct Part;
struct SmoothFemManager;
struct HourglassControl;

/*
 * 包括单元GPU计算所有需要的数据
 * 包括单元，材料，截面属性，积分点，节点等
 */
typedef struct ElementControl
{
	GaussPoint* gauss_array_gpu;
	NodeCuda* node_array_gpu;

	/*
	 * 单gpu版
	 */
	BeamElementCuda** beamArrayGpu;
	SolidElementCuda** solidArrayGpu;
	ShellElementCuda** shellArrayGpu;
	DiscreteCuda** discreteArrayGpu;
	BeamElement2nd** beam2ndArrayGpu_;
	ShellElement2nd** shell2ndArrayGpu_;
	SolidElement2nd** solid2ndArrayGpu_;
	SmoothFEM* smoothArrayGpu;

	/*
	 * 多gpu版
	 */
	BeamElementCuda* beamMultiGpuArray_;
	SolidElementCuda* solidMultiGpuArray_;
	ShellElementCuda* shellMultiGpuArray_;
	DiscreteCuda* discreteMultiGpuArray_;
	BeamElement2nd* beam2ndArrayMultiGpu_;
	ShellElement2nd* shell2ndArrayMultiGpu_;
	SolidElement2nd* solid2ndArrayMultiGpu_;

	int beam_num;

	int solid_num;

	int shell_num;

	int discrete_num;

	int beam2ndNum_;
	int solid2ndNum_;
	int shell2ndNum_;

	int smooth_num;

	int gauss_num;

	int node_num;

	double *dt_cpu;
	double *dt_gpu;

	double *dt_array_gpu;

	Section *section_array_gpu;
	Material *material_array_gpu;
	MaterialNew** matnew_array_gpu;
	HourglassControl *hourglass_array_gpu;

	ElementControl();

	/*
	 * 计算所有节点的应力应变,注意这里只是将节点的应变做累加
	 */
	void accumulateElementStreeStrainToNode(Section *secArrayGpu);

	void allocateAllElementAddMemMultiGpu();

	void freeAllElementAddMemMultiGpu();

	void operatorOnAllElementAddMem(Section *sectionArrayGpu);

	/*
	 * 创建单gpu节点集合
	 */
	void createNodeArraySingleGpu(NodeManager *nodeManager);

	/*
	 * 创建多gpu节点集合
	 */
	void createNodeArrayMultiGpu(NodeManager *nodeManager, const int in,
		const vector<cudaStream_t> &streamArray, const vector<int> &gpu_id_array);

	/*
	 * 创建单Gpu的材料、截面以及沙漏控制等参与单元计算的指针
	 */
	void createMatSecHourgSingGpu(MaterialManager *matManager, SectionManager *secManager, HourglassControlManager *hgManager);

	/*
	 * 创建多Gpu的材料、截面以及沙漏控制等参与单元计算的指针
	 */
	void createMatSecHourgMultiGpu(MaterialManager *matManager, SectionManager *secManager, HourglassControlManager *hgManager);

	/*
	 *创建单元集合
	 */
	void createElementArraySingleGpu(vector<Part> &partArray_);

	/**
	创建多GPU单元集合
	*/
	void createElementArrayMultiGpu(vector<Part> &partArray_, const int in,
		const vector<int> &elm_num_gpu, const vector<cudaStream_t> &streamArray, const vector<int> &gpu_id_array);

	/*
	 * 创建多gpu的指针副本
	 */
	void createElementPtrCopy(const int in, const vector<cudaStream_t> &streamArray, const vector<int> &gpu_id_array);

	/*
	 * 创建单GPU的光滑有限元集合
	 */
	void createSFEMArraySingleGpu(SmoothFemManager *smoothManager);

	/*
	 * 创建多gpu版本的光滑有限元集合
	 */
	void createSFEMArrayMulitlGpu(SmoothFemManager *smoothManager, NodeManager *ndMag, const int in,
		const vector<int> &elm_num_gpu, const vector<cudaStream_t> &streamArray, const vector<int> &gpu_id_array);

	/*
	 * 创建单gpu的积分点集合
	 */
	void createGaussPointArraySingleGpu(IntegratePointManager *integPointManager);

	/*
	 * 创建多gpu的积分点集合
	 */
	void createGaussPointArrayMultiGpu(IntegratePointManager *integPointManager, const int in,
		const vector<cudaStream_t> &streamArray, const vector<int> &gpu_id_array);

	/**
	单元控制器中所有单元联系节点
	*/
	void allElementLinkToNode(NodeCuda *sys_node_array_gpu, cudaStream_t stream_gpu);

	/*
	* 多gpu中实行预索引机制
	*/
	void bulidElementIndexNodeMultiGpu(const int in, const vector<cudaStream_t> &streamArray, const vector<int> &gpu_id_array);

	//以下为并行中的单元计算

	/**
	单元集合求解内力，时间增量，特征长度
	*/
	void allElemenArrayCalInterForGpu(MaterialNew **sys_material_array_gpu, Section *sys_section_array_gpu, HourglassControl *hourglass,
		double dt, cublasHandle_t cu_handle, double &dt_new, double safe_factor);

	void allElementArrayCalInterForMultiGpu(MaterialNew **sys_material_array_gpu, Section *sys_section_array_gpu, HourglassControl *hourglass,
		double dt, cublasHandle_t cu_handle, double &dt_new, double safe_factor, cudaStream_t stream = nullptr);

	/*
	 * 对所有单元进行质量缩放计算
	 */
	void massScaleForElementsGpu(const double min_dt);
	
	/**
	统计质量缩放增加的质量
	*/
	void countAddedMassForMassScaled(const int isMassScaled);

	/**
	求解时间增量
	*/
	double calAllDt(Material *material_array, Section *section_array, const double time_step_scale);

	void callDtMultiGpu(cublasHandle_t cu_handle, double &dt_new, double safe_factor, cudaStream_t stream = nullptr);

	//单元并行计算中的一些特殊处理

	/*
	 * 单元指针按算法进行排序
	 */
	void sortElementPtrBaseType();

	/*
	 * 单元按算法进行排序
	 */
	void sortElementBaseType();

	//以下为单元计算中所需内存的分配与释放
	
	/**
	分配零拷贝内存
	*/
	void allocZeroMemElm();

	/**
	验证所有单元在计算中不被破坏
	*/
	void verAllElementCompete();

	/**
	分配dt_array的gpu内存
	*/
	void allocateDtArrayGpu();

	~ElementControl();

} ElementControl;
