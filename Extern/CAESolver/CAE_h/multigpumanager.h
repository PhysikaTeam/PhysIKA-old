#pragma once
#include"cublas.h"
#include"cublas_api.h"
#include"elementcontrol.h"
#include"hourglasscontrolmanager.h"
#include"materialmanager.h"
#include"sectionmanager.h"
#include"part.h"

struct MaterialManager;
struct ElementControl;
struct TimeControl;
struct HourglassControlManager;
struct SectionManager;
struct Part;

/*
* 当前多gpu的思路一共存在三种
* 1、每块gpu上均保存有节点，不同gpu之间每次迭代步与主gpu交换数据，涉及的通讯量比较大，目前以舍弃
* 2、所有节点均保存在主gpu上，其他gpu通过p2p进行访问，涉及的通讯量较少，易于实现，目前采用这种
* 3、所有gpu不分主次，每个gpu上均保存部分节点，仅在边界上进行通讯，通讯量较少，不易于实现，有待后续开发
*/
typedef struct MultiGpuManager
{
	int num_gpu;

	/**
	负责各个gpu上的单元控制的管理器
	*/
	vector<ElementControl*> elementControlArray;

	/**
	存储各个gpu上节点内力
	*/
	/*vector<double *> multi_intforce_array_gpu;*/

	/**
	存储各个gpu上节点的坐标
	*/
	/*vector<double *> multi_coord_array_gpu;*/

	/**
	存储各个gpu上节点的速度
	*/
	/*vector<double *> multi_vel_array_gpu;*/

	/**
	主gpu上的传递中介变量，共分段为三个部分，第一部分为坐标，第二部分为速度，第三部分为节点内力
	*/
	/*vector<double *> mediun_variable_array;*/

	/**
	存储各个gpu上的节点
	*/
	/*vector<NodeCuda *> multi_node_array_gpu;*/

	/**
	存储各个分区上节点的编号
	*/
	vector<int*> multiNodeIdArrayAidGpu;
	vector<int*> multiNodeIdArrayMainGpu;
	vector<vector<int>> multiNodeIdArrayCpu;

	/**
	各个gpu上单元数目
	*/
	vector<int> elm_num_gpu;

	/**
	各个gpu上使用的节点数目
	*/
	vector<int> multi_node_num;

	/**
	运行在各个gpu上的流
	*/
	vector<cudaStream_t> streamArray;
	vector<cublasHandle_t> cudaHandleArray;

	/**
	可以使用的计算能力大于6.0的gpu_id，第一个gpu为主gpu，其余gpu为辅GPU
	*/
	vector<int> gpu_id_array;

	bool isP2P;

	/**
	存储在各个gpu上的材料
	*/
	/*vector<MaterialNew **> material_array_gpu;*/

	/**
	存储在各个gpu上的属性
	*/
	/*vector<Section *> section_array_gpu;*/

	MultiGpuManager();

	/**
	给定gpu数目
	*/
	void judgeGpuNum();

	/**
	创建与gpu相匹配的流
	*/
	void createStream();

	/*
	* 设备同步
	*/
	void synchronousDevice();

	/*
	 * 流同步
	 */
	void synchronousStream();

	/*
	 * 销毁流
	 */
	void destroyStream();

	/*
	* 创建句柄
	*/
	void createCublasHandle();

	/*
	* 销毁句柄
	*/
	void destroyCublasHandle();

	/**
	各个gpu上节点的分配与复制
	*/
	/*void multiGpuNodeCpy(NodeCuda *sys_node_array_gpu,int tot_node_num);*/

	/**
	各个gpu上的材料与属性的复制
	*/
	void multiGpuMatSectionCpy(MaterialManager *material_manager, SectionManager *section_manager, HourglassControlManager* hgManager);

	/**
	各个gpu上节点力向量的分配
	*/
	/*void allocaIntForArrayMultiGpu();*/

	/**
	各个gpu上节点坐标速度向量的显存分配
	*/
	/*void allocateCoordVelMultiGpu();*/

	/**
	主gpu上中介变量向量的显存分配
	*/
	/*void allocateMediaVariable();*/

	/**
	辅gpu力传递到主gpu上
	*/
	/*void assemblyIntForToMainGpu();*/

	/**
	主gpu速度以及坐标传递到辅gpu上
	*/
	/*void distributDispVelToAidGpu();*/

	/**
	在网格分区完成的基础上，形成gpu与cpu上节点的序列号
	*/
	void createNodeIdCpu(const vector<Part>& partArray_, int node_num);
	void createNodeIdGpu();

	/**
	在网格分区的基础上，创建各个分区的单元控制器
	*/
	void createElementControlForAllGpu(vector<Part> &partArray_);

	/*
	* p2p验证
	*/
	void verP2PForMultiGpu();

	/**
	各个gpu上分区的单元连接节点
	*/
	/*void elementControlLinkNode();*/

	/**
	计算各个gpu分区中的单元管理器中的单元应力
	*/
	void allGpuCalElementIntForAndDt(double &dt_new, double dt, HourglassControl *hourg_control, TimeControl *time_control);

	/*
	 * 重置设备
	 */
	void resetDevice();

}MultiGpuManager;

//typedef struct MultiGpuManager
//{
//	int num_gpu;
//
//	/**
//	负责各个gpu上的单元控制的管理器
//	*/
//	vector<ElementControl*> elementControlArray;
//	
//	/**
//	存储各个gpu上节点内力
//	*/
//	vector<double *> multi_intforce_array_gpu;
//
//	/**
//	存储各个gpu上节点的坐标
//	*/
//	vector<double *> multi_coord_array_gpu;
//
//	/**
//	存储各个gpu上节点的速度
//	*/
//	vector<double *> multi_vel_array_gpu;
//
//	/**
//	主gpu上的传递中介变量，共分段为三个部分，第一部分为坐标，第二部分为速度，第三部分为节点内力
//	*/
//	vector<double *> mediun_variable_array;
//	
//	/**
//	存储各个gpu上的节点
//	*/
//	vector<NodeCuda *> multi_node_array_gpu;
//
//	/**
//	存储各个分区上节点的编号
//	*/
//	vector<int*> multi_nodeid_array_aid_gpu;
//	vector<int*> multi_nodeid_array_main_gpu;
//	vector<vector<int>> multi_nodeid_array_cpu;
//
//	/**
//	各个gpu上单元数目
//	*/
//	vector<int> elm_num_gpu;
//
//	/**
//	各个gpu上使用的节点数目
//	*/
//	vector<int> multi_node_num;
//
//	/**
//	运行在各个gpu上的流
//	*/
//	vector<cudaStream_t> streamArray;
//
//	/**
//	可以使用的计算能力大于6.0的gpu_id，第一个gpu为主gpu，其余gpu为辅GPU
//	*/
//	vector<int> gpu_id_array;
//
//	/**
//	存储在各个gpu上的材料
//	*/
//	vector<MaterialNew **> material_array_gpu;
//
//	/**
//	存储在各个gpu上的属性
//	*/
//	vector<Section *> section_array_gpu;
//
//	/**
//	给定gpu数目
//	*/
//	bool judgeGpuNum();
//
//	/**
//	创建与gpu相匹配的流
//	*/
//	void createStream();
//
//	/**
//	各个gpu上节点的分配与复制
//	*/
//	void multiGpuNodeCpy(NodeCuda *sys_node_array_gpu,int tot_node_num);
//
//	/**
//	各个gpu上的材料与属性的复制
//	*/
//	void multiGpuMatSectionCpy(MaterialManager *material_manager, SectionManager *section_manager);
//
//	/**
//	各个gpu上节点力向量的分配
//	*/
//	void allocaIntForArrayMultiGpu();
//
//	/**
//	各个gpu上节点坐标速度向量的显存分配
//	*/
//	void allocateCoordVelMultiGpu();
//
//	/**
//	主gpu上中介变量向量的显存分配
//	*/
//	void allocateMediaVariable();
//
//	/**
//	辅gpu力传递到主gpu上
//	*/
//	void assemblyIntForToMainGpu();
//
//	/**
//	主gpu速度以及坐标传递到辅gpu上
//	*/
//	void distributDispVelToAidGpu();
//
//	/**
//	在网格分区完成的基础上，形成gpu与cpu上节点的序列号
//	*/
//	void createNodeIdCpu(const vector<Part> partArray_, int node_num);
//	void createNodeIdGpu();
//
//	/**
//	在网格分区的基础上，创建各个分区的单元控制器
//	*/
//	void createElementControlForAllGpu(vector<Part> &partArray_);
//
//	/**
//	各个gpu上分区的单元连接节点
//	*/
//	void elementControlLinkNode();
//
//	/**
//	计算各个gpu分区中的单元管理器中的单元应力
//	*/
//	void allGpuCalElementIntForAndDt(double &dt_new,double dt, HourglassControl *hourg_control, TimeControl *time_control, cublasHandle_t cu_handle);
//
//}MultiGpuManager;