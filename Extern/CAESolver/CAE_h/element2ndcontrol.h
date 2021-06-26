//#pragma once
//#include"gausspoint.h"
//#include"part.h"
//#include"materialold.h"
//#include"sectionold.h"
//#include"hourglasscontrol.h"
//#include"sectionmanager.h"
//#include"materialstatus.h"
//#include "cublas_v2.h"
//#include"cublas_api.h"
//#include"beam2nd.h"
//#include"shell2nd.h"
//#include"solid2nd.h"
//
//typedef 
//{
//	BeamElement2nd** beam2ndArrayGpu_;
//	ShellElement2nd** shell2ndArrayGpu_;
//	SolidElement2nd** solid2ndArrayGpu_;
//
//	int beam2ndNum_;
//	int solid2ndNum_;
//	int shell2ndNum_;
//
//	double *dtArrayGpu_;
//
//	GaussPoint* gaussPointArrayGpu_;
//
//	int gaussPointNum_;
//
//	Section *section_array_gpu;
//	MaterialNew **material_array_gpu;
//
//	/*
//	 *创建单元总集合
//	 */
//	void createElementArraySingleGpu(vector<Part> &partArray_);
//
//	/*
//	 *多GPU创建单元集合
//	 */
//	void createElementArrayMultiGpu(vector<Part> &partArray_, const int in,
//		const vector<int> &elm_num_gpu, const vector<cudaStream_t> &streamArray, const vector<int> &gpu_id_array);
//
//	/*
//	 *计算所有单元的内力
//	 */
//	void allElement2ndArrayCalInterForce(MaterialNew** sys_material_array_gpu, Section *sys_section_array_gpu, HourglassControl *hourg_control,
//		double dt, cublasHandle_t cu_handle, double &dt_new, double safe_factor);
//
//	/*
//	 *链接单元中的所有节点
//	 */
//	void allElementLinkNode(NodeCuda *sys_node_array_gpu, cudaStream_t stream_gpu);
//
//	/*
//	 *二阶单元质量缩放
//	 */
//	void massScaleForElement2nd(const double min_dt);
//
//	/*
//	 *所有单元创建积分点
//	 */
//	void computeGaussPointArrayGpu(SectionManager *sectionManager,MaterialManager *matManager,vector<Part> &partArray_);
//
//	/*
//	 *创建所有单元的时间增量
//	 */
//	void allocateDtArrayGpu();
//
//	/*
//	 *统计所有单元质量缩放增加的质量
//	 */
//	void countAddedMassForMassScaled(const int isMassScaled);
//}Element2ndControl;