#pragma once
#include"materialold.h"
#include<vector>
#include<map>
#include"materialnew.h"
using std::vector;
using std::map;

typedef struct MaterialManager
{
	vector<Material> material_array;
	map<int, Material*> materialMap;
	Material* material_array_gpu;

	vector<MaterialNew*> materialNewArrayCpu_;
	map<int, MaterialNew*>materialNewMap_;
	MaterialNew** materialNewArrayGpu_;

	int material_num;

	MaterialManager();

	/**
	建立材料与id的映射
	*/
	void materialLinkId();

	/**
	根据id返回材料
	*/
	Material* returnMaterial(const int id);
	MaterialNew* returnMaterialNew(const int id);

	/**
	创建gpu端数据
	*/
	void createOldMaterialArrayGpu();

	/*
	 * 建立新式材料与id的映射
	 */
	void bulidIdMapMaterialNew();

	/*
	 * 获取材料指针
	 */
	MaterialNew* giveMaterialNewPtr(const int id);

	/*
	 * 创建新式材料的gpu端数据
	 */
	void createNewMaterialArrayGpu();

	/*
	 * 根据旧式材料创建新式材料
	 */
	void createMaterialCpuData();

	/*
	 * 验证新行材料类的实例化建立成功
	 */
	void verNewMaterialGpuCreate();

	/*
	 * 创建cpu材料状态
	 */
	void createAllMatStatusCpu();

	/*
	 * 创建gpu材料状态
	 */
	void createAllMatStatusGpu(const int gpu_id = 0);

	/*
	 * 将材料状态集合从cpu复制到Gpu
	 */
	void copyMatStatusFromCpuToGpu(MaterialNew **matNewArrayGpu,const int gpu_id=0);

	virtual ~MaterialManager();
}MaterialManager;
