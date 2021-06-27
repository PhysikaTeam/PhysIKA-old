#pragma once
#include"sectionold.h"
#include<vector>
#include<map>
using std::vector;
using std::map;

typedef struct SectionManager
{
	vector<Section> section_array;
	map<int, Section*> section_map;
	Section* section_array_gpu;

	/**
	建立section与id的连接
	*/
	void sectionLinkId();

	/**
	根据id返回section
	*/
	Section* returnSection(const int id);

	/*
	 * 确定截面属性所代表的单元的积分点个数
	 */
	void comuteIntegraPointNumDomainType();

	/*
	 * 确定所有截面属性中所有的额外内存大小
	 */
	void computeAllSectionAddMemSize();

	/**
	创建gpu端数据
	*/
	void createOldSectionArrayGpu();

	virtual ~SectionManager();
}SectionManager;