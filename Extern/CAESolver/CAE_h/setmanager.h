#pragma once
#include"set.h"
#include<map>
using std::map;

struct Set;

typedef struct SetManager
{
	vector<Set> nodeSet_array;
	vector<Set> segmentSet_array;
	vector<Set> partSet_array;
	vector<Set> elementSet_array;

	map<int, Set*> nodeSetLink;
	map<int, Set*> partSetLink;
	map<int, Set*> segmentSetLink;
	map<int, Set*> elementSetLink;

	/**
	将set与id连接起来
	*/
	void setLinkId();

	/**
	给出正确的set
	*/
	Set* returnSet(const int id, Set::Type setType=Set::nodeSet);

	~SetManager()
	{
		nodeSet_array.clear();
		segmentSet_array.clear();
		partSet_array.clear();
		elementSet_array.clear();
		nodeSetLink.clear();
		elementSetLink.clear();
		partSetLink.clear();
		segmentSetLink.clear();
	}
} SetManager;