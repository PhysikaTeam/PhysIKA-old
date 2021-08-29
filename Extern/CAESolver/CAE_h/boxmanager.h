#pragma once
#include"box.h"

//ls:2020-04-06

struct Box;

typedef struct BoxManager
{
	vector<Box> box_array;
	int totBoxNum;

} BoxManager;

//