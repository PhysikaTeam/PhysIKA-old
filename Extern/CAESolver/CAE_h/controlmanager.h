#pragma once
#include "cpucontrol.h"
#include "energycontrol.h"
#include "accuracycontrol.h"
#include "outputcontrol.h"
#include "solidcontrol.h"
#include "shellcontrol.h"
#include "contactcontrol.h"
#include "hourglasscontrol.h"
#include "parallelcontrol.h"
#include "couplingcontrol.h"


typedef struct ControlManager
{
	//vector<TimestepControl> timestepControlArray;
	vector<ContactControl> contactControl_array;
	vector<ShellControl> shellControl_array;
	vector<SolidControl> solidControl_array;
	vector<OutputControl> outputControl_array;
	vector<EnergyControl> energyControl_array;
	vector<AccuracyControl>accuracyControl_array;
	vector<CpuControl> cpuControl_array;
	//vector<HourglassControl> hourglassControl_array;
	vector<ParallelControl> parallelControl_array;
	vector<CouplingControl> couplingControl_array;
} ControlManager;