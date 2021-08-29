#pragma once
#include"contactbasic.h"
#include"SurfaceInteract.h"
#include<map>
#include<set>
#include<vector>

using std::map;
using std::set;
using std::vector;

struct SurfaceInteract;
struct Contact;
struct FEMDynamic;

typedef struct ContactManager
{
	//对所有外表面搜寻接触
	bool isAllExterior;

	vector<Contact> contactArray;

	int surfInteractNum_;

	map<vector<int>, int> surfInteractMap;

	vector<SurfaceInteract*> generalSurfInteractCpu_;

	vector<SurfaceInteract**> generalSurfInteractGpu_;

	ContactManager();

	void produceAllExtContact(FEMDynamic *domain);

	void getSurfaceInteract(FEMDynamic *domain);

	void imposeGeneralSurfaceInteractMultiGpu(FEMDynamic *domain,double dt);

	void imposeGeneralSurfaceInteractGpu(FEMDynamic *domain,double dt);

	void imposeGeneralSurfaceInteractCpu(FEMDynamic *domain,double dt);

	void imposeSpecificInteractCpu(FEMDynamic *domain,double dt);

	void imposeSpecificInteractGpu(FEMDynamic *domain,double dt);

	void imposeSpecificInteractMultiGpu(FEMDynamic *domain,double dt);

	void clearContactForTie(FEMDynamic *domain);

	~ContactManager() { ; }
}ContactManager;