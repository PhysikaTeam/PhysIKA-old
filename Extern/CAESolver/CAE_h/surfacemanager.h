#pragma once
#include"surface.h"
#include<map>
#include"SurfaceInteract.h"
using std::map;

struct Element;
struct Surface;
struct FEMDynamic;
struct SurfaceInteract;

struct ElementMapSegment
{
	int faceId;
	Element* elm;
	SegmentCuda *seg;
};

typedef struct SurfaceManager
{
	int segTotNum;

	vector<Surface> surface_array;
	
	map<int, Surface*> surfaceMap;

	vector<SurfaceInteract *> surfaceInteractArrayCpu;

	SurfaceInteract** surfaceInteractArrayGpu;

	map<SegmentCuda*, Element*> segMapEl;

	map<int, int> segIdMapElId;

	vector<ElementMapSegment> elToSegStructArrayCpu;

	ElementMapSegment* elToSegStructArrayGpu;

	SurfaceManager();

	void computeSegmentInterfaceStiffCpu(FEMDynamic *domain,double scaleFactor=0.1);

	void bulidElementSegmentMap(FEMDynamic *domain);

	void produceSurfaceInteractGpu();

	void produceElSegMapStructArrayCpu();

	void produceElSegMapStructArrayGpu();

	/**
	建立接触面与id的联系
	*/
	void surfaceLinkId();

	void produceFormElementSet(FEMDynamic *domain);

	void resetSurfFlag();

	Surface* returnSurface(const int id);
}SurfaceManager;
