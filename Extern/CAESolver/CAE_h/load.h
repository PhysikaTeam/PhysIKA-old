#pragma once
#include"node.h"
#include"nodemanager.h"
#include"rigid.h"
#include<string>
#include"curve.h"
#include"surface.h"
#include"curve.h"
#include"curvemanager.h"
#include"part.h"
using std::string;

struct Part;
struct Curve;
struct RigidCuda;
struct NodeCuda;
struct Surface;
struct CurveManager;

typedef struct Load
{
	enum Type
	{
		Concentrate, uniform, body
	};
	enum LoadType
	{
		accel, vel, disp
	};
	enum ObjectType
	{
		setObject, partObject, surfaceObject
	};
	int id;
	int dof[6];
	int setId;
	int nodeId;
	int curveId;
	double loadValue;
	Type type;
	LoadType load_type;
	ObjectType object_type;

	/**
	放大因子
	*/
	double scaleFactor;

	/**
	承受的载荷曲线
	*/
	Curve *curve;

	/**
	收集载荷中的所有节点
	*/
	NodeCuda** node_array_gpu;
	int tot_node_num;

	/**
	外力作用在单个节点上
	*/
	NodeCuda *node;

	/**
	外力作用在set中的所有节点上
	*/
	Set *set;

	/**
	外力作用在part的所有节点上
	*/
	Part *part;

	/**
	外力施加在刚体上
	*/
	RigidCuda *rigid;

	/**
	外力施加在表面上
	*/
	Surface *surface;

	/**
	除刚体外承受外力的所有节点
	*/
	vector<NodeCuda *> node_array;

	/**
	收集载荷中的所有节点
	*/
	void createLoadNodeArray(NodeManager *nodeManager);

	/**
	对载所有节点施加载荷
	*/
	void imposeLoadAccel(double current_time);

	/**
	确定当前时间的载荷值
	*/
	void verfLoadValue(double current_time);

	/**
	链接载荷曲线
	*/
	void linkLoadCurve(CurveManager *curveManager);

	/**
	连接施加载荷的部件或表面等
	*/
	void linkLoadObject(SetManager *setManager, vector<Part> &partArray, SurfaceManager *surfaceManager);
} Load;


enum NewLoadType
{
	LNull,
	ConcentratedForce,
	Pressure,
	BodyForce,
	Gravity,
	Moment,
	ShellEdgeForce,
	SegLoad_,
};

enum LoadObject
{
	nodeLoad,
	surfaceLoad,
	elementLoad,
	globalLoad
};

typedef struct BasicLoad
{
	int file_id;

	int curve_id;

	int object_id;

	int dof[6];

	double value_;

	double scaleFactor_;

	Curve* curve;

	NewLoadType lType_;

	LoadObject objectType_;

	bool isSet;

	string objectName_;

	string curveName_;

	BasicLoad();

	BasicLoad(int fid, int cv_id, bool setFlag = false,string name="noname",string ncvname="noname");

	virtual void getLoadObject(SetManager* setMag,SurfaceManager* surfMag, vector<Part>& partMag,NodeManager* nodeMag)=0;

	void linkLoadCurve(CurveManager* cvMag);

	virtual void applyLoadGpu()=0;

	virtual void applyLoadCpu()=0;

	virtual void computeLoadValueAt(double time);

 	virtual ~BasicLoad() { ; }
}BasicLoad;