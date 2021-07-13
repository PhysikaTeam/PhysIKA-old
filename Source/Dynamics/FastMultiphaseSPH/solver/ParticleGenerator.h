#pragma once

#include "../math/math.h"
//#include "utility/cpXMLHelper.h"
#include "sph_common.h"

struct fluidvol
{
    cfloat3 xmin;
    cfloat3 xmax;
    float   volfrac[10];
    int     group;
    int     type;
};

struct FluidSrc
{
    cfloat3 srcpos;
    cfloat3 norm;
    float   radius;
    float   speed;
    char    type;  //geometry type
    int     interval;
};

//void loadFluidVolume(tinyxml2::XMLElement* sceneEle, int typenum, std::vector<fluidvol>& fvs);

struct Particle
{
    cfloat3 pos;
    int     type;
};

class ParticleContainer
{
public:
    std::vector<Particle> particles;
};

class ParticleObject
{
public:
    vecf3 pos;
    vecf3 normal;
    vecf  volfrac;
    veci  type;
    veci  id;
};
//
//class BoundaryGenerator {
//private:
//    ParticleObject* particleObject  = nullptr;
//    Tinyxml_Reader reader;
//public:
//    ~BoundaryGenerator()
//    {
//        if (particleObject)
//            delete particleObject;
//    }
//    float spacing;
//    void parseNode(tinyxml2::XMLElement* e);
//    ParticleObject* loadxml(const char* filepath);
//    ParticleObject* GetPO() {
//        return particleObject;
//    }
//    int GetSize() {
//        if (particleObject)
//            return particleObject->pos.size();
//        else
//            return -1;
//    }
//};

void loadPoints(ParticleObject* particleObject, char* filepath);