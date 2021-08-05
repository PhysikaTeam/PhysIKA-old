#pragma once

#include "../../../math/cpTimer.h"

#include "../../ParticleSolver.h"
#include "../../ParticleGenerator.h"
#include "../../tool/BufferManager.h"
#include "SimDataMultiphase.h"
#include "deformable.cuh"

namespace msph {

enum sortFlags
{
    sortFlagsWithoutReorder = 0,
    sortFlagsWithReorder    = 1
};

class MultiphaseSPHSolver : public ParticleSolver
{

public:
    vecf3 pos;
    vecf4 color;
    vecf3 vel;
    veci  unique_id;
    veci  id_table;
    vecf3 normal;
    veci  type;
    vecf  mass;
    vecf  rest_density;
    vecf  inv_mass;
    veci  group;
    vecf  vol_frac;
    vecf  temperature;
    vecf  heat_buffer;

    vecf3              v_star;
    vecf3              drift_accel;
    std::vector<cmat3> stress;
    veci               localid;
    SimDataMultiphase  simdata;
    MultiphaseParam    h_param;
    MultiphaseParam*   pParam;

    std::vector<std::shared_ptr<fluidvol>> fluid_volumes;
    BufferManager                          bufman;

    int  num_particles;
    int  num_cells;
    int  localid_counter[100];
    uint stepCounter = 0;
    int  frame_count;
    int  dump_count;
    int  case_id;
    int  mode;
    int  max_nump;

    bool emit_particle;
    bool enable_dfsph;
    bool enable_solid;
    bool enable_diffusion;
    bool enable_heat;
    bool enable_dump;
    bool enable_dump_render;

    float system_time;
    float emit_timer;
    float emit_interval;
    float dump_timer;

    std::string outputDir;
    float       frame_rate;

    MultiphaseSPHSolver()
    {
    }

    void preinit();
    void postinit();

    void SetupFluidScene();
    void loadSceneFromFile(char* filePath);
    void SetParameter();
    void updateSimulationParam();
    int  addDefaultParticle();
    void addFluidVolumes();
    void addParticles(int addcount, const cfloat3 pos[], const float volfrac[], int group, int type, int visible);
    void LoadBoundaryParticles(ParticleObject* po);
    void LoadRigidParticles(ParticleObject* po, int groupId);
    void LoadPO(ParticleObject* po,
                int             type);
    void loadScriptObject(ParticleObject* po);

    void emitFluid();
    void emitFluidDisk(float radius, cfloat3 center, cfloat3 xyz, cfloat3 vp);

    void setupDeviceData();
    void allocateDeviceBuffer();
    void copySimulationDataFromDevice();
    void copyPosColorFromDevice();
    void copyData2Device(int start, int end);

    //wcsph
    void solveWCSPH();
    void step();
    void sortParticles(uchar flag);

    void phaseDiffusion();

    void dumpSimulationData();
    void loadSimulationData(char* filepath, const cmat4& materialMat);
    void dumpRenderData();

    void setOutputDir(std::string& dir)
    {
        outputDir = dir;
    }

    vecf3& getPos()
    {
        return pos;
    }
    vecf4& getColor()
    {
        return color;
    }
    cfloat3 getXmin()
    {
        return h_param.gridxmin;
    }
    cfloat3 getXmax()
    {
        return h_param.gridxmax;
    }
    float getDt()
    {
        return h_param.dt;
    }
    void prepareRenderData(cfloat3* pos, cfloat4* clr);
    void handleKeyEvent(char key)
    {
        switch (key)
        {
            case 'b':
                dumpSimulationData();
                break;
        }
    }
};

};  // namespace msph
