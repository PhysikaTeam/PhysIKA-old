#include "demoCallbacks.h"

#include "curand.h"

#include <algorithm>

namespace PhysIKA {

__global__ void PGCallback_init(curandState* curand_states, int num, long clock_for_rand)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num)
        return;

    curand_init(clock_for_rand, tid, 0, &curand_states[tid]);
}

__global__ void PGCallback_generate(
    DeviceDArray<Vector3d>     parPos,
    DeviceDArray<Vector3d>     parVel,
    DeviceDArray<double>       parMass,
    DeviceDArray<ParticleType> parType,
    curandState*               randState,
    int                        startid,
    double                     m0,
    double                     gxmin,
    double                     gxmax,
    double                     gzmin,
    double                     gzmax)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid + startid >= parPos.size())
        return;

    //curand_init(startid, tid, )
    double px              = abs(curand_uniform(randState + tid)) * (gxmax - gxmin) + gxmin;
    double pz              = abs(curand_uniform(randState + tid)) * (gzmax - gzmin) + gzmin;
    parPos[tid + startid]  = Vector3d(px, 0, pz);
    parVel[tid + startid]  = Vector3d();
    parMass[tid + startid] = m0;
    parType[tid + startid] = ParticleType::SAND;
}

void ParticleGenerationCallback::init(float xmin, float xmax, float zmin, float zmax, float m0, float rate, int maxNum, long seed)
{
    gxMin            = xmin;
    gxMax            = xmax;
    gzMin            = zmin;
    gzMax            = zmax;
    particelMass     = m0;
    generationRate   = rate;
    maxGenerationNum = maxNum;

    cuSafeCall(
        cudaMalloc(( void** )&devStates, sizeof(curandState) * maxNum));

    cuExecute(maxNum, PGCallback_init, devStates, maxNum, seed);
}

void ParticleGenerationCallback::handle(ParticleSandRigidInteraction* interactNode, float dt)
{
    int newParNum = generationRate * dt;
    if (newParNum < 1)
        return;
    newParNum = newParNum > maxGenerationNum ? maxGenerationNum : newParNum;

    auto& devPos  = interactNode->getSandSolver()->getParticlePosition();
    auto& devVel  = interactNode->getSandSolver()->getParticleVelocity();
    auto& devMass = interactNode->getSandSolver()->getParticleMass();
    auto& devType = interactNode->getSandSolver()->getParticleTypes();

    int startid   = devPos.size();
    int totalSize = newParNum + devPos.size();
    devPos.resize(totalSize);
    devVel.resize(totalSize);
    devMass.resize(totalSize);
    devType.resize(totalSize);

    cuExecute(newParNum, PGCallback_generate, devPos, devVel, devMass, devType, devStates, startid, ( double )particelMass, gxMin, gxMax, gzMin, gzMax);

    interactNode->getSandSolver()->infoUpdate(dt);
}

void ParticleHeightOnZ::handle(float dt)
{
    if (!particleType || !particlePos || !particleRho2D)
        return;
    if (outputfilename == std::string())
        return;

    std::ofstream outstr;
    outstr.open(outputfilename);

    hostParType.resize(particleType->size());
    Function1Pt::copy(hostParType, *particleType);

    hostParPos.resize(particlePos->size());
    Function1Pt::copy(hostParPos, *particlePos);

    hostParRho2D.resize(particleRho2D->size());
    Function1Pt::copy(hostParRho2D, *particleRho2D);

    std::vector<std::pair<double, double>> xhpair;
    for (int i = 0; i < hostParRho2D.size(); ++i)
    {
        if (hostParType[i] != ParticleType::SAND)
            continue;

        Vector3d pos = hostParPos[i];
        if (pos[2] < zValue - searchWidth || pos[2] > zValue + searchWidth)
            continue;

        if (!(pos[0] > -9e10 && pos[0] < 9e10))
            return;

        double curh = hostParRho2D[i] / rho0;
        //outstr << pos[0] << "   " << curh << std::endl;

        xhpair.push_back(std::make_pair(pos[0], curh));
    }

    std::sort(xhpair.begin(), xhpair.end(), [](const std::pair<double, double>& xh1, const std::pair<double, double>& xh2) {
        return xh1.first < xh2.first;
    });

    for (int i = 0; i < xhpair.size(); ++i)
    {
        outstr << xhpair[i].first << " " << xhpair[i].second << std::endl;
    }

    outstr.close();
}

}  // namespace PhysIKA