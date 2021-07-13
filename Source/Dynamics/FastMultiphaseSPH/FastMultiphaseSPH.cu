#include "FastMultiphaseSPH.h"

#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

#include "Core/Utility.h"
#include "Framework/Topology/PointSet.h"
#include "solver/Multiphase/wcsph/MultiphaseSPHSolver.h"

namespace PhysIKA {
IMPLEMENT_CLASS_1(FastMultiphaseSPH, TDataType)

template <typename TDataType>
FastMultiphaseSPH<TDataType>::FastMultiphaseSPH(std::string name)
    : Node(name)
{
    //		attachField(&m_velocity, MechanicalState::velocity(), "Storing the particle velocities!", false);
    //		attachField(&m_force, MechanicalState::force(), "Storing the force densities!", false);

    m_pSet = std::make_shared<PointSet<TDataType>>();
    this->setTopologyModule(m_pSet);

    m_msph = std::make_shared<msph::MultiphaseSPHSolver>();
    m_msph->preinit();
}
template <typename TDataType>
void FastMultiphaseSPH<TDataType>::initSync()
{

    m_msph->postinit();

    prepareData();

    std::vector<Coord> buffer(num_o);
    m_pSet->setPoints(buffer);
    m_pSet->setNormals(buffer);
    m_phase_concentration.setElementCount(num_o);

    updateTopology();
    // 		m_pointsRender = std::make_shared<PointRenderModule>();
    // 		this->addVisualModule(m_pointsRender);
}
struct OpaquePred
{
    __host__ __device__ bool operator()(Vector4f v)
    {
        return v[3] != 0;
    }
};
template <typename TDataType>
void FastMultiphaseSPH<TDataType>::prepareData()
{
    // get all particles
    int num = m_msph->num_particles;
    printf("%d particles in total\n", num);
    if (num != m_pos.size())
    {
        m_pos.resize(num);
        m_color.resize(num);
    }
    m_msph->prepareRenderData(( cfloat3* )m_pos.getDataPtr(), ( cfloat4* )m_color.getDataPtr());
    num_o = num;
    // then filter transparent particles
    Vector3f* d_pos   = m_pos.getDataPtr();
    Vector4f* d_color = m_color.getDataPtr();
    thrust::copy_if(thrust::device, d_pos, d_pos + num, d_color, d_pos, OpaquePred());
    auto oe = thrust::copy_if(thrust::device, d_color, d_color + num, d_color, OpaquePred());
    num_o   = oe - d_color;
}

template <typename TDataType>
FastMultiphaseSPH<TDataType>::~FastMultiphaseSPH()
{
}

template <typename TDataType>
void FastMultiphaseSPH<TDataType>::advance(Real dt)
{
    // dt not used here as its managed by external solver ...
    m_msph->step();
}

template <typename TDataType>
void FastMultiphaseSPH<TDataType>::loadParticlesFromFile(std::string filename, particle_t type)
{
    std::vector<Coord> vertList;
    std::ifstream      ifs(filename, std::ios::binary);
    if (!ifs)
        printf("cannot open file %s\n", filename.c_str());
    std::string line;
    while (getline(ifs, line))
    {
        auto s1p = line.find(' ');
        if (s1p != std::string::npos && line.substr(0, s1p) == "v")
        {
            Coord c;
            sscanf(line.c_str() + s1p, "%f %f %f", &c[0], &c[1], &c[2]);
            vertList.push_back(c);
        }
    }
    addParticles(vertList, type);
}

template <typename TDataType>
void FastMultiphaseSPH<TDataType>::loadParticlesBallVolume(Coord center, Real r, Real distance, particle_t type)
{
    std::vector<Coord> vertList;
    std::vector<Coord> normalList;

    Coord lo = center - r;
    Coord hi = center + r;

    for (Real x = lo[0]; x <= hi[0]; x += distance)
    {
        for (Real y = lo[1]; y <= hi[1]; y += distance)
        {
            for (Real z = lo[2]; z <= hi[2]; z += distance)
            {
                Coord p = Coord(x, y, z);
                if ((p - center).norm() < r)
                {
                    vertList.push_back(Coord(x, y, z));
                }
            }
        }
    }
    normalList.resize(vertList.size());

    m_pSet->setPoints(vertList);
    m_pSet->setNormals(normalList);

    addParticles(vertList, type);

    vertList.clear();
    normalList.clear();
}

template <typename TDataType>
void FastMultiphaseSPH<TDataType>::loadParticlesAABBVolume(Coord lo, Coord hi, Real distance, particle_t type)
{
    std::vector<Coord> vertList;
    std::vector<Coord> normalList;

    for (Real x = lo[0]; x <= hi[0]; x += distance)
    {
        for (Real y = lo[1]; y <= hi[1]; y += distance)
        {
            for (Real z = lo[2]; z <= hi[2]; z += distance)
            {
                Coord p = Coord(x, y, z);
                vertList.push_back(Coord(x, y, z));
            }
        }
    }
    normalList.resize(vertList.size());

    m_pSet->setPoints(vertList);
    m_pSet->setNormals(normalList);

    std::cout << "particle number: " << vertList.size() << std::endl;

    addParticles(vertList, type);

    vertList.clear();
    normalList.clear();
}

template <typename TDataType>
void FastMultiphaseSPH<TDataType>::loadParticlesAABBSurface(Coord lo, Coord hi, Real distance, particle_t type)
{
    std::vector<Coord> vertList;
    std::vector<Coord> normalList;

    for (Real x = lo[0]; x <= hi[0]; x += distance)
    {
        for (Real y = lo[1]; y <= hi[1]; y += distance)
        {
            vertList.push_back(Coord(x, y, lo[2]));
            vertList.push_back(Coord(x, y, hi[2]));
        }
    }
    for (Real x = lo[0]; x <= hi[0]; x += distance)
    {
        for (Real z = lo[2]; z <= hi[2]; z += distance)
        {
            vertList.push_back(Coord(x, lo[1], z));
            vertList.push_back(Coord(x, hi[1], z));
        }
    }
    for (Real y = lo[1]; y <= hi[1]; y += distance)
    {
        for (Real z = lo[2]; z <= hi[2]; z += distance)
        {
            vertList.push_back(Coord(lo[0], y, z));
            vertList.push_back(Coord(hi[0], y, z));
        }
    }
    normalList.resize(vertList.size());

    m_pSet->setPoints(vertList);
    m_pSet->setNormals(normalList);

    std::cout << "particle number: " << vertList.size() << std::endl;

    addParticles(vertList, type);

    vertList.clear();
    normalList.clear();
}

template <typename TDataType>
TDataType::Real FastMultiphaseSPH<TDataType>::getSpacing()
{
    return m_msph->h_param.spacing;
}
template <typename TDataType>
void FastMultiphaseSPH<TDataType>::setDissolutionFlag(int dissolution)
{
    m_msph->h_param.dissolution = dissolution;
}

template <typename TDataType>
void FastMultiphaseSPH<TDataType>::addParticles(const std::vector<Coord>& points, particle_t type)
{
    if (type == particle_t::SAND)
    {
        float volfrac[] = { 0, 1, 0 };
        m_msph->addParticles(points.size(), ( cfloat3* )points.data(), volfrac, 0, TYPE_GRANULAR, 1);
    }
    else if (type == particle_t::FLUID)
    {
        float volfrac[] = { 1, 0, 0 };
        m_msph->addParticles(points.size(), ( cfloat3* )points.data(), volfrac, 0, TYPE_FLUID, 1);
    }
    else if (type == particle_t::BOUDARY)
    {
        float volfrac[] = { 0, 0, 0 };
        m_msph->addParticles(points.size(), ( cfloat3* )points.data(), volfrac, GROUP_FIXED, TYPE_RIGID, 0);
    }
}

template <typename TDataType>
bool FastMultiphaseSPH<TDataType>::initialize()
{
    return Node::initialize();
}

// 	template<typename TDataType>
// 	void FastMultiphaseSPH<TDataType>::setVisible(bool visible)
// 	{
// 		if (m_pointsRender == nullptr)
// 		{
// 			m_pointsRender = std::make_shared<PointRenderModule>();
// 			this->addVisualModule(m_pointsRender);
// 		}
//
// 		Node::setVisible(visible);
// 	}

struct ColorOp
{
    __host__ __device__
        Vector3f
        operator()(Vector4f color)
    {
        return Vector3f(1 - color[0], 1 - color[0], 1 - color[0]);
    }
};

template <typename TDataType>
void FastMultiphaseSPH<TDataType>::updateTopology()
{
    //if (!this->currentPosition()->isEmpty())
    //{
    //	int num = this->currentPosition()->getElementCount();
    //	auto& pts = m_pSet->getPoints();
    //	if (num != pts.size())
    //	{
    //		pts.resize(num);
    //	}
    //	Function1Pt::copy(pts, this->currentPosition()->getValue());
    //}
    prepareData();
    auto pts = m_pSet->getPoints();
    cudaMemcpy(pts.getDataPtr(), m_pos.getDataPtr(), sizeof(Coord) * num_o, cudaMemcpyDeviceToDevice);
    Vector3f* color_idx = m_phase_concentration.getValue().getDataPtr();
    thrust::transform(thrust::device, m_color.getDataPtr(), m_color.getDataPtr() + num_o, color_idx, ColorOp());
}

template <typename TDataType>
bool FastMultiphaseSPH<TDataType>::resetStatus()
{
    auto pts = m_pSet->getPoints();

    if (pts.size() > 0)
    {
        this->currentPosition()->setElementCount(pts.size());
        this->currentVelocity()->setElementCount(pts.size());
        this->currentForce()->setElementCount(pts.size());

        Function1Pt::copy(this->currentPosition()->getValue(), pts);
        this->currentVelocity()->getReference()->reset();
    }

    return Node::resetStatus();
}

// 	template<typename TDataType>
// 	std::shared_ptr<PointRenderModule> FastMultiphaseSPH<TDataType>::getRenderModule()
// 	{
// // 		if (m_pointsRender == nullptr)
// // 		{
// // 			m_pointsRender = std::make_shared<PointRenderModule>();
// // 			this->addVisualModule(m_pointsRender);
// // 		}
//
// 		return m_pointsRender;
// 	}
}  // namespace PhysIKA