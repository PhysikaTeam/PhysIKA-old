#ifndef PHYSIKA_DYNAMICS_SHALLOW_WATER_SWE_H_
#define PHYSIKA_DYNAMICS_SHALLOW_WATER_SWE_H_
#include <string>
#include <vector>
#include <map>
#include <set>
#include "Physika_Dynamics/Shallow_Water/Utilities/Field.h
#include "Physika_Geometry/Cartesian_Grids/grid.h"
#include "Physika_Dynamics/Utilities/Grid_Generalized_Vectors/uniform_grid_generalized_vector_TV.h"
#include "Physika_Dynamics/Driver/driver_base.h"
namespace Physika{
template<typename Scalar> class DriverPluginBase;
template <typename Scalar, int Dim>
class SWE:public DriverBase<Scalar>
{
public:
    SWE();
    SWE(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file);
    SWE(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file, const Grid<Scalar,Dim> &grid);
    virtual ~SWE();
    //virtual methods
    virtual void initConfiguration(const std::string &file_name);
    virtual void printConfigFileFormat();
    virtual void initSimulationData();
    virtual void addPlugin(DriverPluginBase<Scalar> *plugin);
    virtual bool withRestartSupport() const;
    virtual void write(const std::string &file_name);
    virtual void read(const std::string &file_name);
    const Grid<Scalar,Dim>& grid() const;
    void setGrid(const Grid<Scalar,Dim> &grid);
    void setgravity(const Scalar &g);
    void setdx(const Scalar &x);
    Field<Scalar> advect_height(Scalar time_step) const;
    Field<Scalar> advect_vx(Scalar time_step) const;
    Field<Scalar> advect_vy(Scalar time_step) const;
    void  update_height(Scalar timestep);
    void  update_vx(Scalar timestep);
    void  update_vy(Scalar timestep);
    void elur(Scalar dt);
    void setbondarycondition();
    void setxy(const size_t &x,const size_t &y);
    void setvx(const std::vector<Scalar> &input);
    void setvy(const std::vector<Scalar> &input);
    void setsurface(const std::vector<Scalar> &input);
    void setheight(const std::vector<Scalar> &input);
protected:
    Grid<Scalar,Dim> grid_;
    Field<Scalar> grid_vx_;
    Field<Scalar> grid_vy_;
    Field<Scalar> grid_height_;
    Field<Scalar> grid_surface_;
    size_t xcells;
    size_t ycells;
    Scalar dx;
    Scalar gravity;
    
};
}
#endif
