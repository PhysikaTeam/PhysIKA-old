/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: simple mass spring problem
 * @version    : 1.0
 */
#include <memory>

#include <boost/property_tree/ptree.hpp>

#include "Common/error.h"

// TODO: possible bad idea of having dependence to model in problem module
#include "Model/fem/elas_energy.h"
#include "Model/fem/mass_matrix.h"
#include "Model/mass_spring/mass_spring_obj.h"
#include "Model/mass_spring/para.h"

#include "Problem/energy/basic_energy.h"
#include "Io/io.h"
#include "Geometry/extract_surface.imp"

#include "mass_spring_problem.h"

namespace PhysIKA {
using namespace std;
using namespace Eigen;

template <typename T>
using MAT = Eigen::Matrix<T, -1, -1>;
template <typename T>
using VEC = Eigen::Matrix<T, -1, 1>;

template <typename T>
ms_problem_builder<T>::ms_problem_builder(const T* x, const boost::property_tree::ptree& para_tree)
    : pt_(para_tree)
{
    auto blender            = para_tree.get_child("blender");
    auto simulation_para    = para_tree.get_child("simulation_para");
    auto common             = para_tree.get_child("common");
    para::dt                = common.get<double>("time_step", 0.01);
    para::line_search       = simulation_para.get<int>("line_search", true);  // todo
    para::density           = common.get<double>("density", 10);
    para::frame             = common.get<int>("frame", 100);
    para::newton_fastMS     = simulation_para.get<string>("newton_fastMS");
    para::stiffness         = simulation_para.get<double>("stiffness", 8000);
    para::gravity           = common.get<double>("gravity", 9.8);
    para::object_name       = blender.get<string>("surf");
    para::out_dir_simulator = common.get<string>("out_dir_simulator");
    para::simulation_type   = simulation_para.get<string>("simulation", "static");
    para::weight_line_search =
        simulation_para.get<double>("weight_line_search", 1e-5);
    para::input_object   = common.get<string>("input_object");
    para::force_function = simulation_para.get<string>("force_function");
    para::intensity      = simulation_para.get<double>("intensity");
    para::coll_z         = simulation_para.get<bool>("coll_z", false);
    //TODO: need to check exception

    const string      filename = para::input_object;
    Matrix<T, -1, -1> nods;
    MatrixXi          cells;
    IF_ERR(exit, mesh_read_from_vtk<T, 4>(filename.c_str(), nods, cells));

    const size_t num_nods = nods.cols();
    if (x != nullptr)
        nods = Map<const MAT<T>>(x, nods.rows(), nods.cols());

    REST_  = nods;
    cells_ = cells;

    //read fixed points
    vector<size_t> cons(0);
    if (para_tree.find("input_constraint") != para_tree.not_found())
    {
        const string cons_file_path = common.get<string>("input_constraint");
        /* IF_ERR(exit, read_fixed_verts_from_csv(cons_file_path.c_str(), cons));*/
    }
    cout << "constrint " << cons.size() << " points" << endl;

    //calc mass vector
    Matrix<T, -1, 1> mass_vec(num_nods);
    calc_mass_vector<T>(nods, cells, para::density, mass_vec);
    // mass_calculator<T, 3, 4, 1, 1, basis_func, quadrature>(nods, cells, para::density, mass_vec);

    cout << "build energy" << endl;
    int ELAS = 0;
    int GRAV = 1;
    int KIN  = 2;
    int POS  = 3;
    if (pt_.get<string>("solver_type") == "explicit")
        POS = 2;

    ebf_.resize(POS + 1);
    ebf_[ELAS] = make_shared<MassSpringObj<T>>(para::input_object.c_str(), para::stiffness);
    char axis  = common.get<char>("grav_axis", 'y') | 0x20;

    ebf_[GRAV] = make_shared<gravity_energy<T, 3>>(num_nods, 1, para::gravity, mass_vec, axis);
    kinetic_   = make_shared<momentum<T, 3>>(nods.data(), num_nods, mass_vec, para::dt);

    if (pt_.get<string>("solver_type") == "implicit")
        ebf_[KIN] = kinetic_;
    ebf_[POS] = make_shared<position_constraint<T, 3>>(nods.data(), num_nods, simulation_para.get<double>("w_pos", 1e6), cons);

    //set constraint
    enum constraint_type
    {
        COLL
    };
    cbf_.resize(COLL + 1);
    collider_  = nullptr;
    cbf_[COLL] = collider_;

    if (pt_.get<string>("solver_type") == "explicit")
    {
        Map<Matrix<T, -1, 1>> position(REST_.data(), REST_.size());
        semi_implicit_ = make_shared<semi_implicit<T>>(para::dt, mass_vec, position);
    }
}

template <typename T>
std::shared_ptr<Problem<T, 3>> ms_problem_builder<T>::build_problem() const
{
    cout << "assemble energy" << endl;
    shared_ptr<Functional<T, 3>> energy;
    try
    {
        energy = build_energy_t<T, 3>(ebf_);
    }
    catch (std::exception& e)
    {
        cerr << e.what() << endl;
        exit(EXIT_FAILURE);
    }

    shared_ptr<Constraint<T>> constraint;
    cout << "assemble constraint" << endl;
    bool all_null = true;
    for (auto& c : cbf_)
        if (c != nullptr)
            all_null = false;
    if (all_null)
    {
        constraint = nullptr;
        cout << "WARNGING: No hard constraints." << endl;
    }
    else
    {
        try
        {
            constraint = build_constraint_t<T>(cbf_);
        }
        catch (std::exception& e)
        {
            cerr << e.what() << endl;
            exit(EXIT_FAILURE);
        }
    }

    exit_if(constraint != nullptr && energy->Nx() != constraint->Nx(), "energy and constraint has different dimension.");
    return make_shared<Problem<T, 3>>(energy, constraint);
}

template <typename T>
int ms_problem_builder<T>::update_problem(const T* x, const T* v)
{
    IF_ERR(return, kinetic_->update_location_and_velocity(x, v));
    if (collider_ != nullptr)
        IF_ERR(return, collider_->update(x));
    return 0;
}

template class ms_problem_builder<double>;

template class ms_problem_builder<float>;

}  // namespace PhysIKA
