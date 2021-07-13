/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: embedded elasticity mass spring method problem
 * @version    : 1.0
 */
#include <memory>
#include <string>
#include <boost/property_tree/ptree.hpp>

#include "Common/error.h"

// TODO: possible bad idea of having dependence to model in problem module
#include "Model/fem/elas_energy.h"
#include "Model/fem/mass_matrix.h"
#include "Model/mass_spring/mass_spring_obj.h"
#include "Model/mass_spring/para.h"
#include "Geometry/extract_surface.imp"
#include "Geometry/interpolate.h"

#include "Problem/energy/basic_energy.h"
#include "Io/io.h"
#include "Geometry/extract_surface.imp"
#include "libigl/include/igl/readOBJ.h"

#include "embedded_mass_spring_problem.h"

namespace PhysIKA {
using namespace std;
using namespace Eigen;
using namespace igl;

template <typename T>
using MAT = Eigen::Matrix<T, -1, -1>;
template <typename T>
using VEC = Eigen::Matrix<T, -1, 1>;

template <typename T>
embedded_ms_problem_builder<T>::embedded_ms_problem_builder(const T* x, const boost::property_tree::ptree& para_tree)
{
    pt_                     = para_tree;
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

    const string filename        = common.get<string>("embedded_object", "");
    const string filename_coarse = common.get<string>("input_object", "");
    if (filename_coarse.empty() || filename.empty())
    {
        cerr << "no coarse mesh" << __LINE__ << endl;
        exit(1);
    }

    Matrix<T, -1, -1> nods;
    MatrixXi          cells;
    Matrix<T, -1, -1> nods_coarse;
    MatrixXi          cells_coarse;

    if (filename.rfind(".obj") != string::npos)
    {
        readOBJ(filename.c_str(), nods, cells);
        nods.transposeInPlace();
        cells.transposeInPlace();
    }
    else
    {
        IF_ERR(exit, mesh_read_from_vtk<T, 4>(filename.c_str(), nods, cells));
    }
    IF_ERR(exit, mesh_read_from_vtk<T, 4>(filename_coarse.c_str(), nods_coarse, cells_coarse));

    if (cells.size() == 0)
        cells.resize(4, 0);
    if (cells_coarse.size() == 0)
        cells_coarse.resize(4, 0);
    interp_pts_in_tets<T, 3>(nods, cells, nods_coarse, fine_to_coarse_coef_);
    interp_pts_in_tets<T, 3>(nods_coarse, cells_coarse, nods, coarse_to_fine_coef_);

    const size_t num_nods = nods_coarse.cols();
    if (x != nullptr)
    {
        nods        = Map<const MAT<T>>(x, nods.rows(), nods.cols());
        nods_coarse = nods * fine_to_coarse_coef_;
    }

    REST_           = nods;
    cells_          = cells;
    fine_verts_num_ = REST_.cols();

    //read fixed points
    vector<size_t> cons(0);
    if (para_tree.find("input_constraint") != para_tree.not_found())
    {
        const string cons_file_path = common.get<string>("input_constraint");
        /*  IF_ERR(exit, read_fixed_verts_from_csv(cons_file_path.c_str(), cons));*/
    }
    cout << "constrint " << cons.size() << " points" << endl;

    //calc mass vector
    Matrix<T, -1, 1> mass_vec(num_nods);
    calc_mass_vector<T>(nods_coarse, cells_coarse, para::density, mass_vec);
    // mass_calculator<T, 3, 4, 1, 1, basis_func, quadrature>(nods_coarse, cells_coarse, para::density, mass_vec);

    cout << "build energy" << endl;
    int ELAS = 0;
    int GRAV = 1;
    int KIN  = 2;
    int POS  = 3;
    if (para_tree.get<std::string>("solver_type") == "explicit")
        POS = 2;

    ebf_.resize(POS + 1);
    ebf_[ELAS] = make_shared<MassSpringObj<T>>(para::input_object.c_str(), para::stiffness);
    char axis  = common.get<char>("grav_axis", 'y') | 0x20;
    ebf_[GRAV] = make_shared<gravity_energy<T, 3>>(num_nods, 1, para::gravity, mass_vec, axis);
    kinetic_   = make_shared<momentum<T, 3>>(nods_coarse.data(), num_nods, mass_vec, para::dt);

    if (para_tree.get<string>("solver_type") == "implicit")
        ebf_[KIN] = kinetic_;

    ebf_[POS] = make_shared<position_constraint<T, 3>>(nods_coarse.data(), num_nods, simulation_para.get<double>("w_pos", 1e6), cons);

    //set constraint
    enum constraint_type
    {
        COLL
    };
    cbf_.resize(COLL + 1);
    collider_  = nullptr;
    cbf_[COLL] = collider_;

    shared_ptr<Problem<T, 3>> pb      = make_shared<Problem<T, 3>>(ebf_[0], nullptr);
    auto                      dat_str = make_shared<dat_str_core<T, 3>>(pb->Nx() / 3, para_tree.get<bool>("hes_is_const", false));
    compute_hes_pattern(pb->energy_, dat_str);
    ebf_[0]->Hes(nods_coarse.data(), dat_str);
    SparseMatrix<T> K = dat_str->get_hes();

    embedded_interp_ = make_shared<embedded_interpolate<T>>(nods_coarse, coarse_to_fine_coef_, fine_to_coarse_coef_, K, 5868.03 / 2);

    if (para_tree.get<string>("solver_type") == "explicit")
    {
        Map<Matrix<T, -1, 1>> position(nods_coarse.data(), nods_coarse.size());
        semi_implicit_ = make_shared<semi_implicit<T>>(para::dt, mass_vec, position);
    }
}

template <typename T>
int embedded_ms_problem_builder<T>::update_problem(const T* x, const T* v)
{
    embedded_interp_->update_verts(x, fine_verts_num_);
    const Eigen::Matrix<T, -1, -1>& verts = embedded_interp_->get_verts();

    IF_ERR(return, kinetic_->update_location_and_velocity(verts.data(), v));
    if (collider_ != nullptr)
        IF_ERR(return, collider_->update(verts.data()));
    return 0;
}

template class embedded_ms_problem_builder<double>;

template class embedded_ms_problem_builder<float>;

}  // namespace PhysIKA
