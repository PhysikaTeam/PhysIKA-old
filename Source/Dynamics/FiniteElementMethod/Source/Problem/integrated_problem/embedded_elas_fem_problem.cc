/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: embedded elasticity finite element method problem
 * @version    : 1.0
 */
#include <memory>
#include <iomanip>
#include <boost/property_tree/ptree.hpp>

#include "Common/error.h"

// TODO: possible bad idea of having dependence to model in problem module
#include "Model/fem/elas_energy.h"
#include "Model/fem/mass_matrix.h"

#include "Problem/energy/basic_energy.h"
#include "Io/io.h"
#include "Geometry/extract_surface.imp"
#include "Geometry/interpolate.h"
#include "libigl/include/igl/readOBJ.h"

#include "embedded_elas_fem_problem.h"

namespace PhysIKA {
using namespace std;
using namespace Eigen;
using namespace igl;

template <typename T>
using MAT = Eigen::Matrix<T, -1, -1>;
template <typename T>
using VEC = Eigen::Matrix<T, -1, 1>;

template <typename T>
embedded_elas_problem_builder<T>::embedded_elas_problem_builder(const T* x, const boost::property_tree::ptree& pt)
    : pt_(pt)
{
    //TODO: need to check exception
    using Vector3T               = Eigen::Matrix<T, 3, 1>;
    const string filename        = pt.get<string>("filename");
    const string filename_coarse = pt.get<string>("filename_coarse");
    cout << "filename " << filename << "\n filename_coarse " << filename_coarse << endl;
    MAT<T>   nods;
    MatrixXi cells;

    MAT<T>   nods_coarse;
    MatrixXi cells_coarse;

    string type = pt.get<string>("type", "tet");
    if (type == "vox")
        type = "hex";
    string type_coarse = pt.get<string>("type_coarse", type);
    if (type_coarse == "vox")
        type_coarse = "hex";
    cout << "mesh type is " << type << endl;
    cout << "coarse type is " << type_coarse << endl;
    if (type_coarse == "tet")
    {
        IF_ERR(exit, mesh_read_from_vtk<T, 4>(filename_coarse.c_str(), nods_coarse, cells_coarse));
    }
    else if (type_coarse == "hex")
    {
        exit_if(mesh_read_from_vtk<T, 8>(filename_coarse.c_str(), nods_coarse, cells_coarse));
    }
    else if (type_coarse == "hybrid")
    {
        cout << "read hybrid mesh " << filename << endl;
        exit_if(mesh_read_from_vtk<T, 8>(filename_coarse.c_str(), nods_coarse));
    }
    else
    {
        // error_msg("type:<%s> is not supported.", type.c_str());
    }
    cout << "number of cells is " << cells_coarse.cols() << endl;
#if 1
    if (type_coarse == "hybrid")
    {
        Vector3T nods_min = nods_coarse.col(0);
        Vector3T nods_max = nods_coarse.col(1);
        for (size_t i = 0; i < 3; ++i)
        {
            nods_min(i) = nods_coarse.row(i).minCoeff();
            nods_max(i) = nods_coarse.row(i).maxCoeff();
        }
        Matrix<T, 3, 8> nods_coarsest = Matrix<T, 3, 8>::Ones();
        //set z
        nods_coarsest.block<1, 4>(2, 0) *= nods_min(2);
        nods_coarsest.block<1, 4>(2, 4) *= nods_max(2);
        //set y
        Vector4i y_min{ { 2, 3, 6, 7 } };
        Vector4i y_max{ { 0, 1, 4, 5 } };
        nods_coarsest(1, y_min) *= nods_min(1);
        nods_coarsest(1, y_max) *= nods_max(1);
        //set x
        Vector4i x_min{ { 0, 3, 4, 7 } };
        Vector4i x_max{ { 1, 2, 5, 6 } };
        nods_coarsest(0, x_min) *= nods_min(0);
        nods_coarsest(0, x_max) *= nods_max(0);

        nods_coarse = nods_coarsest;

        cout << nods_coarse << endl;
        cells_coarse.resize(8, 1);
        cells_coarse.col(0).setLinSpaced(8, 0, 7);
        Eigen::MatrixXi hexs2tets = hex_2_tet(cells_coarse);
        cells_coarse              = hexs2tets;

        cout << cells_coarse << endl;
        type_coarse = "tet";
    }
#endif
    if (filename.rfind(".obj") != string::npos)
    {
        readOBJ(filename.c_str(), nods, cells);
        nods.transposeInPlace();
        cells.transposeInPlace();
    }
    else
    {
        //TODO: need to check file reading error
        if (type == "tet")
        {
            IF_ERR(exit, mesh_read_from_vtk<T, 4>(filename.c_str(), nods, cells));
        }
        else if (type == "hex")
        {
            exit_if(mesh_read_from_vtk<T, 8>(filename.c_str(), nods, cells));
        }
        //else if (type == "hybrid") {
        //	cout << "read hybrid mesh " << filename << endl;
        //	exit_if(mesh_read_from_vtk<T, 8>(filename.c_str(), nods));
        //

        //}
        else
        {
            // error_msg("type:<%s> is not supported.", type.c_str());
        }
    }

    if (cells.size() == 0)
        cells.resize(4, 0);
    if (cells_coarse.size() == 0)
        cells_coarse.resize(4, 0);
    interp_pts_in_tets<T, 3>(nods, cells, nods_coarse, fine_to_coarse_coef_);
    //interp_pts_in_point_cloud<T, 3>(nods_coarse, nods, coarse_to_fine_coef_);

    if (type_coarse == "tet")
        interp_pts_in_tets<T, 3>(nods_coarse, cells_coarse, nods, coarse_to_fine_coef_);
    else
    {
        //interp_pts_in_point_cloud<T, 3>(nods_coarse, nods, coarse_to_fine_coef_);
        Eigen::MatrixXi hexs2tets = hex_2_tet(cells_coarse);
        cout << "size of hexs2tets " << hexs2tets.rows() << " " << hexs2tets.cols() << endl;
        interp_pts_in_tets<T, 3>(nods_coarse, hexs2tets, nods, coarse_to_fine_coef_);
    }

    const size_t num_nods = nods_coarse.cols();
    cout << "V" << nods_coarse.rows() << " " << nods_coarse.cols() << endl
         << "T " << cells_coarse.rows() << " " << cells_coarse.cols() << endl;
    if (x != nullptr)
    {
        nods        = Map<const MAT<T>>(x, nods.rows(), nods.cols());
        nods_coarse = nods * fine_to_coarse_coef_;
    }

    cout << "Boundary Box :\n"
         << nods_coarse.rowwise().minCoeff() << endl
         << nods_coarse.rowwise().maxCoeff() << endl;

    REST_  = nods;
    cells_ = cells;
    /*Matrix<T, -1, -1> nods_temp = nods_coarse;*/
    fine_verts_num_ = REST_.cols();

    auto         phy_paras = pt.get_child("physics");
    const T      rho       = phy_paras.get<T>("rho", 20);
    const T      Young     = phy_paras.get<T>("Young", 2000.0);
    const T      poi       = phy_paras.get<T>("poi", 0.3);
    const T      gravity   = phy_paras.get<T>("gravity", 9.8);
    const T      dt        = phy_paras.get<T>("dt", 0.01);
    const T      w_pos     = phy_paras.get<T>("w_pos", 1e6);
    const size_t num_frame = phy_paras.get<size_t>("num_frames", 100);

    //read fixed points
    vector<size_t> cons(0);
    const string   cons_file_path = pt.get<string>("cons", "");
    /*if(cons_file_path != "")
    IF_ERR(exit, read_fixed_verts_from_csv(cons_file_path.c_str(), cons));*/
    cout << "constrint " << cons.size() << " points" << endl;

    //calc mass vector
    Matrix<T, -1, 1> mass_vec(num_nods);
    // calc_mass_vector<T>(nods, cells, rho, mass_vec);
    if (type_coarse == "tet")
        mass_calculator<T, 3, 4, 1, 1, basis_func, quadrature>(nods_coarse, cells_coarse, rho, mass_vec);
    else if (type_coarse == "hex")
        mass_calculator<T, 3, 8, 1, 2, basis_func, quadrature>(nods_coarse, cells_coarse, rho, mass_vec);

    cout << "build energy" << endl;
    int ELAS = 0;
    int GRAV = 1;
    int KIN  = 2;
    int POS  = 3;
    if (pt_.get<string>("solver_type") == "explicit")
        POS = 2;

    ebf_.resize(POS + 1);
    {
        const string csttt_type = phy_paras.get<string>("csttt", "linear");

        gen_elas_energy_intf<T>(type_coarse, csttt_type, nods_coarse, cells_coarse, Young, poi, ebf_[ELAS], &elas_intf_);
        /* nods_coarse = nods_temp;*/
        // to lowercase.
        char axis  = pt.get<char>("grav_axis", 'y') | 0x20;
        ebf_[GRAV] = make_shared<gravity_energy<T, 3>>(num_nods, 1, gravity, mass_vec, axis);
        kinetic_   = make_shared<momentum<T, 3>>(nods_coarse.data(), num_nods, mass_vec, dt);

        if (pt_.get<string>("solver_type") == "implicit")
            ebf_[KIN] = kinetic_;

        ebf_[POS] = make_shared<position_constraint<T, 3>>(nods_coarse.data(), num_nods, w_pos, cons);
    }
    cout << "set up energy done." << endl;

    //set constraint

    enum constraint_type
    {
        COLL
    };
    cbf_.resize(COLL + 1);
    collider_  = nullptr;
    cbf_[COLL] = collider_;

    shared_ptr<Problem<T, 3>> pb      = make_shared<Problem<T, 3>>(ebf_[0], nullptr);
    auto                      dat_str = make_shared<dat_str_core<T, 3>>(pb->Nx() / 3, pt.get<bool>("hes_is_const", false));
    compute_hes_pattern(pb->energy_, dat_str);
    ebf_[0]->Hes(nods_coarse.data(), dat_str);
    SparseMatrix<T> K = dat_str->get_hes();

    embedded_interp_ = make_shared<embedded_interpolate<T>>(nods_coarse, coarse_to_fine_coef_, fine_to_coarse_coef_, K, 0.586803 / 2);

    if (pt_.get<string>("solver_type") == "explicit")
    {
        Map<Matrix<T, -1, 1>> position(nods_coarse.data(), nods_coarse.size());
        semi_implicit_ = make_shared<semi_implicit<T>>(dt, mass_vec, position);
    }
    cout << "init problem done." << endl;
}

template <typename T>
std::shared_ptr<Problem<T, 3>> embedded_elas_problem_builder<T>::build_problem() const
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
int embedded_elas_problem_builder<T>::update_problem(const T* x, const T* v)
{
    embedded_interp_->update_verts(x, fine_verts_num_);
    const Eigen::Matrix<T, -1, -1>& verts = embedded_interp_->get_verts();

    IF_ERR(return, kinetic_->update_location_and_velocity(verts.data(), v));
    if (collider_ != nullptr)
        IF_ERR(return, collider_->update(verts.data()));
    return 0;
}

template class embedded_elas_problem_builder<double>;

template class embedded_elas_problem_builder<float>;

}  // namespace PhysIKA
