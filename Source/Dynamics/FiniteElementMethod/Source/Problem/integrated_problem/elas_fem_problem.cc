/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: elasticity finite element method problem
 * @version    : 1.0
 */
#include <memory>
#include <iomanip>
#include <boost/property_tree/ptree.hpp>

#include "Common/DEFINE_TYPE.h"
#include "Common/error.h"

// TODO: possible bad idea of having dependence to model in problem module
#include "Model/fem/elas_energy.h"
#include "Model/fem/mass_matrix.h"

#include "Problem/energy/basic_energy.h"
#include "Io/io.h"
#include "Geometry/extract_surface.imp"

#include "elas_fem_problem.h"

namespace PhysIKA {
using namespace std;
using namespace Eigen;

template <typename T>
int read_elas_mtr(const char* file, VEC<T>& Young, VEC<T>& Poi, const size_t num_cells)
{
    cout << "read_elas_mtr" << endl;
    Young.resize(num_cells);
    Poi.resize(num_cells);
    ifstream ifs(file);
    if (ifs.fail())
    {
        std::cerr << "[info] "
                  << "can not open file" << file << std::endl;
        return __LINE__;
    }
    size_t cell_id = 0;
    while (!ifs.eof())
    {
        if (cell_id == num_cells)
            break;
        ifs >> Young(cell_id) >> Poi(cell_id);
        ++cell_id;
    }
    ifs.close();
    return 0;
}

template <typename T>
elas_problem_builder<T>::elas_problem_builder(const T* x, const boost::property_tree::ptree& pt)
    : pt_(pt)
{
    if (pt.get<string>("solver_type") == "explicit")
    {
#define SEMI_IMPLICIT
    }

    //TODO: need to check exception
    const string filename = pt.get<string>("filename");
    MAT<T>       nods(1, 1);
    MatrixXi     cells(1, 1);

    const string type = pt.get<string>("type", "tet");

    if (type == "tet")
    {
        IF_ERR(exit, mesh_read_from_vtk<T, 4>(filename.c_str(), nods, cells));
    }
    else if (type == "hex")
        exit_if(mesh_read_from_vtk<T, 8>(filename.c_str(), nods, cells));
    else
    {
        // error_msg("type:<%s> is not supported.", type.c_str());
    }

    const size_t num_nods = nods.cols(), num_cells = cells.cols();
    cout << "V" << nods.rows() << " " << nods.cols() << endl
         << "T " << cells.rows() << " " << cells.cols() << endl;
    if (x != nullptr)
        nods = Map<const MAT<T>>(x, nods.rows(), nods.cols());
    cout << "Boundary Box :\n"
         << nods.rowwise().minCoeff() << endl
         << nods.rowwise().maxCoeff() << endl;

    REST_  = nods;
    cells_ = cells;

    Matrix<T, 3, 3> rot;
    rot << 0, 0, 1, 0, 1, 0, -1, 0, 0;
    cout << "rotatoin matrix is " << rot << endl;
    MAT<T> rotated_nods = rot * nods;

    // const string outdir = argv[3];
    auto phy_paras = pt.get_child("physics");
    //set mtr
    const T      rho       = phy_paras.get<T>("rho", 20);
    const T      Young     = phy_paras.get<T>("Young", 2000.0);
    const T      poi       = phy_paras.get<T>("poi", 0.3);
    const T      gravity   = phy_paras.get<T>("gravity", 9.8);
    const T      dt        = phy_paras.get<T>("dt", 0.01);
    const T      w_pos     = phy_paras.get<T>("w_pos", 1e6);
    const size_t num_frame = phy_paras.get<size_t>("num_frames", 100);

    //set mtr
    VEC<T>       Young_vec(num_cells), Poi_vec(num_cells);
    const string mtr_file = pt.get<string>("mtr_file", "");
    if (mtr_file != "")
        read_elas_mtr(mtr_file.c_str(), Young_vec, Poi_vec, num_cells);

    //read fixed points
    vector<size_t> cons(0);
    const string   cons_file_path = pt.get<string>("cons", "");
    if (cons_file_path != "")
        IF_ERR(exit, read_fixed_verts_from_csv(cons_file_path.c_str(), cons));
    cout << "constrint " << cons.size() << " points" << endl;

    //calc mass vector
    Matrix<T, -1, 1> mass_vec(num_nods);
    // calc_mass_vector<T>(nods, cells, rho, mass_vec);
    if (type == "tet")
        mass_calculator<T, 3, 4, 1, 1, basis_func, quadrature>(nods, cells, rho, mass_vec);
    else if (type == "hex")
        mass_calculator<T, 3, 8, 1, 2, basis_func, quadrature>(nods, cells, rho, mass_vec);

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
        if (pt.get<bool>("rotate", false))
            nods = rotated_nods;
        if (mtr_file != "")
            gen_elas_energy_intf<T>(type, csttt_type, nods, cells, Young_vec, Poi_vec, ebf_[ELAS], &elas_intf_);
        else
            gen_elas_energy_intf<T>(type, csttt_type, nods, cells, Young, poi, ebf_[ELAS], &elas_intf_);
        nods = REST_;
        // to lowercase.
        char axis = pt.get<char>("grav_axis", 'y') | 0x20;
        if (axis > 'z' || axis < 'x')
        {
            // error_msg("grav_axis should be one of x(X), y(Y) or z(Z).");
        }
        ebf_[GRAV] = make_shared<gravity_energy<T, 3>>(num_nods, 1, gravity, mass_vec, axis);

        kinetic_ = pt.get<bool>("dynamics", true) ? make_shared<momentum<T, 3>>(nods.data(), num_nods, mass_vec, dt)
                                                  : nullptr;

        if (pt_.get<string>("solver_type") == "implicit")
            ebf_[KIN] = kinetic_;

        ebf_[POS] = make_shared<position_constraint<T, 3>>(nods.data(), num_nods, w_pos, cons);
    }

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
        semi_implicit_ = make_shared<semi_implicit<T>>(dt, mass_vec, position);
    }
}

template <typename T>
std::shared_ptr<Problem<T, 3>> elas_problem_builder<T>::build_problem() const
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
int elas_problem_builder<T>::update_problem(const T* x, const T* v)
{
    if (kinetic_ != nullptr)
        IF_ERR(return, kinetic_->update_location_and_velocity(x, v));
    if (collider_ != nullptr)
        IF_ERR(return, collider_->update(x));
    return 0;
}

template class elas_problem_builder<double>;

template class elas_problem_builder<float>;

}  // namespace PhysIKA
