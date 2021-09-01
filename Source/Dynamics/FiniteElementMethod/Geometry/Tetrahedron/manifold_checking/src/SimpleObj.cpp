#include "../inc/SimpleObj.h"

#include <iostream>

using namespace std;
using namespace Eigen;

static const double EPS = 1e-9;
static const double PI  = 3.1415926;

#define REPORT_LOCATION __FILE__ << " " << __LINE__ << ": "

static inline bool is_line_invalid(const string& line)
{
    return (line.empty() || line[0] == 13 || line[0] == '#');
}

static inline void report_err(const string& info)
{
    cerr << "Error in " << REPORT_LOCATION << info << endl;
    assert(0);
}

double cot(const double angle)
{
    return abs(tan(angle)) <= EPS ? 1.0 / EPS : 1.0 / tan(angle);
}
void set_min_max(const double value, double& min, double& max)
{
    if (value < min)
        min = value;
    if (value > max)
        max = value;
}
void set_avg(const double value, const size_t size, double& avg)
{
    avg += value / size;
    // cout << avg << endl;
}

//-------Private-------//
void SimpleObj::reset_v_min_max()
{
    v_max = Vector3d(-1e10, -1e10, -1e10);
    v_min = Vector3d(1e10, 1e10, 1e10);
}

void SimpleObj::set_v_min_max(const double x, const double y, const double z)
{
    set_min_max(x, v_min[0], v_max[0]);
    set_min_max(y, v_min[1], v_max[1]);
    set_min_max(z, v_min[2], v_max[2]);
}

double triangle_angle(const double a, const double b, const double c)
{
    // c is the opposite edge
    double cos_value = (a * a + b * b - c * c) / (2 * a * b);
    cos_value        = cos_value >= 1.0 ? 1.0 : (cos_value <= -1.0 ? -1.0 : cos_value);
    assert(acos(cos_value) >= 0 - EPS && acos(cos_value) <= PI + EPS);
    return acos(cos_value);
}
double voronoi_area(const double a, const double b, const double r)
{
    // a, b are two edges of triangle
    // r is the radius of circumcircle
    double h1    = sqrt(r * r - a * a / 4);
    double area1 = h1 * a / 4;
    double h2    = sqrt(r * r - b * b / 4);
    double area2 = h2 * b / 4;
    return area1 + area2;
}
vector<double> SimpleObj::face_property(const size_t idx, const std::string type) const
{
    Vector3d v0 = v_mat.col(f_mat(0, idx));
    Vector3d v1 = v_mat.col(f_mat(1, idx));
    Vector3d v2 = v_mat.col(f_mat(2, idx));
    double   a0 = (v2 - v1).norm();
    double   a1 = (v0 - v2).norm();
    double   a2 = (v1 - v0).norm();

    vector<double> result;
    if (type == "area")
    {
        double p    = (a0 + a1 + a2) / 2;
        double area = sqrt(p * (p - a0) * (p - a1) * (p - a2));
        assert(area >= 0 - EPS);
        result.push_back(area);
    }
    if (type == "angle")
    {
        double angle1    = triangle_angle(a2, a1, a0);
        double angle2    = triangle_angle(a0, a2, a1);
        double angle3    = triangle_angle(a1, a0, a2);
        double angle_max = max(angle1, max(angle2, angle3));
        result.push_back(angle1);
        result.push_back(angle2);
        result.push_back(angle3);
        result.push_back(angle_max);
    }
    if (type == "voronoi")
    {
        double r = (a0 * a1 * a2) / (4 * areas[idx]);
        result.push_back(voronoi_area(a1, a2, r));
        result.push_back(voronoi_area(a0, a2, r));
        result.push_back(voronoi_area(a0, a1, r));
    }
    if (type == "circumference")
    {
        result.push_back(a0 + a1 + a2);
    }
    if (type == "avg_vn_normal")
    {
        assert(vn_num > 0 && fn_mat(0, idx) >= 0 && fn_mat(1, idx) >= 0 && fn_mat(2, idx) >= 0);
        Vector3d normal(0, 0, 0);
        for (int i = 0; i < 3; i++)
        {
            normal += vn_mat.col(fn_mat(i, idx));
        }
        for (int i = 0; i < 3; i++)
        {
            result.push_back(normal(i));
        }
    }
    if (type == "tri_normal")
    {
        Vector3d normal = (v1 - v0).cross(v2 - v1);
        for (int i = 0; i < 3; i++)
        {
            result.push_back(normal(i));
        }
    }
    if (type == "naive_volume")
    {
        double volume = 0;
        volume += v0(0) * v1(1) * v2(2);
        volume += v0(1) * v1(2) * v2(0);
        volume += v0(2) * v1(0) * v2(1);
        volume -= v0(2) * v1(1) * v2(0);
        volume -= v0(1) * v1(0) * v2(2);
        volume -= v0(0) * v1(2) * v2(1);
        volume = abs(volume) / 6.0;
        result.push_back(volume);
    }
    if (type == "vn_volume")
    {
        assert(vn_num > 0 && fn_mat(0, idx) >= 0 && fn_mat(1, idx) >= 0 && fn_mat(2, idx) >= 0);
        double         volume     = face_property(idx, "naive_volume")[0];
        vector<double> normal_vec = face_property(idx, "avg_vn_normal");
        assert(normal_vec.size() == 3);
        Vector3d normal(normal_vec[0], normal_vec[1], normal_vec[2]);
        volume = normal.dot(v0 + v1 + v2) > 0 ? volume : -volume;
        result.push_back(volume);
    }
    if (type == "tri_normal_volume")
    {
        double         volume     = face_property(idx, "naive_volume")[0];
        vector<double> normal_vec = face_property(idx, "tri_normal");
        assert(normal_vec.size() == 3);
        Vector3d normal(normal_vec[0], normal_vec[1], normal_vec[2]);
        volume = normal.dot(v0 + v1 + v2) > 0 ? volume : -volume;
        result.push_back(volume);
    }
    return result;
}

size_t SimpleObj::select_correct_idx_in_face(const size_t v_idx, const size_t f_idx)
{
    size_t correct_idx;
    for (correct_idx = 0; correct_idx < 3; correct_idx++)
    {
        if (f_mat(correct_idx, f_idx) == v_idx)
        {
            break;
        }
    }
    return correct_idx;
}

//---------Public----------//

static void vertex_statistic(ifstream& input, size_t& v_num, size_t& vn_num, size_t& vt_num)
{
    string line, word;
    v_num  = 0;
    vn_num = 0;
    vt_num = 0;
    while (getline(input, line))
    {
        if (is_line_invalid(line))
            continue;
        istringstream instream(line);
        instream >> word;
        if (word == "v" || word == "V")
        {
            v_num++;
        }
        else if (word == "vn" || word == "VN")
        {
            vn_num++;
        }
        else if (word == "vt" || word == "VT")
        {
            vt_num++;
        }
        else
        {
            continue;
        }
    }
    input.clear();
    input.seekg(0, ios::beg);
}

static void face_statistic(ifstream& input, size_t& f_num)
{
    string line, word;
    f_num = 0;
    while (getline(input, line))
    {
        if (is_line_invalid(line))
            continue;
        istringstream instream(line);
        instream >> word;
        if (word == "f" || word == "F")
        {
            f_num++;
        }
        else
        {
            continue;
        }
    }
    input.clear();
    input.seekg(0, ios::beg);
}

static void get_from_face_data(string str, int& f_val, int& ft_val, int& fn_val)
{
    ft_val = -1;
    fn_val = -1;
    if (str.find('/') == string::npos)
    {
        istringstream iss(str);
        iss >> f_val;
        return;
    }
    else if (str.find('/') == str.rfind('/'))
    {
        string        f_str  = str.substr(0, str.find('/'));
        string        ft_str = str.substr(str.find('/') + 1, str.length());
        istringstream iss1(f_str);
        iss1 >> f_val;
        istringstream iss2(ft_str);
        iss2 >> ft_val;
    }
    else
    {
        string f_str  = str.substr(0, str.find('/'));
        string ft_str = str.substr(str.find('/') + 1, str.rfind('/'));
        string fn_str = str.substr(str.rfind('/') + 1, str.length());
        if (f_str.length() > 0)
        {
            istringstream iss1(f_str);
            iss1 >> f_val;
        }
        if (ft_str.length() > 0)
        {
            istringstream iss2(ft_str);
            iss2 >> ft_val;
        }
        if (fn_str.length() > 0)
        {
            istringstream iss3(fn_str);
            iss3 >> fn_val;
        }
    }
}

void SimpleObj::set_obj(const string& obj_file)
{
    ifstream input(obj_file);
    if (!input.is_open())
    {
        cerr << "[ERR] SimpleObj::set(): File open error." << endl;
        exit(0);
    }

    int v_idx = 0, vn_idx = 0, vt_idx = 0, f_idx = 0;
    vertex_statistic(input, v_num, vn_num, vt_num);
    if (v_num > 0)
        v_mat = MatrixXd::Zero(3, v_num);
    if (vn_num > 0)
        vn_mat = MatrixXd::Zero(3, vn_num);
    if (vt_num > 0)
        vt_mat = MatrixXd::Zero(3, vt_num);
    face_statistic(input, f_num);
    if (f_num > 0)
    {
        f_mat  = MatrixXi::Zero(3, f_num);
        fn_mat = MatrixXi::Zero(3, f_num);
        ft_mat = MatrixXi::Zero(3, f_num);
    }

    string line, word;
    int    obj_idx          = -1;
    bool   setting_obj_flag = false;
    while (getline(input, line))
    {
        double val1, val2, val3;
        if (is_line_invalid(line))
            continue;
        istringstream instream(line);
        instream >> word;
        if (word == "usemtl" || word == "mtllib" || word == "g" || word == "G" || word == "s" || word == "S" || word == "o" || word == "O" || word == "L" || word == "l")
        {
            continue;
        }
        else if (word == "v" || word == "V")
        {
            instream >> val1 >> val2 >> val3;
            v_mat(0, v_idx) = val1;
            v_mat(1, v_idx) = val2;
            v_mat(2, v_idx) = val3;
            set_v_min_max(val1, val2, val3);
            v_idx++;
        }
        else if (word == "vt" || word == "VT")
        {
            instream >> val1 >> val2 >> val3;
            vt_mat(0, vt_idx) = val1;
            vt_mat(1, vt_idx) = val2;
            vt_mat(2, vt_idx) = val3;
            vt_idx++;
        }
        else if (word == "vn" || word == "VN")
        {
            instream >> val1 >> val2 >> val3;
            vn_mat(0, vn_idx) = val1;
            vn_mat(1, vn_idx) = val2;
            vn_mat(2, vn_idx) = val3;
            vn_idx++;
        }
        else if (word == "f" || word == "F")
        {
            int      f_val = 0, ft_val = 0, fn_val = 0, f_v_idx = 0;
            Vector3i f_vec, ft_vec, fn_vec;
            while (true)
            {
                string face_data;
                instream >> face_data;
                if (face_data.empty())
                    break;
                if (f_v_idx >= 3)
                    report_err("Error: In OBJ file, some faces are not triangle");
                get_from_face_data(face_data, f_val, ft_val, fn_val);
                f_mat(f_v_idx, f_idx) = f_val - 1;
                if (ft_val > 0)
                    ft_mat(f_v_idx, f_idx) = ft_val - 1;
                else
                    ft_mat(f_v_idx, f_idx) = -1;
                if (fn_val > 0)
                    fn_mat(f_v_idx, f_idx) = fn_val - 1;
                else
                    fn_mat(f_v_idx, f_idx) = -1;
                f_v_idx++;
            }
            if (f_v_idx != 3)
                report_err("Error: In OBJ file, some faces are not triangle");
            f_idx++;
        }
        else if (word == "c" || word == "C")
        {
            continue;
        }
        else
        {
            report_err("Error: In OBJ file, find unexpected parameter " + word);
        }
    }
}

void SimpleObj::write(const std::string& obj_file) const
{
    ofstream output(obj_file);
    for (int i = 0; i < v_num; i++)
    {
        output << "v " << v_mat(0, i) << " " << v_mat(1, i) << " " << v_mat(2, i) << endl;
    }
    for (int i = 0; i < vn_num; i++)
    {
        output << "vn " << vn_mat(0, i) << " " << vn_mat(1, i) << " " << vn_mat(2, i) << endl;
    }
    for (int i = 0; i < vt_num; i++)
    {
        output << "vt " << vt_mat(0, i) << " " << vt_mat(1, i) << " " << vt_mat(2, i) << endl;
    }
    for (int i = 0; i < f_num; i++)
    {
        output << "f";
        for (int k = 0; k < 3; k++)
        {
            output << " " << f_mat(k, i) + 1;
            if (ft_mat(k, i) >= 0)
            {
                output << "/" << ft_mat(k, i) + 1;
                if (fn_mat(k, i) > 0)
                {
                    output << "/" << fn_mat(k, i) + 1;
                }
            }
            else
            {
                if (fn_mat(k, i) > 0)
                {
                    output << "//" << fn_mat(k, i) + 1;
                }
            }
        }
        output << endl;
    }
    output.close();
}

void SimpleObj::normalize(const Vector3d& center, const Vector3d& rate)
{
    v_mat = (v_mat.colwise() - center).array().colwise() / rate.array();
}

void SimpleObj::normalize(Vector3d& center, Vector3d& rate, const string type, const bool inplace)
{
    Vector3d maxs = v_mat.rowwise().maxCoeff();
    Vector3d mins = v_mat.rowwise().minCoeff();
    Vector3d size = (maxs - mins) / 2;
    center        = (maxs + mins) / 2;
    if (type == "max-direction")
    {
        double max_rate = size.maxCoeff();
        rate << max_rate, max_rate, max_rate;
    }
    else if (type == "all-direction")
    {
        rate = size;
    }
    if (inplace)
    {
        normalize(center, rate);
    }
}

void SimpleObj::normalize(const string type)
{
    Vector3d center, rate;
    normalize(center, rate, type, true);
}

void SimpleObj::change_axis(const string x, const string y, const string z)
{
    auto decode_target_axis = [=](const string axis, const double x, const double y, const double z) {
        if (axis == "x")
            return x;
        else if (axis == "-x")
            return -x;
        else if (axis == "y")
            return y;
        else if (axis == "-y")
            return -y;
        else if (axis == "z")
            return z;
        else if (axis == "-z")
            return -z;
        else
            return 0.0;
    };
    for (int i = 0; i < v_num; i++)
    {
        double temp_x, temp_y, temp_z;
        {
            temp_x = v_mat(0, i);
            temp_y = v_mat(1, i);
            temp_z = v_mat(2, i);
        }
        v_mat(0, i) = decode_target_axis(x, temp_x, temp_y, temp_z);
        v_mat(1, i) = decode_target_axis(y, temp_x, temp_y, temp_z);
        v_mat(2, i) = decode_target_axis(z, temp_x, temp_y, temp_z);
    }
}

void SimpleObj::translate(const Vector3d& displacement)
{
    v_mat = v_mat.colwise() + displacement;
}

void SimpleObj::rotate(const string axis, const double angle)
{
    MatrixXd rotate_mat(3, 3);
    if (axis == "x")
    {
        rotate_mat << 1, 0, 0,
            0, cos(angle), -sin(angle),
            0, sin(angle), cos(angle);
    }
    else if (axis == "y")
    {
        rotate_mat << cos(angle), 0, sin(angle),
            0, 1, 0,
            -sin(angle), 0, cos(angle);
    }
    else if (axis == "z")
    {
        rotate_mat << cos(angle), -sin(angle), 0,
            sin(angle), cos(angle), 0,
            0, 0, 1;
    }
    v_mat = rotate_mat * v_mat;
}

void SimpleObj::clean(const double threshold, const bool verbose)
{
    vector<int> face_flat, ft_flat, fn_flat;
    size_t      cnt = 0;
    for (size_t i = 0; i < f_num; i++)
    {
        double circumference = face_property(i, "circumference")[0];
        double area          = face_property(i, "area")[0];
        cout << circumference << " " << sqrt(area) << endl;
        if (circumference < sqrt(area) * threshold)
        {
            cnt++;
            for (int j = 0; j < 3; j++)
            {
                face_flat.push_back(f_mat(j, i));
                ft_flat.push_back(ft_mat(j, i));
                fn_flat.push_back(fn_mat(j, i));
            }
        }
    }
    if (verbose)
        cout << "[INFO] Original f_num: " << f_num << "; After cleaning: " << cnt << endl;
    f_num  = cnt;
    f_mat  = Map<MatrixXi>(face_flat.data(), 3, f_num);
    ft_mat = Map<MatrixXi>(ft_flat.data(), 3, f_num);
    fn_mat = Map<MatrixXi>(fn_flat.data(), 3, f_num);
}

// One Ring

void SimpleObj::build_one_ring_faces_set(const bool verbose)
{
    one_ring_faces_set.resize(v_num);
    for (size_t i = 0; i < f_num; i++)
    {
        for (size_t j = 0; j < 3; j++)
        {
            size_t v_idx = f_mat(j, i);
            one_ring_faces_set[v_idx].insert(i);
        }
    }
    has_built_one_ring_faces_set = true;
    if (verbose)
        cout << "[INFO] build_one_ring_faces_set()" << endl;
}

void SimpleObj::build_one_ring_vertices_dict(const bool verbose)
{
    if (!has_built_one_ring_faces_set)
    {
        build_one_ring_faces_set(verbose);
    }
    one_ring_vertices_dict.resize(v_num);
    for (size_t i = 0; i < v_num; i++)
    {
        for (auto fp = one_ring_faces_set[i].begin(); fp != one_ring_faces_set[i].end(); fp++)
        {
            for (size_t j = 0; j < 3; j++)
            {
                size_t v_new_idx = f_mat(j, *fp);
                if (v_new_idx != i)
                {
                    // TODO
                    if (one_ring_vertices_dict[i][v_new_idx].size() == 2)
                    {
                        cerr << "[WARNING]: build_one_ring_vertices_dict(): Exist one edge with more than two faces: vertex id: " << v_new_idx << endl;
                        continue;
                        // cout << v_new_idx << " ";
                        // for (size_t ii=0; ii<one_ring_vertices_dict[i][v_new_idx].size(); ii++) {
                        // 	cout << one_ring_vertices_dict[i][v_new_idx][ii] << " ";
                        // }
                        // cout << endl;
                    }
                    if (one_ring_vertices_dict[i].find(v_new_idx) == one_ring_vertices_dict[i].end())
                    {
                        vector<size_t> faces;
                        faces.push_back(*fp);
                        one_ring_vertices_dict[i][v_new_idx] = faces;
                    }
                    else
                    {
                        one_ring_vertices_dict[i][v_new_idx].push_back(*fp);
                    }
                }
            }
        }
    }
    has_built_one_ring_vertices_dict = true;
    if (verbose)
        cout << "[INFO] build_one_ring_vertices_dict()" << endl;
}

set<size_t> SimpleObj::find_two_ring_vertices(const size_t idx, const bool verbose)
{
    if (!has_built_one_ring_faces_set)
    {
        build_one_ring_faces_set(verbose);
    }
    set<size_t> two_rings;
    for (auto one_ring : one_ring_vertices_dict[idx])
    {
        size_t neighbor = one_ring.first;
        two_rings.insert(neighbor);
        for (auto two_ring : one_ring_vertices_dict[neighbor])
        {
            size_t neighbor2 = two_ring.first;
            if (neighbor2 != idx)
            {
                two_rings.insert(neighbor2);
            }
        }
    }
    return two_rings;
}

// Boundary

void SimpleObj::build_is_boundary(const bool verbose)
{
    if (!has_built_one_ring_faces_set)
    {
        build_one_ring_faces_set(verbose);
    }
    if (!has_built_one_ring_vertices_dict)
    {
        build_one_ring_vertices_dict(verbose);
    }
    boundary_num = 0;
    is_boundary.resize(v_num);
    for (size_t i = 0; i < v_num; i++)
    {
        is_boundary[i] = one_ring_vertices_dict[i].size() - one_ring_faces_set[i].size();
        if (is_boundary[i] > 0)
            boundary_num++;
        if (is_boundary[i] < 0 && verbose)
            cerr << "[WARNING]: Exists vertex has more one-ring faces than one-ring vertices: vertex id: " << i << endl;
    }
    if (verbose)
        cout << "[INFO] build_is_boundary()" << endl;
}

// Angle

void SimpleObj::compute_angles(const bool verbose)
{
    angles.resize(f_num);
    for (size_t i = 0; i < f_num; i++)
    {
        angles[i] = face_property(i, "angle");
    }
    has_computed_angles = true;
    if (verbose)
        cout << "[INFO] compute_angles()" << endl;
}

// Area

void SimpleObj::compute_face_areas(const bool verbose)
{
    areas.resize(f_num);
    for (size_t i = 0; i < f_num; i++)
    {
        areas[i] = face_property(i, "area")[0];
        if (areas[i] <= 0)
        {
            cerr << "[WARNING]: Exists face area is non-positive: face id: " << i << endl;
        }
    }
    if (verbose)
        cout << "[INFO] compute_face_areas()" << endl;
}

void SimpleObj::compute_face_voronoi_areas(const bool verbose)
{
    face_voronoi_areas.resize(f_num);
    for (size_t i = 0; i < f_num; i++)
    {
        if (angles[i][3] < PI / 2)
        {  // acute triangle
            face_voronoi_areas[i] = face_property(i, "voronoi");
        }
        else
        {  // obtuse triangle
            face_voronoi_areas[i].resize(3);
            for (size_t j = 0; j < 3; j++)
            {
                if (angles[i][j] < PI / 2)
                {
                    face_voronoi_areas[i][j] = areas[i] / 4;
                }
                else
                {
                    face_voronoi_areas[i][j] = areas[i] / 2;
                }
            }
        }
        for (size_t j = 0; j < 3; j++)
        {
            if (face_voronoi_areas[i][j] <= 0)
            {
                cerr << "[WARNING]: Exists face voronoi area is non-positive: face id: " << i << endl;
            }
        }
    }
    if (verbose)
        cout << "[INFO] compute_face_voronoi_areas()" << endl;
}

void SimpleObj::compute_vertex_voronoi_areas(const bool verbose)
{
    vertex_voronoi_areas.resize(v_num);
    for (size_t i = 0; i < v_num; i++)
    {
        // TODO
        if (one_ring_faces_set[i].size() == 0)
        {
            vertex_voronoi_areas[i] = 1e10;
            continue;
        }
        double area = 0;
        for (auto face = one_ring_faces_set[i].begin(); face != one_ring_faces_set[i].end(); face++)
        {
            size_t v_pos = select_correct_idx_in_face(i, *face);
            area += face_voronoi_areas[*face][v_pos];
        }
        vertex_voronoi_areas[i] = area;
        if (vertex_voronoi_areas[i] <= 0)
        {
            cerr << "[WARNING] compute_vertex_voronoi_areas(): Exists vertex voronoi area is 0: vertex id: " << i
                 << ", one-ring face number: " << one_ring_faces_set[i].size() << endl;
        }
    }
    if (verbose)
        cout << "[INFO] compute_vertex_voronoi_areas()" << endl;
}

// Volume

double SimpleObj::compute_directed_volume(const string& type, const bool verbose)
{
    if (type == "vn")
    {
        assert(vn_num > 0);
    }

    double volume = 0;
    for (size_t i = 0; i < f_num; i++)
    {
        vector<double> result;
        if (type == "vn")
        {
            result = face_property(i, "vn_volume");
        }
        if (type == "tri")
        {
            result = face_property(i, "tri_normal_volume");
        }
        volume += result[0];
    }
    if (verbose)
        cout << "[INFO] compute_directed_volume(): volume " << volume << endl;
    return volume;
}

// Curvature

double SimpleObj::sum_cot_angle(const size_t idx, const size_t neighbor)
{
    if (!has_built_one_ring_vertices_dict)
        build_one_ring_vertices_dict();
    if (!has_computed_angles)
        compute_angles();
    double sum_cot_angle = 0;
    for (auto f_idx : one_ring_vertices_dict[idx][neighbor])
    {
        size_t pos;
        for (pos = 0; pos < 3; pos++)
        {
            if (f_mat(pos, f_idx) != idx && f_mat(pos, f_idx) != neighbor)
            {
                break;
            }
        }
        sum_cot_angle += cot(angles[f_idx][pos]);
    }
    return sum_cot_angle;
}

void SimpleObj::compute_normals_KHs(const bool verbose)
{
    KHs.resize(v_num);
    normals = MatrixXd::Zero(3, v_num);
    KH_max = -1e10, KH_min = 1e10, KH_avg = 0;
    for (size_t i = 0; i < v_num; i++)
    {
        Vector3d normal(0.0, 0.0, 0.0);
        for (auto vp = one_ring_vertices_dict[i].begin(); vp != one_ring_vertices_dict[i].end(); vp++)
        {
            assert(vp->second.size() <= 2);
            int      j               = vp->first;
            Vector3d xi_xj           = v_mat.col(i) - v_mat.col(j);
            double   total_cot_angle = 0;
            for (auto fp = vp->second.begin(); fp != vp->second.end(); fp++)
            {
                size_t pos;
                for (pos = 0; pos < 3; pos++)
                {
                    if (f_mat(pos, *fp) != i && f_mat(pos, *fp) != j)
                    {
                        break;
                    }
                }
                total_cot_angle += cot(angles[*fp][pos]);
            }
            normal += total_cot_angle * xi_xj;
        }
        normal /= 2 * vertex_voronoi_areas[i];
        if (normal.norm() <= 0)
        {
            normals.col(i) = Vector3d(0, 0, 0);
        }
        else
        {
            normals.col(i) = normal / normal.norm() * -1;
        }
        KHs[i] = normal.norm() / 2;
        set_min_max(KHs[i], KH_min, KH_max);
        set_avg(KHs[i], v_num, KH_avg);
    }
    cout << KH_max << " " << KH_min << " " << KH_avg << endl;
    if (verbose)
        cout << "[INFO] compute_normals_KHs()" << endl;
}

void SimpleObj::correct_normal_direction(const size_t idx, Vector3d& normal)
{
    if (!has_built_one_ring_faces_set)
        build_one_ring_faces_set();
    Vector3d avg_face_normal(0.0, 0.0, 0.0);
    for (auto face : one_ring_faces_set)
    {
        // TODO
    }
}

void SimpleObj::compute_KGs(const bool verbose)
{
    KGs.resize(v_num);
    KG_max = -1e10, KG_min = 1e10, KG_avg = 0;
    for (size_t i = 0; i < v_num; i++)
    {
        double total_angle = 0;
        for (auto face = one_ring_faces_set[i].begin(); face != one_ring_faces_set[i].end(); face++)
        {
            size_t v_pos = select_correct_idx_in_face(i, *face);
            assert(angles[*face][v_pos] >= 0 && angles[*face][v_pos] <= 1.1 * PI);
            total_angle += angles[*face][v_pos];
        }
        // KGs[i] = 2*PI - total_angle >= 0 ? (2*PI - total_angle) / vertex_voronoi_areas[i] : 0;
        KGs[i] = (2 * PI - total_angle) / vertex_voronoi_areas[i];
        set_min_max(KGs[i], KG_min, KG_max);
        set_avg(KGs[i], v_num, KG_avg);
    }
    if (verbose)
        cout << "[INFO] compute_KGs()" << endl;
}

void SimpleObj::compute_Kmaxs_Kmins(const bool verbose)
{
    K_maxs.resize(v_num);
    K_mins.resize(v_num);
    K_max_max = K_min_max = -1e10, K_max_min = K_min_min = 1e10, K_max_avg = K_min_avg = 0;
    for (size_t i = 0; i < v_num; i++)
    {
        double delta = sqrt(KHs[i] * KHs[i] - KGs[i]);
        delta        = delta >= 0 ? delta : 0;
        K_maxs[i]    = KHs[i] + delta;
        K_mins[i]    = KHs[i] - delta;
        set_min_max(K_maxs[i], K_max_min, K_max_max);
        set_min_max(K_mins[i], K_min_min, K_min_max);
        set_avg(K_maxs[i], v_num, K_max_avg);
        set_avg(K_mins[i], v_num, K_min_avg);
    }
    if (verbose)
        cout << "[INFO] compute_Kmaxs_Kmins()" << endl;
}

void SimpleObj::compute_all_curvatures(const bool verbose)
{
    if (!has_built_one_ring_faces_set)
    {
        build_one_ring_faces_set(verbose);
    }
    if (!has_built_one_ring_vertices_dict)
    {
        build_one_ring_vertices_dict(verbose);
    }

    compute_face_areas(verbose);
    compute_angles(verbose);
    compute_face_voronoi_areas(verbose);
    compute_vertex_voronoi_areas(verbose);

    compute_normals_KHs(verbose);
    compute_KGs(verbose);
    compute_Kmaxs_Kmins(verbose);

    if (verbose)
        cout << "[INFO] compute_all_curvatures()" << endl;
}

// Convex & Concave

void SimpleObj::naive_convex(const bool verbose)
{
    if (!has_built_one_ring_vertices_dict)
        build_one_ring_vertices_dict();

    convex_max = -1e10, convex_min = 1e10,
    convex.resize(v_num);
    for (size_t i = 0; i < v_num; i++)
    {
        double val = 0;
        // set<size_t> two_ring_neighbors = find_two_ring_vertices(i);
        // for (auto neighbor : two_ring_neighbors) {
        for (auto one_ring : one_ring_vertices_dict[i])
        {
            size_t   neighbor = one_ring.first;
            Vector3d vec      = v_mat.col(neighbor) - v_mat.col(i);
            vec               = vec * sum_cot_angle(i, neighbor);
            val += vec.dot(vn_mat.col(i));
        }
        // val /= two_ring_neighbors.size();
        val /= one_ring_vertices_dict[i].size();
        convex[i] = val;
        set_min_max(convex[i], convex_min, convex_max);
    }

    if (verbose)
        cout << "[INFO] naive_convex()" << endl;
}

// Smooth

void SimpleObj::compute_smooth_weight(const double threshold)
{
    smooth_weights.resize(v_num);
    for (size_t i = 0; i < v_num; i++)
    {
        double k_max = fabs(K_maxs[i]);
        double k_min = fabs(K_mins[i]);
        double kh    = fabs(KHs[i]);
        if (k_max <= threshold && k_min <= threshold)
        {
            smooth_weights[i] = 1.0;
        }
        else if (k_max > threshold && k_min > threshold && K_maxs[i] * K_mins[i] > 0)
        {
            smooth_weights[i] = 0.0;
        }
        else if (k_max == min(k_max, min(k_min, kh)))
        {
            smooth_weights[i] = k_max / kh;
        }
        else if (k_min == min(k_max, min(k_min, kh)))
        {
            smooth_weights[i] = k_min / kh;
        }
        else if (kh == min(k_max, min(k_min, kh)))
        {
            smooth_weights[i] = 1.0;
        }
    }
}

void SimpleObj::smooth(const bool anisotropic, const double step, const double smooth_weight, const double threshold, const bool verbose)
{
    reset_v_min_max();
    if (anisotropic)
    {
        compute_smooth_weight(threshold);
    }
    for (size_t i = 0; i < v_num; i++)
    {
        if (anisotropic)
        {
            v_mat.col(i) = v_mat.col(i) + smooth_weights[i] * step * KHs[i] * normals.col(i);
        }
        else
        {
            v_mat.col(i) = v_mat.col(i) + smooth_weight * step * KHs[i] * normals.col(i);
        }
        set_v_min_max(v_mat(0, i), v_mat(1, i), v_mat(2, i));
    }
    if (verbose)
        cout << "[INFO] smooth()" << endl;
}

void SimpleObj::continuously_smooth(const int n_iter, const bool anisotropic, const double step, const double smooth_weight, const double threshold, const bool verbose)
{

    for (int i = 0; i < n_iter; i++)
    {
        if (verbose)
            cout << "[INFO] continuously_smooth() iter: " << i << endl;
        compute_all_curvatures(verbose);
        smooth(anisotropic, step, smooth_weight, threshold, verbose);
    }
    if (verbose)
        cout << "[INFO] continuously_smooth()" << endl;
}

// Output

double SimpleObj::normalize_attr(const double val, double min, double max) const
{
    double res = (val - min) / (max - min);
    res        = (res > 1.0 ? 1.0 : (res < 0.0 ? 0.0 : res));
    return res;
}
double SimpleObj::normalize_attr(const double val, double min, double max, const double avg, const double rate, const string type) const
{
    if (type == "most_min")
    {
        max = min + (avg - min) * (rate + 1);
    }
    if (type == "most_max")
    {
        min = max - (max - avg) * (rate + 1);
    }
    return normalize_attr(val, min, max);
}

void SimpleObj::write_attrs_to_vtk(const string& save_file, const bool verbose)
{
    ofstream output(save_file);
    if (!output.is_open())
    {
        cerr << "[ERR] write_attrs_to_vtk(): File open error." << endl;
        exit(0);
    }

    output << "# vtk DataFile Version 2.0" << endl;
    output << "OBJECT" << endl;
    output << "ASCII" << endl
           << endl;
    output << "DATASET UNSTRUCTURED_GRID" << endl;
    output << "POINTS " << v_num << " double" << endl;
    for (int i = 0; i < v_num; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            output << v_mat(j, i) << " ";
        }
        output << endl;
    }
    output << "CELLS " << f_num << " " << f_num * (3 + 1) << endl;
    for (int i = 0; i < f_num; i++)
    {
        output << 3;
        for (int j = 0; j < 3; j++)
        {
            output << " " << f_mat(j, i);
        }
        output << endl;
    }
    output << "CELL_TYPES " << f_num << endl;
    for (int i = 0; i < f_num; i++)
    {
        output << "5" << endl;
    }
    output << "POINT_DATA " << v_num << endl;
    output << "SCALARS KG double 1" << endl;
    output << "LOOKUP_TABLE data_color" << endl;
    for (int i = 0; i < v_num; i++)
    {
        output << normalize_attr(KGs[i], KG_min, KG_max, KG_avg, 1.5, "most_min") << endl;
    }
    output << "SCALARS KH double 1" << endl;
    output << "LOOKUP_TABLE data_color" << endl;
    for (int i = 0; i < v_num; i++)
    {
        output << normalize_attr(KHs[i], KH_min, KH_max, KH_avg, 1.5, "most_min") << endl;
    }
    output << "SCALARS K_max double 1" << endl;
    output << "LOOKUP_TABLE data_color" << endl;
    for (int i = 0; i < v_num; i++)
    {
        output << normalize_attr(K_maxs[i], K_max_min, K_max_max, K_max_avg, 1.5, "most_min") << endl;
    }
    output << "SCALARS K_min double 1" << endl;
    output << "LOOKUP_TABLE data_color" << endl;
    for (int i = 0; i < v_num; i++)
    {
        output << normalize_attr(K_mins[i], K_min_min, K_min_max, K_min_avg, 1.5, "most_min") << endl;
    }
    output << "LOOKUP_TABLE data_color 101" << endl;
    for (int i = 0; i <= 100; i++)
    {
        output << 1.0 * i / 100 << " " << 1.0 - 1.0 * i / 100 << " 0.0 1.0" << endl;
    }

    output.close();
    if (verbose)
        cout << "[INFO] write_attrs_to_vtk()" << endl;
}

void SimpleObj::write_attrs_to_vtk(const string& save_file, const map<string, vector<double>>& features, const bool verbose)
{
    ofstream output(save_file);
    if (!output.is_open())
    {
        cerr << "[ERR] write_attrs_to_vtk(): File open error." << endl;
        exit(0);
    }

    output << "# vtk DataFile Version 2.0" << endl;
    output << "OBJECT" << endl;
    output << "ASCII" << endl
           << endl;
    output << "DATASET UNSTRUCTURED_GRID" << endl;
    output << "POINTS " << v_num << " double" << endl;
    for (int i = 0; i < v_num; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            output << v_mat(j, i) << " ";
        }
        output << endl;
    }
    output << "CELLS " << f_num << " " << f_num * (3 + 1) << endl;
    for (int i = 0; i < f_num; i++)
    {
        output << 3;
        for (int j = 0; j < 3; j++)
        {
            output << " " << f_mat(j, i);
        }
        output << endl;
    }
    output << "CELL_TYPES " << f_num << endl;
    for (int i = 0; i < f_num; i++)
    {
        output << "5" << endl;
    }
    output << "POINT_DATA " << v_num << endl;
    for (auto feature : features)
    {
        output << "SCALARS " << feature.first << " double 1" << endl;
        output << "LOOKUP_TABLE default" << endl;
        assert(feature.second.size() == v_num);
        for (auto value : feature.second)
        {
            output << value << endl;
        }
    }
    // for (auto vec_feature : vec_features) {
    // 	assert(vec_feature.second.size() == v_num);
    // 	assert(vec_feature.second[0].size() == 3);
    // 	output << "VECTORS " << vec_feature.first << " double" << endl;
    // 	for (auto value : vec_feature.second) {
    // 		output << value[0] << " " << value[1] << " " << value[2] << endl;
    // 	}
    // }
    // output << "LOOKUP_TABLE data_color 101" << endl;
    // for (int i = 0; i <= 100; i++) {
    // 	output << 1.0 * i / 100 << " " << 1.0 - 1.0 * i / 100 << " 0.0 1.0" << endl;
    // }

    output.close();
    if (verbose)
        cout << "[INFO] write_attrs_to_vtk()" << endl;
}
