#include "Physika_Shallow_Water/Shallow_water_model/ShallowWaterSolver.h"
#include "Physika_Shallow_Water/Shallow_Water_Render/render.h"
#include "Physika_Shallow_Water/Shallow_Water_IO/input_txt_io.h"
#include "Physika_Shallow_Water/Shallow_Water_meshs/Field.h"
#include <fstream>
#include <cassert>
#include <vector>
#include <string>
namesapce Physika{
class GLApp{
private:
    size_t x_cells;
    size_t y_cells;
    double dt;
    double dx;
    ShallowWaterSolver solver;
    std::vector<double> initial_height;
	  std::vector<double> initial_velocity_x;
	  std::vector<double> initial_velocity_y;
	  std::vector<double> initial_surface_level;
    VisualEngine visual_engine;
 public:
    GLApp(double time,double deltx);
    ~GLApp();
    void init(std::string const surface,std::string const height,std::string const vx,std::string const vy);
    void run();
}
}
    
