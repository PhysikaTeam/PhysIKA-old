#include "para.h"
#include "input_output.h"
#include "simulator.h"
using namespace std;

template<typename T>
simulator<T>::~simulator()
{
  delete test_mass_spring_obj;
}

template<typename T>
simulator<T>::simulator()
{
  // give the simulator the hardcoded para for simple test 
  out_dir=para::out_dir_simulator;
  simulation_type=para::simulation_type;
  dt=para::dt;
  gravity=para::gravity;
  density=para::density;
  stiffness=para::stiffness;
  newton_fastMS=para::newton_fastMS;
  line_search=para::line_search;
  weight_line_search=para::weight_line_search;
  force_function=para::force_function;

  string input_object=para::input_object;
  string input_constraint=para::input_constraint;
  if(simulation_type=="static")
  {
    //make sure that it is a static simulator 
    dt=100;
    frame=1;
  }
  else if(simulation_type=="dynamic")
  {
    // do not modify the para from bash
  }
  io<T> myio=io<T>();
  test_mass_spring_obj=new mass_spring_obj<T,3>(input_object,dt,density,line_search,weight_line_search,stiffness,newton_fastMS); 
  myio.getConstraintFromCsv(test_mass_spring_obj,input_constraint);

  {
    if(force_function=="gravity")
    {
      omp_set_num_threads(para::num_threads);  
#pragma omp parallel for 
      for(size_t i=0;i<test_mass_spring_obj->num_vertex;++i)
      {
        test_mass_spring_obj->myvertexs[i].force_external=myvector<T,3>(0,0,-1*gravity)*test_mass_spring_obj->myvertexs[i].mass;
        //   printf("force_external given : %lf %lf %lf gravity: %lf mass: %lf\n",test_mass_spring_obj->myvertexs[i].force_external(0),test_mass_spring_obj->myvertexs[i].force_external(1),test_mass_spring_obj->myvertexs[i].force_external(2),gravity,test_mass_spring_obj->myvertexs[i].mass);
      }	 
    }
  }
  test_mass_spring_obj->checkFixedOrFree();
  
  // simulate(para_tree);
}

template<typename T>
int simulator<T>::simulate(int i)
{
  printf("**************************frame %d*************************\n", i);
  io<T> myio=io<T>();
  T energy;
  string object_name=para::object_name;
  test_mass_spring_obj->dynamicSimulator();
  energy=test_mass_spring_obj->calElasticEnergy();
  printf("energy is :%lf \n ",energy);
  
  
  stringstream ss;string frame_str;ss<<i; ss>>frame_str;
  stringstream ee; string gravity_str;ee<<gravity; ee>>gravity_str;
  if(simulation_type=="dynamic")
  {
    string path_name=out_dir+"/"+object_name+"_"+force_function+"_"+simulation_type+"_"+gravity_str+"_"+frame_str+".vtk";
    myio.saveAsVTK(test_mass_spring_obj,path_name);
    // path_name=out_dir+"/"+object_name+"_"+force_function+"_"+simulation_type+"_"+gravity_str+"_"+frame_str+".csv";
    // myio.saveTimeNormPair(test_mass_spring_obj,path_name);
  }
  else if(simulation_type=="static")
  {
    string path_name=out_dir+"/"+object_name+"_"+force_function+"_"+simulation_type+"_"+frame_str+"_"+gravity_str+".vtk";
    myio.saveAsVTK(test_mass_spring_obj,path_name);
    // path_name=out_dir+"/"+object_name+"_"+force_function+"_"+simulation_type+"_"+frame_str+"_"+gravity_str+".csv";
    // myio.saveTimeNormPair(test_mass_spring_obj,path_name);
  }

  
  
  return 0;
}
