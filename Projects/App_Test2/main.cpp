#include "Physika_Shallow_Water/Shallow_Water_GUI/GLApp.h"
int main(){
  GLApp window(0.01,0.1,1200,900);
  window.init("input/surface.txt","input/height.txt","input/velocity_x.txt","input/velocity_y.txt");
  window.showframe();
  return 0;
}
