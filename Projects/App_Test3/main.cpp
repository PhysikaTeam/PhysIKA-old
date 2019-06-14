#include "Physika_Surface_Fuild/Surface_GUI/GLApp.h"
int main(){
  GLApp window(100,100,0.5,1200,900);
  window.set_constants();
  window.init("input/surface.txt","input/height.txt","input/vx.txt","input/vy.txt","input/vz.txt");
  window.showframe(10);
  return 0;
}
