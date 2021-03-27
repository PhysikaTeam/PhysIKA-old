//system
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string>
#include <atlstr.h> 
#include <fstream>
#include <iostream>
#include <vector>
using namespace std;
//GL
//#include "gl/glut.h"

////opencv
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#define MAXVAL 999999999
#define F_ZERO (1.0 / 999999999)
#define M_PI 3.14159265358979323846
#define CLOUD_GROUND_SEG_THRESHOLD 5
#define RADIUS 1
#define EARTH_RADIUS ((6378+6356)/2.0)

#define GEO_INTERVAL 15

#define NFRAME 1
#define K 5
