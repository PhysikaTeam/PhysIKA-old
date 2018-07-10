// comment from Bart: taken from Pieter Peers website, thank you Pieter
/* Image.H : reads and writes image files (PPM, RGBE, PIC) */

#ifndef _Image_H_
#define _Image_H_

#include <string.h>
#include <stdio.h>
#include <math.h>

class Image {

public:

  // data members
  float *map;
  unsigned long channels;
  unsigned long width, height;

  // constructors
  Image() {
    map = NULL;
    channels = 0;
    width = 0;
    height = 0;
  }
  Image(unsigned long x, unsigned long y, unsigned long chans=3)
  {
    map = new float[x*y*chans];
    channels = chans;
    width = x;
    height = y;

    for(unsigned int i=0; i < x*y*chans; i++) map[i] = 0.;
  }
  Image(const char *filename) { map = NULL; load(filename); }

  // destructor
  ~Image() { unload(); }

  // operators
  Image& operator=(Image const & src);
  Image& operator=(const float* pixel);
  Image& operator=(const float value);
  bool   operator==(const Image& src);
  bool   operator!=(const Image& src);
  Image& operator+=(const Image& src);
  Image& operator+=(const float* pixel);
  Image& operator+=(const float value);
  Image& operator-=(const Image& src);
  Image& operator-=(const float* pixel);
  Image& operator-=(const float value);
  Image& operator*=(const Image& src);
  Image& operator*=(const float* pixel);
  Image& operator*=(const float value);
  Image& operator/=(const Image& src);
  Image& operator/=(const float* pixel);
  Image& operator/=(const float value);
  Image  operator+(const Image& src);
  Image  operator+(const float* pixel);
  Image  operator+(const float value);
  Image  operator-(const Image& src);
  Image  operator-(const float* pixel);
  Image  operator-(const float value);
  Image  operator*(const Image& src);
  Image  operator*(const float* pixel);
  Image  operator*(const float value);
  Image  operator/(const Image& src);
  Image  operator/(const float* pixel);
  Image  operator/(const float value);

  void   abs();
  Image  iabs();
  void   trunc(float bottomvalue, float topvalue);
  Image  itrunc(float bottomvalue, float topvalue);

  Image  Channel(unsigned int chan);
  inline Image  R() { return Channel(0); }
  inline Image  G() { return Channel(1); }
  inline Image  B() { return Channel(2); }

  // methods
  void load(const char *filename);
  void save(const char *filename);
  void unload() 
  { 
    if (map) delete[] map; 
    map = NULL;
    channels = 0;
    width = 0;
    height = 0;
  }

  float* pixel(unsigned long col, unsigned long row) { return &map[(col + row*width)*channels]; }
  float pixel(unsigned long col, unsigned long row, unsigned long channel) { return (pixel(col, row))[channel]; }
  void setPixel(unsigned long col, unsigned long row, float *color) { memcpy(&map[(col + row*width)*channels], color, sizeof(float) * channels); }
  void setPixel(unsigned long col, unsigned long row, unsigned long channel, float color) { map[(col + row*width)*channels + channel] = color; }

  void SummedAreaTable(void);
  void UnSummedAreaTable(void);
  float summedPixel(unsigned long col1, unsigned long row1, unsigned long col2, unsigned long row2, unsigned long channel)
  {
    float result = pixel(col2, row2, channel);
    float area = (float)fabs((float)((col2 - col1 + 1) * (row2 - row1 + 1)));

    col1--;    row1--;
    if(col1 >= 0) result = result - pixel(col1, row2, channel);
    if(row1 >= 0) result = result - pixel(col2, row1, channel);
    if((col1 >= 0) && (row1 >= 0)) result = result + pixel(col1, row1, channel);

    return result / area;
  }

  void Error(char *where, char *msg);

private:
  void skipComment(FILE *file);
  void loadPPM(FILE *file);
  void savePPM(FILE *file);

  float RealPixel2RGB(unsigned int c, unsigned int e);
  void RGB2RealPixel(float r, float g, float b, unsigned char Color[4]);
  void ReadPICscanline(FILE *file, float *row, unsigned long width);
  void WritePICscanline(FILE *file, unsigned char *row, unsigned long width);

  void loadPIC(FILE *file);
  void savePIC(FILE *file);

  void loadRGBE(FILE *file);
  void saveRGBE(FILE *file); 
};

#endif /*_Image_H_*/
