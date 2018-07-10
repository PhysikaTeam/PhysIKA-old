// comment from Bart: taken from Pieter Peers website, thank you Pieter
// raw PPM support (read & write)
// RGBE (uncompressed PIC) support (read & write)
// PIC support (adapted from original code) (read & write)

#include "Image.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

unsigned int high(unsigned int val) { return((val >> 8) & 0xFF); }
unsigned int low(unsigned int val) { return(val & 0xFF); }

// operators
Image& Image::operator=(const Image& src)
{
	if (this != &src)
	{
		unload();

		if(src.map == NULL) return *this;

		// copy complete map
		unsigned long amount = src.width * src.height * src.channels;
		map = new float[amount];

		// memcopy is limited in size
		for(unsigned long i=0; i < amount; i++)
			map[i] = src.map[i];

		// copy data
		channels = src.channels;
		width = src.width;
		height = src.height;
	}

	return *this;
}

Image& Image::operator=(const float* pixel)
{
	if(map && pixel)
	{
		// fill complete map
		unsigned long amount = width * height;
		unsigned long pos = 0;
		for(unsigned long i=0; i < amount; i++)
		for(unsigned int c=0; c < channels; c++)
		{
			map[pos] = pixel[c];
			pos++;
		}
	}

	return *this;
}

Image& Image::operator=(const float value)
{
	if(map)
	{
		// fill complete map
		unsigned long amount = width * height * channels;
		for(unsigned long i=0; i < amount; i++)
		map[i] = value;
	}

	return *this;
}

bool Image::operator==(const Image& src)
{
	if(width != src.width) return false;
	if(height != src.height) return false;
	if(channels != src.channels) return false;
	if(!map || !src.map) return false;

	unsigned long amount = width * height * channels;

	while(amount)
	{
		if(map[amount] != src.map[amount]) return false;
		amount--;
	}

	return true;
}

bool Image::operator!=(const Image& src)
{
	if(width != src.width) return true;
	if(height != src.height) return true;
	if(channels != src.channels) return true;
	if(!map || !src.map) return true;

	unsigned long amount = width * height * channels;

	while(amount)
	{
		if(map[amount] == src.map[amount]) return false;
		amount--;
	}

	return true;
}

Image& Image::operator+=(const Image& src)
{
	// map != NULL
	// src.map != NULL
	// size and channels are equal!
	if ((map) && (src.map) && (src.width == width) && (src.height == height) && (src.channels == channels))
	{
		unsigned long amount = src.width * src.height * src.channels;

		for(unsigned long i=0; i < amount; i++)
			map[i] += src.map[i];
	}

	return *this;
}

Image& Image::operator+=(const float *pixel)
{
	// map != NULL
	// pixel != NULL
	if (map && pixel)
	{
		unsigned long amount = width * height;
		unsigned long pos = 0;

		for(unsigned long i=0; i < amount; i++)
			for(unsigned int c=0; c < channels; c++)
			{
				map[pos] += pixel[c];
				pos++;
			}
	}

	return *this;
}

Image& Image::operator+=(const float value)
{
	// map != NULL
	if (map)
	{
		unsigned long amount = width * height * channels;

		for(unsigned long i=0; i < amount; i++)
			map[i] += value;
	}

	return *this;
}

Image& Image::operator-=(const Image& src)
{
	// map != NULL
	// src.map != NULL
	// size and channels are equal!
	if ((map) && (src.map) && (src.width == width) && (src.height == height) && (src.channels == channels))
	{
		unsigned long amount = src.width * src.height * src.channels;

		for(unsigned long i=0; i < amount; i++)
			map[i] -= src.map[i];
	}

	return *this;
}

Image& Image::operator-=(const float *pixel)
{
	// map != NULL
	// pixel != NULL
	if (map && pixel)
	{
		unsigned long amount = width * height;
		unsigned long pos = 0;

		for(unsigned long i=0; i < amount; i++)
			for(unsigned int c=0; c < channels; c++)
			{
				map[pos] -= pixel[c];
				pos++;
			}
	}

	return *this;
}

Image& Image::operator-=(const float value)
{
	// map != NULL
	if (map)
	{
		unsigned long amount = width * height * channels;

		for(unsigned long i=0; i < amount; i++)
			map[i] -= value;
	}

	return *this;
}

Image& Image::operator*=(const Image& src)
{
	// map != NULL
	// src.map != NULL
	// size and channels are equal!
	if ((map) && (src.map) && (src.width == width) && (src.height == height) && (src.channels == channels))
	{
		unsigned long amount = src.width * src.height * src.channels;

		for(unsigned long i=0; i < amount; i++)
			map[i] *= src.map[i];
	}

	return *this;
}

Image& Image::operator*=(const float *pixel)
{
	// map != NULL
	// pixel != NULL
	if (map && pixel)
	{
		unsigned long amount = width * height;
		unsigned long pos = 0;

		for(unsigned long i=0; i < amount; i++)
			for(unsigned int c=0; c < channels; c++)
			{
				map[pos] *= pixel[c];
				pos++;
			}
	}

	return *this;
}

Image& Image::operator*=(const float value)
{
  // map != NULL
  if (map)
  {
    unsigned long amount = width * height * channels;

    for(unsigned long i=0; i < amount; i++)
	map[i] *= value;
  }
  
  return *this;
}

Image& Image::operator/=(const Image& src)
{
  // map != NULL
  // src.map != NULL
  // size and channels are equal!
  if ((map) && (src.map) && (src.width == width) && (src.height == height) && (src.channels == channels))
  {
    unsigned long amount = src.width * src.height * src.channels;

    for(unsigned long i=0; i < amount; i++)
      map[i] /= src.map[i];
  }
  
  return *this;
}

Image& Image::operator/=(const float *pixel)
{
	// map != NULL
	// pixel != NULL
	if (map && pixel)
	{
		unsigned long amount = width * height;
		unsigned long pos = 0;

		for(unsigned long i=0; i < amount; i++)
			for(unsigned int c=0; c < channels; c++)
			{
				map[pos] /= pixel[c];
				pos++;
			}
	}

	return *this;
}

Image& Image::operator/=(const float value)
{
	// map != NULL
	if (map)
	{
		unsigned long amount = width * height * channels;

		for(unsigned long i=0; i < amount; i++)
			map[i] /= value;
	}

	return *this;
}

Image Image::operator+(const Image& src)
{
	// map != NULL
	// src.map != NULL
	// size and channels are equal!
	if ((src.width == width) && (src.height == height) && (src.channels == channels))
	{
		Image result(width, height, channels);

		if(map && src.map)
		{
			unsigned long amount = src.width * src.height * src.channels;

			for(unsigned long i=0; i < amount; i++)
				result = map[i] + src.map[i];
		}
		else if(map) result = *this;
		else result = src;

		return result;
	}
	else return Image();
}

Image Image::operator+(const float *pixel)
{
	Image result(width, height, channels);

	// map != NULL
	// pixel != NULL
	if (map && pixel)
	{
		unsigned long amount = width * height;
		unsigned long pos = 0;

		for(unsigned long i=0; i < amount; i++)
			for(unsigned int c=0; c < channels; c++)
			{
				result.map[pos] = map[pos] + pixel[c];
				pos++;
			}
	}
	else if(!map) result = pixel;
	else result = *this;

	return result;
}

Image Image::operator+(const float value)
{
	Image result(width, height, channels);

	// map != NULL
	if (map)
	{
		unsigned long amount = width * height * channels;

		for(unsigned long i=0; i < amount; i++)
			result.map[i] = map[i] + value;
	}
	else result = value;

	return result;
}

Image Image::operator-(const Image& src)
{
	// map != NULL
	// src.map != NULL
	// size and channels are equal!
	if ((src.width == width) && (src.height == height) && (src.channels == channels))
	{
		Image result(width, height, channels);

		if(map && src.map)
		{
			unsigned long amount = src.width * src.height * src.channels;

			for(unsigned long i=0; i < amount; i++)
				result.map[i] = map[i] - src.map[i];
		}
		else if(map) result = *this;
		else result -= src;

		return result;
	}
	else return Image();
}

Image Image::operator-(const float *pixel)
{
	Image result(width, height, channels);

	// map != NULL
	// pixel != NULL
	if (map && pixel)
	{
		unsigned long amount = width * height;
		unsigned long pos = 0;

		for(unsigned long i=0; i < amount; i++)
			for(unsigned int c=0; c < channels; c++)
			{
				result.map[pos] = map[pos] - pixel[c];
				pos++;
			}
	}
	else if(!map) result -= pixel;
	else result = *this;

	return result;
}

Image Image::operator-(const float value)
{
	Image result(width, height, channels);

	// map != NULL
	if (map)
	{
		unsigned long amount = width * height * channels;

		for(unsigned long i=0; i < amount; i++)
			result.map[i] = map[i] - value;
	}
	else result -= value;

	return result;
}

Image Image::operator*(const Image& src)
{
	// map != NULL
	// src.map != NULL
	// size and channels are equal!
	if ((src.width == width) && (src.height == height) && (src.channels == channels))
	{
		Image result(width, height, channels);

		if(map && src.map)
		{
			unsigned long amount = src.width * src.height * src.channels;

			for(unsigned long i=0; i < amount; i++)
				result.map[i] = map[i] * src.map[i];
		}

		return result;
	}
	else return Image();
}

Image Image::operator*(const float *pixel)
{
	Image result(width, height, channels);

	// map != NULL
	// pixel != NULL
	if (map && pixel)
	{
		unsigned long amount = width * height;
		unsigned long pos = 0;

		for(unsigned long i=0; i < amount; i++)
			for(unsigned int c=0; c < channels; c++)
			{
				result.map[pos] = map[pos] * pixel[c];
				pos++;
			}
	}

	return result;
}

Image Image::operator*(const float value)
{
	Image result(width, height, channels);

	// map != NULL
	if (map)
	{
		unsigned long amount = width * height * channels;

		for(unsigned long i=0; i < amount; i++)
			result.map[i] = map[i] * value;
	}

	return result;
}

Image Image::operator/(const Image& src)
{
	// map != NULL
	// src.map != NULL
	// size and channels are equal!
	if ((src.width == width) && (src.height == height) && (src.channels == channels))
	{
		Image result(width, height, channels);

		if(map && src.map)
		{
			unsigned long amount = src.width * src.height * src.channels;

			for(unsigned long i=0; i < amount; i++)
				result.map[i] = map[i] / src.map[i];
		}

		return result;
	}
	else return Image();
}

Image Image::operator/(const float *pixel)
{
	Image result(width, height, channels);

	// map != NULL
	// pixel != NULL
	if (map && pixel)
	{
		unsigned long amount = width * height;
		unsigned long pos = 0;

		for(unsigned long i=0; i < amount; i++)
			for(unsigned int c=0; c < channels; c++)
			{
				result.map[pos] = map[pos] / pixel[c];
				pos++;
			}
	}

	return result;
}

Image Image::operator/(const float value)
{
	Image result(width, height, channels);

	// map != NULL
	if (map)
	{
		unsigned long amount = width * height * channels;

		for(unsigned long i=0; i < amount; i++)
			result.map[i] = map[i] / value;
	}

	return result;
}

// take abs of each pixel of this image
void Image::abs(void)
{
	if(map)
	{
		unsigned long amount = width * height * channels;
		for(unsigned long i=0; i < amount; i++)
			map[i] = (float)fabs(map[i]);
	}
}

// return image which has each pixel abs of this image
Image Image::iabs(void)
{
	if(map)
	{
		Image result(width, height, channels);
		unsigned long amount = width * height * channels;
		for(unsigned long i=0; i < amount; i++)
			result.map[i] = (float)fabs(map[i]); 
			return result;
	}
	else return *this;
}

// truncate each pixel value of this inage between bottom and top value
void Image::trunc(float bottomvalue, float topvalue)
{
	if(map)
	{
		unsigned long amount = width * height * channels;
		for(unsigned long i=0; i < amount; i++)
			if(map[i] < bottomvalue) map[i] = bottomvalue;
			else if(map[i] > topvalue) map[i] = topvalue;
			//else map[i] = map[i];
	} 
}

// return truncated Image in which  each pixel value of this inage between bottom and top value
Image Image::itrunc(float bottomvalue, float topvalue)
{
	if(map)
	{
		Image result(width, height, channels);
		unsigned long amount = width * height * channels;
		for(unsigned long i=0; i < amount; i++)
			if(map[i] < bottomvalue) result.map[i] = bottomvalue;
			else if(map[i] > topvalue) result.map[i] = topvalue;
			else result.map[i] = map[i];

		return result;
	} 
	else return *this;
}

// return 1 channel image which is a specific channel from this image
Image Image::Channel(unsigned int chan)
{
	if((map || channels > 1) && (chan <= channels))
	{
		Image result(width, height, 1);
		unsigned long amount = width * height;
		unsigned long pos = chan-1;
		for(unsigned long i = 0; i < amount; i++)
		{
			result.map[i] = map[pos];
			pos += channels;
		}

		return result; 
	}
	else return *this;
}

// Error Function
void Image::Error(char *where, char *msg)
{
  fprintf(stderr, "Error(%s) : %s.\n", where, msg);
  exit(0);
}

// convert image from summed area table to normal
void Image::UnSummedAreaTable(void)
{
	if(!map) return;
	unsigned long pos = 0;

	fprintf(stderr, "Converting Image from Summed Area Table...");

	float *map2 = new float[width*height*channels];

	for(unsigned int y=0; y < height; y++)
		for(unsigned int x=0; x < width; x++)
			for(unsigned int c=0; c < channels; c++)
			{
				if((x != 0) && (y != 0))
					map2[pos] = map[pos] + map[pos - (channels*(width+1))] - map[pos - channels] - map[pos - (width*channels)];

				else if((x != 0) && (y == 0))
					map2[pos] = map[pos] - map[pos - channels];

				else if((x == 0) && (y != 0))
					map2[pos] = map[pos] - map[pos - (channels*width)];

				else map2[pos] = map[pos];

				pos++;
			}

	delete[] map;
	map = map2;
	fprintf(stderr, "Done.\n");
}

// convert image to summed area table
void Image::SummedAreaTable(void)
{
	if(!map) return;
	unsigned long pos = 0;

	fprintf(stderr, "Converting Image to Summed Area Table...");

	for(unsigned int y=0; y < height; y++)
	{
		for(unsigned int x=0; x < width; x++)
		{
			for(unsigned int c=0; c < channels; c++)
			{
				// if not first column and not first line
				if((x != 0) && (y != 0))
					// sum = pixel + sum_block_above + sum_block_before - sum_block_above_before
					map[pos] = map[pos] + map[pos - (channels*width)] + map[pos - channels] - map[pos - (channels*(width+1))];

				// if first line (but not first 0,0)
				else if((y == 0) && (x != 0))
					// sum = pixel + sum_block_before
					map[pos] = map[pos] + map[pos - channels];

				// first column (but not first 0,0)
				else if((x == 0) && (y != 0))
					// sum = pixel + sum_column_above
					map[pos] = map[pos] + map[pos - (channels*width)];

				// next pos
				pos++;
			}
		}
	}

	fprintf(stderr, "Done.\n");
}

// read past comments in ppm and pic files
// a comment is a line beginning with a # and ending with a newline (\n)
void Image::skipComment(FILE *file)
{
	char b = fgetc(file);

	while(b == '#')
	{
		while (b != '\n')
		b = fgetc(file);

		b = fgetc(file);
	}
	ungetc(b, file);
}

// read a RAW (P6) ppm file.
void Image::loadPPM(FILE *file)
{
	unsigned int  colorsize;

	// read header (P6)
	fscanf(file, "P6\n");

	// skip comment (#)
	skipComment(file);

	// read dimensions
	char dummy;
	unsigned int r = fscanf(file, "%ld %ld\n%d%c", &width, &height, &colorsize, &dummy);
	if (r != 4) Error(NULL, "not a PPM (raw) file");

	// calc amount of memory needed and allocate it
	unsigned long amount = width * height * 3;
	map = new float[amount];

	// load image
	for (unsigned int j=0; j<height; j++) 
	{
		float *row = &(map[3*width*j]);
		for(unsigned int i = 0; i < 3*width; i++)
		row[i] = (float)(fgetc(file)) / colorsize;
	}

	// set nr of channels
	channels = 3;
}

// save to a RAW (P6) ppm file
void Image::savePPM(FILE *file)
{
	unsigned int colorsize = 255;

	// sanity check
	if(!map) Error(NULL, "can't write empty image");

	// write header
	fprintf(file, "P6\n%ld %ld\n%d\n", width, height, colorsize);

	// save image data
	// if less then 3 channels, add with 0-value chans
	// if more then 3 channe;s, drop excess channels
	for(unsigned int j=0; j<height; j++)
	{
		float *row  = &map[channels*width*j];
		for(unsigned int i=0; i < width; i++)
		{
			for(unsigned int c=0; c<3; c++)
			{
				if(c < channels) 
				{
					unsigned int color = (unsigned int)(row[i*channels + c] * colorsize);
					if(color > colorsize) color = colorsize;

					fputc(color, file);
				}
				else fputc(0, file);
			}
		}
	}

	// done!
}

#define FORMAT "FORMAT"
#define FORMAT_TYPE "32-bit_rle_rgbe"
#define EXPOSURE "EXPOSURE"
#define MINLEN   (unsigned long)(8)
#define MAXLEN   (unsigned long)(32767)
#define MINRUN   (unsigned long)(4)

// given a real pixels value and its exponent (4th value) return its RGB component
//
// ABCE => R = RealPixel2RGB(A, E)
//         G = RealPixel2RGB(B, E)
//         B = RealPixel2RGB(C, E)
float Image::RealPixel2RGB(unsigned int c, unsigned int e)
{
	if(e == 0) return 0;
	else {
		float v = (float)(ldexp(1./256, e - 128));
		return (float)((c + .5)*v);
	}
}

// given an RGB triplet create a real pixel color and store in Color
//
// RGB2RealPixel(R, G, B, ABCE)
//
void Image::RGB2RealPixel(float r, float g, float b, unsigned char Color[4])
{
	float max = r;
	if(g > max) max = g;
	if(b > max) max = b;

	if(max <= 1e-32) { Color[0] = Color[1] = Color[2] = Color[3] = 0; }
	else 
	{
		int exp;
		max = (float)(frexp(max, &exp) * 255.9999 / max);

		Color[0] = (unsigned char)(r * max);
		Color[1] = (unsigned char)(g * max);
		Color[2] = (unsigned char)(b * max);
		Color[3] = (unsigned char)(exp + 128);
	}
}

// read a scanline from a PIC file into row with width.
void Image::ReadPICscanline(FILE *file, float *row, unsigned long width)
{
	unsigned long pos = 0;
	unsigned int r, g, b, e, i;

	while(pos < (3*width))
	{
		// read 4 bytes
		r = fgetc(file);
		g = fgetc(file);
		b = fgetc(file);
		e = fgetc(file);

		// check compresion method
		if((r == 1) && (g == 1) && (b == 1))           // Old Run-Length encoding
		{
			unsigned long prev = pos - 3;
			unsigned long l = e, t = 8;
			bool done = false;

			// stupidity check
			if(pos < 3) Error(NULL, "PIC: Illegal Old RLE compression");

			// check l
			while(!done) 
			{
				if(t > 32) Error(NULL, "PIC: Old RLE overflow");

				r = fgetc(file);
				g = fgetc(file);
				b = fgetc(file);
				e = fgetc(file);

				if(e == EOF) done = true;
				else if((r == 1) && (g == 1) && (b ==1))
				{
					l += (e << t);
					t += 8;
				}
				else done = true;
			}

			if(e != EOF)
			{
				ungetc(e, file);
				ungetc(b, file);
				ungetc(g, file);
				ungetc(r, file);
			}

			// repeat
			for(i=0; i < l; i++)
			{
				row[pos++] = row[prev + 0];
				row[pos++] = row[prev + 1];
				row[pos++] = row[prev + 2];
			}
		}
		else if((r == 2) && (g == 2))           // Run-Length encoding
		{
			if((b << 8 | e) != width) Error(NULL, "PIC: RLE length mismatch");

			unsigned int *R = new unsigned int[width];
			unsigned int *G = new unsigned int[width];
			unsigned int *B = new unsigned int[width];
			unsigned int *E = new unsigned int[width];
			unsigned int *T = NULL;

			// repeat per channel
			for(i=0; i < 4; i++)
			{
				unsigned int p = 0;
				if(i==0) T = R;
				if(i==1) T = G;
				if(i==2) T = B;
				if(i==3) T = E;

				while(p < width)
				{
					e = fgetc(file);
					if(p >= width) Error(NULL, "PIC: RLE out of bounds");
					if(e == EOF) Error(NULL, "PIC: RLE unexpected end");
					if(e > 128)
					{
						e &= 127;
						r = fgetc(file);

						for(unsigned int j=0; j < e; j++) T[p++] = r;
					}
					else for(unsigned int j=0; j < e; j++) T[p++] = fgetc(file);
				}
			}

			// copy & convert
			for(i=0; i < width; i++)
			{	  
				r = R[i];
				g = G[i];
				b = B[i];
				e = E[i];

				row[pos++] = RealPixel2RGB(r, e);
				row[pos++] = RealPixel2RGB(g, e);
				row[pos++] = RealPixel2RGB(b, e);
			}

			// free mem
			if(R) delete[] R;
			if(G) delete[] G;
			if(B) delete[] B;
			if(E) delete[] E;
		}
		else                           // Uncompressed
		{
			row[pos++] = RealPixel2RGB(r, e);
			row[pos++] = RealPixel2RGB(g, e);
			row[pos++] = RealPixel2RGB(b, e);
		}
	}
}

// write a scanline from row with width to file
void Image::WritePICscanline(FILE *file, unsigned char *row, unsigned long width)
{
	// adapted from the incomprehensible radiance code

	// out of range => plain write
	if(width < MINLEN || width > MAXLEN) 
	{
		fwrite(row, 4, width, file);
		return;
	}

	// write compression header (2,2, high(width), low(width))  RLE COMPRESSION
	fputc(2, file);
	fputc(2, file);
	fputc(high(width), file);
	fputc(low(width), file);

	// threat every chan seperately
	for(unsigned int c = 0; c < 4; c++)
	{
		// find next run, that is longer then MINLEN
		unsigned long pos = 0;
		while(pos < width)
		{
			unsigned long begin = pos;
			unsigned long runLength = 0;
			while((begin < width) && (runLength < MINLEN))
			{
				begin += runLength;
				runLength = 1;
				while((runLength < 127) && (begin + runLength < width) && (row[begin*4 + c] == row[(begin + runLength) * 4 + c]))
				{
					runLength++;
				}
			}
      
			// short run check (long run is proceeded by short run)
			unsigned int start = begin - pos;
			if((start > 1) && (start < MINRUN))
			{
				unsigned int shortpos = pos + 1;
	
				// check if present
				while((row[shortpos*4 + c] == row[pos*4 + c]) && (shortpos < begin)) shortpos++;

				// write out if present
				if(shortpos == begin)
				{
					fputc(128+start, file);
					fputc(row[pos*4 + c], file);
					pos = begin;
				}
			}

			// write out non-run
			while(pos < begin)
			{
				unsigned int length = begin - pos;
				if(length > 128) length = 128;    // max non-run
				fputc(length, file);
				while(length--) 
				{
					fputc(row[pos*4 + c], file);
					pos++;
				}
			}

			// if long-run was found...write it out
			if(runLength >= MINLEN)
			{
				fputc(128+runLength, file);
				fputc(row[begin*4 + c], file);
				pos += runLength;
			}
		}
	}
}

// load a PIC file from a file
void Image::loadPIC(FILE *file)
{
	// Based on the original code for PIC readers.
	float exposure = 1.;
	bool done = false;
	char buf[80];

	// read header
	fscanf(file, "#?RADIANCE\n");

	while(!done)
	{
		skipComment(file);

		// read line
		fgets(buf, 80, file);

		// check format (only support rgbe format...NOT xyze)
		if(!strncmp(buf, FORMAT, strlen(FORMAT)))
		{
			char *string = strchr(buf, '=') + 1;
			if(strncmp(string, FORMAT_TYPE, strlen(FORMAT_TYPE))) Error(NULL, "Unsupported PIC format");
		}

		// read exposure
		else if(!strncmp(buf, EXPOSURE, strlen(EXPOSURE)))
		{
			char *value = strchr(buf, '=') + 1;
			exposure = (float)(1. / atof(value));
		}

		// end?
		else if(!strncmp(buf, "\n", strlen("\n"))) done = true;

		/* else skip line */
	}

	// get resolution (ignore orientation & sign)
	unsigned char sx, sy, X, Y;
	if( fscanf(file, "%c%c %ld %c%c %ld\n", &sy, &Y, &height, &sx, &X, &width) != 6)
	{
		fprintf(stderr, "PIC resolution error. (%c%c %ld %c%c %ld)", sy, Y, height, sx, X, width);
		Error(NULL, "PIC res error.");
	}

	channels = 3;

	// calc amount of memory needed and allocate it
	long amount = width * height * 3;
	map = new float[amount];

	// read image (in top to bottom order)
	for(unsigned int y = 0; y < height; y++) {
		ReadPICscanline(file, map + 3*width*y, width);
	}

	// multiply read values with exposure
	float *m = map;
	for (unsigned int i=0; i<width*height*3; i++, m++) {
		*m *= exposure;
	}
}


// save as PIC
void Image::savePIC(FILE *file)
{
	// save data in PIC format (RLE compressed)

	// sanity check
	if(!map) Error(NULL, "can't write empty image");

	// scanline
	unsigned char *Color  = new unsigned char[4 * width];

	// write header
	fprintf(file, "#?RADIANCE\n");
	fprintf(file, "%s=%s\n\n", FORMAT, FORMAT_TYPE);
	fprintf(file, "-Y %ld +X %ld\n", height, width);

	for(unsigned int y=0; y < height; y++)
	{
		// convert row into RGBE
		float *row = &(map[channels*width*y]);
		for(unsigned int x=0; x < width; x++)
		{
			float r = (channels > 0) ? row[x*channels + 0] : 0;
			float g = (channels > 1) ? row[x*channels + 1] : 0;
			float b = (channels > 2) ? row[x*channels + 2] : 0;
			RGB2RealPixel(r, g, b, &(Color[x*4]));
		}

		// compress and write out rgbe
		WritePICscanline(file, Color, width);
	}

	delete[] Color;
}

// load from RGBE file
void Image::loadRGBE(FILE *file)
{
	float exposure = 1.;
	bool done = false;
	char buf[80];

	// read header
	fscanf(file, "#?RGBE\n");

	while(!done)
	{
		skipComment(file);

		// read line
		fgets(buf, 80, file);

		// check format (only support rgbe format...NOT xyze)
		if(!strncmp(buf, FORMAT, strlen(FORMAT)))
		{
			char *string = strchr(buf, '=') + 1;
			if(strncmp(string, FORMAT_TYPE, strlen(FORMAT_TYPE))) Error(NULL, "Unsupported RGBE format");
		}

		// read exposure
		else if(!strncmp(buf, EXPOSURE, strlen(EXPOSURE)))
		{
			char *value = strchr(buf, '=') + 1;
			exposure = (float)(1. / atof(value));
		}

		// end?
		else if(!strncmp(buf, "\n", strlen("\n"))) done = true;

		/* else skip line */
	}

	// get resolution (ignore orientation & sign)
	unsigned char sx, sy, X, Y;
	if( fscanf(file, "%c%c %ld %c%c %ld\n", &sy, &Y, &height, &sx, &X, &width) != 6)
	{
		fprintf(stderr, "RGBE resolution error. (%c%c %ld %c%c %ld)", sy, Y, height, sx, X, width);
		Error(NULL, "RGBE res error.");
	}


	channels = 3;

	// calc amount of memory needed and allocate it
	unsigned long amount = width * height * 3;
	map = new float[amount];

	// read image (in top to bottom order)
	for(unsigned int y=0; y < height; y++) 
	{
		float *row = &(map[y*width*3]);
		for(unsigned int x=0; x < width; x++)
		{
			unsigned char r, g, b, e;
			fscanf(file, "%c%c%c%c", &r, &g, &b, &e);

			row[x*3 + 0] = RealPixel2RGB(r, e);
			row[x*3 + 1] = RealPixel2RGB(g, e);
			row[x*3 + 2] = RealPixel2RGB(b, e);
		}
	}

	// multiply read values with exposure
	float *m = map;
	for (unsigned int i=0; i<width*height*3; i++, m++) {
		*m *= exposure;
	}
}

// save to RGBE formated file
void Image::saveRGBE(FILE *file)
{
	// save data in RGBE format (uncompressed PIC)

	// sanity check
	if(!map) Error(NULL, "can't write empty image");

	// write header
	fprintf(file, "#?RGBE\n");
	fprintf(file, "%s=%s\n\n", FORMAT, FORMAT_TYPE);
	fprintf(file, "-Y %ld +X %ld\n", height, width);

	// write out uncompressed
	// drop chans above 3
	// fill chans under 3 if not exist
	for(unsigned int y=0; y < height; y++)
	{
		float *row = &(map[width*channels*y]);
		for(unsigned int x=0; x < width; x++)
		{
			// get first 3 channels
			float r = (channels > 0) ? row[x*channels + 0] : 0;
			float g = (channels > 1) ? row[x*channels + 1] : 0;
			float b = (channels > 2) ? row[x*channels + 2] : 0;

			// convert to realpixel format
			unsigned char Color[4];
			RGB2RealPixel(r, g, b, Color);

			// write out
			fwrite(Color, 4, 1, file);
		}
	}

}

// load image file
void Image::load(const char *filename)
{
        // UNLOAD
        unload();

	FILE *file = fopen(filename, "rb");

	if (!file) Error("Image::load()", "couldn't open texture image file");

	// determine file type
	const char *name = filename;
	if(strstr(name, ".ppm") || strstr(name, ".pnm"))
	{
		loadPPM(file);
		fprintf(stderr, "PPM ");
	}
	else if(strstr(name, ".pic"))
	{
		loadPIC(file);
		fprintf(stderr, "PIC ");
	}
	else if(strstr(name, ".rgbe"))
	{
		loadRGBE(file);
		fprintf(stderr, "RGBE");
	}
	else
	{
		fprintf(stderr, "Image::load()", "unknown image format for '%s'", name);
	}    

	// close file
	fclose(file);

	// Display nice message
	fprintf(stderr, "read: %s: %ld KBytes, Size=%ldx%ld, Channels=%ld\n", filename, (unsigned long)(width * height * 3 * sizeof(float) >> 10), width, height, channels);
}

// save image
void Image::save(const char *filename)
{
	// save buffer to image texture
	if (map == NULL) Error(NULL, "Image::save(): empty map");

	FILE* file = fopen(filename, "wb");

	if (!file) {
		Error("Image::save()", "couldn't create texture image file");
		return;
	}

	// determine file type
	const char *name = filename;
	if(strstr(name, ".ppm") || strstr(name, ".pnm"))
	{
		savePPM(file);
		fprintf(stderr, "PPM ");
	}
	else if(strstr(name, ".pic"))
	{
		savePIC(file);
		fprintf(stderr, "PIC ");
	}
	else if(strstr(name, ".rgbe"))
	{
		saveRGBE(file);
		fprintf(stderr, "RGBE ");
	}
	else
	{
		fprintf(stderr, "Image::save()", "unknown image format for '%s'", name);
	}    

	// close file
	fclose(file);

	// Display nice message
	fprintf(stderr, "write: %s: %ld KBytes, Size=%ldx%ld, Channels=%ld\n", filename, (unsigned long)(width * height * 3 * sizeof(float) >> 10), width, height, channels);
}


