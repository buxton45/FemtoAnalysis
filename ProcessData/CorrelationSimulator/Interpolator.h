/* Interpolator.h */

#ifndef INTERPOLATOR_H
#define INTERPOLATOR_H

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <iomanip>
#include <complex>
#include <math.h>
#include <vector>
#include <ctime>
#include <random>
#include <chrono>
#include <algorithm>
#include <limits>
#include <cassert>

#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "THn.h"

#include <omp.h>

#include <omp.h>

using std::cout;
using std::endl;
using std::vector;


class Interpolator {

public:
  //Constructor, destructor, copy constructor, assignment operator
  Interpolator();
  virtual ~Interpolator();

  static int GetBinNumber(double aBinSize, int aNbins, double aValue);
  static int GetBinNumber(int aNbins, double aMin, double aMax, double aValue);
  static int GetBinNumber(double aBinWidth, double aMin, double aMax, double aValue);

  //Note:  Linear, Bilinear, and Trilinear will essentially be copies of TH1::, TH2::, and TH3::Interpolate
  //       Rewriting these allows me more control, and allows me to find where and why error flags are thrown
  static double LinearInterpolate(TH1* a1dHisto, double aX);
  static double BilinearInterpolate(TH2* a2dHisto, double aX, double aY);
  static double BilinearInterpolateVector(vector<vector<double> > &a2dVec, double aX, int aNbinsX, double aMinX, double aMaxX, double aY, int aNbinsY, double aMinY, double aMaxY);
  static double TrilinearInterpolate(TH3* a3dHisto, double aX, double aY, double aZ);
  static double QuadrilinearInterpolate(THn* a4dHisto, double aT, double aX, double aY, double aZ);

protected:


#ifdef __ROOT__
  ClassDef(Interpolator, 1)
#endif
};



#endif
