// file NumIntLednickyCf.h


#ifndef NUMINTLEDNICKYCF_H
#define NUMINTLEDNICKYCF_H

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

#include "TVector3.h"
#include "TNtuple.h"
#include "TRandom.h"
#include "TFile.h"
#include "TList.h"

#include "ChronoTimer.h"

#include <omp.h>

#include <gsl/gsl_math.h>
#include <gsl/gsl_monte.h>
#include <gsl/gsl_monte_plain.h>
#include <gsl/gsl_monte_miser.h>
#include <gsl/gsl_monte_vegas.h>

using namespace std;

#include "Types.h"

class NumIntLednickyCf {

public:
  //Constructor, destructor, copy constructor, assignment operator
  NumIntLednickyCf(int aIntegrationType=2, int aNCalls=500000, double aMaxIntRadius=100);
  virtual ~NumIntLednickyCf();

  static double FunctionToIntegrate(double *k, size_t dim, void *params);

  complex<double> GetStrongOnlyWaveFunction(const TVector3 &aKStar3Vec, const TVector3 &aRStar3Vec, const double aRef0, const double aImf0, const double ad0);
  double GetStrongOnlyWaveFunctionSq(const TVector3 &aKStar3Vec, const TVector3 &aRStar3Vec, const double aRef0, const double aImf0, const double ad0); 

  double GetFitCfContent(double aKStar, double *par);

protected:
  int fIntegrationType;  //0=plain, 1=miser, 2=vegas
  int fNCalls;
  double fMaxIntRadius;


#ifdef __ROOT__
  ClassDef(NumIntLednickyCf, 1)
#endif
};



#endif
