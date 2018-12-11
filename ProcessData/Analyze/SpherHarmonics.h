
#ifndef _SPHERHARMONICS_H_
#define _SPHERHARMONICS_H_

//Taken from Adam Kisiel's methods within CorrFit
//  More specifically, taken from sf.h and ylm.cc

#include <gsl/gsl_sf.h>
#include <cstdlib>
#include <cmath>
#include <complex>

using namespace std;

class SpherHarmonics {
 public:
  SpherHarmonics();
  ~SpherHarmonics();

  static complex<double> ceiphi(double phi);
  static double legendre(int ell,double ctheta);
  static complex<double> Ylm(int ell,int m,double theta,double phi);
  static complex <double> Ylm(int ell, int m, double x, double y, double z);

  static void YlmUpToL(int lmax, double x, double y, double z, complex<double> *ylms);
  static void YlmUpToL(int lmax, double ctheta, double phi, complex<double> *ylms);

  static double ReYlm(int ell, int m, double theta, double phi);
  static double ReYlm(int ell, int m, double x, double y, double z);
  static double ImYlm(int ell, int m, double theta, double phi);
  static double ImYlm(int ell, int m, double x, double y, double z);
};

#endif

