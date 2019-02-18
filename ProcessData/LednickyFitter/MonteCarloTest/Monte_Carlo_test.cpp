//  file: Monte_Carlo_test.cpp
//
//  C++ Program to demonstrate use of Monte Carlo integration routines from
//   the gsl numerical library.
//
//  Programmer:  Dick Furnstahl  furnstahl.1@osu.edu
//
//  Revision history:
//      12/26/03  original C++ version, modified from C version
//
//  Notes:  
//   * Example taken from the GNU Scientific Library Reference Manual
//      Edition 1.1, for GSL Version 1.1 9 January 2002
//   * Compile and link with:
//       g++ -Wall -o Monte_Carlo_test Monte_Carlo_test.cpp -lgsl -lgslcblas
//   * gsl routines have built-in 
//       extern "C" {
//          <header stuff>
//       }
//      so they can be called from C++ programs without modification
//
//*********************************************************************//

// The following documention is from the GSL reference manual

//
// The example program below uses the Monte Carlo routines to estimate
// the value of the following 3-dimensional integral from the theory of
// random walks,
//
// I = \int_{-pi}^{+pi} {dk_x/(2 pi)} 
//     \int_{-pi}^{+pi} {dk_y/(2 pi)} 
//     \int_{-pi}^{+pi} {dk_z/(2 pi)} 
//      1 / (1 - cos(k_x)cos(k_y)cos(k_z))
//
// The analytic value of this integral can be shown to be I =
// \Gamma(1/4)^4/(4 \pi^3) = 1.393203929685676859.... The integral gives
// the mean time spent at the origin by a random walk on a body-centered
// cubic lattice in three dimensions.
//
// For simplicity we will compute the integral over the region (0,0,0)
// to (\pi,\pi,\pi) and multiply by 8 to obtain the full result. The
// integral is slowly varying in the middle of the region but has
// integrable singularities at the corners (0,0,0), (0,\pi,\pi),
// (\pi,0,\pi) and (\pi,\pi,0). The Monte Carlo routines only select
// points which are strictly within the integration region and so no
// special measures are needed to avoid these singularities.

//
// With 500,000 function calls the plain Monte Carlo algorithm achieves
// a fractional error of 0.6%. The estimated error sigma is consistent
// with the actual error, and the computed result differs from the true
// result by about one standard deviation,
//
// plain ==================
// result =  1.385867
// sigma  =  0.007938
// exact  =  1.393204
// error  = -0.007337 = 0.9 sigma
//
// The MISER algorithm reduces the error by a factor of two, and also
// correctly estimates the error,
//
// miser ==================
// result =  1.390656
// sigma  =  0.003743
// exact  =  1.393204
// error  = -0.002548 = 0.7 sigma
//
// In the case of the VEGAS algorithm the program uses an initial
// warm-up run of 10,000 function calls to prepare, or "warm up", the
// grid. This is followed by a main run with five iterations of 100,000
// function calls. The chi-squared per degree of freedom for the five
// iterations are checked for consistency with 1, and the run is
// repeated if the results have not converged. In this case the
// estimates are consistent on the first pass.
//
// vegas warm-up ==================
// result =  1.386925
// sigma  =  0.002651
// exact  =  1.393204
// error  = -0.006278 = 2 sigma
// converging...
// result =  1.392957 sigma =  0.000452 chisq/dof = 1.1
// vegas final ==================
// result =  1.392957
// sigma  =  0.000452
// exact  =  1.393204
// error  = -0.000247 = 0.5 sigma
//
// If the value of chisq had differed significantly from 1 it would
// indicate inconsistent results, with a correspondingly underestimated
// error. The final estimate from VEGAS (using a similar number of
// function calls) is significantly more accurate than the other two
// algorithms.
//
//*********************************************************************//

// include files
#include <iostream>
#include <iomanip>
#include <fstream>
#include <complex>
#include <math.h>
using namespace std;

#include "Faddeeva.hh"

#include <gsl/gsl_math.h>
#include <gsl/gsl_monte.h>
#include <gsl/gsl_monte_plain.h>
#include <gsl/gsl_monte_miser.h>
#include <gsl/gsl_monte_vegas.h>

// Computation of the integral,
//
//    I = int (dx dy dz)/(2pi)^3  1/(1-cos(x)cos(y)cos(z))
//
// over (-pi,-pi,-pi) to (+pi, +pi, +pi).  The exact answer
// is Gamma(1/4)^4/(4 pi^3).  This example is taken from
// C.Itzykson, J.M.Drouffe, "Statistical Field Theory -
// Volume 1", Section 1.1, p21, which cites the original
// paper M.L.Glasser, I.J.Zucker, Proc.Natl.Acad.Sci.USA 74
// 1800 (1977) */
//
// For simplicity we compute the integral over the region 
// (0,0,0) -> (pi,pi,pi) and multiply by 8 */
//

// Function prototypes

double g (double *k, size_t dim, void *params);

void display_results (char *title, double result, double error, double exact);

//*********************************************************************//
//________________________________________________________________________
double GetLednickyF1(double z)
{
  double result = (1./z)*Faddeeva::Dawson(z);
  return result;
}

//________________________________________________________________________
double GetLednickyF2(double z)
{
  double result = (1./z)*(1.-exp(-z*z));
  return result;
}

//________________________________________________________________________
double LednickyEq(double aKStar, double aRadius, double aRef0, double aImf0, double ad0)
{
  // par[0] = Lambda 
  // par[1] = Radius
  // par[2] = Ref0
  // par[3] = Imf0
  // par[4] = d0
  // par[5] = Norm

  double hbarc = 0.197327;
  complex<double> tImI = complex<double>(0., 1.);

  std::complex<double> f0 (aRef0, aImf0);
  double z = 2.*(aKStar/hbarc)*aRadius;  //z = 2k*R, to be fed to GetLednickyF1(2)

  std::complex<double> ScattAmp = pow( (1./f0) + 0.5*ad0*(aKStar/hbarc)*(aKStar/hbarc) - tImI*(aKStar/hbarc),-1);

  double C_FSI = 0.5*norm(ScattAmp)/(aRadius*aRadius)*(1.-1./(2*sqrt(M_PI))*(ad0/aRadius)) + 2.*real(ScattAmp)/(aRadius*sqrt(M_PI))*GetLednickyF1(z) - (imag(ScattAmp)/aRadius)*GetLednickyF2(z);

  double Cf = 1. + C_FSI;


  return Cf;
}




int
main (void)
{
  double exact;
  double result, error;		// result and error

  int tNdim = 5;
  double xl[tNdim] = { 0, 0, 0, 0, 0};
  double xu[tNdim] = { 1000, M_PI, 2*M_PI, M_PI, 2*M_PI};

  double tKStar = 0.1;
  double tRadius = 5.0;
  double tRef0 = -0.5;
  double tImf0 = 0.5;
  double td0 = 0.0;
  double tMuOut = 0.;
  double tParams[6] = {tKStar, tRadius, tRef0, tImf0, td0, tMuOut};

  exact = LednickyEq(tKStar, tRadius, tRef0, tImf0, td0);
/*
  int tNdim = 3;
  double xl[tNdim] = { 0, 0, 0};
  double xu[tNdim] = { 100, M_PI, 2*M_PI};
*/
  const gsl_rng_type *T;
  gsl_rng *r;

  gsl_monte_function G = { &g, tNdim, tParams };

  size_t calls = 500000;

  gsl_rng_env_setup ();

  T = gsl_rng_default;
  r = gsl_rng_alloc (T);

  {
    gsl_monte_plain_state *s = gsl_monte_plain_alloc (tNdim);
    gsl_monte_plain_integrate (&G, xl, xu, tNdim, calls, r, s, &result, &error);
    gsl_monte_plain_free (s);

    display_results ("plain", result, error, exact);
  }

  {
    gsl_monte_miser_state *s = gsl_monte_miser_alloc (tNdim);
    gsl_monte_miser_integrate (&G, xl, xu, tNdim, calls, r, s, &result, &error);
    gsl_monte_miser_free (s);

    display_results ("miser", result, error, exact);
  }

  {
    gsl_monte_vegas_state *s = gsl_monte_vegas_alloc (tNdim);

    gsl_monte_vegas_integrate (&G, xl, xu, tNdim, 10000, r, s, &result, &error);
    display_results ("vegas warm-up", result, error, exact);

    cout << "converging... " << endl;

    do
      {
	gsl_monte_vegas_integrate (&G, xl, xu, tNdim, calls / 5, r, s,
				   &result, &error);
	cout
	  << "result = " << setprecision (6) << result
	  << " sigma = " << setprecision (6) << error
	  << " chisq/dof = " << setprecision (1) << s->chisq << endl;
      }
    while (fabs (s->chisq - 1.0) > 0.5);

    display_results ("vegas final", result, error, exact);

    gsl_monte_vegas_free (s);
  }

  return 0;
}

//*********************************************************************//
/*
double
g (double *k, size_t dim, void *params)
{
  double A = 1.0 / (M_PI * M_PI * M_PI);	// A = 1/\pi^3

  return A / (1.0 - cos (k[0]) * cos (k[1]) * cos (k[2]));
}
*/


double
g (double *k, size_t dim, void *params)
{
  double *tParams = (double*)params;

  double hbarc = 0.197327;

  double tR0 = tParams[1];

  double tkStar = tParams[0];
  tkStar /= hbarc;

  double tkO = tkStar*sin(k[3])*cos(k[4]);
  double tkS = tkStar*sin(k[3])*sin(k[4]);
  double tkL = tkStar*cos(k[3]);

  double tA = 1.0/(pow((4*M_PI), 1.5)*tR0*tR0*tR0);
  double tAPrime = tA/(4*M_PI);  //because integrating over all kstar angles!

  double trO = k[0]*sin(k[1])*cos(k[2]);
  double trS = k[0]*sin(k[1])*sin(k[2]);
  double trL = k[0]*cos(k[1]);
  double tr = k[0];

  complex<double> tImI = complex<double>(0., 1.);
  double tkDotr = tkO*trO + tkS*trS + tkL*trL;

  std::complex<double> f0 (tParams[2],tParams[3]);
  double d0=tParams[4];
  std::complex<double> ScattAmp = pow( (1./f0) + 0.5*d0*tkStar*tkStar - tImI*tkStar,-1);

  std::complex<double> tWf = exp(-tImI*tkDotr)+(ScattAmp/tr)*exp(tImI*tkStar*tr);
  double tWfSq = norm(tWf);

  double tRealF = k[0]*k[0]*sin(k[1])*sin(k[3])*tAPrime*exp(-pow((trO-tParams[5]), 2)/(4*tR0*tR0))*exp(-trS*trS/(4*tR0*tR0))*exp(-trL*trL/(4*tR0*tR0))*tWfSq;

  return tRealF;
}


/*
double
g (double *k, size_t dim, void *params)
{
  double *tParams = (double*)params;

  double hbarc = 0.197327;

  double tR0 = tParams[1];

  double tkStar = tParams[0];
  tkStar /= hbarc;

  double tkO = tkStar;
  double tkS = 0;
  double tkL = 0;

  double tA = 1.0/(pow((4*M_PI), 1.5)*tR0*tR0*tR0);

  double trO = k[0]*sin(k[1])*cos(k[2]);
  double trS = k[0]*sin(k[1])*sin(k[2]);
  double trL = k[0]*cos(k[1]);
  double tr = k[0];

  complex<double> tImI = complex<double>(0., 1.);
  double tkDotr = tkO*trO + tkS*trS + tkL*trL;

  std::complex<double> f0 (tParams[2],tParams[3]);
  double d0=tParams[4];
  std::complex<double> ScattAmp = pow( (1./f0) + 0.5*d0*tkStar*tkStar - tImI*tkStar,-1);

  std::complex<double> tWf = exp(-tImI*tkDotr)+(ScattAmp/tr)*exp(tImI*tkStar*tr);
  double tWfSq = norm(tWf);

//  double tRealF = k[0]*k[0]*sin(k[1])*tA*exp(-tr*tr/(4*tR0*tR0))*tWfSq;
  double tRealF = k[0]*k[0]*sin(k[1])*tA*exp(-pow((trO-tParams[5]), 2)/(4*tR0*tR0))*exp(-trS*trS/(4*tR0*tR0))*exp(-trL*trL/(4*tR0*tR0))*tWfSq;

  return tRealF;
}
*/


/*
double
g (double *k, size_t dim, void *params)
{
  double tR0 = 5.0;
  double tA = 1.0/(pow((4*M_PI), 1.5)*tR0*tR0*tR0);

  double tr = k[0];
  double tRealF = k[0]*k[0]*sin(k[1])*tA*exp(-tr*tr/(4*tR0*tR0));

  return tRealF;
}
*/
//*********************************************************************//

void
display_results (char *title, double result, double error, double exact)
{
  cout.setf (ios::fixed, ios::floatfield);	// output in fixed format
  cout.precision (6);		// 6 digits past the decimal point

  cout << title << " ==================" << endl;
  cout << "result = " << setw (9) << result << endl;
  cout << "sigma  = " << setw (9) << error << endl;
  cout << "exact  = " << setw (9) << exact << endl;
  cout << "error  = " << setw (9) << result - exact
    << " = " << setprecision (1) << setw (2)
    << fabs (result - exact) / error << " sigma " << endl << endl;
}

