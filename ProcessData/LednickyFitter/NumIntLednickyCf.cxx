// file NumIntLednickyCf.cxx

#include "NumIntLednickyCf.h"

#ifdef __ROOT__
ClassImp(NumIntLednickyCf)
#endif


//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________


//________________________________________________________________________________________________________________
NumIntLednickyCf::NumIntLednickyCf(int aIntegrationType, int aNCalls, double aMaxIntRadius) :
  fIntegrationType(aIntegrationType),
  fNCalls(aNCalls),
  fMaxIntRadius(aMaxIntRadius)
{

}


//________________________________________________________________________________________________________________
NumIntLednickyCf::~NumIntLednickyCf()
{}


//________________________________________________________________________________________________________________
double NumIntLednickyCf::FunctionToIntegrate(double *k, size_t dim, void *params)
{
  double *tParams = (double*)params;

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

  complex<double> ImI = complex<double>(0., 1.);
  double tkDotr = tkO*trO + tkS*trS + tkL*trL;

  std::complex<double> f0 (tParams[2],tParams[3]);
  double d0=tParams[4];
  std::complex<double> ScattAmp = pow( (1./f0) + 0.5*d0*tkStar*tkStar - ImI*tkStar,-1);

  std::complex<double> tWf = exp(-ImI*tkDotr)+(ScattAmp/tr)*exp(ImI*tkStar*tr);
  double tWfSq = norm(tWf);

//  double tRealF = k[0]*k[0]*sin(k[1])*tA*exp(-tr*tr/(4*tR0*tR0))*tWfSq;
  double tRealF = k[0]*k[0]*sin(k[1])*tAPrime*exp(-pow((trO-tParams[5]), 2)/(4*tR0*tR0))*exp(-trS*trS/(4*tR0*tR0))*exp(-trL*trL/(4*tR0*tR0))*tWfSq;

  return tRealF;

}



//________________________________________________________________________________________________________________
complex<double> NumIntLednickyCf::GetStrongOnlyWaveFunction(const TVector3 &aKStar3Vec, const TVector3 &aRStar3Vec, const double aRef0, const double aImf0, const double ad0)
{
  complex<double> tf0 (aRef0, aImf0);

  double tKdotR = aKStar3Vec.Dot(aRStar3Vec);
    tKdotR /= hbarc;
  double tKStarMag = aKStar3Vec.Mag();
    tKStarMag /= hbarc;
  double tRStarMag = aRStar3Vec.Mag();

  complex<double> tScattLenLastTerm (0., tKStarMag);
  complex<double> tScattAmp = pow((1./tf0) + 0.5*ad0*tKStarMag*tKStarMag - tScattLenLastTerm,-1);

  complex<double> tReturnWf = exp(-ImI*tKdotR) + tScattAmp*exp(ImI*tKStarMag*tRStarMag)/tRStarMag;
  return tReturnWf;
}

//________________________________________________________________________________________________________________
double NumIntLednickyCf::GetStrongOnlyWaveFunctionSq(const TVector3 &aKStar3Vec, const TVector3 &aRStar3Vec, const double aRef0, const double aImf0, const double ad0)
{
  complex<double> tWf = GetStrongOnlyWaveFunction(aKStar3Vec, aRStar3Vec, aRef0, aImf0, ad0);
  double tWfSq = norm(tWf);
  return tWfSq;
}


//________________________________________________________________________________________________________________
double NumIntLednickyCf::GetFitCfContent(double aKStar, double *par)
{
  // par[0] = Lambda 
  // par[1] = Radius
  // par[2] = Ref0
  // par[3] = Imf0
  // par[4] = d0
  // par[5] = norm
  // par[6] = muOut


  double tResult, tError;	// result and error
  double tReturnCfContent = 0.;

  double tParams[6] = {aKStar, par[1], par[2], par[3], par[4], par[6]};

  int tNdim = 5;
  double xl[tNdim] = { 0, 0, 0, 0, 0};
  double xu[tNdim] = { fMaxIntRadius, M_PI, 2*M_PI, M_PI, 2*M_PI};

  const gsl_rng_type *T;
  gsl_rng *r;

  gsl_monte_function G = { &FunctionToIntegrate, tNdim, tParams };

  size_t calls = fNCalls;

  gsl_rng_env_setup ();

  T = gsl_rng_default;
  r = gsl_rng_alloc (T);

  if(fIntegrationType==0)
  {
    gsl_monte_plain_state *s = gsl_monte_plain_alloc (tNdim);
    gsl_monte_plain_integrate (&G, xl, xu, tNdim, calls, r, s, &tResult, &tError);
    gsl_monte_plain_free (s);
  }

  else if(fIntegrationType==1)
  {
    gsl_monte_miser_state *s = gsl_monte_miser_alloc (tNdim);
    gsl_monte_miser_integrate (&G, xl, xu, tNdim, calls, r, s, &tResult, &tError);
    gsl_monte_miser_free (s);
  }

  else if(fIntegrationType==2)
  {
    gsl_monte_vegas_state *s = gsl_monte_vegas_alloc (tNdim);

    gsl_monte_vegas_integrate (&G, xl, xu, tNdim, 10000, r, s, &tResult, &tError);
//    cout << "converging... " << endl;

    do
      {
	gsl_monte_vegas_integrate (&G, xl, xu, tNdim, calls / 5, r, s,
				   &tResult, &tError);
/*
	cout
	  << "tResult = " << setprecision (6) << tResult
	  << " sigma = " << setprecision (6) << tError
	  << " chisq/dof = " << setprecision (1) << s->chisq << endl;
*/
      }
    while (fabs (s->chisq - 1.0) > 0.5);
    gsl_monte_vegas_free (s);
  }
  else assert(0);
  //--------------------------------------------------------

  tReturnCfContent = (par[0]*tResult + (1.-par[0]));  //C = (Lam*C_gen + (1-Lam));
  return tReturnCfContent;
}

