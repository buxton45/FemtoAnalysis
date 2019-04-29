/* BackgroundFitter.h */

#ifndef BACKGROUNDFITTER_H
#define BACKGROUNDFITTER_H

//includes and any constant variable declarations
#include <iostream>
#include <iomanip>
#include <math.h>
#include <vector>
#include <algorithm>  //std::sort

#include "TString.h"
#include "TMinuit.h"
#include "TF1.h"
#include "TH1.h"
#include "TMath.h"

using std::cout;
using std::cin;
using std::endl;
using std::vector;

#include "Types.h"

class BackgroundFitter {

public:

  //Constructor, destructor, copy constructor, assignment operator
  BackgroundFitter(TH1* aNum, TH1* aDen, TH1* aCf, NonFlatBgdFitType aBgdFitType, FitType aFitType=kChi2PML, bool aNormalizeFitToCf=false, 
                   double aMinBgdFit=0.60, double aMaxBgdFit=0.90, double aMaxBgdBuild=2., double aKStarMinNorm=0.32, double aKStarMaxNorm=0.40);
  virtual ~BackgroundFitter();

  void PrintFitFunctionInfo();
  //--------------------------------------------------------------------
  static double FitFunctionLinear(double *x, double *par);
  static double FitFunctionQuadratic(double *x, double *par);
  static double FitFunctionGaussian(double *x, double *par);
  static double FitFunctionPolynomial(double *x, double *par);

  //These are used when plotting and fFitType==kChi2PML
  static double NormalizedFitFunctionLinear(double *x, double *par);
  static double NormalizedFitFunctionQuadratic(double *x, double *par);
  static double NormalizedFitFunctionGaussian(double *x, double *par);
  static double NormalizedFitFunctionPolynomial(double *x, double *par);
  static double NormalizedFitFunctionPolynomialwithOffset(double *x, double *par);
  //--------------------------------------------------------------------
  //These are used in FitPairAnalysis
  static double AddTwoFitFunctionsLinear(double *x, double *par);
  static double AddTwoFitFunctionsQuadratic(double *x, double *par);
  static double AddTwoFitFunctionsGaussian(double *x, double *par);
  static double AddTwoFitFunctionsPolynomial(double *x, double *par);

  static double AddTwoNormalizedFitFunctionsLinear(double *x, double *par);
  static double AddTwoNormalizedFitFunctionsQuadratic(double *x, double *par);
  static double AddTwoNormalizedFitFunctionsGaussian(double *x, double *par);
  static double AddTwoNormalizedFitFunctionsPolynomial(double *x, double *par);
  static double AddTwoNormalizedFitFunctionsPolynomialwithOffset(double *x, double *par);
  //--------------------------------------------------------------------

  static double GetPmlValue(double aNumContent, double aDenContent, double aCfContent);
  void CalculateBgdFitFunction(int &npar, double &chi2, double *par);

  TF1* FitNonFlatBackgroundPML();
  TF1* FitNonFlatBackgroundSimple();
  TF1* FitNonFlatBackground();

  //inline 
  TMinuit* GetMinuitObject();

private:

  TH1 *fNum, *fDen, *fCf;
  NonFlatBgdFitType fNonFlatBgdFitType;
  FitType fFitType;

  bool fNormalizeFitToCf;  //In the case of kChi2PML fit type, I want to treat this fit exactly as is done in LednickyFitter
                           //  i.e. fit the un-normalized numerator and denominator.  However, to plot the NonFlatBgd, it is
                           //  helpful/necessary to normalize up to the Cf.  This will be accomplished by simply multiplying 
                           //  the NonFlatBgd by DenScale/NumScale, just as is done for the Cf
  double fScale;

  double fMinBgdFit, fMaxBgdFit;
  double fMaxBgdBuild;
  double fKStarMinNorm, fKStarMaxNorm;

  TMinuit* fMinuit;



#ifdef __ROOT__
  ClassDef(BackgroundFitter, 1)
#endif
};


//inline stuff
inline TMinuit* BackgroundFitter::GetMinuitObject() {return fMinuit;}

#endif


