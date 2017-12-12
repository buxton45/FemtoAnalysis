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
  BackgroundFitter(TH1* aNum, TH1* aDen, TH1* aCf, NonFlatBgdFitType aBgdFitType, FitType aFitType=kChi2PML, 
                   double aMinBgdFit=0.60, double aMaxBgdFit=0.90, double aKStarMinNorm=0.32, double aKStarMaxNorm=0.40);
  virtual ~BackgroundFitter();

  static double FitFunctionLinear(double *x, double *par);
  static double FitFunctionQuadratic(double *x, double *par);
  static double FitFunctionGaussian(double *x, double *par);

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
  double fMinBgdFit, fMaxBgdFit;
  double fKStarMinNorm, fKStarMaxNorm;

  TMinuit* fMinuit;



#ifdef __ROOT__
  ClassDef(BackgroundFitter, 1)
#endif
};


//inline stuff
inline TMinuit* BackgroundFitter::GetMinuitObject() {return fMinuit;}

#endif


