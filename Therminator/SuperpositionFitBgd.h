/* SuperpositionFitBgd.h */

#ifndef SUPERPOSITIONFITBGD_H
#define SUPERPOSITIONFITBGD_H

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

class SuperpositionFitBgd {

public:

  //Constructor, destructor, copy constructor, assignment operator
  SuperpositionFitBgd(TH1D* aData, TH1D* aCf1, TH1D* aCf2, double aMinBgdFit=0.40, double aMaxBgdFit=2.0);
  virtual ~SuperpositionFitBgd();

  void CalculateBgdFitFunction(int &npar, double &chi2, double *par);
  void DoFit();

  TMinuit* GetMinuitObject();
  TH1D* GetSupCf();
  double GetN1();
  double GetN2();

private:
  TH1D *fData, *fCf1, *fCf2, *fSupCf12;
  double fMinBgdFit, fMaxBgdFit;
  int fMinBgdFitBin, fMaxBgdFitBin;
  double fN1, fN2;

  TMinuit* fMinuit;


#ifdef __ROOT__
  ClassDef(SuperpositionFitBgd, 1)
#endif
};

inline TMinuit* SuperpositionFitBgd::GetMinuitObject() {return fMinuit;}
inline TH1D* SuperpositionFitBgd::GetSupCf() {return fSupCf12;}
inline double SuperpositionFitBgd::GetN1() {return fN1;}
inline double SuperpositionFitBgd::GetN2() {return fN2;}
#endif


