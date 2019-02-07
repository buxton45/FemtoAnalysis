///////////////////////////////////////////////////////////////////////////
// StrippedSimpleFitter:                                                 //
///////////////////////////////////////////////////////////////////////////

#ifndef STRIPPEDSIMPLEFITTER_H
#define STRIPPEDSIMPLEFITTER_H

//includes and any constant variable declarations
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

#include "Faddeeva.hh"

#include "TF1.h"
#include "TH1F.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TApplication.h"
#include "TPaveText.h"
#include "TStyle.h"
#include "TAxis.h"
#include "TGraph.h"
#include "TMath.h"
#include "TVector3.h"
#include "TLegend.h"

#include "TSystem.h"
#include "TMinuit.h"
#include "TVirtualFitter.h"


using std::cout;
using std::endl;
using std::vector;

typedef vector<double> td1dVec;
static const double hbarc = 0.197327;
static const std::complex<double> ImI (0.,1.);

class StrippedSimpleFitter {

public:
  enum FitType {kChi2PML=0, kChi2=1};
  StrippedSimpleFitter(TH1* aNum, TH1* aDen, double aMaxFitKStar = 0.3, double aMinNormKStar=0.32, double aMaxNormKStar=0.40);
  virtual ~StrippedSimpleFitter();

  static double GetLednickyF1(double z);
  static double GetLednickyF2(double z);
  static double LednickyEq(double *x, double *par);
  static double LednickyEqWithNorm(double *x, double *par);

  double GetNumScale();
  double GetDenScale();
  void BuildCf();

  static void PrintCurrentParamValues(int aNpar, double* aPar);
  static double GetChi2Value(int aKStarBin, TH1* aCfToFit, double aFitCfContent);
  static double GetPmlValue(double aNumContent, double aDenContent, double aCfContent);
  static void ApplyNormalization(double aNorm, td1dVec &aCf);

  void CreateMinuitParameters();
  void CalculateFitFunction(int &npar, double &chi2, double *par);
  TF1* CreateFitFunction(TString aName);

  void InitializeFitter();  //Called within DoFit
  void DoFit();
  void Finalize();

  TPaveText* CreateParamFinalValuesText(TF1* aFit, double aTextXmin, double aTextYmin, double aTextWidth, double aTextHeight);
  void DrawCfWithFit(TPad *aPad, TString aDrawOption="");

  void DrawCfNumDen(TPad *aPad, TString aDrawOption="");


  //inline
  vector<double> GetMinParams();
  vector<double> GetParErrors();

  void SetVerbose(bool aSet);

  double GetChi2();
  int GetNDF();
  TMinuit* GetMinuitObject();

  void SetFitType(FitType aFitType);

protected:
  TH1 *fNum, *fDen, *fCf;

  FitType fFitType;
  bool fVerbose;
  TMinuit* fMinuit;

  double fMaxFitKStar;
  double fMinNormKStar, fMaxNormKStar;
  int fNbinsXToBuild;
  int fNbinsXToFit;
  double fKStarBinWidth;
  td1dVec fKStarBinCenters;

  bool fRejectOmega;

  double fChi2;
  double fChi2GlobalMin;
  vector<double> fChi2Vec;

  double fEdm, fErrDef;
  int fNvpar, fNparx, fIcstat;

  int fNpFits;

  int fNDF;
  int fErrFlg;

  vector<double> fMinParams;
  vector<double> fParErrors;

#ifdef __ROOT__
  ClassDef(StrippedSimpleFitter, 1)
#endif
};


//inline stuff
inline vector<double> StrippedSimpleFitter::GetMinParams() {return fMinParams;}
inline vector<double> StrippedSimpleFitter::GetParErrors() {return fParErrors;}

inline void StrippedSimpleFitter::SetVerbose(bool aSet) {fVerbose=aSet;}

inline double StrippedSimpleFitter::GetChi2() {return fChi2;}
inline int StrippedSimpleFitter::GetNDF() {return fNDF;}

inline TMinuit* StrippedSimpleFitter::GetMinuitObject() {return fMinuit;}

inline void StrippedSimpleFitter::SetFitType(FitType aFitType) {fFitType = aFitType;}

#endif
