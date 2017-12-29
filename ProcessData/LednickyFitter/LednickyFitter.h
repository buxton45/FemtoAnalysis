///////////////////////////////////////////////////////////////////////////
// LednickyFitter:                                                       //
///////////////////////////////////////////////////////////////////////////

#ifndef LEDNICKYFITTER_H
#define LEDNICKYFITTER_H

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

#include "TSystem.h"
#include "TMinuit.h"
#include "TVirtualFitter.h"

//const double hbarc = 0.197327;
//const std::complex<double> ImI (0.,1.);

#include <omp.h>

#include "ChronoTimer.h"

using std::cout;
using std::endl;
using std::vector;

#include "FitSharedAnalyses.h"
class FitSharedAnalyses;

class LednickyFitter {

public:
  //Any enum types

  //Constructor, destructor, copy constructor, assignment operator
  LednickyFitter(FitSharedAnalyses* aFitSharedAnalyses, double aMaxFitKStar = 0.3);
  LednickyFitter(AnalysisType aAnalysisType, double aMaxFitKStar = 0.3);  //Currently, just used for CoulombFitter::CoulombFitter(AnalysisType aAnalysisType, double aMaxFitKStar)
  virtual ~LednickyFitter();

  static double GetLednickyF1(double z);
  static double GetLednickyF2(double z);
  static double LednickyEq(double *x, double *par);


  static void PrintCurrentParamValues(int aNpar, double* aPar);
  static double GetChi2Value(int aKStarBin, TH1* aCfToFit, double* aPar);
  static double GetChi2Value(int aKStarBin, TH1* aCfToFit, double aFitCfContent);
  static double GetPmlValue(double aNumContent, double aDenContent, double aCfContent);

  static void ApplyNonFlatBackgroundCorrection(vector<double> &aCf, vector<double> &aKStarBinCenters, TF1* aNonFlatBgd);
  static vector<double> ApplyMomResCorrection(vector<double> &aCf, vector<double> &aKStarBinCenters, TH2* aMomResMatrix);
  static void ApplyNormalization(double aNorm, td1dVec &aCf);

  vector<double> GetFitCfIncludingResiduals(FitPairAnalysis* aFitPairAnalysis, vector<double> &aPrimaryFitCfContent, double *aParamSet);

  void CalculateFitFunction(int &npar, double &chi2, double *par);
  void CalculateFitFunctionOnce(int &npar, double &chi2, double *par, double *parErr, double aChi2, int aNDF);
  TF1* CreateFitFunction(TString aName, int aAnalysisNumber);
  TF1* CreateFitFunction(int aAnalysisNumber, double *par, double *parErr, double aChi2, int aNDF);  //special case, used with PlotAllFitsCentral.C

  void InitializeFitter();  //Called within DoFit
  void DoFit();
  void Finalize();  //Send things back to analyses, etc.


  vector<double> FindGoodInitialValues();

  //inline (i.e. simple) functions
  FitSharedAnalyses* GetFitSharedAnalyses();

  void SetApplyNonFlatBackgroundCorrection(bool aApply);
  void SetNonFlatBgdFitType(NonFlatBgdFitType aNonFlatBgdFitType);
  void SetApplyMomResCorrection(bool aApplyMomResCorrection);
  virtual void SetIncludeResidualCorrelationsType(IncludeResidualsType aIncludeResidualsType);
  void SetChargedResidualsType(ChargedResidualsType aChargedResidualsType);
  void SetResPrimMaxDecayType(ResPrimMaxDecayType aResPrimMaxDecayType);

  vector<double> GetMinParams();
  vector<double> GetParErrors();

  void SetVerbose(bool aSet);

  double GetChi2();
  int GetNDF();

  void SetUsemTScalingOfResidualRadii(bool aUse=true, double aPower=-0.5);

protected:
  bool fVerbose;
  FitSharedAnalyses* fFitSharedAnalyses;
  TMinuit* fMinuit;
  int fNAnalyses;
  td3dVec fCorrectedFitVecs;

  double fMaxFitKStar;
  int fNbinsXToBuild;
  int fNbinsXToFit;
  double fKStarBinWidth;
  td1dVec fKStarBinCenters;
  //vector<double> fMaxFitKStarVec;

  bool fRejectOmega;
  bool fApplyNonFlatBackgroundCorrection;
  bool fApplyMomResCorrection;

  IncludeResidualsType fIncludeResidualsType;
  ChargedResidualsType fChargedResidualsType;
  ResPrimMaxDecayType fResPrimMaxDecayType;

  bool fResidualsInitiated;
  bool fReturnPrimaryWithResidualsToAnalyses;
  NonFlatBgdFitType fNonFlatBgdFitType;

  bool fUsemTScalingOfResidualRadii;
  double fmTScalingPowerOfResidualRadii;

  double fChi2;
  double fChi2GlobalMin;
  vector<double> fChi2Vec;

  double fEdm, fErrDef;
  int fNvpar, fNparx, fIcstat;

  int fNpFits;
  vector<int> fNpFitsVec;

  int fNDF;
  int fErrFlg;

  vector<double> fMinParams;
  vector<double> fParErrors;


#ifdef __ROOT__
  ClassDef(LednickyFitter, 1)
#endif
};


//inline stuff
inline FitSharedAnalyses* LednickyFitter::GetFitSharedAnalyses() {return fFitSharedAnalyses;}

inline void LednickyFitter::SetApplyNonFlatBackgroundCorrection(bool aApply) {fApplyNonFlatBackgroundCorrection = aApply;}
inline void LednickyFitter::SetNonFlatBgdFitType(NonFlatBgdFitType aNonFlatBgdFitType) {fNonFlatBgdFitType = aNonFlatBgdFitType;}
inline void LednickyFitter::SetApplyMomResCorrection(bool aApplyMomResCorrection) {fApplyMomResCorrection = aApplyMomResCorrection;}
inline void LednickyFitter::SetIncludeResidualCorrelationsType(IncludeResidualsType aIncludeResidualsType) {fIncludeResidualsType = aIncludeResidualsType;}
inline void LednickyFitter::SetChargedResidualsType(ChargedResidualsType aChargedResidualsType) {fChargedResidualsType = aChargedResidualsType;}
inline void LednickyFitter::SetResPrimMaxDecayType(ResPrimMaxDecayType aResPrimMaxDecayType) {fResPrimMaxDecayType = aResPrimMaxDecayType;}

inline vector<double> LednickyFitter::GetMinParams() {return fMinParams;}
inline vector<double> LednickyFitter::GetParErrors() {return fParErrors;}

inline void LednickyFitter::SetVerbose(bool aSet) {fVerbose=aSet;}

inline double LednickyFitter::GetChi2() {return fChi2;}
inline int LednickyFitter::GetNDF() {return fNDF;}

inline void LednickyFitter::SetUsemTScalingOfResidualRadii(bool aUse, double aPower) {fUsemTScalingOfResidualRadii = aUse; fmTScalingPowerOfResidualRadii = aPower;}

#endif
