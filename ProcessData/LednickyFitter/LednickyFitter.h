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
const std::complex<double> ImI (0.,1.);

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
  virtual ~LednickyFitter();

  double GetLednickyMomResCorrectedPoint(double aKStar, double* aPar, TH2* aMomResMatrix);
  double ApplyResidualCorrelationToPoint(double aKStar, double* aPar, TH2* aTransformMatrix);

  //void CalculateChi2(int &npar, double &chi2, double *par);
  void CalculateChi2PML(int &npar, double &chi2, double *par);
  void CalculateChi2PMLwMomResCorrection(int &npar, double &chi2, double *par);


  bool AreParamsSame(double *aCurrent, double *aNew, int aNEntries);
  void ApplyNonFlatBackgroundCorrection(vector<double> &aCf, vector<double> &aKStarBinCenters, TF1* aNonFlatBgd);
  vector<double> ApplyMomResCorrection(vector<double> &aCf, vector<double> &aKStarBinCenters, TH2* aMomResMatrix);
  vector<double> GetResidualCorrelation(vector<double> &aParentCf, vector<double> &aKStarBinCenters, TH2* aTransformMatrix);
  void CalculateChi2PMLwMomResCorrectionv2(int &npar, double &chi2, double *par);

  double GetChi2Value(int aKStarBin, TH1* aCfToFit, double* aPar);
  double GetPmlValue(double aNumContent, double aDenContent, double aCfContent);
  void CalculateFitFunction(int &npar, double &chi2, double *par);


  void CalculateChi2PMLwCorrectedCfs(int &npar, double &chi2, double *par);
  void DoFit();
  TF1* CreateFitFunction(TString aName, int aAnalysisNumber);

  vector<double> FindGoodInitialValues();


  //inline (i.e. simple) functions
  FitSharedAnalyses* GetFitSharedAnalyses();

  void SetApplyNonFlatBackgroundCorrection(bool aApply);
  void SetApplyMomResCorrection(bool aApplyMomResCorrection);
  void SetIncludeResidualCorrelations(bool aInclude);

  vector<double> GetMinParams();
  vector<double> GetParErrors();

  void SetVerbose(bool aSet);

  double GetChi2();

private:
  bool fVerbose;
  FitSharedAnalyses* fFitSharedAnalyses;
  TMinuit* fMinuit;
  int fNAnalyses;
  vector<TH1F*> fCfsToFit;
  vector<TF1*> fFits;

  double fMaxFitKStar;
  //vector<double> fMaxFitKStarVec;

  bool fRejectOmega;
  bool fApplyNonFlatBackgroundCorrection;
  bool fApplyMomResCorrection;
  bool fIncludeResidualCorrelations;

  double fChi2;
  double fChi2GlobalMin;

  double fEdm, fErrDef;
  int fNvpar, fNparx, fIcstat;

  vector<double> fChi2Vec;



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
inline void LednickyFitter::SetApplyMomResCorrection(bool aApplyMomResCorrection) {fApplyMomResCorrection = aApplyMomResCorrection;}
inline void LednickyFitter::SetIncludeResidualCorrelations(bool aInclude) {fIncludeResidualCorrelations = aInclude;}

inline vector<double> LednickyFitter::GetMinParams() {return fMinParams;}
inline vector<double> LednickyFitter::GetParErrors() {return fParErrors;}

inline void LednickyFitter::SetVerbose(bool aSet) {fVerbose=aSet;}

inline double LednickyFitter::GetChi2() {return fChi2;}

#endif
