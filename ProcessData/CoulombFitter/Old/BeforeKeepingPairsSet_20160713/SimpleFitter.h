///////////////////////////////////////////////////////////////////////////
// SimpleFitter:                                                        //
///////////////////////////////////////////////////////////////////////////

#ifndef SIMPLEFITTER_H
#define SIMPLEFITTER_H

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
#include "TH3.h"
#include "THn.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TApplication.h"
#include "TPaveText.h"
#include "TStyle.h"
#include "TAxis.h"
#include "TGraph.h"
#include "TMath.h"
#include "TVector3.h"
#include "TNtuple.h"
#include "TRandom.h"
#include "TObjectTable.h"

#include "TSystem.h"
#include "TMinuit.h"
#include "TVirtualFitter.h"

#include <omp.h>

#include "ChronoTimer.h"

using std::cout;
using std::endl;
using std::vector;


class SimpleFitter {

public:
  SimpleFitter(TString aName, double aMean, double aSigma, int aNBins, double aMin, double aMax);
  virtual ~SimpleFitter();

  double GetFitCfContentNumerically(double aMin, double aMax, double *par);
  TH1D* GetEntireFitCfContentNumerically(double *par);
  void CalculateChi2Numerically(int &npar, double &chi2, double *par);
  void CalculateChi2(int &npar, double &chi2, double *par);
  void DoFit();


  //inline
  TMinuit* GetMinuitObject();

private:
  TMinuit* fMinuit;
  TF1* fIdealGaussian;
  TH1D* fIdealGaussianHistogram;
  double fIdealMean, fIdealSigma;
  int fNBinsIdeal;
  double fIdealMin, fIdealMax;

  double fChi2;

  double fEdm, fErrDef;
  int fNvpar, fNparx, fIcstat;

  int fErrFlg;
  int fNpFits;
  int fNDF;


#ifdef __ROOT__
  ClassDef(SimpleFitter, 1)
#endif
};

inline TMinuit* SimpleFitter::GetMinuitObject() {return fMinuit;}

#endif
