///////////////////////////////////////////////////////////////////////////
// Purity:                                                               //
//                                                                       //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

#ifndef PURITY_H
#define PURITY_H

#include <iostream>
#include <iomanip>
#include <math.h>
#include <vector>
#include <assert.h>

#include "TF1.h"
#include "TH1.h"
#include "TH2.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TApplication.h"
#include "TPaveText.h"
#include "TStyle.h"
#include "TAxis.h"
#include "TGraph.h"
#include "TMath.h"
#include "TVectorD.h"
#include "TLegend.h"

using std::cout;
using std::endl;
using std::vector;

#include "Types.h"


class Purity {

public:

  Purity(TString aCombinedPurityName, ParticleType aParticleType, vector<TH1*> aPurityHistos);

  virtual ~Purity();

  void CombineHistos();  //simple addition of histograms
                         //different from CombineCfs(= weighted average)

  void CalculatePurity();
  void DrawPurity(TPad* aPad, bool aZoomBg=false);
  void DrawPurityAndBgd(TPad* aPad);


  void AddHisto(TH1* aHisto);

  void SetBgFitLow(double aMinLow, double aMaxLow);
  void SetBgFitHigh(double aMinHigh, double aMaxHigh);
  void SetROI(double aMinROI, double aMaxROI);


  //inline


private:

  TString fCombinedPurityName;
  vector<TH1*> fPurityHistos;
  ParticleType fParticleType;

  vector<double> fBgFitLow;
  vector<double> fBgFitHigh;
  vector<double> fROI;

  TH1* fCombinedPurity;
  TObjArray* fPurityFitInfo;

  bool fOutputPurityFitInfo;
  double fPurityValue;
  TF1* fFitBgd;










#ifdef __ROOT__
  ClassDef(Purity, 1)
#endif
};



#endif

