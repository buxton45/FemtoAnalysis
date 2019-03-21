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
#include "TGaxis.h"

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
  void DrawPurity(TPad* aPad, bool aZoomBg=false, bool aPrintPurity=false, double aPadScaleX=1., double aPadScaleY=1.);
  void DrawPurityAndBgd(TPad* aPad, bool aPrintPurity=true);


  void AddHisto(TH1* aHisto);

  void SetBgFitLow(double aMinLow, double aMaxLow);
  void SetBgFitHigh(double aMinHigh, double aMaxHigh);
  void SetROI(double aMinROI, double aMaxROI);


  //inline
  double GetPurity();
  double GetSignal();
  double GetSignalPlusBgd();
  double GetBgd();
  ParticleType GetParticleType();
  TH1* GetCombinedPurity();

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
  double fSignal;
  double fSignalPlusBgd;
  double fBgd;
  TF1* fFitBgd;


#ifdef __ROOT__
  ClassDef(Purity, 1)
#endif
};

inline double Purity::GetPurity() {return fPurityValue;}
inline double Purity::GetSignal() {return fSignal;}
inline double Purity::GetSignalPlusBgd() {return fSignalPlusBgd;}
inline double Purity::GetBgd() {return fBgd;}

inline ParticleType Purity::GetParticleType() {return fParticleType;}
inline TH1* Purity::GetCombinedPurity() {return fCombinedPurity;}

#endif

