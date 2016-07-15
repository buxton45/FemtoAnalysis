///////////////////////////////////////////////////////////////////////////
// Cf2dLite:                                                             //
//                                                                       //
//   This class holds a single Cf2d and the method to build it.          //
//   It will also contain methods to project out 1d histograms, and      //
//   will contain a collection of these histograms                       //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

#ifndef CF2DLITE_H
#define CF2DLITE_H

#include <iostream>
#include <iomanip>
#include <math.h>
#include <vector>
#include <assert.h>

#include "TF1.h"
#include "TH2.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TApplication.h"
#include "TPaveText.h"
#include "TStyle.h"
#include "TAxis.h"
#include "TGraph.h"
#include "TMath.h"

using std::cout;
using std::endl;
using std::vector;

#include "CfLite.h"
class CfLite;

#include "Types.h"

class Cf2dLite {

public:

  Cf2dLite(TString aDaughterCfsBaseName, TH2* aMotherNum2d, TH2* aMotherDen2d, AxisType aProjectionAxis, vector<vector<int> > &aProjectionBins, double aMinNorm, double aMaxNorm);
  virtual ~Cf2dLite();

  void DoProjection(int aRebinFactor=1);
  void Rebin(int aRebinFactor);
  CfLite* GetDaughterCf(int aDaughterCf);



  //inline
  TH2* GetMotherNum2d();
  TH2* GetMotherDen2d();

  int GetNDaughterCfs();


private:

  TH2 *fMotherNum2d, *fMotherDen2d;

  int fNbinsX, fNbinsY;
  TString fXName, fYName;

  
  AxisType fProjectionAxis;
  vector<vector<int> > fProjectionBins;
  double fMinNorm, fMaxNorm;
  int fNDaughterCfs;
  TString fDaughterCfsBaseName;
  vector<CfLite*> fDaughterCfs;





#ifdef __ROOT__
  ClassDef(Cf2dLite, 1)
#endif
};

inline TH2* Cf2dLite::GetMotherNum2d() {return fMotherNum2d;}
inline TH2* Cf2dLite::GetMotherDen2d() {return fMotherDen2d;}

inline int Cf2dLite::GetNDaughterCfs() {return fNDaughterCfs;}






#endif











