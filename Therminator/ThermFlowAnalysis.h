/* ThermFlowAnalysis.h */
//Dirty AF flow analysis, just checking for v2 and v3

#ifndef THERMFLOWANALYSIS_H
#define THERMFLOWANALYSIS_H

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <iomanip>
#include <cassert>
#include <complex>

#include "TGraph.h"
#include "TGraphErrors.h"
#include "TCanvas.h"

#include "ThermEvent.h"
class ThermEvent;

#include "PIDMapping.h"

using namespace std;

class ThermFlowAnalysis {

public:
  ThermFlowAnalysis(int aNpTBins=60, double apTBinSize=0.1, int aPID=0);
  virtual ~ThermFlowAnalysis();

  void BuildVnEPIngredients(ThermEvent &aEvent, double aHarmonic);
  void BuildVnEPIngredients(ThermEvent &aEvent);
  void BuildVnGraphs();
  TObjArray* GetVnGraphs();
  TCanvas* DrawFlowHarmonics();
  void SaveGraphs(TFile* aFile);
  void Finalize();

private:
  int fPID;  //fPID=0 means unidentified analysis
  int fNpTBins;
  double fpTBinSize;

  double fEtaA, fEtaB, fEtaOI;  //Typically, fEtaOI (of interest) set to 0.8, to investigate |eta| < 0.8 region
                                //Usually, fEtaA = 2.8 and fEtaB = -2.8 to give an eta gap of 2.0, however, my
                                //THERMINATOR simulation only goes to |eta|=2.0 to save disk space, so I set fEtaA=1. and fEtaB=-1.

  int fNEvTot;
  vector<int> fNEv_pTBins;

  double fEns_Res_v2, fEns_Res_v2_Sq, fVarEns_Res_v2;
  vector<double> fEns_v2, fEns_v2_Sq, fVarEns_v2;

  double fEns_Res_v3, fEns_Res_v3_Sq, fVarEns_Res_v3;
  vector<double> fEns_v3, fEns_v3_Sq, fVarEns_v3;

  vector<double> fEns_pT, fEns_pT_Sq, fVarEns_pT;

  TGraphErrors *fGraphV2, *fGraphV3;

#ifdef __ROOT__
  ClassDef(ThermFlowAnalysis, 1)
#endif
};



#endif















