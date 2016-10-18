///////////////////////////////////////////////////////////////////////////
// PlotPartners:                                                         //
//                                                                       //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

#ifndef PLOTPARTNERS_H
#define PLOTPARTNERS_H

#include <iostream>
#include <iomanip>
#include <math.h>
#include <vector>
#include <assert.h>

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
#include "TList.h"

#include "TApplication.h"

using std::cout;
using std::endl;
using std::vector;

#include "PartialAnalysis.h"
class PartialAnalysis;

#include "Analysis.h"
class Analysis;

#include "DataAndModel.h"
class DataAndModel;

#include "MultGraph.h"
class MultGraph;

class PlotPartners {

public:
  PlotPartners(TString aFileLocationBase, AnalysisType aAnalysisType, CentralityType aCentralityType, int aNPartialAnalysis=5, bool aIsTrainResults=false);
  PlotPartners(TString aFileLocationBase, TString aFileLocationBaseMC, AnalysisType aAnalysisType, CentralityType aCentralityType, int aNPartialAnalysis=5, bool aIsTrainResults=false);
  virtual ~PlotPartners();

  void SetupAxis(TAxis* aAxis, double aMin, double aMax, TString aTitle, 
                 float aTitleSize=0.05, float aTitleOffset=0.9, bool aCenterTitle=false, float aLabelSize=0.03, float aLabelOffset=0.005, int aNdivisions=510);
  void SetupAxis(TAxis* aAxis, TString aTitle, 
                 float aTitleSize=0.05, float aTitleOffset=0.9, bool aCenterTitle=false, float aLabelSize=0.03, float aLabelOffset=0.005, int aNdivisions=510);

//  virtual TCanvas* DrawPurity();
//  virtual TCanvas* DrawKStarCfs();
//  virtual TCanvas* DrawAvgSepCfs();
//  virtual TCanvas* DrawKStarTrueVsRec();


protected:
  bool fContainsMC;

  Analysis* fAnalysis1;     //eg LamKchP,  LamK0,  XiKchP
  Analysis* fConjAnalysis1; //eg ALamKchM, ALamK0, AXiKchM

  Analysis* fAnalysis2;     //eg LamKchM,  XiKchM
  Analysis* fConjAnalysis2; //eg ALamKchP, AXiKchP

  //-----MC results
  Analysis* fAnalysisMC1;     //eg LamKchP,  LamK0,  XiKchP
  Analysis* fConjAnalysisMC1; //eg ALamKchM, ALamK0, AXiKchM

  Analysis* fAnalysisMC2;     //eg LamKchM,  XiKchM
  Analysis* fConjAnalysisMC2; //eg ALamKchP, AXiKchP

#ifdef __ROOT__
  ClassDef(PlotPartners, 1)
#endif
};


#endif
