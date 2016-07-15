///////////////////////////////////////////////////////////////////////////
// DataAndModel:                                                         //
//                                                                       //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

#ifndef DATAANDMODEL_H
#define DATAANDMODEL_H

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
#include "TLegend.h"

using std::cout;
using std::cin;
using std::endl;
using std::vector;

#include "Analysis.h"
class Analysis;

#include "PartialAnalysis.h"
class PartialAnalysis;

#include "CfHeavy.h"
class CfHeavy;

#include "Cf2dHeavy.h"
class Cf2dHeavy;

#include "Purity.h"
class Purity;

class DataAndModel {

public:

  DataAndModel(Analysis *aAnalysisData, Analysis *aAnalysisModel, double aMinNorm=0.32, double aMaxNorm=0.40, int aRebin=1);
  virtual ~DataAndModel();

  void PrepareHistToPlot(TH1* aHist, double aMarkerStyle=20, double aColor=1);

  void DrawAllCorrectedCfs(TPad* aPad);
  void DrawTrueCorrectionwFit(TPad* aPad);
  void DrawFakeCorrectionwFit(TPad* aPad);
//  void DrawAllModelCfs(TPad* aPad);

  TH1D* MatchBinSize(CfHeavy* aHeavyCf, TH2* aTrueVsRecHist);

  TH1D* GetKStarCorrectedwMatrix(int aMethod, KStarTrueVsRecType aType, double aKStarLow, double aKStarHigh, bool aGetCorrectionFactorInstead=false);
  TH1D* GetKStarCorrectedwMatrixNumDenSmeared(KStarTrueVsRecType aType, double aKStarLow, double aKStarHigh, bool aGetCorrectionFactorInstead=false);

  //inline
  TH1* GetKStarCfCorrectedwTrueHist();
  TH1* GetKStarCfCorrectedwFakeHist();
  TH1* GetKStarCfUncorrected();

  TH1* GetTrueCorrectionHist();
  TH1* GetFakeCorrectionHist();

  AnalysisType GetAnalysisType();
  Analysis* GetAnalysisData();
  Analysis* GetAnalysisModel();

private:


  Analysis* fAnalysisData;
  Analysis* fAnalysisModel;

  AnalysisType fAnalysisType;

  TH1* fKStarCfUncorrected;
  TH1* fKStarCfCorrectedwTrueHist;
  TH1* fKStarCfCorrectedwTrueFit;
  TH1* fKStarCfCorrectedwFakeHist;
  TH1* fKStarCfCorrectedwFakeFit;

  TH1* fKStarCfTrue;
  TH1* fKStarCfTrueIdeal;
  TH1* fKStarCfFake;
  TH1* fKStarCfFakeIdeal;

  TH1* fTrueCorrectionHist;
  TF1* fTrueCorrectionFit;

  TH1* fFakeCorrectionHist;
  TF1* fFakeCorrectionFit;

#ifdef __ROOT__
  ClassDef(DataAndModel, 1)
#endif
};

inline TH1* DataAndModel::GetKStarCfCorrectedwTrueHist() {return fKStarCfCorrectedwTrueHist;}
inline TH1* DataAndModel::GetKStarCfCorrectedwFakeHist() {return fKStarCfCorrectedwFakeHist;}
inline TH1* DataAndModel::GetKStarCfUncorrected() {return fKStarCfUncorrected;}

inline TH1* DataAndModel::GetTrueCorrectionHist() {return fTrueCorrectionHist;}
inline TH1* DataAndModel::GetFakeCorrectionHist() {return fFakeCorrectionHist;}

inline AnalysisType DataAndModel::GetAnalysisType() {return fAnalysisType;}
inline Analysis* DataAndModel::GetAnalysisData() {return fAnalysisData;}
inline Analysis* DataAndModel::GetAnalysisModel() {return fAnalysisModel;}

#endif
