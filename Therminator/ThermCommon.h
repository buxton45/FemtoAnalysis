/* ThermCommon.h */

/* Some common macros used by post-processing programs */

#ifndef THERMCOMMON_H
#define THERMCOMMON_H

#include <iostream>
#include <iomanip>
#include <math.h>
#include <vector>
#include <cassert>

#include "TApplication.h"
#include "TStyle.h"
#include "TPaveText.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "TFile.h"
#include "TPad.h"
#include "TAxis.h"

#include "Types.h"
#include "PIDMapping.h"

using namespace std;


extern TH1D* Get1dHisto(TString FileName, TString HistoName);
extern TH2D* Get2dHisto(TString FileName, TString HistoName);
extern TH3D* Get3dHisto(TString FileName, TString HistoName);

//-----------------------------------------------------------
extern const char* tXAxisLabels_LamKchP[13];
extern const char* tXAxisLabels_ALamKchM[13];

extern const char* tXAxisLabels_LamKchM[13];
extern const char* tXAxisLabels_ALamKchP[13];

extern const char* tXAxisLabels_LamK0[13];
extern const char* tXAxisLabels_ALamK0[13];

extern void SetXAxisLabels(AnalysisType aAnType, TH1D* aHist);

//-----------------------------------------------------------

extern void PrintLambdaValues(TPad* aPad, TH1D* aHisto);
extern void PrintLambdaValuesv2(TPad* aPad, TH1D* aHisto);
extern void DrawPairFractions(TPad* aPad, TH1D* aHisto, bool aSave=false, TString aSaveName = "", double aPairPurity=0.95);
extern void DrawPairFractionsv2(TPad* aPad, TH1D* aHisto, bool aSave=false, TString aSaveName = "", double aPairPurity=0.95);

//-----------------------------------------------------------

extern vector<int> GetParentsPidVector(ParticlePDGType aType);
extern void SetParentPidBinLabels(TAxis* aAxis, ParticlePDGType aType);
extern void DrawParentsMatrixBackground(AnalysisType aAnType, TPad* aPad, TH2D* aMatrix);
extern void DrawOnlyPairsInOthers(AnalysisType aAnType, TPad* aPad, TH2D* aMatrix, double aMaxDecayLength=-1.);
extern void DrawParentsMatrix(AnalysisType aAnType, TPad* aPad, TH2D* aMatrix, bool aZoomROI=false, bool aSetLogZ=false, bool aSave=false, TString aSaveName="", TString aFileType = ".pdf");
extern TH2D* BuildCondensedParentsMatrix(TH2D* aMatrix, TString aReturnName);
extern void DrawCondensedParentsMatrix(AnalysisType aAnType, TPad* aPad, TH2D* aMatrix, bool aSetLogZ=false, bool aSave=false, TString aSaveName="", TString aFileType = ".pdf");

//-----------------------------------------------------------

extern TH2D* BuildCondensed2dRadiiVsBeta(TH2D* a2dHist, TString aReturnName);
extern void DrawCondensed2dRadiiVsPid(ParticlePDGType aType, TPad* aPad, TH2D* a2dHist, bool aSetLogZ=true, bool aSave=false, TString aSaveName="");

//-----------------------------------------------------------

extern TH1D* BuildCondensed1dParentsHistogram(TH1D* a1dHist, TString aReturnName);
extern void DrawCondensed1dParentsHistogram(ParticlePDGType aType, TPad* aPad, TH1D* a1dHist, bool aSave=false, TString aSaveName="");

//-----------------------------------------------------------
extern TH1D* BuildCf(TH1* aNum, TH1* aDen, TString aName, double aMinNorm=0.32, double aMaxNorm=0.40, int aRebin=1);

//-----------------------------------------------------------

#endif
