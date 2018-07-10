#include "TSystem.h"
#include "TLegend.h"
#include <TGraph.h>
#include <TCanvas.h>
#include <TColor.h>
#include <TH1F.h>
#include <TGraphErrors.h>
#include <TGraphAsymmErrors.h>
#include <TMarker.h>
#include <TLatex.h>
#include <TStyle.h>
#include <TLine.h>
#include <TMath.h>
#include <TString.h>
#include "TApplication.h"

#include "Types.h"
#include "Types_LambdaValues.h"
#include "Types_FitParamValues.h"

#include <iostream>
#include <vector>
#include <cassert>
typedef std::vector<double> td1dVec;
typedef std::vector<std::vector<double> > td2dVec;

using namespace std;
//---------------------------------------------------------------------------------------------------------------------------------
struct FitValWriterInfo
{
  AnalysisType analysisType;

  TString masterFileLocation;
  TString resultsDate;
  TString fitInfoTString;

  TString legendDescriptor;
  int markerColor;
  int markerStyle;
  double markerSize;

  FitValWriterInfo(AnalysisType aAnType, TString aMasterFileLocation, TString aResultsDate, TString aFitInfoTString, TString aLegendDescriptor, int aMarkerColor, int aMarkerStyle, double aMarkerSize)
  {
    analysisType       = aAnType;

    masterFileLocation = aMasterFileLocation;
    resultsDate        = aResultsDate;
    fitInfoTString     = aFitInfoTString;

    legendDescriptor   = aLegendDescriptor;
    markerColor        = aMarkerColor;
    markerStyle        = aMarkerStyle;
    markerSize         = aMarkerSize;
  }
};

//---------------------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************************
//---------------------------------------------------------------------------------------------------------------------------------

  TString tFileLocation = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_20180505/MasterFitResults_20180505.txt";
  TString tResultsDate = "20180505";

  int tColorLamK0   = kBlack;
  int tColorLamKchP = kRed+1;
  int tColorLamKchM = kBlue+1;

  int tColorFixedRadiiLamK0 = kGray+1;
  int tColorFixedRadiiLamKchP = kPink+10;
  int tColorFixedRadiiLamKchM = kAzure+10;

  int tMarkerStyle_PolyBgd = 20;
  int tMarkerStyle_LinrBgd = 21;
  int tMarkerStyle_StavCf = 34;

  int tConjMarkerStyle_PolyBgd = 24;
  int tConjMarkerStyle_LinrBgd = 25;
  int tConjMarkerStyle_StavCf = 28;

  double tMarkerSize = 1.5;

//---------------------------------------------------------------------------------------------------------------------------------

  TString tFitInfoTString_SepR_Seplam_PolyBgd      = "_MomResCrctn_NonFlatBgdCrctnPolynomial_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly";
  TString tLegDesc_SepR_Seplam_PolyBgd = "3 Res., Poly. Bgd";
  const FitValWriterInfo tFVWI_SepR_Seplam_PolyBgd_LamKchP = FitValWriterInfo(kLamKchP, tFileLocation, tResultsDate, tFitInfoTString_SepR_Seplam_PolyBgd, 
                                                                              tLegDesc_SepR_Seplam_PolyBgd, tColorLamKchP, tMarkerStyle_PolyBgd, tMarkerSize);
  const FitValWriterInfo tFVWI_SepR_Seplam_PolyBgd_ALamKchM = FitValWriterInfo(kALamKchM, tFileLocation, tResultsDate, tFitInfoTString_SepR_Seplam_PolyBgd, 
                                                                              tLegDesc_SepR_Seplam_PolyBgd, tColorLamKchP, tConjMarkerStyle_PolyBgd, tMarkerSize);
  const FitValWriterInfo tFVWI_SepR_Seplam_PolyBgd_LamKchM = FitValWriterInfo(kLamKchM, tFileLocation, tResultsDate, tFitInfoTString_SepR_Seplam_PolyBgd, 
                                                                              tLegDesc_SepR_Seplam_PolyBgd, tColorLamKchM, tMarkerStyle_PolyBgd, tMarkerSize);
  const FitValWriterInfo tFVWI_SepR_Seplam_PolyBgd_ALamKchP = FitValWriterInfo(kALamKchP, tFileLocation, tResultsDate, tFitInfoTString_SepR_Seplam_PolyBgd, 
                                                                              tLegDesc_SepR_Seplam_PolyBgd, tColorLamKchM, tConjMarkerStyle_PolyBgd, tMarkerSize);
  //-------------------------
  TString tFitInfoTString_SepR_Seplam_LinrBgd      = "_MomResCrctn_NonFlatBgdCrctnLinear_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly";
  TString tLegDesc_SepR_Seplam_LinrBgd = "3 Res., Linr. Bgd";
  const FitValWriterInfo tFVWI_SepR_Seplam_LinrBgd_LamKchP = FitValWriterInfo(kLamKchP, tFileLocation, tResultsDate, tFitInfoTString_SepR_Seplam_LinrBgd, 
                                                                              tLegDesc_SepR_Seplam_LinrBgd, tColorLamKchP, tMarkerStyle_LinrBgd, tMarkerSize);
  const FitValWriterInfo tFVWI_SepR_Seplam_LinrBgd_ALamKchM = FitValWriterInfo(kALamKchM, tFileLocation, tResultsDate, tFitInfoTString_SepR_Seplam_LinrBgd, 
                                                                              tLegDesc_SepR_Seplam_LinrBgd, tColorLamKchP, tConjMarkerStyle_LinrBgd, tMarkerSize);
  const FitValWriterInfo tFVWI_SepR_Seplam_LinrBgd_LamKchM = FitValWriterInfo(kLamKchM, tFileLocation, tResultsDate, tFitInfoTString_SepR_Seplam_LinrBgd, 
                                                                              tLegDesc_SepR_Seplam_LinrBgd, tColorLamKchM, tMarkerStyle_LinrBgd, tMarkerSize);
  const FitValWriterInfo tFVWI_SepR_Seplam_LinrBgd_ALamKchP = FitValWriterInfo(kALamKchP, tFileLocation, tResultsDate, tFitInfoTString_SepR_Seplam_LinrBgd, 
                                                                              tLegDesc_SepR_Seplam_LinrBgd, tColorLamKchM, tConjMarkerStyle_LinrBgd, tMarkerSize);
  //-------------------------
  TString tFitInfoTString_SepR_Seplam_StavCf_NoBgd = "_MomResCrctn_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_MomResCrctn_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_StavCf";
  TString tLegDesc_SepR_Seplam_StavCf_NoBgd = "3 Res., Stav. (No Bgd)";
  const FitValWriterInfo tFVWI_SepR_Seplam_StavCf_NoBgd_LamKchP = FitValWriterInfo(kLamKchP, tFileLocation, tResultsDate, tFitInfoTString_SepR_Seplam_StavCf_NoBgd, 
                                                                              tLegDesc_SepR_Seplam_StavCf_NoBgd, tColorLamKchP, tMarkerStyle_StavCf, tMarkerSize);
  const FitValWriterInfo tFVWI_SepR_Seplam_StavCf_NoBgd_ALamKchM = FitValWriterInfo(kALamKchM, tFileLocation, tResultsDate, tFitInfoTString_SepR_Seplam_StavCf_NoBgd, 
                                                                              tLegDesc_SepR_Seplam_StavCf_NoBgd, tColorLamKchP, tConjMarkerStyle_StavCf, tMarkerSize);
  const FitValWriterInfo tFVWI_SepR_Seplam_StavCf_NoBgd_LamKchM = FitValWriterInfo(kLamKchM, tFileLocation, tResultsDate, tFitInfoTString_SepR_Seplam_StavCf_NoBgd, 
                                                                              tLegDesc_SepR_Seplam_StavCf_NoBgd, tColorLamKchM, tMarkerStyle_StavCf, tMarkerSize);
  const FitValWriterInfo tFVWI_SepR_Seplam_StavCf_NoBgd_ALamKchP = FitValWriterInfo(kALamKchP, tFileLocation, tResultsDate, tFitInfoTString_SepR_Seplam_StavCf_NoBgd, 
                                                                              tLegDesc_SepR_Seplam_StavCf_NoBgd, tColorLamKchM, tConjMarkerStyle_StavCf, tMarkerSize);
  //-------------------------
//---------------------------------------------------------------------------------------------------------------------------------
  TString tFitInfoTString_SepR_Sharelam_PolyBgd      = "_MomResCrctn_NonFlatBgdCrctnPolynomial_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_ShareLam";
  TString tLegDesc_SepR_Sharelam_PolyBgd = "3 Res., Share #lambda, Poly. Bgd";
  const FitValWriterInfo tFVWI_SepR_Sharelam_PolyBgd_LamKchP = FitValWriterInfo(kLamKchP, tFileLocation, tResultsDate, tFitInfoTString_SepR_Sharelam_PolyBgd, 
                                                                              tLegDesc_SepR_Sharelam_PolyBgd, tColorLamKchP, tMarkerStyle_PolyBgd, tMarkerSize);
  const FitValWriterInfo tFVWI_SepR_Sharelam_PolyBgd_LamKchM = FitValWriterInfo(kLamKchM, tFileLocation, tResultsDate, tFitInfoTString_SepR_Sharelam_PolyBgd, 
                                                                              tLegDesc_SepR_Sharelam_PolyBgd, tColorLamKchM, tMarkerStyle_PolyBgd, tMarkerSize);
  //-------------------------
  TString tFitInfoTString_SepR_Sharelam_LinrBgd      = "_MomResCrctn_NonFlatBgdCrctnLinear_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_ShareLam";
  TString tLegDesc_SepR_Sharelam_LinrBgd = "3 Res., Share #lambda, Linr. Bgd";
  const FitValWriterInfo tFVWI_SepR_Sharelam_LinrBgd_LamKchP = FitValWriterInfo(kLamKchP, tFileLocation, tResultsDate, tFitInfoTString_SepR_Sharelam_LinrBgd, 
                                                                              tLegDesc_SepR_Sharelam_LinrBgd, tColorLamKchP, tMarkerStyle_LinrBgd, tMarkerSize);
  const FitValWriterInfo tFVWI_SepR_Sharelam_LinrBgd_LamKchM = FitValWriterInfo(kLamKchM, tFileLocation, tResultsDate, tFitInfoTString_SepR_Sharelam_LinrBgd, 
                                                                              tLegDesc_SepR_Sharelam_LinrBgd, tColorLamKchM, tMarkerStyle_LinrBgd, tMarkerSize);
  //-------------------------
  TString tFitInfoTString_SepR_Sharelam_StavCf_NoBgd = "_MomResCrctn_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_MomResCrctn_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_StavCf_ShareLam";
  TString tLegDesc_SepR_Sharelam_StavCf_NoBgd = "3 Res., Share #lambda, Stav. (No Bgd)";
  const FitValWriterInfo tFVWI_SepR_Sharelam_StavCf_NoBgd_LamKchP = FitValWriterInfo(kLamKchP, tFileLocation, tResultsDate, tFitInfoTString_SepR_Sharelam_StavCf_NoBgd, 
                                                                              tLegDesc_SepR_Sharelam_StavCf_NoBgd, tColorLamKchP, tMarkerStyle_StavCf, tMarkerSize);
  const FitValWriterInfo tFVWI_SepR_Sharelam_StavCf_NoBgd_LamKchM = FitValWriterInfo(kLamKchM, tFileLocation, tResultsDate, tFitInfoTString_SepR_Sharelam_StavCf_NoBgd, 
                                                                              tLegDesc_SepR_Sharelam_StavCf_NoBgd, tColorLamKchM, tMarkerStyle_StavCf, tMarkerSize);
  //-------------------------
//---------------------------------------------------------------------------------------------------------------------------------
  TString tFitInfoTString_ShareR_Sharelam_PolyBgd      = "_MomResCrctn_NonFlatBgdCrctnPolynomial_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_ShareLam_Dualie_ShareLam_ShareRadii";
  TString tLegDesc_ShareR_Sharelam_PolyBgd = "3 Res., Share R and #lambda, Poly. Bgd";
  const FitValWriterInfo tFVWI_ShareR_Sharelam_PolyBgd_LamKchP = FitValWriterInfo(kLamKchP, tFileLocation, tResultsDate, tFitInfoTString_ShareR_Sharelam_PolyBgd, 
                                                                              tLegDesc_ShareR_Sharelam_PolyBgd, tColorLamKchP, tMarkerStyle_PolyBgd, tMarkerSize);
  const FitValWriterInfo tFVWI_ShareR_Sharelam_PolyBgd_LamKchM = FitValWriterInfo(kLamKchM, tFileLocation, tResultsDate, tFitInfoTString_ShareR_Sharelam_PolyBgd, 
                                                                              tLegDesc_ShareR_Sharelam_PolyBgd, tColorLamKchM, tMarkerStyle_PolyBgd, tMarkerSize);
  //-------------------------
  TString tFitInfoTString_ShareR_Sharelam_LinrBgd      = "_MomResCrctn_NonFlatBgdCrctnLinear_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_ShareLam_Dualie_ShareLam_ShareRadii";
  TString tLegDesc_ShareR_Sharelam_LinrBgd = "3 Res., Share R and #lambda, Linr. Bgd";
  const FitValWriterInfo tFVWI_ShareR_Sharelam_LinrBgd_LamKchP = FitValWriterInfo(kLamKchP, tFileLocation, tResultsDate, tFitInfoTString_ShareR_Sharelam_LinrBgd, 
                                                                              tLegDesc_ShareR_Sharelam_LinrBgd, tColorLamKchP, tMarkerStyle_LinrBgd, tMarkerSize);
  const FitValWriterInfo tFVWI_ShareR_Sharelam_LinrBgd_LamKchM = FitValWriterInfo(kLamKchM, tFileLocation, tResultsDate, tFitInfoTString_ShareR_Sharelam_LinrBgd, 
                                                                              tLegDesc_ShareR_Sharelam_LinrBgd, tColorLamKchM, tMarkerStyle_LinrBgd, tMarkerSize);
  //-------------------------
  TString tFitInfoTString_ShareR_Sharelam_StavCf_NoBgd = "_MomResCrctn_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_MomResCrctn_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_StavCf_ShareLam_Dualie_ShareLam_ShareRadii";
  TString tLegDesc_ShareR_Sharelam_StavCf_NoBgd = "3 Res., Share R and #lambda, Stav. (No Bgd)";
  const FitValWriterInfo tFVWI_ShareR_Sharelam_StavCf_NoBgd_LamKchP = FitValWriterInfo(kLamKchP, tFileLocation, tResultsDate, tFitInfoTString_ShareR_Sharelam_StavCf_NoBgd, 
                                                                              tLegDesc_ShareR_Sharelam_StavCf_NoBgd, tColorLamKchP, tMarkerStyle_StavCf, tMarkerSize);
  const FitValWriterInfo tFVWI_ShareR_Sharelam_StavCf_NoBgd_LamKchM = FitValWriterInfo(kLamKchM, tFileLocation, tResultsDate, tFitInfoTString_ShareR_Sharelam_StavCf_NoBgd, 
                                                                              tLegDesc_ShareR_Sharelam_StavCf_NoBgd, tColorLamKchM, tMarkerStyle_StavCf, tMarkerSize);
  //-------------------------



//---------------------------------------------------------------------------------------------------------------------------------


  vector<FitValWriterInfo> tFVWIVec_SepR_Seplam = {tFVWI_SepR_Seplam_PolyBgd_LamKchP, tFVWI_SepR_Seplam_LinrBgd_LamKchP, tFVWI_SepR_Seplam_StavCf_NoBgd_LamKchP,
                                                   tFVWI_SepR_Seplam_PolyBgd_ALamKchM, tFVWI_SepR_Seplam_LinrBgd_ALamKchM, tFVWI_SepR_Seplam_StavCf_NoBgd_ALamKchM,
                                                   tFVWI_SepR_Seplam_PolyBgd_LamKchM, tFVWI_SepR_Seplam_LinrBgd_LamKchM, tFVWI_SepR_Seplam_StavCf_NoBgd_LamKchM,
                                                   tFVWI_SepR_Seplam_PolyBgd_ALamKchP, tFVWI_SepR_Seplam_LinrBgd_ALamKchP, tFVWI_SepR_Seplam_StavCf_NoBgd_ALamKchP};

  vector<FitValWriterInfo> tFVWIVec_SepR_Sharelam = {tFVWI_SepR_Sharelam_PolyBgd_LamKchP, tFVWI_SepR_Sharelam_LinrBgd_LamKchP, tFVWI_SepR_Sharelam_StavCf_NoBgd_LamKchP,
                                                     tFVWI_SepR_Sharelam_PolyBgd_LamKchM, tFVWI_SepR_Sharelam_LinrBgd_LamKchM, tFVWI_SepR_Sharelam_StavCf_NoBgd_LamKchM};

  vector<FitValWriterInfo> tFVWIVec_ShareR_Sharelam = {tFVWI_ShareR_Sharelam_PolyBgd_LamKchP, tFVWI_ShareR_Sharelam_LinrBgd_LamKchP, tFVWI_ShareR_Sharelam_StavCf_NoBgd_LamKchP,
                                                       tFVWI_ShareR_Sharelam_PolyBgd_LamKchM, tFVWI_ShareR_Sharelam_LinrBgd_LamKchM, tFVWI_ShareR_Sharelam_StavCf_NoBgd_LamKchM};


  vector<FitValWriterInfo> tFVWIVec_SharevsSepR = {FitValWriterInfo(kLamKchP, tFileLocation, tResultsDate, tFitInfoTString_ShareR_Sharelam_PolyBgd, 
                                                                    tLegDesc_ShareR_Sharelam_PolyBgd, tColorLamKchP, 20, tMarkerSize), 
                                                   FitValWriterInfo(kLamKchP, tFileLocation, tResultsDate, tFitInfoTString_SepR_Sharelam_PolyBgd, 
                                                                    tLegDesc_SepR_Sharelam_PolyBgd, tColorLamKchP, 21, tMarkerSize), 
                                                   FitValWriterInfo(kLamKchP, tFileLocation, tResultsDate, tFitInfoTString_SepR_Seplam_PolyBgd, 
                                                                    tLegDesc_SepR_Seplam_PolyBgd, tColorLamKchP, 34, tMarkerSize), 
                                                   FitValWriterInfo(kALamKchM, tFileLocation, tResultsDate, tFitInfoTString_SepR_Seplam_PolyBgd, 
                                                                    tLegDesc_SepR_Seplam_PolyBgd, tColorLamKchP, 28, tMarkerSize), 
                                                   FitValWriterInfo(kLamKchM, tFileLocation, tResultsDate, tFitInfoTString_ShareR_Sharelam_PolyBgd, 
                                                                    tLegDesc_ShareR_Sharelam_PolyBgd, tColorLamKchM, 20, tMarkerSize), 
                                                   FitValWriterInfo(kLamKchM, tFileLocation, tResultsDate, tFitInfoTString_SepR_Sharelam_PolyBgd, 
                                                                    tLegDesc_SepR_Sharelam_PolyBgd, tColorLamKchM, 21, tMarkerSize), 
                                                   FitValWriterInfo(kLamKchM, tFileLocation, tResultsDate, tFitInfoTString_SepR_Seplam_PolyBgd, 
                                                                    tLegDesc_SepR_Seplam_PolyBgd, tColorLamKchM, 34, tMarkerSize), 
                                                   FitValWriterInfo(kALamKchP, tFileLocation, tResultsDate, tFitInfoTString_SepR_Seplam_PolyBgd, 
                                                                    tLegDesc_SepR_Seplam_PolyBgd, tColorLamKchM, 28, tMarkerSize)};





