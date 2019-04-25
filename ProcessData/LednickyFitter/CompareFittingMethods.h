#ifndef COMPAREFITTINGMETHODS_H_
#define COMPAREFITTINGMETHODS_H_

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
#include "FitValuesWriter.h"

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
  bool lamKchCombined;
  bool allLamKCombined;

  FitValWriterInfo(AnalysisType aAnType, TString aMasterFileLocation, TString aResultsDate, TString aFitInfoTString, TString aLegendDescriptor, int aMarkerColor, int aMarkerStyle, double aMarkerSize, bool aLamKchCombined, bool aAllLamKCombined=false)
  {
    analysisType       = aAnType;

    masterFileLocation = aMasterFileLocation;
    resultsDate        = aResultsDate;
    fitInfoTString     = aFitInfoTString;

    legendDescriptor   = aLegendDescriptor;
    markerColor        = aMarkerColor;
    markerStyle        = aMarkerStyle;
    markerSize         = aMarkerSize;
    lamKchCombined     = aLamKchCombined;
    allLamKCombined    = aAllLamKCombined;
  }
};

//---------------------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************************
//-------------------------------------------------------- List of FitInfo --------------------------------------------------------
/*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ LamKch ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//--------------- Separate Radii -----------------------------------------------------------
_MomResCrctn_NonFlatBgdCrctnPolynomial_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly
_MomResCrctn_NonFlatBgdCrctnLinear_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly
_MomResCrctn_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_StavCf
_MomResCrctn_NonFlatBgdCrctnPolynomial_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_ShareLam
_MomResCrctn_NonFlatBgdCrctnLinear_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_ShareLam
_MomResCrctn_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_StavCf_ShareLam
//----- Fix d0 -----
_MomResCrctn_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_FixedD0_StavCf
_MomResCrctn_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_FixedD0_StavCf_ShareLam
_MomResCrctn_NonFlatBgdCrctnPolynomial_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_FixedD0
_MomResCrctn_NonFlatBgdCrctnPolynomial_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_FixedD0_ShareLam
_MomResCrctn_NonFlatBgdCrctnLinear_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_FixedD0
_MomResCrctn_NonFlatBgdCrctnLinear_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_FixedD0_ShareLam
//----- Fix lam ----
_MomResCrctn_NonFlatBgdCrctnPolynomial_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_FixAllLambdaTo1
_MomResCrctn_NonFlatBgdCrctnLinear_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_FixAllLambdaTo1
_MomResCrctn_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_StavCf_FixAllLambdaTo1


//--------------- Shared Radii -----------------------------------------------------------
_MomResCrctn_NonFlatBgdCrctnPolynomial_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_ShareLam_Dualie_ShareLam_ShareRadii
_MomResCrctn_NonFlatBgdCrctnLinear_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_ShareLam_Dualie_ShareLam_ShareRadii
_MomResCrctn_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_StavCf_ShareLam_Dualie_ShareLam_ShareRadii
_MomResCrctn_NonFlatBgdCrctnPolynomial_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_ShareLam_Dualie_ShareRadii
_MomResCrctn_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_StavCf_ShareLam_Dualie_ShareRadii
_MomResCrctn_NonFlatBgdCrctnLinear_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_ShareLam_Dualie_ShareRadii
//----- Fix d0 -----
_MomResCrctn_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_FixedD0_StavCf_ShareLam_Dualie_ShareLam_ShareRadii
_MomResCrctn_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_FixedD0_StavCf_ShareLam_Dualie_ShareRadii
_MomResCrctn_NonFlatBgdCrctnPolynomial_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_FixedD0_ShareLam_Dualie_ShareLam_ShareRadii
_MomResCrctn_NonFlatBgdCrctnPolynomial_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_FixedD0_ShareLam_Dualie_ShareRadii
_MomResCrctn_NonFlatBgdCrctnLinear_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_FixedD0_ShareLam_Dualie_ShareLam_ShareRadii
_MomResCrctn_NonFlatBgdCrctnLinear_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_FixedD0_ShareLam_Dualie_ShareRadii
//----- Fix lam ----
_MomResCrctn_NonFlatBgdCrctnPolynomial_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_FixAllLambdaTo1_ShareLam_Dualie_ShareLam_ShareRadii
_MomResCrctn_NonFlatBgdCrctnLinear_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_FixAllLambdaTo1_ShareLam_Dualie_ShareLam_ShareRadii
_MomResCrctn_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_StavCf_FixAllLambdaTo1_ShareLam_Dualie_ShareLam_ShareRadii



^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ LamK0 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_MomResCrctn_NonFlatBgdCrctnPolynomial_NoRes_SingleLamParam
_MomResCrctn_NonFlatBgdCrctnPolynomial_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_SingleLamParam
_MomResCrctn_NonFlatBgdCrctnPolynomial_10Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_SingleLamParam
_MomResCrctn_NonFlatBgdCrctnPolynomial_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_FixAllLambdaTo1
_MomResCrctn_NonFlatBgdCrctnPolynomial_10Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_FixAllLambdaTo1
_MomResCrctn_NonFlatBgdCrctnPolynomial_NoRes_FixAllLambdaTo1
_MomResCrctn_NonFlatBgdCrctnPolynomial_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly
_MomResCrctn_NonFlatBgdCrctnPolynomial_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_FixAllNormTo1_SingleLamParam
_MomResCrctn_NonFlatBgdCrctnPolynomial_NoRes_ShareLam_SingleLamParam
_MomResCrctn_NonFlatBgdCrctnPolynomial_NoRes_FixAllNormTo1_SingleLamParam
_MomResCrctn_NonFlatBgdCrctnLinear_NoRes_FixAllNormTo1_SingleLamParam
_MomResCrctn_NonFlatBgdCrctnPolynomial_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_FixAllNormTo1
_MomResCrctn_NonFlatBgdCrctnPolynomial_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_FixAllNormTo1_ShareLam
_MomResCrctn_NonFlatBgdCrctnPolynomial_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_ShareLam
_MomResCrctn_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_StavCf_SingleLamParam
_MomResCrctn_NoRes_StavCf_SingleLamParam
_MomResCrctn_10Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_StavCf_SingleLamParam
_MomResCrctn_NonFlatBgdCrctnLinear_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_SingleLamParam
*/


//---------------------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************************
//---------------------------------------------------------------------------------------------------------------------------------

  TString tFileLocation_LamKch = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_20180505/MasterFitResults_20180505.txt";
  TString tFileLocation_LamK0 = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamK0_20180505/MasterFitResults_20180505.txt";
  TString tResultsDate = "20180505";

  int tColorLamK0   = kBlack;
  int tColorLamKchP = kRed+1;
  int tColorLamKchM = kBlue+1;

  int tColorFixedRadiiLamK0 = kGray+1;
  int tColorFixedRadiiLamKchP = kPink+10;
  int tColorFixedRadiiLamKchM = kAzure+10;
  //---------------
  int tMarkerStyle_PolyBgd = 20;
  int tMarkerStyle_LinrBgd = 21;
  int tMarkerStyle_StavCf = 34;

  int tConjMarkerStyle_PolyBgd = 24;
  int tConjMarkerStyle_LinrBgd = 25;
  int tConjMarkerStyle_StavCf = 28;
  //---------------
  int tMarkerStyle_FixParam_PolyBgd = 22;
  int tMarkerStyle_FixParam_LinrBgd = 33;
  int tMarkerStyle_FixParam_StavCf = 47;

  int tConjMarkerStyle_FixParam_PolyBgd = 26;
  int tConjMarkerStyle_FixParam_LinrBgd = 27;
  int tConjMarkerStyle_FixParam_StavCf = 46;
  //---------------
  double tMarkerSize = 1.5;

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Separate Radii %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

//---------------------------------------------------------------------------------------------------------------------------------

  TString tFitInfoTString_SepR_Seplam_PolyBgd      = "_MomResCrctn_NonFlatBgdCrctnPolynomial_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly";
  TString tLegDesc_SepR_Seplam_PolyBgd = "3 Res., Poly. Bgd";
  const FitValWriterInfo tFVWI_SepR_Seplam_PolyBgd_LamKchP = FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Seplam_PolyBgd, 
                                                                              tLegDesc_SepR_Seplam_PolyBgd, tColorLamKchP, tMarkerStyle_PolyBgd, tMarkerSize, false);
  const FitValWriterInfo tFVWI_SepR_Seplam_PolyBgd_ALamKchM = FitValWriterInfo(kALamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Seplam_PolyBgd, 
                                                                              tLegDesc_SepR_Seplam_PolyBgd, tColorLamKchP, tConjMarkerStyle_PolyBgd, tMarkerSize, false);
  const FitValWriterInfo tFVWI_SepR_Seplam_PolyBgd_LamKchM = FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Seplam_PolyBgd, 
                                                                              tLegDesc_SepR_Seplam_PolyBgd, tColorLamKchM, tMarkerStyle_PolyBgd, tMarkerSize, false);
  const FitValWriterInfo tFVWI_SepR_Seplam_PolyBgd_ALamKchP = FitValWriterInfo(kALamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Seplam_PolyBgd, 
                                                                              tLegDesc_SepR_Seplam_PolyBgd, tColorLamKchM, tConjMarkerStyle_PolyBgd, tMarkerSize, false);
  //-------------------------
  TString tFitInfoTString_SepR_Seplam_LinrBgd      = "_MomResCrctn_NonFlatBgdCrctnLinear_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly";
  TString tLegDesc_SepR_Seplam_LinrBgd = "3 Res., Linr. Bgd";
  const FitValWriterInfo tFVWI_SepR_Seplam_LinrBgd_LamKchP = FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Seplam_LinrBgd, 
                                                                              tLegDesc_SepR_Seplam_LinrBgd, tColorLamKchP, tMarkerStyle_LinrBgd, tMarkerSize, false);
  const FitValWriterInfo tFVWI_SepR_Seplam_LinrBgd_ALamKchM = FitValWriterInfo(kALamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Seplam_LinrBgd, 
                                                                              tLegDesc_SepR_Seplam_LinrBgd, tColorLamKchP, tConjMarkerStyle_LinrBgd, tMarkerSize, false);
  const FitValWriterInfo tFVWI_SepR_Seplam_LinrBgd_LamKchM = FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Seplam_LinrBgd, 
                                                                              tLegDesc_SepR_Seplam_LinrBgd, tColorLamKchM, tMarkerStyle_LinrBgd, tMarkerSize, false);
  const FitValWriterInfo tFVWI_SepR_Seplam_LinrBgd_ALamKchP = FitValWriterInfo(kALamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Seplam_LinrBgd, 
                                                                              tLegDesc_SepR_Seplam_LinrBgd, tColorLamKchM, tConjMarkerStyle_LinrBgd, tMarkerSize, false);
  //-------------------------
  TString tFitInfoTString_SepR_Seplam_StavCf_NoBgd = "_MomResCrctn_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_StavCf";
  TString tLegDesc_SepR_Seplam_StavCf_NoBgd = "3 Res., Stav. (No Bgd)";
  const FitValWriterInfo tFVWI_SepR_Seplam_StavCf_NoBgd_LamKchP = FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Seplam_StavCf_NoBgd, 
                                                                              tLegDesc_SepR_Seplam_StavCf_NoBgd, tColorLamKchP, tMarkerStyle_StavCf, tMarkerSize, false);
  const FitValWriterInfo tFVWI_SepR_Seplam_StavCf_NoBgd_ALamKchM = FitValWriterInfo(kALamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Seplam_StavCf_NoBgd, 
                                                                              tLegDesc_SepR_Seplam_StavCf_NoBgd, tColorLamKchP, tConjMarkerStyle_StavCf, tMarkerSize, false);
  const FitValWriterInfo tFVWI_SepR_Seplam_StavCf_NoBgd_LamKchM = FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Seplam_StavCf_NoBgd, 
                                                                              tLegDesc_SepR_Seplam_StavCf_NoBgd, tColorLamKchM, tMarkerStyle_StavCf, tMarkerSize, false);
  const FitValWriterInfo tFVWI_SepR_Seplam_StavCf_NoBgd_ALamKchP = FitValWriterInfo(kALamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Seplam_StavCf_NoBgd, 
                                                                              tLegDesc_SepR_Seplam_StavCf_NoBgd, tColorLamKchM, tConjMarkerStyle_StavCf, tMarkerSize, false);
  //-------------------------
//---------------------------------------------------------------------------------------------------------------------------------
  TString tFitInfoTString_SepR_Sharelam_PolyBgd      = "_MomResCrctn_NonFlatBgdCrctnPolynomial_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_ShareLam";
  TString tLegDesc_SepR_Sharelam_PolyBgd = "3 Res., Share #lambda, Poly. Bgd";
  const FitValWriterInfo tFVWI_SepR_Sharelam_PolyBgd_LamKchP = FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Sharelam_PolyBgd, 
                                                                              tLegDesc_SepR_Sharelam_PolyBgd, tColorLamKchP, tMarkerStyle_PolyBgd, tMarkerSize, false);
  const FitValWriterInfo tFVWI_SepR_Sharelam_PolyBgd_LamKchM = FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Sharelam_PolyBgd, 
                                                                              tLegDesc_SepR_Sharelam_PolyBgd, tColorLamKchM, tMarkerStyle_PolyBgd, tMarkerSize, false);
  //-------------------------
  TString tFitInfoTString_SepR_Sharelam_LinrBgd      = "_MomResCrctn_NonFlatBgdCrctnLinear_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_ShareLam";
  TString tLegDesc_SepR_Sharelam_LinrBgd = "3 Res., Share #lambda, Linr. Bgd";
  const FitValWriterInfo tFVWI_SepR_Sharelam_LinrBgd_LamKchP = FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Sharelam_LinrBgd, 
                                                                              tLegDesc_SepR_Sharelam_LinrBgd, tColorLamKchP, tMarkerStyle_LinrBgd, tMarkerSize, false);
  const FitValWriterInfo tFVWI_SepR_Sharelam_LinrBgd_LamKchM = FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Sharelam_LinrBgd, 
                                                                              tLegDesc_SepR_Sharelam_LinrBgd, tColorLamKchM, tMarkerStyle_LinrBgd, tMarkerSize, false);
  //-------------------------
  TString tFitInfoTString_SepR_Sharelam_StavCf_NoBgd = "_MomResCrctn_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_StavCf_ShareLam";
  TString tLegDesc_SepR_Sharelam_StavCf_NoBgd = "3 Res., Share #lambda, Stav. (No Bgd)";
  const FitValWriterInfo tFVWI_SepR_Sharelam_StavCf_NoBgd_LamKchP = FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Sharelam_StavCf_NoBgd, 
                                                                              tLegDesc_SepR_Sharelam_StavCf_NoBgd, tColorLamKchP, tMarkerStyle_StavCf, tMarkerSize, false);
  const FitValWriterInfo tFVWI_SepR_Sharelam_StavCf_NoBgd_LamKchM = FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Sharelam_StavCf_NoBgd, 
                                                                              tLegDesc_SepR_Sharelam_StavCf_NoBgd, tColorLamKchM, tMarkerStyle_StavCf, tMarkerSize, false);
  //-------------------------
//################################### Fix lambda
//---------------------------------------------------------------------------------------------------------------------------------
  TString tFitInfoTString_SepR_Fixlam_PolyBgd      = "_MomResCrctn_NonFlatBgdCrctnPolynomial_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_FixAllLambdaTo1";
  TString tLegDesc_SepR_Fixlam_PolyBgd = "3 Res., Fix #lambda, Poly. Bgd";
  const FitValWriterInfo tFVWI_SepR_Fixlam_PolyBgd_LamKchP = FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Fixlam_PolyBgd, 
                                                                              tLegDesc_SepR_Fixlam_PolyBgd, tColorLamKchP, tMarkerStyle_FixParam_PolyBgd, tMarkerSize, false);
  const FitValWriterInfo tFVWI_SepR_Fixlam_PolyBgd_LamKchM = FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Fixlam_PolyBgd, 
                                                                              tLegDesc_SepR_Fixlam_PolyBgd, tColorLamKchM, tMarkerStyle_FixParam_PolyBgd, tMarkerSize, false);
  //-------------------------
  TString tFitInfoTString_SepR_Fixlam_LinrBgd      = "_MomResCrctn_NonFlatBgdCrctnLinear_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_FixAllLambdaTo1";
  TString tLegDesc_SepR_Fixlam_LinrBgd = "3 Res., Fix #lambda, Linr. Bgd";
  const FitValWriterInfo tFVWI_SepR_Fixlam_LinrBgd_LamKchP = FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Fixlam_LinrBgd, 
                                                                              tLegDesc_SepR_Fixlam_LinrBgd, tColorLamKchP, tMarkerStyle_FixParam_LinrBgd, tMarkerSize, false);
  const FitValWriterInfo tFVWI_SepR_Fixlam_LinrBgd_LamKchM = FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Fixlam_LinrBgd, 
                                                                              tLegDesc_SepR_Fixlam_LinrBgd, tColorLamKchM, tMarkerStyle_FixParam_LinrBgd, tMarkerSize, false);
  //-------------------------
  TString tFitInfoTString_SepR_Fixlam_StavCf_NoBgd = "_MomResCrctn_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_StavCf_FixAllLambdaTo1";
  TString tLegDesc_SepR_Fixlam_StavCf_NoBgd = "3 Res., Fix #lambda, Stav. (No Bgd)";
  const FitValWriterInfo tFVWI_SepR_Fixlam_StavCf_NoBgd_LamKchP = FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Fixlam_StavCf_NoBgd, 
                                                                              tLegDesc_SepR_Fixlam_StavCf_NoBgd, tColorLamKchP, tMarkerStyle_FixParam_StavCf, tMarkerSize, false);
  const FitValWriterInfo tFVWI_SepR_Fixlam_StavCf_NoBgd_LamKchM = FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Fixlam_StavCf_NoBgd, 
                                                                              tLegDesc_SepR_Fixlam_StavCf_NoBgd, tColorLamKchM, tMarkerStyle_FixParam_StavCf, tMarkerSize, false);
  //-------------------------

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Shared Radii %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

//---------------------------------------------------------------------------------------------------------------------------------
  TString tFitInfoTString_ShareR_Sharelam_PolyBgd      = "_MomResCrctn_NonFlatBgdCrctnPolynomial_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_ShareLam_Dualie_ShareLam_ShareRadii";
  TString tLegDesc_ShareR_Sharelam_PolyBgd = "3 Res., Share R and #lambda, Poly. Bgd";
  const FitValWriterInfo tFVWI_ShareR_Sharelam_PolyBgd_LamKchP = FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Sharelam_PolyBgd, 
                                                                              tLegDesc_ShareR_Sharelam_PolyBgd, tColorLamKchP, tMarkerStyle_PolyBgd, tMarkerSize, true);
  const FitValWriterInfo tFVWI_ShareR_Sharelam_PolyBgd_LamKchM = FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Sharelam_PolyBgd, 
                                                                              tLegDesc_ShareR_Sharelam_PolyBgd, tColorLamKchM, tMarkerStyle_PolyBgd, tMarkerSize, true);
  //-------------------------
  TString tFitInfoTString_ShareR_Sharelam_LinrBgd      = "_MomResCrctn_NonFlatBgdCrctnLinear_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_ShareLam_Dualie_ShareLam_ShareRadii";
  TString tLegDesc_ShareR_Sharelam_LinrBgd = "3 Res., Share R and #lambda, Linr. Bgd";
  const FitValWriterInfo tFVWI_ShareR_Sharelam_LinrBgd_LamKchP = FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Sharelam_LinrBgd, 
                                                                              tLegDesc_ShareR_Sharelam_LinrBgd, tColorLamKchP, tMarkerStyle_LinrBgd, tMarkerSize, true);
  const FitValWriterInfo tFVWI_ShareR_Sharelam_LinrBgd_LamKchM = FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Sharelam_LinrBgd, 
                                                                              tLegDesc_ShareR_Sharelam_LinrBgd, tColorLamKchM, tMarkerStyle_LinrBgd, tMarkerSize, true);
  //-------------------------
  TString tFitInfoTString_ShareR_Sharelam_StavCf_NoBgd = "_MomResCrctn_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_StavCf_ShareLam_Dualie_ShareLam_ShareRadii";
  TString tLegDesc_ShareR_Sharelam_StavCf_NoBgd = "3 Res., Share R and #lambda, Stav. (No Bgd)";
  const FitValWriterInfo tFVWI_ShareR_Sharelam_StavCf_NoBgd_LamKchP = FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Sharelam_StavCf_NoBgd, 
                                                                              tLegDesc_ShareR_Sharelam_StavCf_NoBgd, tColorLamKchP, tMarkerStyle_StavCf, tMarkerSize, true);
  const FitValWriterInfo tFVWI_ShareR_Sharelam_StavCf_NoBgd_LamKchM = FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Sharelam_StavCf_NoBgd, 
                                                                              tLegDesc_ShareR_Sharelam_StavCf_NoBgd, tColorLamKchM, tMarkerStyle_StavCf, tMarkerSize, true);
  //-------------------------
//---------------------------------------------------------------------------------------------------------------------------------
  TString tFitInfoTString_ShareR_SharelamConj_PolyBgd      = "_MomResCrctn_NonFlatBgdCrctnPolynomial_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_ShareLam_Dualie_ShareRadii";
  TString tLegDesc_ShareR_SharelamConj_PolyBgd = "3 Res., Share R and #lambda_{Conj}, Poly. Bgd";
  const FitValWriterInfo tFVWI_ShareR_SharelamConj_PolyBgd_LamKchP = FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_SharelamConj_PolyBgd, 
                                                                              tLegDesc_ShareR_SharelamConj_PolyBgd, tColorLamKchP, tMarkerStyle_PolyBgd, tMarkerSize, true);
  const FitValWriterInfo tFVWI_ShareR_SharelamConj_PolyBgd_LamKchM = FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_SharelamConj_PolyBgd, 
                                                                              tLegDesc_ShareR_SharelamConj_PolyBgd, tColorLamKchM, tMarkerStyle_PolyBgd, tMarkerSize, true);
  //-------------------------
  TString tFitInfoTString_ShareR_SharelamConj_LinrBgd      = "_MomResCrctn_NonFlatBgdCrctnLinear_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_ShareLam_Dualie_ShareRadii";
  TString tLegDesc_ShareR_SharelamConj_LinrBgd = "3 Res., Share R and #lambda_{Conj}, Linr. Bgd";
  const FitValWriterInfo tFVWI_ShareR_SharelamConj_LinrBgd_LamKchP = FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_SharelamConj_LinrBgd, 
                                                                              tLegDesc_ShareR_SharelamConj_LinrBgd, tColorLamKchP, tMarkerStyle_LinrBgd, tMarkerSize, true);
  const FitValWriterInfo tFVWI_ShareR_SharelamConj_LinrBgd_LamKchM = FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_SharelamConj_LinrBgd, 
                                                                              tLegDesc_ShareR_SharelamConj_LinrBgd, tColorLamKchM, tMarkerStyle_LinrBgd, tMarkerSize, true);
  //-------------------------
  TString tFitInfoTString_ShareR_SharelamConj_StavCf_NoBgd = "_MomResCrctn_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_StavCf_ShareLam_Dualie_ShareRadii";
  TString tLegDesc_ShareR_SharelamConj_StavCf_NoBgd = "3 Res., Share R and #lambda_{Conj}, Stav. (No Bgd)";
  const FitValWriterInfo tFVWI_ShareR_SharelamConj_StavCf_NoBgd_LamKchP = FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_SharelamConj_StavCf_NoBgd, 
                                                                              tLegDesc_ShareR_SharelamConj_StavCf_NoBgd, tColorLamKchP, tMarkerStyle_StavCf, tMarkerSize, true);
  const FitValWriterInfo tFVWI_ShareR_SharelamConj_StavCf_NoBgd_LamKchM = FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_SharelamConj_StavCf_NoBgd, 
                                                                              tLegDesc_ShareR_SharelamConj_StavCf_NoBgd, tColorLamKchM, tMarkerStyle_StavCf, tMarkerSize, true);
  //-------------------------
//################################### Fix lambda
//---------------------------------------------------------------------------------------------------------------------------------
  TString tFitInfoTString_ShareR_Fixlam_PolyBgd      = "_MomResCrctn_NonFlatBgdCrctnPolynomial_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_FixAllLambdaTo1_ShareLam_Dualie_ShareLam_ShareRadii";
  TString tLegDesc_ShareR_Fixlam_PolyBgd = "3 Res., Share R, Fix #lambda, Poly. Bgd";
  const FitValWriterInfo tFVWI_ShareR_Fixlam_PolyBgd_LamKchP = FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Fixlam_PolyBgd, 
                                                                              tLegDesc_ShareR_Fixlam_PolyBgd, tColorLamKchP, tMarkerStyle_FixParam_PolyBgd, tMarkerSize, true);
  const FitValWriterInfo tFVWI_ShareR_Fixlam_PolyBgd_LamKchM = FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Fixlam_PolyBgd, 
                                                                              tLegDesc_ShareR_Fixlam_PolyBgd, tColorLamKchM, tMarkerStyle_FixParam_PolyBgd, tMarkerSize, true);
  //-------------------------
  TString tFitInfoTString_ShareR_Fixlam_LinrBgd      = "_MomResCrctn_NonFlatBgdCrctnLinear_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_FixAllLambdaTo1_ShareLam_Dualie_ShareLam_ShareRadii";
  TString tLegDesc_ShareR_Fixlam_LinrBgd = "3 Res., Share R, Fix #lambda, Linr. Bgd";
  const FitValWriterInfo tFVWI_ShareR_Fixlam_LinrBgd_LamKchP = FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Fixlam_LinrBgd, 
                                                                              tLegDesc_ShareR_Fixlam_LinrBgd, tColorLamKchP, tMarkerStyle_FixParam_LinrBgd, tMarkerSize, true);
  const FitValWriterInfo tFVWI_ShareR_Fixlam_LinrBgd_LamKchM = FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Fixlam_LinrBgd, 
                                                                              tLegDesc_ShareR_Fixlam_LinrBgd, tColorLamKchM, tMarkerStyle_FixParam_LinrBgd, tMarkerSize, true);
  //-------------------------
  TString tFitInfoTString_ShareR_Fixlam_StavCf_NoBgd = "_MomResCrctn_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_StavCf_FixAllLambdaTo1_ShareLam_Dualie_ShareLam_ShareRadii";
  TString tLegDesc_ShareR_Fixlam_StavCf_NoBgd = "3 Res., Share R, Fix #lambda, Stav. (No Bgd)";
  const FitValWriterInfo tFVWI_ShareR_Fixlam_StavCf_NoBgd_LamKchP = FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Fixlam_StavCf_NoBgd, 
                                                                              tLegDesc_ShareR_Fixlam_StavCf_NoBgd, tColorLamKchP, tMarkerStyle_FixParam_StavCf, tMarkerSize, true);
  const FitValWriterInfo tFVWI_ShareR_Fixlam_StavCf_NoBgd_LamKchM = FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Fixlam_StavCf_NoBgd, 
                                                                              tLegDesc_ShareR_Fixlam_StavCf_NoBgd, tColorLamKchM, tMarkerStyle_FixParam_StavCf, tMarkerSize, true);
  //-------------------------

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


  vector<FitValWriterInfo> tFVWIVec_SepR_Seplam = {tFVWI_SepR_Seplam_PolyBgd_LamKchP, tFVWI_SepR_Seplam_LinrBgd_LamKchP, tFVWI_SepR_Seplam_StavCf_NoBgd_LamKchP,
                                                   tFVWI_SepR_Seplam_PolyBgd_ALamKchM, tFVWI_SepR_Seplam_LinrBgd_ALamKchM, tFVWI_SepR_Seplam_StavCf_NoBgd_ALamKchM,
                                                   tFVWI_SepR_Seplam_PolyBgd_LamKchM, tFVWI_SepR_Seplam_LinrBgd_LamKchM, tFVWI_SepR_Seplam_StavCf_NoBgd_LamKchM,
                                                   tFVWI_SepR_Seplam_PolyBgd_ALamKchP, tFVWI_SepR_Seplam_LinrBgd_ALamKchP, tFVWI_SepR_Seplam_StavCf_NoBgd_ALamKchP};

  vector<FitValWriterInfo> tFVWIVec_SepR_Sharelam = {tFVWI_SepR_Sharelam_PolyBgd_LamKchP, tFVWI_SepR_Sharelam_LinrBgd_LamKchP, tFVWI_SepR_Sharelam_StavCf_NoBgd_LamKchP,
                                                     tFVWI_SepR_Sharelam_PolyBgd_LamKchM, tFVWI_SepR_Sharelam_LinrBgd_LamKchM, tFVWI_SepR_Sharelam_StavCf_NoBgd_LamKchM};

  vector<FitValWriterInfo> tFVWIVec_ShareR_Sharelam = {tFVWI_ShareR_Sharelam_PolyBgd_LamKchP, tFVWI_ShareR_Sharelam_LinrBgd_LamKchP, tFVWI_ShareR_Sharelam_StavCf_NoBgd_LamKchP,
                                                       tFVWI_ShareR_Sharelam_PolyBgd_LamKchM, tFVWI_ShareR_Sharelam_LinrBgd_LamKchM, tFVWI_ShareR_Sharelam_StavCf_NoBgd_LamKchM};

  vector<FitValWriterInfo> tFVWIVec_ShareR_SharelamConj = {tFVWI_ShareR_SharelamConj_PolyBgd_LamKchP, tFVWI_ShareR_SharelamConj_LinrBgd_LamKchP, tFVWI_ShareR_SharelamConj_StavCf_NoBgd_LamKchP,
                                                           tFVWI_ShareR_SharelamConj_PolyBgd_LamKchM, tFVWI_ShareR_SharelamConj_LinrBgd_LamKchM, tFVWI_ShareR_SharelamConj_StavCf_NoBgd_LamKchM};


  vector<FitValWriterInfo> tFVWIVec_SharevsSepR = {FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Sharelam_PolyBgd, 
                                                                    tLegDesc_ShareR_Sharelam_PolyBgd, tColorLamKchP, 20, tMarkerSize, true), 
                                                   FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Sharelam_PolyBgd, 
                                                                    tLegDesc_SepR_Sharelam_PolyBgd, tColorLamKchP, 21, tMarkerSize, false), 
                                                   FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Seplam_PolyBgd, 
                                                                    tLegDesc_SepR_Seplam_PolyBgd, tColorLamKchP, 34, tMarkerSize, false), 
                                                   FitValWriterInfo(kALamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Seplam_PolyBgd, 
                                                                    tLegDesc_SepR_Seplam_PolyBgd, tColorLamKchP, 28, tMarkerSize, false), 
                                                   FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Sharelam_PolyBgd, 
                                                                    tLegDesc_ShareR_Sharelam_PolyBgd, tColorLamKchM, 20, tMarkerSize, true), 
                                                   FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Sharelam_PolyBgd, 
                                                                    tLegDesc_SepR_Sharelam_PolyBgd, tColorLamKchM, 21, tMarkerSize, false), 
                                                   FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Seplam_PolyBgd, 
                                                                    tLegDesc_SepR_Seplam_PolyBgd, tColorLamKchM, 34, tMarkerSize, false), 
                                                   FitValWriterInfo(kALamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Seplam_PolyBgd, 
                                                                    tLegDesc_SepR_Seplam_PolyBgd, tColorLamKchM, 28, tMarkerSize, false)};


//-----------------------------------------------------------------

  vector<FitValWriterInfo> tFVWIVec_FreevsFixlam_SepR = {tFVWI_SepR_Sharelam_PolyBgd_LamKchP, tFVWI_SepR_Sharelam_LinrBgd_LamKchP, tFVWI_SepR_Sharelam_StavCf_NoBgd_LamKchP,
                                                         tFVWI_SepR_Sharelam_PolyBgd_LamKchM, tFVWI_SepR_Sharelam_LinrBgd_LamKchM, tFVWI_SepR_Sharelam_StavCf_NoBgd_LamKchM,
                                                         FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Fixlam_PolyBgd, 
                                                                          tLegDesc_SepR_Fixlam_PolyBgd, tColorLamKchP, 24, tMarkerSize, false), 
                                                         FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Fixlam_LinrBgd, 
                                                                          tLegDesc_SepR_Fixlam_LinrBgd, tColorLamKchP, 25, tMarkerSize, false), 
                                                         FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Fixlam_StavCf_NoBgd, 
                                                                          tLegDesc_SepR_Fixlam_StavCf_NoBgd, tColorLamKchP, 28, tMarkerSize, false), 
                                                         FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Fixlam_PolyBgd, 
                                                                          tLegDesc_SepR_Fixlam_PolyBgd, tColorLamKchM, 24, tMarkerSize, false), 
                                                         FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Fixlam_LinrBgd, 
                                                                          tLegDesc_SepR_Fixlam_LinrBgd, tColorLamKchM, 25, tMarkerSize, false), 
                                                         FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Fixlam_StavCf_NoBgd, 
                                                                          tLegDesc_SepR_Fixlam_StavCf_NoBgd, tColorLamKchM, 28, tMarkerSize, false)};

  vector<FitValWriterInfo> tFVWIVec_FreevsFixlam_SepR_NoStav = {tFVWI_SepR_Sharelam_PolyBgd_LamKchP, tFVWI_SepR_Sharelam_LinrBgd_LamKchP,
                                                                tFVWI_SepR_Sharelam_PolyBgd_LamKchM, tFVWI_SepR_Sharelam_LinrBgd_LamKchM,
                                                               FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Fixlam_PolyBgd, 
                                                                                tLegDesc_SepR_Fixlam_PolyBgd, tColorLamKchP, 24, tMarkerSize, false), 
                                                               FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Fixlam_LinrBgd, 
                                                                                tLegDesc_SepR_Fixlam_LinrBgd, tColorLamKchP, 25, tMarkerSize, false), 
                                                               FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Fixlam_PolyBgd, 
                                                                                tLegDesc_SepR_Fixlam_PolyBgd, tColorLamKchM, 24, tMarkerSize, false), 
                                                               FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Fixlam_LinrBgd, 
                                                                                tLegDesc_SepR_Fixlam_LinrBgd, tColorLamKchM, 25, tMarkerSize, false)};

  vector<FitValWriterInfo> tFVWIVec_FreevsFixlam_SepR_PolyBgd = {tFVWI_SepR_Sharelam_PolyBgd_LamKchP, 
                                                                 tFVWI_SepR_Sharelam_PolyBgd_LamKchM,
                                                                 FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Fixlam_PolyBgd, 
                                                                                  tLegDesc_SepR_Fixlam_PolyBgd, tColorLamKchP, 24, tMarkerSize, false), 
                                                                 FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Fixlam_PolyBgd, 
                                                                                  tLegDesc_SepR_Fixlam_PolyBgd, tColorLamKchM, 24, tMarkerSize, false)};

  vector<FitValWriterInfo> tFVWIVec_FreevsFixlam_SepR_LinrBgd = {tFVWI_SepR_Sharelam_LinrBgd_LamKchP, 
                                                                 tFVWI_SepR_Sharelam_LinrBgd_LamKchM,
                                                                 FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Fixlam_LinrBgd, 
                                                                                  tLegDesc_SepR_Fixlam_LinrBgd, tColorLamKchP, 24, tMarkerSize, false), 
                                                                 FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Fixlam_LinrBgd, 
                                                                                  tLegDesc_SepR_Fixlam_LinrBgd, tColorLamKchM, 24, tMarkerSize, false)};

  vector<FitValWriterInfo> tFVWIVec_FreevsFixlam_SepR_StavCf_NoBgd = {tFVWI_SepR_Sharelam_StavCf_NoBgd_LamKchP, 
                                                                 tFVWI_SepR_Sharelam_StavCf_NoBgd_LamKchM,
                                                                 FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Fixlam_StavCf_NoBgd, 
                                                                                  tLegDesc_SepR_Fixlam_StavCf_NoBgd, tColorLamKchP, 24, tMarkerSize, false), 
                                                                 FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Fixlam_StavCf_NoBgd, 
                                                                                  tLegDesc_SepR_Fixlam_StavCf_NoBgd, tColorLamKchM, 24, tMarkerSize, false)};

//-----------------------------------------------------------------

  vector<FitValWriterInfo> tFVWIVec_FreevsFixlam_ShareR = {tFVWI_ShareR_Sharelam_PolyBgd_LamKchP, tFVWI_ShareR_Sharelam_LinrBgd_LamKchP, tFVWI_ShareR_Sharelam_StavCf_NoBgd_LamKchP,
                                                           tFVWI_ShareR_Sharelam_PolyBgd_LamKchM, tFVWI_ShareR_Sharelam_LinrBgd_LamKchM, tFVWI_ShareR_Sharelam_StavCf_NoBgd_LamKchM,
                                                           FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Fixlam_PolyBgd, 
                                                                            tLegDesc_ShareR_Fixlam_PolyBgd, tColorLamKchP, 24, tMarkerSize, true), 
                                                           FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Fixlam_LinrBgd, 
                                                                            tLegDesc_ShareR_Fixlam_LinrBgd, tColorLamKchP, 25, tMarkerSize, true), 
                                                           FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Fixlam_StavCf_NoBgd, 
                                                                            tLegDesc_ShareR_Fixlam_StavCf_NoBgd, tColorLamKchP, 28, tMarkerSize, true), 
                                                           FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Fixlam_PolyBgd, 
                                                                            tLegDesc_ShareR_Fixlam_PolyBgd, tColorLamKchM, 24, tMarkerSize, true), 
                                                           FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Fixlam_LinrBgd, 
                                                                            tLegDesc_ShareR_Fixlam_LinrBgd, tColorLamKchM, 25, tMarkerSize, true), 
                                                           FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Fixlam_StavCf_NoBgd, 
                                                                            tLegDesc_ShareR_Fixlam_StavCf_NoBgd, tColorLamKchM, 28, tMarkerSize, true)};

  vector<FitValWriterInfo> tFVWIVec_FreevsFixlam_ShareR_NoStav = {tFVWI_ShareR_Sharelam_PolyBgd_LamKchP, tFVWI_ShareR_Sharelam_LinrBgd_LamKchP,
                                                                  tFVWI_ShareR_Sharelam_PolyBgd_LamKchM, tFVWI_ShareR_Sharelam_LinrBgd_LamKchM,
                                                                  FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Fixlam_PolyBgd, 
                                                                                   tLegDesc_ShareR_Fixlam_PolyBgd, tColorLamKchP, 24, tMarkerSize, true), 
                                                                  FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Fixlam_LinrBgd, 
                                                                                   tLegDesc_ShareR_Fixlam_LinrBgd, tColorLamKchP, 25, tMarkerSize, true), 
                                                                  FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Fixlam_PolyBgd, 
                                                                                   tLegDesc_ShareR_Fixlam_PolyBgd, tColorLamKchM, 24, tMarkerSize, true), 
                                                                  FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Fixlam_LinrBgd, 
                                                                                   tLegDesc_ShareR_Fixlam_LinrBgd, tColorLamKchM, 25, tMarkerSize, true)};

  vector<FitValWriterInfo> tFVWIVec_FreevsFixlam_ShareR_PolyBgd = {tFVWI_ShareR_Sharelam_PolyBgd_LamKchP,
                                                                   tFVWI_ShareR_Sharelam_PolyBgd_LamKchM,
                                                                   FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Fixlam_PolyBgd, 
                                                                                    tLegDesc_ShareR_Fixlam_PolyBgd, tColorLamKchP, 24, tMarkerSize, true), 
                                                                   FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Fixlam_PolyBgd, 
                                                                                    tLegDesc_ShareR_Fixlam_PolyBgd, tColorLamKchM, 24, tMarkerSize, true)};

  vector<FitValWriterInfo> tFVWIVec_FreevsFixlam_ShareR_LinrBgd = {tFVWI_ShareR_Sharelam_LinrBgd_LamKchP,
                                                                   tFVWI_ShareR_Sharelam_LinrBgd_LamKchM,
                                                                   FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Fixlam_LinrBgd, 
                                                                                    tLegDesc_ShareR_Fixlam_LinrBgd, tColorLamKchP, 24, tMarkerSize, true), 
                                                                   FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Fixlam_LinrBgd, 
                                                                                    tLegDesc_ShareR_Fixlam_LinrBgd, tColorLamKchM, 24, tMarkerSize, true)};

  vector<FitValWriterInfo> tFVWIVec_FreevsFixlam_ShareR_StavCf_NoBgd = {tFVWI_ShareR_Sharelam_StavCf_NoBgd_LamKchP,
                                                                   tFVWI_ShareR_Sharelam_StavCf_NoBgd_LamKchM,
                                                                   FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Fixlam_StavCf_NoBgd, 
                                                                                    tLegDesc_ShareR_Fixlam_StavCf_NoBgd, tColorLamKchP, 24, tMarkerSize, true), 
                                                                   FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Fixlam_StavCf_NoBgd, 
                                                                                    tLegDesc_ShareR_Fixlam_StavCf_NoBgd, tColorLamKchM, 24, tMarkerSize, true)};



//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
/*
  static TString BuildFitInfoTString(bool aApplyMomResCorrection, bool aApplyNonFlatBackgroundCorrection, NonFlatBgdFitType aNonFlatBgdFitType, 
                                     IncludeResidualsType aIncludeResidualsType, ResPrimMaxDecayType aResPrimMaxDecayType=k4fm, 
                                     ChargedResidualsType aChargedResidualsType=kUseXiDataAndCoulombOnlyInterp, bool aFixD0=false,
                                     bool aUseStavCf=false, bool aFixAllLambdaTo1=false, bool aFixAllNormTo1=false, bool aFixRadii=false, bool aFixAllScattParams=false, 
                                     bool aShareLambdaParams=false, bool aAllShareSingleLambdaParam=false, bool aUsemTScalingOfResidualRadii=false, bool aIsDualie=false, 
                                     bool aDualieShareLambda=false, bool aDualieShareRadii=false);
*/


  TString tFitInfoTString_LamKch_3Res_PolyBgd = FitValuesWriter::BuildFitInfoTString(true, true, kPolynomial, 
                                                                                     kInclude3Residuals, k4fm, 
                                                                                     kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                     false, false, false, false, false, 
                                                                                     true, false, false, true, 
                                                                                     true, true);

  TString tFitInfoTString_LamKch_10Res_PolyBgd = FitValuesWriter::BuildFitInfoTString(true, true, kPolynomial, 
                                                                                     kInclude10Residuals, k4fm, 
                                                                                     kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                     false, false, false, false, false, 
                                                                                     true, false, false, true, 
                                                                                     true, true);

  TString tFitInfoTString_LamKch_NoRes_PolyBgd = FitValuesWriter::BuildFitInfoTString(true, true, kPolynomial, 
                                                                                     kIncludeNoResiduals, k4fm, 
                                                                                     kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                     false, false, false, false, false, 
                                                                                     true, false, false, true, 
                                                                                     true, true);

  //-------------------------------------------------------------------------

  TString tFitInfoTString_LamK0_3Res_PolyBgd = FitValuesWriter::BuildFitInfoTString(true, true, kPolynomial, 
                                                                                    kInclude3Residuals, k4fm, 
                                                                                    kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                    false, false, false, false, false, 
                                                                                    false, true, false, false, 
                                                                                    false, false);

  TString tFitInfoTString_LamK0_10Res_PolyBgd = FitValuesWriter::BuildFitInfoTString(true, true, kPolynomial, 
                                                                                    kInclude10Residuals, k4fm, 
                                                                                    kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                    false, false, false, false, false, 
                                                                                    false, true, false, false, 
                                                                                    false, false);

  TString tFitInfoTString_LamK0_NoRes_PolyBgd = FitValuesWriter::BuildFitInfoTString(true, true, kPolynomial, 
                                                                                    kIncludeNoResiduals, k4fm, 
                                                                                    kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                    false, false, false, false, false, 
                                                                                    false, true, false, false, 
                                                                                    false, false);

  //---------------

  TString tFitInfoTString_LamK0_3Res_LinrBgd = FitValuesWriter::BuildFitInfoTString(true, true, kLinear, 
                                                                                    kInclude3Residuals, k4fm, 
                                                                                    kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                    false, false, false, false, false, 
                                                                                    false, true, false, false, 
                                                                                    false, false);

  TString tFitInfoTString_LamK0_10Res_LinrBgd = FitValuesWriter::BuildFitInfoTString(true, true, kLinear, 
                                                                                    kInclude10Residuals, k4fm, 
                                                                                    kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                    false, false, false, false, false, 
                                                                                    false, true, false, false, 
                                                                                    false, false);

  TString tFitInfoTString_LamK0_NoRes_LinrBgd = FitValuesWriter::BuildFitInfoTString(true, true, kLinear, 
                                                                                    kIncludeNoResiduals, k4fm, 
                                                                                    kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                    false, false, false, false, false, 
                                                                                    false, true, false, false, 
                                                                                    false, false);

  //---------------

  TString tFitInfoTString_LamK0_3Res_StavCf = FitValuesWriter::BuildFitInfoTString(true, false, kLinear, 
                                                                                    kInclude3Residuals, k4fm, 
                                                                                    kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                    true, false, false, false, false, 
                                                                                    false, true, false, false, 
                                                                                    false, false);

  TString tFitInfoTString_LamK0_10Res_StavCf = FitValuesWriter::BuildFitInfoTString(true, false, kLinear, 
                                                                                    kInclude10Residuals, k4fm, 
                                                                                    kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                    true, false, false, false, false, 
                                                                                    false, true, false, false, 
                                                                                    false, false);

  TString tFitInfoTString_LamK0_NoRes_StavCf = FitValuesWriter::BuildFitInfoTString(true, false, kLinear, 
                                                                                    kIncludeNoResiduals, k4fm, 
                                                                                    kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                    true, false, false, false, false, 
                                                                                    false, true, false, false, 
                                                                                    false, false);

  //-------------------------------------------------------------------------


  vector<FitValWriterInfo> tFVWIVec_CompNumRes_ShareR_PolyBgd = {FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_LamKch_3Res_PolyBgd, 
                                                                                  "3 Res., Share R and #lambda, Poly. Bgd", tColorLamKchP, 34, tMarkerSize, true), 
                                                                 FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_LamKch_10Res_PolyBgd, 
                                                                                  "10 Res., Share R and #lambda, Poly. Bgd", tColorLamKchP, 47, tMarkerSize, true), 
                                                                 FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_LamKch_NoRes_PolyBgd, 
                                                                                  "No Res., Share R and #lambda, Poly. Bgd", tColorLamKchP, 20, tMarkerSize, true), 

                                                                 FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_LamKch_3Res_PolyBgd, 
                                                                                  "3 Res., Share R and #lambda, Poly. Bgd", tColorLamKchM, 34, tMarkerSize, true), 
                                                                 FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_LamKch_10Res_PolyBgd, 
                                                                                  "10 Res., Share R and #lambda, Poly. Bgd", tColorLamKchM, 47, tMarkerSize, true), 
                                                                 FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_LamKch_NoRes_PolyBgd, 
                                                                                  "No Res., Share R and #lambda, Poly. Bgd", tColorLamKchM, 20, tMarkerSize, true), 

                                                                 FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, tFitInfoTString_LamK0_3Res_PolyBgd, 
                                                                                  "3 Res., Share R and #lambda, Poly. Bgd", tColorLamK0, 34, tMarkerSize, false), 
                                                                 FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, tFitInfoTString_LamK0_10Res_PolyBgd, 
                                                                                  "10 Res., Share R and #lambda, Poly. Bgd", tColorLamK0, 47, tMarkerSize, false), 
                                                                 FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, tFitInfoTString_LamK0_NoRes_PolyBgd, 
                                                                                  "No Res., Share R and #lambda, Poly. Bgd", tColorLamK0, 20, tMarkerSize, false)};


//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Comparison plots for results sections of analysis note  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  vector<FitValWriterInfo> tFVWIVec_Comp3An_3Res = {FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_LamKch_3Res_PolyBgd, 
                                                                     "3 Residuals (Suppress Markers)", tColorLamKchP, 20, tMarkerSize, true), 
                                                    FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_LamKch_3Res_PolyBgd, 
                                                                     "#LambdaK^{#pm}: Share R and #lambda, Poly. Bgd (Suppress Markers)", tColorLamKchM, 20, tMarkerSize, true), 
                                                    FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, tFitInfoTString_LamK0_3Res_PolyBgd, 
                                                                     "#LambdaK^{0}_{S}: Single #lambda, Poly. Bgd (Suppress Markers)", tColorLamK0, 20, tMarkerSize, false)};
/*
  vector<FitValWriterInfo> tFVWIVec_Comp3An_3Res = {FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_LamKch_3Res_PolyBgd, 
                                                                     "3 Residuals (Suppress Markers)", tColorLamKchP, 20, tMarkerSize, true), 
                                                    FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_LamKch_3Res_PolyBgd, 
                                                                     "#LambdaK^{#pm}: Share R and #lambda, Poly. Bgd (Suppress Markers)", tColorLamKchM, 20, tMarkerSize, true), 
                                                    FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, tFitInfoTString_LamK0_3Res_LinrBgd, 
                                                                     "#LambdaK^{0}_{S}: Single #lambda, Linr. Bgd (Suppress Markers)", tColorLamK0, 20, tMarkerSize, false)};
*/



  vector<FitValWriterInfo> tFVWIVec_Comp3An_10Res = {FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_LamKch_10Res_PolyBgd, 
                                                                      "3 Residuals (Suppress Markers)", tColorLamKchP, 20, tMarkerSize, true), 
                                                    FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_LamKch_10Res_PolyBgd, 
                                                                      "#LambdaK^{#pm}: Share R and #lambda, Poly. Bgd (Suppress Markers)", tColorLamKchM, 20, tMarkerSize, true), 
                                                    FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, tFitInfoTString_LamK0_10Res_PolyBgd, 
                                                                      "#LambdaK^{0}_{S}: Single #lambda, Poly. Bgd (Suppress Markers)", tColorLamK0, 20, tMarkerSize, false)};
/*
  vector<FitValWriterInfo> tFVWIVec_Comp3An_10Res = {FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_LamKch_10Res_PolyBgd, 
                                                                      "3 Residuals (Suppress Markers)", tColorLamKchP, 20, tMarkerSize, true), 
                                                    FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_LamKch_10Res_PolyBgd, 
                                                                     "#LambdaK^{#pm}: Share R and #lambda, Poly. Bgd (Suppress Markers)", tColorLamKchM, 20, tMarkerSize, true), 
                                                    FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, tFitInfoTString_LamK0_10Res_LinrBgd, 
                                                                     "#LambdaK^{0}_{S}: Single #lambda, Linr. Bgd (Suppress Markers)", tColorLamK0, 20, tMarkerSize, false)};
*/


  vector<FitValWriterInfo> tFVWIVec_Comp3An_NoRes = {FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_LamKch_NoRes_PolyBgd, 
                                                                      "3 Residuals (Suppress Markers)", tColorLamKchP, 20, tMarkerSize, true), 
                                                    FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_LamKch_NoRes_PolyBgd, 
                                                                     "#LambdaK^{#pm}: Share R and #lambda, Poly. Bgd (Suppress Markers)", tColorLamKchM, 20, tMarkerSize, true), 
                                                    FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, tFitInfoTString_LamK0_NoRes_PolyBgd, 
                                                                     "#LambdaK^{0}_{S}: Single #lambda, Poly. Bgd (Suppress Markers)", tColorLamK0, 20, tMarkerSize, false)};
/*
  vector<FitValWriterInfo> tFVWIVec_Comp3An_NoRes = {FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_LamKch_NoRes_PolyBgd, 
                                                                      "3 Residuals (Suppress Markers)", tColorLamKchP, 20, tMarkerSize, true), 
                                                    FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_LamKch_NoRes_PolyBgd, 
                                                                     "#LambdaK^{#pm}: Share R and #lambda, Poly. Bgd (Suppress Markers)", tColorLamKchM, 20, tMarkerSize, true), 
                                                    FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, tFitInfoTString_LamK0_NoRes_LinrBgd, 
                                                                     "#LambdaK^{0}_{S}: Single #lambda, Linr. Bgd (Suppress Markers)", tColorLamK0, 20, tMarkerSize, false)};
*/




//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Comparison plots for comparison section of results  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

//--------- #Residuals used (Poly Bgd, Share R, Share lam)
  vector<FitValWriterInfo> tFVWIVec_CompNumRes = {FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_LamKch_3Res_PolyBgd, 
                                                                   "3 Residuals", tColorLamKchP, 34, tMarkerSize, true), 
                                                  FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_LamKch_10Res_PolyBgd, 
                                                                   "10 Residuals", tColorLamKchP, 47, tMarkerSize, true), 
                                                  FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_LamKch_NoRes_PolyBgd, 
                                                                   "No Residuals", tColorLamKchP, 20, tMarkerSize, true), 

                                                  FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_LamKch_3Res_PolyBgd, 
                                                                   "3 Residuals", tColorLamKchM, 34, tMarkerSize, true), 
                                                  FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_LamKch_10Res_PolyBgd, 
                                                                   "10 Residuals", tColorLamKchM, 47, tMarkerSize, true), 
                                                  FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_LamKch_NoRes_PolyBgd, 
                                                                   "No Residuals", tColorLamKchM, 20, tMarkerSize, true), 

                                                  FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, tFitInfoTString_LamK0_3Res_PolyBgd, 
                                                                   "3 Residuals", tColorLamK0, 34, tMarkerSize, false), 
                                                  FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, tFitInfoTString_LamK0_10Res_PolyBgd, 
                                                                   "10 Residuals", tColorLamK0, 47, tMarkerSize, false), 
                                                  FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, tFitInfoTString_LamK0_NoRes_PolyBgd, 
                                                                   "No Residuals", tColorLamK0, 20, tMarkerSize, false)};

//--------- Background treatment (3Res, Share R, Share lam)
  vector<FitValWriterInfo> tFVWIVec_CompBgdTreatment = {FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Sharelam_PolyBgd, 
                                                                         "Polynomial Bgd.", tColorLamKchP, tMarkerStyle_PolyBgd, tMarkerSize, true), 
                                                        FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Sharelam_LinrBgd, 
                                                                         "Linear Bgd.", tColorLamKchP, tMarkerStyle_LinrBgd, tMarkerSize, true), 
                                                        FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Sharelam_StavCf_NoBgd, 
                                                                         "Stav Cf (Assumed No Bgd.)", tColorLamKchP, tMarkerStyle_StavCf, tMarkerSize, true), 
      
                                                        FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Sharelam_PolyBgd, 
                                                                         "Polynomial Bgd.", tColorLamKchM, tMarkerStyle_PolyBgd, tMarkerSize, true), 
                                                        FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Sharelam_LinrBgd, 
                                                                         "Linear Bgd.", tColorLamKchM, tMarkerStyle_LinrBgd, tMarkerSize, true), 
                                                        FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Sharelam_StavCf_NoBgd, 
                                                                         "Stav Cf (Assumed No Bgd.)", tColorLamKchM, tMarkerStyle_StavCf, tMarkerSize, true),

                                                        FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, tFitInfoTString_LamK0_3Res_PolyBgd, 
                                                                         "Polynomial Bgd.", tColorLamK0, tMarkerStyle_PolyBgd, tMarkerSize, false), 
                                                        FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, tFitInfoTString_LamK0_3Res_LinrBgd, 
                                                                         "Linear Bgd.", tColorLamK0, tMarkerStyle_LinrBgd, tMarkerSize, false), 
                                                        FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, tFitInfoTString_LamK0_3Res_StavCf, 
                                                                         "Stav Cf (Assumed No Bgd.)", tColorLamK0, tMarkerStyle_StavCf, tMarkerSize, false)};

//--------- Free vs fixed lam, sharing radii (3Res, Poly Bgd, Share R, Share lam)
  TString tFitInfoTString_LamK0_3Res_PolyBgd_Fixlam = FitValuesWriter::BuildFitInfoTString(true, true, kPolynomial, 
                                                                                           kInclude3Residuals, k4fm, 
                                                                                           kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                           false, true, false, false, false, 
                                                                                           false, false, false, false, 
                                                                                           false, false);

  vector<FitValWriterInfo> tFVWIVec_CompFreevsFixedlam_ShareR = {FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Sharelam_PolyBgd, 
                                                                           "Free #lambda", tColorLamKchP, tMarkerStyle_PolyBgd, tMarkerSize, true),
                                                                 FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Fixlam_PolyBgd, 
                                                                                  "#lambda = 1", tColorLamKchP, 24, tMarkerSize, true), 

                                                                 FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Sharelam_PolyBgd, 
                                                                                  "Free #lambda", tColorLamKchM, tMarkerStyle_PolyBgd, tMarkerSize, true),
                                                                 FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Fixlam_PolyBgd, 
                                                                                  "#lambda = 1", tColorLamKchM, 24, tMarkerSize, true),

                                                                 FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, tFitInfoTString_LamK0_3Res_PolyBgd, 
                                                                                  "Free #lambda", tColorLamK0, tMarkerStyle_PolyBgd, tMarkerSize, false),
                                                                 FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, tFitInfoTString_LamK0_3Res_PolyBgd_Fixlam, 
                                                                                  "#lambda = 1", tColorLamK0, 24, tMarkerSize, false)};



//--------- Free vs fixed lam, separate radii (3Res, Poly Bgd, Sep R, Share lamconj)
  vector<FitValWriterInfo> tFVWIVec_CompFreevsFixedlam_SepR = {FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Sharelam_PolyBgd, 
                                                                                "Free #lambda", tColorLamKchP, tMarkerStyle_PolyBgd, tMarkerSize, false), 
                                                               FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Fixlam_PolyBgd, 
                                                                                "#lambda = 1", tColorLamKchP, 24, tMarkerSize, false), 

                                                               FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Sharelam_PolyBgd, 
                                                                                "Free #lambda", tColorLamKchM, tMarkerStyle_PolyBgd, tMarkerSize, false),
                                                               FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Fixlam_PolyBgd, 
                                                                                "#lambda = 1", tColorLamKchM, 24, tMarkerSize, false),

                                                               FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, tFitInfoTString_LamK0_3Res_PolyBgd, 
                                                                                "Free #lambda", tColorLamK0, tMarkerStyle_PolyBgd, tMarkerSize, false),
                                                               FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, tFitInfoTString_LamK0_3Res_PolyBgd_Fixlam, 
                                                                                "#lambda = 1", tColorLamK0, 24, tMarkerSize, false)};

//--------- Free vs fixed lam, separate radii, unique lambda (3Res, Poly Bgd, Sep R, Share lamconj)
  vector<FitValWriterInfo> tFVWIVec_CompFreevsFixedlam_SepR_Seplam = {FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Seplam_PolyBgd, 
                                                                                "Free #lambda", tColorLamKchP, tMarkerStyle_PolyBgd, tMarkerSize, false), 
                                                               FitValWriterInfo(kALamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Seplam_PolyBgd, 
                                                                                "Free #lambda", tColorLamKchP, tMarkerStyle_PolyBgd, tMarkerSize, false), 
                                                               FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Fixlam_PolyBgd, 
                                                                                "#lambda = 1", tColorLamKchP, 24, tMarkerSize, false), 

                                                               FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Seplam_PolyBgd, 
                                                                                "Free #lambda", tColorLamKchM, tMarkerStyle_PolyBgd, tMarkerSize, false),
                                                               FitValWriterInfo(kALamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Seplam_PolyBgd, 
                                                                                "Free #lambda", tColorLamKchM, tMarkerStyle_PolyBgd, tMarkerSize, false),
                                                               FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Fixlam_PolyBgd, 
                                                                                "#lambda = 1", tColorLamKchM, 24, tMarkerSize, false),

                                                               FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, tFitInfoTString_LamK0_3Res_PolyBgd, 
                                                                                "Free #lambda", tColorLamK0, tMarkerStyle_PolyBgd, tMarkerSize, false),
                                                               FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, tFitInfoTString_LamK0_3Res_PolyBgd_Fixlam, 
                                                                                "#lambda = 1", tColorLamK0, 24, tMarkerSize, false)};





//--------- Shared vs separate R (3Res, Poly Bgd, Share lam)
  vector<FitValWriterInfo> tFVWIVec_CompSharesvsSepR = {FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Sharelam_PolyBgd, 
                                                                         "Shared Radii", tColorLamKchP, 20, tMarkerSize, true), 
                                                        FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Sharelam_PolyBgd, 
                                                                         "Separate Radii", tColorLamKchP, 21, tMarkerSize, false), 

                                                        FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Sharelam_PolyBgd, 
                                                                         "Shared Radii", tColorLamKchM, 20, tMarkerSize, true), 
                                                        FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Sharelam_PolyBgd, 
                                                                         "Separate Radii", tColorLamKchM, 21, tMarkerSize, false)};


//--------- Share lamconj vs unique lam (3Res, Poly Bgd, Sep R)
  vector<FitValWriterInfo> tFVWIVec_CompSharelam_SepR = {FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Sharelam_PolyBgd, 
                                                                          "Share #lambda_{Conj}", tColorLamKchP, 21, tMarkerSize, false), 
                                                         FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Seplam_PolyBgd, 
                                                                          "Unqiue #lambda", tColorLamKchP, 34, tMarkerSize, false), 
                                                         FitValWriterInfo(kALamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Seplam_PolyBgd, 
                                                                          "Unqiue #lambda", tColorLamKchP, 28, tMarkerSize, false), 
 
                                                         FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Sharelam_PolyBgd, 
                                                                          "Share #lambda_{Conj}", tColorLamKchM, 21, tMarkerSize, false), 
                                                         FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Seplam_PolyBgd, 
                                                                          "Unqiue #lambda", tColorLamKchM, 34, tMarkerSize, false), 
                                                         FitValWriterInfo(kALamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Seplam_PolyBgd, 
                                                                          "Unqiue #lambda", tColorLamKchM, 28, tMarkerSize, false)};

//--------- Share lam vs share lamconj vs unique lam (3Res, Poly Bgd, Share R)
  TString tFitInfoTString_ShareR_UniqueLam_PolyBgd      = "_MomResCrctn_NonFlatBgdCrctnPolynomial_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_Dualie_ShareRadii";

  vector<FitValWriterInfo> tFVWIVec_CompSharelam_SharedR = {FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Sharelam_PolyBgd, 
                                                                              "Share #lambda", tColorLamKchP, 20, tMarkerSize, true), 
                                                            FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_SharelamConj_PolyBgd, 
                                                                              "Share #lambda_{Conj}", tColorLamKchP, 21, tMarkerSize, true), 
                                                            FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_UniqueLam_PolyBgd, 
                                                                              "Unique #lambda", tColorLamKchP, 34, tMarkerSize, true),
                                                            FitValWriterInfo(kALamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_UniqueLam_PolyBgd, 
                                                                              "Unique #lambda", tColorLamKchP, 34, tMarkerSize, true),

                                                            FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Sharelam_PolyBgd, 
                                                                              "Share #lambda", tColorLamKchM, 20, tMarkerSize, true), 
                                                            FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_SharelamConj_PolyBgd, 
                                                                              "Share #lambda_{Conj}", tColorLamKchM, 21, tMarkerSize, true), 
                                                            FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_UniqueLam_PolyBgd, 
                                                                              "Unique #lambda", tColorLamKchM, 34, tMarkerSize, true),
                                                            FitValWriterInfo(kALamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_UniqueLam_PolyBgd, 
                                                                              "Unique #lambda", tColorLamKchM, 34, tMarkerSize, true)};


//**************************************************************************************************************************************************************************
  extern void DrawAnalysisStamps(TPad* aPad, vector<AnalysisType> &aAnTypes, double aStartX, double aStartY, double aIncrementX, double aIncrementY, double aTextSize=0.04, int aMarkerStyle=20);
  extern void DrawAnalysisStamps(TPad* aPad, vector<AnalysisType> &aAnTypes, double aStartX, double aStartY, double aIncrementX, double aIncrementY, double aTextSize, vector<int> &aMarkerStyles);
  extern void DrawAnalysisAndConjStamps(TPad* aPad, vector<AnalysisType> &aAnTypes, double aStartX, double aStartY, double aIncrementX, double aIncrementY, double aSecondColumnShiftX, double aTextSize=0.04, int aMarkerStyle=20, int aConjMarkerStyle=24, bool aLamKchCombined=false, bool aLamKchSeparate=true, bool aAllLamKCombined=false);
  extern void DrawFixedRadiiStamps(TPad* aPad, double aStartX, double aStartY, double aIncrementX, double aIncrementY, double aTextSize=0.04, int aMarkerStyle=20);
  extern void SetupReF0vsImF0Axes(TPad* aPad, double aMinReF0=-2., double aMaxReF0=1., double aMinImF0=0., double aMaxImF0=1.5);
  extern void SetupReF0vsImF0AndD0Axes(TPad* aPadReF0vsImF0, TPad* aPadD0, 
                                       double aMinReF0=-2.09, double aMaxReF0=1.12, double aMinImF0=-0.09, double aMaxImF0=1.49,
                                       double aMinD0=-10.5, double aMaxD0=5.5);
  extern void SetupRadiusvsLambdaAxes(TPad* aPad, double aMinR=0., double aMaxR=8., double aMinLam=0., double aMaxLam=2.49);
  extern bool MarkerStylesConsistent(int aMarkerStyle1, int aMarkerStyle2);
  extern bool DescriptorAlreadyIncluded(vector<TString> &aUsedDescriptors, vector<int> &aUsedMarkerStyles, TString aNewDescriptor, int aNewMarkerStyle);
  extern TString StripSuppressMarkersFlat(TString aString);
  extern TCanvas* CompareImF0vsReF0(vector<FitValWriterInfo> &aFitValWriterInfo, bool aDrawPredictions=false, TString aCanNameMod="", bool aSuppressDescs=false, bool aSuppressAnStamps=false);
  extern TCanvas* CompareLambdavsRadius(vector<FitValWriterInfo> &aFitValWriterInfo, CentralityType aCentType, TString aCanNameMod="", bool aSuppressDescs=false, bool aSuppressAnStamps=false);
  extern TCanvas* CompareAll(vector<FitValWriterInfo> &aFitValWriterInfo, bool aDrawPredictions=false, TString aCanNameMod="");

#endif
