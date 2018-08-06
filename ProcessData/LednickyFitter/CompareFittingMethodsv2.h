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
                                                                              tLegDesc_SepR_Seplam_PolyBgd, tColorLamKchP, tMarkerStyle_PolyBgd, tMarkerSize);
  const FitValWriterInfo tFVWI_SepR_Seplam_PolyBgd_ALamKchM = FitValWriterInfo(kALamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Seplam_PolyBgd, 
                                                                              tLegDesc_SepR_Seplam_PolyBgd, tColorLamKchP, tConjMarkerStyle_PolyBgd, tMarkerSize);
  const FitValWriterInfo tFVWI_SepR_Seplam_PolyBgd_LamKchM = FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Seplam_PolyBgd, 
                                                                              tLegDesc_SepR_Seplam_PolyBgd, tColorLamKchM, tMarkerStyle_PolyBgd, tMarkerSize);
  const FitValWriterInfo tFVWI_SepR_Seplam_PolyBgd_ALamKchP = FitValWriterInfo(kALamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Seplam_PolyBgd, 
                                                                              tLegDesc_SepR_Seplam_PolyBgd, tColorLamKchM, tConjMarkerStyle_PolyBgd, tMarkerSize);
  //-------------------------
  TString tFitInfoTString_SepR_Seplam_LinrBgd      = "_MomResCrctn_NonFlatBgdCrctnLinear_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly";
  TString tLegDesc_SepR_Seplam_LinrBgd = "3 Res., Linr. Bgd";
  const FitValWriterInfo tFVWI_SepR_Seplam_LinrBgd_LamKchP = FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Seplam_LinrBgd, 
                                                                              tLegDesc_SepR_Seplam_LinrBgd, tColorLamKchP, tMarkerStyle_LinrBgd, tMarkerSize);
  const FitValWriterInfo tFVWI_SepR_Seplam_LinrBgd_ALamKchM = FitValWriterInfo(kALamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Seplam_LinrBgd, 
                                                                              tLegDesc_SepR_Seplam_LinrBgd, tColorLamKchP, tConjMarkerStyle_LinrBgd, tMarkerSize);
  const FitValWriterInfo tFVWI_SepR_Seplam_LinrBgd_LamKchM = FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Seplam_LinrBgd, 
                                                                              tLegDesc_SepR_Seplam_LinrBgd, tColorLamKchM, tMarkerStyle_LinrBgd, tMarkerSize);
  const FitValWriterInfo tFVWI_SepR_Seplam_LinrBgd_ALamKchP = FitValWriterInfo(kALamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Seplam_LinrBgd, 
                                                                              tLegDesc_SepR_Seplam_LinrBgd, tColorLamKchM, tConjMarkerStyle_LinrBgd, tMarkerSize);
  //-------------------------
  TString tFitInfoTString_SepR_Seplam_StavCf_NoBgd = "_MomResCrctn_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_StavCf";
  TString tLegDesc_SepR_Seplam_StavCf_NoBgd = "3 Res., Stav. (No Bgd)";
  const FitValWriterInfo tFVWI_SepR_Seplam_StavCf_NoBgd_LamKchP = FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Seplam_StavCf_NoBgd, 
                                                                              tLegDesc_SepR_Seplam_StavCf_NoBgd, tColorLamKchP, tMarkerStyle_StavCf, tMarkerSize);
  const FitValWriterInfo tFVWI_SepR_Seplam_StavCf_NoBgd_ALamKchM = FitValWriterInfo(kALamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Seplam_StavCf_NoBgd, 
                                                                              tLegDesc_SepR_Seplam_StavCf_NoBgd, tColorLamKchP, tConjMarkerStyle_StavCf, tMarkerSize);
  const FitValWriterInfo tFVWI_SepR_Seplam_StavCf_NoBgd_LamKchM = FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Seplam_StavCf_NoBgd, 
                                                                              tLegDesc_SepR_Seplam_StavCf_NoBgd, tColorLamKchM, tMarkerStyle_StavCf, tMarkerSize);
  const FitValWriterInfo tFVWI_SepR_Seplam_StavCf_NoBgd_ALamKchP = FitValWriterInfo(kALamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Seplam_StavCf_NoBgd, 
                                                                              tLegDesc_SepR_Seplam_StavCf_NoBgd, tColorLamKchM, tConjMarkerStyle_StavCf, tMarkerSize);
  //-------------------------
//---------------------------------------------------------------------------------------------------------------------------------
  TString tFitInfoTString_SepR_Sharelam_PolyBgd      = "_MomResCrctn_NonFlatBgdCrctnPolynomial_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_ShareLam";
  TString tLegDesc_SepR_Sharelam_PolyBgd = "3 Res., Share #lambda, Poly. Bgd";
  const FitValWriterInfo tFVWI_SepR_Sharelam_PolyBgd_LamKchP = FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Sharelam_PolyBgd, 
                                                                              tLegDesc_SepR_Sharelam_PolyBgd, tColorLamKchP, tMarkerStyle_PolyBgd, tMarkerSize);
  const FitValWriterInfo tFVWI_SepR_Sharelam_PolyBgd_LamKchM = FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Sharelam_PolyBgd, 
                                                                              tLegDesc_SepR_Sharelam_PolyBgd, tColorLamKchM, tMarkerStyle_PolyBgd, tMarkerSize);
  //-------------------------
  TString tFitInfoTString_SepR_Sharelam_LinrBgd      = "_MomResCrctn_NonFlatBgdCrctnLinear_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_ShareLam";
  TString tLegDesc_SepR_Sharelam_LinrBgd = "3 Res., Share #lambda, Linr. Bgd";
  const FitValWriterInfo tFVWI_SepR_Sharelam_LinrBgd_LamKchP = FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Sharelam_LinrBgd, 
                                                                              tLegDesc_SepR_Sharelam_LinrBgd, tColorLamKchP, tMarkerStyle_LinrBgd, tMarkerSize);
  const FitValWriterInfo tFVWI_SepR_Sharelam_LinrBgd_LamKchM = FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Sharelam_LinrBgd, 
                                                                              tLegDesc_SepR_Sharelam_LinrBgd, tColorLamKchM, tMarkerStyle_LinrBgd, tMarkerSize);
  //-------------------------
  TString tFitInfoTString_SepR_Sharelam_StavCf_NoBgd = "_MomResCrctn_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_StavCf_ShareLam";
  TString tLegDesc_SepR_Sharelam_StavCf_NoBgd = "3 Res., Share #lambda, Stav. (No Bgd)";
  const FitValWriterInfo tFVWI_SepR_Sharelam_StavCf_NoBgd_LamKchP = FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Sharelam_StavCf_NoBgd, 
                                                                              tLegDesc_SepR_Sharelam_StavCf_NoBgd, tColorLamKchP, tMarkerStyle_StavCf, tMarkerSize);
  const FitValWriterInfo tFVWI_SepR_Sharelam_StavCf_NoBgd_LamKchM = FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Sharelam_StavCf_NoBgd, 
                                                                              tLegDesc_SepR_Sharelam_StavCf_NoBgd, tColorLamKchM, tMarkerStyle_StavCf, tMarkerSize);
  //-------------------------
//################################### Fix lambda
//---------------------------------------------------------------------------------------------------------------------------------
  TString tFitInfoTString_SepR_Fixlam_PolyBgd      = "_MomResCrctn_NonFlatBgdCrctnPolynomial_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_FixAllLambdaTo1";
  TString tLegDesc_SepR_Fixlam_PolyBgd = "3 Res., Fix #lambda, Poly. Bgd";
  const FitValWriterInfo tFVWI_SepR_Fixlam_PolyBgd_LamKchP = FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Fixlam_PolyBgd, 
                                                                              tLegDesc_SepR_Fixlam_PolyBgd, tColorLamKchP, tMarkerStyle_FixParam_PolyBgd, tMarkerSize);
  const FitValWriterInfo tFVWI_SepR_Fixlam_PolyBgd_LamKchM = FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Fixlam_PolyBgd, 
                                                                              tLegDesc_SepR_Fixlam_PolyBgd, tColorLamKchM, tMarkerStyle_FixParam_PolyBgd, tMarkerSize);
  //-------------------------
  TString tFitInfoTString_SepR_Fixlam_LinrBgd      = "_MomResCrctn_NonFlatBgdCrctnLinear_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_FixAllLambdaTo1";
  TString tLegDesc_SepR_Fixlam_LinrBgd = "3 Res., Fix #lambda, Linr. Bgd";
  const FitValWriterInfo tFVWI_SepR_Fixlam_LinrBgd_LamKchP = FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Fixlam_LinrBgd, 
                                                                              tLegDesc_SepR_Fixlam_LinrBgd, tColorLamKchP, tMarkerStyle_FixParam_LinrBgd, tMarkerSize);
  const FitValWriterInfo tFVWI_SepR_Fixlam_LinrBgd_LamKchM = FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Fixlam_LinrBgd, 
                                                                              tLegDesc_SepR_Fixlam_LinrBgd, tColorLamKchM, tMarkerStyle_FixParam_LinrBgd, tMarkerSize);
  //-------------------------
  TString tFitInfoTString_SepR_Fixlam_StavCf_NoBgd = "_MomResCrctn_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_StavCf_FixAllLambdaTo1";
  TString tLegDesc_SepR_Fixlam_StavCf_NoBgd = "3 Res., Fix #lambda, Stav. (No Bgd)";
  const FitValWriterInfo tFVWI_SepR_Fixlam_StavCf_NoBgd_LamKchP = FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Fixlam_StavCf_NoBgd, 
                                                                              tLegDesc_SepR_Fixlam_StavCf_NoBgd, tColorLamKchP, tMarkerStyle_FixParam_StavCf, tMarkerSize);
  const FitValWriterInfo tFVWI_SepR_Fixlam_StavCf_NoBgd_LamKchM = FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Fixlam_StavCf_NoBgd, 
                                                                              tLegDesc_SepR_Fixlam_StavCf_NoBgd, tColorLamKchM, tMarkerStyle_FixParam_StavCf, tMarkerSize);
  //-------------------------

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Shared Radii %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

//---------------------------------------------------------------------------------------------------------------------------------
  TString tFitInfoTString_ShareR_Sharelam_PolyBgd      = "_MomResCrctn_NonFlatBgdCrctnPolynomial_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_ShareLam_Dualie_ShareLam_ShareRadii";
  TString tLegDesc_ShareR_Sharelam_PolyBgd = "3 Res., Share R and #lambda, Poly. Bgd";
  const FitValWriterInfo tFVWI_ShareR_Sharelam_PolyBgd_LamKchP = FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Sharelam_PolyBgd, 
                                                                              tLegDesc_ShareR_Sharelam_PolyBgd, tColorLamKchP, tMarkerStyle_PolyBgd, tMarkerSize);
  const FitValWriterInfo tFVWI_ShareR_Sharelam_PolyBgd_LamKchM = FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Sharelam_PolyBgd, 
                                                                              tLegDesc_ShareR_Sharelam_PolyBgd, tColorLamKchM, tMarkerStyle_PolyBgd, tMarkerSize);
  //-------------------------
  TString tFitInfoTString_ShareR_Sharelam_LinrBgd      = "_MomResCrctn_NonFlatBgdCrctnLinear_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_ShareLam_Dualie_ShareLam_ShareRadii";
  TString tLegDesc_ShareR_Sharelam_LinrBgd = "3 Res., Share R and #lambda, Linr. Bgd";
  const FitValWriterInfo tFVWI_ShareR_Sharelam_LinrBgd_LamKchP = FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Sharelam_LinrBgd, 
                                                                              tLegDesc_ShareR_Sharelam_LinrBgd, tColorLamKchP, tMarkerStyle_LinrBgd, tMarkerSize);
  const FitValWriterInfo tFVWI_ShareR_Sharelam_LinrBgd_LamKchM = FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Sharelam_LinrBgd, 
                                                                              tLegDesc_ShareR_Sharelam_LinrBgd, tColorLamKchM, tMarkerStyle_LinrBgd, tMarkerSize);
  //-------------------------
  TString tFitInfoTString_ShareR_Sharelam_StavCf_NoBgd = "_MomResCrctn_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_StavCf_ShareLam_Dualie_ShareLam_ShareRadii";
  TString tLegDesc_ShareR_Sharelam_StavCf_NoBgd = "3 Res., Share R and #lambda, Stav. (No Bgd)";
  const FitValWriterInfo tFVWI_ShareR_Sharelam_StavCf_NoBgd_LamKchP = FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Sharelam_StavCf_NoBgd, 
                                                                              tLegDesc_ShareR_Sharelam_StavCf_NoBgd, tColorLamKchP, tMarkerStyle_StavCf, tMarkerSize);
  const FitValWriterInfo tFVWI_ShareR_Sharelam_StavCf_NoBgd_LamKchM = FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Sharelam_StavCf_NoBgd, 
                                                                              tLegDesc_ShareR_Sharelam_StavCf_NoBgd, tColorLamKchM, tMarkerStyle_StavCf, tMarkerSize);
  //-------------------------
//---------------------------------------------------------------------------------------------------------------------------------
  TString tFitInfoTString_ShareR_SharelamConj_PolyBgd      = "_MomResCrctn_NonFlatBgdCrctnPolynomial_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_ShareLam_Dualie_ShareRadii";
  TString tLegDesc_ShareR_SharelamConj_PolyBgd = "3 Res., Share R and #lambda_{Conj}, Poly. Bgd";
  const FitValWriterInfo tFVWI_ShareR_SharelamConj_PolyBgd_LamKchP = FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_SharelamConj_PolyBgd, 
                                                                              tLegDesc_ShareR_SharelamConj_PolyBgd, tColorLamKchP, tMarkerStyle_PolyBgd, tMarkerSize);
  const FitValWriterInfo tFVWI_ShareR_SharelamConj_PolyBgd_LamKchM = FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_SharelamConj_PolyBgd, 
                                                                              tLegDesc_ShareR_SharelamConj_PolyBgd, tColorLamKchM, tMarkerStyle_PolyBgd, tMarkerSize);
  //-------------------------
  TString tFitInfoTString_ShareR_SharelamConj_LinrBgd      = "_MomResCrctn_NonFlatBgdCrctnLinear_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_ShareLam_Dualie_ShareRadii";
  TString tLegDesc_ShareR_SharelamConj_LinrBgd = "3 Res., Share R and #lambda_{Conj}, Linr. Bgd";
  const FitValWriterInfo tFVWI_ShareR_SharelamConj_LinrBgd_LamKchP = FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_SharelamConj_LinrBgd, 
                                                                              tLegDesc_ShareR_SharelamConj_LinrBgd, tColorLamKchP, tMarkerStyle_LinrBgd, tMarkerSize);
  const FitValWriterInfo tFVWI_ShareR_SharelamConj_LinrBgd_LamKchM = FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_SharelamConj_LinrBgd, 
                                                                              tLegDesc_ShareR_SharelamConj_LinrBgd, tColorLamKchM, tMarkerStyle_LinrBgd, tMarkerSize);
  //-------------------------
  TString tFitInfoTString_ShareR_SharelamConj_StavCf_NoBgd = "_MomResCrctn_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_StavCf_ShareLam_Dualie_ShareRadii";
  TString tLegDesc_ShareR_SharelamConj_StavCf_NoBgd = "3 Res., Share R and #lambda_{Conj}, Stav. (No Bgd)";
  const FitValWriterInfo tFVWI_ShareR_SharelamConj_StavCf_NoBgd_LamKchP = FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_SharelamConj_StavCf_NoBgd, 
                                                                              tLegDesc_ShareR_SharelamConj_StavCf_NoBgd, tColorLamKchP, tMarkerStyle_StavCf, tMarkerSize);
  const FitValWriterInfo tFVWI_ShareR_SharelamConj_StavCf_NoBgd_LamKchM = FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_SharelamConj_StavCf_NoBgd, 
                                                                              tLegDesc_ShareR_SharelamConj_StavCf_NoBgd, tColorLamKchM, tMarkerStyle_StavCf, tMarkerSize);
  //-------------------------
//################################### Fix lambda
//---------------------------------------------------------------------------------------------------------------------------------
  TString tFitInfoTString_ShareR_Fixlam_PolyBgd      = "_MomResCrctn_NonFlatBgdCrctnPolynomial_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_FixAllLambdaTo1_ShareLam_Dualie_ShareLam_ShareRadii";
  TString tLegDesc_ShareR_Fixlam_PolyBgd = "3 Res., Share R, Fix #lambda, Poly. Bgd";
  const FitValWriterInfo tFVWI_ShareR_Fixlam_PolyBgd_LamKchP = FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Fixlam_PolyBgd, 
                                                                              tLegDesc_ShareR_Fixlam_PolyBgd, tColorLamKchP, tMarkerStyle_FixParam_PolyBgd, tMarkerSize);
  const FitValWriterInfo tFVWI_ShareR_Fixlam_PolyBgd_LamKchM = FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Fixlam_PolyBgd, 
                                                                              tLegDesc_ShareR_Fixlam_PolyBgd, tColorLamKchM, tMarkerStyle_FixParam_PolyBgd, tMarkerSize);
  //-------------------------
  TString tFitInfoTString_ShareR_Fixlam_LinrBgd      = "_MomResCrctn_NonFlatBgdCrctnLinear_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_FixAllLambdaTo1_ShareLam_Dualie_ShareLam_ShareRadii";
  TString tLegDesc_ShareR_Fixlam_LinrBgd = "3 Res., Share R, Fix #lambda, Linr. Bgd";
  const FitValWriterInfo tFVWI_ShareR_Fixlam_LinrBgd_LamKchP = FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Fixlam_LinrBgd, 
                                                                              tLegDesc_ShareR_Fixlam_LinrBgd, tColorLamKchP, tMarkerStyle_FixParam_LinrBgd, tMarkerSize);
  const FitValWriterInfo tFVWI_ShareR_Fixlam_LinrBgd_LamKchM = FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Fixlam_LinrBgd, 
                                                                              tLegDesc_ShareR_Fixlam_LinrBgd, tColorLamKchM, tMarkerStyle_FixParam_LinrBgd, tMarkerSize);
  //-------------------------
  TString tFitInfoTString_ShareR_Fixlam_StavCf_NoBgd = "_MomResCrctn_3Res_PrimMaxDecay4fm_UsingXiDataAndCoulombOnly_StavCf_FixAllLambdaTo1_ShareLam_Dualie_ShareLam_ShareRadii";
  TString tLegDesc_ShareR_Fixlam_StavCf_NoBgd = "3 Res., Share R, Fix #lambda, Stav. (No Bgd)";
  const FitValWriterInfo tFVWI_ShareR_Fixlam_StavCf_NoBgd_LamKchP = FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Fixlam_StavCf_NoBgd, 
                                                                              tLegDesc_ShareR_Fixlam_StavCf_NoBgd, tColorLamKchP, tMarkerStyle_FixParam_StavCf, tMarkerSize);
  const FitValWriterInfo tFVWI_ShareR_Fixlam_StavCf_NoBgd_LamKchM = FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Fixlam_StavCf_NoBgd, 
                                                                              tLegDesc_ShareR_Fixlam_StavCf_NoBgd, tColorLamKchM, tMarkerStyle_FixParam_StavCf, tMarkerSize);
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
                                                                    tLegDesc_ShareR_Sharelam_PolyBgd, tColorLamKchP, 20, tMarkerSize), 
                                                   FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Sharelam_PolyBgd, 
                                                                    tLegDesc_SepR_Sharelam_PolyBgd, tColorLamKchP, 21, tMarkerSize), 
                                                   FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Seplam_PolyBgd, 
                                                                    tLegDesc_SepR_Seplam_PolyBgd, tColorLamKchP, 34, tMarkerSize), 
                                                   FitValWriterInfo(kALamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Seplam_PolyBgd, 
                                                                    tLegDesc_SepR_Seplam_PolyBgd, tColorLamKchP, 28, tMarkerSize), 
                                                   FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Sharelam_PolyBgd, 
                                                                    tLegDesc_ShareR_Sharelam_PolyBgd, tColorLamKchM, 20, tMarkerSize), 
                                                   FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Sharelam_PolyBgd, 
                                                                    tLegDesc_SepR_Sharelam_PolyBgd, tColorLamKchM, 21, tMarkerSize), 
                                                   FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Seplam_PolyBgd, 
                                                                    tLegDesc_SepR_Seplam_PolyBgd, tColorLamKchM, 34, tMarkerSize), 
                                                   FitValWriterInfo(kALamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Seplam_PolyBgd, 
                                                                    tLegDesc_SepR_Seplam_PolyBgd, tColorLamKchM, 28, tMarkerSize)};


//-----------------------------------------------------------------

  vector<FitValWriterInfo> tFVWIVec_FreevsFixlam_SepR = {tFVWI_SepR_Sharelam_PolyBgd_LamKchP, tFVWI_SepR_Sharelam_LinrBgd_LamKchP, tFVWI_SepR_Sharelam_StavCf_NoBgd_LamKchP,
                                                         tFVWI_SepR_Sharelam_PolyBgd_LamKchM, tFVWI_SepR_Sharelam_LinrBgd_LamKchM, tFVWI_SepR_Sharelam_StavCf_NoBgd_LamKchM,
                                                         FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Fixlam_PolyBgd, 
                                                                          tLegDesc_SepR_Fixlam_PolyBgd, tColorLamKchP, 24, tMarkerSize), 
                                                         FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Fixlam_LinrBgd, 
                                                                          tLegDesc_SepR_Fixlam_LinrBgd, tColorLamKchP, 25, tMarkerSize), 
                                                         FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Fixlam_StavCf_NoBgd, 
                                                                          tLegDesc_SepR_Fixlam_StavCf_NoBgd, tColorLamKchP, 28, tMarkerSize), 
                                                         FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Fixlam_PolyBgd, 
                                                                          tLegDesc_SepR_Fixlam_PolyBgd, tColorLamKchM, 24, tMarkerSize), 
                                                         FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Fixlam_LinrBgd, 
                                                                          tLegDesc_SepR_Fixlam_LinrBgd, tColorLamKchM, 25, tMarkerSize), 
                                                         FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Fixlam_StavCf_NoBgd, 
                                                                          tLegDesc_SepR_Fixlam_StavCf_NoBgd, tColorLamKchM, 28, tMarkerSize)};

  vector<FitValWriterInfo> tFVWIVec_FreevsFixlam_SepR_NoStav = {tFVWI_SepR_Sharelam_PolyBgd_LamKchP, tFVWI_SepR_Sharelam_LinrBgd_LamKchP,
                                                                tFVWI_SepR_Sharelam_PolyBgd_LamKchM, tFVWI_SepR_Sharelam_LinrBgd_LamKchM,
                                                               FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Fixlam_PolyBgd, 
                                                                                tLegDesc_SepR_Fixlam_PolyBgd, tColorLamKchP, 24, tMarkerSize), 
                                                               FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Fixlam_LinrBgd, 
                                                                                tLegDesc_SepR_Fixlam_LinrBgd, tColorLamKchP, 25, tMarkerSize), 
                                                               FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Fixlam_PolyBgd, 
                                                                                tLegDesc_SepR_Fixlam_PolyBgd, tColorLamKchM, 24, tMarkerSize), 
                                                               FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Fixlam_LinrBgd, 
                                                                                tLegDesc_SepR_Fixlam_LinrBgd, tColorLamKchM, 25, tMarkerSize)};

  vector<FitValWriterInfo> tFVWIVec_FreevsFixlam_SepR_PolyBgd = {tFVWI_SepR_Sharelam_PolyBgd_LamKchP, 
                                                                 tFVWI_SepR_Sharelam_PolyBgd_LamKchM,
                                                                 FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Fixlam_PolyBgd, 
                                                                                  tLegDesc_SepR_Fixlam_PolyBgd, tColorLamKchP, 24, tMarkerSize), 
                                                                 FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Fixlam_PolyBgd, 
                                                                                  tLegDesc_SepR_Fixlam_PolyBgd, tColorLamKchM, 24, tMarkerSize)};

  vector<FitValWriterInfo> tFVWIVec_FreevsFixlam_SepR_LinrBgd = {tFVWI_SepR_Sharelam_LinrBgd_LamKchP, 
                                                                 tFVWI_SepR_Sharelam_LinrBgd_LamKchM,
                                                                 FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Fixlam_LinrBgd, 
                                                                                  tLegDesc_SepR_Fixlam_LinrBgd, tColorLamKchP, 24, tMarkerSize), 
                                                                 FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Fixlam_LinrBgd, 
                                                                                  tLegDesc_SepR_Fixlam_LinrBgd, tColorLamKchM, 24, tMarkerSize)};

  vector<FitValWriterInfo> tFVWIVec_FreevsFixlam_SepR_StavCf_NoBgd = {tFVWI_SepR_Sharelam_StavCf_NoBgd_LamKchP, 
                                                                 tFVWI_SepR_Sharelam_StavCf_NoBgd_LamKchM,
                                                                 FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Fixlam_StavCf_NoBgd, 
                                                                                  tLegDesc_SepR_Fixlam_StavCf_NoBgd, tColorLamKchP, 24, tMarkerSize), 
                                                                 FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_SepR_Fixlam_StavCf_NoBgd, 
                                                                                  tLegDesc_SepR_Fixlam_StavCf_NoBgd, tColorLamKchM, 24, tMarkerSize)};

//-----------------------------------------------------------------

  vector<FitValWriterInfo> tFVWIVec_FreevsFixlam_ShareR = {tFVWI_ShareR_Sharelam_PolyBgd_LamKchP, tFVWI_ShareR_Sharelam_LinrBgd_LamKchP, tFVWI_ShareR_Sharelam_StavCf_NoBgd_LamKchP,
                                                           tFVWI_ShareR_Sharelam_PolyBgd_LamKchM, tFVWI_ShareR_Sharelam_LinrBgd_LamKchM, tFVWI_ShareR_Sharelam_StavCf_NoBgd_LamKchM,
                                                           FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Fixlam_PolyBgd, 
                                                                            tLegDesc_ShareR_Fixlam_PolyBgd, tColorLamKchP, 24, tMarkerSize), 
                                                           FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Fixlam_LinrBgd, 
                                                                            tLegDesc_ShareR_Fixlam_LinrBgd, tColorLamKchP, 25, tMarkerSize), 
                                                           FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Fixlam_StavCf_NoBgd, 
                                                                            tLegDesc_ShareR_Fixlam_StavCf_NoBgd, tColorLamKchP, 28, tMarkerSize), 
                                                           FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Fixlam_PolyBgd, 
                                                                            tLegDesc_ShareR_Fixlam_PolyBgd, tColorLamKchM, 24, tMarkerSize), 
                                                           FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Fixlam_LinrBgd, 
                                                                            tLegDesc_ShareR_Fixlam_LinrBgd, tColorLamKchM, 25, tMarkerSize), 
                                                           FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Fixlam_StavCf_NoBgd, 
                                                                            tLegDesc_ShareR_Fixlam_StavCf_NoBgd, tColorLamKchM, 28, tMarkerSize)};

  vector<FitValWriterInfo> tFVWIVec_FreevsFixlam_ShareR_NoStav = {tFVWI_ShareR_Sharelam_PolyBgd_LamKchP, tFVWI_ShareR_Sharelam_LinrBgd_LamKchP,
                                                                  tFVWI_ShareR_Sharelam_PolyBgd_LamKchM, tFVWI_ShareR_Sharelam_LinrBgd_LamKchM,
                                                                  FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Fixlam_PolyBgd, 
                                                                                   tLegDesc_ShareR_Fixlam_PolyBgd, tColorLamKchP, 24, tMarkerSize), 
                                                                  FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Fixlam_LinrBgd, 
                                                                                   tLegDesc_ShareR_Fixlam_LinrBgd, tColorLamKchP, 25, tMarkerSize), 
                                                                  FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Fixlam_PolyBgd, 
                                                                                   tLegDesc_ShareR_Fixlam_PolyBgd, tColorLamKchM, 24, tMarkerSize), 
                                                                  FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Fixlam_LinrBgd, 
                                                                                   tLegDesc_ShareR_Fixlam_LinrBgd, tColorLamKchM, 25, tMarkerSize)};

  vector<FitValWriterInfo> tFVWIVec_FreevsFixlam_ShareR_PolyBgd = {tFVWI_ShareR_Sharelam_PolyBgd_LamKchP,
                                                                   tFVWI_ShareR_Sharelam_PolyBgd_LamKchM,
                                                                   FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Fixlam_PolyBgd, 
                                                                                    tLegDesc_ShareR_Fixlam_PolyBgd, tColorLamKchP, 24, tMarkerSize), 
                                                                   FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Fixlam_PolyBgd, 
                                                                                    tLegDesc_ShareR_Fixlam_PolyBgd, tColorLamKchM, 24, tMarkerSize)};

  vector<FitValWriterInfo> tFVWIVec_FreevsFixlam_ShareR_LinrBgd = {tFVWI_ShareR_Sharelam_LinrBgd_LamKchP,
                                                                   tFVWI_ShareR_Sharelam_LinrBgd_LamKchM,
                                                                   FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Fixlam_LinrBgd, 
                                                                                    tLegDesc_ShareR_Fixlam_LinrBgd, tColorLamKchP, 24, tMarkerSize), 
                                                                   FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Fixlam_LinrBgd, 
                                                                                    tLegDesc_ShareR_Fixlam_LinrBgd, tColorLamKchM, 24, tMarkerSize)};

  vector<FitValWriterInfo> tFVWIVec_FreevsFixlam_ShareR_StavCf_NoBgd = {tFVWI_ShareR_Sharelam_StavCf_NoBgd_LamKchP,
                                                                   tFVWI_ShareR_Sharelam_StavCf_NoBgd_LamKchM,
                                                                   FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Fixlam_StavCf_NoBgd, 
                                                                                    tLegDesc_ShareR_Fixlam_StavCf_NoBgd, tColorLamKchP, 24, tMarkerSize), 
                                                                   FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_ShareR_Fixlam_StavCf_NoBgd, 
                                                                                    tLegDesc_ShareR_Fixlam_StavCf_NoBgd, tColorLamKchM, 24, tMarkerSize)};



//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

  //-----------

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


  vector<FitValWriterInfo> tFVWIVec_CompNumRes_ShareR_PolyBgd = {FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_LamKch_3Res_PolyBgd, 
                                                                                  "3 Res., Share R and #lambda, Poly. Bgd", tColorLamKchP, 34, tMarkerSize), 
                                                                 FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_LamKch_10Res_PolyBgd, 
                                                                                  "10 Res., Share R and #lambda, Poly. Bgd", tColorLamKchP, 47, tMarkerSize), 
                                                                 FitValWriterInfo(kLamKchP, tFileLocation_LamKch, tResultsDate, tFitInfoTString_LamKch_NoRes_PolyBgd, 
                                                                                  "No Res., Share R and #lambda, Poly. Bgd", tColorLamKchP, 20, tMarkerSize), 

                                                                 FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_LamKch_3Res_PolyBgd, 
                                                                                  "3 Res., Share R and #lambda, Poly. Bgd", tColorLamKchM, 34, tMarkerSize), 
                                                                 FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_LamKch_10Res_PolyBgd, 
                                                                                  "10 Res., Share R and #lambda, Poly. Bgd", tColorLamKchM, 47, tMarkerSize), 
                                                                 FitValWriterInfo(kLamKchM, tFileLocation_LamKch, tResultsDate, tFitInfoTString_LamKch_NoRes_PolyBgd, 
                                                                                  "No Res., Share R and #lambda, Poly. Bgd", tColorLamKchM, 20, tMarkerSize), 

                                                                 FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, tFitInfoTString_LamK0_3Res_PolyBgd, 
                                                                                  "3 Res., Share R and #lambda, Poly. Bgd", tColorLamK0, 34, tMarkerSize), 
                                                                 FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, tFitInfoTString_LamK0_10Res_PolyBgd, 
                                                                                  "10 Res., Share R and #lambda, Poly. Bgd", tColorLamK0, 47, tMarkerSize), 
                                                                 FitValWriterInfo(kLamK0, tFileLocation_LamK0, tResultsDate, tFitInfoTString_LamK0_NoRes_PolyBgd, 
                                                                                  "No Res., Share R and #lambda, Poly. Bgd", tColorLamK0, 20, tMarkerSize)};


