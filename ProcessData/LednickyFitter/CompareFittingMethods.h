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

#include <iostream>
#include <vector>
#include <cassert>
typedef std::vector<double> td1dVec;
typedef std::vector<std::vector<double> > td2dVec;

using namespace std;

//---------------------------------------------------------------------------------------------------------------------------------
struct FitInfo
{
  TString descriptor;
  AnalysisType analysisType;

  double lambda1, lambda2, lambda3;
  double radius1, radius2, radius3;
  double ref0, imf0, d0;


  double lambdaStatErr1, lambdaStatErr2, lambdaStatErr3;
  double radiusStatErr1, radiusStatErr2, radiusStatErr3; 
  double ref0StatErr, imf0StatErr, d0StatErr;

  double chi2;
  int ndf;

  bool freeLambda;
  bool freeRadii;
  bool freeD0;
  bool all10ResidualsUsed;

  Color_t markerColor;
  int markerStyle;

  FitInfo(TString aDescriptor, 
          AnalysisType aAnalysisType,
          bool aFreeLambda, bool aFreeRadii, bool aFreeD0, bool aAll10ResidualsUsed,
          double aLambda1, double aLambdaStatErr1, 
          double aLambda2, double aLambdaStatErr2, 
          double aLambda3, double aLambdaStatErr3, 
          double aRadius1, double aRadiusStatErr1, 
          double aRadius2, double aRadiusStatErr2, 
          double aRadius3, double aRadiusStatErr3, 
          double aReF0,   double aReF0StatErr, 
          double aImF0,   double aImF0StatErr, 
          double aD0,     double aD0StatErr, 
          double aChi2, int aNDF,
          Color_t aMarkerColor, int aMarkerStyle)
  {
    descriptor = aDescriptor;
    analysisType = aAnalysisType;

    freeLambda = aFreeLambda;
    freeRadii = aFreeRadii;
    freeD0 = aFreeD0;
    all10ResidualsUsed = aAll10ResidualsUsed;

    lambda1 = aLambda1;
    lambda2 = aLambda2;
    lambda3 = aLambda3;

    radius1 = aRadius1;
    radius2 = aRadius2;
    radius3 = aRadius3;

    ref0   = aReF0;
    imf0   = aImF0;
    d0     = aD0;

    lambdaStatErr1 = aLambdaStatErr1;
    lambdaStatErr2 = aLambdaStatErr2;
    lambdaStatErr3 = aLambdaStatErr3;

    radiusStatErr1 = aRadiusStatErr1;
    radiusStatErr2 = aRadiusStatErr2;
    radiusStatErr3 = aRadiusStatErr3;

    ref0StatErr   = aReF0StatErr;
    imf0StatErr   = aImF0StatErr;
    d0StatErr     = aD0StatErr;

    chi2 = aChi2;
    ndf = aNDF;

    markerColor = aMarkerColor;
    markerStyle = aMarkerStyle;
  }

};


//---------------------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************************
//---------------------------------------------------------------------------------------------------------------------------------

//-----------------------------------------------------------------------------
  Color_t tColor1 = kBlack;
  Color_t tColor2 = kBlue;
  Color_t tColor3 = kRed;
  Color_t tColor4 = kGreen+2;
  Color_t tColor5 = kCyan;
  Color_t tColor6 = kYellow+2;

  int tMarkerStyleA1 = 20;
  int tMarkerStyleA2 = 22;

  int tMarkerStyleB1 = 24;
  int tMarkerStyleB2 = 26;

  //---------------------------------------------------------------------------
  //---------------------------- LamKchP --------------------------------------
  //---------------------------------------------------------------------------

  //--------------- 10 Residuals ----------
  const FitInfo tFitInfo1a_LamKchP = FitInfo(TString("FreeRadii_FreeD0_10Res"), 
                                       kLamKchP,
                                       true, true, true, true, 
                                       0.5*(0.96+0.94), 0.5*(0.37+0.36),
                                       0.5*(1.18+0.99), 0.5*(0.51+0.42),
                                       0.5*(1.01+0.98), 0.5*(0.30+0.29),

                                       4.98, 0.96,
                                       4.76, 0.99,
                                       3.55, 0.52,

                                      -1.51, 0.37,
                                       0.65, 0.40,
                                       1.13, 0.74,

                                       424.2,
                                       336,

                                       tColor1, tMarkerStyleA1);

  const FitInfo tFitInfo2a_LamKchP = FitInfo(TString("FreeRadii_FixedD0_10Res"), 
                                       kLamKchP,
                                       true, true, false, true, 
                                       0.5*(1.33+1.32), 0.5*(0.40+0.40),
                                       0.5*(1.56+1.30), 0.5*(0.63+0.52),
                                       0.5*(1.16+1.12), 0.5*(0.31+0.30),

                                       5.26, 0.87,
                                       4.91, 1.00,
                                       3.37, 0.47,

                                      -1.08, 0.14,
                                       0.59, 0.30,
                                       0.00, 0.00,

                                       425.9,
                                       337,

                                       tColor1, tMarkerStyleA2);


  const FitInfo tFitInfo3a_LamKchP = FitInfo(TString("FixedLambda_FreeD0_10Res"), 
                                       kLamKchP,
                                       false, true, true, true, 
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00),
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00),
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00),

                                       5.09, 0.54,
                                       4.57, 0.48,
                                       3.55, 0.35,

                                      -1.49, 0.24,
                                       0.66, 0.38,
                                       1.10, 0.59,

                                       428.3,
                                       342,

                                       tColor2, tMarkerStyleA1);

  const FitInfo tFitInfo4a_LamKchP = FitInfo(TString("FixedLambda_FixedD0_10Res"), 
                                       kLamKchP,
                                       false, true, false, true, 
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00),
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00),
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00),

                                       4.67, 0.43,
                                       4.17, 0.37,
                                       3.22, 0.26,

                                      -1.18, 0.13,
                                       0.60, 0.30,
                                       0.00, 0.00,

                                       431.3,
                                       343,

                                       tColor2, tMarkerStyleA2);

  const FitInfo tFitInfo5a_LamKchP = FitInfo(TString("FixedRadii_FreeD0_10Res"), 
                                       kLamKchP,
                                       true, false, true, true, 
                                       0.5*(0.84+0.83), 0.5*(0.27+0.27),
                                       0.5*(0.96+0.81), 0.5*(0.31+0.26),
                                       0.5*(0.81+0.79), 0.5*(0.27+0.27),

                                       3.50, 0.00,
                                       3.25, 0.00,
                                       2.50, 0.00,

                                      -1.02, 0.36,
                                       0.08, 0.06,
                                       0.92, 0.38,

                                       432.6,
                                       339,

                                       tColor3, tMarkerStyleA1);

  const FitInfo tFitInfo6a_LamKchP = FitInfo(TString("FixedRadii_FixedD0_10Res"), 
                                       kLamKchP,
                                       true, false, false, true, 
                                       0.5*(1.03+1.03), 0.5*(0.32+0.32),
                                       0.5*(1.18+1.00), 0.5*(0.36+0.30),
                                       0.5*(1.05+1.02), 0.5*(0.32+0.32),

                                       3.50, 0.00,
                                       3.25, 0.00,
                                       2.50, 0.00,

                                      -0.76, 0.23,
                                       0.12, 0.07,
                                       0.00, 0.00,

                                       435.1,
                                       340,

                                       tColor3, tMarkerStyleA2);


  //--------------- 3 Residuals ----------
  const FitInfo tFitInfo1b_LamKchP = FitInfo(TString("FreeRadii_FreeD0_3Res"), 
                                       kLamKchP,
                                       true, true, true, false, 
                                       0.5*(0.84+0.82), 0.5*(0.31+0.30),
                                       0.5*(1.09+0.90), 0.5*(0.47+0.39),
                                       0.5*(1.02+0.97), 0.5*(0.30+0.29),

                                       4.43, 0.86,
                                       4.34, 0.94,
                                       3.38, 0.50,

                                      -1.24, 0.32,
                                       0.50, 0.35,
                                       1.11, 0.51,

                                       427.2,
                                       336,

                                       tColor1, tMarkerStyleB1);

  const FitInfo tFitInfo2b_LamKchP = FitInfo(TString("FreeRadii_FixedD0_3Res"), 
                                       kLamKchP,
                                       true, true, false, false, 
                                       0.5*(1.23+1.21), 0.5*(0.37+0.37),
                                       0.5*(1.54+1.27), 0.5*(0.59+0.48),
                                       0.5*(1.22+1.17), 0.5*(0.32+0.31),

                                       4.82, 0.81,
                                       4.62, 0.90,
                                       3.27, 0.44,

                                      -0.85, 0.13,
                                       0.49, 0.24,
                                       0.00, 0.00,

                                       430.0,
                                       337,

                                       tColor1, tMarkerStyleB2);


  const FitInfo tFitInfo3b_LamKchP = FitInfo(TString("FixedLambda_FreeD0_3Res"), 
                                       kLamKchP,
                                       false, true, true, false, 
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00),
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00),
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00),

                                       4.84, 0.60,
                                       4.34, 0.53,
                                       3.37, 0.38,

                                      -1.16, 0.19,
                                       0.55, 0.33,
                                       1.02, 0.43,

                                       432.1,
                                       342,

                                       tColor2, tMarkerStyleB1);

  const FitInfo tFitInfo4b_LamKchP = FitInfo(TString("FixedLambda_FixedD0_3Res"), 
                                       kLamKchP,
                                       false, true, false, false, 
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00),
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00),
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00),

                                       4.48, 0.42,
                                       4.01, 0.36,
                                       3.09, 0.26,

                                      -0.94, 0.10,
                                       0.51, 0.23,
                                       0.00, 0.00,

                                       435.8,
                                       343,

                                       tColor2, tMarkerStyleB2);

  const FitInfo tFitInfo5b_LamKchP = FitInfo(TString("FixedRadii_FreeD0_3Res"), 
                                       kLamKchP,
                                       true, false, true, false, 
                                       0.5*(0.83+0.82), 0.5*(0.27+0.27),
                                       0.5*(0.96+0.80), 0.5*(0.31+0.26),
                                       0.5*(0.82+0.80), 0.5*(0.28+0.28),

                                       3.50, 0.00,
                                       3.25, 0.00,
                                       2.50, 0.00,

                                      -0.90, 0.32,
                                       0.12, 0.06,
                                       1.06, 0.27,

                                       432.6,
                                       339,

                                       tColor3, tMarkerStyleB1);

  const FitInfo tFitInfo6b_LamKchP = FitInfo(TString("FixedRadii_FixedD0_3Res"), 
                                       kLamKchP,
                                       true, false, false, false, 
                                       0.5*(1.04+1.03), 0.5*(0.29+0.29),
                                       0.5*(1.20+1.00), 0.5*(0.33+0.28),
                                       0.5*(1.08+1.04), 0.5*(0.31+0.30),

                                       3.50, 0.00,
                                       3.25, 0.00,
                                       2.50, 0.00,

                                      -0.66, 0.18,
                                       0.15, 0.07,
                                       0.00, 0.00,

                                       437.0,
                                       340,

                                       tColor3, tMarkerStyleB2);



  //---------------------------------------------------------------------------
  //---------------------------- LamKchM --------------------------------------
  //---------------------------------------------------------------------------

  //--------------- 10 Residuals ----------
  const FitInfo tFitInfo1a_LamKchM = FitInfo(TString("FreeRadii_FreeD0_10Res"), 
                                       kLamKchM,
                                       true, true, true, true, 
                                       0.5*(1.50+1.49), 0.5*(0.72+0.69),
                                       0.5*(1.15+1.41), 0.5*(0.51+0.61),
                                       0.5*(1.07+0.80), 0.5*(0.73+0.37),

                                       6.21, 1.51,
                                       4.86, 1.15,
                                       2.86, 0.81,

                                       0.45, 0.23,
                                       0.52, 0.21,
                                      -4.81, 2.62,

                                       286.2,
                                       288,

                                       tColor1, tMarkerStyleA1);

  const FitInfo tFitInfo2a_LamKchM = FitInfo(TString("FreeRadii_FixedD0_10Res"), 
                                       kLamKchM,
                                       true, true, false, true, 
                                       0.5*(1.16+1.17), 0.5*(0.63+0.61),
                                       0.5*(0.97+1.19), 0.5*(0.52+0.63),
                                       0.5*(0.89+0.72), 0.5*(0.70+0.41),

                                       4.73, 0.77,
                                       3.91, 0.70,
                                       2.34, 0.53,

                                       0.34, 0.17,
                                       0.52, 0.27,
                                       0.00, 0.00,

                                       290.1,
                                       289,

                                       tColor1, tMarkerStyleA2);


  const FitInfo tFitInfo3a_LamKchM = FitInfo(TString("FixedLambda_FreeD0_10Res"), 
                                       kLamKchM,
                                       false, true, true, true, 
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00),
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00),
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00),

                                       5.15, 0.26,
                                       4.44, 0.28,
                                       3.22, 0.31,

                                       0.54, 0.18,
                                       0.69, 0.15,
                                      -3.31, 1.41,

                                       293.4,
                                       294,

                                       tColor2, tMarkerStyleA1);

  const FitInfo tFitInfo4a_LamKchM = FitInfo(TString("FixedLambda_FixedD0_10Res"), 
                                       kLamKchM,
                                       false, true, false, true, 
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00),
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00),
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00),

                                       4.52, 0.36,
                                       3.89, 0.34,
                                       2.77, 0.32,

                                       0.40, 0.17,
                                       0.60, 0.15,
                                       0.00, 0.00,

                                       296.2,
                                       295,

                                       tColor2, tMarkerStyleA2);

  const FitInfo tFitInfo5a_LamKchM = FitInfo(TString("FixedRadii_FreeD0_10Res"), 
                                       kLamKchM,
                                       true, false, true, true, 
                                       0.5*(0.58+0.60), 0.5*(0.27+0.27),
                                       0.5*(0.58+0.72), 0.5*(0.26+0.32),
                                       0.5*(0.63+0.56), 0.5*(0.34+0.24),

                                       3.50, 0.00,
                                       3.25, 0.00,
                                       2.50, 0.00,

                                       0.23, 0.20,
                                       0.64, 0.51,
                                       1.81, 4.68,

                                       297.3,
                                       291,

                                       tColor3, tMarkerStyleA1);

  const FitInfo tFitInfo6a_LamKchM = FitInfo(TString("FixedRadii_FixedD0_10Res"), 
                                       kLamKchM,
                                       true, false, false, true, 
                                       0.5*(0.55+0.57), 0.5*(0.15+0.15),
                                       0.5*(0.56+0.69), 0.5*(0.15+0.18),
                                       0.5*(0.60+0.54), 0.5*(0.22+0.16),

                                       3.50, 0.00,
                                       3.25, 0.00,
                                       2.50, 0.00,

                                       0.25, 0.15,
                                       0.71, 0.33,
                                       0.00, 0.00,

                                       297.2,
                                       292,

                                       tColor3, tMarkerStyleA2);



  //--------------- 3 Residuals ----------
  const FitInfo tFitInfo1b_LamKchM = FitInfo(TString("FreeRadii_FreeD0_3Res"), 
                                       kLamKchM,
                                       true, true, true, false, 
                                       0.5*(1.55+1.54), 0.5*(0.72+0.69),
                                       0.5*(1.19+1.46), 0.5*(0.51+0.61),
                                       0.5*(1.08+0.80), 0.5*(0.71+0.36),

                                       6.02, 1.46,
                                       4.74, 1.11,
                                       2.75, 0.80,

                                       0.34, 0.21,
                                       0.42, 0.19,
                                      -5.72, 3.39,

                                       285.0,
                                       288,

                                       tColor1, tMarkerStyleB1);

  const FitInfo tFitInfo2b_LamKchM = FitInfo(TString("FreeRadii_FixedD0_3Res"), 
                                       kLamKchM,
                                       true, true, false, false, 
                                       0.5*(1.20+1.20), 0.5*(0.60+0.58),
                                       0.5*(1.01+1.24), 0.5*(0.51+0.62),
                                       0.5*(0.94+0.75), 0.5*(0.73+0.41),

                                       4.33, 0.64,
                                       3.61, 0.62,
                                       2.12, 0.46,

                                       0.21, 0.14,
                                       0.39, 0.22,
                                       0.00, 0.00,

                                       290.6,
                                       289,

                                       tColor1, tMarkerStyleB2);


  const FitInfo tFitInfo3b_LamKchM = FitInfo(TString("FixedLambda_FreeD0_3Res"), 
                                       kLamKchM,
                                       false, true, false, false, 
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00),
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00),
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00),

                                       4.95, 0.33,
                                       4.27, 0.32,
                                       3.13, 0.32,

                                       0.40, 0.16,
                                       0.57, 0.13,
                                      -3.83, 1.52,

                                       292.8,
                                       294,

                                       tColor2, tMarkerStyleB1);

  const FitInfo tFitInfo4b_LamKchM = FitInfo(TString("FixedLambda_FixedD0_3Res"), 
                                       kLamKchM,
                                       false, true, false, false, 
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00),
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00),
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00),

                                       4.12, 0.44,
                                       3.56, 0.39,
                                       2.50, 0.35,

                                       0.25, 0.14,
                                       0.45, 0.13,
                                       0.00, 0.00,

                                       297.0,
                                       295,

                                       tColor2, tMarkerStyleB2);

  const FitInfo tFitInfo5b_LamKchM = FitInfo(TString("FixedRadii_FreeD0_3Res"), 
                                       kLamKchM,
                                       true, false, true, false, 
                                       0.5*(0.65+0.67), 0.5*(0.32+0.32),
                                       0.5*(0.65+0.81), 0.5*(0.31+0.38),
                                       0.5*(0.72+0.62), 0.5*(0.44+0.29),

                                       3.50, 0.00,
                                       3.25, 0.00,
                                       2.50, 0.00,

                                       0.12, 0.13,
                                       0.45, 0.36,
                                      -4.68, 4.91,

                                       295.3,
                                       291,

                                       tColor3, tMarkerStyleB1);

  const FitInfo tFitInfo6b_LamKchM = FitInfo(TString("FixedRadii_FixedD0_3Res"), 
                                       kLamKchM,
                                       true, false, false, false, 
                                       0.5*(0.78+0.80), 0.5*(0.38+0.38),
                                       0.5*(0.78+0.96), 0.5*(0.37+0.45),
                                       0.5*(0.93+0.74), 0.5*(0.62+0.36),

                                       3.50, 0.00,
                                       3.25, 0.00,
                                       2.50, 0.00,

                                       0.15, 0.10,
                                       0.42, 0.28,
                                       0.00, 0.00,

                                       297.2,
                                       292,

                                       tColor3, tMarkerStyleB2);


  //---------------------------------------------------------------------------
  //---------------------------- LamK0 --------------------------------------
  //---------------------------------------------------------------------------

  //--------------- 10 Residuals ----------
  const FitInfo tFitInfo1a_LamK0 = FitInfo(TString("FreeRadii_FreeD0_10Res"), 
                                       kLamK0,
                                       true, true, true, true, 
                                       0.5*(0.60+0.60), 0.5*(0.76+0.76),
                                       0.5*(0.60+0.60), 0.5*(0.76+0.76),
                                       0.5*(0.60+0.60), 0.5*(0.76+0.76),

                                       2.97, 0.49,
                                       2.30, 0.39,
                                       1.70, 0.29,

                                      -0.26, 0.07,
                                       0.17, 0.07,
                                       2.53, 0.68,

                                       357.7,
                                       341,

                                       tColor1, tMarkerStyleA1);

  const FitInfo tFitInfo2a_LamK0 = FitInfo(TString("FreeRadii_FixedD0_10Res"), 
                                       kLamK0,
                                       true, true, false, true, 
                                       0.5*(1.50+1.44), 0.5*(0.65+0.16),
                                       0.5*(0.60+0.60), 0.5*(0.17+0.75),
                                       0.5*(1.50+1.40), 0.5*(0.59+0.24),

                                       3.53, 1.01,
                                       1.72, 0.45,
                                       1.99, 0.53,

                                      -0.12, 0.03,
                                       0.11, 0.10,
                                       0.00, 0.00,

                                       354.1,
                                       337,

                                       tColor1, tMarkerStyleA2);


  const FitInfo tFitInfo3a_LamK0 = FitInfo(TString("FixedLambda_FreeD0_10Res"), 
                                       kLamK0,
                                       false, true, true, true, 
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00),
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00),
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00),

                                       3.16, 0.52,
                                       2.44, 0.42,
                                       1.79, 0.31,

                                      -0.15, 0.04,
                                       0.13, 0.05,
                                       3.50, 1.14,

                                       358.0,
                                       342,

                                       tColor2, tMarkerStyleA1);

  const FitInfo tFitInfo4a_LamK0 = FitInfo(TString("FixedLambda_FixedD0_10Res"), 
                                       kLamK0,
                                       false, true, false, true, 
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00),
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00),
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00),

                                       3.32, 0.88,
                                       2.54, 0.65,
                                       1.84, 0.45,

                                      -0.13, 0.04,
                                       0.16, 0.12,
                                       0.00, 0.00,

                                       360.1,
                                       343,

                                       tColor2, tMarkerStyleA2);

  const FitInfo tFitInfo5a_LamK0 = FitInfo(TString("FixedRadii_FreeD0_10Res"), 
                                       kLamK0,
                                       true, false, true, true, 
                                       0.5*(1.50+1.50), 0.5*(0.83+0.83),
                                       0.5*(1.50+1.50), 0.5*(0.83+0.83),
                                       0.5*(1.50+1.50), 0.5*(0.83+0.83),

                                       3.25, 0.00,
                                       2.75, 0.00,
                                       2.25, 0.00,

                                      -0.11, 0.03,
                                       0.10, 0.01,
                                      -0.73, 2.57,

                                       370.1,
                                       344,

                                       tColor3, tMarkerStyleA1);

  const FitInfo tFitInfo6a_LamK0 = FitInfo(TString("FixedRadii_FixedD0_10Res"), 
                                       kLamK0,
                                       true, false, false, true, 
                                       0.5*(0.99+0.93), 0.5*(0.16+0.16),
                                       0.5*(1.11+1.14), 0.5*(0.20+0.20),
                                       0.5*(1.50+1.40), 0.5*(0.89+0.65),

                                       3.25, 0.00,
                                       2.75, 0.00,
                                       2.25, 0.00,

                                      -0.12, 0.04,
                                       0.16, 0.04,
                                       0.00, 0.00,

                                       360.6,
                                       340,

                                       tColor3, tMarkerStyleA2);

  //--------------- 3 Residuals ----------
  const FitInfo tFitInfo1b_LamK0 = FitInfo(TString("FreeRadii_FreeD0_3Res"), 
                                       kLamK0,
                                       true, true, true, false, 
                                       0.5*(0.60+0.60), 0.5*(0.77+0.77),
                                       0.5*(0.60+0.60), 0.5*(0.77+0.77),
                                       0.5*(0.60+0.60), 0.5*(0.77+0.77),

                                       2.91, 0.51,
                                       2.22, 0.40,
                                       1.64, 0.30,

                                      -0.27, 0.06,
                                       0.21, 0.10,
                                       2.66, 0.58,

                                       357.1,
                                       341,

                                       tColor1, tMarkerStyleB1);

  const FitInfo tFitInfo2b_LamK0 = FitInfo(TString("FreeRadii_FixedD0_3Res"), 
                                       kLamK0,
                                       true, true, false, false, 
                                       0.5*(1.50+1.50), 0.5*(0.90+0.90),
                                       0.5*(1.50+1.50), 0.5*(0.90+0.90),
                                       0.5*(1.50+1.50), 0.5*(0.90+0.90),

                                       3.48, 0.96,
                                       2.63, 0.70,
                                       1.89, 0.49,

                                      -0.08, 0.03,
                                       0.15, 0.11,
                                       0.00, 0.00,

                                       359.9,
                                       342,

                                       tColor1, tMarkerStyleB2);


  const FitInfo tFitInfo3b_LamK0 = FitInfo(TString("FixedLambda_FreeD0_3Res"), 
                                       kLamK0,
                                       false, true, true, false, 
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00),
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00),
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00),

                                       3.10, 0.55,
                                       2.36, 0.43,
                                       1.73, 0.32,

                                      -0.15, 0.03,
                                       0.15, 0.07,
                                       3.58, 1.03,

                                       357.4,
                                       342,

                                       tColor2, tMarkerStyleB1);

  const FitInfo tFitInfo4b_LamK0 = FitInfo(TString("FixedLambda_FixedD0_3Res"), 
                                       kLamK0,
                                       false, true, false, false, 
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00),
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00),
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00),

                                       3.25, 0.83,
                                       2.45, 0.60,
                                       1.76, 0.42,

                                      -0.12, 0.04,
                                       0.19, 0.15,
                                       0.00, 0.00,

                                       360.3,
                                       343,

                                       tColor2, tMarkerStyleB2);

  const FitInfo tFitInfo5b_LamK0 = FitInfo(TString("FixedRadii_FreeD0_3Res"), 
                                       kLamK0,
                                       true, false, true, false, 
                                       0.5*(0.60+0.60), 0.5*(0.63+0.63),
                                       0.5*(0.60+0.60), 0.5*(0.63+0.63),
                                       0.5*(0.60+0.60), 0.5*(0.63+0.63),

                                       3.25, 0.00,
                                       2.75, 0.00,
                                       2.25, 0.00,

                                      -0.35, 0.08,
                                       0.27, 0.05,
                                       2.72, 0.54,

                                       367.9,
                                       344,

                                       tColor3, tMarkerStyleB1);

  const FitInfo tFitInfo6b_LamK0 = FitInfo(TString("FixedRadii_FixedD0_3Res"), 
                                       kLamK0,
                                       true, false, false, false, 
                                       0.5*(1.50+1.50), 0.5*(0.79+0.79),
                                       0.5*(1.50+1.50), 0.5*(0.79+0.79),
                                       0.5*(1.50+1.50), 0.5*(0.79+0.79),

                                       3.25, 0.00,
                                       2.75, 0.00,
                                       2.25, 0.00,

                                      -0.10, 0.03,
                                       0.14, 0.02,
                                       0.00, 0.00,

                                       372.0,
                                       345,

                                       tColor3, tMarkerStyleB2);

//------------------------------------------------------------------------------------------------
//              QM 2017 Results, i.e. without residuals
//------------------------------------------------------------------------------------------------
  Color_t tColorQM_LamKchP = kRed+1;
  Color_t tColorQM_LamKchM = kBlue+1;
  Color_t tColorQM_LamK0 = kBlack;


  const FitInfo tFitInfoQM_LamKchP = FitInfo(TString("QM2017"), 
                                       kLamKchP,
                                       true, true, true, false, 
                                       0.5*(0.38+0.37), 0.5*(0.09+0.08),
                                       0.5*(0.48+0.41), 0.5*(0.13+0.11),
                                       0.5*(0.64+0.62), 0.5*(0.20+0.19),

                                       4.04, 0.38,
                                       3.92, 0.45,
                                       3.72, 0.55,

                                      -0.69, 0.16,
                                       0.39, 0.14,
                                       0.64, 0.53,

                                       425.8,
                                       336,

                                       tColorQM_LamKchP, tMarkerStyleA1);

  const FitInfo tFitInfoQM_LamKchM = FitInfo(TString("QM2017"), 
                                       kLamKchM,
                                       true, true, true, false, 
                                       0.5*(0.45+0.48), 0.5*(0.16+0.17),
                                       0.5*(0.40+0.49), 0.5*(0.15+0.18),
                                       0.5*(0.20+0.22), 0.5*(0.08+0.08),

                                       4.79, 0.79,
                                       4.00, 0.72,
                                       2.11, 0.52,

                                       0.18, 0.13,
                                       0.45, 0.18,
                                      -5.29, 2.94,

                                       284.0,
                                       288,

                                       tColorQM_LamKchM, tMarkerStyleA1);

  const FitInfo tFitInfoQM_LamK0 = FitInfo(TString("QM2017"), 
                                       kLamK0,
                                       true, true, true, false, 
                                       0.5*(0.40+0.40), 0.5*(0.19+0.19),
                                       0.5*(0.40+0.40), 0.5*(0.19+0.19),
                                       0.5*(0.40+0.40), 0.5*(0.19+0.19),

                                       3.02, 0.54,
                                       2.27, 0.41,
                                       1.67, 0.30,

                                      -0.16, 0.03,
                                       0.18, 0.08,
                                       3.57, 0.95,

                                       357.0,
                                       341,

                                       tColorQM_LamK0, tMarkerStyleA1);


//------------------------------------------------------------------------------------------------

  const vector<FitInfo> tFitInfoVec_LamKchP{tFitInfo1a_LamKchP, tFitInfo2a_LamKchP, tFitInfo3a_LamKchP, tFitInfo4a_LamKchP, tFitInfo5a_LamKchP, tFitInfo6a_LamKchP,
                                            tFitInfo1b_LamKchP, tFitInfo2b_LamKchP, tFitInfo3b_LamKchP, tFitInfo4b_LamKchP, tFitInfo5b_LamKchP, tFitInfo6b_LamKchP};

  const vector<FitInfo> tFitInfoVec_LamKchM{tFitInfo1a_LamKchM, tFitInfo2a_LamKchM, tFitInfo3a_LamKchM, tFitInfo4a_LamKchM, tFitInfo5a_LamKchM, tFitInfo6a_LamKchM,
                                            tFitInfo1b_LamKchM, tFitInfo2b_LamKchM, tFitInfo3b_LamKchM, tFitInfo4b_LamKchM, tFitInfo5b_LamKchM, tFitInfo6b_LamKchM};

  const vector<FitInfo> tFitInfoVec_LamK0{tFitInfo1a_LamK0, tFitInfo2a_LamK0, tFitInfo3a_LamK0, tFitInfo4a_LamK0, tFitInfo5a_LamK0, tFitInfo6a_LamK0,
                                          tFitInfo1b_LamK0, tFitInfo2b_LamK0, tFitInfo3b_LamK0, tFitInfo4b_LamK0, tFitInfo5b_LamK0, tFitInfo6b_LamK0};




