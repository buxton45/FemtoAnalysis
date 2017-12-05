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
enum AverageType {kAverage=0, kWeightedMean=1};

enum IncludeResType {kInclude10ResOnly=0, kInclude3ResOnly=1, kInclude10ResAnd3Res=2, kIncludeNoRes=3};
const char* const cIncludeResTypeTags[4] = {"_10Res", "_3Res", "_10ResAnd3Res", ""};

enum IncludeD0Type {kFreeD0Only=0, kFixedD0Only=1, kFreeAndFixedD0=2};
const char* const cIncludeD0TypeTags[3] = {"_FreeD0Only", "_FixedD0Only", "_FreeAndFixedD0"};

enum IncludeRadiiType {kFreeRadiiOnly=0, kFixedRadiiOnly=1, kFreeAndFixedRadii=2};
const char* const cIncludeRadiiTypeTags[3] = {"_FreeRadiiOnly", "_FixedRadiiOnly", "_FreeAndFixedRadii"};

enum IncludeLambdaType {kFreeLambdaOnly=0, kFixedLambdaOnly=1, kFreeAndFixedLambda=2};
const char* const cIncludeLambdaTypeTags[3] = {"_FreeLambdaOnly", "_FixedLambdaOnly", "_FreeAndFixedLambda"};


enum Plot10and3Type {kPlot10and3SeparateOnly=0, kPlot10and3AvgOnly=1, kPlot10and3SeparateAndAvg=2};
const char* const cPlot10and3TypeTage[3] = {"_10and3SeparateOnly", "_10and3AvgOnly", "_10and3SeparateandAvg"};

//---------------------------------------------------------------------------------------------------------------------------------
struct FitInfo
{
  TString descriptor;
  AnalysisType analysisType;

  double lambda1, lambda2, lambda3;
  vector<double> lambdaVec;

  double radius1, radius2, radius3;
  vector<double> radiusVec;

  double ref0, imf0, d0;


  double lambdaStatErr1, lambdaStatErr2, lambdaStatErr3;
  vector<double> lambdaStatErrVec;

  double radiusStatErr1, radiusStatErr2, radiusStatErr3; 
  vector<double> radiusStatErrVec;

  double ref0StatErr, imf0StatErr, d0StatErr;

  double lambdaSysErr1, lambdaSysErr2, lambdaSysErr3;
  vector<double> lambdaSysErrVec;

  double radiusSysErr1, radiusSysErr2, radiusSysErr3; 
  vector<double> radiusSysErrVec;

  double ref0SysErr, imf0SysErr, d0SysErr;

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
          double aLambda1, double aLambdaStatErr1, double aLambdaSysErr1, 
          double aLambda2, double aLambdaStatErr2, double aLambdaSysErr2, 
          double aLambda3, double aLambdaStatErr3, double aLambdaSysErr3, 
          double aRadius1, double aRadiusStatErr1, double aRadiusSysErr1, 
          double aRadius2, double aRadiusStatErr2, double aRadiusSysErr2, 
          double aRadius3, double aRadiusStatErr3, double aRadiusSysErr3, 
          double aReF0,    double aReF0StatErr,    double aReF0SysErr,
          double aImF0,    double aImF0StatErr,    double aImF0SysErr,
          double aD0,      double aD0StatErr,      double aD0SysErr,
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

    lambdaSysErr1 = aLambdaSysErr1;
    lambdaSysErr2 = aLambdaSysErr2;
    lambdaSysErr3 = aLambdaSysErr3;

    radiusSysErr1 = aRadiusSysErr1;
    radiusSysErr2 = aRadiusSysErr2;
    radiusSysErr3 = aRadiusSysErr3;

    ref0SysErr   = aReF0SysErr;
    imf0SysErr   = aImF0SysErr;
    d0SysErr     = aD0SysErr;

    chi2 = aChi2;
    ndf = aNDF;

    markerColor = aMarkerColor;
    markerStyle = aMarkerStyle;

    //-------------------------
    lambdaVec = vector<double>{lambda1, lambda2, lambda3};
    radiusVec = vector<double>{radius1, radius2, radius3};
    lambdaStatErrVec = vector<double>{lambdaStatErr1, lambdaStatErr2, lambdaStatErr3};
    radiusStatErrVec = vector<double>{radiusStatErr1, radiusStatErr2, radiusStatErr3};
    lambdaSysErrVec = vector<double>{lambdaSysErr1, lambdaSysErr2, lambdaSysErr3};
    radiusSysErrVec = vector<double>{radiusSysErr1, radiusSysErr2, radiusSysErr3};
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
                                       0.5*(0.96+0.94), 0.5*(0.37+0.36), 0.5*(0.00+0.00),
                                       0.5*(1.18+0.99), 0.5*(0.51+0.42), 0.5*(0.00+0.00),
                                       0.5*(1.01+0.98), 0.5*(0.30+0.29), 0.5*(0.00+0.00),

                                       4.98, 0.96, 0.00,
                                       4.76, 0.99, 0.00,
                                       3.55, 0.52, 0.00,

                                      -1.51, 0.37, 0.00,
                                       0.65, 0.40, 0.00,
                                       1.13, 0.74, 0.00,

                                       424.2,
                                       336,

                                       tColor1, tMarkerStyleA1);

  const FitInfo tFitInfo2a_LamKchP = FitInfo(TString("FreeRadii_FixedD0_10Res"), 
                                       kLamKchP,
                                       true, true, false, true, 
                                       0.5*(1.33+1.32), 0.5*(0.40+0.40), 0.5*(0.00+0.00),
                                       0.5*(1.56+1.30), 0.5*(0.63+0.52), 0.5*(0.00+0.00),
                                       0.5*(1.16+1.12), 0.5*(0.31+0.30), 0.5*(0.00+0.00),

                                       5.26, 0.87, 0.00,
                                       4.91, 1.00, 0.00,
                                       3.37, 0.47, 0.00,

                                      -1.08, 0.14, 0.00,
                                       0.59, 0.30, 0.00,
                                       0.00, 0.00, 0.00,

                                       425.9,
                                       337,

                                       tColor1, tMarkerStyleA2);


  const FitInfo tFitInfo3a_LamKchP = FitInfo(TString("FixedLambda_FreeD0_10Res"), 
                                       kLamKchP,
                                       false, true, true, true, 
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00), 0.5*(0.00+0.00),
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00), 0.5*(0.00+0.00),
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00), 0.5*(0.00+0.00),

                                       5.09, 0.54, 0.00,
                                       4.57, 0.48, 0.00,
                                       3.55, 0.35, 0.00,

                                      -1.49, 0.24, 0.00,
                                       0.66, 0.38, 0.00,
                                       1.10, 0.59, 0.00,

                                       428.3,
                                       342,

                                       tColor2, tMarkerStyleA1);

  const FitInfo tFitInfo4a_LamKchP = FitInfo(TString("FixedLambda_FixedD0_10Res"), 
                                       kLamKchP,
                                       false, true, false, true, 
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00), 0.5*(0.00+0.00),
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00), 0.5*(0.00+0.00),
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00), 0.5*(0.00+0.00),

                                       4.67, 0.43, 0.00,
                                       4.17, 0.37, 0.00,
                                       3.22, 0.26, 0.00,

                                      -1.18, 0.13, 0.00,
                                       0.60, 0.30, 0.00,
                                       0.00, 0.00, 0.00,

                                       431.3,
                                       343,

                                       tColor2, tMarkerStyleA2);

  const FitInfo tFitInfo5a_LamKchP = FitInfo(TString("FixedRadii_FreeD0_10Res"), 
                                       kLamKchP,
                                       true, false, true, true, 
                                       0.5*(0.84+0.83), 0.5*(0.27+0.27), 0.5*(0.00+0.00),
                                       0.5*(0.96+0.81), 0.5*(0.31+0.26), 0.5*(0.00+0.00),
                                       0.5*(0.81+0.79), 0.5*(0.27+0.27), 0.5*(0.00+0.00),

                                       3.50, 0.00, 0.00,
                                       3.25, 0.00, 0.00,
                                       2.50, 0.00, 0.00,

                                      -1.02, 0.36, 0.00,
                                       0.08, 0.06, 0.00,
                                       0.92, 0.38, 0.00,

                                       432.6,
                                       339,

                                       tColor3, tMarkerStyleA1);

  const FitInfo tFitInfo6a_LamKchP = FitInfo(TString("FixedRadii_FixedD0_10Res"), 
                                       kLamKchP,
                                       true, false, false, true, 
                                       0.5*(1.03+1.03), 0.5*(0.32+0.32), 0.5*(0.00+0.00),
                                       0.5*(1.18+1.00), 0.5*(0.36+0.30), 0.5*(0.00+0.00),
                                       0.5*(1.05+1.02), 0.5*(0.32+0.32), 0.5*(0.00+0.00),

                                       3.50, 0.00, 0.00,
                                       3.25, 0.00, 0.00,
                                       2.50, 0.00, 0.00,

                                      -0.76, 0.23, 0.00,
                                       0.12, 0.07, 0.00,
                                       0.00, 0.00, 0.00,

                                       435.1,
                                       340,

                                       tColor3, tMarkerStyleA2);


  //--------------- 3 Residuals ----------
  const FitInfo tFitInfo1b_LamKchP = FitInfo(TString("FreeRadii_FreeD0_3Res"), 
                                       kLamKchP,
                                       true, true, true, false, 
                                       0.5*(0.84+0.82), 0.5*(0.31+0.30), 0.5*(0.00+0.00),
                                       0.5*(1.09+0.90), 0.5*(0.47+0.39), 0.5*(0.00+0.00),
                                       0.5*(1.02+0.97), 0.5*(0.30+0.29), 0.5*(0.00+0.00),

                                       4.43, 0.86, 0.00,
                                       4.34, 0.94, 0.00,
                                       3.38, 0.50, 0.00,

                                      -1.24, 0.32, 0.00,
                                       0.50, 0.35, 0.00,
                                       1.11, 0.51, 0.00,

                                       427.2,
                                       336,

                                       tColor1, tMarkerStyleB1);

  const FitInfo tFitInfo2b_LamKchP = FitInfo(TString("FreeRadii_FixedD0_3Res"), 
                                       kLamKchP,
                                       true, true, false, false, 
                                       0.5*(1.23+1.21), 0.5*(0.37+0.37), 0.5*(0.00+0.00),
                                       0.5*(1.54+1.27), 0.5*(0.59+0.48), 0.5*(0.00+0.00),
                                       0.5*(1.22+1.17), 0.5*(0.32+0.31), 0.5*(0.00+0.00),

                                       4.82, 0.81, 0.00,
                                       4.62, 0.90, 0.00,
                                       3.27, 0.44, 0.00,

                                      -0.85, 0.13, 0.00,
                                       0.49, 0.24, 0.00,
                                       0.00, 0.00, 0.00,

                                       430.0,
                                       337,

                                       tColor1, tMarkerStyleB2);


  const FitInfo tFitInfo3b_LamKchP = FitInfo(TString("FixedLambda_FreeD0_3Res"), 
                                       kLamKchP,
                                       false, true, true, false, 
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00), 0.5*(0.00+0.00),
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00), 0.5*(0.00+0.00),
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00), 0.5*(0.00+0.00),

                                       4.84, 0.60, 0.00,
                                       4.34, 0.53, 0.00,
                                       3.37, 0.38, 0.00,

                                      -1.16, 0.19, 0.00,
                                       0.55, 0.33, 0.00,
                                       1.02, 0.43, 0.00,

                                       432.1,
                                       342,

                                       tColor2, tMarkerStyleB1);

  const FitInfo tFitInfo4b_LamKchP = FitInfo(TString("FixedLambda_FixedD0_3Res"), 
                                       kLamKchP,
                                       false, true, false, false, 
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00), 0.5*(0.00+0.00),
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00), 0.5*(0.00+0.00),
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00), 0.5*(0.00+0.00),

                                       4.48, 0.42, 0.00,
                                       4.01, 0.36, 0.00,
                                       3.09, 0.26, 0.00,

                                      -0.94, 0.10, 0.00,
                                       0.51, 0.23, 0.00,
                                       0.00, 0.00, 0.00,

                                       435.8,
                                       343,

                                       tColor2, tMarkerStyleB2);

  const FitInfo tFitInfo5b_LamKchP = FitInfo(TString("FixedRadii_FreeD0_3Res"), 
                                       kLamKchP,
                                       true, false, true, false, 
                                       0.5*(0.83+0.82), 0.5*(0.27+0.27), 0.5*(0.00+0.00),
                                       0.5*(0.96+0.80), 0.5*(0.31+0.26), 0.5*(0.00+0.00),
                                       0.5*(0.82+0.80), 0.5*(0.28+0.28), 0.5*(0.00+0.00),

                                       3.50, 0.00, 0.00,
                                       3.25, 0.00, 0.00,
                                       2.50, 0.00, 0.00,

                                      -0.90, 0.32, 0.00,
                                       0.12, 0.06, 0.00,
                                       1.06, 0.27, 0.00,

                                       432.6,
                                       339,

                                       tColor3, tMarkerStyleB1);

  const FitInfo tFitInfo6b_LamKchP = FitInfo(TString("FixedRadii_FixedD0_3Res"), 
                                       kLamKchP,
                                       true, false, false, false, 
                                       0.5*(1.04+1.03), 0.5*(0.29+0.29), 0.5*(0.00+0.00),
                                       0.5*(1.20+1.00), 0.5*(0.33+0.28), 0.5*(0.00+0.00),
                                       0.5*(1.08+1.04), 0.5*(0.31+0.30), 0.5*(0.00+0.00),

                                       3.50, 0.00, 0.00,
                                       3.25, 0.00, 0.00,
                                       2.50, 0.00, 0.00,

                                      -0.66, 0.18, 0.00,
                                       0.15, 0.07, 0.00,
                                       0.00, 0.00, 0.00,

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
                                       0.5*(1.50+1.49), 0.5*(0.72+0.69), 0.5*(0.00+0.00),
                                       0.5*(1.15+1.41), 0.5*(0.51+0.61), 0.5*(0.00+0.00),
                                       0.5*(1.07+0.80), 0.5*(0.73+0.37), 0.5*(0.00+0.00),

                                       6.21, 1.51, 0.00,
                                       4.86, 1.15, 0.00,
                                       2.86, 0.81, 0.00,

                                       0.45, 0.23, 0.00,
                                       0.52, 0.21, 0.00,
                                      -4.81, 2.62, 0.00,

                                       286.2,
                                       288,

                                       tColor1, tMarkerStyleA1);

  const FitInfo tFitInfo2a_LamKchM = FitInfo(TString("FreeRadii_FixedD0_10Res"), 
                                       kLamKchM,
                                       true, true, false, true, 
                                       0.5*(1.16+1.17), 0.5*(0.63+0.61), 0.5*(0.00+0.00),
                                       0.5*(0.97+1.19), 0.5*(0.52+0.63), 0.5*(0.00+0.00),
                                       0.5*(0.89+0.72), 0.5*(0.70+0.41), 0.5*(0.00+0.00),

                                       4.73, 0.77, 0.00,
                                       3.91, 0.70, 0.00,
                                       2.34, 0.53, 0.00,

                                       0.34, 0.17, 0.00,
                                       0.52, 0.27, 0.00,
                                       0.00, 0.00, 0.00,

                                       290.1,
                                       289,

                                       tColor1, tMarkerStyleA2);


  const FitInfo tFitInfo3a_LamKchM = FitInfo(TString("FixedLambda_FreeD0_10Res"), 
                                       kLamKchM,
                                       false, true, true, true, 
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00), 0.5*(0.00+0.00),
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00), 0.5*(0.00+0.00),
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00), 0.5*(0.00+0.00),

                                       5.15, 0.26, 0.00,
                                       4.44, 0.28, 0.00,
                                       3.22, 0.31, 0.00,

                                       0.54, 0.18, 0.00,
                                       0.69, 0.15, 0.00,
                                      -3.31, 1.41, 0.00,

                                       293.4,
                                       294,

                                       tColor2, tMarkerStyleA1);

  const FitInfo tFitInfo4a_LamKchM = FitInfo(TString("FixedLambda_FixedD0_10Res"), 
                                       kLamKchM,
                                       false, true, false, true, 
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00), 0.5*(0.00+0.00),
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00), 0.5*(0.00+0.00),
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00), 0.5*(0.00+0.00),

                                       4.52, 0.36, 0.00,
                                       3.89, 0.34, 0.00,
                                       2.77, 0.32, 0.00,

                                       0.40, 0.17, 0.00,
                                       0.60, 0.15, 0.00,
                                       0.00, 0.00, 0.00,

                                       296.2,
                                       295,

                                       tColor2, tMarkerStyleA2);

  const FitInfo tFitInfo5a_LamKchM = FitInfo(TString("FixedRadii_FreeD0_10Res"), 
                                       kLamKchM,
                                       true, false, true, true, 
                                       0.5*(0.58+0.60), 0.5*(0.27+0.27), 0.5*(0.00+0.00),
                                       0.5*(0.58+0.72), 0.5*(0.26+0.32), 0.5*(0.00+0.00),
                                       0.5*(0.63+0.56), 0.5*(0.34+0.24), 0.5*(0.00+0.00),

                                       3.50, 0.00, 0.00,
                                       3.25, 0.00, 0.00,
                                       2.50, 0.00, 0.00,

                                       0.23, 0.20, 0.00,
                                       0.64, 0.51, 0.00,
                                       1.81, 4.68, 0.00,

                                       297.3,
                                       291,

                                       tColor3, tMarkerStyleA1);

  const FitInfo tFitInfo6a_LamKchM = FitInfo(TString("FixedRadii_FixedD0_10Res"), 
                                       kLamKchM,
                                       true, false, false, true, 
                                       0.5*(0.55+0.57), 0.5*(0.15+0.15), 0.5*(0.00+0.00),
                                       0.5*(0.56+0.69), 0.5*(0.15+0.18), 0.5*(0.00+0.00),
                                       0.5*(0.60+0.54), 0.5*(0.22+0.16), 0.5*(0.00+0.00),

                                       3.50, 0.00, 0.00,
                                       3.25, 0.00, 0.00,
                                       2.50, 0.00, 0.00,

                                       0.25, 0.15, 0.00,
                                       0.71, 0.33, 0.00,
                                       0.00, 0.00, 0.00,

                                       297.2,
                                       292,

                                       tColor3, tMarkerStyleA2);



  //--------------- 3 Residuals ----------
  const FitInfo tFitInfo1b_LamKchM = FitInfo(TString("FreeRadii_FreeD0_3Res"), 
                                       kLamKchM,
                                       true, true, true, false, 
                                       0.5*(1.55+1.54), 0.5*(0.72+0.69), 0.5*(0.00+0.00),
                                       0.5*(1.19+1.46), 0.5*(0.51+0.61), 0.5*(0.00+0.00),
                                       0.5*(1.08+0.80), 0.5*(0.71+0.36), 0.5*(0.00+0.00),

                                       6.02, 1.46, 0.00,
                                       4.74, 1.11, 0.00,
                                       2.75, 0.80, 0.00,

                                       0.34, 0.21, 0.00,
                                       0.42, 0.19, 0.00,
                                      -5.72, 3.39, 0.00,

                                       285.0,
                                       288,

                                       tColor1, tMarkerStyleB1);

  const FitInfo tFitInfo2b_LamKchM = FitInfo(TString("FreeRadii_FixedD0_3Res"), 
                                       kLamKchM,
                                       true, true, false, false, 
                                       0.5*(1.20+1.20), 0.5*(0.60+0.58), 0.5*(0.00+0.00),
                                       0.5*(1.01+1.24), 0.5*(0.51+0.62), 0.5*(0.00+0.00),
                                       0.5*(0.94+0.75), 0.5*(0.73+0.41), 0.5*(0.00+0.00),

                                       4.33, 0.64, 0.00,
                                       3.61, 0.62, 0.00,
                                       2.12, 0.46, 0.00,

                                       0.21, 0.14, 0.00,
                                       0.39, 0.22, 0.00,
                                       0.00, 0.00, 0.00,

                                       290.6,
                                       289,

                                       tColor1, tMarkerStyleB2);


  const FitInfo tFitInfo3b_LamKchM = FitInfo(TString("FixedLambda_FreeD0_3Res"), 
                                       kLamKchM,
                                       false, true, true, false, 
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00), 0.5*(0.00+0.00),
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00), 0.5*(0.00+0.00),
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00), 0.5*(0.00+0.00),

                                       4.95, 0.33, 0.00,
                                       4.27, 0.32, 0.00,
                                       3.13, 0.32, 0.00,

                                       0.40, 0.16, 0.00,
                                       0.57, 0.13, 0.00,
                                      -3.83, 1.52, 0.00,

                                       292.8,
                                       294,

                                       tColor2, tMarkerStyleB1);

  const FitInfo tFitInfo4b_LamKchM = FitInfo(TString("FixedLambda_FixedD0_3Res"), 
                                       kLamKchM,
                                       false, true, false, false, 
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00), 0.5*(0.00+0.00),
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00), 0.5*(0.00+0.00),
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00), 0.5*(0.00+0.00),

                                       4.12, 0.44, 0.00,
                                       3.56, 0.39, 0.00,
                                       2.50, 0.35, 0.00,

                                       0.25, 0.14, 0.00,
                                       0.45, 0.13, 0.00,
                                       0.00, 0.00, 0.00,

                                       297.0,
                                       295,

                                       tColor2, tMarkerStyleB2);

  const FitInfo tFitInfo5b_LamKchM = FitInfo(TString("FixedRadii_FreeD0_3Res"), 
                                       kLamKchM,
                                       true, false, true, false, 
                                       0.5*(0.65+0.67), 0.5*(0.32+0.32), 0.5*(0.00+0.00),
                                       0.5*(0.65+0.81), 0.5*(0.31+0.38), 0.5*(0.00+0.00),
                                       0.5*(0.72+0.62), 0.5*(0.44+0.29), 0.5*(0.00+0.00),

                                       3.50, 0.00, 0.00,
                                       3.25, 0.00, 0.00,
                                       2.50, 0.00, 0.00,

                                       0.12, 0.13, 0.00,
                                       0.45, 0.36, 0.00,
                                      -4.68, 4.91, 0.00,

                                       295.3,
                                       291,

                                       tColor3, tMarkerStyleB1);

  const FitInfo tFitInfo6b_LamKchM = FitInfo(TString("FixedRadii_FixedD0_3Res"), 
                                       kLamKchM,
                                       true, false, false, false, 
                                       0.5*(0.78+0.80), 0.5*(0.38+0.38), 0.5*(0.00+0.00),
                                       0.5*(0.78+0.96), 0.5*(0.37+0.45), 0.5*(0.00+0.00),
                                       0.5*(0.93+0.74), 0.5*(0.62+0.36), 0.5*(0.00+0.00),

                                       3.50, 0.00, 0.00,
                                       3.25, 0.00, 0.00,
                                       2.50, 0.00, 0.00,

                                       0.15, 0.10, 0.00,
                                       0.42, 0.28, 0.00,
                                       0.00, 0.00, 0.00,

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
                                       0.5*(0.60+0.60), 0.5*(0.76+0.76), 0.5*(0.00+0.00),
                                       0.5*(0.60+0.60), 0.5*(0.76+0.76), 0.5*(0.00+0.00),
                                       0.5*(0.60+0.60), 0.5*(0.76+0.76), 0.5*(0.00+0.00),

                                       2.97, 0.49, 0.00,
                                       2.30, 0.39, 0.00,
                                       1.70, 0.29, 0.00,

                                      -0.26, 0.07, 0.00,
                                       0.17, 0.07, 0.00,
                                       2.53, 0.68, 0.00,

                                       357.7,
                                       341,

                                       tColor1, tMarkerStyleA1);

  const FitInfo tFitInfo2a_LamK0 = FitInfo(TString("FreeRadii_FixedD0_10Res"), 
                                       kLamK0,
                                       true, true, false, true, 
                                       0.5*(1.50+1.44), 0.5*(0.65+0.16), 0.5*(0.00+0.00),
                                       0.5*(0.60+0.60), 0.5*(0.17+0.75), 0.5*(0.00+0.00),
                                       0.5*(1.50+1.40), 0.5*(0.59+0.24), 0.5*(0.00+0.00),

                                       3.53, 1.01, 0.00,
                                       1.72, 0.45, 0.00,
                                       1.99, 0.53, 0.00,

                                      -0.12, 0.03, 0.00,
                                       0.11, 0.10, 0.00,
                                       0.00, 0.00, 0.00,

                                       354.1,
                                       337,

                                       tColor1, tMarkerStyleA2);


  const FitInfo tFitInfo3a_LamK0 = FitInfo(TString("FixedLambda_FreeD0_10Res"), 
                                       kLamK0,
                                       false, true, true, true, 
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00), 0.5*(0.00+0.00),
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00), 0.5*(0.00+0.00),
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00), 0.5*(0.00+0.00),

                                       3.16, 0.52, 0.00,
                                       2.44, 0.42, 0.00,
                                       1.79, 0.31, 0.00,

                                      -0.15, 0.04, 0.00,
                                       0.13, 0.05, 0.00,
                                       3.50, 1.14, 0.00,

                                       358.0,
                                       342,

                                       tColor2, tMarkerStyleA1);

  const FitInfo tFitInfo4a_LamK0 = FitInfo(TString("FixedLambda_FixedD0_10Res"), 
                                       kLamK0,
                                       false, true, false, true, 
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00), 0.5*(0.00+0.00),
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00), 0.5*(0.00+0.00),
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00), 0.5*(0.00+0.00),

                                       3.32, 0.88, 0.00,
                                       2.54, 0.65, 0.00,
                                       1.84, 0.45, 0.00,

                                      -0.13, 0.04, 0.00,
                                       0.16, 0.12, 0.00,
                                       0.00, 0.00, 0.00,

                                       360.1,
                                       343,

                                       tColor2, tMarkerStyleA2);

  const FitInfo tFitInfo5a_LamK0 = FitInfo(TString("FixedRadii_FreeD0_10Res"), 
                                       kLamK0,
                                       true, false, true, true, 
                                       0.5*(1.50+1.50), 0.5*(0.83+0.83), 0.5*(0.00+0.00),
                                       0.5*(1.50+1.50), 0.5*(0.83+0.83), 0.5*(0.00+0.00),
                                       0.5*(1.50+1.50), 0.5*(0.83+0.83), 0.5*(0.00+0.00),

                                       3.25, 0.00, 0.00,
                                       2.75, 0.00, 0.00,
                                       2.25, 0.00, 0.00,

                                      -0.11, 0.03, 0.00,
                                       0.10, 0.01, 0.00,
                                      -0.73, 2.57, 0.00,

                                       370.1,
                                       344,

                                       tColor3, tMarkerStyleA1);

  const FitInfo tFitInfo6a_LamK0 = FitInfo(TString("FixedRadii_FixedD0_10Res"), 
                                       kLamK0,
                                       true, false, false, true, 
                                       0.5*(0.99+0.93), 0.5*(0.16+0.16), 0.5*(0.00+0.00),
                                       0.5*(1.11+1.14), 0.5*(0.20+0.20), 0.5*(0.00+0.00),
                                       0.5*(1.50+1.40), 0.5*(0.89+0.65), 0.5*(0.00+0.00),

                                       3.25, 0.00, 0.00,
                                       2.75, 0.00, 0.00,
                                       2.25, 0.00, 0.00,

                                      -0.12, 0.04, 0.00,
                                       0.16, 0.04, 0.00,
                                       0.00, 0.00, 0.00,

                                       360.6,
                                       340,

                                       tColor3, tMarkerStyleA2);

  //--------------- 3 Residuals ----------
  const FitInfo tFitInfo1b_LamK0 = FitInfo(TString("FreeRadii_FreeD0_3Res"), 
                                       kLamK0,
                                       true, true, true, false, 
                                       0.5*(0.60+0.60), 0.5*(0.77+0.77), 0.5*(0.00+0.00),
                                       0.5*(0.60+0.60), 0.5*(0.77+0.77), 0.5*(0.00+0.00),
                                       0.5*(0.60+0.60), 0.5*(0.77+0.77), 0.5*(0.00+0.00),

                                       2.91, 0.51, 0.00,
                                       2.22, 0.40, 0.00,
                                       1.64, 0.30, 0.00,

                                      -0.27, 0.06, 0.00,
                                       0.21, 0.10, 0.00,
                                       2.66, 0.58, 0.00,

                                       357.1,
                                       341,

                                       tColor1, tMarkerStyleB1);

  const FitInfo tFitInfo2b_LamK0 = FitInfo(TString("FreeRadii_FixedD0_3Res"), 
                                       kLamK0,
                                       true, true, false, false, 
                                       0.5*(1.50+1.50), 0.5*(0.90+0.90), 0.5*(0.00+0.00),
                                       0.5*(1.50+1.50), 0.5*(0.90+0.90), 0.5*(0.00+0.00),
                                       0.5*(1.50+1.50), 0.5*(0.90+0.90), 0.5*(0.00+0.00),

                                       3.48, 0.96, 0.00,
                                       2.63, 0.70, 0.00,
                                       1.89, 0.49, 0.00,

                                      -0.08, 0.03, 0.00,
                                       0.15, 0.11, 0.00,
                                       0.00, 0.00, 0.00,

                                       359.9,
                                       342,

                                       tColor1, tMarkerStyleB2);


  const FitInfo tFitInfo3b_LamK0 = FitInfo(TString("FixedLambda_FreeD0_3Res"), 
                                       kLamK0,
                                       false, true, true, false, 
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00), 0.5*(0.00+0.00),
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00), 0.5*(0.00+0.00),
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00), 0.5*(0.00+0.00),

                                       3.10, 0.55, 0.00,
                                       2.36, 0.43, 0.00,
                                       1.73, 0.32, 0.00,

                                      -0.15, 0.03, 0.00,
                                       0.15, 0.07, 0.00,
                                       3.58, 1.03, 0.00,

                                       357.4,
                                       342,

                                       tColor2, tMarkerStyleB1);

  const FitInfo tFitInfo4b_LamK0 = FitInfo(TString("FixedLambda_FixedD0_3Res"), 
                                       kLamK0,
                                       false, true, false, false, 
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00), 0.5*(0.00+0.00),
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00), 0.5*(0.00+0.00),
                                       0.5*(1.00+1.00), 0.5*(0.00+0.00), 0.5*(0.00+0.00),

                                       3.25, 0.83, 0.00,
                                       2.45, 0.60, 0.00,
                                       1.76, 0.42, 0.00,

                                      -0.12, 0.04, 0.00,
                                       0.19, 0.15, 0.00,
                                       0.00, 0.00, 0.00,

                                       360.3,
                                       343,

                                       tColor2, tMarkerStyleB2);

  const FitInfo tFitInfo5b_LamK0 = FitInfo(TString("FixedRadii_FreeD0_3Res"), 
                                       kLamK0,
                                       true, false, true, false, 
                                       0.5*(0.60+0.60), 0.5*(0.63+0.63), 0.5*(0.00+0.00),
                                       0.5*(0.60+0.60), 0.5*(0.63+0.63), 0.5*(0.00+0.00),
                                       0.5*(0.60+0.60), 0.5*(0.63+0.63), 0.5*(0.00+0.00),

                                       3.25, 0.00, 0.00,
                                       2.75, 0.00, 0.00,
                                       2.25, 0.00, 0.00,

                                      -0.35, 0.08, 0.00,
                                       0.27, 0.05, 0.00,
                                       2.72, 0.54, 0.00,

                                       367.9,
                                       344,

                                       tColor3, tMarkerStyleB1);

  const FitInfo tFitInfo6b_LamK0 = FitInfo(TString("FixedRadii_FixedD0_3Res"), 
                                       kLamK0,
                                       true, false, false, false, 
                                       0.5*(1.50+1.50), 0.5*(0.79+0.79), 0.5*(0.00+0.00),
                                       0.5*(1.50+1.50), 0.5*(0.79+0.79), 0.5*(0.00+0.00),
                                       0.5*(1.50+1.50), 0.5*(0.79+0.79), 0.5*(0.00+0.00),

                                       3.25, 0.00, 0.00,
                                       2.75, 0.00, 0.00,
                                       2.25, 0.00, 0.00,

                                      -0.10, 0.03, 0.00,
                                       0.14, 0.02, 0.00,
                                       0.00, 0.00, 0.00,

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
                                       0.5*(0.38+0.37), 0.5*(0.09+0.08), 0.5*(0.22+0.22),
                                       0.5*(0.48+0.41), 0.5*(0.13+0.11), 0.5*(0.24+0.20),
                                       0.5*(0.64+0.62), 0.5*(0.20+0.19), 0.5*(0.20+0.20),

                                       4.045, 0.381, 0.830,
                                       3.923, 0.454, 0.663,
                                       3.717, 0.554, 0.420,

                                      -0.69, 0.16, 0.22,
                                       0.39, 0.14, 0.11,
                                       0.64, 0.53, 1.62,

                                       425.8,
                                       336,

                                       tColorQM_LamKchP, tMarkerStyleA1);

  const FitInfo tFitInfoQM_LamKchM = FitInfo(TString("QM2017"), 
                                       kLamKchM,
                                       true, true, true, false, 
                                       0.5*(0.45+0.48), 0.5*(0.16+0.17), 0.5*(0.19+0.15),
                                       0.5*(0.40+0.49), 0.5*(0.15+0.18), 0.5*(0.20+0.15),
                                       0.5*(0.20+0.22), 0.5*(0.08+0.08), 0.5*(0.13+0.11),

                                       4.787, 0.788, 1.375,
                                       4.001, 0.719, 0.978,
                                       2.112, 0.517, 0.457,

                                       0.18, 0.13, 0.10,
                                       0.45, 0.18, 0.18,
                                      -5.29, 2.94, 7.66,

                                       284.0,
                                       288,

                                       tColorQM_LamKchM, tMarkerStyleA1);

  const FitInfo tFitInfoQM_LamK0 = FitInfo(TString("QM2017"), 
                                       kLamK0,
                                       true, true, true, false, 
                                       0.5*(0.40+0.40), 0.5*(0.19+0.19), 0.5*(0.12+0.12),
                                       0.5*(0.40+0.40), 0.5*(0.19+0.19), 0.5*(0.12+0.12),
                                       0.5*(0.40+0.40), 0.5*(0.19+0.19), 0.5*(0.12+0.12),

                                       3.024, 0.541, 0.329,
                                       2.270, 0.413, 0.324,
                                       1.669, 0.307, 0.280,

                                      -0.16, 0.03, 0.04,
                                       0.18, 0.08, 0.06,
                                       3.57, 0.95, 2.84,

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

//---------------------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************************
//---------------------------------------------------------------------------------------------------------------------------------
td1dVec GetMean(td2dVec &aVecOfPointsWithErrors/*, AverageType tAvgType=kWeightedMean*/)
{
  //Each td1dVec in aVecOfPointsWithErrors = [Point, PointStatError, PointSysError]
  //Return td1dVec of same structure

  vector<double> tReturnVec{0.,0.,0.};

  double tNum=0., tDenStat=0., tDenSys=0.;
  double tMean=0., tMeanStatErr=0., tMeanSysErr=0.;

  double tPoint=0., tPointStatErr=0., tPointSysErr=0.;
  for(unsigned int i=0; i<aVecOfPointsWithErrors.size(); i++)
  {
    tPoint =        aVecOfPointsWithErrors[i][0];
    tPointStatErr = aVecOfPointsWithErrors[i][1];
    tPointSysErr =  aVecOfPointsWithErrors[i][2];

    if(tPointStatErr > 0.)
    {
      tNum += tPoint/(tPointStatErr*tPointStatErr);
      tDenStat += 1./(tPointStatErr*tPointStatErr);
    }
    else
    {
      tNum += tPoint;
      tDenStat += 1.;
    }
    if(tPointSysErr > 0.) tDenSys += 1./(tPointSysErr*tPointSysErr);
  }

  assert(tDenStat > 0.);
  tMean = tNum/tDenStat;
  if(tDenStat==(double)aVecOfPointsWithErrors.size()) tMeanStatErr=0.;  //in this case, not weighted avg
  else tMeanStatErr = sqrt(1./tDenStat);
  if(tDenSys > 0.) tMeanSysErr = sqrt(1./tDenSys);

  //--------------------------------------------
  tReturnVec[0] = tMean;
  tReturnVec[1] = tMeanStatErr;
  tReturnVec[2] = tMeanSysErr;

  return tReturnVec;
}



//---------------------------------------------------------------------------------------------------------------------------------
vector<FitInfo> GetFitInfoVec(AnalysisType aAnType, IncludeResType aIncludeResType=kInclude10ResAnd3Res, IncludeD0Type aIncludeD0Type=kFreeAndFixedD0,
                                                    IncludeRadiiType aIncludeRadiiType=kFreeAndFixedRadii, IncludeLambdaType aIncludeLambdaType=kFreeAndFixedLambda)
{
  vector<FitInfo> aFitInfoVec;
  if     (aAnType==kLamKchP) aFitInfoVec = tFitInfoVec_LamKchP;
  else if(aAnType==kLamKchM) aFitInfoVec = tFitInfoVec_LamKchM;
  else if(aAnType==kLamK0) aFitInfoVec = tFitInfoVec_LamK0;
  else assert(0);
  //------------------------------
  vector<FitInfo> tReturnVec;
  //------------------------------

  bool bPassRes=false, bPassD0=false, bPassRadii=false, bPassLambda=false;
  for(unsigned int i=0; i<aFitInfoVec.size(); i++)
  {
    if(aIncludeResType==kInclude10ResAnd3Res) bPassRes = true;
    else if(aIncludeResType==kInclude10ResOnly && aFitInfoVec[i].all10ResidualsUsed) bPassRes = true;
    else if(aIncludeResType==kInclude3ResOnly && !aFitInfoVec[i].all10ResidualsUsed) bPassRes = true;
    else bPassRes = false;

    if(aIncludeD0Type==kFreeAndFixedD0) bPassD0 = true;
    else if(aIncludeD0Type==kFreeD0Only && aFitInfoVec[i].freeD0) bPassD0 = true;
    else if(aIncludeD0Type==kFixedD0Only && !aFitInfoVec[i].freeD0) bPassD0 = true;
    else bPassD0 = false;

    if(aIncludeRadiiType==kFreeAndFixedRadii) bPassRadii = true;
    else if(aIncludeRadiiType==kFreeRadiiOnly && aFitInfoVec[i].freeRadii) bPassRadii = true;
    else if(aIncludeRadiiType==kFixedRadiiOnly && !aFitInfoVec[i].freeRadii) bPassRadii = true;
    else bPassRadii = false;

    if(aIncludeLambdaType==kFreeAndFixedLambda) bPassLambda = true;
    else if(aIncludeLambdaType==kFreeLambdaOnly && aFitInfoVec[i].freeLambda) bPassLambda = true;
    else if(aIncludeLambdaType==kFixedLambdaOnly && !aFitInfoVec[i].freeLambda) bPassLambda = true;
    else bPassLambda = false;

    if(bPassRes && bPassD0 && bPassRadii && bPassLambda) tReturnVec.push_back(aFitInfoVec[i]);
  }

  return tReturnVec;
}

//---------------------------------------------------------------------------------------------------------------------------------
bool IncludeFitInfoInMeanCalculation(FitInfo &aFitInfo, IncludeResType aIncludeResType, IncludeD0Type aIncludeD0Type)
{
  if(aIncludeResType==kInclude10ResOnly && !aFitInfo.all10ResidualsUsed) return false;
  if(aIncludeResType==kInclude3ResOnly  && aFitInfo.all10ResidualsUsed) return false;

  if(aIncludeD0Type==kFreeD0Only && !aFitInfo.freeD0) return false;
  if(aIncludeD0Type==kFixedD0Only && aFitInfo.freeD0) return false;

  //Exclude fixed radius results from all average/mean calculations
  if(!aFitInfo.freeRadii) return false;
  //Exclude fixed lambda results from all average/mean calculations
  if(!aFitInfo.freeLambda) return false;

  return true;
}





//---------------------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************************
//---------------------------------------------------------------------------------------------------------------------------------
struct DrawAcrossAnalysesInfo
{
  TString descriptor;
  TString saveDescriptor;
  IncludeResType incResType;
  IncludeD0Type incD0Type;
  int markerStyle;
  double markerSize;

  DrawAcrossAnalysesInfo(TString aDescriptor, TString aSaveDescriptor, IncludeResType aIncludeResType, IncludeD0Type aIncD0Type, int aMarkerStyle, double aMarkerSize)
  {
    descriptor = aDescriptor;
    saveDescriptor = aSaveDescriptor;
    incResType = aIncludeResType;
    incD0Type = aIncD0Type;
    markerStyle = aMarkerStyle;
    markerSize = aMarkerSize;
  }

};


//-------------------------------------------

  double tMarkerSizeSingle = 1.0;
  double tMarkerSizeAvg = 2.0;

  //-----

  int tMarkerStyle_QM = 20;

  int tMarkerStyle_10and3_Avg_FreeD0 = 21;
  int tMarkerStyle_10and3_Avg_FixedD0 = 25;
  int tMarkerStyle_10and3_Avg = 36;

  int tMarkerStyle_10_FreeD0 = 47;
  int tMarkerStyle_10_FixedD0 = 46;
  int tMarkerStyle_10_Avg = 48;

  int tMarkerStyle_3_FreeD0 = 34;
  int tMarkerStyle_3_FixedD0 = 28;
  int tMarkerStyle_3_Avg = 49;

//-------------------------------------------

  const DrawAcrossAnalysesInfo tDrawInfo_QM = DrawAcrossAnalysesInfo(TString("QM 2017"),
                                                                     TString("_QM2017"),
                                                                     kIncludeNoRes, kFreeD0Only,
                                                                     tMarkerStyle_QM, tMarkerSizeSingle);

  //-----

  const DrawAcrossAnalysesInfo tDrawInfo_10and3_Avg_FreeD0 = DrawAcrossAnalysesInfo(TString("10&3 Res., Avg., Free d_{0}"),
                                                                     TString("_10And3Res_Avg_FreeD0"),
                                                                     kInclude10ResAnd3Res, kFreeD0Only,
                                                                     tMarkerStyle_10and3_Avg_FreeD0, tMarkerSizeSingle);

  const DrawAcrossAnalysesInfo tDrawInfo_10and3_Avg_FixedD0 = DrawAcrossAnalysesInfo(TString("10&3 Res., Avg., Fix d_{0}"),
                                                                     TString("_10And3Res_Avg_FixedD0"),
                                                                     kInclude10ResAnd3Res, kFixedD0Only,
                                                                     tMarkerStyle_10and3_Avg_FixedD0, tMarkerSizeSingle);

  const DrawAcrossAnalysesInfo tDrawInfo_10and3_Avg = DrawAcrossAnalysesInfo(TString("10 & 3 Res., Avg."),
                                                                     TString("_10And3Res_Avg"),
                                                                     kInclude10ResAnd3Res, kFreeAndFixedD0,
                                                                     tMarkerStyle_10and3_Avg, tMarkerSizeAvg);

  //-----

  const DrawAcrossAnalysesInfo tDrawInfo_10_FreeD0 = DrawAcrossAnalysesInfo(TString("10 Res., Free d_{0}"),
                                                                     TString("_10Res_FreeD0"),
                                                                     kInclude10ResOnly, kFreeD0Only,
                                                                     tMarkerStyle_10_FreeD0, tMarkerSizeSingle);

  const DrawAcrossAnalysesInfo tDrawInfo_10_FixedD0 = DrawAcrossAnalysesInfo(TString("10 Res., Fix d_{0}"),
                                                                     TString("_10Res_FixedD0"),
                                                                     kInclude10ResOnly, kFixedD0Only,
                                                                     tMarkerStyle_10_FixedD0, tMarkerSizeSingle);

  const DrawAcrossAnalysesInfo tDrawInfo_10_Avg = DrawAcrossAnalysesInfo(TString("10 Res., Avg."),
                                                                     TString("_10Res_Avg"),
                                                                     kInclude10ResOnly, kFreeAndFixedD0,
                                                                     tMarkerStyle_10_Avg, tMarkerSizeAvg);

  //-----

  const DrawAcrossAnalysesInfo tDrawInfo_3_FreeD0 = DrawAcrossAnalysesInfo(TString("3 Res., Free d_{0}"),
                                                                     TString("_3Res_FreeD0"),
                                                                     kInclude3ResOnly, kFreeD0Only,
                                                                     tMarkerStyle_3_FreeD0, tMarkerSizeSingle);

  const DrawAcrossAnalysesInfo tDrawInfo_3_FixedD0 = DrawAcrossAnalysesInfo(TString("3 Res., Fix d_{0}"),
                                                                     TString("_3Res_FixedD0"),
                                                                     kInclude3ResOnly, kFixedD0Only,
                                                                     tMarkerStyle_3_FixedD0, tMarkerSizeSingle);

  const DrawAcrossAnalysesInfo tDrawInfo_3_Avg = DrawAcrossAnalysesInfo(TString("3 Res., Avg."),
                                                                     TString("_3Res_Avg"),
                                                                     kInclude3ResOnly, kFreeAndFixedD0,
                                                                     tMarkerStyle_3_Avg, tMarkerSizeAvg);


//------------------------------------------------------------------------------------------------

  const vector<DrawAcrossAnalysesInfo> tDrawAcrossAnalysesInfoVec{tDrawInfo_QM,
                                                                  tDrawInfo_10and3_Avg_FreeD0, tDrawInfo_10and3_Avg_FixedD0, tDrawInfo_10and3_Avg, 
                                                                  tDrawInfo_10_FreeD0,         tDrawInfo_10_FixedD0,         tDrawInfo_10_Avg,
                                                                  tDrawInfo_3_FreeD0,          tDrawInfo_3_FixedD0,          tDrawInfo_3_Avg};

  const vector<bool> tIncludePlotsv1{true,
                                     false, false, true, 
                                     true,  false, true, 
                                     true,  false, true};

  const vector<bool> tIncludePlotsv2{true,
                                     true, true, false, 
                                     true, true, false, 
                                     true, true, false};

