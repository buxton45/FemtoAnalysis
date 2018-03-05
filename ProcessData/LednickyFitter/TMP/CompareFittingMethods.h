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

enum ErrorType {kStat=0, kSys=1, kStatAndSys=2};
const char* const cErrorTypeTags[3] = {"Stat", "Sys", ""};

//---------------------------------------------------------------------------------------------------------------------------------
struct FitInfo
{
  TString descriptor;
  AnalysisType analysisType;

  IncludeLambdaType lambdaType;
  IncludeRadiiType radiiType;
  IncludeD0Type d0Type;
  IncludeResType resType;

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

  Color_t markerColor;
  int markerStyle;

  FitInfo(TString aDescriptor, 
          AnalysisType aAnalysisType,
          IncludeLambdaType aLamType, IncludeRadiiType aRadiiType, IncludeD0Type aD0Type, IncludeResType aResType, 
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

    lambdaType = aLamType;
    radiiType = aRadiiType;
    d0Type = aD0Type;
    resType = aResType;

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


  FitInfo(TString aDescriptor, 
          AnalysisType aAnalysisType, AnalysisType aConjAnType, IncludeResidualsType aIncResType, 
          IncludeLambdaType aLamType, IncludeRadiiType aRadiiType, IncludeD0Type aD0Type, IncludeResType aResType, 
          double aChi2, int aNDF,
          Color_t aMarkerColor, int aMarkerStyle)
  {
    descriptor = aDescriptor;
    analysisType = aAnalysisType;

    lambdaType = aLamType;
    radiiType = aRadiiType;
    d0Type = aD0Type;
    resType = aResType;

    //-------------------------------------------------------------------------------

    double lambda1a = cFitParamValues[aIncResType][aAnalysisType][k0010][kLambda][kValue];
    double lambda2a = cFitParamValues[aIncResType][aAnalysisType][k1030][kLambda][kValue];
    double lambda3a = cFitParamValues[aIncResType][aAnalysisType][k3050][kLambda][kValue];

    double lambda1b = cFitParamValues[aIncResType][aConjAnType][k0010][kLambda][kValue];
    double lambda2b = cFitParamValues[aIncResType][aConjAnType][k1030][kLambda][kValue];
    double lambda3b = cFitParamValues[aIncResType][aConjAnType][k3050][kLambda][kValue];

    lambda1 = 0.5*(lambda1a + lambda1b);
    lambda2 = 0.5*(lambda2a + lambda2b);
    lambda3 = 0.5*(lambda3a + lambda3b);


    radius1 = cFitParamValues[aIncResType][aAnalysisType][k0010][kRadius][kValue];
    radius2 = cFitParamValues[aIncResType][aAnalysisType][k1030][kRadius][kValue];
    radius3 = cFitParamValues[aIncResType][aAnalysisType][k3050][kRadius][kValue];

    ref0 = cFitParamValues[aIncResType][aAnalysisType][k0010][kRef0][kValue];
    imf0 = cFitParamValues[aIncResType][aAnalysisType][k0010][kImf0][kValue];
    d0 =   cFitParamValues[aIncResType][aAnalysisType][k0010][kd0][kValue];

    //--------------------------------------

    double lambdaStatErr1a = cFitParamValues[aIncResType][aAnalysisType][k0010][kLambda][kStatErr];
    double lambdaStatErr2a = cFitParamValues[aIncResType][aAnalysisType][k1030][kLambda][kStatErr];
    double lambdaStatErr3a = cFitParamValues[aIncResType][aAnalysisType][k3050][kLambda][kStatErr];

    double lambdaStatErr1b = cFitParamValues[aIncResType][aConjAnType][k0010][kLambda][kStatErr];
    double lambdaStatErr2b = cFitParamValues[aIncResType][aConjAnType][k1030][kLambda][kStatErr];
    double lambdaStatErr3b = cFitParamValues[aIncResType][aConjAnType][k3050][kLambda][kStatErr];

    lambdaStatErr1 = 0.5*(lambdaStatErr1a + lambdaStatErr1b);
    lambdaStatErr2 = 0.5*(lambdaStatErr2a + lambdaStatErr2b);
    lambdaStatErr3 = 0.5*(lambdaStatErr3a + lambdaStatErr3b);


    radiusStatErr1 = cFitParamValues[aIncResType][aAnalysisType][k0010][kRadius][kStatErr];
    radiusStatErr2 = cFitParamValues[aIncResType][aAnalysisType][k1030][kRadius][kStatErr];
    radiusStatErr3 = cFitParamValues[aIncResType][aAnalysisType][k3050][kRadius][kStatErr];

    ref0StatErr = cFitParamValues[aIncResType][aAnalysisType][k0010][kRef0][kStatErr];
    imf0StatErr = cFitParamValues[aIncResType][aAnalysisType][k0010][kImf0][kStatErr];
    d0StatErr =   cFitParamValues[aIncResType][aAnalysisType][k0010][kd0][kStatErr];

    //--------------------------------------

    double lambdaSysErr1a = cFitParamValues[aIncResType][aAnalysisType][k0010][kLambda][kSystErr];
    double lambdaSysErr2a = cFitParamValues[aIncResType][aAnalysisType][k1030][kLambda][kSystErr];
    double lambdaSysErr3a = cFitParamValues[aIncResType][aAnalysisType][k3050][kLambda][kSystErr];

    double lambdaSysErr1b = cFitParamValues[aIncResType][aConjAnType][k0010][kLambda][kSystErr];
    double lambdaSysErr2b = cFitParamValues[aIncResType][aConjAnType][k1030][kLambda][kSystErr];
    double lambdaSysErr3b = cFitParamValues[aIncResType][aConjAnType][k3050][kLambda][kSystErr];

    lambdaSysErr1 = 0.5*(lambdaSysErr1a + lambdaSysErr1b);
    lambdaSysErr2 = 0.5*(lambdaSysErr2a + lambdaSysErr2b);
    lambdaSysErr3 = 0.5*(lambdaSysErr3a + lambdaSysErr3b);


    radiusSysErr1 = cFitParamValues[aIncResType][aAnalysisType][k0010][kRadius][kSystErr];
    radiusSysErr2 = cFitParamValues[aIncResType][aAnalysisType][k1030][kRadius][kSystErr];
    radiusSysErr3 = cFitParamValues[aIncResType][aAnalysisType][k3050][kRadius][kSystErr];

    ref0SysErr = cFitParamValues[aIncResType][aAnalysisType][k0010][kRef0][kSystErr];
    imf0SysErr = cFitParamValues[aIncResType][aAnalysisType][k0010][kImf0][kSystErr];
    d0SysErr =   cFitParamValues[aIncResType][aAnalysisType][k0010][kd0][kSystErr];

    //-------------------------------------------------------------------------------

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

  Color_t tColorQM_LamKchP = kRed+1;
  Color_t tColorQM_LamKchM = kBlue+1;
  Color_t tColorQM_LamK0 = kBlack;

  //---------------------------------------------------------------------------
  //---------------------------- LamKchP --------------------------------------
  //---------------------------------------------------------------------------

  //--------------- 10 Residuals ----------
  const FitInfo tFitInfo1a_LamKchP = FitInfo(TString("FreeRadii_FreeD0_10Res"), 
                                       kLamKchP, kALamKchM, kInclude10Residuals, 
                                       kFreeLambdaOnly, kFreeRadiiOnly, kFreeD0Only, kInclude10ResOnly, 
                                       411.3, 336,
                                       tColor1, tMarkerStyleA1);

  const FitInfo tFitInfo2a_LamKchP = FitInfo(TString("FreeRadii_FixedD0_10Res"), 
                                       kLamKchP,
                                       kFreeLambdaOnly, kFreeRadiiOnly, kFixedD0Only, kInclude10ResOnly, 

                                       //ResPrimMaxDecayType = k4fm (20171227)
                                       0.5*(1.93+1.96), 0.5*(0.54+0.56), 0.5*(0.33+0.37),
                                       0.5*(1.82+1.57), 0.5*(0.62+0.53), 0.5*(0.35+0.25),
                                       0.5*(1.24+1.20), 0.5*(0.31+0.29), 0.5*(0.38+0.17),

                                       6.07, 0.97, 0.33,
                                       5.01, 0.86, 0.31,
                                       3.21, 0.41, 0.22,

                                      -1.05, 0.13, 0.20,
                                       0.53, 0.26, 0.12,
                                       0.00, 0.00, 0.00,

                                       413.0,
                                       337,

                                       tColor1, tMarkerStyleA2);


  const FitInfo tFitInfo3a_LamKchP = FitInfo(TString("FixedLambda_FreeD0_10Res"), 
                                       kLamKchP,
                                       kFixedLambdaOnly, kFreeRadiiOnly, kFreeD0Only, kInclude10ResOnly, 
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
                                       kFixedLambdaOnly, kFreeRadiiOnly, kFixedD0Only, kInclude10ResOnly, 
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
                                       kFreeLambdaOnly, kFixedRadiiOnly, kFreeD0Only, kInclude10ResOnly, 

                                       //ResPrimMaxDecayType = k4fm (20171227)
                                       0.5*(0.97+0.98), 0.5*(0.30+0.31), 0.5*(0.00+0.00),
                                       0.5*(1.14+1.00), 0.5*(0.36+0.31), 0.5*(0.00+0.00),
                                       0.5*(1.02+1.00), 0.5*(0.34+0.34), 0.5*(0.00+0.00),

                                       3.50, 0.00, 0.00,
                                       3.25, 0.00, 0.00,
                                       2.50, 0.00, 0.00,

                                      -0.92, 0.32, 0.00,
                                       0.05, 0.04, 0.00,
                                       1.05, 0.29, 0.00,

                                       425.9,
                                       339,

                                       kPink+10, 47);

  const FitInfo tFitInfo6a_LamKchP = FitInfo(TString("FixedRadii_FixedD0_10Res"), 
                                       kLamKchP,
                                       kFreeLambdaOnly, kFixedRadiiOnly, kFixedD0Only, kInclude10ResOnly, 

                                       //ResPrimMaxDecayType = k4fm (20171227)
                                       0.5*(1.19+1.21), 0.5*(0.26+0.27), 0.5*(0.00+0.00),
                                       0.5*(1.41+1.23), 0.5*(0.30+0.26), 0.5*(0.00+0.00),
                                       0.5*(1.33+1.28), 0.5*(0.30+0.29), 0.5*(0.00+0.00),

                                       3.50, 0.00, 0.00,
                                       3.25, 0.00, 0.00,
                                       2.50, 0.00, 0.00,

                                      -0.69, 0.15, 0.00,
                                       0.10, 0.04, 0.00,
                                       0.00, 0.00, 0.00,

                                       430.6,
                                       340,

                                       kPink+10, 46);


  //--------------- 3 Residuals ----------
  const FitInfo tFitInfo1b_LamKchP = FitInfo(TString("FreeRadii_FreeD0_3Res"), 
                                       kLamKchP, kALamKchM, kInclude3Residuals, 
                                       kFreeLambdaOnly, kFreeRadiiOnly, kFreeD0Only, kInclude3ResOnly, 
                                       415.1, 336,
                                       tColor1, tMarkerStyleB1);

  const FitInfo tFitInfo2b_LamKchP = FitInfo(TString("FreeRadii_FixedD0_3Res"), 
                                       kLamKchP,
                                       kFreeLambdaOnly, kFreeRadiiOnly, kFixedD0Only, kInclude3ResOnly, 

                                       //ResPrimMaxDecayType = k4fm (20171227)
                                       0.5*(2.06+2.08), 0.5*(0.56+0.57), 0.5*(0.35+0.38),
                                       0.5*(2.08+1.78), 0.5*(0.67+0.56), 0.5*(0.38+0.26),
                                       0.5*(1.41+1.35), 0.5*(0.32+0.31), 0.5*(0.42+0.19),

                                       5.71, 0.92, 0.28,
                                       4.87, 0.79, 0.28,
                                       3.08, 0.36, 0.20,

                                      -0.85, 0.11, 0.17,
                                       0.47, 0.21, 0.10,
                                       0.00, 0.00, 0.00,

                                       418.8,
                                       337,

                                       tColor1, tMarkerStyleB2);


  const FitInfo tFitInfo3b_LamKchP = FitInfo(TString("FixedLambda_FreeD0_3Res"), 
                                       kLamKchP,
                                       kFixedLambdaOnly, kFreeRadiiOnly, kFreeD0Only, kInclude3ResOnly, 
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
                                       kFixedLambdaOnly, kFreeRadiiOnly, kFixedD0Only, kInclude3ResOnly, 
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
                                       kFreeLambdaOnly, kFixedRadiiOnly, kFreeD0Only, kInclude3ResOnly, 

                                       //ResPrimMaxDecayType = k4fm (20171227)
                                       0.5*(1.04+1.05), 0.5*(0.24+0.25), 0.5*(0.00+0.00),
                                       0.5*(1.23+1.07), 0.5*(0.29+0.25), 0.5*(0.00+0.00),
                                       0.5*(1.12+1.09), 0.5*(0.29+0.28), 0.5*(0.00+0.00),

                                       3.50, 0.00, 0.00,
                                       3.25, 0.00, 0.00,
                                       2.50, 0.00, 0.00,

                                      -0.87, 0.23, 0.00,
                                       0.09, 0.04, 0.00,
                                       1.21, 0.21, 0.00,

                                       425.4,
                                       339,

                                       kPink+10, 34);

  const FitInfo tFitInfo6b_LamKchP = FitInfo(TString("FixedRadii_FixedD0_3Res"), 
                                       kLamKchP,
                                       kFreeLambdaOnly, kFixedRadiiOnly, kFixedD0Only, kInclude3ResOnly, 

                                       //ResPrimMaxDecayType = k4fm (20171227)
                                       0.5*(1.26+1.27), 0.5*(0.27+0.28), 0.5*(0.00+0.00),
                                       0.5*(1.51+1.30), 0.5*(0.32+0.28), 0.5*(0.00+0.00),
                                       0.5*(1.46+1.40), 0.5*(0.33+0.32), 0.5*(0.00+0.00),

                                       3.50, 0.00, 0.00,
                                       3.25, 0.00, 0.00,
                                       2.50, 0.00, 0.00,

                                      -0.64, 0.14, 0.00,
                                       0.14, 0.05, 0.00,
                                       0.00, 0.00, 0.00,

                                       434.3,
                                       340,

                                       kPink+10, 28);


  //--------------- No Residuals ----------
  //CustomLambdaLimits
  const FitInfo tFitInfo1c_LamKchP = FitInfo(TString("FreeRadii_FreeD0_NoRes"), 
                                       kLamKchP,
                                       kFreeLambdaOnly, kFreeRadiiOnly, kFreeD0Only, kIncludeNoRes, 
                                       0.5*(0.47+0.46), 0.5*(0.17+0.17), 0.5*(0.22+0.22),
                                       0.5*(0.50+0.44), 0.5*(0.17+0.03), 0.5*(0.24+0.20),
                                       0.5*(0.59+0.58), 0.5*(0.14+0.13), 0.5*(0.20+0.20),

                                       4.990, 0.830, 0.830,
                                       4.350, 0.600, 0.663,
                                       3.790, 0.700, 0.420,

                                      -0.65, 0.13, 0.22,
                                       0.56, 0.28, 0.11,
                                       0.86, 0.54, 1.62,

                                       415.7,
                                       336,

                                       tColorQM_LamKchP, tMarkerStyleA1);
 



  const FitInfo tFitInfo2c_LamKchP = FitInfo(TString("FreeRadii_FixedD0_NoRes"), 
                                       kLamKchP,
                                       kFreeLambdaOnly, kFreeRadiiOnly, kFixedD0Only, kIncludeNoRes, 
                                       0.5*(0.75+0.74), 0.5*(0.31+0.30), 0.5*(0.22+0.22),
                                       0.5*(0.79+0.69), 0.5*(0.29+0.25), 0.5*(0.24+0.20),
                                       0.5*(0.87+0.86), 0.5*(0.37+0.36), 0.5*(0.20+0.20),

                                       5.550, 1.190, 0.830,
                                       4.790, 0.880, 0.663,
                                       4.060, 0.860, 0.420,

                                      -0.42, 0.12, 0.22,
                                       0.45, 0.22, 0.11,
                                       0.00, 0.00, 1.62,

                                       415.0,
                                       336,

                                       tColorQM_LamKchP, tMarkerStyleA1);



  //---------------------------------------------------------------------------
  //---------------------------- LamKchM --------------------------------------
  //---------------------------------------------------------------------------

  //--------------- 10 Residuals ----------
  const FitInfo tFitInfo1a_LamKchM = FitInfo(TString("FreeRadii_FreeD0_10Res"), 
                                       kLamKchM, kALamKchP, kInclude10Residuals, 
                                       kFreeLambdaOnly, kFreeRadiiOnly, kFreeD0Only, kInclude10ResOnly, 
                                       282.2, 288,
                                       tColor1, tMarkerStyleA1);

  const FitInfo tFitInfo2a_LamKchM = FitInfo(TString("FreeRadii_FixedD0_10Res"), 
                                       kLamKchM,
                                       kFreeLambdaOnly, kFreeRadiiOnly, kFixedD0Only, kInclude10ResOnly, 

                                       //ResPrimMaxDecayType = k4fm (20171227)
                                       0.5*(1.35+1.36), 0.5*(0.66+0.64), 0.5*(0.74+0.65),
                                       0.5*(1.08+1.14), 0.5*(0.52+0.55), 0.5*(0.57+0.69),
                                       0.5*(1.12+0.75), 0.5*(0.83+0.38), 0.5*(0.88+0.66),

                                       5.01, 0.73, 0.76,
                                       3.97, 0.64, 0.64,
                                       2.56, 0.49, 0.42,

                                       0.40, 0.17, 0.27,
                                       0.57, 0.27, 0.41,
                                       0.00, 0.00, 0.00,

                                       287.1,
                                       289,

                                       tColor1, tMarkerStyleA2);


  const FitInfo tFitInfo3a_LamKchM = FitInfo(TString("FixedLambda_FreeD0_10Res"), 
                                       kLamKchM,
                                       kFixedLambdaOnly, kFreeRadiiOnly, kFreeD0Only, kInclude10ResOnly, 
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
                                       kFixedLambdaOnly, kFreeRadiiOnly, kFixedD0Only, kInclude10ResOnly, 
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
                                       kFreeLambdaOnly, kFixedRadiiOnly, kFreeD0Only, kInclude10ResOnly, 

                                       //ResPrimMaxDecayType = k4fm (20171227)
                                       0.5*(0.54+0.56), 0.5*(0.11+0.11), 0.5*(0.00+0.00),
                                       0.5*(0.55+0.58), 0.5*(0.11+0.12), 0.5*(0.00+0.00),
                                       0.5*(0.56+0.47), 0.5*(0.16+0.12), 0.5*(0.00+0.00),

                                       3.50, 0.00, 0.00,
                                       3.25, 0.00, 0.00,
                                       2.50, 0.00, 0.00,

                                       0.22, 0.16, 0.00,
                                       0.77, 0.35, 0.00,
                                      -2.64, 2.08, 0.00,

                                       295.1,
                                       291,

                                       kAzure+10, 47);

  const FitInfo tFitInfo6a_LamKchM = FitInfo(TString("FixedRadii_FixedD0_10Res"), 
                                       kLamKchM,
                                       kFreeLambdaOnly, kFixedRadiiOnly, kFixedD0Only, kInclude10ResOnly, 

                                       //ResPrimMaxDecayType = k4fm (20171227)
                                       0.5*(0.57+0.59), 0.5*(0.13+0.13), 0.5*(0.00+0.00),
                                       0.5*(0.58+0.61), 0.5*(0.13+0.14), 0.5*(0.00+0.00),
                                       0.5*(0.60+0.50), 0.5*(0.18+0.13), 0.5*(0.00+0.00),

                                       3.50, 0.00, 0.00,
                                       3.25, 0.00, 0.00,
                                       2.50, 0.00, 0.00,

                                       0.25, 0.15, 0.00,
                                       0.79, 0.33, 0.00,
                                       0.00, 0.00, 0.00,

                                       296.2,
                                       292,

                                       kAzure+10, 46);



  //--------------- 3 Residuals ----------
  const FitInfo tFitInfo1b_LamKchM = FitInfo(TString("FreeRadii_FreeD0_3Res"), 
                                       kLamKchM, kALamKchP, kInclude3Residuals, 
                                       kFreeLambdaOnly, kFreeRadiiOnly, kFreeD0Only, kInclude3ResOnly, 
                                       281.2, 288,
                                       tColor1, tMarkerStyleB1);

  const FitInfo tFitInfo2b_LamKchM = FitInfo(TString("FreeRadii_FixedD0_3Res"), 
                                       kLamKchM,
                                       kFreeLambdaOnly, kFreeRadiiOnly, kFixedD0Only, kInclude3ResOnly, 

                                       //ResPrimMaxDecayType = k4fm (20171227)
                                       0.5*(1.46+1.46), 0.5*(0.53+0.51), 0.5*(0.67+0.55),
                                       0.5*(1.18+1.25), 0.5*(0.44+0.47), 0.5*(0.48+0.66),
                                       0.5*(1.31+0.83), 0.5*(0.84+0.35), 0.5*(0.82+0.68),

                                       4.51, 0.57, 0.75,
                                       3.61, 0.54, 0.67,
                                       2.31, 0.41, 0.41,

                                       0.26, 0.13, 0.21,
                                       0.44, 0.18, 0.30,
                                       0.00, 0.00, 0.00,

                                       288.8,
                                       289,

                                       tColor1, tMarkerStyleB2);


  const FitInfo tFitInfo3b_LamKchM = FitInfo(TString("FixedLambda_FreeD0_3Res"), 
                                       kLamKchM,
                                       kFixedLambdaOnly, kFreeRadiiOnly, kFreeD0Only, kInclude3ResOnly, 
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
                                       kFixedLambdaOnly, kFreeRadiiOnly, kFixedD0Only, kInclude3ResOnly, 
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
                                       kFreeLambdaOnly, kFixedRadiiOnly, kFreeD0Only, kInclude3ResOnly, 

                                       //ResPrimMaxDecayType = k4fm (20171227)
                                       0.5*(0.79+0.81), 0.5*(0.33+0.34), 0.5*(0.00+0.00),
                                       0.5*(0.78+0.83), 0.5*(0.32+0.34), 0.5*(0.00+0.00),
                                       0.5*(0.87+0.66), 0.5*(0.47+0.27), 0.5*(0.00+0.00),

                                       3.50, 0.00, 0.00,
                                       3.25, 0.00, 0.00,
                                       2.50, 0.00, 0.00,

                                       0.12, 0.12, 0.00,
                                       0.47, 0.33, 0.00,
                                      -4.26, 4.08, 0.00,

                                       294.6,
                                       291,

                                       kAzure+10, 34);

  const FitInfo tFitInfo6b_LamKchM = FitInfo(TString("FixedRadii_FixedD0_3Res"), 
                                       kLamKchM,
                                       kFreeLambdaOnly, kFixedRadiiOnly, kFixedD0Only, kInclude3ResOnly, 

                                       //ResPrimMaxDecayType = k4fm (20171227)
                                       0.5*(0.90+0.92), 0.5*(0.32+0.32), 0.5*(0.00+0.00),
                                       0.5*(0.89+0.95), 0.5*(0.31+0.33), 0.5*(0.00+0.00),
                                       0.5*(1.07+0.75), 0.5*(0.56+0.28), 0.5*(0.00+0.00),

                                       3.50, 0.00, 0.00,
                                       3.25, 0.00, 0.00,
                                       2.50, 0.00, 0.00,

                                       0.15, 0.09, 0.00,
                                       0.46, 0.23, 0.00,
                                       0.00, 0.00, 0.00,

                                       296.9,
                                       292,

                                       kAzure+10, 28);

  //--------------- No Residuals ----------
  const FitInfo tFitInfo1c_LamKchM = FitInfo(TString("FreeRadii_FreeD0_NoRes"), 
                                       kLamKchM,
                                       kFreeLambdaOnly, kFreeRadiiOnly, kFreeD0Only, kIncludeNoRes, 
                                       0.5*(0.46+0.49), 0.5*(0.16+0.16), 0.5*(0.19+0.15),
                                       0.5*(0.37+0.40), 0.5*(0.12+0.13), 0.5*(0.20+0.15),
                                       0.5*(0.21+0.22), 0.5*(0.08+0.08), 0.5*(0.13+0.11),

                                       4.990, 0.750, 1.375,
                                       3.950, 0.630, 0.978,
                                       2.340, 0.530, 0.457,

                                       0.22, 0.12, 0.10,
                                       0.50, 0.16, 0.18,
                                      -4.51, 2.05, 7.66,

                                       279.1,
                                       288,

                                       tColorQM_LamKchM, tMarkerStyleA1);

  const FitInfo tFitInfo2c_LamKchM = FitInfo(TString("FreeRadii_FixedD0_NoRes"), 
                                       kLamKchM,
                                       kFreeLambdaOnly, kFreeRadiiOnly, kFixedD0Only, kIncludeNoRes, 
                                       0.5*(0.82+0.87), 0.5*(0.48+0.50), 0.5*(0.19+0.15),
                                       0.5*(0.67+0.71), 0.5*(0.40+0.43), 0.5*(0.20+0.15),
                                       0.5*(0.38+0.40), 0.5*(0.24+0.25), 0.5*(0.13+0.11),

                                       4.800, 0.660, 1.375,
                                       3.840, 0.590, 0.978,
                                       2.290, 0.490, 0.457,

                                       0.12, 0.08, 0.10,
                                       0.28, 0.15, 0.18,
                                       0.00, 0.00, 7.66,

                                       282.2,
                                       288,

                                       tColorQM_LamKchM, tMarkerStyleA1);


  //---------------------------------------------------------------------------
  //---------------------------- LamK0 --------------------------------------
  //---------------------------------------------------------------------------

  //--------------- 10 Residuals ----------
  const FitInfo tFitInfo1a_LamK0 = FitInfo(TString("FreeRadii_FreeD0_10Res"), 
                                       kLamK0, kALamK0, kInclude10Residuals, 
                                       kFreeLambdaOnly, kFreeRadiiOnly, kFreeD0Only, kInclude10ResOnly, 
                                       362.5, 341,
                                       tColor1, tMarkerStyleA1);

  const FitInfo tFitInfo2a_LamK0 = FitInfo(TString("FreeRadii_FixedD0_10Res"), 
                                       kLamK0,
                                       kFreeLambdaOnly, kFreeRadiiOnly, kFixedD0Only, kInclude10ResOnly, 

                                       //ResPrimMaxDecayType = k4fm (20171227)
                                       0.5*(1.50+1.50), 0.5*(0.89+0.89), 0.5*(0.36+0.36),
                                       0.5*(1.50+1.50), 0.5*(0.89+0.89), 0.5*(0.36+0.36),
                                       0.5*(1.50+1.50), 0.5*(0.89+0.89), 0.5*(0.36+0.36),

                                       3.48, 0.88, 0.42,
                                       2.81, 0.71, 0.25,
                                       2.09, 0.50, 0.24,

                                      -0.15, 0.04, 0.03,
                                       0.13, 0.10, 0.03,
                                       0.00, 0.00, 0.00,

                                       365.0,
                                       342,

                                       tColor1, tMarkerStyleA2);


  const FitInfo tFitInfo3a_LamK0 = FitInfo(TString("FixedLambda_FreeD0_10Res"), 
                                       kLamK0,
                                       kFixedLambdaOnly, kFreeRadiiOnly, kFreeD0Only, kInclude10ResOnly, 
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
                                       kFixedLambdaOnly, kFreeRadiiOnly, kFixedD0Only, kInclude10ResOnly, 
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
                                       kFreeLambdaOnly, kFixedRadiiOnly, kFreeD0Only, kInclude10ResOnly, 

                                       //ResPrimMaxDecayType = k4fm (20171227)
                                       0.5*(0.60+0.60), 0.5*(0.16+0.16), 0.5*(0.00+0.00),
                                       0.5*(0.60+0.60), 0.5*(0.16+0.16), 0.5*(0.00+0.00),
                                       0.5*(0.60+0.60), 0.5*(0.16+0.16), 0.5*(0.00+0.00),

                                       3.25, 0.00, 0.00,
                                       2.75, 0.00, 0.00,
                                       2.25, 0.00, 0.00,

                                      -0.56, 0.12, 0.00,
                                       0.18, 0.09, 0.00,
                                       1.84, 0.29, 0.00,

                                       366.5,
                                       344,

                                       kGray+1, 47);

  const FitInfo tFitInfo6a_LamK0 = FitInfo(TString("FixedRadii_FixedD0_10Res"), 
                                       kLamK0,
                                       kFreeLambdaOnly, kFixedRadiiOnly, kFixedD0Only, kInclude10ResOnly, 

                                       //ResPrimMaxDecayType = k4fm (20171227)
                                       0.5*(1.50+1.50), 0.5*(0.86+0.86), 0.5*(0.00+0.00),
                                       0.5*(1.50+1.50), 0.5*(0.86+0.86), 0.5*(0.00+0.00),
                                       0.5*(1.50+1.50), 0.5*(0.86+0.86), 0.5*(0.00+0.00),

                                       3.25, 0.00, 0.00,
                                       2.75, 0.00, 0.00,
                                       2.25, 0.00, 0.00,

                                      -0.16, 0.03, 0.00,
                                       0.11, 0.02, 0.00,
                                       0.00, 0.00, 0.00,

                                       370.0,
                                       345,

                                       kGray+1, 46);

  //--------------- 3 Residuals ----------
  const FitInfo tFitInfo1b_LamK0 = FitInfo(TString("FreeRadii_FreeD0_3Res"), 
                                       kLamK0, kALamK0, kInclude3Residuals, 
                                       kFreeLambdaOnly, kFreeRadiiOnly, kFreeD0Only, kInclude3ResOnly, 
                                       361.8, 341,
                                       tColor1, tMarkerStyleB1);

  const FitInfo tFitInfo2b_LamK0 = FitInfo(TString("FreeRadii_FixedD0_3Res"), 
                                       kLamK0,
                                       kFreeLambdaOnly, kFreeRadiiOnly, kFixedD0Only, kInclude3ResOnly, 

                                       //ResPrimMaxDecayType = k4fm (20171227)
                                       0.5*(1.50+1.50), 0.5*(0.87+0.87), 0.5*(0.49+0.49),
                                       0.5*(1.50+1.50), 0.5*(0.87+0.87), 0.5*(0.49+0.49),
                                       0.5*(1.50+1.50), 0.5*(0.87+0.87), 0.5*(0.49+0.49),

                                       3.47, 0.94, 0.50,
                                       2.75, 0.73, 0.33,
                                       2.02, 0.51, 0.28,

                                      -0.14, 0.04, 0.06,
                                       0.19, 0.15, 0.08,
                                       0.00, 0.00, 0.00,

                                       365.1,
                                       342,

                                       tColor1, tMarkerStyleB2);


  const FitInfo tFitInfo3b_LamK0 = FitInfo(TString("FixedLambda_FreeD0_3Res"), 
                                       kLamK0,
                                       kFixedLambdaOnly, kFreeRadiiOnly, kFreeD0Only, kInclude3ResOnly, 
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
                                       kFixedLambdaOnly, kFreeRadiiOnly, kFixedD0Only, kInclude3ResOnly, 
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
                                       kFreeLambdaOnly, kFixedRadiiOnly, kFreeD0Only, kInclude3ResOnly, 

                                       //ResPrimMaxDecayType = k4fm (20171227)
                                       0.5*(0.60+0.60), 0.5*(0.61+0.61), 0.5*(0.00+0.00),
                                       0.5*(0.60+0.60), 0.5*(0.61+0.61), 0.5*(0.00+0.00),
                                       0.5*(0.60+0.60), 0.5*(0.61+0.61), 0.5*(0.00+0.00),

                                       3.25, 0.00, 0.00,
                                       2.75, 0.00, 0.00,
                                       2.25, 0.00, 0.00,

                                      -0.58, 0.13, 0.00,
                                       0.28, 0.10, 0.00,
                                       2.16, 0.35, 0.00,

                                       367.6,
                                       344,

                                       kGray+1, 34);

  const FitInfo tFitInfo6b_LamK0 = FitInfo(TString("FixedRadii_FixedD0_3Res"), 
                                       kLamK0,
                                       kFreeLambdaOnly, kFixedRadiiOnly, kFixedD0Only, kInclude3ResOnly, 

                                       //ResPrimMaxDecayType = k4fm (20171227)
                                       0.5*(1.50+1.50), 0.5*(0.75+0.75), 0.5*(0.00+0.00),
                                       0.5*(1.50+1.50), 0.5*(0.75+0.75), 0.5*(0.00+0.00),
                                       0.5*(1.50+1.50), 0.5*(0.75+0.75), 0.5*(0.00+0.00),

                                       3.25, 0.00, 0.00,
                                       2.75, 0.00, 0.00,
                                       2.25, 0.00, 0.00,

                                      -0.15, 0.03, 0.00,
                                       0.17, 0.02, 0.00,
                                       0.00, 0.00, 0.00,

                                       371.7,
                                       345,

                                       kGray+1, 28);

  //--------------- No Residuals ----------
  const FitInfo tFitInfo1c_LamK0 = FitInfo(TString("FreeRadii_FreeD0_NoRes"), 
                                       kLamK0,
                                       kFreeLambdaOnly, kFreeRadiiOnly, kFreeD0Only, kIncludeNoRes, 
                                       0.5*(0.40+0.40), 0.5*(0.20+0.20), 0.5*(0.12+0.12),
                                       0.5*(0.40+0.40), 0.5*(0.20+0.20), 0.5*(0.12+0.12),
                                       0.5*(0.40+0.40), 0.5*(0.20+0.20), 0.5*(0.12+0.12),

                                       3.080, 0.810, 0.329,
                                       2.320, 0.410, 0.324,
                                       1.750, 0.300, 0.280,

                                      -0.20, 0.04, 0.04,
                                       0.19, 0.08, 0.06,
                                       3.08, 0.81, 2.84,

                                       361.9,
                                       341,

                                       tColorQM_LamK0, tMarkerStyleA1);

  const FitInfo tFitInfo2c_LamK0 = FitInfo(TString("FreeRadii_FixedD0_NoRes"), 
                                       kLamK0,
                                       kFreeLambdaOnly, kFreeRadiiOnly, kFixedD0Only, kIncludeNoRes, 
                                       0.5*(0.60+0.60), 0.5*(0.19+0.19), 0.5*(0.12+0.12),
                                       0.5*(0.60+0.60), 0.5*(0.19+0.19), 0.5*(0.12+0.12),
                                       0.5*(0.60+0.60), 0.5*(0.19+0.19), 0.5*(0.12+0.12),

                                       3.150, 0.690, 0.329,
                                       2.460, 0.530, 0.324,
                                       1.820, 0.380, 0.280,

                                      -0.11, 0.02, 0.04,
                                       0.16, 0.10, 0.06,
                                       0.00, 0.00, 2.84,

                                       364.8,
                                       341,

                                       tColorQM_LamK0, tMarkerStyleA1);



//------------------------------------------------------------------------------------------------
  const vector<FitInfo> tFitInfoVec{tFitInfo1a_LamKchP, tFitInfo2a_LamKchP, tFitInfo3a_LamKchP, tFitInfo4a_LamKchP, tFitInfo5a_LamKchP, tFitInfo6a_LamKchP,
                                    tFitInfo1b_LamKchP, tFitInfo2b_LamKchP, tFitInfo3b_LamKchP, tFitInfo4b_LamKchP, tFitInfo5b_LamKchP, tFitInfo6b_LamKchP,
                                    tFitInfo1c_LamKchP, tFitInfo2c_LamKchP,

                                    tFitInfo1a_LamKchM, tFitInfo2a_LamKchM, tFitInfo3a_LamKchM, tFitInfo4a_LamKchM, tFitInfo5a_LamKchM, tFitInfo6a_LamKchM,
                                    tFitInfo1b_LamKchM, tFitInfo2b_LamKchM, tFitInfo3b_LamKchM, tFitInfo4b_LamKchM, tFitInfo5b_LamKchM, tFitInfo6b_LamKchM, 
                                    tFitInfo1c_LamKchM, tFitInfo2c_LamKchM, 

                                    tFitInfo1a_LamK0, tFitInfo2a_LamK0, tFitInfo3a_LamK0, tFitInfo4a_LamK0, tFitInfo5a_LamK0, tFitInfo6a_LamK0,
                                    tFitInfo1b_LamK0, tFitInfo2b_LamK0, tFitInfo3b_LamK0, tFitInfo4b_LamK0, tFitInfo5b_LamK0, tFitInfo6b_LamK0, 
                                    tFitInfo1c_LamK0, tFitInfo2c_LamK0};


//---------------------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************************
//---------------------------------------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------------------------------------
vector<FitInfo> GetFitInfoVec(AnalysisType aAnType, IncludeResType aIncludeResType=kInclude10ResAnd3Res, IncludeD0Type aIncludeD0Type=kFreeAndFixedD0,
                                                    IncludeRadiiType aIncludeRadiiType=kFreeAndFixedRadii, IncludeLambdaType aIncludeLambdaType=kFreeAndFixedLambda)
{
  //------------------------------
  vector<FitInfo> tReturnVec;
  //------------------------------

  bool bPassAnType=false, bPassRes=false, bPassD0=false, bPassRadii=false, bPassLambda=false;
  for(unsigned int i=0; i<tFitInfoVec.size(); i++)
  {
    if(aAnType == tFitInfoVec[i].analysisType) bPassAnType = true;
    else bPassAnType = false;

    if(aIncludeResType==kInclude10ResAnd3Res) bPassRes = true;
    else if(aIncludeResType == tFitInfoVec[i].resType) bPassRes = true;
    else bPassRes = false;

    if(aIncludeD0Type==kFreeAndFixedD0) bPassD0 = true;
    else if(aIncludeD0Type == tFitInfoVec[i].d0Type) bPassD0 = true;
    else bPassD0 = false;

    if(aIncludeRadiiType==kFreeAndFixedRadii) bPassRadii = true;
    else if(aIncludeRadiiType == tFitInfoVec[i].radiiType) bPassRadii = true;
    else bPassRadii = false;

    if(aIncludeLambdaType==kFreeAndFixedLambda) bPassLambda = true;
    else if(aIncludeLambdaType == tFitInfoVec[i].lambdaType) bPassLambda = true;
    else bPassLambda = false;

    if(bPassAnType && bPassRes && bPassD0 && bPassRadii && bPassLambda) tReturnVec.push_back(tFitInfoVec[i]);
  }

  return tReturnVec;
}

//---------------------------------------------------------------------------------------------------------------------------------
vector<FitInfo> GetFitInfoVec(vector<AnalysisType> aAnTypes, IncludeResType aIncludeResType=kInclude10ResAnd3Res, IncludeD0Type aIncludeD0Type=kFreeAndFixedD0,
                              IncludeRadiiType aIncludeRadiiType=kFreeAndFixedRadii, IncludeLambdaType aIncludeLambdaType=kFreeAndFixedLambda)
{
  //------------------------------
  vector<FitInfo> tReturnVec(0);
  vector<FitInfo> tTempVec(0);
  //------------------------------

  for(unsigned int i=0; i<aAnTypes.size(); i++)
  {
    tTempVec.clear();
    tTempVec = GetFitInfoVec(aAnTypes[i], aIncludeResType, aIncludeD0Type, aIncludeRadiiType, aIncludeLambdaType);
    tReturnVec.insert(tReturnVec.end(), tTempVec.begin(), tTempVec.end());
  }

  return tReturnVec;
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

  double tMarkerSizeSingle = 1.5;
  double tMarkerSizeAvg = 2.0;

  //-----

  int tMarkerStyle_QM = 20;
  int tMarkerStyle_QM_FixedD0 = 24;

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
/*
  const DrawAcrossAnalysesInfo tDrawInfo_QM = DrawAcrossAnalysesInfo(TString("QM 2017"),
                                                                     TString("_QM2017"),
                                                                     kIncludeNoRes, kFreeD0Only,
                                                                     tMarkerStyle_QM, tMarkerSizeSingle);
*/

  const DrawAcrossAnalysesInfo tDrawInfo_QM = DrawAcrossAnalysesInfo(TString("No Res."),
                                                                     TString("_NoRes_FreeD0"),
                                                                     kIncludeNoRes, kFreeD0Only,
                                                                     tMarkerStyle_QM, tMarkerSizeSingle);

  const DrawAcrossAnalysesInfo tDrawInfo_QM_FixD0 = DrawAcrossAnalysesInfo(TString("No Res., Fix d_{0}"),
                                                                     TString("_NoRes_FixD0"),
                                                                     kIncludeNoRes, kFixedD0Only,
                                                                     tMarkerStyle_QM_FixedD0, tMarkerSizeSingle);

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

  const DrawAcrossAnalysesInfo tDrawInfo_10_FreeD0 = DrawAcrossAnalysesInfo(TString("10 Res."),
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

  const DrawAcrossAnalysesInfo tDrawInfo_3_FreeD0 = DrawAcrossAnalysesInfo(TString("3 Res."),
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

  const vector<DrawAcrossAnalysesInfo> tDrawAcrossAnalysesInfoVec{tDrawInfo_QM, tDrawInfo_QM_FixD0,
                                                                  tDrawInfo_10and3_Avg_FreeD0, tDrawInfo_10and3_Avg_FixedD0, tDrawInfo_10and3_Avg, 
                                                                  tDrawInfo_10_FreeD0,         tDrawInfo_10_FixedD0,         tDrawInfo_10_Avg,
                                                                  tDrawInfo_3_FreeD0,          tDrawInfo_3_FixedD0,          tDrawInfo_3_Avg};

  const vector<bool> tIncludePlotsv1{true, false, 
                                     false, false, true, 
                                     true,  false, true, 
                                     true,  false, true};

  const vector<bool> tIncludePlotsv2{true, false,
                                     true, true, false, 
                                     true, true, false, 
                                     true, true, false};

