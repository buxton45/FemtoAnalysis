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

  double lambda1, lambda2, lambda3;
  double radius1, radius2, radius3;
  double ref0, imf0, d0;


  double lambdaStatErr1, lambdaStatErr2, lambdaStatErr3;
  double radiusStatErr1, radiusStatErr2, radiusStatErr3; 
  double ref0StatErr, imf0StatErr, d0StatErr;

  Color_t markerColor;
  int markerStyle;

  FitInfo(TString aDescriptor, 
          double aLambda1, double aLambdaStatErr1, 
          double aLambda2, double aLambdaStatErr2, 
          double aLambda3, double aLambdaStatErr3, 
          double aRadius1, double aRadiusStatErr1, 
          double aRadius2, double aRadiusStatErr2, 
          double aRadius3, double aRadiusStatErr3, 
          double aReF0,   double aReF0StatErr, 
          double aImF0,   double aImF0StatErr, 
          double aD0,     double aD0StatErr, 
          Color_t aMarkerColor, int aMarkerStyle)
  {
    descriptor = aDescriptor;

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

    markerColor = aMarkerColor;
    markerStyle = aMarkerStyle;
  }

};

//---------------------------------------------------------------------------------------------------------------------------------
void DrawReF0vsImF0(TPad* aPad, FitInfo &aFitInfo, TString aDrawOption="epsame")
{
  aPad->cd();
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  TGraphAsymmErrors* tGr = new TGraphAsymmErrors(1);

  tGr->SetName(aFitInfo.descriptor);
  tGr->SetMarkerStyle(aFitInfo.markerStyle);
  tGr->SetMarkerColor(aFitInfo.markerColor);
  tGr->SetFillColor(aFitInfo.markerColor);
  tGr->SetFillStyle(1000);
  tGr->SetLineColor(aFitInfo.markerColor);
  tGr->SetLineWidth(1);

  tGr->SetPoint(0, aFitInfo.ref0, aFitInfo.imf0);
  tGr->SetPointError(0, aFitInfo.ref0StatErr, aFitInfo.ref0StatErr, aFitInfo.imf0StatErr, aFitInfo.imf0StatErr);

  tGr->Draw(aDrawOption);
}

//---------------------------------------------------------------------------------------------------------------------------------
void DrawWeightedMeanF0vsImF0(TPad* aPad, vector<FitInfo> &aFitInfoVec, bool bInclude10Res=true, bool bInclude3Res=true, TString aDrawOption="epsame")
{
  double tNumX = 0., tDenX = 0.;
  double tNumY = 0., tDenY = 0.;

  for(unsigned int i=0; i<aFitInfoVec.size(); i++)
  {
    if(!bInclude10Res && i < aFitInfoVec.size()/2) continue; 
    if(!bInclude3Res && i >= aFitInfoVec.size()/2) continue;

    tNumX += aFitInfoVec[i].ref0/(aFitInfoVec[i].ref0StatErr*aFitInfoVec[i].ref0StatErr);
    tDenX += 1./(aFitInfoVec[i].ref0StatErr*aFitInfoVec[i].ref0StatErr);

    tNumY += aFitInfoVec[i].imf0/(aFitInfoVec[i].imf0StatErr*aFitInfoVec[i].imf0StatErr);
    tDenY += 1./(aFitInfoVec[i].imf0StatErr*aFitInfoVec[i].imf0StatErr);
  }

  double tAvgX = tNumX/tDenX;
  double tErrX = sqrt(1./tDenX);

  double tAvgY = tNumY/tDenY;
  double tErrY = sqrt(1./tDenY);

  //--------------------------------------
  aPad->cd();
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  TGraphAsymmErrors* tGr = new TGraphAsymmErrors(1);

  tGr->SetName("Average Re[f0] vs Im[f0]");
  tGr->SetMarkerStyle(29);
  tGr->SetMarkerSize(2.0);
  tGr->SetMarkerColor(6);
  tGr->SetFillColor(6);
  tGr->SetFillStyle(1000);
  tGr->SetLineColor(6);
  tGr->SetLineWidth(1);

  tGr->SetPoint(0, tAvgX, tAvgY);
  tGr->SetPointError(0, tErrX, tErrX, tErrY, tErrY);

  tGr->Draw(aDrawOption);

}

//---------------------------------------------------------------------------------------------------------------------------------
void DrawAverageF0vsImF0(TPad* aPad, vector<FitInfo> &aFitInfoVec, bool bInclude10Res=true, bool bInclude3Res=true, TString aDrawOption="epsame")
{
  double tNumX = 0., tDenX = 0.;
  double tNumY = 0., tDenY = 0.;

  for(unsigned int i=0; i<aFitInfoVec.size(); i++)
  {
    if(!bInclude10Res && i < aFitInfoVec.size()/2) continue; 
    if(!bInclude3Res && i >= aFitInfoVec.size()/2) continue;

    tNumX += aFitInfoVec[i].ref0;
    tDenX += 1.;

    tNumY += aFitInfoVec[i].imf0;
    tDenY += 1.;
  }

  double tAvgX = tNumX/tDenX;
  double tErrX = 0.;

  double tAvgY = tNumY/tDenY;
  double tErrY = 0.;

  //--------------------------------------
  aPad->cd();
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  TGraphAsymmErrors* tGr = new TGraphAsymmErrors(1);

  tGr->SetName("Average Re[f0] vs Im[f0]");
  tGr->SetMarkerStyle(30);
  tGr->SetMarkerSize(2.0);
  tGr->SetMarkerColor(6);
  tGr->SetFillColor(6);
  tGr->SetFillStyle(1000);
  tGr->SetLineColor(6);
  tGr->SetLineWidth(1);

  tGr->SetPoint(0, tAvgX, tAvgY);
  tGr->SetPointError(0, tErrX, tErrX, tErrY, tErrY);

  tGr->Draw(aDrawOption);

}

//---------------------------------------------------------------------------------------------------------------------------------
void DrawAllReF0vsImF0(AnalysisType aAnType, TPad* aPad, vector<FitInfo> &aFitInfoVec, bool bInclude10Res=true, bool bInclude3Res=true, bool bSaveImage=false, double aMinX=-2., double aMaxX=2., double aMinY=0., double aMaxY=2.)
{
  aPad->cd();
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  TH1D* tTrash = new TH1D("tTrash", "tTrash", 10, aMinX, aMaxX);
  tTrash->GetXaxis()->SetRangeUser(aMinX, aMaxX);
  tTrash->GetYaxis()->SetRangeUser(aMinY, aMaxY);

  tTrash->GetXaxis()->SetTitle("Re[f0]");
  tTrash->GetYaxis()->SetTitle("Im[f0]");

  tTrash->DrawCopy("axis");


  TLatex* tTex = new TLatex();
  tTex->SetTextAlign(12);
  tTex->SetLineWidth(2);
  tTex->SetTextFont(62);
  tTex->SetTextSize(0.08);

  if(aAnType == kLamKchM) tTex->DrawLatex(0.25, 1.35, cAnalysisRootTags[aAnType]);
  else tTex->DrawLatex(-1.75, 1.35, cAnalysisRootTags[aAnType]);

  tTex->SetTextFont(42);
  tTex->SetTextSize(0.04);

  double tStartX = -0.25;
  double tStartY = 1.4;

  if(aAnType == kLamKchM) tStartX = -1.75;

  double tIncrementY = 0.08;

  const Size_t tMarkerSize=1.6;
  TMarker *tMarker = new TMarker();
  tMarker->SetMarkerSize(tMarkerSize);

  for(unsigned int i=0; i<aFitInfoVec.size(); i++)
  {
    if(!bInclude10Res && i < aFitInfoVec.size()/2) continue; 
    if(!bInclude3Res && i >= aFitInfoVec.size()/2) continue; 

    DrawReF0vsImF0(aPad, aFitInfoVec[i]);

    tTex->DrawLatex(tStartX, tStartY-i*tIncrementY, aFitInfoVec[i].descriptor);
    tMarker->SetMarkerStyle(aFitInfoVec[i].markerStyle);
    tMarker->SetMarkerColor(aFitInfoVec[i].markerColor);
    tMarker->DrawMarker(tStartX-0.05, tStartY-i*tIncrementY);
  }

  //------------------------------------------------
  DrawWeightedMeanF0vsImF0(aPad, aFitInfoVec, bInclude10Res, bInclude3Res);
  if(bInclude10Res && !bInclude3Res) tTex->DrawLatex(tStartX, tStartY-(aFitInfoVec.size()/2)*tIncrementY, "Weighted Mean");
  else tTex->DrawLatex(tStartX, tStartY-aFitInfoVec.size()*tIncrementY, "Weighted Mean");
  tMarker->SetMarkerStyle(29);
  tMarker->SetMarkerColor(6);
  if(bInclude10Res && !bInclude3Res) tMarker->DrawMarker(tStartX-0.05, tStartY-(aFitInfoVec.size()/2)*tIncrementY);
  else tMarker->DrawMarker(tStartX-0.05, tStartY-aFitInfoVec.size()*tIncrementY);
  //------------------------------------------------
  DrawAverageF0vsImF0(aPad, aFitInfoVec, bInclude10Res, bInclude3Res);
  if(bInclude10Res && !bInclude3Res) tTex->DrawLatex(tStartX, tStartY-(aFitInfoVec.size()/2+1)*tIncrementY, "Average");
  else tTex->DrawLatex(tStartX, tStartY-(aFitInfoVec.size()+1)*tIncrementY, "Average");
  tMarker->SetMarkerStyle(30);
  tMarker->SetMarkerColor(6);
  if(bInclude10Res && !bInclude3Res) tMarker->DrawMarker(tStartX-0.05, tStartY-(aFitInfoVec.size()/2+1)*tIncrementY);
  else tMarker->DrawMarker(tStartX-0.05, tStartY-(aFitInfoVec.size()+1)*tIncrementY);
  //------------------------------------------------
  if(bSaveImage)
  {
    TString tSaveLocationBase = "/home/jesse/Analysis/Presentations/GroupMeetings/20171102/Figures/";
    TString tSaveLocationFull_ReF0vsImF0;

    tSaveLocationFull_ReF0vsImF0 = tSaveLocationBase + TString::Format("%s/ReF0vsImF0", cAnalysisBaseTags[aAnType]);
    if(bInclude10Res && bInclude3Res) tSaveLocationFull_ReF0vsImF0 += TString("_10ResAnd3Res");
    else if(bInclude10Res) tSaveLocationFull_ReF0vsImF0 += TString("_10Res");
    else if(bInclude3Res) tSaveLocationFull_ReF0vsImF0 += TString("_3Res");
    else assert(0);

    tSaveLocationFull_ReF0vsImF0 += TString(".eps");
    aPad->SaveAs(tSaveLocationFull_ReF0vsImF0);
  }

  tTrash->Delete();
}


//---------------------------------------------------------------------------------------------------------------------------------
void DrawRadiusvsLambda(TPad* aPad, FitInfo &aFitInfo, CentralityType aCentType=k0010, TString aDrawOption="epsame")
{
  aPad->cd();
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  TGraphAsymmErrors* tGr = new TGraphAsymmErrors(1);

  tGr->SetName(aFitInfo.descriptor);

  if(aCentType==k0010) tGr->SetMarkerStyle(aFitInfo.markerStyle);
  else if(aCentType==k1030) tGr->SetMarkerStyle(aFitInfo.markerStyle+1);
  else if(aCentType==k3050) tGr->SetMarkerStyle(aFitInfo.markerStyle+2);
  else assert(0);

  tGr->SetMarkerColor(aFitInfo.markerColor);
  tGr->SetFillColor(aFitInfo.markerColor);
  tGr->SetFillStyle(1000);
  tGr->SetLineColor(aFitInfo.markerColor);
  tGr->SetLineWidth(1);

  if(aCentType==k0010)
  {
    tGr->SetPoint(0, aFitInfo.radius1, aFitInfo.lambda1);
    tGr->SetPointError(0, aFitInfo.radiusStatErr1, aFitInfo.radiusStatErr1, aFitInfo.lambdaStatErr1, aFitInfo.lambdaStatErr1);
  }
  else if(aCentType==k1030)
  {
    tGr->SetPoint(0, aFitInfo.radius2, aFitInfo.lambda2);
    tGr->SetPointError(0, aFitInfo.radiusStatErr2, aFitInfo.radiusStatErr2, aFitInfo.lambdaStatErr2, aFitInfo.lambdaStatErr2);
  }
  else if(aCentType==k3050)
  {
    tGr->SetPoint(0, aFitInfo.radius3, aFitInfo.lambda3);
    tGr->SetPointError(0, aFitInfo.radiusStatErr3, aFitInfo.radiusStatErr3, aFitInfo.lambdaStatErr3, aFitInfo.lambdaStatErr3);
  }
  else assert(0);

  tGr->Draw(aDrawOption);
}

//---------------------------------------------------------------------------------------------------------------------------------
void DrawWeightedMeanRadiusvsLambda(TPad* aPad, vector<FitInfo> &aFitInfoVec, CentralityType aCentType, bool bInclude10Res=true, bool bInclude3Res=true, TString aDrawOption="epsame")
{
  double tNumX = 0., tDenX = 0.;
  double tNumY = 0., tDenY = 0.;

  double tRadius=0., tRadiusErr=0.;
  double tLambda=0., tLambdaErr=0.;

  for(unsigned int i=0; i<aFitInfoVec.size(); i++)
  {
    if(!bInclude10Res && i < aFitInfoVec.size()/2) continue; 
    if(!bInclude3Res && i >= aFitInfoVec.size()/2) continue;

    if(aCentType==k0010)
    {
      tRadius = aFitInfoVec[i].radius1;
      tRadiusErr = aFitInfoVec[i].radiusStatErr1;

      tLambda = aFitInfoVec[i].lambda1;
      tLambdaErr = aFitInfoVec[i].lambdaStatErr1;
    }
    else if(aCentType==k1030)
    {
      tRadius = aFitInfoVec[i].radius2;
      tRadiusErr = aFitInfoVec[i].radiusStatErr2;

      tLambda = aFitInfoVec[i].lambda2;
      tLambdaErr = aFitInfoVec[i].lambdaStatErr2;
    }
    else if(aCentType==k3050)
    {
      tRadius = aFitInfoVec[i].radius3;
      tRadiusErr = aFitInfoVec[i].radiusStatErr3;

      tLambda = aFitInfoVec[i].lambda3;
      tLambdaErr = aFitInfoVec[i].lambdaStatErr3;
    }
    else assert(0);

    //Problematic for two points with tRadiusErr = 0.
    //tNumX += tRadius/(tRadiusErr*tRadiusErr);
    //tDenX += 1./(tRadiusErr*tRadiusErr);
    tNumX += tRadius;
    tDenX += 1;

    tNumY += tLambda/(tLambdaErr*tLambdaErr);
    tDenY += 1./(tLambdaErr*tLambdaErr);
  }

  double tAvgX = tNumX/tDenX;
  //double tErrX = sqrt(1./tDenX);
  double tErrX = 0.;

  double tAvgY = tNumY/tDenY;
  double tErrY = sqrt(1./tDenY);

  //--------------------------------------
  aPad->cd();
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  TGraphAsymmErrors* tGr = new TGraphAsymmErrors(1);

  tGr->SetName("Average R vs #lambda");
  tGr->SetMarkerStyle(29);
  tGr->SetMarkerSize(2.0);
  tGr->SetMarkerColor(6);
  tGr->SetFillColor(6);
  tGr->SetFillStyle(1000);
  tGr->SetLineColor(6);
  tGr->SetLineWidth(1);

  tGr->SetPoint(0, tAvgX, tAvgY);
  tGr->SetPointError(0, tErrX, tErrX, tErrY, tErrY);

  tGr->Draw(aDrawOption);

}

//---------------------------------------------------------------------------------------------------------------------------------
void DrawAverageRadiusvsLambda(TPad* aPad, vector<FitInfo> &aFitInfoVec, CentralityType aCentType, bool bInclude10Res=true, bool bInclude3Res=true, TString aDrawOption="epsame")
{
  double tNumX = 0., tDenX = 0.;
  double tNumY = 0., tDenY = 0.;

  double tRadius=0., tRadiusErr=0.;
  double tLambda=0., tLambdaErr=0.;

  for(unsigned int i=0; i<aFitInfoVec.size(); i++)
  {
    if(!bInclude10Res && i < aFitInfoVec.size()/2) continue; 
    if(!bInclude3Res && i >= aFitInfoVec.size()/2) continue;

    if(aCentType==k0010)
    {
      tRadius = aFitInfoVec[i].radius1;
      tRadiusErr = aFitInfoVec[i].radiusStatErr1;

      tLambda = aFitInfoVec[i].lambda1;
      tLambdaErr = aFitInfoVec[i].lambdaStatErr1;
    }
    else if(aCentType==k1030)
    {
      tRadius = aFitInfoVec[i].radius2;
      tRadiusErr = aFitInfoVec[i].radiusStatErr2;

      tLambda = aFitInfoVec[i].lambda2;
      tLambdaErr = aFitInfoVec[i].lambdaStatErr2;
    }
    else if(aCentType==k3050)
    {
      tRadius = aFitInfoVec[i].radius3;
      tRadiusErr = aFitInfoVec[i].radiusStatErr3;

      tLambda = aFitInfoVec[i].lambda3;
      tLambdaErr = aFitInfoVec[i].lambdaStatErr3;
    }
    else assert(0);

    tNumX += tRadius;
    tDenX += 1;

    tNumY += tLambda;
    tDenY += 1.;
  }

  double tAvgX = tNumX/tDenX;
  double tErrX = 0.;

  double tAvgY = tNumY/tDenY;
  double tErrY = 0.;

  //--------------------------------------
  aPad->cd();
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  TGraphAsymmErrors* tGr = new TGraphAsymmErrors(1);

  tGr->SetName("Average R vs #lambda");
  tGr->SetMarkerStyle(30);
  tGr->SetMarkerSize(2.0);
  tGr->SetMarkerColor(6);
  tGr->SetFillColor(6);
  tGr->SetFillStyle(1000);
  tGr->SetLineColor(6);
  tGr->SetLineWidth(1);

  tGr->SetPoint(0, tAvgX, tAvgY);
  tGr->SetPointError(0, tErrX, tErrX, tErrY, tErrY);

  tGr->Draw(aDrawOption);

}

//---------------------------------------------------------------------------------------------------------------------------------
void DrawAllRadiusvsLambda(AnalysisType aAnType, TPad* aPad, vector<FitInfo> &aFitInfoVec, CentralityType aCentType=k0010, bool bInclude10Res=true, bool bInclude3Res=true, bool bSaveImage=false, double aMinX=0., double aMaxX=8., double aMinY=0., double aMaxY=2.)
{
  aPad->cd();
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  TH1D* tTrash = new TH1D("tTrash", "tTrash", 10, aMinX, aMaxX);
  tTrash->GetXaxis()->SetRangeUser(aMinX, aMaxX);
  tTrash->GetYaxis()->SetRangeUser(aMinY, aMaxY);

  tTrash->GetXaxis()->SetTitle("Radius");
  tTrash->GetYaxis()->SetTitle("#lambda");

  tTrash->DrawCopy("axis");


  TLatex* tTex = new TLatex();
  tTex->SetTextAlign(12);
  tTex->SetLineWidth(2);

  tTex->SetTextFont(62);
  tTex->SetTextSize(0.08);
  tTex->DrawLatex(0.5, 1.8, cAnalysisRootTags[aAnType]);

  tTex->SetTextFont(42);
  tTex->SetTextSize(0.06);
  tTex->DrawLatex(0.5, 1.6, cPrettyCentralityTags[aCentType]);

  tTex->SetTextSize(0.03);


  double tStartX = 5.5;
  double tStartY = 0.8;

  double tIncrementY = 0.08;

  const Size_t tMarkerSize=1.2;
  TMarker *tMarker = new TMarker();
  tMarker->SetMarkerSize(tMarkerSize);

  for(unsigned int i=0; i<aFitInfoVec.size(); i++)
  {

    if(!bInclude10Res && i < aFitInfoVec.size()/2) continue; 
    if(!bInclude3Res && i >= aFitInfoVec.size()/2) continue; 

    if(aCentType==kMB)
    {
      DrawRadiusvsLambda(aPad, aFitInfoVec[i], k0010);
      DrawRadiusvsLambda(aPad, aFitInfoVec[i], k1030);
      DrawRadiusvsLambda(aPad, aFitInfoVec[i], k3050);
    }
    else DrawRadiusvsLambda(aPad, aFitInfoVec[i], aCentType);

    tTex->DrawLatex(tStartX, tStartY-i*tIncrementY, aFitInfoVec[i].descriptor);
    tMarker->SetMarkerStyle(aFitInfoVec[i].markerStyle);
    tMarker->SetMarkerColor(aFitInfoVec[i].markerColor);
    tMarker->DrawMarker(tStartX-0.10, tStartY-i*tIncrementY);

  }

  //------------------------------------------------
  DrawWeightedMeanRadiusvsLambda(aPad, aFitInfoVec, aCentType, bInclude10Res, bInclude3Res);
  if(bInclude10Res && !bInclude3Res) tTex->DrawLatex(tStartX, tStartY-(aFitInfoVec.size()/2)*tIncrementY, "Weighted Mean");
  else tTex->DrawLatex(tStartX, tStartY-aFitInfoVec.size()*tIncrementY, "Weighted Mean");
  tMarker->SetMarkerStyle(29);
  tMarker->SetMarkerColor(6);
  if(bInclude10Res && !bInclude3Res) tMarker->DrawMarker(tStartX-0.05, tStartY-(aFitInfoVec.size()/2)*tIncrementY);
  else tMarker->DrawMarker(tStartX-0.05, tStartY-aFitInfoVec.size()*tIncrementY);
  //------------------------------------------------
  DrawAverageRadiusvsLambda(aPad, aFitInfoVec, aCentType, bInclude10Res, bInclude3Res);
  if(bInclude10Res && !bInclude3Res) tTex->DrawLatex(tStartX, tStartY-(aFitInfoVec.size()/2+1)*tIncrementY, "Average");
  else tTex->DrawLatex(tStartX, tStartY-(aFitInfoVec.size()+1)*tIncrementY, "Average");
  tMarker->SetMarkerStyle(30);
  tMarker->SetMarkerColor(6);
  if(bInclude10Res && !bInclude3Res) tMarker->DrawMarker(tStartX-0.05, tStartY-(aFitInfoVec.size()/2+1)*tIncrementY);
  else tMarker->DrawMarker(tStartX-0.05, tStartY-(aFitInfoVec.size()+1)*tIncrementY);

  //------------------------------------------------
  if(bSaveImage)
  {
    TString tSaveLocationBase = "/home/jesse/Analysis/Presentations/GroupMeetings/20171102/Figures/";
    TString tSaveLocationFull_RadiusvsLambda;

    tSaveLocationFull_RadiusvsLambda = tSaveLocationBase + TString::Format("%s/RadiusvsLambda%s", cAnalysisBaseTags[aAnType], cCentralityTags[aCentType]);
    if(bInclude10Res && bInclude3Res) tSaveLocationFull_RadiusvsLambda += TString("_10ResAnd3Res");
    else if(bInclude10Res) tSaveLocationFull_RadiusvsLambda += TString("_10Res");
    else if(bInclude3Res) tSaveLocationFull_RadiusvsLambda += TString("_3Res");
    else assert(0);

    tSaveLocationFull_RadiusvsLambda += TString(".eps");
    aPad->SaveAs(tSaveLocationFull_RadiusvsLambda);
  }


  tTrash->Delete();
}

//---------------------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************************
//---------------------------------------------------------------------------------------------------------------------------------
int main(int argc, char **argv) 
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything

//-----------------------------------------------------------------------------
  Color_t tColor1 = kBlack;
  Color_t tColor2 = kBlue;
  Color_t tColor3 = kRed;
  Color_t tColor4 = kGreen+2;

  int tMarkerStyleA = 20;
  int tMarkerStyleB = 24;

  //---------------------------------------------------------------------------
  //---------------------------- LamKchP --------------------------------------
  //---------------------------------------------------------------------------

  //--------------- 10 Residuals ----------
  FitInfo tFitInfo1a_LamKchP = FitInfo(TString("FreeRadii_FreeD0_10Res"), 
                                       0.5*(0.96+0.94), 0.5*(0.37+0.36),
                                       0.5*(1.18+0.99), 0.5*(0.51+0.42),
                                       0.5*(1.01+0.98), 0.5*(0.30+0.29),

                                       4.98, 0.96,
                                       4.76, 0.99,
                                       3.55, 0.52,

                                      -1.51, 0.37,
                                       0.65, 0.40,
                                       1.13, 0.74,
                                       tColor1, tMarkerStyleA);

  FitInfo tFitInfo2a_LamKchP = FitInfo(TString("FreeRadii_FixedD0_10Res"), 
                                       0.5*(1.33+1.32), 0.5*(0.40+0.40),
                                       0.5*(1.56+1.30), 0.5*(0.63+0.52),
                                       0.5*(1.16+1.12), 0.5*(0.31+0.30),

                                       5.26, 0.87,
                                       4.91, 1.00,
                                       3.37, 0.47,

                                      -1.08, 0.14,
                                       0.59, 0.30,
                                       0.00, 0.00,
                                       tColor2, tMarkerStyleA);

  FitInfo tFitInfo3a_LamKchP = FitInfo(TString("FixedRadii_FreeD0_10Res"), 
                                       0.5*(0.84+0.83), 0.5*(0.27+0.27),
                                       0.5*(0.96+0.81), 0.5*(0.31+0.26),
                                       0.5*(0.81+0.79), 0.5*(0.27+0.27),

                                       3.50, 0.00,
                                       3.25, 0.00,
                                       2.50, 0.00,

                                      -1.02, 0.36,
                                       0.08, 0.06,
                                       0.92, 0.38,
                                       tColor3, tMarkerStyleA);

  FitInfo tFitInfo4a_LamKchP = FitInfo(TString("FixedRadii_FixedD0_10Res"), 
                                       0.5*(1.03+1.03), 0.5*(0.32+0.32),
                                       0.5*(1.18+1.00), 0.5*(0.36+0.30),
                                       0.5*(1.05+1.02), 0.5*(0.32+0.32),

                                       3.50, 0.00,
                                       3.25, 0.00,
                                       2.50, 0.00,

                                      -0.76, 0.23,
                                       0.12, 0.07,
                                       0.00, 0.00,
                                       tColor4, tMarkerStyleA);


  //--------------- 3 Residuals ----------
  FitInfo tFitInfo1b_LamKchP = FitInfo(TString("FreeRadii_FreeD0_3Res"), 
                                       0.5*(0.84+0.82), 0.5*(0.31+0.30),
                                       0.5*(1.09+0.90), 0.5*(0.47+0.39),
                                       0.5*(1.02+0.97), 0.5*(0.30+0.29),

                                       4.43, 0.86,
                                       4.34, 0.94,
                                       3.38, 0.50,

                                      -1.24, 0.32,
                                       0.50, 0.35,
                                       1.11, 0.51,
                                       tColor1, tMarkerStyleB);

  FitInfo tFitInfo2b_LamKchP = FitInfo(TString("FreeRadii_FixedD0_3Res"), 
                                       0.5*(1.23+1.21), 0.5*(0.37+0.37),
                                       0.5*(1.54+1.27), 0.5*(0.59+0.48),
                                       0.5*(1.22+1.17), 0.5*(0.32+0.31),

                                       4.82, 0.81,
                                       4.62, 0.90,
                                       3.27, 0.44,

                                      -0.85, 0.13,
                                       0.49, 0.24,
                                       0.00, 0.00,
                                       tColor2, tMarkerStyleB);

  FitInfo tFitInfo3b_LamKchP = FitInfo(TString("FixedRadii_FreeD0_3Res"), 
                                       0.5*(0.83+0.82), 0.5*(0.27+0.27),
                                       0.5*(0.96+0.80), 0.5*(0.31+0.26),
                                       0.5*(0.82+0.80), 0.5*(0.28+0.28),

                                       3.50, 0.00,
                                       3.25, 0.00,
                                       2.50, 0.00,

                                      -0.90, 0.32,
                                       0.12, 0.06,
                                       1.06, 0.27,
                                       tColor3, tMarkerStyleB);

  FitInfo tFitInfo4b_LamKchP = FitInfo(TString("FixedRadii_FixedD0_3Res"), 
                                       0.5*(1.04+1.03), 0.5*(0.29+0.29),
                                       0.5*(1.20+1.00), 0.5*(0.33+0.28),
                                       0.5*(1.08+1.04), 0.5*(0.31+0.30),

                                       3.50, 0.00,
                                       3.25, 0.00,
                                       2.50, 0.00,

                                      -0.66, 0.18,
                                       0.15, 0.07,
                                       0.00, 0.00,
                                       tColor4, tMarkerStyleB);



  //---------------------------------------------------------------------------
  //---------------------------- LamKchM --------------------------------------
  //---------------------------------------------------------------------------

  //--------------- 10 Residuals ----------
  FitInfo tFitInfo1a_LamKchM = FitInfo(TString("FreeRadii_FreeD0_10Res"), 
                                       0.5*(1.50+1.49), 0.5*(0.72+0.69),
                                       0.5*(1.15+1.41), 0.5*(0.51+0.61),
                                       0.5*(1.07+0.80), 0.5*(0.73+0.37),

                                       6.21, 1.51,
                                       4.86, 1.15,
                                       2.86, 0.81,

                                       0.45, 0.23,
                                       0.52, 0.21,
                                      -4.81, 2.62,
                                       tColor1, tMarkerStyleA);

  FitInfo tFitInfo2a_LamKchM = FitInfo(TString("FreeRadii_FixedD0_10Res"), 
                                       0.5*(1.16+1.17), 0.5*(0.63+0.61),
                                       0.5*(0.97+1.19), 0.5*(0.52+0.63),
                                       0.5*(0.89+0.72), 0.5*(0.70+0.41),

                                       4.73, 0.77,
                                       3.91, 0.70,
                                       2.34, 0.53,

                                       0.34, 0.17,
                                       0.52, 0.27,
                                       0.00, 0.00,
                                       tColor2, tMarkerStyleA);

  FitInfo tFitInfo3a_LamKchM = FitInfo(TString("FixedRadii_FreeD0_10Res"), 
                                       0.5*(0.58+0.60), 0.5*(0.27+0.27),
                                       0.5*(0.58+0.72), 0.5*(0.26+0.32),
                                       0.5*(0.63+0.56), 0.5*(0.34+0.24),

                                       3.50, 0.00,
                                       3.25, 0.00,
                                       2.50, 0.00,

                                       0.23, 0.20,
                                       0.64, 0.51,
                                       1.81, 4.68,
                                       tColor3, tMarkerStyleA);

  FitInfo tFitInfo4a_LamKchM = FitInfo(TString("FixedRadii_FixedD0_10Res"), 
                                       0.5*(0.55+0.57), 0.5*(0.15+0.15),
                                       0.5*(0.56+0.69), 0.5*(0.15+0.18),
                                       0.5*(0.60+0.54), 0.5*(0.22+0.16),

                                       3.50, 0.00,
                                       3.25, 0.00,
                                       2.50, 0.00,

                                       0.25, 0.15,
                                       0.71, 0.33,
                                       0.00, 0.00,
                                       tColor4, tMarkerStyleA);


  //--------------- 3 Residuals ----------
  FitInfo tFitInfo1b_LamKchM = FitInfo(TString("FreeRadii_FreeD0_3Res"), 
                                       0.5*(1.55+1.54), 0.5*(0.72+0.69),
                                       0.5*(1.19+1.46), 0.5*(0.51+0.61),
                                       0.5*(1.08+0.80), 0.5*(0.71+0.36),

                                       6.02, 1.46,
                                       4.74, 1.11,
                                       2.75, 0.80,

                                       0.34, 0.21,
                                       0.42, 0.19,
                                      -5.72, 3.39,
                                       tColor1, tMarkerStyleB);

  FitInfo tFitInfo2b_LamKchM = FitInfo(TString("FreeRadii_FixedD0_3Res"), 
                                       0.5*(1.20+1.20), 0.5*(0.60+0.58),
                                       0.5*(1.01+1.24), 0.5*(0.51+0.62),
                                       0.5*(0.94+0.75), 0.5*(0.73+0.41),

                                       4.33, 0.64,
                                       3.61, 0.62,
                                       2.12, 0.46,

                                       0.21, 0.14,
                                       0.39, 0.22,
                                       0.00, 0.00,
                                       tColor2, tMarkerStyleB);

  FitInfo tFitInfo3b_LamKchM = FitInfo(TString("FixedRadii_FreeD0_3Res"), 
                                       0.5*(0.65+0.67), 0.5*(0.32+0.32),
                                       0.5*(0.65+0.81), 0.5*(0.31+0.38),
                                       0.5*(0.72+0.62), 0.5*(0.44+0.29),

                                       3.50, 0.00,
                                       3.25, 0.00,
                                       2.50, 0.00,

                                       0.12, 0.13,
                                       0.45, 0.36,
                                      -4.68, 4.91,
                                       tColor3, tMarkerStyleB);

  FitInfo tFitInfo4b_LamKchM = FitInfo(TString("FixedRadii_FixedD0_3Res"), 
                                       0.5*(0.78+0.80), 0.5*(0.38+0.38),
                                       0.5*(0.78+0.96), 0.5*(0.37+0.45),
                                       0.5*(0.93+0.74), 0.5*(0.62+0.36),

                                       3.50, 0.00,
                                       3.25, 0.00,
                                       2.50, 0.00,

                                       0.15, 0.10,
                                       0.42, 0.28,
                                       0.00, 0.00,
                                       tColor4, tMarkerStyleB);



  //---------------------------------------------------------------------------
  //---------------------------- LamK0 --------------------------------------
  //---------------------------------------------------------------------------

  //--------------- 10 Residuals ----------
  FitInfo tFitInfo1a_LamK0 = FitInfo(TString("FreeRadii_FreeD0_10Res"), 
                                       0.5*(0.60+0.60), 0.5*(0.76+0.76),
                                       0.5*(0.60+0.60), 0.5*(0.76+0.76),
                                       0.5*(0.60+0.60), 0.5*(0.76+0.76),

                                       2.97, 0.49,
                                       2.30, 0.39,
                                       1.70, 0.29,

                                      -0.26, 0.07,
                                       0.17, 0.07,
                                       2.53, 0.68,
                                       tColor1, tMarkerStyleA);

  FitInfo tFitInfo2a_LamK0 = FitInfo(TString("FreeRadii_FixedD0_10Res"), 
                                       0.5*(1.50+1.44), 0.5*(0.65+0.16),
                                       0.5*(0.60+0.60), 0.5*(0.17+0.75),
                                       0.5*(1.50+1.40), 0.5*(0.59+0.24),

                                       3.53, 1.01,
                                       1.72, 0.45,
                                       1.99, 0.53,

                                      -0.12, 0.03,
                                       0.11, 0.10,
                                       0.00, 0.00,
                                       tColor2, tMarkerStyleA);

  FitInfo tFitInfo3a_LamK0 = FitInfo(TString("FixedRadii_FreeD0_10Res"), 
                                       0.5*(1.50+1.50), 0.5*(0.83+0.83),
                                       0.5*(1.50+1.50), 0.5*(0.83+0.83),
                                       0.5*(1.50+1.50), 0.5*(0.83+0.83),

                                       3.50, 0.00,
                                       3.25, 0.00,
                                       2.50, 0.00,

                                      -0.11, 0.03,
                                       0.10, 0.01,
                                      -0.73, 2.57,
                                       tColor3, tMarkerStyleA);

  FitInfo tFitInfo4a_LamK0 = FitInfo(TString("FixedRadii_FixedD0_10Res"), 
                                       0.5*(0.99+0.93), 0.5*(0.16+0.16),
                                       0.5*(1.11+1.14), 0.5*(0.20+0.20),
                                       0.5*(1.50+1.40), 0.5*(0.89+0.65),

                                       3.50, 0.00,
                                       3.25, 0.00,
                                       2.50, 0.00,

                                      -0.12, 0.04,
                                       0.16, 0.04,
                                       0.00, 0.00,
                                       tColor4, tMarkerStyleA);


  //--------------- 3 Residuals ----------
  FitInfo tFitInfo1b_LamK0 = FitInfo(TString("FreeRadii_FreeD0_3Res"), 
                                       0.5*(0.60+0.60), 0.5*(0.77+0.77),
                                       0.5*(0.60+0.60), 0.5*(0.77+0.77),
                                       0.5*(0.60+0.60), 0.5*(0.77+0.77),

                                       2.91, 0.51,
                                       2.22, 0.40,
                                       1.64, 0.30,

                                      -0.27, 0.06,
                                       0.21, 0.10,
                                       2.66, 0.58,
                                       tColor1, tMarkerStyleB);

  FitInfo tFitInfo2b_LamK0 = FitInfo(TString("FreeRadii_FixedD0_3Res"), 
                                       0.5*(1.50+1.50), 0.5*(0.90+0.90),
                                       0.5*(1.50+1.50), 0.5*(0.90+0.90),
                                       0.5*(1.50+1.50), 0.5*(0.90+0.90),

                                       3.48, 0.96,
                                       2.63, 0.70,
                                       1.89, 0.49,

                                      -0.08, 0.03,
                                       0.15, 0.11,
                                       0.00, 0.00,
                                       tColor2, tMarkerStyleB);

  FitInfo tFitInfo3b_LamK0 = FitInfo(TString("FixedRadii_FreeD0_3Res"), 
                                       0.5*(0.60+0.60), 0.5*(0.63+0.63),
                                       0.5*(0.60+0.60), 0.5*(0.63+0.63),
                                       0.5*(0.60+0.60), 0.5*(0.63+0.63),

                                       3.50, 0.00,
                                       3.25, 0.00,
                                       2.50, 0.00,

                                      -0.35, 0.08,
                                       0.27, 0.05,
                                       2.72, 0.54,
                                       tColor3, tMarkerStyleB);

  FitInfo tFitInfo4b_LamK0 = FitInfo(TString("FixedRadii_FixedD0_3Res"), 
                                       0.5*(1.50+1.50), 0.5*(0.79+0.79),
                                       0.5*(1.50+1.50), 0.5*(0.79+0.79),
                                       0.5*(1.50+1.50), 0.5*(0.79+0.79),

                                       3.50, 0.00,
                                       3.25, 0.00,
                                       2.50, 0.00,

                                      -0.10, 0.03,
                                       0.14, 0.02,
                                       0.00, 0.00,
                                       tColor4, tMarkerStyleB);


//---------------------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************************
//---------------------------------------------------------------------------------------------------------------------------------

  bool bInclude10Res = true;
  bool bInclude3Res = true;
  CentralityType tCentType = k0010;

  bool bSaveFigures = false;

//-------------------------------------------------------------------------------
  vector<FitInfo> tFitInfoVec_LamKchP{tFitInfo1a_LamKchP, tFitInfo2a_LamKchP, tFitInfo3a_LamKchP, tFitInfo4a_LamKchP,
                                      tFitInfo1b_LamKchP, tFitInfo2b_LamKchP, tFitInfo3b_LamKchP, tFitInfo4b_LamKchP};

  TCanvas* tCanReF0vsImF0_LamKchP = new TCanvas("tCanReF0vsImF0_LamKchP", "tCanReF0vsImF0_LamKchP");
  double aMinX_LamKchP = -2.;
  double aMaxX_LamKchP = 1.;

  double aMinY_LamKchP = 0.;
  double aMaxY_LamKchP = 1.5;

  DrawAllReF0vsImF0(kLamKchP, (TPad*)tCanReF0vsImF0_LamKchP, tFitInfoVec_LamKchP, bInclude10Res, bInclude3Res, bSaveFigures, aMinX_LamKchP, aMaxX_LamKchP, aMinY_LamKchP, aMaxY_LamKchP);

  //-------------
  TCanvas* tCanRadiusvsLambda_LamKchP = new TCanvas("tCanRadiusvsLambda_LamKchP", "tCanRadiusvsLambda_LamKchP");
  DrawAllRadiusvsLambda(kLamKchP, (TPad*)tCanRadiusvsLambda_LamKchP, tFitInfoVec_LamKchP, tCentType, bInclude10Res, bInclude3Res, bSaveFigures);

  

//-------------------------------------------------------------------------------

  vector<FitInfo> tFitInfoVec_LamKchM{tFitInfo1a_LamKchM, tFitInfo2a_LamKchM, tFitInfo3a_LamKchM, tFitInfo4a_LamKchM,
                                      tFitInfo1b_LamKchM, tFitInfo2b_LamKchM, tFitInfo3b_LamKchM, tFitInfo4b_LamKchM};

  TCanvas* tCanReF0vsImF0_LamKchM = new TCanvas("tCanReF0vsImF0_LamKchM", "tCanReF0vsImF0_LamKchM");
  double aMinX_LamKchM = -2.;
  double aMaxX_LamKchM = 1.;

  double aMinY_LamKchM = 0.;
  double aMaxY_LamKchM = 1.5;

  DrawAllReF0vsImF0(kLamKchM, (TPad*)tCanReF0vsImF0_LamKchM, tFitInfoVec_LamKchM, bInclude10Res, bInclude3Res, bSaveFigures, aMinX_LamKchM, aMaxX_LamKchM, aMinY_LamKchM, aMaxY_LamKchM);

  //-------------
  TCanvas* tCanRadiusvsLambda_LamKchM = new TCanvas("tCanRadiusvsLambda_LamKchM", "tCanRadiusvsLambda_LamKchM");
  DrawAllRadiusvsLambda(kLamKchM, (TPad*)tCanRadiusvsLambda_LamKchM, tFitInfoVec_LamKchM, tCentType, bInclude10Res, bInclude3Res, bSaveFigures);

//-------------------------------------------------------------------------------

  vector<FitInfo> tFitInfoVec_LamK0{tFitInfo1a_LamK0, tFitInfo2a_LamK0, tFitInfo3a_LamK0, tFitInfo4a_LamK0,
                                      tFitInfo1b_LamK0, tFitInfo2b_LamK0, tFitInfo3b_LamK0, tFitInfo4b_LamK0};

  TCanvas* tCanReF0vsImF0_LamK0 = new TCanvas("tCanReF0vsImF0_LamK0", "tCanReF0vsImF0_LamK0");
  double aMinX_LamK0 = -2.;
  double aMaxX_LamK0 = 1.;

  double aMinY_LamK0 = 0.;
  double aMaxY_LamK0 = 1.5;

  DrawAllReF0vsImF0(kLamK0, (TPad*)tCanReF0vsImF0_LamK0, tFitInfoVec_LamK0, bInclude10Res, bInclude3Res, bSaveFigures, aMinX_LamK0, aMaxX_LamK0, aMinY_LamK0, aMaxY_LamK0);

  //-------------
  TCanvas* tCanRadiusvsLambda_LamK0 = new TCanvas("tCanRadiusvsLambda_LamK0", "tCanRadiusvsLambda_LamK0");
  DrawAllRadiusvsLambda(kLamK0, (TPad*)tCanRadiusvsLambda_LamK0, tFitInfoVec_LamK0, tCentType, bInclude10Res, bInclude3Res, bSaveFigures);


//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.
cout << "DONE" << endl;
  return 0;
}








