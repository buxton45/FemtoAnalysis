#include "TSystem.h"
#include "TLegend.h"

#include "CompareFittingMethods.h"
TString gSaveLocationBase = "/home/jesse/Analysis/Presentations/PWGCF/LamKPaperProposal/ALICE_MiniWeek_20180115/Figures/";
//TString gSaveType = "eps";
TString gSaveType = "pdf";  // must save as pdf for transparency to work

//---------------------------------------------------------------------------------------------------------------------------------
void DrawChi2PerNDF(double aStartX, double aStartY, TPad* aPad, vector<FitInfo> &aFitInfoVec)
{
  aPad->cd();
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  TLatex* tTex = new TLatex();
  tTex->SetTextAlign(12);
  tTex->SetLineWidth(2);
  tTex->SetTextFont(42);
  tTex->SetTextSize(0.02);

  const Size_t tMarkerSize=1.6;
  TMarker *tMarker = new TMarker();
  tMarker->SetMarkerSize(tMarkerSize);

  double tIncrementY = 0.08;
  TString tText;
  int iTex = 0;
  for(unsigned int i=0; i<aFitInfoVec.size(); i++)
  {
    tText = TString::Format("#chi^{2}/NDF = %0.1f/%i = %0.3f", aFitInfoVec[i].chi2, aFitInfoVec[i].ndf, (aFitInfoVec[i].chi2/aFitInfoVec[i].ndf));
    tTex->DrawLatex(aStartX, aStartY-iTex*tIncrementY, tText);
    tMarker->SetMarkerStyle(aFitInfoVec[i].markerStyle);
    tMarker->SetMarkerColor(aFitInfoVec[i].markerColor);
    tMarker->DrawMarker(aStartX-0.10, aStartY-iTex*tIncrementY);
    iTex++;
  }

}

//---------------------------------------------------------------------------------------------------------------------------------
void DrawAnalysisStamps(TPad* aPad, double aStartX, double aStartY, double aIncrementX, double aIncrementY, double aTextSize=0.04, int aMarkerStyle=20)
{
  aPad->cd();

  TLatex* tTex = new TLatex();
  tTex->SetTextAlign(12);
  tTex->SetLineWidth(2);
  tTex->SetTextFont(42);
  tTex->SetTextSize(aTextSize);

  const Size_t tMarkerSize=1.6;
  TMarker *tMarker = new TMarker();
  tMarker->SetMarkerSize(tMarkerSize);
  tMarker->SetMarkerStyle(aMarkerStyle);

  int iTex = 0;

  tTex->DrawLatex(aStartX, aStartY-iTex*aIncrementY, cAnalysisRootTags[kLamKchP]);
  tMarker->SetMarkerColor(kRed+1);
  tMarker->DrawMarker(aStartX-aIncrementX, aStartY-iTex*aIncrementY);
  iTex++;

  tTex->DrawLatex(aStartX, aStartY-iTex*aIncrementY, cAnalysisRootTags[kLamKchM]);
  tMarker->SetMarkerColor(kBlue+1);
  tMarker->DrawMarker(aStartX-aIncrementX, aStartY-iTex*aIncrementY);
  iTex++;

  tTex->DrawLatex(aStartX, aStartY-iTex*aIncrementY, cAnalysisRootTags[kLamK0]);
  tMarker->SetMarkerColor(kBlack);
  tMarker->DrawMarker(aStartX-aIncrementX, aStartY-iTex*aIncrementY);
  iTex++;
}

//---------------------------------------------------------------------------------------------------------------------------------
void DrawFixedRadiiStamps(TPad* aPad, double aStartX, double aStartY, double aIncrementX, double aIncrementY, double aTextSize=0.04, int aMarkerStyle=20)
{
  aPad->cd();

  TLatex* tTex = new TLatex();
  tTex->SetTextAlign(12);
  tTex->SetLineWidth(2);
  tTex->SetTextFont(42);
  tTex->SetTextSize(aTextSize);

  const Size_t tMarkerSize=1.6;
  TMarker *tMarker = new TMarker();
  tMarker->SetMarkerSize(tMarkerSize);
  tMarker->SetMarkerStyle(aMarkerStyle);

  int iTex = 0;

  tTex->DrawLatex(aStartX, aStartY-(iTex-0.75)*aIncrementY, "Fixed Radii");

  tMarker->SetMarkerColor(tFitInfo5a_LamKchP.markerColor);
  tMarker->DrawMarker(aStartX+(2.*aIncrementX), aStartY-iTex*aIncrementY);
  iTex++;

  tMarker->SetMarkerColor(tFitInfo5a_LamKchM.markerColor);
  tMarker->DrawMarker(aStartX+(2.*aIncrementX), aStartY-iTex*aIncrementY);
  iTex++;

  tMarker->SetMarkerColor(tFitInfo5a_LamK0.markerColor);
  tMarker->DrawMarker(aStartX+(2.*aIncrementX), aStartY-iTex*aIncrementY);
  iTex++;
}


//---------------------------------------------------------------------------------------------------------------------------------
void SetupReF0vsImF0Axes(TPad* aPad, double aMinReF0=-2., double aMaxReF0=1., double aMinImF0=0., double aMaxImF0=1.5)
{
  aPad->cd();
  TH1D* tTrash = new TH1D("tTrash", "tTrash", 10, aMinReF0, aMaxReF0);
  tTrash->GetXaxis()->SetRangeUser(aMinReF0, aMaxReF0);
  tTrash->GetYaxis()->SetRangeUser(aMinImF0, aMaxImF0);

  tTrash->GetXaxis()->SetTitle("Re[f_{0}]");
  tTrash->GetYaxis()->SetTitle("Im[f_{0}]");

  tTrash->DrawCopy("axis");
  tTrash->Delete();
}


//---------------------------------------------------------------------------------------------------------------------------------
void SetupReF0vsImF0AndD0Axes(TPad* aPadReF0vsImF0, TPad* aPadD0, 
                              double aMinReF0=-2., double aMaxReF0=1., double aMinImF0=0., double aMaxImF0=1.5,
                              double aMinD0=-10., double aMaxD0=5.)
{
  aPadReF0vsImF0->cd();
  TH1D* tTrash1 = new TH1D("tTrash1", "tTrash1", 10, aMinReF0, aMaxReF0);
  tTrash1->GetXaxis()->SetRangeUser(aMinReF0, aMaxReF0);
  tTrash1->GetYaxis()->SetRangeUser(aMinImF0, aMaxImF0);

  tTrash1->GetXaxis()->SetTitle("Re[f_{0}]");
  tTrash1->GetYaxis()->SetTitle("Im[f_{0}]");

  tTrash1->DrawCopy("axis");

  //------------------------
  aPadD0->cd();
  TH1D* tTrash2 = new TH1D("tTrash2", "tTrash2", 1, 0., 1.);
  tTrash2->GetXaxis()->SetRangeUser(0., 1.);
  tTrash2->GetYaxis()->SetRangeUser(aMinD0, aMaxD0);

  double tScale = aPadReF0vsImF0->GetAbsWNDC()/aPadD0->GetAbsWNDC();
  tTrash2->GetYaxis()->SetLabelSize(tScale*tTrash1->GetYaxis()->GetLabelSize());
  tTrash2->GetYaxis()->SetLabelOffset(0.05);
  tTrash2->GetYaxis()->SetTitleSize(tScale*tTrash1->GetYaxis()->GetTitleSize());

  tTrash2->GetYaxis()->SetTitle("d_{0}");

  tTrash2->GetXaxis()->SetLabelSize(0.0);
  tTrash2->GetXaxis()->SetTickLength(0.0);

  tTrash2->DrawCopy("Y+ axis");
  //------------------------
  tTrash1->Delete();
  tTrash2->Delete();
}


//---------------------------------------------------------------------------------------------------------------------------------
void SetupRadiusvsLambdaAxes(TPad* aPad, double aMinR=0., double aMaxR=8., double aMinLam=0., double aMaxLam=2.)
{
  aPad->cd();
  TH1D* tTrash = new TH1D("tTrash", "tTrash", 10, aMinR, aMaxR);
  tTrash->GetXaxis()->SetRangeUser(aMinR, aMaxR);
  tTrash->GetYaxis()->SetRangeUser(aMinLam, aMaxLam);

  tTrash->GetXaxis()->SetTitle("Radius");
  tTrash->GetYaxis()->SetTitle("#lambda");

  tTrash->DrawCopy("axis");
  tTrash->Delete();
}



//---------------------------------------------------------------------------------------------------------------------------------
TGraphAsymmErrors* GetReF0vsImF0(const FitInfo &aFitInfo, ErrorType aErrType=kStat)
{
  TGraphAsymmErrors* tReturnGr = new TGraphAsymmErrors(1);
  tReturnGr->SetPoint(0, aFitInfo.ref0, aFitInfo.imf0);
  if(aErrType==kStat) tReturnGr->SetPointError(0, aFitInfo.ref0StatErr, aFitInfo.ref0StatErr, aFitInfo.imf0StatErr, aFitInfo.imf0StatErr);
  else if(aErrType==kSys) tReturnGr->SetPointError(0, aFitInfo.ref0SysErr, aFitInfo.ref0SysErr, aFitInfo.imf0SysErr, aFitInfo.imf0SysErr);
  else assert(0);

  return tReturnGr;
}

//---------------------------------------------------------------------------------------------------------------------------------
TGraphAsymmErrors* GetD0(const FitInfo &aFitInfo, ErrorType aErrType=kStat, double aXOffset=0.5)
{
  TGraphAsymmErrors* tReturnGr = new TGraphAsymmErrors(1);
  tReturnGr->SetPoint(0, aXOffset, aFitInfo.d0);
  if(aErrType==kStat) tReturnGr->SetPointError(0, 0., 0., aFitInfo.d0StatErr, aFitInfo.d0StatErr);
  else if(aErrType==kSys) tReturnGr->SetPointError(0, 0.05, 0.05, aFitInfo.d0SysErr, aFitInfo.d0SysErr);
  else assert(0);

  return tReturnGr;
}

//---------------------------------------------------------------------------------------------------------------------------------
TGraphAsymmErrors* GetRadiusvsLambda(const FitInfo &aFitInfo, CentralityType aCentType=k0010, ErrorType aErrType=kStat)
{
  TGraphAsymmErrors* tReturnGr = new TGraphAsymmErrors(1);
  tReturnGr->SetName(aFitInfo.descriptor);

  assert(aCentType != kMB);
  tReturnGr->SetPoint(0, aFitInfo.radiusVec[aCentType], aFitInfo.lambdaVec[aCentType]);
  if(aErrType==kStat)
  {
    tReturnGr->SetPointError(0, aFitInfo.radiusStatErrVec[aCentType], aFitInfo.radiusStatErrVec[aCentType], 
                                aFitInfo.lambdaStatErrVec[aCentType], aFitInfo.lambdaStatErrVec[aCentType]);
  }
  else if(aErrType==kSys)
  {
    tReturnGr->SetPointError(0, aFitInfo.radiusSysErrVec[aCentType], aFitInfo.radiusSysErrVec[aCentType], 
                                aFitInfo.lambdaSysErrVec[aCentType], aFitInfo.lambdaSysErrVec[aCentType]);
  }
  else assert(0);

  return tReturnGr;
}

//---------------------------------------------------------------------------------------------------------------------------------
void DrawReF0vsImF0(TPad* aPad, FitInfo &aFitInfo, ErrorType aErrType=kStat)
{
  aPad->cd();
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  TString tDrawOption_Stat, tDrawOption_Syst;
  int tColor_Stat, tColor_Syst;

  //With statistical errors
  TGraphAsymmErrors* tGr_Stat = GetReF0vsImF0(aFitInfo, kStat);
  tDrawOption_Stat = TString("pzsame");
  tColor_Stat = aFitInfo.markerColor;
    tGr_Stat->SetName(aFitInfo.descriptor + cErrorTypeTags[kStat]);
    tGr_Stat->SetLineWidth(1);
    tGr_Stat->SetMarkerStyle(aFitInfo.markerStyle);
    tGr_Stat->SetMarkerColor(tColor_Stat);
    tGr_Stat->SetFillColor(tColor_Stat);
    tGr_Stat->SetFillStyle(1000);
    tGr_Stat->SetLineColor(tColor_Stat);

  //With systematic errors
  TGraphAsymmErrors* tGr_Syst = GetReF0vsImF0(aFitInfo, kSys);
  tDrawOption_Syst = TString("e2same");
  tColor_Syst = TColor::GetColorTransparent(aFitInfo.markerColor, 0.3);
    tGr_Syst->SetName(aFitInfo.descriptor + cErrorTypeTags[kSys]);
    tGr_Syst->SetLineWidth(0);
    tGr_Syst->SetMarkerStyle(aFitInfo.markerStyle);
    tGr_Syst->SetMarkerColor(tColor_Syst);
    tGr_Syst->SetFillColor(tColor_Syst);
    tGr_Syst->SetFillStyle(1000);
    tGr_Syst->SetLineColor(tColor_Syst);

  if(aErrType==kStatAndSys)
  {
    tGr_Syst->Draw(tDrawOption_Syst);
    tGr_Stat->Draw(tDrawOption_Stat);
  }
  else if(aErrType==kStat) tGr_Stat->Draw(tDrawOption_Stat);
  else if(aErrType==kSys)  tGr_Syst->Draw(tDrawOption_Syst);
  else assert(0);
}

//---------------------------------------------------------------------------------------------------------------------------------
void DrawD0(TPad* aPad, FitInfo &aFitInfo, ErrorType aErrType=kStat, double aXOffset=0.5)
{
  aPad->cd();
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  TString tDrawOption_Stat, tDrawOption_Syst;
  int tColor_Stat, tColor_Syst;


  //With statistical errors
  TGraphAsymmErrors* tGr_Stat = GetD0(aFitInfo, kStat, aXOffset);
  tDrawOption_Stat = TString("pzsame");
  tColor_Stat = aFitInfo.markerColor;
    tGr_Stat->SetName(aFitInfo.descriptor + cErrorTypeTags[kStat]);
    tGr_Stat->SetLineWidth(1);
    tGr_Stat->SetMarkerStyle(aFitInfo.markerStyle);
    tGr_Stat->SetMarkerColor(tColor_Stat);
    tGr_Stat->SetFillColor(tColor_Stat);
    tGr_Stat->SetFillStyle(1000);
    tGr_Stat->SetLineColor(tColor_Stat);

  //With systematic errors
  TGraphAsymmErrors* tGr_Syst = GetD0(aFitInfo, kSys, aXOffset);
  tDrawOption_Syst = TString("e2same");
  tColor_Syst = TColor::GetColorTransparent(aFitInfo.markerColor, 0.3);
    tGr_Syst->SetName(aFitInfo.descriptor + cErrorTypeTags[kSys]);   
    tGr_Syst->SetLineWidth(0);
    tGr_Syst->SetMarkerStyle(aFitInfo.markerStyle);
    tGr_Syst->SetMarkerColor(tColor_Syst);
    tGr_Syst->SetFillColor(tColor_Syst);
    tGr_Syst->SetFillStyle(1000);
    tGr_Syst->SetLineColor(tColor_Syst);


  double tX=0., tY=0;
  tGr_Stat->GetPoint(0, tX, tY);
  if(tY != 0.)
  {
    if(aErrType==kStatAndSys)
    {
      tGr_Syst->Draw(tDrawOption_Syst);
      tGr_Stat->Draw(tDrawOption_Stat);
    }
    else if(aErrType==kStat) tGr_Stat->Draw(tDrawOption_Stat);
    else if(aErrType==kSys)  tGr_Syst->Draw(tDrawOption_Syst);
    else assert(0);
  }
}

//---------------------------------------------------------------------------------------------------------------------------------
void DrawRadiusvsLambda(TPad* aPad, FitInfo &aFitInfo, CentralityType aCentType=k0010, ErrorType aErrType=kStat)
{
  aPad->cd();
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  TString tDrawOption_Stat, tDrawOption_Syst;
  int tColor_Stat, tColor_Syst;


  //With statistical errors
  TGraphAsymmErrors* tGr_Stat = GetRadiusvsLambda(aFitInfo, aCentType, kStat);
  tDrawOption_Stat = TString("pzsame");
  tColor_Stat = aFitInfo.markerColor;
    tGr_Stat->SetName(aFitInfo.descriptor + cErrorTypeTags[kStat]);
    tGr_Stat->SetLineWidth(1);
    tGr_Stat->SetMarkerStyle(aFitInfo.markerStyle);
    tGr_Stat->SetMarkerColor(tColor_Stat);
    tGr_Stat->SetFillColor(tColor_Stat);
    tGr_Stat->SetFillStyle(1000);
    tGr_Stat->SetLineColor(tColor_Stat);


  //With systematic errors
  TGraphAsymmErrors* tGr_Syst = GetRadiusvsLambda(aFitInfo, aCentType, kSys);
  tDrawOption_Syst = TString("e2same");
  tColor_Syst = TColor::GetColorTransparent(aFitInfo.markerColor, 0.3);
    tGr_Syst->SetName(aFitInfo.descriptor + cErrorTypeTags[kSys]);
    tGr_Syst->SetLineWidth(0);
    tGr_Syst->SetMarkerStyle(aFitInfo.markerStyle);
    tGr_Syst->SetMarkerColor(tColor_Syst);
    tGr_Syst->SetFillColor(tColor_Syst);
    tGr_Syst->SetFillStyle(1000);
    tGr_Syst->SetLineColor(tColor_Syst);

  if(aErrType==kStatAndSys)
  {
    tGr_Syst->Draw(tDrawOption_Syst);
    tGr_Stat->Draw(tDrawOption_Stat);
  }
  else if(aErrType==kStat) tGr_Stat->Draw(tDrawOption_Stat);
  else if(aErrType==kSys)  tGr_Syst->Draw(tDrawOption_Syst);
  else assert(0);
}


//---------------------------------------------------------------------------------------------------------------------------------
TCanvas* DrawMultipleReF0vsImF0(AnalysisType aAnType, 
                                IncludeResType aIncludeResType=kInclude10ResAnd3Res, IncludeD0Type aIncludeD0Type=kFreeAndFixedD0,
                                IncludeRadiiType aIncludeRadiiType=kFreeAndFixedRadii, IncludeLambdaType aIncludeLambdaType=kFreeAndFixedLambda, 
                                ErrorType aErrType=kStatAndSys)
{
/*
  TCanvas* tReturnCan = new TCanvas(TString::Format("tCanRadiusvsLambda_%s%s", cAnalysisBaseTags[aAnType], cCentralityTags[aCentType]), 
                                    TString::Format("tCanRadiusvsLambda_%s%s", cAnalysisBaseTags[aAnType], cCentralityTags[aCentType]));
*/

  TCanvas* tReturnCan = new TCanvas(TString::Format("tCanReF0vsImF0_%s", cAnalysisBaseTags[aAnType]), 
                                    TString::Format("tCanReF0vsImF0_%s", cAnalysisBaseTags[aAnType]));
  tReturnCan->cd();
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);
  //------------------------
  vector<FitInfo> aFitInfoVec = GetFitInfoVec(aAnType, aIncludeResType, aIncludeD0Type, aIncludeRadiiType, aIncludeLambdaType);
  //------------------------
  TPad* tPadReF0vsImF0 = new TPad("PadReF0vsImF0", "PadReF0vsImF0", 0.0, 0.0, 0.8, 1.0);
  tPadReF0vsImF0->SetRightMargin(0.01);
  tPadReF0vsImF0->Draw();

  TPad* tPadD0 = new TPad("PadD0", "PadD0", 0.8, 0.0, 1.0, 1.0);
  tPadD0->SetRightMargin(0.4);
  tPadD0->SetLeftMargin(0.);
  tPadD0->Draw();

  SetupReF0vsImF0AndD0Axes(tPadReF0vsImF0, tPadD0);
  //------------------------

  for(unsigned int i=0; i<aFitInfoVec.size(); i++)
  {
    DrawReF0vsImF0((TPad*)tPadReF0vsImF0, aFitInfoVec[i], aErrType);
    DrawD0((TPad*)tPadD0, aFitInfoVec[i], aErrType);

  }
  tPadReF0vsImF0->cd();

  //--------------------------------------------------------
  double tStartXChi2 = -0.50;
  double tStartYChi2 = 1.4;
  if(aAnType == kLamKchM) tStartXChi2 = -0.75;
  DrawChi2PerNDF(tStartXChi2, tStartYChi2, (TPad*)tPadReF0vsImF0, aFitInfoVec);
  //--------------------------------------------------------

  return tReturnCan;
}


//TODO For use in CompareAllReF0vsImF0AcrossAnalyses!!!
//---------------------------------------------------------------------------------------------------------------------------------
void DrawMultipleReF0vsImF0(TPad* aPadReF0vsImF0, TPad* aPadD0, 
                            AnalysisType aAnType, 
                            IncludeResType aIncludeResType=kInclude10ResAnd3Res, IncludeD0Type aIncludeD0Type=kFreeAndFixedD0,
                            IncludeRadiiType aIncludeRadiiType=kFreeAndFixedRadii, IncludeLambdaType aIncludeLambdaType=kFreeAndFixedLambda, 
                            ErrorType aErrType=kStatAndSys, double aD0XOffset=0.5, double aD0XOffsetIncrement=0.)
{
  //------------------------
  vector<FitInfo> aFitInfoVec = GetFitInfoVec(aAnType, aIncludeResType, aIncludeD0Type, aIncludeRadiiType, aIncludeLambdaType);
  //------------------------

  int iD0Inc=0;
  for(unsigned int i=0; i<aFitInfoVec.size(); i++)
  {
    DrawReF0vsImF0((TPad*)aPadReF0vsImF0, aFitInfoVec[i], aErrType);
    DrawD0((TPad*)aPadD0, aFitInfoVec[i], aErrType, aD0XOffset+iD0Inc*aD0XOffsetIncrement);

    if(aFitInfoVec[i].d0 != 0.) iD0Inc++;
  }
}


//---------------------------------------------------------------------------------------------------------------------------------
void DrawMultipleRadiusvsLambda(TPad* aPad, AnalysisType aAnType, CentralityType aCentType=k0010, 
                                IncludeResType aIncludeResType=kInclude10ResAnd3Res, IncludeD0Type aIncludeD0Type=kFreeAndFixedD0, 
                                IncludeRadiiType aIncludeRadiiType=kFreeAndFixedRadii, IncludeLambdaType aIncludeLambdaType=kFreeAndFixedLambda, 
                                ErrorType aErrType=kStatAndSys)
{
  aPad->cd();
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  SetupRadiusvsLambdaAxes(aPad);
  //------------------------
  vector<FitInfo> aFitInfoVec = GetFitInfoVec(aAnType, aIncludeResType, aIncludeD0Type, aIncludeRadiiType, aIncludeLambdaType);
  //------------------------
  for(unsigned int i=0; i<aFitInfoVec.size(); i++)
  {
    DrawRadiusvsLambda(aPad, aFitInfoVec[i], aCentType, aErrType);
  }

  //--------------------------------------------------------
  double tStartXChi2 = 0.50;
  double tStartYChi2 = 1.2;
  if(aAnType==kLamK0 || aCentType==k3050) tStartXChi2 = 4.5;
  DrawChi2PerNDF(tStartXChi2, tStartYChi2, aPad, aFitInfoVec);
  //--------------------------------------------------------
}

//---------------------------------------------------------------------------------------------------------------------------------
void SetStandardPlotAttributes(TGraphAsymmErrors* aGr, AnalysisType aAnType, ErrorType aErrType, int aMarkerStyle=20, double aMarkerSize=1.)
{
  assert(aErrType != kStatAndSys);

  Color_t tColor;
  if     (aAnType==kLamKchP || aAnType==kALamKchM) tColor = kRed+1;
  else if(aAnType==kLamKchM || aAnType==kALamKchP) tColor = kBlue+1;
  else if(aAnType==kLamK0 || aAnType==kALamK0)     tColor = kBlack;
  else assert(0);

  if(aErrType==kStat)
  {
    aGr->SetLineWidth(1);

  }
  else if(aErrType==kSys)
  {
    aGr->SetLineWidth(0);
    tColor = TColor::GetColorTransparent(tColor, 0.3);
  }

  aGr->SetMarkerColor(tColor);
  aGr->SetMarkerStyle(aMarkerStyle);
  aGr->SetMarkerSize(aMarkerSize);
  aGr->SetFillColor(tColor);
  aGr->SetFillStyle(1000);
  aGr->SetLineColor(tColor);
}




//---------------------------------------------------------------------------------------------------------------------------------
TCanvas* CompareAllRadiusvsLambdaAcrossAnalyses(CentralityType aCentType, IncludeResType aIncludeResType=kInclude10ResAnd3Res, IncludeD0Type aIncludeD0Type=kFreeAndFixedD0, Plot10and3Type aPlot10and3Type=kPlot10and3SeparateAndAvg, bool aIncludeFreeFixedD0Avgs=true, ErrorType aErrType=kStatAndSys, bool bSaveImage=false)
{
  vector<FitInfo> tIncludedPlots = GetFitInfoVec({kLamKchP, kLamKchM, kLamK0}, aIncludeResType, aIncludeD0Type, aIncludeRadiiType, aIncludeLambdaType);
  //----------------------------------------

  TString tCanBaseName =  "CompareAllRadiusvsLambdaAcrossAnalyses";
  TString tModifier = "";
  if(aIncludeD0Type==kFreeAndFixedD0 && aIncludeFreeFixedD0Avgs) tModifier = TString("_IncludeFreeFixedD0Avgs");

  TString tCanName = TString::Format("%s%s%s%s%s%s", tCanBaseName.Data(),
                                                     cCentralityTags[aCentType], 
                                                     cIncludeResTypeTags[aIncludeResType], cPlot10and3TypeTage[aPlot10and3Type],
                                                     cIncludeD0TypeTags[aIncludeD0Type], tModifier.Data());



  TCanvas* tReturnCan = new TCanvas(TString::Format("tCan%s", tCanName.Data()), 
                                    TString::Format("tCan%s", tCanName.Data()));
  tReturnCan->cd();
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  //------------------------------------------------------

  int tNPlots = tIncludedPlots.size();

  double tStartX = 5.8;
  double tStartY = 0.75;
  double tIncrementX = 0.14;
  double tIncrementY = 0.11;
  double tTextSize = 0.03;

  if(tNPlots>7) 
  {
    SetupRadiusvsLambdaAxes((TPad*)tReturnCan, 0., 10.);
    tStartX = 6.8;
    tStartY = 1.15;
    tIncrementX = 0.175;
  }
  else SetupRadiusvsLambdaAxes((TPad*)tReturnCan);

  TLatex* tTex = new TLatex();
  tTex->SetTextAlign(12);
  tTex->SetLineWidth(2);
  tTex->SetTextFont(42);
  tTex->SetTextSize(tTextSize);

  const Size_t tDescriptorMarkerSize=1.6;
  TMarker *tMarker = new TMarker();
  tMarker->SetMarkerSize(tDescriptorMarkerSize);
  tMarker->SetMarkerStyle(20);
  tMarker->SetMarkerColor(kBlack);

  //------------------------------------------------------
  AnalysisType tAnType;
  IncludeResType tIncResType;
  IncludeD0Type tIncD0Type;
  IncludeRadiiType tRadiiType;
  IncludeLambdaType tLambdaType;
  int tMarkerStyle;
  double tMarkerSize;
  TString tDescriptor;

  int iTex = 0;tIncludedPlots
  {
    tAnType = tIncludedPlots[i].analysisType;
    tIncResType = tIncludedPlots[i].resType;
    tIncD0Type = tIncludedPlots[i].d0Type;
    tRadiiType = tIncludedPlots[i].radiiType;
    tLambdaType = tIncludedPlots[i].lambdaType;
    tMarkerStyle = tIncludedPlots[i].markerStyle;
    tMarkerSize = tIncludedPlots[i].markerSize;
    tDescriptor = tIncludedPlots[i].descriptor;

    DrawMultipleRadiusvsLambda((TPad*)tReturnCan, tAnType, aCentType, tIncResType, tIncD0Type, tRadiiType, tLambdaType, aErrType);

    if(tDescriptor.Contains("d_{0}")) tTex->DrawLatex(tStartX, tStartY-iTex*tIncrementY-0.1*tIncrementY, tDescriptor);
    else tTex->DrawLatex(tStartX, tStartY-iTex*tIncrementY, tDescriptor);
    tMarker->SetMarkerStyle(tMarkerStyle);
    tMarker->DrawMarker(tStartX-tIncrementX, tStartY-iTex*tIncrementY);
    iTex++;

  }

  //------------------------------------------------------

  tTex->SetTextFont(42);
  tTex->SetTextSize(0.06);
  tTex->DrawLatex(0.5, 1.8, cPrettyCentralityTags[aCentType]);

  //------------------------------------------------------

  double tStartXStamp = 0.7;
  double tStartYStamp = 1.6;
  double tIncrementXStamp = 0.125;
  if(tNPlots>7) tIncrementXStamp = 0.15;
  double tIncrementYStamp = 0.125;
  double tTextSizeStamp = 0.04;
  int tMarkerStyleStamp = 21;
  DrawAnalysisStamps((TPad*)tReturnCan, tStartXStamp, tStartYStamp, tIncrementXStamp, tIncrementYStamp, tTextSizeStamp, tMarkerStyleStamp);

  //------------------------------------------------------

  if(bSaveImage)
  {
    gSystem->mkdir(gSaveLocationBase.Data());
    TString tSaveLocationFull = TString::Format("%s%s.%s", gSaveLocationBase.Data(), tCanName.Data(), gSaveType.Data());
    tReturnCan->SaveAs(tSaveLocationFull);
  }


  return tReturnCan;
}


//---------------------------------------------------------------------------------------------------------------------------------
void DrawScattParamPredictions(TPad* aPadReF0vsImF0, double aLegX1=0.825, double aLegY1=0.725, double aLegX2=0.975, double aLegY2=0.875)
{
  aPadReF0vsImF0->cd();

  TGraphAsymmErrors *tGr_0607100_Set1 = new TGraphAsymmErrors(1);
    tGr_0607100_Set1->SetPoint(0, 0.17, 0.34);
    tGr_0607100_Set1->SetPointError(0, 0.06, 0.06, 0.00, 0.00);
    tGr_0607100_Set1->SetMarkerStyle(39);
    tGr_0607100_Set1->SetMarkerSize(1.5);
    tGr_0607100_Set1->SetMarkerColor(kGreen+2);
    tGr_0607100_Set1->SetLineColor(kGreen+2);
    tGr_0607100_Set1->Draw("pzsame");

  TGraphAsymmErrors *tGr_0607100_Set2 = new TGraphAsymmErrors(1);
    tGr_0607100_Set2->SetPoint(0, 0.09, 0.34);
    tGr_0607100_Set2->SetPointError(0, 0.06, 0.06, 0.00, 0.00);
    tGr_0607100_Set2->SetMarkerStyle(37);
    tGr_0607100_Set2->SetMarkerSize(1.5);
    tGr_0607100_Set2->SetMarkerColor(kGreen+2);
    tGr_0607100_Set2->SetLineColor(kGreen+2);
    tGr_0607100_Set2->Draw("pzsame");

  //-----------

  TGraphAsymmErrors *tGr_PhysRevD_KLam = new TGraphAsymmErrors(1);
    tGr_PhysRevD_KLam->SetPoint(0, 0.19, 0.14);
    tGr_PhysRevD_KLam->SetPointError(0, 0.56, 0.55, 0.00, 0.00);
    tGr_PhysRevD_KLam->SetMarkerStyle(29);
    tGr_PhysRevD_KLam->SetMarkerSize(1.5);
    tGr_PhysRevD_KLam->SetMarkerColor(kOrange);
    tGr_PhysRevD_KLam->SetLineColor(kOrange);
    tGr_PhysRevD_KLam->Draw("pzsame");

  TGraphAsymmErrors *tGr_PhysRevD_AKLam = new TGraphAsymmErrors(1);
    tGr_PhysRevD_AKLam->SetPoint(0, 0.04, 0.18);
    tGr_PhysRevD_AKLam->SetPointError(0, 0.56, 0.55, 0.00, 0.00);
    tGr_PhysRevD_AKLam->SetMarkerStyle(30);
    tGr_PhysRevD_AKLam->SetMarkerSize(1.5);
    tGr_PhysRevD_AKLam->SetMarkerColor(kOrange);
    tGr_PhysRevD_AKLam->SetLineColor(kOrange);
    tGr_PhysRevD_AKLam->Draw("pzsame");

  TLegend* tLegPredictions = new TLegend(aLegX1, aLegY1, aLegX2, aLegY2);
    tLegPredictions->SetLineWidth(0);
    tLegPredictions->AddEntry(tGr_0607100_Set1, "[1] Set 1", "p");
    tLegPredictions->AddEntry(tGr_0607100_Set2, "[1] Set 2", "p");
    tLegPredictions->AddEntry(tGr_PhysRevD_KLam, "[2] K#Lambda", "p");
    tLegPredictions->AddEntry(tGr_PhysRevD_AKLam, "[2] #bar{K}#Lambda", "p");
  tLegPredictions->Draw();
}

//---------------------------------------------------------------------------------------------------------------------------------
TCanvas* CompareAllReF0vsImF0AcrossAnalyses(IncludeResType aIncludeResType=kInclude10ResAnd3Res, IncludeD0Type aIncludeD0Type=kFreeAndFixedD0, IncludeRadiiType aIncludeRadiiType=kFreeAndFixedRadii, IncludeLambdaType aIncludeLambdaType=kFreeAndFixedLambda, Plot10and3Type aPlot10and3Type=kPlot10and3SeparateAndAvg, ErrorType aErrType=kStatAndSys, bool aDrawFixedRadii=false, bool aDrawPredictions=false, bool bSaveImage=false)
{
  vector<FitInfo> tIncludedPlots = GetFitInfoVec({kLamKchP, kLamKchM, kLamK0}, aIncludeResType, aIncludeD0Type, aIncludeRadiiType, aIncludeLambdaType);

  //----------------------------------------
  TString tCanBaseName =  "CompareAllReF0vsImF0AcrossAnalyses";
  TString tModifier = "";
  if(aIncludeD0Type==kFreeAndFixedD0) tModifier = TString("_IncludeFreeFixedD0Avgs");

  TString tCanName = TString::Format("%s%s%s%s%s", tCanBaseName.Data(),
                                                   cIncludeResTypeTags[aIncludeResType], cPlot10and3TypeTage[aPlot10and3Type],
                                                   cIncludeD0TypeTags[aIncludeD0Type], tModifier.Data());


  TCanvas* tReturnCan = new TCanvas(TString::Format("tCan%s", tCanName.Data()), TString::Format("tCan%s", tCanName.Data()));
  tReturnCan->cd();

  TPad* tPadReF0vsImF0 = new TPad(TString::Format("tPadReF0vsImF0%s", tCanName.Data()), TString::Format("tPadReF0vsImF0%s", tCanName.Data()), 
                                  0.0, 0.0, 0.8, 1.0);
  tPadReF0vsImF0->SetRightMargin(0.01);
  tPadReF0vsImF0->Draw();

  TPad* tPadD0 = new TPad(TString::Format("tPadD0%s", tCanName.Data()), TString::Format("tPadD0%s", tCanName.Data()), 
                          0.8, 0.0, 1.0, 1.0);
  tPadD0->SetRightMargin(0.4);
  tPadD0->SetLeftMargin(0.);
  tPadD0->Draw();

  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  SetupReF0vsImF0AndD0Axes(tPadReF0vsImF0, tPadD0);

  //------------------------------------------------------
  double tStartX = -0.5;
  double tStartY = 1.4;
  double tIncrementX = 0.075;
  double tIncrementY = 0.10;
  double tTextSize = 0.04;

  int tNPlots = tIncludedPlots.size();
  if(tNPlots>7) tTextSize = 0.035;

  TLatex* tTex = new TLatex();
  tTex->SetTextAlign(12);
  tTex->SetLineWidth(2);
  tTex->SetTextFont(42);
  tTex->SetTextSize(tTextSize);

  const Size_t tDescriptorMarkerSize=1.6;
  TMarker *tMarker = new TMarker();
  tMarker->SetMarkerSize(tDescriptorMarkerSize);
  tMarker->SetMarkerStyle(20);
  tMarker->SetMarkerColor(kBlack);

  //------------------------------------------------------

  int tND0Increments = 0;
  for(unsigned int i=0; i<tIncludedPlots.size(); i++) if(tIncludedPlots[i].d0Type != kFixedD0Only) tND0Increments++;

  tND0Increments +=1;  //To give some room at left and right of plot
  double tIncrementSize = 1./tND0Increments;

  //------------------------------------------------------
  AnalysisType tAnType;
  IncludeResType tIncResType;
  IncludeD0Type tIncD0Type;
  IncludeRadiiType tRadiiType;
  IncludeLambdaType tLambdaType;
  int tMarkerStyle;
  double tMarkerSize;
  TString tDescriptor;

  bool tDrawFixedRadii = false;

  int iTex = 0;
  int iD0Inc = 0;
  for(unsigned int i=0; i<tIncludedPlots.size(); i++)
  {
    tAnType = tIncludedPlots[i].analysisType;
    tIncResType = tIncludedPlots[i].resType;
    tIncD0Type = tIncludedPlots[i].d0Type;
    tRadiiType = tIncludedPlots[i].radiiType;
    tLambdaType = tIncludedPlots[i].lambdaType;
    tMarkerStyle = tIncludedPlots[i].markerStyle;
    tMarkerSize = tIncludedPlots[i].markerSize;
    tDescriptor = tIncludedPlots[i].descriptor;

    if(tRadiiType == kFixedRadiiOnly) tDrawFixedRadii = true;

    DrawMultipleReF0vsImF0(tPadReF0vsImF0, tPadD0, tAnType, tIncResType, tIncD0Type, tRadiiType, tLambdaType, aErrType, (iD0Inc+1)*tIncrementSize);


    tPadReF0vsImF0->cd();
    if(tDescriptor.Contains("d_{0}")) tTex->DrawLatex(tStartX, tStartY-iTex*tIncrementY-0.1*tIncrementY, tDescriptor);
    else tTex->DrawLatex(tStartX, tStartY-iTex*tIncrementY, tDescriptor);
    tMarker->SetMarkerStyle(tMarkerStyle);
    tMarker->DrawMarker(tStartX-tIncrementX, tStartY-iTex*tIncrementY);
    iTex++;
    if(tIncludedPlots[i].d0Type != kFixedD0Only) iD0Inc++;

  }

  //------------------------------------------------------
  if(aDrawPredictions) DrawScattParamPredictions(tPadReF0vsImF0, 0.825, 0.725, 0.975, 0.875);
  //------------------------------------------------------

  double tStartXStamp = -1.75;
  double tStartYStamp = 1.35;
  double tIncrementXStamp = 0.05;
  double tIncrementYStamp = 0.10;
  double tTextSizeStamp = 0.04;
  int tMarkerStyleStamp = 21;
  DrawAnalysisStamps((TPad*)tPadReF0vsImF0, tStartXStamp, tStartYStamp, tIncrementXStamp, tIncrementYStamp, tTextSizeStamp, tMarkerStyleStamp);

  //------------------------------------------------------
  if(aDrawFixedRadii) DrawFixedRadiiStamps((TPad*)tPadReF0vsImF0, tStartXStamp+0.35, tStartYStamp, tIncrementXStamp, tIncrementYStamp, 0.75*tTextSizeStamp, tMarkerStyleStamp);
  //------------------------------------------------------

  if(bSaveImage)
  {
    gSystem->mkdir(gSaveLocationBase.Data());

    TString tModifier = "";
    if(tDrawFixedRadii) tModifier = TString("_wFixedRadiiResults");
    if(aDrawPredictions) tModifier += TString("_wScattLenPredictions");

    TString tSaveLocationFull = TString::Format("%s%s%s.%s", gSaveLocationBase.Data(), tCanName.Data(), tModifier.Data(), gSaveType.Data());
    tReturnCan->SaveAs(tSaveLocationFull);
  }

  return tReturnCan;
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



//---------------------------------------------------------------------------------------------------------------------------------
//*********************************************************************************************************************************
//---------------------------------------------------------------------------------------------------------------------------------

  bool bSaveFigures = false;
  CentralityType tCentType = kMB;

  IncludeResType tIncludeResType;
    //tIncludeResType = kInclude10ResAnd3Res;
    //tIncludeResType = kInclude10ResOnly;
    tIncludeResType = kInclude3ResOnly;

  IncludeD0Type tIncludeD0Type;
    //tIncludeD0Type = kFreeAndFixedD0;
    tIncludeD0Type = kFreeD0Only;
    //tIncludeD0Type = kFixedD0Only;

  Plot10and3Type tPlot10and3Type;
    //tPlot10and3Type=kPlot10and3SeparateAndAvg;
    tPlot10and3Type=kPlot10and3SeparateOnly;
    //tPlot10and3Type=kPlot10and3AvgOnly;

  bool tIncludeFreeFixedD0Avgs=false;
  bool tDrawFixedRadiiInCompareAllReF0vsImF0AcrossAnalyses = true;
  bool tDrawScattLenPredictions = true;

  //--------------------------------------------------------------------------

  // tIncludeRadiiType and tIncludeLambdaType only for single analysis methods
  //  i.e. only for DrawAll... methods
  //  For Compare...AcrossAnalyses methods, tIncludeRadiiType = kFreeRadiiOnly and tIncludeLambdaType = kFreeLambdaOnly
  IncludeRadiiType tIncludeRadiiType;
    tIncludeRadiiType = kFreeAndFixedRadii;
    //tIncludeRadiiType = kFreeRadiiOnly;
    //tIncludeRadiiType = kFixedRadiiOnly;

  IncludeLambdaType tIncludeLambdaType;
    tIncludeLambdaType = kFreeAndFixedLambda;
    //tIncludeLambdaType = kFreeLambdaOnly;
    //tIncludeLambdaType = kFixedLambdaOnly;

  ErrorType tErrorType;
    tErrorType = kStatAndSys;
    //tErrorType = kStat;
    //tErrorType = kSys;



//-------------------------------------------------------------------------------
/*
  TCanvas* tCanReF0vsImF0_LamKchP;
  tCanReF0vsImF0_LamKchP = DrawMultipleReF0vsImF0(kLamKchP, tIncludeResType, tIncludeD0Type, tIncludeRadiiType, tIncludeLambdaType, tErrorType, bSaveFigures);

  //-------------
  TCanvas *tCanRadiusvsLambda_LamKchP1, *tCanRadiusvsLambda_LamKchP2, *tCanRadiusvsLambda_LamKchP3;
  if(tCentType != kMB)
  {
    tCanRadiusvsLambda_LamKchP1 = DrawMultipleRadiusvsLambda(kLamKchP, tCentType, tIncludeResType, tIncludeD0Type, tIncludeRadiiType, tIncludeLambdaType, tErrorType, bSaveFigures);
  }
  else
  {
    tCanRadiusvsLambda_LamKchP1 = DrawMultipleRadiusvsLambda(kLamKchP, k0010, tIncludeResType, tIncludeD0Type, tIncludeRadiiType, tIncludeLambdaType, tErrorType, bSaveFigures);
    tCanRadiusvsLambda_LamKchP2 = DrawMultipleRadiusvsLambda(kLamKchP, k1030, tIncludeResType, tIncludeD0Type, tIncludeRadiiType, tIncludeLambdaType, tErrorType, bSaveFigures);
    tCanRadiusvsLambda_LamKchP3 = DrawMultipleRadiusvsLambda(kLamKchP, k3050, tIncludeResType, tIncludeD0Type, tIncludeRadiiType, tIncludeLambdaType, tErrorType, bSaveFigures);
  }

//-------------------------------------------------------------------------------
  TCanvas* tCanReF0vsImF0_LamKchM;
  tCanReF0vsImF0_LamKchM = DrawMultipleReF0vsImF0(kLamKchM, tIncludeResType, tIncludeD0Type, tIncludeRadiiType, tIncludeLambdaType, tErrorType, bSaveFigures);

  //-------------
  TCanvas *tCanRadiusvsLambda_LamKchM1, *tCanRadiusvsLambda_LamKchM2, *tCanRadiusvsLambda_LamKchM3;
  if(tCentType != kMB)
  {
    tCanRadiusvsLambda_LamKchM1 = DrawMultipleRadiusvsLambda(kLamKchM, tCentType, tIncludeResType, tIncludeD0Type, tIncludeRadiiType, tIncludeLambdaType, tErrorType, bSaveFigures);
  }
  else
  {
    tCanRadiusvsLambda_LamKchM1 = DrawMultipleRadiusvsLambda(kLamKchM, k0010, tIncludeResType, tIncludeD0Type, tIncludeRadiiType, tIncludeLambdaType, tErrorType, bSaveFigures);
    tCanRadiusvsLambda_LamKchM2 = DrawMultipleRadiusvsLambda(kLamKchM, k1030, tIncludeResType, tIncludeD0Type, tIncludeRadiiType, tIncludeLambdaType, tErrorType, bSaveFigures);
    tCanRadiusvsLambda_LamKchM3 = DrawMultipleRadiusvsLambda(kLamKchM, k3050, tIncludeResType, tIncludeD0Type, tIncludeRadiiType, tIncludeLambdaType, tErrorType, bSaveFigures);
  }

//-------------------------------------------------------------------------------
  TCanvas* tCanReF0vsImF0_LamK0;
  tCanReF0vsImF0_LamK0 = DrawMultipleReF0vsImF0(kLamK0, tIncludeResType, tIncludeD0Type, tIncludeRadiiType, tIncludeLambdaType, tErrorType, bSaveFigures);

  //-------------
  TCanvas *tCanRadiusvsLambda_LamK01, *tCanRadiusvsLambda_LamK02, *tCanRadiusvsLambda_LamK03;
  if(tCentType != kMB)
  {
    tCanRadiusvsLambda_LamK01 = DrawMultipleRadiusvsLambda(kLamK0, tCentType, tIncludeResType, tIncludeD0Type, tIncludeRadiiType, tIncludeLambdaType, tErrorType, bSaveFigures);
  }
  else
  {
    tCanRadiusvsLambda_LamK01 = DrawMultipleRadiusvsLambda(kLamK0, k0010, tIncludeResType, tIncludeD0Type, tIncludeRadiiType, tIncludeLambdaType, tErrorType, bSaveFigures);
    tCanRadiusvsLambda_LamK02 = DrawMultipleRadiusvsLambda(kLamK0, k1030, tIncludeResType, tIncludeD0Type, tIncludeRadiiType, tIncludeLambdaType, tErrorType, bSaveFigures);
    tCanRadiusvsLambda_LamK03 = DrawMultipleRadiusvsLambda(kLamK0, k3050, tIncludeResType, tIncludeD0Type, tIncludeRadiiType, tIncludeLambdaType, tErrorType, bSaveFigures);
  }
*/
//-------------------------------------------------------------------------------
//*******************************************************************************
//-------------------------------------------------------------------------------

  TCanvas *tCanCompareAllRadiusvsLambdaAcrossAnalyses1, *tCanCompareAllRadiusvsLambdaAcrossAnalyses2, *tCanCompareAllRadiusvsLambdaAcrossAnalyses3;
  if(tCentType != kMB)
  {
    tCanCompareAllRadiusvsLambdaAcrossAnalyses1 = CompareAllRadiusvsLambdaAcrossAnalyses(tCentType, tIncludeResType, tIncludeD0Type, tPlot10and3Type, tIncludeFreeFixedD0Avgs, tErrorType, bSaveFigures);
  }
  else
  {
    tCanCompareAllRadiusvsLambdaAcrossAnalyses1 = CompareAllRadiusvsLambdaAcrossAnalyses(k0010, tIncludeResType, tIncludeD0Type, tPlot10and3Type, tIncludeFreeFixedD0Avgs, tErrorType, bSaveFigures);
    tCanCompareAllRadiusvsLambdaAcrossAnalyses2 = CompareAllRadiusvsLambdaAcrossAnalyses(k1030, tIncludeResType, tIncludeD0Type, tPlot10and3Type, tIncludeFreeFixedD0Avgs, tErrorType, bSaveFigures);
    tCanCompareAllRadiusvsLambdaAcrossAnalyses3 = CompareAllRadiusvsLambdaAcrossAnalyses(k3050, tIncludeResType, tIncludeD0Type, tPlot10and3Type, tIncludeFreeFixedD0Avgs, tErrorType, bSaveFigures);
  }

  TCanvas* tCanCompareAllReF0vsImF0AcrossAnalyses = CompareAllReF0vsImF0AcrossAnalyses(tIncludeResType, tIncludeD0Type, tPlot10and3Type, tIncludeFreeFixedD0Avgs, tErrorType, 
                                                                                       tDrawFixedRadiiInCompareAllReF0vsImF0AcrossAnalyses, tDrawScattLenPredictions, bSaveFigures);


//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.
/*
  delete tCanReF0vsImF0_LamKchP;
  delete tCanRadiusvsLambda_LamKchP1;
  delete tCanRadiusvsLambda_LamKchP2;
  delete tCanRadiusvsLambda_LamKchP3;

  delete tCanReF0vsImF0_LamKchM;
  delete tCanRadiusvsLambda_LamKchM1;
  delete tCanRadiusvsLambda_LamKchM2;
  delete tCanRadiusvsLambda_LamKchM3;

  delete tCanReF0vsImF0_LamK0;
  delete tCanRadiusvsLambda_LamK01;
  delete tCanRadiusvsLambda_LamK02;
  delete tCanRadiusvsLambda_LamK03;
*/

  delete tCanCompareAllRadiusvsLambdaAcrossAnalyses1;
  delete tCanCompareAllRadiusvsLambdaAcrossAnalyses2;
  delete tCanCompareAllRadiusvsLambdaAcrossAnalyses3;

  delete tCanCompareAllReF0vsImF0AcrossAnalyses;

  cout << "DONE" << endl;
  return 0;
}








