#include "CompareFittingMethods.h"

//---------------------------------------------------------------------------------------------------------------------------------
void DrawChi2PerNDF(double aStartX, double aStartY, TPad* aPad, vector<FitInfo> &aFitInfoVec, bool bInclude10Res=true, bool bInclude3Res=true)
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
    if(!bInclude10Res && aFitInfoVec[i].all10ResidualsUsed) continue;
    if(!bInclude3Res && !aFitInfoVec[i].all10ResidualsUsed) continue;

    tText = TString::Format("#chi^{2}/NDF = %0.1f/%i = %0.3f", aFitInfoVec[i].chi2, aFitInfoVec[i].ndf, (aFitInfoVec[i].chi2/aFitInfoVec[i].ndf));
    tTex->DrawLatex(aStartX, aStartY-iTex*tIncrementY, tText);
    tMarker->SetMarkerStyle(aFitInfoVec[i].markerStyle);
    tMarker->SetMarkerColor(aFitInfoVec[i].markerColor);
    tMarker->DrawMarker(aStartX-0.10, aStartY-iTex*tIncrementY);
    iTex++;
  }

}

//---------------------------------------------------------------------------------------------------------------------------------
TGraphAsymmErrors* GetReF0vsImF0(FitInfo &aFitInfo)
{
  TGraphAsymmErrors* tReturnGr = new TGraphAsymmErrors(1);
  tReturnGr->SetPoint(0, aFitInfo.ref0, aFitInfo.imf0);
  tReturnGr->SetPointError(0, aFitInfo.ref0StatErr, aFitInfo.ref0StatErr, aFitInfo.imf0StatErr, aFitInfo.imf0StatErr);

  return tReturnGr;
}

//---------------------------------------------------------------------------------------------------------------------------------
void DrawReF0vsImF0(TPad* aPad, FitInfo &aFitInfo, TString aDrawOption="epsame")
{
  aPad->cd();
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  TGraphAsymmErrors* tGr = GetReF0vsImF0(aFitInfo);

  tGr->SetName(aFitInfo.descriptor);
  tGr->SetMarkerStyle(aFitInfo.markerStyle);
  tGr->SetMarkerColor(aFitInfo.markerColor);
  tGr->SetFillColor(aFitInfo.markerColor);
  tGr->SetFillStyle(1000);
  tGr->SetLineColor(aFitInfo.markerColor);
  tGr->SetLineWidth(1);

  tGr->Draw(aDrawOption);
}

//---------------------------------------------------------------------------------------------------------------------------------
TGraphAsymmErrors* GetWeightedMeanReF0vsImF0(vector<FitInfo> &aFitInfoVec, bool bInclude10Res=true, bool bInclude3Res=true)
{
  double tNumReF0 = 0., tDenReF0 = 0.;
  double tNumImF0 = 0., tDenImF0 = 0.;

  for(unsigned int i=0; i<aFitInfoVec.size(); i++)
  {
    if(!bInclude10Res && aFitInfoVec[i].all10ResidualsUsed) continue;
    if(!bInclude3Res && !aFitInfoVec[i].all10ResidualsUsed) continue;

    //Exclude fixed radius results from all average/mean calculations
    if(!aFitInfoVec[i].freeRadii) continue;

    tNumReF0 += aFitInfoVec[i].ref0/(aFitInfoVec[i].ref0StatErr*aFitInfoVec[i].ref0StatErr);
    tDenReF0 += 1./(aFitInfoVec[i].ref0StatErr*aFitInfoVec[i].ref0StatErr);

    tNumImF0 += aFitInfoVec[i].imf0/(aFitInfoVec[i].imf0StatErr*aFitInfoVec[i].imf0StatErr);
    tDenImF0 += 1./(aFitInfoVec[i].imf0StatErr*aFitInfoVec[i].imf0StatErr);
  }

  double tAvgReF0 = tNumReF0/tDenReF0;
  double tErrReF0 = sqrt(1./tDenReF0);

  double tAvgImF0 = tNumImF0/tDenImF0;
  double tErrImF0 = sqrt(1./tDenImF0);

  //--------------------------------------

  TGraphAsymmErrors* tReturnGr = new TGraphAsymmErrors(1);
  tReturnGr->SetPoint(0, tAvgReF0, tAvgImF0);
  tReturnGr->SetPointError(0, tErrReF0, tErrReF0, tErrImF0, tErrImF0);

  return tReturnGr;
}

//---------------------------------------------------------------------------------------------------------------------------------
void DrawWeightedMeanReF0vsImF0(TPad* aPad, vector<FitInfo> &aFitInfoVec, bool bInclude10Res=true, bool bInclude3Res=true, TString aDrawOption="epsame")
{
  aPad->cd();
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  TGraphAsymmErrors* tGr = GetWeightedMeanReF0vsImF0(aFitInfoVec, bInclude10Res, bInclude3Res);

  tGr->SetName("Weighted Mean Re[f0] vs Im[f0]");
  tGr->SetMarkerStyle(30);
  tGr->SetMarkerSize(2.0);
  tGr->SetMarkerColor(6);
  tGr->SetFillColor(6);
  tGr->SetFillStyle(1000);
  tGr->SetLineColor(6);
  tGr->SetLineWidth(1);

  tGr->Draw(aDrawOption);
}


//---------------------------------------------------------------------------------------------------------------------------------
TGraphAsymmErrors* GetAverageReF0vsImF0(vector<FitInfo> &aFitInfoVec, bool bInclude10Res=true, bool bInclude3Res=true)
{
  double tNumReF0 = 0., tDenReF0 = 0.;
  double tNumImF0 = 0., tDenImF0 = 0.;

  for(unsigned int i=0; i<aFitInfoVec.size(); i++)
  {
    if(!bInclude10Res && aFitInfoVec[i].all10ResidualsUsed) continue;
    if(!bInclude3Res && !aFitInfoVec[i].all10ResidualsUsed) continue;

    //Exclude fixed radius results from all average/mean calculations
    if(!aFitInfoVec[i].freeRadii) continue;

    tNumReF0 += aFitInfoVec[i].ref0;
    tDenReF0 += 1.;

    tNumImF0 += aFitInfoVec[i].imf0;
    tDenImF0 += 1.;
  }

  double tAvgReF0 = tNumReF0/tDenReF0;
  double tErrReF0 = 0.;

  double tAvgImF0 = tNumImF0/tDenImF0;
  double tErrImF0 = 0.;

  //--------------------------------------
  TGraphAsymmErrors* tReturnGr = new TGraphAsymmErrors(1);
  tReturnGr->SetPoint(0, tAvgReF0, tAvgImF0);
  tReturnGr->SetPointError(0, tErrReF0, tErrReF0, tErrImF0, tErrImF0);

  return tReturnGr;
}


//---------------------------------------------------------------------------------------------------------------------------------
void DrawAverageReF0vsImF0(TPad* aPad, vector<FitInfo> &aFitInfoVec, bool bInclude10Res=true, bool bInclude3Res=true, TString aDrawOption="epsame")
{
  aPad->cd();
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  TGraphAsymmErrors* tGr = GetAverageReF0vsImF0(aFitInfoVec, bInclude10Res, bInclude3Res);

  tGr->SetName("Average Re[f0] vs Im[f0]");
  tGr->SetMarkerStyle(29);
  tGr->SetMarkerSize(2.0);
  tGr->SetMarkerColor(6);
  tGr->SetFillColor(6);
  tGr->SetFillStyle(1000);
  tGr->SetLineColor(6);
  tGr->SetLineWidth(1);

  tGr->Draw(aDrawOption);
}

//---------------------------------------------------------------------------------------------------------------------------------
TCanvas* DrawAllReF0vsImF0(AnalysisType aAnType, bool bInclude10Res=true, bool bInclude3Res=true, bool bSaveImage=false, bool bDrawAverage=false, bool bDrawWeightedMean=false)
{
  TCanvas* tReturnCan = new TCanvas(TString::Format("tCanReF0vsImF0_%s", cAnalysisBaseTags[aAnType]), 
                                    TString::Format("tCanReF0vsImF0_%s", cAnalysisBaseTags[aAnType]));
  tReturnCan->cd();
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);
  //------------------------
  double aMinX = -2.;
  double aMaxX = 1.;

  double aMinY = 0.;
  double aMaxY = 1.5;
  //------------------------
  vector<FitInfo> aFitInfoVec;
  if     (aAnType==kLamKchP) aFitInfoVec = tFitInfoVec_LamKchP;
  else if(aAnType==kLamKchM) aFitInfoVec = tFitInfoVec_LamKchM;
  else if(aAnType==kLamK0) aFitInfoVec = tFitInfoVec_LamK0;
  else assert(0);

  //------------------------

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

  double tStartX = 0.30;
  double tStartY = 1.4;

  if(aAnType == kLamKchM) tStartX = -1.75;

  double tIncrementY = 0.08;

  const Size_t tMarkerSize=1.6;
  TMarker *tMarker = new TMarker();
  tMarker->SetMarkerSize(tMarkerSize);

  int iTex=0;
  TString tDescriptorFull, tDescriptor;
  int tDescriptorEnd = 0;
  for(unsigned int i=0; i<aFitInfoVec.size(); i++)
  {
    if(!bInclude10Res && aFitInfoVec[i].all10ResidualsUsed) continue;
    if(!bInclude3Res && !aFitInfoVec[i].all10ResidualsUsed) continue; 

    DrawReF0vsImF0((TPad*)tReturnCan, aFitInfoVec[i]);

    if(i%2==0)
    {
      if(iTex>2) continue;

      tDescriptorFull = aFitInfoVec[i].descriptor;
      tDescriptorEnd = tDescriptorFull.Index("_");
      if(tDescriptorEnd == -1) tDescriptorEnd = tDescriptorFull.Length();
      tDescriptor = tDescriptorFull(0, tDescriptorEnd);

      tTex->DrawLatex(tStartX, tStartY-iTex*tIncrementY, tDescriptor);
      tMarker->SetMarkerStyle(aFitInfoVec[i].markerStyle+1);
      tMarker->SetMarkerColor(aFitInfoVec[i].markerColor);
      tMarker->DrawMarker(tStartX-0.05, tStartY-iTex*tIncrementY);
      iTex++;
    }
  }

  //------------------------------------------------
  if(bDrawAverage)
  {
    DrawAverageReF0vsImF0((TPad*)tReturnCan, aFitInfoVec, bInclude10Res, bInclude3Res);
    tTex->DrawLatex(tStartX, tStartY-(iTex)*tIncrementY, "Average");
    tMarker->SetMarkerStyle(29);
    tMarker->SetMarkerColor(6);
    tMarker->DrawMarker(tStartX-0.05, tStartY-(iTex)*tIncrementY);
    iTex++;
  }
  //------------------------------------------------
  if(bDrawWeightedMean)
  {
    DrawWeightedMeanReF0vsImF0((TPad*)tReturnCan, aFitInfoVec, bInclude10Res, bInclude3Res);
    tTex->DrawLatex(tStartX, tStartY-(iTex)*tIncrementY, "Weighted Mean");
    tMarker->SetMarkerStyle(30);
    tMarker->SetMarkerColor(6);
    tMarker->DrawMarker(tStartX-0.05, tStartY-(iTex)*tIncrementY);
    iTex++;
  }
  //------------------------------------------------
  iTex++;
  tTex->DrawLatex(tStartX, tStartY-(iTex)*tIncrementY, "Free D0");
  tMarker->SetMarkerStyle(20);
  tMarker->SetMarkerColor(1);
  tMarker->DrawMarker(tStartX-0.05, tStartY-(iTex)*tIncrementY);
  iTex++;

  tTex->DrawLatex(tStartX, tStartY-(iTex)*tIncrementY, "Fixed D0");
  tMarker->SetMarkerStyle(22);
  tMarker->SetMarkerColor(1);
  tMarker->DrawMarker(tStartX-0.05, tStartY-(iTex)*tIncrementY);
  iTex+=2;
  //------------------------------------------------
  if(bInclude10Res && bInclude3Res)
  {
    tTex->DrawLatex(tStartX, tStartY-(iTex)*tIncrementY, "10 Res.");
    tMarker->SetMarkerStyle(21);
    tMarker->SetMarkerColor(1);
    tMarker->DrawMarker(tStartX-0.05, tStartY-(iTex)*tIncrementY);
    iTex++;

    tTex->DrawLatex(tStartX, tStartY-(iTex)*tIncrementY, "3 Res.");
    tMarker->SetMarkerStyle(25);
    tMarker->SetMarkerColor(1);
    tMarker->DrawMarker(tStartX-0.05, tStartY-(iTex)*tIncrementY);
    iTex++;
  }
  else if(bInclude10Res)
  {
    tTex->DrawLatex(tStartX, tStartY-(iTex)*tIncrementY, "10 Res.");
    tMarker->SetMarkerStyle(21);
    tMarker->SetMarkerColor(1);
    tMarker->DrawMarker(tStartX-0.05, tStartY-(iTex)*tIncrementY);
    iTex++;
  }
  else if(bInclude3Res)
  {
    tTex->DrawLatex(tStartX, tStartY-(iTex)*tIncrementY, "3 Res.");
    tMarker->SetMarkerStyle(25);
    tMarker->SetMarkerColor(1);
    tMarker->DrawMarker(tStartX-0.05, tStartY-(iTex)*tIncrementY);
    iTex++;
  }
  else assert(0);

  //--------------------------------------------------------
  double tStartXChi2 = -0.50;
  double tStartYChi2 = 1.4;
  if(aAnType == kLamKchM) tStartXChi2 = -0.75;
  DrawChi2PerNDF(tStartXChi2, tStartYChi2, (TPad*)tReturnCan, aFitInfoVec, bInclude10Res, bInclude3Res);
  //--------------------------------------------------------

  if(bSaveImage)
  {
    TString tSaveLocationBase = "/home/jesse/Analysis/Presentations/AliFemto/20171108/Figures/";
    TString tSaveLocationFull_ReF0vsImF0;

    tSaveLocationFull_ReF0vsImF0 = tSaveLocationBase + TString::Format("%s/ReF0vsImF0", cAnalysisBaseTags[aAnType]);
    if(bInclude10Res && bInclude3Res) tSaveLocationFull_ReF0vsImF0 += TString("_10ResAnd3Res");
    else if(bInclude10Res) tSaveLocationFull_ReF0vsImF0 += TString("_10Res");
    else if(bInclude3Res) tSaveLocationFull_ReF0vsImF0 += TString("_3Res");
    else assert(0);

    tSaveLocationFull_ReF0vsImF0 += TString(".eps");
    tReturnCan->SaveAs(tSaveLocationFull_ReF0vsImF0);
  }

  tTrash->Delete();
  return tReturnCan;
}

//---------------------------------------------------------------------------------------------------------------------------------
TGraphAsymmErrors* GetRadiusvsLambda(FitInfo &aFitInfo, CentralityType aCentType=k0010)
{
  TGraphAsymmErrors* tReturnGr = new TGraphAsymmErrors(1);
  tReturnGr->SetName(aFitInfo.descriptor);

  if(aCentType==k0010)
  {
    tReturnGr->SetPoint(0, aFitInfo.radius1, aFitInfo.lambda1);
    tReturnGr->SetPointError(0, aFitInfo.radiusStatErr1, aFitInfo.radiusStatErr1, aFitInfo.lambdaStatErr1, aFitInfo.lambdaStatErr1);
  }
  else if(aCentType==k1030)
  {
    tReturnGr->SetPoint(0, aFitInfo.radius2, aFitInfo.lambda2);
    tReturnGr->SetPointError(0, aFitInfo.radiusStatErr2, aFitInfo.radiusStatErr2, aFitInfo.lambdaStatErr2, aFitInfo.lambdaStatErr2);
  }
  else if(aCentType==k3050)
  {
    tReturnGr->SetPoint(0, aFitInfo.radius3, aFitInfo.lambda3);
    tReturnGr->SetPointError(0, aFitInfo.radiusStatErr3, aFitInfo.radiusStatErr3, aFitInfo.lambdaStatErr3, aFitInfo.lambdaStatErr3);
  }
  else assert(0);

  return tReturnGr;
}

//---------------------------------------------------------------------------------------------------------------------------------
void DrawRadiusvsLambda(TPad* aPad, FitInfo &aFitInfo, CentralityType aCentType=k0010, bool aDrawingAllCentralities=false, TString aDrawOption="epsame")
{
  aPad->cd();
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  TGraphAsymmErrors* tGr = GetRadiusvsLambda(aFitInfo, aCentType);

  if(aDrawingAllCentralities)
  {
    if(aCentType==k0010) tGr->SetMarkerStyle(aFitInfo.markerStyle);
    else if(aCentType==k1030) tGr->SetMarkerStyle(aFitInfo.markerStyle+1);
    else if(aCentType==k3050) tGr->SetMarkerStyle(aFitInfo.markerStyle+2);
    else assert(0);
  }
  else tGr->SetMarkerStyle(aFitInfo.markerStyle);

  tGr->SetMarkerColor(aFitInfo.markerColor);
  tGr->SetFillColor(aFitInfo.markerColor);
  tGr->SetFillStyle(1000);
  tGr->SetLineColor(aFitInfo.markerColor);
  tGr->SetLineWidth(1);

  tGr->Draw(aDrawOption);
}

//---------------------------------------------------------------------------------------------------------------------------------
TGraphAsymmErrors* GetWeightedMeanRadiusvsLambda(vector<FitInfo> &aFitInfoVec, CentralityType aCentType, bool bInclude10Res=true, bool bInclude3Res=true)
{
  TGraphAsymmErrors* tReturnGr = new TGraphAsymmErrors(1);
  if(aCentType==kMB) return tReturnGr;

  double tNumRadius = 0., tDenRadius = 0.;
  double tNumLambda = 0., tDenLambda = 0.;

  double tRadius=0., tRadiusErr=0.;
  double tLambda=0., tLambdaErr=0.;

  for(unsigned int i=0; i<aFitInfoVec.size(); i++)
  {
    if(!bInclude10Res && aFitInfoVec[i].all10ResidualsUsed) continue;
    if(!bInclude3Res && !aFitInfoVec[i].all10ResidualsUsed) continue;

    //Exclude fixed radius results from all average/mean calculations
    if(!aFitInfoVec[i].freeRadii) continue;

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

    tNumRadius += tRadius/(tRadiusErr*tRadiusErr);
    tDenRadius += 1./(tRadiusErr*tRadiusErr);

    if(aFitInfoVec[i].freeLambda)
    {
      tNumLambda += tLambda/(tLambdaErr*tLambdaErr);
      tDenLambda += 1./(tLambdaErr*tLambdaErr);
    }
  }

  double tAvgRadius = tNumRadius/tDenRadius;
  double tErrRadius = sqrt(1./tDenRadius);

  double tAvgLambda = tNumLambda/tDenLambda;
  double tErrLambda = sqrt(1./tDenLambda);
  //--------------------------------------

  tReturnGr->SetPoint(0, tAvgRadius, tAvgLambda);
  tReturnGr->SetPointError(0, tErrRadius, tErrRadius, tErrLambda, tErrLambda);

  return tReturnGr;
}

//---------------------------------------------------------------------------------------------------------------------------------
void DrawWeightedMeanRadiusvsLambda(TPad* aPad, vector<FitInfo> &aFitInfoVec, CentralityType aCentType, bool bInclude10Res=true, bool bInclude3Res=true, TString aDrawOption="epsame")
{
  if(aCentType==kMB) return;

  aPad->cd();
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  TGraphAsymmErrors* tGr = GetWeightedMeanRadiusvsLambda(aFitInfoVec, aCentType, bInclude10Res, bInclude3Res);

  tGr->SetName("Weighted Mean R vs #lambda");
  tGr->SetMarkerStyle(30);
  tGr->SetMarkerSize(2.0);
  tGr->SetMarkerColor(6);
  tGr->SetFillColor(6);
  tGr->SetFillStyle(1000);
  tGr->SetLineColor(6);
  tGr->SetLineWidth(1);

  tGr->Draw(aDrawOption);
}

//---------------------------------------------------------------------------------------------------------------------------------
TGraphAsymmErrors* GetAverageRadiusvsLambda(vector<FitInfo> &aFitInfoVec, CentralityType aCentType, bool bInclude10Res=true, bool bInclude3Res=true)
{
  TGraphAsymmErrors* tReturnGr = new TGraphAsymmErrors(1);
  if(aCentType==kMB) return tReturnGr;

  double tNumRadius = 0., tDenRadius = 0.;
  double tNumLambda = 0., tDenLambda = 0.;

  double tRadius=0., tRadiusErr=0.;
  double tLambda=0., tLambdaErr=0.;

  for(unsigned int i=0; i<aFitInfoVec.size(); i++)
  {
    if(!bInclude10Res && aFitInfoVec[i].all10ResidualsUsed) continue;
    if(!bInclude3Res && !aFitInfoVec[i].all10ResidualsUsed) continue;

    //Exclude fixed radius results from all average/mean calculations
    if(!aFitInfoVec[i].freeRadii) continue;

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

    tNumRadius += tRadius;
    tDenRadius += 1;

    if(aFitInfoVec[i].freeLambda)
    {
      tNumLambda += tLambda;
      tDenLambda += 1.;
    }
  }

  double tAvgRadius = tNumRadius/tDenRadius;
  double tErrRadius = 0.;

  double tAvgLambda = tNumLambda/tDenLambda;
  double tErrLambda = 0.;

  //--------------------------------------
  tReturnGr->SetPoint(0, tAvgRadius, tAvgLambda);
  tReturnGr->SetPointError(0, tErrRadius, tErrRadius, tErrLambda, tErrLambda);

  return tReturnGr;
}


//---------------------------------------------------------------------------------------------------------------------------------
void DrawAverageRadiusvsLambda(TPad* aPad, vector<FitInfo> &aFitInfoVec, CentralityType aCentType, bool bInclude10Res=true, bool bInclude3Res=true, TString aDrawOption="epsame")
{
  if(aCentType==kMB) return;

  aPad->cd();
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  TGraphAsymmErrors* tGr = GetAverageRadiusvsLambda(aFitInfoVec, aCentType, bInclude10Res, bInclude3Res);

  tGr->SetName("Average R vs #lambda");
  tGr->SetMarkerStyle(29);
  tGr->SetMarkerSize(2.0);
  tGr->SetMarkerColor(6);
  tGr->SetFillColor(6);
  tGr->SetFillStyle(1000);
  tGr->SetLineColor(6);
  tGr->SetLineWidth(1);

  tGr->Draw(aDrawOption);
}

//---------------------------------------------------------------------------------------------------------------------------------
TCanvas* DrawAllRadiusvsLambda(AnalysisType aAnType, CentralityType aCentType=k0010, bool bInclude10Res=true, bool bInclude3Res=true, bool bSaveImage=false, bool bDrawAverage=false, bool bDrawWeightedMean=false)
{
  TCanvas* tReturnCan = new TCanvas(TString::Format("tCanRadiusvsLambda_%s%s", cAnalysisBaseTags[aAnType], cCentralityTags[aCentType]), 
                                    TString::Format("tCanRadiusvsLambda_%s%s", cAnalysisBaseTags[aAnType], cCentralityTags[aCentType]));
  tReturnCan->cd();
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);
  //------------------------
  double aMinX = 0.;
  double aMaxX = 8.;

  double aMinY = 0.;
  double aMaxY = 2.;
  //------------------------
  vector<FitInfo> aFitInfoVec;
  if     (aAnType==kLamKchP) aFitInfoVec = tFitInfoVec_LamKchP;
  else if(aAnType==kLamKchM) aFitInfoVec = tFitInfoVec_LamKchM;
  else if(aAnType==kLamK0) aFitInfoVec = tFitInfoVec_LamK0;
  else assert(0);
  //------------------------


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


  double tStartX = 6.5;
  double tStartY = 1.0;

  double tIncrementY = 0.08;

  const Size_t tMarkerSize=1.2;
  TMarker *tMarker = new TMarker();
  tMarker->SetMarkerSize(tMarkerSize);

  int iTex=0;
  TString tDescriptorFull, tDescriptor;
  int tDescriptorEnd = 0;
  for(unsigned int i=0; i<aFitInfoVec.size(); i++)
  {
    if(!bInclude10Res && aFitInfoVec[i].all10ResidualsUsed) continue;
    if(!bInclude3Res && !aFitInfoVec[i].all10ResidualsUsed) continue;

    if(aCentType==kMB)
    {
      DrawRadiusvsLambda((TPad*)tReturnCan, aFitInfoVec[i], k0010, true);
      DrawRadiusvsLambda((TPad*)tReturnCan, aFitInfoVec[i], k1030, true);
      DrawRadiusvsLambda((TPad*)tReturnCan, aFitInfoVec[i], k3050, true);
    }
    else DrawRadiusvsLambda((TPad*)tReturnCan, aFitInfoVec[i], aCentType);

    if(i%2 == 0)
    {
      if(iTex>2) continue;

      tDescriptorFull = aFitInfoVec[i].descriptor;
      tDescriptorEnd = tDescriptorFull.Index("_");
      if(tDescriptorEnd == -1) tDescriptorEnd = tDescriptorFull.Length();
      tDescriptor = tDescriptorFull(0, tDescriptorEnd);

      tTex->DrawLatex(tStartX, tStartY-iTex*tIncrementY, tDescriptor);
      tMarker->SetMarkerStyle(aFitInfoVec[i].markerStyle+1);
      tMarker->SetMarkerColor(aFitInfoVec[i].markerColor);
      tMarker->DrawMarker(tStartX-0.10, tStartY-iTex*tIncrementY);
      iTex++;
    }
  }

  //------------------------------------------------
  if(bDrawAverage)
  {
    DrawAverageRadiusvsLambda((TPad*)tReturnCan, aFitInfoVec, aCentType, bInclude10Res, bInclude3Res);
    tTex->DrawLatex(tStartX, tStartY-(iTex)*tIncrementY, "Average");
    tMarker->SetMarkerStyle(29);
    tMarker->SetMarkerColor(6);
    tMarker->DrawMarker(tStartX-0.05, tStartY-(iTex)*tIncrementY);
    iTex++;
  }
  //------------------------------------------------
  if(bDrawWeightedMean)
  {
    DrawWeightedMeanRadiusvsLambda((TPad*)tReturnCan, aFitInfoVec, aCentType, bInclude10Res, bInclude3Res);
    tTex->DrawLatex(tStartX, tStartY-(iTex)*tIncrementY, "Weighted Mean");
    tMarker->SetMarkerStyle(30);
    tMarker->SetMarkerColor(6);
    tMarker->DrawMarker(tStartX-0.05, tStartY-(iTex)*tIncrementY);
    iTex++;
  }
  //------------------------------------------------
  iTex++;
  tTex->DrawLatex(tStartX, tStartY-(iTex)*tIncrementY, "Free D0");
  tMarker->SetMarkerStyle(20);
  tMarker->SetMarkerColor(1);
  tMarker->DrawMarker(tStartX-0.05, tStartY-(iTex)*tIncrementY);
  iTex++;

  tTex->DrawLatex(tStartX, tStartY-(iTex)*tIncrementY, "Fixed D0");
  tMarker->SetMarkerStyle(22);
  tMarker->SetMarkerColor(1);
  tMarker->DrawMarker(tStartX-0.05, tStartY-(iTex)*tIncrementY);
  iTex+=2;
  //------------------------------------------------
  if(bInclude10Res && bInclude3Res)
  {
    tTex->DrawLatex(tStartX, tStartY-(iTex)*tIncrementY, "10 Res.");
    tMarker->SetMarkerStyle(21);
    tMarker->SetMarkerColor(1);
    tMarker->DrawMarker(tStartX-0.05, tStartY-(iTex)*tIncrementY);
    iTex++;

    tTex->DrawLatex(tStartX, tStartY-(iTex)*tIncrementY, "3 Res.");
    tMarker->SetMarkerStyle(25);
    tMarker->SetMarkerColor(1);
    tMarker->DrawMarker(tStartX-0.05, tStartY-(iTex)*tIncrementY);
    iTex++;
  }
  else if(bInclude10Res)
  {
    tTex->DrawLatex(tStartX, tStartY-(iTex)*tIncrementY, "10 Res.");
    tMarker->SetMarkerStyle(21);
    tMarker->SetMarkerColor(1);
    tMarker->DrawMarker(tStartX-0.05, tStartY-(iTex)*tIncrementY);
    iTex++;
  }
  else if(bInclude3Res)
  {
    tTex->DrawLatex(tStartX, tStartY-(iTex)*tIncrementY, "3 Res.");
    tMarker->SetMarkerStyle(25);
    tMarker->SetMarkerColor(1);
    tMarker->DrawMarker(tStartX-0.05, tStartY-(iTex)*tIncrementY);
    iTex++;
  }
  else assert(0);

  //--------------------------------------------------------
  double tStartXChi2 = 0.50;
  double tStartYChi2 = 1.2;
  if(aAnType==kLamK0 || aCentType==k3050) tStartXChi2 = 4.5;
  DrawChi2PerNDF(tStartXChi2, tStartYChi2, (TPad*)tReturnCan, aFitInfoVec, bInclude10Res, bInclude3Res);
  //--------------------------------------------------------

  if(bSaveImage)
  {
    TString tSaveLocationBase = "/home/jesse/Analysis/Presentations/AliFemto/20171108/Figures/";
    TString tSaveLocationFull_RadiusvsLambda;

    tSaveLocationFull_RadiusvsLambda = tSaveLocationBase + TString::Format("%s/RadiusvsLambda%s", cAnalysisBaseTags[aAnType], cCentralityTags[aCentType]);
    if(bInclude10Res && bInclude3Res) tSaveLocationFull_RadiusvsLambda += TString("_10ResAnd3Res");
    else if(bInclude10Res) tSaveLocationFull_RadiusvsLambda += TString("_10Res");
    else if(bInclude3Res) tSaveLocationFull_RadiusvsLambda += TString("_3Res");
    else assert(0);

    tSaveLocationFull_RadiusvsLambda += TString(".eps");
    tReturnCan->SaveAs(tSaveLocationFull_RadiusvsLambda);
  }


  tTrash->Delete();
  return tReturnCan;
}

//---------------------------------------------------------------------------------------------------------------------------------
TCanvas* CompareRadiusvsLambdaAcrossAnalyses(CentralityType aCentType=k0010, bool aUseWeightedMean=false)
{
  TCanvas* tReturnCan = new TCanvas(TString::Format("tCanCompareRadiusvsLambdaAcrossAnalyses_%s", cCentralityTags[aCentType]), 
                                    TString::Format("tCanCompareRadiusvsLambdaAcrossAnalyses_%s", cCentralityTags[aCentType]));


  tReturnCan->cd();
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);
  //------------------------
  double aMinX = 0.;
  double aMaxX = 8.;

  double aMinY = 0.;
  double aMaxY = 2.;
  //------------------------

  TH1D* tTrash = new TH1D("tTrash", "tTrash", 10, aMinX, aMaxX);
  tTrash->GetXaxis()->SetRangeUser(aMinX, aMaxX);
  tTrash->GetYaxis()->SetRangeUser(aMinY, aMaxY);

  tTrash->GetXaxis()->SetTitle("Radius");
  tTrash->GetYaxis()->SetTitle("#lambda");

  tTrash->DrawCopy("axis");
  //------------------------
  vector<FitInfo> aFitInfoVec_LamKchP = tFitInfoVec_LamKchP;
  vector<FitInfo> aFitInfoVec_LamKchM = tFitInfoVec_LamKchM;
  vector<FitInfo> aFitInfoVec_LamK0 = tFitInfoVec_LamK0;

  //------------------------

  TGraphAsymmErrors *tGr_LamKchP, *tGr_LamKchM, *tGr_LamK0;
  if(aUseWeightedMean)
  {
    tGr_LamKchP = GetWeightedMeanRadiusvsLambda(aFitInfoVec_LamKchP, aCentType, true, true);
    tGr_LamKchM = GetWeightedMeanRadiusvsLambda(aFitInfoVec_LamKchM, aCentType, true, true);
    tGr_LamK0   = GetWeightedMeanRadiusvsLambda(aFitInfoVec_LamK0, aCentType, true, true);
  }
  else
  {
    tGr_LamKchP = GetRadiusvsLambda(aFitInfoVec_LamKchP[0], aCentType);
    tGr_LamKchM = GetRadiusvsLambda(aFitInfoVec_LamKchM[0], aCentType);
    tGr_LamK0   = GetRadiusvsLambda(aFitInfoVec_LamK0[0], aCentType);
  }


  tGr_LamKchP->SetMarkerColor(kRed);
  tGr_LamKchP->SetMarkerStyle(20);
  tGr_LamKchP->SetFillColor(kRed);
  tGr_LamKchP->SetFillStyle(1000);
  tGr_LamKchP->SetLineColor(kRed);
  tGr_LamKchP->SetLineWidth(1);


  tGr_LamKchM->SetMarkerColor(kBlue);
  tGr_LamKchM->SetMarkerStyle(20);
  tGr_LamKchM->SetFillColor(kBlue);
  tGr_LamKchM->SetFillStyle(1000);
  tGr_LamKchM->SetLineColor(kBlue);
  tGr_LamKchM->SetLineWidth(1);


  tGr_LamK0->SetMarkerColor(kBlack);
  tGr_LamK0->SetMarkerStyle(20);
  tGr_LamK0->SetFillColor(kBlack);
  tGr_LamK0->SetFillStyle(1000);
  tGr_LamK0->SetLineColor(kBlack);
  tGr_LamK0->SetLineWidth(1);
  //------------------------
  tGr_LamKchP->Draw("epsame");
  tGr_LamKchM->Draw("epsame");
  tGr_LamK0->Draw("epsame");


  return tReturnCan;
}




//---------------------------------------------------------------------------------------------------------------------------------
TCanvas* CompareReF0vsImF0AcrossAnalyses(bool aUseWeightedMean=false)
{
/*
  TCanvas* tReturnCan = new TCanvas(TString::Format("tCanCompareReF0vsImF0AcrossAnalyses_%s", cAnalysisBaseTags[aAnType]), 
                                    TString::Format("tCanCompareReF0vsImF0AcrossAnalyses_%s", cAnalysisBaseTags[aAnType]));
*/
  TCanvas* tReturnCan = new TCanvas("tCanCompareReF0vsImF0AcrossAnalyses_", "tCanCompareReF0vsImF0AcrossAnalyses_");

  tReturnCan->cd();
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);
  //------------------------
  double aMinX = -2.;
  double aMaxX = 1.;

  double aMinY = 0.;
  double aMaxY = 1.5;
  //------------------------

  TH1D* tTrash = new TH1D("tTrash", "tTrash", 10, aMinX, aMaxX);
  tTrash->GetXaxis()->SetRangeUser(aMinX, aMaxX);
  tTrash->GetYaxis()->SetRangeUser(aMinY, aMaxY);

  tTrash->GetXaxis()->SetTitle("Re[f0]");
  tTrash->GetYaxis()->SetTitle("Im[f0]");

  tTrash->DrawCopy("axis");
  //------------------------
  vector<FitInfo> aFitInfoVec_LamKchP = tFitInfoVec_LamKchP;
  vector<FitInfo> aFitInfoVec_LamKchM = tFitInfoVec_LamKchM;
  vector<FitInfo> aFitInfoVec_LamK0 = tFitInfoVec_LamK0;

  //------------------------

  TGraphAsymmErrors *tGr_LamKchP, *tGr_LamKchM, *tGr_LamK0;
  if(aUseWeightedMean)
  {
    tGr_LamKchP = GetWeightedMeanReF0vsImF0(aFitInfoVec_LamKchP, true, true);
    tGr_LamKchM = GetWeightedMeanReF0vsImF0(aFitInfoVec_LamKchM, true, true);
    tGr_LamK0   = GetWeightedMeanReF0vsImF0(aFitInfoVec_LamK0, true, true);
  }
  else
  {
    tGr_LamKchP = GetReF0vsImF0(aFitInfoVec_LamKchP[0]);
    tGr_LamKchM = GetReF0vsImF0(aFitInfoVec_LamKchM[0]);
    tGr_LamK0   = GetReF0vsImF0(aFitInfoVec_LamK0[0]);
  }

  tGr_LamKchP->SetMarkerColor(kRed);
  tGr_LamKchP->SetMarkerStyle(20);
  tGr_LamKchP->SetFillColor(kRed);
  tGr_LamKchP->SetFillStyle(1000);
  tGr_LamKchP->SetLineColor(kRed);
  tGr_LamKchP->SetLineWidth(1);


  tGr_LamKchM->SetMarkerColor(kBlue);
  tGr_LamKchM->SetMarkerStyle(20);
  tGr_LamKchM->SetFillColor(kBlue);
  tGr_LamKchM->SetFillStyle(1000);
  tGr_LamKchM->SetLineColor(kBlue);
  tGr_LamKchM->SetLineWidth(1);


  tGr_LamK0->SetMarkerColor(kBlack);
  tGr_LamK0->SetMarkerStyle(20);
  tGr_LamK0->SetFillColor(kBlack);
  tGr_LamK0->SetFillStyle(1000);
  tGr_LamK0->SetLineColor(kBlack);
  tGr_LamK0->SetLineWidth(1);
  //------------------------
  tGr_LamKchP->Draw("epsame");
  tGr_LamKchM->Draw("epsame");
  tGr_LamK0->Draw("epsame");


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

  bool bInclude10Res = true;
  bool bInclude3Res = true;
  CentralityType tCentType = kMB;

  bool bDrawAllCentralitiesOnSinglePlot = false;
  if(bDrawAllCentralitiesOnSinglePlot) tCentType = kMB;

  bool bDrawAverage = true;
  bool bDrawWeightedMean = true;

  bool bSaveFigures = false;

//-------------------------------------------------------------------------------
/*
  TCanvas* tCanReF0vsImF0_LamKchP;
  tCanReF0vsImF0_LamKchP = DrawAllReF0vsImF0(kLamKchP, bInclude10Res, bInclude3Res, bSaveFigures, bDrawAverage, bDrawWeightedMean);

  //-------------
  TCanvas *tCanRadiusvsLambda_LamKchP1, *tCanRadiusvsLambda_LamKchP2, *tCanRadiusvsLambda_LamKchP3;
  if(tCentType != kMB || bDrawAllCentralitiesOnSinglePlot)
  {
    tCanRadiusvsLambda_LamKchP1 = DrawAllRadiusvsLambda(kLamKchP, tCentType, bInclude10Res, bInclude3Res, bSaveFigures, bDrawAverage, bDrawWeightedMean);
  }
  else
  {
    tCanRadiusvsLambda_LamKchP1 = DrawAllRadiusvsLambda(kLamKchP, k0010, bInclude10Res, bInclude3Res, bSaveFigures, bDrawAverage, bDrawWeightedMean);
    tCanRadiusvsLambda_LamKchP2 = DrawAllRadiusvsLambda(kLamKchP, k1030, bInclude10Res, bInclude3Res, bSaveFigures, bDrawAverage, bDrawWeightedMean);
    tCanRadiusvsLambda_LamKchP3 = DrawAllRadiusvsLambda(kLamKchP, k3050, bInclude10Res, bInclude3Res, bSaveFigures, bDrawAverage, bDrawWeightedMean);
  }

//-------------------------------------------------------------------------------
  TCanvas* tCanReF0vsImF0_LamKchM;
  tCanReF0vsImF0_LamKchM = DrawAllReF0vsImF0(kLamKchM, bInclude10Res, bInclude3Res, bSaveFigures, bDrawAverage, bDrawWeightedMean);

  //-------------
  TCanvas *tCanRadiusvsLambda_LamKchM1, *tCanRadiusvsLambda_LamKchM2, *tCanRadiusvsLambda_LamKchM3;
  if(tCentType != kMB || bDrawAllCentralitiesOnSinglePlot)
  {
    tCanRadiusvsLambda_LamKchM1 = DrawAllRadiusvsLambda(kLamKchM, tCentType, bInclude10Res, bInclude3Res, bSaveFigures, bDrawAverage, bDrawWeightedMean);
  }
  else
  {
    tCanRadiusvsLambda_LamKchM1 = DrawAllRadiusvsLambda(kLamKchM, k0010, bInclude10Res, bInclude3Res, bSaveFigures, bDrawAverage, bDrawWeightedMean);
    tCanRadiusvsLambda_LamKchM2 = DrawAllRadiusvsLambda(kLamKchM, k1030, bInclude10Res, bInclude3Res, bSaveFigures, bDrawAverage, bDrawWeightedMean);
    tCanRadiusvsLambda_LamKchM3 = DrawAllRadiusvsLambda(kLamKchM, k3050, bInclude10Res, bInclude3Res, bSaveFigures, bDrawAverage, bDrawWeightedMean);
  }

//-------------------------------------------------------------------------------
  TCanvas* tCanReF0vsImF0_LamK0;
  tCanReF0vsImF0_LamK0 = DrawAllReF0vsImF0(kLamK0, bInclude10Res, bInclude3Res, bSaveFigures, bDrawAverage, bDrawWeightedMean);

  //-------------
  TCanvas *tCanRadiusvsLambda_LamK01, *tCanRadiusvsLambda_LamK02, *tCanRadiusvsLambda_LamK03;
  if(tCentType != kMB || bDrawAllCentralitiesOnSinglePlot)
  {
    tCanRadiusvsLambda_LamK01 = DrawAllRadiusvsLambda(kLamK0, tCentType, bInclude10Res, bInclude3Res, bSaveFigures, bDrawAverage, bDrawWeightedMean);
  }
  else
  {
    tCanRadiusvsLambda_LamK01 = DrawAllRadiusvsLambda(kLamK0, k0010, bInclude10Res, bInclude3Res, bSaveFigures, bDrawAverage, bDrawWeightedMean);
    tCanRadiusvsLambda_LamK02 = DrawAllRadiusvsLambda(kLamK0, k1030, bInclude10Res, bInclude3Res, bSaveFigures, bDrawAverage, bDrawWeightedMean);
    tCanRadiusvsLambda_LamK03 = DrawAllRadiusvsLambda(kLamK0, k3050, bInclude10Res, bInclude3Res, bSaveFigures, bDrawAverage, bDrawWeightedMean);
  }
*/
//-------------------------------------------------------------------------------
//*******************************************************************************
//-------------------------------------------------------------------------------
  TCanvas* tCanCompareRadiusvsLambdaAcrossAnalyses = CompareRadiusvsLambdaAcrossAnalyses(k0010, true);
  TCanvas* tCanCompareReF0vsImF0AcrossAnalyses = CompareReF0vsImF0AcrossAnalyses(true);

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
  delete tCanCompareRadiusvsLambdaAcrossAnalyses;
  delete tCanCompareReF0vsImF0AcrossAnalyses;

  cout << "DONE" << endl;
  return 0;
}








