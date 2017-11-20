#include "CompareFittingMethods.h"
TString gSaveLocationBase = "/home/jesse/Analysis/Presentations/GroupMeetings/20171123/Figures/";


//---------------------------------------------------------------------------------------------------------------------------------
void DrawChi2PerNDF(double aStartX, double aStartY, TPad* aPad, vector<FitInfo> &aFitInfoVec, IncludeResType aIncludeResType=kInclude10ResAnd3Res)
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
    if(aIncludeResType==kInclude10ResOnly && !aFitInfoVec[i].all10ResidualsUsed) continue;
    if(aIncludeResType==kInclude3ResOnly  && aFitInfoVec[i].all10ResidualsUsed) continue;

    tText = TString::Format("#chi^{2}/NDF = %0.1f/%i = %0.3f", aFitInfoVec[i].chi2, aFitInfoVec[i].ndf, (aFitInfoVec[i].chi2/aFitInfoVec[i].ndf));
    tTex->DrawLatex(aStartX, aStartY-iTex*tIncrementY, tText);
    tMarker->SetMarkerStyle(aFitInfoVec[i].markerStyle);
    tMarker->SetMarkerColor(aFitInfoVec[i].markerColor);
    tMarker->DrawMarker(aStartX-0.10, aStartY-iTex*tIncrementY);
    iTex++;
  }

}

//---------------------------------------------------------------------------------------------------------------------------------
TGraphAsymmErrors* GetReF0vsImF0(const FitInfo &aFitInfo)
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
TGraphAsymmErrors* GetWeightedMeanReF0vsImF0(vector<FitInfo> &aFitInfoVec, IncludeResType aIncludeResType=kInclude10ResAnd3Res)
{
  double tNumReF0 = 0., tDenReF0 = 0.;
  double tNumImF0 = 0., tDenImF0 = 0.;

  for(unsigned int i=0; i<aFitInfoVec.size(); i++)
  {
    if(aIncludeResType==kInclude10ResOnly && !aFitInfoVec[i].all10ResidualsUsed) continue;
    if(aIncludeResType==kInclude3ResOnly  && aFitInfoVec[i].all10ResidualsUsed) continue;

    //Exclude fixed radius results from all average/mean calculations
    if(!aFitInfoVec[i].freeRadii) continue;
    //Exclude fixed lambda results from all average/mean calculations
    if(!aFitInfoVec[i].freeLambda) continue;

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
TCanvas* DrawAllReF0vsImF0(AnalysisType aAnType, IncludeResType aIncludeResType=kInclude10ResAnd3Res, bool bSaveImage=false)
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
    if(aIncludeResType==kInclude10ResOnly && !aFitInfoVec[i].all10ResidualsUsed) continue;
    if(aIncludeResType==kInclude3ResOnly  && aFitInfoVec[i].all10ResidualsUsed) continue; 

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
  if(aIncludeResType==kInclude10ResAnd3Res)
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
  else if(aIncludeResType==kInclude10ResOnly)
  {
    tTex->DrawLatex(tStartX, tStartY-(iTex)*tIncrementY, "10 Res.");
    tMarker->SetMarkerStyle(21);
    tMarker->SetMarkerColor(1);
    tMarker->DrawMarker(tStartX-0.05, tStartY-(iTex)*tIncrementY);
    iTex++;
  }
  else if(aIncludeResType==kInclude3ResOnly)
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
  DrawChi2PerNDF(tStartXChi2, tStartYChi2, (TPad*)tReturnCan, aFitInfoVec, aIncludeResType);
  //--------------------------------------------------------

  if(bSaveImage)
  {
    TString tSaveLocationFull_ReF0vsImF0;

    tSaveLocationFull_ReF0vsImF0 = gSaveLocationBase + TString::Format("%s/ReF0vsImF0%s.eps", cAnalysisBaseTags[aAnType], cIncludeResTypeTags[aIncludeResType]);
    tReturnCan->SaveAs(tSaveLocationFull_ReF0vsImF0);
  }

  tTrash->Delete();
  return tReturnCan;
}

//---------------------------------------------------------------------------------------------------------------------------------
TGraphAsymmErrors* GetRadiusvsLambda(const FitInfo &aFitInfo, CentralityType aCentType=k0010)
{
  TGraphAsymmErrors* tReturnGr = new TGraphAsymmErrors(1);
  tReturnGr->SetName(aFitInfo.descriptor);

  assert(aCentType != kMB);
  tReturnGr->SetPoint(0, aFitInfo.radiusVec[aCentType], aFitInfo.lambdaVec[aCentType]);
  tReturnGr->SetPointError(0, aFitInfo.radiusStatErrVec[aCentType], aFitInfo.radiusStatErrVec[aCentType], 
                              aFitInfo.lambdaStatErrVec[aCentType], aFitInfo.lambdaStatErrVec[aCentType]);


  return tReturnGr;
}

//---------------------------------------------------------------------------------------------------------------------------------
void DrawRadiusvsLambda(TPad* aPad, FitInfo &aFitInfo, CentralityType aCentType=k0010, TString aDrawOption="epsame")
{
  aPad->cd();
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  TGraphAsymmErrors* tGr = GetRadiusvsLambda(aFitInfo, aCentType);

  tGr->SetMarkerStyle(aFitInfo.markerStyle);
  tGr->SetMarkerColor(aFitInfo.markerColor);
  tGr->SetFillColor(aFitInfo.markerColor);
  tGr->SetFillStyle(1000);
  tGr->SetLineColor(aFitInfo.markerColor);
  tGr->SetLineWidth(1);

  tGr->Draw(aDrawOption);
}

//---------------------------------------------------------------------------------------------------------------------------------
TGraphAsymmErrors* GetWeightedMeanRadiusvsLambda(vector<FitInfo> &aFitInfoVec, CentralityType aCentType, IncludeResType aIncludeResType=kInclude10ResAnd3Res)
{
  TGraphAsymmErrors* tReturnGr = new TGraphAsymmErrors(1);
  if(aCentType==kMB) return tReturnGr;

  double tNumRadius = 0., tDenRadius = 0.;
  double tNumLambda = 0., tDenLambda = 0.;

  double tRadius=0., tRadiusErr=0.;
  double tLambda=0., tLambdaErr=0.;

  for(unsigned int i=0; i<aFitInfoVec.size(); i++)
  {
    if(aIncludeResType==kInclude10ResOnly && !aFitInfoVec[i].all10ResidualsUsed) continue;
    if(aIncludeResType==kInclude3ResOnly  && aFitInfoVec[i].all10ResidualsUsed) continue;

    //Exclude fixed radius results from all average/mean calculations
    if(!aFitInfoVec[i].freeRadii) continue;
    //Exclude fixed lambda results from all average/mean calculations
    if(!aFitInfoVec[i].freeLambda) continue;

    tRadius = aFitInfoVec[i].radiusVec[aCentType];
    tRadiusErr = aFitInfoVec[i].radiusStatErrVec[aCentType];

    tLambda = aFitInfoVec[i].lambdaVec[aCentType];
    tLambdaErr = aFitInfoVec[i].lambdaStatErrVec[aCentType];


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
TCanvas* DrawAllRadiusvsLambda(AnalysisType aAnType, CentralityType aCentType=k0010, IncludeResType aIncludeResType=kInclude10ResAnd3Res, bool bSaveImage=false)
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
    if(aIncludeResType==kInclude10ResOnly && !aFitInfoVec[i].all10ResidualsUsed) continue;
    if(aIncludeResType==kInclude3ResOnly  && aFitInfoVec[i].all10ResidualsUsed) continue;

    DrawRadiusvsLambda((TPad*)tReturnCan, aFitInfoVec[i], aCentType);

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
  if(aIncludeResType==kInclude10ResAnd3Res)
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
  else if(aIncludeResType==kInclude10ResOnly)
  {
    tTex->DrawLatex(tStartX, tStartY-(iTex)*tIncrementY, "10 Res.");
    tMarker->SetMarkerStyle(21);
    tMarker->SetMarkerColor(1);
    tMarker->DrawMarker(tStartX-0.05, tStartY-(iTex)*tIncrementY);
    iTex++;
  }
  else if(aIncludeResType==kInclude3ResOnly)
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
  DrawChi2PerNDF(tStartXChi2, tStartYChi2, (TPad*)tReturnCan, aFitInfoVec, aIncludeResType);
  //--------------------------------------------------------

  if(bSaveImage)
  {
    TString tSaveLocationFull_RadiusvsLambda;

    tSaveLocationFull_RadiusvsLambda = gSaveLocationBase + TString::Format("%s/RadiusvsLambda%s%s.eps", cAnalysisBaseTags[aAnType], cCentralityTags[aCentType], cIncludeResTypeTags[aIncludeResType]);
    tReturnCan->SaveAs(tSaveLocationFull_RadiusvsLambda);
  }


  tTrash->Delete();
  return tReturnCan;
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
void DrawRadiusvsLambdaAcrossAnalyses(TPad* aPad, int aMarkerStyle=20, CentralityType aCentType=k0010, bool aUseWeightedMean=false, IncludeResType aIncludeResType=kInclude10ResAnd3Res)
{
  aPad->cd();
  //------------------------
  vector<FitInfo> aFitInfoVec_LamKchP = tFitInfoVec_LamKchP;
  vector<FitInfo> aFitInfoVec_LamKchM = tFitInfoVec_LamKchM;
  vector<FitInfo> aFitInfoVec_LamK0 = tFitInfoVec_LamK0;

  Color_t tColor_LamKchP = kRed+1;
  Color_t tColor_LamKchM = kBlue+1;
  Color_t tColor_LamK0 = kBlack;

  //------------------------

  TGraphAsymmErrors *tGr_LamKchP, *tGr_LamKchM, *tGr_LamK0;
  if(aUseWeightedMean)
  {
    tGr_LamKchP = GetWeightedMeanRadiusvsLambda(aFitInfoVec_LamKchP, aCentType, aIncludeResType);
    tGr_LamKchM = GetWeightedMeanRadiusvsLambda(aFitInfoVec_LamKchM, aCentType, aIncludeResType);
    tGr_LamK0   = GetWeightedMeanRadiusvsLambda(aFitInfoVec_LamK0, aCentType, aIncludeResType);
  }
  else
  {
    assert(!(aIncludeResType==kInclude10ResAnd3Res));
    if(aIncludeResType==kInclude10ResOnly)
    {
      tGr_LamKchP = GetRadiusvsLambda(aFitInfoVec_LamKchP[0], aCentType);
      tGr_LamKchM = GetRadiusvsLambda(aFitInfoVec_LamKchM[0], aCentType);
      tGr_LamK0   = GetRadiusvsLambda(aFitInfoVec_LamK0[0], aCentType);
    }
    else if(aIncludeResType==kInclude3ResOnly)
    {
      tGr_LamKchP = GetRadiusvsLambda(aFitInfoVec_LamKchP[6], aCentType);
      tGr_LamKchM = GetRadiusvsLambda(aFitInfoVec_LamKchM[6], aCentType);
      tGr_LamK0   = GetRadiusvsLambda(aFitInfoVec_LamK0[6], aCentType);
    }
    else assert(0);
  }


  tGr_LamKchP->SetMarkerColor(tColor_LamKchP);
  tGr_LamKchP->SetMarkerStyle(aMarkerStyle);
  tGr_LamKchP->SetFillColor(tColor_LamKchP);
  tGr_LamKchP->SetFillStyle(1000);
  tGr_LamKchP->SetLineColor(tColor_LamKchP);
  tGr_LamKchP->SetLineWidth(1);


  tGr_LamKchM->SetMarkerColor(tColor_LamKchM);
  tGr_LamKchM->SetMarkerStyle(aMarkerStyle);
  tGr_LamKchM->SetFillColor(tColor_LamKchM);
  tGr_LamKchM->SetFillStyle(1000);
  tGr_LamKchM->SetLineColor(tColor_LamKchM);
  tGr_LamKchM->SetLineWidth(1);


  tGr_LamK0->SetMarkerColor(tColor_LamK0);
  tGr_LamK0->SetMarkerStyle(aMarkerStyle);
  tGr_LamK0->SetFillColor(tColor_LamK0);
  tGr_LamK0->SetFillStyle(1000);
  tGr_LamK0->SetLineColor(tColor_LamK0);
  tGr_LamK0->SetLineWidth(1);
  //------------------------
  tGr_LamKchP->Draw("epsame");
  tGr_LamKchM->Draw("epsame");
  tGr_LamK0->Draw("epsame");

}



//---------------------------------------------------------------------------------------------------------------------------------
void DrawRadiusvsLambdaAcrossAnalysesQMResults(TPad* aPad, int aMarkerStyle=20, CentralityType aCentType=k0010)
{
  aPad->cd();
  //------------------------

  Color_t tColor_LamKchP = kRed+1;
  Color_t tColor_LamKchM = kBlue+1;
  Color_t tColor_LamK0 = kBlack;
  //------------------------

  TGraphAsymmErrors *tGr_LamKchP, *tGr_LamKchM, *tGr_LamK0;
  tGr_LamKchP = GetRadiusvsLambda(tFitInfoQM_LamKchP, aCentType);
  tGr_LamKchM = GetRadiusvsLambda(tFitInfoQM_LamKchM, aCentType);
  tGr_LamK0   = GetRadiusvsLambda(tFitInfoQM_LamK0, aCentType);
  //------------------------

  tGr_LamKchP->SetMarkerColor(tColor_LamKchP);
  tGr_LamKchP->SetMarkerStyle(aMarkerStyle);
  tGr_LamKchP->SetFillColor(tColor_LamKchP);
  tGr_LamKchP->SetFillStyle(1000);
  tGr_LamKchP->SetLineColor(tColor_LamKchP);
  tGr_LamKchP->SetLineWidth(1);


  tGr_LamKchM->SetMarkerColor(tColor_LamKchM);
  tGr_LamKchM->SetMarkerStyle(aMarkerStyle);
  tGr_LamKchM->SetFillColor(tColor_LamKchM);
  tGr_LamKchM->SetFillStyle(1000);
  tGr_LamKchM->SetLineColor(tColor_LamKchM);
  tGr_LamKchM->SetLineWidth(1);


  tGr_LamK0->SetMarkerColor(tColor_LamK0);
  tGr_LamK0->SetMarkerStyle(aMarkerStyle);
  tGr_LamK0->SetFillColor(tColor_LamK0);
  tGr_LamK0->SetFillStyle(1000);
  tGr_LamK0->SetLineColor(tColor_LamK0);
  tGr_LamK0->SetLineWidth(1);
  //------------------------
  tGr_LamKchP->Draw("epsame");
  tGr_LamKchM->Draw("epsame");
  tGr_LamK0->Draw("epsame");
}

//---------------------------------------------------------------------------------------------------------------------------------
TCanvas* CompareAllRadiusvsLambdaAcrossAnalyses(CentralityType aCentType=k0010, IncludeResType aIncludeResType=kInclude10ResAnd3Res, bool bSaveImage=false)
{
  TCanvas* tReturnCan = new TCanvas(TString::Format("tCanCompareAllRadiusvsLambdaAcrossAnalyses%s", cCentralityTags[aCentType]), 
                                    TString::Format("tCanCompareAllRadiusvsLambdaAcrossAnalyses%s", cCentralityTags[aCentType]));


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
  bool bInclude_QM = true;
  int tMarkerStyle_QM = 20;
  bool bIsMean_QM = false;
  IncludeResType tIncResType_QM = kIncludeNoRes;
  TString tDescriptor_QM = "QM 2017";

  //------------------------
  bool bInclude_10and3_WeightedMean = true;
  int tMarkerStyle_10and3_WeightedMean = 24;
  bool bIsMean_10and3_WeightedMean = true;
  IncludeResType tIncResType_10and3_WeightedMean = kInclude10ResAnd3Res;
  TString tDescriptor_10and3_WeightedMean = "10 & 3 Res., Avg.";

  //------------------------
  bool bInclude_10_WeightedMean = true;
  int tMarkerStyle_10_WeightedMean = 29;
  bool bIsMean_10_WeightedMean = true;
  IncludeResType tIncResType_10_WeightedMean = kInclude10ResOnly;
  TString tDescriptor_10_WeightedMean = "10 Res., Avg.";

  bool bInclude_3_WeightedMean = true;
  int tMarkerStyle_3_WeightedMean = 30;
  bool bIsMean_3_WeightedMean = true;
  IncludeResType tIncResType_3_WeightedMean = kInclude3ResOnly;
  TString tDescriptor_3_WeightedMean = "3 Res., Avg.";

  //------------------------
  bool bInclude_10_AllFree = true;
  int tMarkerStyle_10_AllFree = 33;
  bool bIsMean_10_AllFree = false;
  IncludeResType tIncResType_10_AllFree = kInclude10ResOnly;
  TString tDescriptor_10_AllFree = "10 Res., All Free";

  bool bInclude_3_AllFree = true;
  int tMarkerStyle_3_AllFree = 27;
  bool bIsMean_3_AllFree = false;
  IncludeResType tIncResType_3_AllFree = kInclude3ResOnly;
  TString tDescriptor_3_AllFree = "3 Res., All Free";

  //----------------------------------------
  if(aIncludeResType==kInclude10ResAnd3Res)
  {
    bInclude_QM = true;

    bInclude_10and3_WeightedMean = true;

    bInclude_10_WeightedMean = true;
    bInclude_3_WeightedMean = true;

    bInclude_10_AllFree = true;
    bInclude_3_AllFree = true;
  }
  else if(aIncludeResType==kInclude10ResOnly)
  {
    bInclude_QM = true;

    bInclude_10and3_WeightedMean = false;

    bInclude_10_WeightedMean = true;
    bInclude_3_WeightedMean = false;

    bInclude_10_AllFree = true;
    bInclude_3_AllFree = false;
  }
  else if(aIncludeResType==kInclude3ResOnly)
  {
    bInclude_QM = true;

    bInclude_10and3_WeightedMean = false;

    bInclude_10_WeightedMean = false;
    bInclude_3_WeightedMean = true;

    bInclude_10_AllFree = false;
    bInclude_3_AllFree = true;
  }
  else assert(0);

  //----------------------------------------
  vector<bool> tIsMeanVec{bIsMean_QM,
                          bIsMean_10and3_WeightedMean,
                          bIsMean_10_WeightedMean,     bIsMean_3_WeightedMean,
                          bIsMean_10_AllFree,          bIsMean_3_AllFree};

  vector<IncludeResType> tIncResTypes{tIncResType_QM,
                                      tIncResType_10and3_WeightedMean,
                                      tIncResType_10_WeightedMean,     tIncResType_3_WeightedMean,
                                      tIncResType_10_AllFree,          tIncResType_3_AllFree};


  vector<int> tMarkerStyles{tMarkerStyle_QM, 
                            tMarkerStyle_10and3_WeightedMean,
                            tMarkerStyle_10_WeightedMean,     tMarkerStyle_3_WeightedMean, 
                            tMarkerStyle_10_AllFree,          tMarkerStyle_3_AllFree};

  vector<int> tIncludePlots{bInclude_QM, 
                            bInclude_10and3_WeightedMean, 
                            bInclude_10_WeightedMean,    bInclude_3_WeightedMean,
                            bInclude_10_AllFree,         bInclude_3_AllFree};

  vector<TString> tDescriptors{tDescriptor_QM,
                               tDescriptor_10and3_WeightedMean,
                               tDescriptor_10_WeightedMean,     tDescriptor_3_WeightedMean, 
                               tDescriptor_10_AllFree,          tDescriptor_3_AllFree};

  assert(tIsMeanVec.size() == tIncResTypes.size());
  assert(tIsMeanVec.size() == tMarkerStyles.size());
  assert(tIsMeanVec.size() == tIncludePlots.size());
  assert(tIsMeanVec.size() == tDescriptors.size());

  //------------------------------------------------------
  double tStartX = 5.8;
  double tStartY = 0.7;
  double tIncrementX = 0.10;
  double tIncrementY = 0.10;
  double tTextSize = 0.04;

  TLatex* tTex = new TLatex();
  tTex->SetTextAlign(12);
  tTex->SetLineWidth(2);
  tTex->SetTextFont(42);
  tTex->SetTextSize(0.04);

  const Size_t tMarkerSize=1.6;
  TMarker *tMarker = new TMarker();
  tMarker->SetMarkerSize(tMarkerSize);
  tMarker->SetMarkerStyle(20);
  tMarker->SetMarkerColor(kBlack);

  int iTex = 0;

  for(unsigned int i=0; i<tIsMeanVec.size(); i++)
  {
    if(tIncludePlots[i])
    {
      if(tIncResTypes[i]==kIncludeNoRes) DrawRadiusvsLambdaAcrossAnalysesQMResults((TPad*)tReturnCan, tMarkerStyles[i], aCentType);
      else DrawRadiusvsLambdaAcrossAnalyses((TPad*)tReturnCan, tMarkerStyles[i], aCentType, tIsMeanVec[i], tIncResTypes[i]);

      tTex->DrawLatex(tStartX, tStartY-iTex*tIncrementY, tDescriptors[i]);
      tMarker->SetMarkerStyle(tMarkerStyles[i]);
      tMarker->DrawMarker(tStartX-tIncrementX, tStartY-iTex*tIncrementY);
      iTex++;
    }
  }

  //------------------------------------------------------

  tTex->SetTextFont(42);
  tTex->SetTextSize(0.06);
  tTex->DrawLatex(0.5, 1.8, cPrettyCentralityTags[aCentType]);

  //------------------------------------------------------

  double tStartXStamp = 0.5;
  double tStartYStamp = 1.6;
  double tIncrementXStamp = 0.10;
  double tIncrementYStamp = 0.10;
  double tTextSizeStamp = 0.04;
  int tMarkerStyleStamp = 21;
  DrawAnalysisStamps((TPad*)tReturnCan, tStartXStamp, tStartYStamp, tIncrementXStamp, tIncrementYStamp, tTextSizeStamp, tMarkerStyleStamp);

  //------------------------------------------------------


  if(bSaveImage)
  {
    TString tSaveLocationFull;
    tSaveLocationFull = gSaveLocationBase + TString::Format("AllRadiusvsLambdaAcrossAnalyses%s%s.eps", cCentralityTags[aCentType], cIncludeResTypeTags[aIncludeResType]);
    tReturnCan->SaveAs(tSaveLocationFull);
  }

  return tReturnCan;

}


//---------------------------------------------------------------------------------------------------------------------------------
void DrawReF0vsImF0AcrossAnalyses(TPad* aPad, int aMarkerStyle=20, bool aUseWeightedMean=false, IncludeResType aIncludeResType=kInclude10ResAnd3Res)
{
  aPad->cd();
  //------------------------
  vector<FitInfo> aFitInfoVec_LamKchP = tFitInfoVec_LamKchP;
  vector<FitInfo> aFitInfoVec_LamKchM = tFitInfoVec_LamKchM;
  vector<FitInfo> aFitInfoVec_LamK0 = tFitInfoVec_LamK0;

  Color_t tColor_LamKchP = kRed+1;
  Color_t tColor_LamKchM = kBlue+1;
  Color_t tColor_LamK0 = kBlack;

  //------------------------
  TGraphAsymmErrors *tGr_LamKchP, *tGr_LamKchM, *tGr_LamK0;
  if(aUseWeightedMean)
  {
    tGr_LamKchP = GetWeightedMeanReF0vsImF0(aFitInfoVec_LamKchP, aIncludeResType);
    tGr_LamKchM = GetWeightedMeanReF0vsImF0(aFitInfoVec_LamKchM, aIncludeResType);
    tGr_LamK0   = GetWeightedMeanReF0vsImF0(aFitInfoVec_LamK0, aIncludeResType);
  }
  else
  {
    assert(!(aIncludeResType==kInclude10ResAnd3Res));
    if(aIncludeResType==kInclude10ResOnly)
    {
      tGr_LamKchP = GetReF0vsImF0(aFitInfoVec_LamKchP[0]);
      tGr_LamKchM = GetReF0vsImF0(aFitInfoVec_LamKchM[0]);
      tGr_LamK0   = GetReF0vsImF0(aFitInfoVec_LamK0[0]);
    }
    else if(aIncludeResType==kInclude3ResOnly)
    {
      tGr_LamKchP = GetReF0vsImF0(aFitInfoVec_LamKchP[6]);
      tGr_LamKchM = GetReF0vsImF0(aFitInfoVec_LamKchM[6]);
      tGr_LamK0   = GetReF0vsImF0(aFitInfoVec_LamK0[6]);
    }
    else assert(0);
  }

  tGr_LamKchP->SetMarkerColor(tColor_LamKchP);
  tGr_LamKchP->SetMarkerStyle(aMarkerStyle);
  tGr_LamKchP->SetFillColor(tColor_LamKchP);
  tGr_LamKchP->SetFillStyle(1000);
  tGr_LamKchP->SetLineColor(tColor_LamKchP);
  tGr_LamKchP->SetLineWidth(1);


  tGr_LamKchM->SetMarkerColor(tColor_LamKchM);
  tGr_LamKchM->SetMarkerStyle(aMarkerStyle);
  tGr_LamKchM->SetFillColor(tColor_LamKchM);
  tGr_LamKchM->SetFillStyle(1000);
  tGr_LamKchM->SetLineColor(tColor_LamKchM);
  tGr_LamKchM->SetLineWidth(1);


  tGr_LamK0->SetMarkerColor(tColor_LamK0);
  tGr_LamK0->SetMarkerStyle(aMarkerStyle);
  tGr_LamK0->SetFillColor(tColor_LamK0);
  tGr_LamK0->SetFillStyle(1000);
  tGr_LamK0->SetLineColor(tColor_LamK0);
  tGr_LamK0->SetLineWidth(1);
  //------------------------
  tGr_LamKchP->Draw("epsame");
  tGr_LamKchM->Draw("epsame");
  tGr_LamK0->Draw("epsame");

  //------------------------------------------------------

}

//---------------------------------------------------------------------------------------------------------------------------------
void DrawReF0vsImF0AcrossAnalysesQMResults(TPad* aPad, int aMarkerStyle=20)
{
  aPad->cd();
  //------------------------
  Color_t tColor_LamKchP = kRed+1;
  Color_t tColor_LamKchM = kBlue+1;
  Color_t tColor_LamK0 = kBlack;

  //------------------------
  TGraphAsymmErrors *tGr_LamKchP, *tGr_LamKchM, *tGr_LamK0;
  tGr_LamKchP = GetReF0vsImF0(tFitInfoQM_LamKchP);
  tGr_LamKchM = GetReF0vsImF0(tFitInfoQM_LamKchM);
  tGr_LamK0   = GetReF0vsImF0(tFitInfoQM_LamK0);

  //------------------------
  tGr_LamKchP->SetMarkerColor(tColor_LamKchP);
  tGr_LamKchP->SetMarkerStyle(aMarkerStyle);
  tGr_LamKchP->SetFillColor(tColor_LamKchP);
  tGr_LamKchP->SetFillStyle(1000);
  tGr_LamKchP->SetLineColor(tColor_LamKchP);
  tGr_LamKchP->SetLineWidth(1);


  tGr_LamKchM->SetMarkerColor(tColor_LamKchM);
  tGr_LamKchM->SetMarkerStyle(aMarkerStyle);
  tGr_LamKchM->SetFillColor(tColor_LamKchM);
  tGr_LamKchM->SetFillStyle(1000);
  tGr_LamKchM->SetLineColor(tColor_LamKchM);
  tGr_LamKchM->SetLineWidth(1);


  tGr_LamK0->SetMarkerColor(tColor_LamK0);
  tGr_LamK0->SetMarkerStyle(aMarkerStyle);
  tGr_LamK0->SetFillColor(tColor_LamK0);
  tGr_LamK0->SetFillStyle(1000);
  tGr_LamK0->SetLineColor(tColor_LamK0);
  tGr_LamK0->SetLineWidth(1);
  //------------------------
  tGr_LamKchP->Draw("epsame");
  tGr_LamKchM->Draw("epsame");
  tGr_LamK0->Draw("epsame");

  //------------------------------------------------------

}

//---------------------------------------------------------------------------------------------------------------------------------
TCanvas* CompareAllReF0vsImF0AcrossAnalyses(IncludeResType aIncludeResType=kInclude10ResAnd3Res, bool bSaveImage=false)
{
/*
  TCanvas* tReturnCan = new TCanvas(TString::Format("tCanCompareAllReF0vsImF0AcrossAnalyses_%s", cAnalysisBaseTags[aAnType]), 
                                    TString::Format("tCanCompareAllReF0vsImF0AcrossAnalyses_%s", cAnalysisBaseTags[aAnType]));
*/
  TCanvas* tReturnCan = new TCanvas("tCanCompareAllReF0vsImF0AcrossAnalyses_", "tCanCompareAllReF0vsImF0AcrossAnalyses_");

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
  bool bInclude_QM = true;
  int tMarkerStyle_QM = 20;
  bool bIsMean_QM = false;
  IncludeResType tIncResType_QM = kIncludeNoRes;
  TString tDescriptor_QM = "QM 2017";

  //------------------------
  bool bInclude_10and3_WeightedMean = true;
  int tMarkerStyle_10and3_WeightedMean = 24;
  bool bIsMean_10and3_WeightedMean = true;
  IncludeResType tIncResType_10and3_WeightedMean = kInclude10ResAnd3Res;
  TString tDescriptor_10and3_WeightedMean = "10 & 3 Res., Avg.";

  //------------------------
  bool bInclude_10_WeightedMean = true;
  int tMarkerStyle_10_WeightedMean = 29;
  bool bIsMean_10_WeightedMean = true;
  IncludeResType tIncResType_10_WeightedMean = kInclude10ResOnly;
  TString tDescriptor_10_WeightedMean = "10 Res., Avg.";

  bool bInclude_3_WeightedMean = true;
  int tMarkerStyle_3_WeightedMean = 30;
  bool bIsMean_3_WeightedMean = true;
  IncludeResType tIncResType_3_WeightedMean = kInclude3ResOnly;
  TString tDescriptor_3_WeightedMean = "3 Res., Avg.";

  //------------------------
  bool bInclude_10_AllFree = true;
  int tMarkerStyle_10_AllFree = 33;
  bool bIsMean_10_AllFree = false;
  IncludeResType tIncResType_10_AllFree = kInclude10ResOnly;
  TString tDescriptor_10_AllFree = "10 Res., All Free";

  bool bInclude_3_AllFree = true;
  int tMarkerStyle_3_AllFree = 27;
  bool bIsMean_3_AllFree = false;
  IncludeResType tIncResType_3_AllFree = kInclude3ResOnly;
  TString tDescriptor_3_AllFree = "3 Res., All Free";

  //----------------------------------------
  if(aIncludeResType==kInclude10ResAnd3Res)
  {
    bInclude_QM = true;

    bInclude_10and3_WeightedMean = true;

    bInclude_10_WeightedMean = true;
    bInclude_3_WeightedMean = true;

    bInclude_10_AllFree = true;
    bInclude_3_AllFree = true;
  }
  else if(aIncludeResType==kInclude10ResOnly)
  {
    bInclude_QM = true;

    bInclude_10and3_WeightedMean = false;

    bInclude_10_WeightedMean = true;
    bInclude_3_WeightedMean = false;

    bInclude_10_AllFree = true;
    bInclude_3_AllFree = false;
  }
  else if(aIncludeResType==kInclude3ResOnly)
  {
    bInclude_QM = true;

    bInclude_10and3_WeightedMean = false;

    bInclude_10_WeightedMean = false;
    bInclude_3_WeightedMean = true;

    bInclude_10_AllFree = false;
    bInclude_3_AllFree = true;
  }
  else assert(0);

  //----------------------------------------
  vector<bool> tIsMeanVec{bIsMean_QM,
                          bIsMean_10and3_WeightedMean,
                          bIsMean_10_WeightedMean,     bIsMean_3_WeightedMean,
                          bIsMean_10_AllFree,          bIsMean_3_AllFree};

  vector<IncludeResType> tIncResTypes{tIncResType_QM,
                                      tIncResType_10and3_WeightedMean,
                                      tIncResType_10_WeightedMean,     tIncResType_3_WeightedMean,
                                      tIncResType_10_AllFree,          tIncResType_3_AllFree};


  vector<int> tMarkerStyles{tMarkerStyle_QM, 
                            tMarkerStyle_10and3_WeightedMean,
                            tMarkerStyle_10_WeightedMean,     tMarkerStyle_3_WeightedMean, 
                            tMarkerStyle_10_AllFree,          tMarkerStyle_3_AllFree};

  vector<int> tIncludePlots{bInclude_QM, 
                            bInclude_10and3_WeightedMean, 
                            bInclude_10_WeightedMean,    bInclude_3_WeightedMean,
                            bInclude_10_AllFree,         bInclude_3_AllFree};

  vector<TString> tDescriptors{tDescriptor_QM,
                               tDescriptor_10and3_WeightedMean,
                               tDescriptor_10_WeightedMean,     tDescriptor_3_WeightedMean, 
                               tDescriptor_10_AllFree,          tDescriptor_3_AllFree};

  assert(tIsMeanVec.size() == tIncResTypes.size());
  assert(tIsMeanVec.size() == tMarkerStyles.size());
  assert(tIsMeanVec.size() == tIncludePlots.size());
  assert(tIsMeanVec.size() == tDescriptors.size());

  //------------------------------------------------------
  double tStartX = 0.0;
  double tStartY = 1.4;
  double tIncrementX = 0.05;
  double tIncrementY = 0.08;
  double tTextSize = 0.04;

  TLatex* tTex = new TLatex();
  tTex->SetTextAlign(12);
  tTex->SetLineWidth(2);
  tTex->SetTextFont(42);
  tTex->SetTextSize(0.04);

  const Size_t tMarkerSize=1.6;
  TMarker *tMarker = new TMarker();
  tMarker->SetMarkerSize(tMarkerSize);
  tMarker->SetMarkerStyle(20);
  tMarker->SetMarkerColor(kBlack);

  int iTex = 0;

  for(unsigned int i=0; i<tIsMeanVec.size(); i++)
  {
    if(tIncludePlots[i])
    {
      if(tIncResTypes[i]==kIncludeNoRes) DrawReF0vsImF0AcrossAnalysesQMResults((TPad*)tReturnCan, tMarkerStyles[i]);
      else DrawReF0vsImF0AcrossAnalyses((TPad*)tReturnCan, tMarkerStyles[i], tIsMeanVec[i], tIncResTypes[i]);

      tTex->DrawLatex(tStartX, tStartY-iTex*tIncrementY, tDescriptors[i]);
      tMarker->SetMarkerStyle(tMarkerStyles[i]);
      tMarker->DrawMarker(tStartX-tIncrementX, tStartY-iTex*tIncrementY);
      iTex++;
    }
  }

  //------------------------------------------------------

  double tStartXStamp = -1.75;
  double tStartYStamp = 1.4;
  double tIncrementXStamp = 0.05;
  double tIncrementYStamp = 0.08;
  double tTextSizeStamp = 0.04;
  int tMarkerStyleStamp = 21;
  DrawAnalysisStamps((TPad*)tReturnCan, tStartXStamp, tStartYStamp, tIncrementXStamp, tIncrementYStamp, tTextSizeStamp, tMarkerStyleStamp);

  //------------------------------------------------------


  if(bSaveImage)
  {
    TString tSaveLocationFull;

//    tSaveLocationFull = gSaveLocationBase + TString::Format("%s/ReF0vsImF0", cAnalysisBaseTags[aAnType]);
    tSaveLocationFull = gSaveLocationBase + TString::Format("AllReF0vsImF0AcrossAnalyses%s.eps", cIncludeResTypeTags[aIncludeResType]);
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

  IncludeResType tIncludeResType;
    tIncludeResType = kInclude10ResAnd3Res;
//    tIncludeResType = kInclude10ResOnly;
//    tIncludeResType = kInclude3ResOnly;

  CentralityType tCentType = kMB;

  bool bSaveFigures = false;

//-------------------------------------------------------------------------------
/*
  TCanvas* tCanReF0vsImF0_LamKchP;
  tCanReF0vsImF0_LamKchP = DrawAllReF0vsImF0(kLamKchP, tIncludeResType, bSaveFigures);

  //-------------
  TCanvas *tCanRadiusvsLambda_LamKchP1, *tCanRadiusvsLambda_LamKchP2, *tCanRadiusvsLambda_LamKchP3;
  if(tCentType != kMB)
  {
    tCanRadiusvsLambda_LamKchP1 = DrawAllRadiusvsLambda(kLamKchP, tCentType, tIncludeResType, bSaveFigures);
  }
  else
  {
    tCanRadiusvsLambda_LamKchP1 = DrawAllRadiusvsLambda(kLamKchP, k0010, tIncludeResType, bSaveFigures);
    tCanRadiusvsLambda_LamKchP2 = DrawAllRadiusvsLambda(kLamKchP, k1030, tIncludeResType, bSaveFigures);
    tCanRadiusvsLambda_LamKchP3 = DrawAllRadiusvsLambda(kLamKchP, k3050, tIncludeResType, bSaveFigures);
  }

//-------------------------------------------------------------------------------
  TCanvas* tCanReF0vsImF0_LamKchM;
  tCanReF0vsImF0_LamKchM = DrawAllReF0vsImF0(kLamKchM, tIncludeResType, bSaveFigures);

  //-------------
  TCanvas *tCanRadiusvsLambda_LamKchM1, *tCanRadiusvsLambda_LamKchM2, *tCanRadiusvsLambda_LamKchM3;
  if(tCentType != kMB)
  {
    tCanRadiusvsLambda_LamKchM1 = DrawAllRadiusvsLambda(kLamKchM, tCentType, tIncludeResType, bSaveFigures);
  }
  else
  {
    tCanRadiusvsLambda_LamKchM1 = DrawAllRadiusvsLambda(kLamKchM, k0010, tIncludeResType, bSaveFigures);
    tCanRadiusvsLambda_LamKchM2 = DrawAllRadiusvsLambda(kLamKchM, k1030, tIncludeResType, bSaveFigures);
    tCanRadiusvsLambda_LamKchM3 = DrawAllRadiusvsLambda(kLamKchM, k3050, tIncludeResType, bSaveFigures);
  }

//-------------------------------------------------------------------------------
  TCanvas* tCanReF0vsImF0_LamK0;
  tCanReF0vsImF0_LamK0 = DrawAllReF0vsImF0(kLamK0, tIncludeResType, bSaveFigures);

  //-------------
  TCanvas *tCanRadiusvsLambda_LamK01, *tCanRadiusvsLambda_LamK02, *tCanRadiusvsLambda_LamK03;
  if(tCentType != kMB)
  {
    tCanRadiusvsLambda_LamK01 = DrawAllRadiusvsLambda(kLamK0, tCentType, tIncludeResType, bSaveFigures);
  }
  else
  {
    tCanRadiusvsLambda_LamK01 = DrawAllRadiusvsLambda(kLamK0, k0010, tIncludeResType, bSaveFigures);
    tCanRadiusvsLambda_LamK02 = DrawAllRadiusvsLambda(kLamK0, k1030, tIncludeResType, bSaveFigures);
    tCanRadiusvsLambda_LamK03 = DrawAllRadiusvsLambda(kLamK0, k3050, tIncludeResType, bSaveFigures);
  }
*/
//-------------------------------------------------------------------------------
//*******************************************************************************
//-------------------------------------------------------------------------------

  TCanvas *tCanCompareAllRadiusvsLambdaAcrossAnalyses1, *tCanCompareAllRadiusvsLambdaAcrossAnalyses2, *tCanCompareAllRadiusvsLambdaAcrossAnalyses3;
  if(tCentType != kMB)
  {
    tCanCompareAllRadiusvsLambdaAcrossAnalyses1 = CompareAllRadiusvsLambdaAcrossAnalyses(tCentType, tIncludeResType, bSaveFigures);
  }
  else
  {
    tCanCompareAllRadiusvsLambdaAcrossAnalyses1 = CompareAllRadiusvsLambdaAcrossAnalyses(k0010, tIncludeResType, bSaveFigures);
    tCanCompareAllRadiusvsLambdaAcrossAnalyses2 = CompareAllRadiusvsLambdaAcrossAnalyses(k1030, tIncludeResType, bSaveFigures);
    tCanCompareAllRadiusvsLambdaAcrossAnalyses3 = CompareAllRadiusvsLambdaAcrossAnalyses(k3050, tIncludeResType, bSaveFigures);
  }


  TCanvas* tCanCompareAllReF0vsImF0AcrossAnalyses = CompareAllReF0vsImF0AcrossAnalyses(tIncludeResType, bSaveFigures);

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








