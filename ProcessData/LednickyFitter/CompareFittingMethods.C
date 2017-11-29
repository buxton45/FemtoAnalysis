#include "CompareFittingMethods.h"
TString gSaveLocationBase = "/home/jesse/Analysis/Presentations/GroupMeetings/20171123/Figures/";


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
/*
//---------------------------------------------------------------------------------------------------------------------------------
vector<FitInfo> GetFitInfoVec(AnalysisType aAnType, IncludeResType aIncludeResType=kInclude10ResAnd3Res)
{
  vector<FitInfo> aFitInfoVec;
  if     (aAnType==kLamKchP) aFitInfoVec = tFitInfoVec_LamKchP;
  else if(aAnType==kLamKchM) aFitInfoVec = tFitInfoVec_LamKchM;
  else if(aAnType==kLamK0) aFitInfoVec = tFitInfoVec_LamK0;
  else assert(0);
  //------------------------------
  vector<FitInfo> tReturnVec;

  if(aIncludeResType==kInclude10ResAnd3Res) tReturnVec = aFitInfoVec;
  else if(aIncludeResType==kInclude10ResOnly)
  {
    tReturnVec.push_back(aFitInfoVec[0]);
    tReturnVec.push_back(aFitInfoVec[1]);
    tReturnVec.push_back(aFitInfoVec[2]);
    tReturnVec.push_back(aFitInfoVec[3]);
    tReturnVec.push_back(aFitInfoVec[4]);
    tReturnVec.push_back(aFitInfoVec[5]);
  }
  else if(aIncludeResType==kInclude3ResOnly)
  {
    tReturnVec.push_back(aFitInfoVec[6]);
    tReturnVec.push_back(aFitInfoVec[7]);
    tReturnVec.push_back(aFitInfoVec[8]);
    tReturnVec.push_back(aFitInfoVec[9]);
    tReturnVec.push_back(aFitInfoVec[10]);
    tReturnVec.push_back(aFitInfoVec[11]);
  }
  else assert(0);

  return tReturnVec;
}
*/


//---------------------------------------------------------------------------------------------------------------------------------
vector<FitInfo> GetFitInfoVec(AnalysisType aAnType, IncludeResType aIncludeResType=kInclude10ResAnd3Res, IncludeD0Type aIncludeD0Type=kFreeAndFixedD0)
{
  vector<FitInfo> aFitInfoVec;
  if     (aAnType==kLamKchP) aFitInfoVec = tFitInfoVec_LamKchP;
  else if(aAnType==kLamKchM) aFitInfoVec = tFitInfoVec_LamKchM;
  else if(aAnType==kLamK0) aFitInfoVec = tFitInfoVec_LamK0;
  else assert(0);
  //------------------------------
  vector<FitInfo> tReturnVec;
  //------------------------------

  bool bPassRes=false, bPassD0=false;
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

    if(bPassRes && bPassD0) tReturnVec.push_back(aFitInfoVec[i]);
  }

  return tReturnVec;
}


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
TGraphAsymmErrors* GetReF0vsImF0(const FitInfo &aFitInfo)
{
  TGraphAsymmErrors* tReturnGr = new TGraphAsymmErrors(1);
  tReturnGr->SetPoint(0, aFitInfo.ref0, aFitInfo.imf0);
  tReturnGr->SetPointError(0, aFitInfo.ref0StatErr, aFitInfo.ref0StatErr, aFitInfo.imf0StatErr, aFitInfo.imf0StatErr);

  return tReturnGr;
}

//---------------------------------------------------------------------------------------------------------------------------------
TGraphAsymmErrors* GetD0(const FitInfo &aFitInfo)
{
  TGraphAsymmErrors* tReturnGr = new TGraphAsymmErrors(1);
  tReturnGr->SetPoint(0, 0.5, aFitInfo.d0);
  tReturnGr->SetPointError(0, 0., 0., aFitInfo.d0StatErr, aFitInfo.d0StatErr);

  return tReturnGr;
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
void DrawD0(TPad* aPad, FitInfo &aFitInfo, TString aDrawOption="epsame")
{
  aPad->cd();
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  TGraphAsymmErrors* tGr = GetD0(aFitInfo);

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
TGraphAsymmErrors* GetWeightedMeanReF0vsImF0(vector<FitInfo> &aFitInfoVec, IncludeResType aIncludeResType=kInclude10ResAnd3Res, IncludeD0Type aIncludeD0Type=kFreeAndFixedD0)
{
  td2dVec tReF0Vec;
  td2dVec tImF0Vec;

  for(unsigned int i=0; i<aFitInfoVec.size(); i++)
  {
    if(aIncludeResType==kInclude10ResOnly && !aFitInfoVec[i].all10ResidualsUsed) continue;
    if(aIncludeResType==kInclude3ResOnly  && aFitInfoVec[i].all10ResidualsUsed) continue;

    if(aIncludeD0Type==kFreeD0Only && !aFitInfoVec[i].freeD0) continue;
    if(aIncludeD0Type==kFixedD0Only && aFitInfoVec[i].freeD0) continue;

    //Exclude fixed radius results from all average/mean calculations
    if(!aFitInfoVec[i].freeRadii) continue;
    //Exclude fixed lambda results from all average/mean calculations
    if(!aFitInfoVec[i].freeLambda) continue;

    tReF0Vec.push_back(vector<double>{aFitInfoVec[i].ref0, aFitInfoVec[i].ref0StatErr, aFitInfoVec[i].ref0SysErr});
    tImF0Vec.push_back(vector<double>{aFitInfoVec[i].imf0, aFitInfoVec[i].imf0StatErr, aFitInfoVec[i].imf0SysErr});
  }

  td1dVec tMeanInfo_ReF0 = GetMean(tReF0Vec);
  td1dVec tMeanInfo_ImF0 = GetMean(tImF0Vec);

  //--------------------------------------

  double tMean_ReF0 =        tMeanInfo_ReF0[0];
  double tMeanStatErr_ReF0 = tMeanInfo_ReF0[1];
//  double tMeanSysErr_ReF0 =  tMeanInfo_ReF0[2];  //Currently not used

  double tMean_ImF0 =        tMeanInfo_ImF0[0];
  double tMeanStatErr_ImF0 = tMeanInfo_ImF0[1];
//  double tMeanSysErr_ImF0 =  tMeanInfo_ImF0[2];  //Currently not used

  //--------------------------------------

  TGraphAsymmErrors* tReturnGr = new TGraphAsymmErrors(1);
  tReturnGr->SetPoint(0, tMean_ReF0, tMean_ImF0);
  tReturnGr->SetPointError(0, tMeanStatErr_ReF0, tMeanStatErr_ReF0, tMeanStatErr_ImF0, tMeanStatErr_ImF0);

  return tReturnGr;
}

//---------------------------------------------------------------------------------------------------------------------------------
TGraphAsymmErrors* GetWeightedMeanD0(vector<FitInfo> &aFitInfoVec, IncludeResType aIncludeResType=kInclude10ResAnd3Res, IncludeD0Type aIncludeD0Type=kFreeAndFixedD0)
{
  td2dVec tD0Vec;
  for(unsigned int i=0; i<aFitInfoVec.size(); i++) 
  {
    if(aIncludeResType==kInclude10ResOnly && !aFitInfoVec[i].all10ResidualsUsed) continue;
    if(aIncludeResType==kInclude3ResOnly  && aFitInfoVec[i].all10ResidualsUsed) continue;

    if(aIncludeD0Type==kFreeD0Only && !aFitInfoVec[i].freeD0) continue;
    if(aIncludeD0Type==kFixedD0Only && aFitInfoVec[i].freeD0) continue;

    //Exclude fixed radius results from all average/mean calculations
    if(!aFitInfoVec[i].freeRadii) continue;
    //Exclude fixed lambda results from all average/mean calculations
    if(!aFitInfoVec[i].freeLambda) continue;

    tD0Vec.push_back(vector<double>{aFitInfoVec[i].d0, aFitInfoVec[i].d0StatErr, aFitInfoVec[i].d0SysErr});
  }
  td1dVec tMeanInfo_D0 = GetMean(tD0Vec);

  //--------------------------------------

  double tMean_D0 =        tMeanInfo_D0[0];
  double tMeanStatErr_D0 = tMeanInfo_D0[1];
//  double tMeanSysErr_D0 =  tMeanInfo_D0[2];  //Currently not used

  //--------------------------------------

  TGraphAsymmErrors* tReturnGr = new TGraphAsymmErrors(1);
  tReturnGr->SetPoint(0, 0.5, tMean_D0);
  tReturnGr->SetPointError(0, 0., 0., tMeanStatErr_D0, tMeanStatErr_D0);

  return tReturnGr;
}


//---------------------------------------------------------------------------------------------------------------------------------
TGraphAsymmErrors* GetWeightedMeanRadiusvsLambda(vector<FitInfo> &aFitInfoVec, CentralityType aCentType, IncludeResType aIncludeResType=kInclude10ResAnd3Res, IncludeD0Type aIncludeD0Type=kFreeAndFixedD0)
{
  TGraphAsymmErrors* tReturnGr = new TGraphAsymmErrors(1);
  if(aCentType==kMB) return tReturnGr;

  td2dVec tRadiusVec;
  td2dVec tLambdaVec;

  for(unsigned int i=0; i<aFitInfoVec.size(); i++)
  {
    if(aIncludeResType==kInclude10ResOnly && !aFitInfoVec[i].all10ResidualsUsed) continue;
    if(aIncludeResType==kInclude3ResOnly  && aFitInfoVec[i].all10ResidualsUsed) continue;

    if(aIncludeD0Type==kFreeD0Only && !aFitInfoVec[i].freeD0) continue;
    if(aIncludeD0Type==kFixedD0Only && aFitInfoVec[i].freeD0) continue;

    //Exclude fixed radius results from all average/mean calculations
    if(!aFitInfoVec[i].freeRadii) continue;
    //Exclude fixed lambda results from all average/mean calculations
    if(!aFitInfoVec[i].freeLambda) continue;

    tRadiusVec.push_back(vector<double>{aFitInfoVec[i].radiusVec[aCentType], aFitInfoVec[i].radiusStatErrVec[aCentType], aFitInfoVec[i].radiusSysErrVec[aCentType]});
    tLambdaVec.push_back(vector<double>{aFitInfoVec[i].lambdaVec[aCentType], aFitInfoVec[i].lambdaStatErrVec[aCentType], aFitInfoVec[i].lambdaSysErrVec[aCentType]});
  }

  td1dVec tMeanInfo_Radius = GetMean(tRadiusVec);
  td1dVec tMeanInfo_Lambda = GetMean(tLambdaVec);

  //--------------------------------------

  double tMean_Radius =        tMeanInfo_Radius[0];
  double tMeanStatErr_Radius = tMeanInfo_Radius[1];
//  double tMeanSysErr_Radius =  tMeanInfo_Radius[2];  //Currently not used

  double tMean_Lambda =        tMeanInfo_Lambda[0];
  double tMeanStatErr_Lambda = tMeanInfo_Lambda[1];
//  double tMeanSysErr_Lambda =  tMeanInfo_Lambda[2];  //Currently not used

  //--------------------------------------

  tReturnGr->SetPoint(0, tMean_Radius, tMean_Lambda);
  tReturnGr->SetPointError(0, tMeanStatErr_Radius, tMeanStatErr_Radius, tMeanStatErr_Lambda, tMeanStatErr_Lambda);

  return tReturnGr;
}


//---------------------------------------------------------------------------------------------------------------------------------
TCanvas* DrawAllReF0vsImF0(AnalysisType aAnType, IncludeResType aIncludeResType=kInclude10ResAnd3Res, IncludeD0Type aIncludeD0Type=kFreeAndFixedD0, bool bSaveImage=false)
{
  TCanvas* tReturnCan = new TCanvas(TString::Format("tCanReF0vsImF0_%s", cAnalysisBaseTags[aAnType]), 
                                    TString::Format("tCanReF0vsImF0_%s", cAnalysisBaseTags[aAnType]));
  tReturnCan->cd();
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);
  //------------------------
  vector<FitInfo> aFitInfoVec = GetFitInfoVec(aAnType, aIncludeResType, aIncludeD0Type);
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
    DrawReF0vsImF0((TPad*)tPadReF0vsImF0, aFitInfoVec[i]);
    DrawD0((TPad*)tPadD0, aFitInfoVec[i]);

    if(i%2==0)
    {
      if(iTex>2) continue;

      tPadReF0vsImF0->cd();
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
  tPadReF0vsImF0->cd();
  //------------------------------------------------
  iTex++;
  tTex->DrawLatex(tStartX, tStartY-(iTex)*tIncrementY, "Free D0");
  tMarker->SetMarkerStyle(20);
  tMarker->SetMarkerColor(1);
  tMarker->DrawMarker(tStartX-0.05, tStartY-(iTex)*tIncrementY);
  iTex++;

  tTex->DrawLatex(tStartX, tStartY-(iTex)*tIncrementY, "Fix D0");
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
  DrawChi2PerNDF(tStartXChi2, tStartYChi2, (TPad*)tPadReF0vsImF0, aFitInfoVec);
  //--------------------------------------------------------

  if(bSaveImage)
  {
    TString tSaveLocationFull_ReF0vsImF0;

    tSaveLocationFull_ReF0vsImF0 = gSaveLocationBase + TString::Format("%s/ReF0vsImF0%s.eps", cAnalysisBaseTags[aAnType], cIncludeResTypeTags[aIncludeResType]);
    tReturnCan->SaveAs(tSaveLocationFull_ReF0vsImF0);
  }

  return tReturnCan;
}


//---------------------------------------------------------------------------------------------------------------------------------
TCanvas* DrawAllRadiusvsLambda(AnalysisType aAnType, CentralityType aCentType=k0010, IncludeResType aIncludeResType=kInclude10ResAnd3Res, IncludeD0Type aIncludeD0Type=kFreeAndFixedD0, bool bSaveImage=false)
{
  TCanvas* tReturnCan = new TCanvas(TString::Format("tCanRadiusvsLambda_%s%s", cAnalysisBaseTags[aAnType], cCentralityTags[aCentType]), 
                                    TString::Format("tCanRadiusvsLambda_%s%s", cAnalysisBaseTags[aAnType], cCentralityTags[aCentType]));
  tReturnCan->cd();
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  SetupRadiusvsLambdaAxes((TPad*)tReturnCan);
  //------------------------
  vector<FitInfo> aFitInfoVec = GetFitInfoVec(aAnType, aIncludeResType, aIncludeD0Type);
  //------------------------
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

  tTex->DrawLatex(tStartX, tStartY-(iTex)*tIncrementY, "Fix D0");
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
  DrawChi2PerNDF(tStartXChi2, tStartYChi2, (TPad*)tReturnCan, aFitInfoVec);
  //--------------------------------------------------------

  if(bSaveImage)
  {
    TString tSaveLocationFull_RadiusvsLambda;

    tSaveLocationFull_RadiusvsLambda = gSaveLocationBase + TString::Format("%s/RadiusvsLambda%s%s.eps", cAnalysisBaseTags[aAnType], cCentralityTags[aCentType], cIncludeResTypeTags[aIncludeResType]);
    tReturnCan->SaveAs(tSaveLocationFull_RadiusvsLambda);
  }

  return tReturnCan;
}

//---------------------------------------------------------------------------------------------------------------------------------
void DrawRadiusvsLambdaAcrossAnalyses(TPad* aPad, int aMarkerStyle=20, double aMarkerSize=1., CentralityType aCentType=k0010, IncludeResType aIncludeResType=kInclude10ResAnd3Res, IncludeD0Type aIncludeD0Type=kFreeAndFixedD0)
{
  aPad->cd();
  //------------------------
  vector<FitInfo> aFitInfoVec_LamKchP = GetFitInfoVec(kLamKchP, aIncludeResType, aIncludeD0Type);
  vector<FitInfo> aFitInfoVec_LamKchM = GetFitInfoVec(kLamKchM, aIncludeResType, aIncludeD0Type);
  vector<FitInfo> aFitInfoVec_LamK0 = GetFitInfoVec(kLamK0, aIncludeResType, aIncludeD0Type);

  Color_t tColor_LamKchP = kRed+1;
  Color_t tColor_LamKchM = kBlue+1;
  Color_t tColor_LamK0 = kBlack;

  //------------------------

  TGraphAsymmErrors *tGr_LamKchP, *tGr_LamKchM, *tGr_LamK0;
  if(aIncludeResType==kInclude10ResAnd3Res || aIncludeD0Type==kFreeAndFixedD0)
  {
    tGr_LamKchP = GetWeightedMeanRadiusvsLambda(aFitInfoVec_LamKchP, aCentType, aIncludeResType, aIncludeD0Type);
    tGr_LamKchM = GetWeightedMeanRadiusvsLambda(aFitInfoVec_LamKchM, aCentType, aIncludeResType, aIncludeD0Type);
    tGr_LamK0   = GetWeightedMeanRadiusvsLambda(aFitInfoVec_LamK0, aCentType, aIncludeResType, aIncludeD0Type);
  }
  else
  {
    assert(!(aIncludeResType==kInclude10ResAnd3Res));
    if(aIncludeResType==kInclude10ResOnly) assert(aFitInfoVec_LamKchP[0].all10ResidualsUsed);
    if(aIncludeResType==kInclude3ResOnly) assert(!aFitInfoVec_LamKchP[0].all10ResidualsUsed);

    assert(!(aIncludeD0Type==kFreeAndFixedD0));
    if(aIncludeD0Type==kFreeD0Only) assert(aFitInfoVec_LamKchP[0].freeD0);
    if(aIncludeD0Type==kFixedD0Only) assert(!aFitInfoVec_LamKchP[0].freeD0);

    tGr_LamKchP = GetRadiusvsLambda(aFitInfoVec_LamKchP[0], aCentType);
    tGr_LamKchM = GetRadiusvsLambda(aFitInfoVec_LamKchM[0], aCentType);
    tGr_LamK0   = GetRadiusvsLambda(aFitInfoVec_LamK0[0], aCentType);
  }


  tGr_LamKchP->SetMarkerColor(tColor_LamKchP);
  tGr_LamKchP->SetMarkerStyle(aMarkerStyle);
  tGr_LamKchP->SetMarkerSize(aMarkerSize);
  tGr_LamKchP->SetFillColor(tColor_LamKchP);
  tGr_LamKchP->SetFillStyle(1000);
  tGr_LamKchP->SetLineColor(tColor_LamKchP);
  tGr_LamKchP->SetLineWidth(1);


  tGr_LamKchM->SetMarkerColor(tColor_LamKchM);
  tGr_LamKchM->SetMarkerStyle(aMarkerStyle);
  tGr_LamKchM->SetMarkerSize(aMarkerSize);
  tGr_LamKchM->SetFillColor(tColor_LamKchM);
  tGr_LamKchM->SetFillStyle(1000);
  tGr_LamKchM->SetLineColor(tColor_LamKchM);
  tGr_LamKchM->SetLineWidth(1);


  tGr_LamK0->SetMarkerColor(tColor_LamK0);
  tGr_LamK0->SetMarkerStyle(aMarkerStyle);
  tGr_LamK0->SetMarkerSize(aMarkerSize);
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
void DrawRadiusvsLambdaAcrossAnalysesQMResults(TPad* aPad, int aMarkerStyle=20, double aMarkerSize=1., CentralityType aCentType=k0010)
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
  tGr_LamKchP->SetMarkerSize(aMarkerSize);
  tGr_LamKchP->SetFillColor(tColor_LamKchP);
  tGr_LamKchP->SetFillStyle(1000);
  tGr_LamKchP->SetLineColor(tColor_LamKchP);
  tGr_LamKchP->SetLineWidth(1);


  tGr_LamKchM->SetMarkerColor(tColor_LamKchM);
  tGr_LamKchM->SetMarkerStyle(aMarkerStyle);
  tGr_LamKchM->SetMarkerSize(aMarkerSize);
  tGr_LamKchM->SetFillColor(tColor_LamKchM);
  tGr_LamKchM->SetFillStyle(1000);
  tGr_LamKchM->SetLineColor(tColor_LamKchM);
  tGr_LamKchM->SetLineWidth(1);


  tGr_LamK0->SetMarkerColor(tColor_LamK0);
  tGr_LamK0->SetMarkerStyle(aMarkerStyle);
  tGr_LamK0->SetMarkerSize(aMarkerSize);
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
TCanvas* CompareAllRadiusvsLambdaAcrossAnalyses(TString aCanBaseName, CentralityType aCentType, vector<IncludeResType> &aIncResTypes, vector<IncludeD0Type> &aIncD0Types, vector<int> &aMarkerStyles, vector<double> &aMarkerSizes, vector<bool> &aIncludePlots, vector<TString> &aDescriptors, vector<double> &aDescriptorsPositionInfo)
{
  //Note aDescriptorsPositionInfo = [tStartX, tStartY, tIncrementX, tIncrementY, tTextSize]
  assert(aDescriptorsPositionInfo.size()==5);

  assert(aIncResTypes.size() == aIncD0Types.size());
  assert(aIncResTypes.size() == aMarkerStyles.size());
  assert(aIncResTypes.size() == aIncludePlots.size());
  assert(aIncResTypes.size() == aDescriptors.size());
  //----------------------------------------

  TCanvas* tReturnCan = new TCanvas(TString::Format("tCan%s%s", aCanBaseName.Data(), cCentralityTags[aCentType]), 
                                    TString::Format("tCan%s%s", aCanBaseName.Data(), cCentralityTags[aCentType]));
  tReturnCan->cd();
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  SetupRadiusvsLambdaAxes((TPad*)tReturnCan);

  //------------------------------------------------------
  double tStartX =     aDescriptorsPositionInfo[0];
  double tStartY =     aDescriptorsPositionInfo[1];
  double tIncrementX = aDescriptorsPositionInfo[2];
  double tIncrementY = aDescriptorsPositionInfo[3];
  double tTextSize =   aDescriptorsPositionInfo[4];

  TLatex* tTex = new TLatex();
  tTex->SetTextAlign(12);
  tTex->SetLineWidth(2);
  tTex->SetTextFont(42);
  tTex->SetTextSize(tTextSize);

  const Size_t tMarkerSize=1.6;
  TMarker *tMarker = new TMarker();
  tMarker->SetMarkerSize(tMarkerSize);
  tMarker->SetMarkerStyle(20);
  tMarker->SetMarkerColor(kBlack);

  int iTex = 0;

  for(unsigned int i=0; i<aIncResTypes.size(); i++)
  {
    if(aIncludePlots[i])
    {
      if(aIncResTypes[i]==kIncludeNoRes) DrawRadiusvsLambdaAcrossAnalysesQMResults((TPad*)tReturnCan, aMarkerStyles[i], aMarkerSizes[i], aCentType);
      else DrawRadiusvsLambdaAcrossAnalyses((TPad*)tReturnCan, aMarkerStyles[i], aMarkerSizes[i], aCentType, aIncResTypes[i], aIncD0Types[i]);

      tTex->DrawLatex(tStartX, tStartY-iTex*tIncrementY, aDescriptors[i]);
      tMarker->SetMarkerStyle(aMarkerStyles[i]);
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

  return tReturnCan;
}



//---------------------------------------------------------------------------------------------------------------------------------
TCanvas* CompareAllRadiusvsLambdaAcrossAnalysesv1(CentralityType aCentType=k0010, IncludeResType aIncludeResType=kInclude10ResAnd3Res, IncludeD0Type aIncludeD0Type=kFreeAndFixedD0, bool bSaveImage=false)
{
  //------------------------
  bool bInclude_QM = true;
  int tMarkerStyle_QM = 20;
  double tMarkerSize_QM = 1.0;
  IncludeResType tIncResType_QM = kIncludeNoRes;
  IncludeD0Type tIncD0Type_QM = kFreeD0Only;
  TString tDescriptor_QM = "QM 2017";

  //------------------------
  bool bInclude_10and3_Avg = true;
  int tMarkerStyle_10and3_Avg = 24;
  double tMarkerSize_10and3_Avg = 2.0;
  IncludeResType tIncResType_10and3_Avg = kInclude10ResAnd3Res;
  IncludeD0Type tIncD0Type_10and3_Avg = kFreeAndFixedD0;
  TString tDescriptor_10and3_Avg = "10 & 3 Res., Avg.";

  //------------------------
  bool bInclude_10_Avg = true;
  int tMarkerStyle_10_Avg = 29;
  double tMarkerSize_10_Avg = 2.0;
  IncludeResType tIncResType_10_Avg = kInclude10ResOnly;
  IncludeD0Type tIncD0Type_10_Avg = kFreeAndFixedD0;
  TString tDescriptor_10_Avg = "10 Res., Avg.";

  bool bInclude_3_Avg = true;
  int tMarkerStyle_3_Avg = 30;
  double tMarkerSize_3_Avg = 2.0;
  IncludeResType tIncResType_3_Avg = kInclude3ResOnly;
  IncludeD0Type tIncD0Type_3_Avg = kFreeAndFixedD0;
  TString tDescriptor_3_Avg = "3 Res., Avg.";

  //------------------------
  bool bInclude_10_FreeD0 = true;
  int tMarkerStyle_10_FreeD0 = 33;
  double tMarkerSize_10_FreeD0 = 1.0;
  IncludeResType tIncResType_10_FreeD0 = kInclude10ResOnly;
  IncludeD0Type tIncD0Type_10_FreeD0 = kFreeD0Only;
  TString tDescriptor_10_FreeD0 = "10 Res., All Free";

  bool bInclude_3_FreeD0 = true;
  int tMarkerStyle_3_FreeD0 = 27;
  double tMarkerSize_3_FreeD0 = 1.0;
  IncludeResType tIncResType_3_FreeD0 = kInclude3ResOnly;
  IncludeD0Type tIncD0Type_3_FreeD0 = kFreeD0Only;
  TString tDescriptor_3_FreeD0 = "3 Res., All Free";

  //----------------------------------------
  if(aIncludeResType==kInclude10ResAnd3Res)
  {
    bInclude_QM = true;

    bInclude_10and3_Avg = true;

    bInclude_10_Avg = true;
    bInclude_3_Avg = true;

    bInclude_10_FreeD0 = true;
    bInclude_3_FreeD0 = true;
  }
  else if(aIncludeResType==kInclude10ResOnly)
  {
    bInclude_QM = true;

    bInclude_10and3_Avg = false;

    bInclude_10_Avg = true;
    bInclude_3_Avg = false;

    bInclude_10_FreeD0 = true;
    bInclude_3_FreeD0 = false;
  }
  else if(aIncludeResType==kInclude3ResOnly)
  {
    bInclude_QM = true;

    bInclude_10and3_Avg = false;

    bInclude_10_Avg = false;
    bInclude_3_Avg = true;

    bInclude_10_FreeD0 = false;
    bInclude_3_FreeD0 = true;
  }
  else assert(0);

  //----------------------------------------
  vector<IncludeResType> tIncResTypes{tIncResType_QM,
                                      tIncResType_10and3_Avg,
                                      tIncResType_10_Avg,     tIncResType_3_Avg,
                                      tIncResType_10_FreeD0,  tIncResType_3_FreeD0};

  vector<IncludeD0Type> tIncD0Types{tIncD0Type_QM,
                                    tIncD0Type_10and3_Avg,
                                    tIncD0Type_10_Avg,     tIncD0Type_3_Avg,
                                    tIncD0Type_10_FreeD0,  tIncD0Type_3_FreeD0};



  vector<int> tMarkerStyles{tMarkerStyle_QM, 
                            tMarkerStyle_10and3_Avg,
                            tMarkerStyle_10_Avg,     tMarkerStyle_3_Avg, 
                            tMarkerStyle_10_FreeD0,  tMarkerStyle_3_FreeD0};

  vector<double> tMarkerSizes{tMarkerSize_QM, 
                              tMarkerSize_10and3_Avg,
                              tMarkerSize_10_Avg,     tMarkerSize_3_Avg, 
                              tMarkerSize_10_FreeD0,  tMarkerSize_3_FreeD0};

  vector<bool> tIncludePlots{bInclude_QM, 
                             bInclude_10and3_Avg, 
                             bInclude_10_Avg,    bInclude_3_Avg,
                             bInclude_10_FreeD0, bInclude_3_FreeD0};

  vector<TString> tDescriptors{tDescriptor_QM,
                               tDescriptor_10and3_Avg,
                               tDescriptor_10_Avg,     tDescriptor_3_Avg, 
                               tDescriptor_10_FreeD0,  tDescriptor_3_FreeD0};


  //------------------------------------------------------

  TString tCanBaseName = "CompareAllRadiusvsLambdaAcrossAnalysesv1";

  double tStartX = 5.8;
  double tStartY = 0.7;
  double tIncrementX = 0.10;
  double tIncrementY = 0.10;
  double tTextSize = 0.04;
  vector<double> tDescriptorsPositionInfo{tStartX, tStartY, tIncrementX, tIncrementY, tTextSize};

  TCanvas* tReturnCan = CompareAllRadiusvsLambdaAcrossAnalyses(tCanBaseName, aCentType, tIncResTypes, tIncD0Types, tMarkerStyles, tMarkerSizes, tIncludePlots, tDescriptors, tDescriptorsPositionInfo);

  //------------------------------------------------------

  if(bSaveImage)
  {
    TString tSaveLocationFull;
    tSaveLocationFull = gSaveLocationBase + TString::Format("AllRadiusvsLambdaAcrossAnalysesv1%s%s.eps", cCentralityTags[aCentType], cIncludeResTypeTags[aIncludeResType]);
    tReturnCan->SaveAs(tSaveLocationFull);
  }

  return tReturnCan;
}

//---------------------------------------------------------------------------------------------------------------------------------
TCanvas* CompareAllRadiusvsLambdaAcrossAnalysesv2(CentralityType aCentType=k0010, IncludeResType aIncludeResType=kInclude10ResAnd3Res, IncludeD0Type aIncludeD0Type=kFreeAndFixedD0, bool bSaveImage=false)
{
  //------------------------

  bool bInclude_QM = true;
  int tMarkerStyle_QM = 20;
  double tMarkerSize_QM = 1.0;
  IncludeResType tIncResType_QM = kIncludeNoRes;
  IncludeD0Type tIncD0Type_QM = kFreeD0Only;
  TString tDescriptor_QM = "QM 2017";

  //------------------------
  bool bInclude_10and3_Avg_FreeD0 = true;
  int tMarkerStyle_10and3_Avg_FreeD0 = 21;
  double tMarkerSize_10and3_Avg_FreeD0 = 2.0;
  IncludeResType tIncResType_10and3_Avg_FreeD0 = kInclude10ResAnd3Res;
  IncludeD0Type tIncD0Type_10and3_Avg_FreeD0 = kFreeD0Only;
  TString tDescriptor_10and3_Avg_FreeD0 = "10&3 Res., Avg., Free d_{0}";

  bool bInclude_10and3_Avg_FixedD0 = true;
  int tMarkerStyle_10and3_Avg_FixedD0 = 25;
  double tMarkerSize_10and3_Avg_FixedD0 = 2.0;
  IncludeResType tIncResType_10and3_Avg_FixedD0 = kInclude10ResAnd3Res;
  IncludeD0Type tIncD0Type_10and3_Avg_FixedD0 = kFixedD0Only;
  TString tDescriptor_10and3_Avg_FixedD0 = "10&3 Res., Avg., Fix d_{0}";

  //------------------------
  bool bInclude_10_FreeD0 = true;
  int tMarkerStyle_10_FreeD0 = 33;
  double tMarkerSize_10_FreeD0 = 1.0;
  IncludeResType tIncResType_10_FreeD0 = kInclude10ResOnly;
  IncludeD0Type tIncD0Type_10_FreeD0 = kFreeD0Only;
  TString tDescriptor_10_FreeD0 = "10 Res., Free d_{0}";

  bool bInclude_10_FixedD0 = true;
  int tMarkerStyle_10_FixedD0 = 27;
  double tMarkerSize_10_FixedD0 = 1.0;
  IncludeResType tIncResType_10_FixedD0 = kInclude10ResOnly;
  IncludeD0Type tIncD0Type_10_FixedD0 = kFixedD0Only;
  TString tDescriptor_10_FixedD0 = "10 Res., Fix d_{0}";

  //------------------------
  bool bInclude_3_FreeD0 = true;
  int tMarkerStyle_3_FreeD0 = 29;
  double tMarkerSize_3_FreeD0 = 1.0;
  IncludeResType tIncResType_3_FreeD0 = kInclude3ResOnly;
  IncludeD0Type tIncD0Type_3_FreeD0 = kFreeD0Only;
  TString tDescriptor_3_FreeD0 = "3 Res., Free d_{0}";

  bool bInclude_3_FixedD0 = true;
  int tMarkerStyle_3_FixedD0 = 30;
  double tMarkerSize_3_FixedD0 = 1.0;
  IncludeResType tIncResType_3_FixedD0 = kInclude3ResOnly;
  IncludeD0Type tIncD0Type_3_FixedD0 = kFixedD0Only;
  TString tDescriptor_3_FixedD0 = "3 Res., Fix d_{0}";

  //----------------------------------------
  vector<IncludeResType> tIncResTypes{tIncResType_QM,
                                      tIncResType_10and3_Avg_FreeD0, tIncResType_10and3_Avg_FixedD0,
                                      tIncResType_10_FreeD0,     tIncResType_10_FixedD0,
                                      tIncResType_3_FreeD0,          tIncResType_3_FixedD0};

  vector<IncludeD0Type> tIncD0Types{tIncD0Type_QM,
                                    tIncD0Type_10and3_Avg_FreeD0, tIncD0Type_10and3_Avg_FixedD0,
                                    tIncD0Type_10_FreeD0,     tIncD0Type_10_FixedD0,
                                    tIncD0Type_3_FreeD0,          tIncD0Type_3_FixedD0};


  vector<int> tMarkerStyles{tMarkerStyle_QM, 
                            tMarkerStyle_10and3_Avg_FreeD0, tMarkerStyle_10and3_Avg_FixedD0,
                            tMarkerStyle_10_FreeD0,         tMarkerStyle_10_FixedD0, 
                            tMarkerStyle_3_FreeD0,          tMarkerStyle_3_FixedD0};

  vector<double> tMarkerSizes{tMarkerSize_QM, 
                              tMarkerSize_10and3_Avg_FreeD0, tMarkerSize_10and3_Avg_FixedD0,
                              tMarkerSize_10_FreeD0,         tMarkerSize_10_FixedD0, 
                              tMarkerSize_3_FreeD0,          tMarkerSize_3_FixedD0};

  vector<bool> tIncludePlots{bInclude_QM, 
                             bInclude_10and3_Avg_FreeD0, bInclude_10and3_Avg_FixedD0,
                             bInclude_10_FreeD0,    bInclude_10_FixedD0,
                             bInclude_3_FreeD0,         bInclude_3_FixedD0};

  vector<TString> tDescriptors{tDescriptor_QM,
                               tDescriptor_10and3_Avg_FreeD0, tDescriptor_10and3_Avg_FixedD0,
                               tDescriptor_10_FreeD0,     tDescriptor_10_FixedD0, 
                               tDescriptor_3_FreeD0,          tDescriptor_3_FixedD0};

  //------------------------------------------------------

  TString tCanBaseName = "CompareAllRadiusvsLambdaAcrossAnalysesv2";

  double tStartX = 5.8;
  double tStartY = 0.75;
  double tIncrementX = 0.11;
  double tIncrementY = 0.11;
  double tTextSize = 0.03;
  vector<double> tDescriptorsPositionInfo{tStartX, tStartY, tIncrementX, tIncrementY, tTextSize};

  TCanvas* tReturnCan = CompareAllRadiusvsLambdaAcrossAnalyses(tCanBaseName, aCentType, tIncResTypes, tIncD0Types, tMarkerStyles, tMarkerSizes, tIncludePlots, tDescriptors, tDescriptorsPositionInfo);

  //------------------------------------------------------

  if(bSaveImage)
  {
    TString tSaveLocationFull;
    tSaveLocationFull = gSaveLocationBase + TString::Format("AllRadiusvsLambdaAcrossAnalysesv2%s%s.eps", cCentralityTags[aCentType], cIncludeResTypeTags[aIncludeResType]);
    tReturnCan->SaveAs(tSaveLocationFull);
  }

  return tReturnCan;
}













//---------------------------------------------------------------------------------------------------------------------------------
void DrawReF0vsImF0AcrossAnalyses(TPad* aPad, int aMarkerStyle=20, double aMarkerSize=1., IncludeResType aIncludeResType=kInclude10ResAnd3Res, IncludeD0Type aIncludeD0Type=kFreeAndFixedD0)
{
  aPad->cd();
  //------------------------
  vector<FitInfo> aFitInfoVec_LamKchP = GetFitInfoVec(kLamKchP, aIncludeResType, aIncludeD0Type);
  vector<FitInfo> aFitInfoVec_LamKchM = GetFitInfoVec(kLamKchM, aIncludeResType, aIncludeD0Type);
  vector<FitInfo> aFitInfoVec_LamK0 = GetFitInfoVec(kLamK0, aIncludeResType, aIncludeD0Type);

  Color_t tColor_LamKchP = kRed+1;
  Color_t tColor_LamKchM = kBlue+1;
  Color_t tColor_LamK0 = kBlack;

  //------------------------
  TGraphAsymmErrors *tGr_LamKchP, *tGr_LamKchM, *tGr_LamK0;
  if(aIncludeResType==kInclude10ResAnd3Res || aIncludeD0Type==kFreeAndFixedD0)
  {
    tGr_LamKchP = GetWeightedMeanReF0vsImF0(aFitInfoVec_LamKchP, aIncludeResType, aIncludeD0Type);
    tGr_LamKchM = GetWeightedMeanReF0vsImF0(aFitInfoVec_LamKchM, aIncludeResType, aIncludeD0Type);
    tGr_LamK0   = GetWeightedMeanReF0vsImF0(aFitInfoVec_LamK0, aIncludeResType, aIncludeD0Type);
  }
  else
  {
    assert(!(aIncludeResType==kInclude10ResAnd3Res));
    if(aIncludeResType==kInclude10ResOnly) assert(aFitInfoVec_LamKchP[0].all10ResidualsUsed);
    if(aIncludeResType==kInclude3ResOnly) assert(!aFitInfoVec_LamKchP[0].all10ResidualsUsed);

    assert(!(aIncludeD0Type==kFreeAndFixedD0));
    if(aIncludeD0Type==kFreeD0Only) assert(aFitInfoVec_LamKchP[0].freeD0);
    if(aIncludeD0Type==kFixedD0Only) assert(!aFitInfoVec_LamKchP[0].freeD0);

    tGr_LamKchP = GetReF0vsImF0(aFitInfoVec_LamKchP[0]);
    tGr_LamKchM = GetReF0vsImF0(aFitInfoVec_LamKchM[0]);
    tGr_LamK0   = GetReF0vsImF0(aFitInfoVec_LamK0[0]);
  }

  tGr_LamKchP->SetMarkerColor(tColor_LamKchP);
  tGr_LamKchP->SetMarkerStyle(aMarkerStyle);
  tGr_LamKchP->SetMarkerSize(aMarkerSize);
  tGr_LamKchP->SetFillColor(tColor_LamKchP);
  tGr_LamKchP->SetFillStyle(1000);
  tGr_LamKchP->SetLineColor(tColor_LamKchP);
  tGr_LamKchP->SetLineWidth(1);


  tGr_LamKchM->SetMarkerColor(tColor_LamKchM);
  tGr_LamKchM->SetMarkerStyle(aMarkerStyle);
  tGr_LamKchM->SetMarkerSize(aMarkerSize);
  tGr_LamKchM->SetFillColor(tColor_LamKchM);
  tGr_LamKchM->SetFillStyle(1000);
  tGr_LamKchM->SetLineColor(tColor_LamKchM);
  tGr_LamKchM->SetLineWidth(1);


  tGr_LamK0->SetMarkerColor(tColor_LamK0);
  tGr_LamK0->SetMarkerStyle(aMarkerStyle);
  tGr_LamK0->SetMarkerSize(aMarkerSize);
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
void DrawReF0vsImF0AcrossAnalysesQMResults(TPad* aPad, int aMarkerStyle=20, double aMarkerSize=1.)
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
  tGr_LamKchP->SetMarkerSize(aMarkerSize);
  tGr_LamKchP->SetFillColor(tColor_LamKchP);
  tGr_LamKchP->SetFillStyle(1000);
  tGr_LamKchP->SetLineColor(tColor_LamKchP);
  tGr_LamKchP->SetLineWidth(1);


  tGr_LamKchM->SetMarkerColor(tColor_LamKchM);
  tGr_LamKchM->SetMarkerStyle(aMarkerStyle);
  tGr_LamKchM->SetMarkerSize(aMarkerSize);
  tGr_LamKchM->SetFillColor(tColor_LamKchM);
  tGr_LamKchM->SetFillStyle(1000);
  tGr_LamKchM->SetLineColor(tColor_LamKchM);
  tGr_LamKchM->SetLineWidth(1);


  tGr_LamK0->SetMarkerColor(tColor_LamK0);
  tGr_LamK0->SetMarkerStyle(aMarkerStyle);
  tGr_LamK0->SetMarkerSize(aMarkerSize);
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
void DrawD0AcrossAnalyses(TPad* aPad, int aMarkerStyle=20, double aMarkerSize=1., IncludeResType aIncludeResType=kInclude10ResAnd3Res, IncludeD0Type aIncludeD0Type=kFreeAndFixedD0)
{
  aPad->cd();
  //------------------------
  vector<FitInfo> aFitInfoVec_LamKchP = GetFitInfoVec(kLamKchP, aIncludeResType, aIncludeD0Type);
  vector<FitInfo> aFitInfoVec_LamKchM = GetFitInfoVec(kLamKchM, aIncludeResType, aIncludeD0Type);
  vector<FitInfo> aFitInfoVec_LamK0 = GetFitInfoVec(kLamK0, aIncludeResType, aIncludeD0Type);

  Color_t tColor_LamKchP = kRed+1;
  Color_t tColor_LamKchM = kBlue+1;
  Color_t tColor_LamK0 = kBlack;

  //------------------------
  TGraphAsymmErrors *tGr_LamKchP, *tGr_LamKchM, *tGr_LamK0;
  if(aIncludeResType==kInclude10ResAnd3Res || aIncludeD0Type==kFreeAndFixedD0)
  {
    tGr_LamKchP = GetWeightedMeanD0(aFitInfoVec_LamKchP, aIncludeResType, aIncludeD0Type);
    tGr_LamKchM = GetWeightedMeanD0(aFitInfoVec_LamKchM, aIncludeResType, aIncludeD0Type);
    tGr_LamK0   = GetWeightedMeanD0(aFitInfoVec_LamK0, aIncludeResType, aIncludeD0Type);
  }
  else
  {
    assert(!(aIncludeResType==kInclude10ResAnd3Res));
    if(aIncludeResType==kInclude10ResOnly) assert(aFitInfoVec_LamKchP[0].all10ResidualsUsed);
    if(aIncludeResType==kInclude3ResOnly) assert(!aFitInfoVec_LamKchP[0].all10ResidualsUsed);

    assert(!(aIncludeD0Type==kFreeAndFixedD0));
    if(aIncludeD0Type==kFreeD0Only) assert(aFitInfoVec_LamKchP[0].freeD0);
    if(aIncludeD0Type==kFixedD0Only) assert(!aFitInfoVec_LamKchP[0].freeD0);

    tGr_LamKchP = GetD0(aFitInfoVec_LamKchP[0]);
    tGr_LamKchM = GetD0(aFitInfoVec_LamKchM[0]);
    tGr_LamK0   = GetD0(aFitInfoVec_LamK0[0]);
  }

  tGr_LamKchP->SetMarkerColor(tColor_LamKchP);
  tGr_LamKchP->SetMarkerStyle(aMarkerStyle);
  tGr_LamKchP->SetMarkerSize(aMarkerSize);
  tGr_LamKchP->SetFillColor(tColor_LamKchP);
  tGr_LamKchP->SetFillStyle(1000);
  tGr_LamKchP->SetLineColor(tColor_LamKchP);
  tGr_LamKchP->SetLineWidth(1);


  tGr_LamKchM->SetMarkerColor(tColor_LamKchM);
  tGr_LamKchM->SetMarkerStyle(aMarkerStyle);
  tGr_LamKchM->SetMarkerSize(aMarkerSize);
  tGr_LamKchM->SetFillColor(tColor_LamKchM);
  tGr_LamKchM->SetFillStyle(1000);
  tGr_LamKchM->SetLineColor(tColor_LamKchM);
  tGr_LamKchM->SetLineWidth(1);


  tGr_LamK0->SetMarkerColor(tColor_LamK0);
  tGr_LamK0->SetMarkerStyle(aMarkerStyle);
  tGr_LamK0->SetMarkerSize(aMarkerSize);
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
void DrawD0AcrossAnalysesQMResults(TPad* aPad, int aMarkerStyle=20, double aMarkerSize=1.)
{
  aPad->cd();
  //------------------------
  Color_t tColor_LamKchP = kRed+1;
  Color_t tColor_LamKchM = kBlue+1;
  Color_t tColor_LamK0 = kBlack;

  //------------------------
  TGraphAsymmErrors *tGr_LamKchP, *tGr_LamKchM, *tGr_LamK0;
  tGr_LamKchP = GetD0(tFitInfoQM_LamKchP);
  tGr_LamKchM = GetD0(tFitInfoQM_LamKchM);
  tGr_LamK0   = GetD0(tFitInfoQM_LamK0);

  //------------------------
  tGr_LamKchP->SetMarkerColor(tColor_LamKchP);
  tGr_LamKchP->SetMarkerStyle(aMarkerStyle);
  tGr_LamKchP->SetMarkerSize(aMarkerSize);
  tGr_LamKchP->SetFillColor(tColor_LamKchP);
  tGr_LamKchP->SetFillStyle(1000);
  tGr_LamKchP->SetLineColor(tColor_LamKchP);
  tGr_LamKchP->SetLineWidth(1);


  tGr_LamKchM->SetMarkerColor(tColor_LamKchM);
  tGr_LamKchM->SetMarkerStyle(aMarkerStyle);
  tGr_LamKchM->SetMarkerSize(aMarkerSize);
  tGr_LamKchM->SetFillColor(tColor_LamKchM);
  tGr_LamKchM->SetFillStyle(1000);
  tGr_LamKchM->SetLineColor(tColor_LamKchM);
  tGr_LamKchM->SetLineWidth(1);


  tGr_LamK0->SetMarkerColor(tColor_LamK0);
  tGr_LamK0->SetMarkerStyle(aMarkerStyle);
  tGr_LamK0->SetMarkerSize(aMarkerSize);
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
TCanvas* CompareAllReF0vsImF0AcrossAnalyses(TString aCanBaseName, vector<IncludeResType> &aIncResTypes, vector<IncludeD0Type> &aIncD0Types, vector<int> &aMarkerStyles, vector<double> &aMarkerSizes, vector<bool> &aIncludePlots, vector<TString> &aDescriptors, vector<double> &aDescriptorsPositionInfo)
{
  //Note aDescriptorsPositionInfo = [tStartX, tStartY, tIncrementX, tIncrementY, tTextSize]
  assert(aDescriptorsPositionInfo.size()==5);

  assert(aIncResTypes.size() == aIncD0Types.size());
  assert(aIncResTypes.size() == aMarkerStyles.size());
  assert(aIncResTypes.size() == aMarkerSizes.size());
  assert(aIncResTypes.size() == aIncludePlots.size());
  assert(aIncResTypes.size() == aDescriptors.size());
  //----------------------------------------

  TCanvas* tReturnCan = new TCanvas(TString::Format("tCan%s", aCanBaseName.Data()), TString::Format("tCan%s", aCanBaseName.Data()));
  tReturnCan->cd();

  TPad* tPadReF0vsImF0 = new TPad(TString::Format("tPadReF0vsImF0%s", aCanBaseName.Data()), TString::Format("tPadReF0vsImF0%s", aCanBaseName.Data()), 
                                  0.0, 0.0, 0.8, 1.0);
  tPadReF0vsImF0->SetRightMargin(0.01);
  tPadReF0vsImF0->Draw();

  TPad* tPadD0 = new TPad(TString::Format("tPadD0%s", aCanBaseName.Data()), TString::Format("tPadD0%s", aCanBaseName.Data()), 
                          0.8, 0.0, 1.0, 1.0);
  tPadD0->SetRightMargin(0.4);
  tPadD0->SetLeftMargin(0.);
  tPadD0->Draw();

  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  SetupReF0vsImF0AndD0Axes(tPadReF0vsImF0, tPadD0);

  //------------------------------------------------------
  double tStartX =     aDescriptorsPositionInfo[0];
  double tStartY =     aDescriptorsPositionInfo[1];
  double tIncrementX = aDescriptorsPositionInfo[2];
  double tIncrementY = aDescriptorsPositionInfo[3];
  double tTextSize =   aDescriptorsPositionInfo[4];

  TLatex* tTex = new TLatex();
  tTex->SetTextAlign(12);
  tTex->SetLineWidth(2);
  tTex->SetTextFont(42);
  tTex->SetTextSize(tTextSize);

  const Size_t tMarkerSize=1.6;
  TMarker *tMarker = new TMarker();
  tMarker->SetMarkerSize(tMarkerSize);
  tMarker->SetMarkerStyle(20);
  tMarker->SetMarkerColor(kBlack);

  int iTex = 0;

  for(unsigned int i=0; i<aIncResTypes.size(); i++)
  {
    if(aIncludePlots[i])
    {
      if(aIncResTypes[i]==kIncludeNoRes)  //Signifies QM results
      {
        DrawReF0vsImF0AcrossAnalysesQMResults((TPad*)tPadReF0vsImF0, aMarkerStyles[i], aMarkerSizes[i]);
        DrawD0AcrossAnalysesQMResults((TPad*)tPadD0, aMarkerStyles[i], aMarkerSizes[i]);
      }
      else 
      {
        DrawReF0vsImF0AcrossAnalyses((TPad*)tPadReF0vsImF0, aMarkerStyles[i], aMarkerSizes[i], aIncResTypes[i], aIncD0Types[i]);
        DrawD0AcrossAnalyses((TPad*)tPadD0, aMarkerStyles[i], aMarkerSizes[i], aIncResTypes[i], aIncD0Types[i]);
      }

      tPadReF0vsImF0->cd();
      tTex->DrawLatex(tStartX, tStartY-iTex*tIncrementY, aDescriptors[i]);
      tMarker->SetMarkerStyle(aMarkerStyles[i]);
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
  DrawAnalysisStamps((TPad*)tPadReF0vsImF0, tStartXStamp, tStartYStamp, tIncrementXStamp, tIncrementYStamp, tTextSizeStamp, tMarkerStyleStamp);

  //------------------------------------------------------

  return tReturnCan;
}













//---------------------------------------------------------------------------------------------------------------------------------
TCanvas* CompareAllReF0vsImF0AcrossAnalysesv1(IncludeResType aIncludeResType=kInclude10ResAnd3Res, IncludeD0Type aIncludeD0Type=kFreeAndFixedD0, bool bSaveImage=false)
{
  //------------------------
  bool bInclude_QM = true;
  int tMarkerStyle_QM = 20;
  double tMarkerSize_QM = 1.0;
  IncludeResType tIncResType_QM = kIncludeNoRes;
  IncludeD0Type tIncD0Type_QM = kFreeD0Only;
  TString tDescriptor_QM = "QM 2017";

  //------------------------
  bool bInclude_10and3_Avg = true;
  int tMarkerStyle_10and3_Avg = 24;
  double tMarkerSize_10and3_Avg = 2.0;
  IncludeResType tIncResType_10and3_Avg = kInclude10ResAnd3Res;
  IncludeD0Type tIncD0Type_10and3_Avg = kFreeAndFixedD0;
  TString tDescriptor_10and3_Avg = "10 & 3 Res., Avg.";

  //------------------------
  bool bInclude_10_Avg = true;
  int tMarkerStyle_10_Avg = 29;
  double tMarkerSize_10_Avg = 2.0;
  IncludeResType tIncResType_10_Avg = kInclude10ResOnly;
  IncludeD0Type tIncD0Type_10_Avg = kFreeAndFixedD0;
  TString tDescriptor_10_Avg = "10 Res., Avg.";

  bool bInclude_3_Avg = true;
  int tMarkerStyle_3_Avg = 30;
  double tMarkerSize_3_Avg = 2.0;
  IncludeResType tIncResType_3_Avg = kInclude3ResOnly;
  IncludeD0Type tIncD0Type_3_Avg = kFreeAndFixedD0;
  TString tDescriptor_3_Avg = "3 Res., Avg.";

  //------------------------
  bool bInclude_10_FreeD0 = true;
  int tMarkerStyle_10_FreeD0 = 33;
  double tMarkerSize_10_FreeD0 = 1.0;
  IncludeResType tIncResType_10_FreeD0 = kInclude10ResOnly;
  IncludeD0Type tIncD0Type_10_FreeD0 = kFreeD0Only;
  TString tDescriptor_10_FreeD0 = "10 Res., All Free";

  bool bInclude_3_FreeD0 = true;
  int tMarkerStyle_3_FreeD0 = 27;
  double tMarkerSize_3_FreeD0 = 1.0;
  IncludeResType tIncResType_3_FreeD0 = kInclude3ResOnly;
  IncludeD0Type tIncD0Type_3_FreeD0 = kFreeD0Only;
  TString tDescriptor_3_FreeD0 = "3 Res., All Free";

  //----------------------------------------
  if(aIncludeResType==kInclude10ResOnly)
  {
    bInclude_10and3_Avg = false;

    bInclude_10_Avg = true;
    bInclude_3_Avg = false;

    bInclude_10_FreeD0 = true;
    bInclude_3_FreeD0 = false;
  }
  else if(aIncludeResType==kInclude3ResOnly)
  {
    bInclude_10and3_Avg = false;

    bInclude_10_Avg = false;
    bInclude_3_Avg = true;

    bInclude_10_FreeD0 = false;
    bInclude_3_FreeD0 = true;
  }


  //----------------------------------------
  vector<IncludeResType> tIncResTypes{tIncResType_QM,
                                      tIncResType_10and3_Avg,
                                      tIncResType_10_Avg,     tIncResType_3_Avg,
                                      tIncResType_10_FreeD0,          tIncResType_3_FreeD0};

  vector<IncludeD0Type> tIncD0Types{tIncD0Type_QM,
                                    tIncD0Type_10and3_Avg,
                                    tIncD0Type_10_Avg,     tIncD0Type_3_Avg,
                                    tIncD0Type_10_FreeD0,          tIncD0Type_3_FreeD0};


  vector<int> tMarkerStyles{tMarkerStyle_QM, 
                            tMarkerStyle_10and3_Avg,
                            tMarkerStyle_10_Avg,     tMarkerStyle_3_Avg, 
                            tMarkerStyle_10_FreeD0,  tMarkerStyle_3_FreeD0};

  vector<double> tMarkerSizes{tMarkerSize_QM, 
                              tMarkerSize_10and3_Avg,
                              tMarkerSize_10_Avg,     tMarkerSize_3_Avg, 
                              tMarkerSize_10_FreeD0,  tMarkerSize_3_FreeD0};

  vector<bool> tIncludePlots{bInclude_QM, 
                             bInclude_10and3_Avg, 
                             bInclude_10_Avg,    bInclude_3_Avg,
                             bInclude_10_FreeD0,         bInclude_3_FreeD0};

  vector<TString> tDescriptors{tDescriptor_QM,
                               tDescriptor_10and3_Avg,
                               tDescriptor_10_Avg,     tDescriptor_3_Avg, 
                               tDescriptor_10_FreeD0,          tDescriptor_3_FreeD0};

  //----------------------------------------

  TString tCanBaseName = "CompareAllReF0vsImF0AcrossAnalysesv1";

  double tStartX = 0.0;
  double tStartY = 1.4;
  double tIncrementX = 0.05;
  double tIncrementY = 0.08;
  double tTextSize = 0.04;
  vector<double> tDescriptorsPositionInfo{tStartX, tStartY, tIncrementX, tIncrementY, tTextSize};

  TCanvas* tReturnCan = CompareAllReF0vsImF0AcrossAnalyses(tCanBaseName, tIncResTypes, tIncD0Types, tMarkerStyles, tMarkerSizes, tIncludePlots, tDescriptors, tDescriptorsPositionInfo);

  //----------------------------------------
  if(bSaveImage)
  {
    TString tSaveLocationFull;

//    tSaveLocationFull = gSaveLocationBase + TString::Format("%s/ReF0vsImF0", cAnalysisBaseTags[aAnType]);
    tSaveLocationFull = gSaveLocationBase + TString::Format("AllReF0vsImF0AcrossAnalysesv1%s.eps", cIncludeResTypeTags[aIncludeResType]);
    tReturnCan->SaveAs(tSaveLocationFull);
  }

  return tReturnCan;
}


//---------------------------------------------------------------------------------------------------------------------------------
TCanvas* CompareAllReF0vsImF0AcrossAnalysesv2(IncludeResType aIncludeResType=kInclude10ResAnd3Res, IncludeD0Type aIncludeD0Type=kFreeAndFixedD0, bool bSaveImage=false)
{
  //------------------------

  bool bInclude_QM = true;
  int tMarkerStyle_QM = 20;
  double tMarkerSize_QM = 1.0;
  IncludeResType tIncResType_QM = kIncludeNoRes;
  IncludeD0Type tIncD0Type_QM = kFreeD0Only;
  TString tDescriptor_QM = "QM 2017";

  //------------------------
  bool bInclude_10and3_Avg_FreeD0 = true;
  int tMarkerStyle_10and3_Avg_FreeD0 = 21;
  double tMarkerSize_10and3_Avg_FreeD0 = 2.0;
  IncludeResType tIncResType_10and3_Avg_FreeD0 = kInclude10ResAnd3Res;
  IncludeD0Type tIncD0Type_10and3_Avg_FreeD0 = kFreeD0Only;
  TString tDescriptor_10and3_Avg_FreeD0 = "10&3 Res., Avg., Free d_{0}";

  bool bInclude_10and3_Avg_FixedD0 = true;
  int tMarkerStyle_10and3_Avg_FixedD0 = 25;
  double tMarkerSize_10and3_Avg_FixedD0 = 2.0;
  IncludeResType tIncResType_10and3_Avg_FixedD0 = kInclude10ResAnd3Res;
  IncludeD0Type tIncD0Type_10and3_Avg_FixedD0 = kFixedD0Only;
  TString tDescriptor_10and3_Avg_FixedD0 = "10&3 Res., Avg., Fix d_{0}";

  //------------------------
  bool bInclude_10_FreeD0 = true;
  int tMarkerStyle_10_FreeD0 = 33;
  double tMarkerSize_10_FreeD0 = 1.0;
  IncludeResType tIncResType_10_FreeD0 = kInclude10ResOnly;
  IncludeD0Type tIncD0Type_10_FreeD0 = kFreeD0Only;
  TString tDescriptor_10_FreeD0 = "10 Res., Free d_{0}";

  bool bInclude_10_FixedD0 = true;
  int tMarkerStyle_10_FixedD0 = 27;
  double tMarkerSize_10_FixedD0 = 1.0;
  IncludeResType tIncResType_10_FixedD0 = kInclude10ResOnly;
  IncludeD0Type tIncD0Type_10_FixedD0 = kFixedD0Only;
  TString tDescriptor_10_FixedD0 = "10 Res., Fix d_{0}";

  //------------------------
  bool bInclude_3_FreeD0 = true;
  int tMarkerStyle_3_FreeD0 = 29;
  double tMarkerSize_3_FreeD0 = 1.0;
  IncludeResType tIncResType_3_FreeD0 = kInclude3ResOnly;
  IncludeD0Type tIncD0Type_3_FreeD0 = kFreeD0Only;
  TString tDescriptor_3_FreeD0 = "3 Res., Free d_{0}";

  bool bInclude_3_FixedD0 = true;
  int tMarkerStyle_3_FixedD0 = 30;
  double tMarkerSize_3_FixedD0 = 1.0;
  IncludeResType tIncResType_3_FixedD0 = kInclude3ResOnly;
  IncludeD0Type tIncD0Type_3_FixedD0 = kFixedD0Only;
  TString tDescriptor_3_FixedD0 = "3 Res., Fix d_{0}";

  //----------------------------------------
  vector<IncludeResType> tIncResTypes{tIncResType_QM,
                                      tIncResType_10and3_Avg_FreeD0, tIncResType_10and3_Avg_FixedD0,
                                      tIncResType_10_FreeD0,     tIncResType_10_FixedD0,
                                      tIncResType_3_FreeD0,          tIncResType_3_FixedD0};

  vector<IncludeD0Type> tIncD0Types{tIncD0Type_QM,
                                    tIncD0Type_10and3_Avg_FreeD0, tIncD0Type_10and3_Avg_FixedD0,
                                    tIncD0Type_10_FreeD0,     tIncD0Type_10_FixedD0,
                                    tIncD0Type_3_FreeD0,          tIncD0Type_3_FixedD0};


  vector<int> tMarkerStyles{tMarkerStyle_QM, 
                            tMarkerStyle_10and3_Avg_FreeD0, tMarkerStyle_10and3_Avg_FixedD0,
                            tMarkerStyle_10_FreeD0,     tMarkerStyle_10_FixedD0, 
                            tMarkerStyle_3_FreeD0,          tMarkerStyle_3_FixedD0};

  vector<double> tMarkerSizes{tMarkerSize_QM, 
                              tMarkerSize_10and3_Avg_FreeD0, tMarkerSize_10and3_Avg_FixedD0,
                              tMarkerSize_10_FreeD0,     tMarkerSize_10_FixedD0, 
                              tMarkerSize_3_FreeD0,          tMarkerSize_3_FixedD0};

  vector<bool> tIncludePlots{bInclude_QM, 
                             bInclude_10and3_Avg_FreeD0, bInclude_10and3_Avg_FixedD0,
                             bInclude_10_FreeD0,    bInclude_10_FixedD0,
                             bInclude_3_FreeD0,         bInclude_3_FixedD0};

  vector<TString> tDescriptors{tDescriptor_QM,
                               tDescriptor_10and3_Avg_FreeD0, tDescriptor_10and3_Avg_FixedD0,
                               tDescriptor_10_FreeD0,     tDescriptor_10_FixedD0, 
                               tDescriptor_3_FreeD0,          tDescriptor_3_FixedD0};

  //----------------------------------------

  TString tCanBaseName = "CompareAllReF0vsImF0AcrossAnalysesv2";

  double tStartX = -0.5;
  double tStartY = 1.4;
  double tIncrementX = 0.05;
  double tIncrementY = 0.10;
  double tTextSize = 0.04;
  vector<double> tDescriptorsPositionInfo{tStartX, tStartY, tIncrementX, tIncrementY, tTextSize};

  TCanvas* tReturnCan = CompareAllReF0vsImF0AcrossAnalyses(tCanBaseName, tIncResTypes, tIncD0Types, tMarkerStyles, tMarkerSizes, tIncludePlots, tDescriptors, tDescriptorsPositionInfo);

  //----------------------------------------
  if(bSaveImage)
  {
    TString tSaveLocationFull;

//    tSaveLocationFull = gSaveLocationBase + TString::Format("%s/ReF0vsImF0", cAnalysisBaseTags[aAnType]);
    tSaveLocationFull = gSaveLocationBase + TString::Format("AllReF0vsImF0AcrossAnalysesv2%s.eps", cIncludeResTypeTags[aIncludeResType]);
    tReturnCan->SaveAs(tSaveLocationFull);
  }


  return tReturnCan;
}


//---------------------------------------------------------------------------------------------------------------------------------
TCanvas* CompareAllReF0vsImF0AcrossAnalysesv3(IncludeResType aIncludeResType=kInclude10ResAnd3Res, IncludeD0Type aIncludeD0Type=kFreeAndFixedD0, bool bSaveImage=false)
{
  //TODO
  bool bDrawv1 = false;
  bool bDrawv2 = false;

  assert(!(bDrawv1&&bDrawv2));

  //------------------------

  bool bInclude_QM = true;
  int tMarkerStyle_QM = 20;
  double tMarkerSize_QM = 1.0;
  IncludeResType tIncResType_QM = kIncludeNoRes;
  IncludeD0Type tIncD0Type_QM = kFreeD0Only;
  TString tDescriptor_QM = "QM 2017";

  //------------------------
  bool bInclude_10and3_Avg_FreeD0 = true;
  int tMarkerStyle_10and3_Avg_FreeD0 = 21;
  double tMarkerSize_10and3_Avg_FreeD0 = 1.0;
  IncludeResType tIncResType_10and3_Avg_FreeD0 = kInclude10ResAnd3Res;
  IncludeD0Type tIncD0Type_10and3_Avg_FreeD0 = kFreeD0Only;
  TString tDescriptor_10and3_Avg_FreeD0 = "10&3 Res., Avg., Free d_{0}";

  bool bInclude_10and3_Avg_FixedD0 = true;
  int tMarkerStyle_10and3_Avg_FixedD0 = 25;
  double tMarkerSize_10and3_Avg_FixedD0 = 1.0;
  IncludeResType tIncResType_10and3_Avg_FixedD0 = kInclude10ResAnd3Res;
  IncludeD0Type tIncD0Type_10and3_Avg_FixedD0 = kFixedD0Only;
  TString tDescriptor_10and3_Avg_FixedD0 = "10&3 Res., Avg., Fix d_{0}";

  bool bInclude_10and3_Avg = true;
  int tMarkerStyle_10and3_Avg = 35;
  double tMarkerSize_10and3_Avg = 2.0;
  IncludeResType tIncResType_10and3_Avg = kInclude10ResAnd3Res;
  IncludeD0Type tIncD0Type_10and3_Avg = kFreeAndFixedD0;
  TString tDescriptor_10and3_Avg = "10 & 3 Res., Avg.";

  //------------------------
  bool bInclude_10_FreeD0 = true;
  int tMarkerStyle_10_FreeD0 = 47;
  double tMarkerSize_10_FreeD0 = 1.0;
  IncludeResType tIncResType_10_FreeD0 = kInclude10ResOnly;
  IncludeD0Type tIncD0Type_10_FreeD0 = kFreeD0Only;
  TString tDescriptor_10_FreeD0 = "10 Res., Free d_{0}";

  bool bInclude_10_FixedD0 = true;
  int tMarkerStyle_10_FixedD0 = 46;
  double tMarkerSize_10_FixedD0 = 1.0;
  IncludeResType tIncResType_10_FixedD0 = kInclude10ResOnly;
  IncludeD0Type tIncD0Type_10_FixedD0 = kFixedD0Only;
  TString tDescriptor_10_FixedD0 = "10 Res., Fix d_{0}";

  bool bInclude_10_Avg = true;
  int tMarkerStyle_10_Avg = 48;
  double tMarkerSize_10_Avg = 2.0;
  IncludeResType tIncResType_10_Avg = kInclude10ResOnly;
  IncludeD0Type tIncD0Type_10_Avg = kFreeAndFixedD0;
  TString tDescriptor_10_Avg = "10 Res., Avg.";

  //------------------------
  bool bInclude_3_FreeD0 = true;
  int tMarkerStyle_3_FreeD0 = 34;
  double tMarkerSize_3_FreeD0 = 1.0;
  IncludeResType tIncResType_3_FreeD0 = kInclude3ResOnly;
  IncludeD0Type tIncD0Type_3_FreeD0 = kFreeD0Only;
  TString tDescriptor_3_FreeD0 = "3 Res., Free d_{0}";

  bool bInclude_3_FixedD0 = true;
  int tMarkerStyle_3_FixedD0 = 28;
  double tMarkerSize_3_FixedD0 = 1.0;
  IncludeResType tIncResType_3_FixedD0 = kInclude3ResOnly;
  IncludeD0Type tIncD0Type_3_FixedD0 = kFixedD0Only;
  TString tDescriptor_3_FixedD0 = "3 Res., Fix d_{0}";

  bool bInclude_3_Avg = true;
  int tMarkerStyle_3_Avg = 49;
  double tMarkerSize_3_Avg = 2.0;
  IncludeResType tIncResType_3_Avg = kInclude3ResOnly;
  IncludeD0Type tIncD0Type_3_Avg = kFreeAndFixedD0;
  TString tDescriptor_3_Avg = "3 Res., Avg.";

  //----------------------------------------

  if(aIncludeResType==kInclude10ResOnly)
  {
    bInclude_10and3_Avg_FreeD0 = false;
    bInclude_10and3_Avg_FixedD0 = false;
    bInclude_10and3_Avg = false;

    bInclude_10_FreeD0 = true;
    bInclude_10_FixedD0 = true;
    bInclude_10_Avg = true;


    bInclude_3_FreeD0 = false;
    bInclude_3_FixedD0 = false;
    bInclude_3_Avg = false;
  }
  else if(aIncludeResType==kInclude3ResOnly)
  {
    bInclude_10and3_Avg_FreeD0 = false;
    bInclude_10and3_Avg_FixedD0 = false;
    bInclude_10and3_Avg = false;

    bInclude_10_FreeD0 = false;
    bInclude_10_FixedD0 = false;
    bInclude_10_Avg = false;


    bInclude_3_FreeD0 = true;
    bInclude_3_FixedD0 = true;
    bInclude_3_Avg = true;
  }

  //----------------------------------------

  if(aIncludeD0Type==kFreeD0Only)
  {
    if(aIncludeResType==kInclude10ResAnd3Res) bInclude_10and3_Avg_FreeD0 = true;
    else bInclude_10and3_Avg_FreeD0 = false;
    bInclude_10and3_Avg_FixedD0 = false;
    bInclude_10and3_Avg = false;

    if(aIncludeResType != kInclude3ResOnly) bInclude_10_FreeD0 = true;
    else bInclude_10_FreeD0 = false;
    bInclude_10_FixedD0 = false;
    bInclude_10_Avg = false;


    if(aIncludeResType != kInclude10ResOnly) bInclude_3_FreeD0 = true;
    else bInclude_3_FreeD0 = false;
    bInclude_3_FixedD0 = false;
    bInclude_3_Avg = false;
  }
  else if(aIncludeD0Type==kFixedD0Only)
  {
    bInclude_10and3_Avg_FreeD0 = false;
    if(aIncludeResType==kInclude10ResAnd3Res) bInclude_10and3_Avg_FixedD0 = true;
    else bInclude_10and3_Avg_FixedD0 = false;
    bInclude_10and3_Avg = false;

    bInclude_10_FreeD0 = false;
    if(aIncludeResType != kInclude3ResOnly) bInclude_10_FixedD0 = true;
    else bInclude_10_FixedD0 = false;
    bInclude_10_Avg = false;


    bInclude_3_FreeD0 = false;
    if(aIncludeResType != kInclude10ResOnly) bInclude_3_FixedD0 = true;
    else bInclude_3_FixedD0 = false;
    bInclude_3_Avg = false;
  }
 

  //----------------------------------------
  vector<IncludeResType> tIncResTypes{tIncResType_QM,
                                      tIncResType_10and3_Avg_FreeD0, tIncResType_10and3_Avg_FixedD0, tIncResType_10and3_Avg,
                                      tIncResType_10_FreeD0,         tIncResType_10_FixedD0,         tIncResType_10_Avg,
                                      tIncResType_3_FreeD0,          tIncResType_3_FixedD0,          tIncResType_3_Avg};

  vector<IncludeD0Type> tIncD0Types{tIncD0Type_QM,
                                    tIncD0Type_10and3_Avg_FreeD0, tIncD0Type_10and3_Avg_FixedD0, tIncD0Type_10and3_Avg,
                                    tIncD0Type_10_FreeD0,         tIncD0Type_10_FixedD0,         tIncD0Type_10_Avg,
                                    tIncD0Type_3_FreeD0,          tIncD0Type_3_FixedD0,          tIncD0Type_3_Avg};


  vector<int> tMarkerStyles{tMarkerStyle_QM, 
                            tMarkerStyle_10and3_Avg_FreeD0, tMarkerStyle_10and3_Avg_FixedD0, tMarkerStyle_10and3_Avg,
                            tMarkerStyle_10_FreeD0,         tMarkerStyle_10_FixedD0,         tMarkerStyle_10_Avg, 
                            tMarkerStyle_3_FreeD0,          tMarkerStyle_3_FixedD0,          tMarkerStyle_3_Avg};

  vector<double> tMarkerSizes{tMarkerSize_QM, 
                              tMarkerSize_10and3_Avg_FreeD0, tMarkerSize_10and3_Avg_FixedD0, tMarkerSize_10and3_Avg,
                              tMarkerSize_10_FreeD0,         tMarkerSize_10_FixedD0,         tMarkerSize_10_Avg, 
                              tMarkerSize_3_FreeD0,          tMarkerSize_3_FixedD0,          tMarkerSize_3_Avg};

  vector<bool> tIncludePlots{bInclude_QM, 
                             bInclude_10and3_Avg_FreeD0, bInclude_10and3_Avg_FixedD0, bInclude_10and3_Avg,
                             bInclude_10_FreeD0,         bInclude_10_FixedD0,         bInclude_10_Avg,
                             bInclude_3_FreeD0,          bInclude_3_FixedD0,          bInclude_3_Avg};

  vector<TString> tDescriptors{tDescriptor_QM,
                               tDescriptor_10and3_Avg_FreeD0, tDescriptor_10and3_Avg_FixedD0, tDescriptor_10and3_Avg,
                               tDescriptor_10_FreeD0,         tDescriptor_10_FixedD0,         tDescriptor_10_Avg,
                               tDescriptor_3_FreeD0,          tDescriptor_3_FixedD0,          tDescriptor_3_Avg};

  //----------------------------------------
  if(bDrawv1) tIncludePlots = {true,
                               false, false, true, 
                               true,  false, true, 
                               true,  false, true};

  if(bDrawv2) tIncludePlots = {true,
                               true, true, false, 
                               true, true, false, 
                               true, true, false};

  //----------------------------------------

  TString tCanBaseName = "CompareAllReF0vsImF0AcrossAnalysesv3";

  double tStartX = -0.5;
  double tStartY = 1.4;
  double tIncrementX = 0.05;
  double tIncrementY = 0.10;
  double tTextSize = 0.04;

  if(bDrawv1)
  {
    tStartX = 0.0;
    tStartY = 1.4;
    tIncrementX = 0.05;
    tIncrementY = 0.08;
    tTextSize = 0.04;
  }
  if(bDrawv2)
  {
    tStartX = -0.5;
    tStartY = 1.4;
    tIncrementX = 0.05;
    tIncrementY = 0.10;
    tTextSize = 0.04;
  }

  vector<double> tDescriptorsPositionInfo{tStartX, tStartY, tIncrementX, tIncrementY, tTextSize};

  TCanvas* tReturnCan = CompareAllReF0vsImF0AcrossAnalyses(tCanBaseName, tIncResTypes, tIncD0Types, tMarkerStyles, tMarkerSizes, tIncludePlots, tDescriptors, tDescriptorsPositionInfo);

  //----------------------------------------
  if(bSaveImage)
  {
    TString tSaveLocationFull;

//    tSaveLocationFull = gSaveLocationBase + TString::Format("%s/ReF0vsImF0", cAnalysisBaseTags[aAnType]);
    tSaveLocationFull = gSaveLocationBase + TString::Format("AllReF0vsImF0AcrossAnalysesv3%s.eps", cIncludeResTypeTags[aIncludeResType]);
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

  IncludeD0Type tIncludeD0Type;
    tIncludeD0Type = kFreeAndFixedD0;
//    tIncludeD0Type = kFreeD0Only;
//    tIncludeD0Type = kFixedD0Only;

  CentralityType tCentType = kMB;

  bool bSaveFigures = false;

//-------------------------------------------------------------------------------
/*
  TCanvas* tCanReF0vsImF0_LamKchP;
  tCanReF0vsImF0_LamKchP = DrawAllReF0vsImF0(kLamKchP, tIncludeResType, tIncludeD0Type, bSaveFigures);

  //-------------
  TCanvas *tCanRadiusvsLambda_LamKchP1, *tCanRadiusvsLambda_LamKchP2, *tCanRadiusvsLambda_LamKchP3;
  if(tCentType != kMB)
  {
    tCanRadiusvsLambda_LamKchP1 = DrawAllRadiusvsLambda(kLamKchP, tCentType, tIncludeResType, tIncludeD0Type, bSaveFigures);
  }
  else
  {
    tCanRadiusvsLambda_LamKchP1 = DrawAllRadiusvsLambda(kLamKchP, k0010, tIncludeResType, tIncludeD0Type, bSaveFigures);
    tCanRadiusvsLambda_LamKchP2 = DrawAllRadiusvsLambda(kLamKchP, k1030, tIncludeResType, tIncludeD0Type, bSaveFigures);
    tCanRadiusvsLambda_LamKchP3 = DrawAllRadiusvsLambda(kLamKchP, k3050, tIncludeResType, tIncludeD0Type, bSaveFigures);
  }

//-------------------------------------------------------------------------------
  TCanvas* tCanReF0vsImF0_LamKchM;
  tCanReF0vsImF0_LamKchM = DrawAllReF0vsImF0(kLamKchM, tIncludeResType, tIncludeD0Type, bSaveFigures);

  //-------------
  TCanvas *tCanRadiusvsLambda_LamKchM1, *tCanRadiusvsLambda_LamKchM2, *tCanRadiusvsLambda_LamKchM3;
  if(tCentType != kMB)
  {
    tCanRadiusvsLambda_LamKchM1 = DrawAllRadiusvsLambda(kLamKchM, tCentType, tIncludeResType, tIncludeD0Type, bSaveFigures);
  }
  else
  {
    tCanRadiusvsLambda_LamKchM1 = DrawAllRadiusvsLambda(kLamKchM, k0010, tIncludeResType, tIncludeD0Type, bSaveFigures);
    tCanRadiusvsLambda_LamKchM2 = DrawAllRadiusvsLambda(kLamKchM, k1030, tIncludeResType, tIncludeD0Type, bSaveFigures);
    tCanRadiusvsLambda_LamKchM3 = DrawAllRadiusvsLambda(kLamKchM, k3050, tIncludeResType, tIncludeD0Type, bSaveFigures);
  }

//-------------------------------------------------------------------------------
  TCanvas* tCanReF0vsImF0_LamK0;
  tCanReF0vsImF0_LamK0 = DrawAllReF0vsImF0(kLamK0, tIncludeResType, tIncludeD0Type, bSaveFigures);

  //-------------
  TCanvas *tCanRadiusvsLambda_LamK01, *tCanRadiusvsLambda_LamK02, *tCanRadiusvsLambda_LamK03;
  if(tCentType != kMB)
  {
    tCanRadiusvsLambda_LamK01 = DrawAllRadiusvsLambda(kLamK0, tCentType, tIncludeResType, tIncludeD0Type, bSaveFigures);
  }
  else
  {
    tCanRadiusvsLambda_LamK01 = DrawAllRadiusvsLambda(kLamK0, k0010, tIncludeResType, tIncludeD0Type, bSaveFigures);
    tCanRadiusvsLambda_LamK02 = DrawAllRadiusvsLambda(kLamK0, k1030, tIncludeResType, tIncludeD0Type, bSaveFigures);
    tCanRadiusvsLambda_LamK03 = DrawAllRadiusvsLambda(kLamK0, k3050, tIncludeResType, tIncludeD0Type, bSaveFigures);
  }
*/
//-------------------------------------------------------------------------------
//*******************************************************************************
//-------------------------------------------------------------------------------
/*
  TCanvas *tCanCompareAllRadiusvsLambdaAcrossAnalyses1, *tCanCompareAllRadiusvsLambdaAcrossAnalyses2, *tCanCompareAllRadiusvsLambdaAcrossAnalyses3;
  if(tCentType != kMB)
  {
    tCanCompareAllRadiusvsLambdaAcrossAnalyses1 = CompareAllRadiusvsLambdaAcrossAnalysesv1(tCentType, tIncludeResType, kFreeAndFixedD0, bSaveFigures);
  }
  else
  {
    tCanCompareAllRadiusvsLambdaAcrossAnalyses1 = CompareAllRadiusvsLambdaAcrossAnalysesv2(k0010, tIncludeResType, kFreeAndFixedD0, bSaveFigures);
    tCanCompareAllRadiusvsLambdaAcrossAnalyses2 = CompareAllRadiusvsLambdaAcrossAnalysesv2(k1030, tIncludeResType, kFreeAndFixedD0, bSaveFigures);
    tCanCompareAllRadiusvsLambdaAcrossAnalyses3 = CompareAllRadiusvsLambdaAcrossAnalysesv2(k3050, tIncludeResType, kFreeAndFixedD0, bSaveFigures);
  }
*/

  TCanvas* tCanCompareAllReF0vsImF0AcrossAnalysesv1 = CompareAllReF0vsImF0AcrossAnalysesv1(tIncludeResType, kFreeAndFixedD0, bSaveFigures);
  TCanvas* tCanCompareAllReF0vsImF0AcrossAnalysesv2 = CompareAllReF0vsImF0AcrossAnalysesv2(tIncludeResType, kFreeAndFixedD0, bSaveFigures);
  TCanvas* tCanCompareAllReF0vsImF0AcrossAnalysesv3 = CompareAllReF0vsImF0AcrossAnalysesv3(tIncludeResType, kFreeAndFixedD0, bSaveFigures);

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
/*
  delete tCanCompareAllRadiusvsLambdaAcrossAnalyses1;
  delete tCanCompareAllRadiusvsLambdaAcrossAnalyses2;
  delete tCanCompareAllRadiusvsLambdaAcrossAnalyses3;
*/
  delete tCanCompareAllReF0vsImF0AcrossAnalysesv1;
  delete tCanCompareAllReF0vsImF0AcrossAnalysesv2;
  delete tCanCompareAllReF0vsImF0AcrossAnalysesv3;

  cout << "DONE" << endl;
  return 0;
}








