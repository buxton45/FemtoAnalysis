#include "TSystem.h"
#include "TLegend.h"

#include "CompareFittingMethods.h"
TString gSaveLocationBase = "/home/jesse/Analysis/Presentations/PWGCF/LamKPaperProposal/ALICE_MiniWeek_20180115/Figures/";
//TString gSaveType = "eps";
TString gSaveType = "pdf";  // must save as pdf for transparency to work

/*
//---------------------------------------------------------------------------------------------------------------------------------
vector<bool> GetIncludePlotsVec(IncludeResType aIncludeResType, IncludeD0Type aIncludeD0Type)
{
  bool bInclude_QM = true;
  //------------------------
  bool bInclude_10and3_Avg_FreeD0 = true;
  bool bInclude_10and3_Avg_FixedD0 = true;
  bool bInclude_10and3_Avg = true;
  //------------------------
  bool bInclude_10_FreeD0 = true;
  bool bInclude_10_FixedD0 = true;
  bool bInclude_10_Avg = true;
  //------------------------
  bool bInclude_3_FreeD0 = true;
  bool bInclude_3_FixedD0 = true;
  bool bInclude_3_Avg = true;
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
  vector<bool> tIncludePlots{bInclude_QM, 
                             bInclude_10and3_Avg_FreeD0, bInclude_10and3_Avg_FixedD0, bInclude_10and3_Avg,
                             bInclude_10_FreeD0,         bInclude_10_FixedD0,         bInclude_10_Avg,
                             bInclude_3_FreeD0,          bInclude_3_FixedD0,          bInclude_3_Avg};
  return tIncludePlots;
}
*/

//---------------------------------------------------------------------------------------------------------------------------------
vector<bool> GetIncludePlotsVec(IncludeResType aIncludeResType, IncludeD0Type aIncludeD0Type, Plot10and3Type aPlot10and3Type=kPlot10and3SeparateAndAvg, bool bIncludeFreeFixedD0Avgs=true)
{
  //According to storage in vector<DrawAcrossAnalysesInfo> tDrawAcrossAnalysesInfoVec (contained in CompareFittingMethods.h)
  //The ordering of the return vector<bool> should be...
  //vector<bool> tIncludePlots{bInclude_QM, 
  //                           bInclude_10and3_Avg_FreeD0, bInclude_10and3_Avg_FixedD0, bInclude_10and3_Avg,
  //                           bInclude_10_FreeD0,         bInclude_10_FixedD0,         bInclude_10_Avg,
  //                           bInclude_3_FreeD0,          bInclude_3_FixedD0,          bInclude_3_Avg};

  bool bInclude_QM = true;  //Always true
  bool bInclude_QM_FixD0 = true;  

  //---------------------------------------------------------
  //Step 1: Build vector<bool> tD0TruthVec(3) for each grouping of [bInclude..._FreeD0, bInclude..._FixedD0, bInclude..._Avg]
  vector<bool> tD0TruthVec;
  if(aIncludeD0Type==kFreeD0Only)       tD0TruthVec = vector<bool>{true,  false, false};
  else if(aIncludeD0Type==kFixedD0Only) tD0TruthVec = vector<bool>{false, true,  false};
  else if(aIncludeD0Type==kFreeAndFixedD0)
  {
    if(bIncludeFreeFixedD0Avgs) tD0TruthVec = vector<bool>{true, true, true};
    else tD0TruthVec = vector<bool>{true, true, false};
  }
  else assert(0);

  assert(tD0TruthVec.size()==3);

  //---------------------------------------------------------
  //Step 2: Build vector<bool> tResTruthVec(3) for grouping of [10and3, 10, 3]
  vector<bool> tResTruthVec;
  if(aIncludeResType==kInclude10ResAnd3Res)
  {
    if(aPlot10and3Type==kPlot10and3SeparateOnly)        tResTruthVec = vector<bool>{false, true,  true};
    else if(aPlot10and3Type==kPlot10and3AvgOnly)        tResTruthVec = vector<bool>{true,  false, false};
    else if(aPlot10and3Type==kPlot10and3SeparateAndAvg) tResTruthVec = vector<bool>{true,  true,  true};
    else assert(0);
  }
  else if(aIncludeResType==kInclude10ResOnly) tResTruthVec = vector<bool>{false, true,  false};
  else if(aIncludeResType==kInclude3ResOnly)  tResTruthVec = vector<bool>{false, false, true};
  else if(aIncludeResType==kIncludeNoRes)     tResTruthVec = vector<bool>{false, false, false};
  else assert(0);

  assert(tResTruthVec.size()==3);

  //---------------------------------------------------------
  //Step 3: Combine tD0TruthVec and tResTruthVec
  vector<bool> tIncludePlots;
  if(aIncludeD0Type == kFixedD0Only) bInclude_QM = false;
  if(aIncludeD0Type == kFreeD0Only) bInclude_QM_FixD0 = false;

  tIncludePlots.push_back(bInclude_QM);
  tIncludePlots.push_back(bInclude_QM_FixD0);


  for(unsigned int i=0; i<tResTruthVec.size(); i++)
  {
    if(!tResTruthVec[i]) tIncludePlots.insert(tIncludePlots.end(), 3, false);
    else tIncludePlots.insert(tIncludePlots.end(), tD0TruthVec.begin(), tD0TruthVec.end());
  }

  assert(tIncludePlots.size() == tDrawAcrossAnalysesInfoVec.size());
  return tIncludePlots;
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

  TString tDrawOption;
  int tColor;
  TGraphAsymmErrors* tGr = GetReF0vsImF0(aFitInfo, aErrType);
  tGr->SetName(aFitInfo.descriptor + cErrorTypeTags[aErrType]);
  if(aErrType==kStat)
  {
    tColor = aFitInfo.markerColor;
    tGr->SetLineWidth(1);
    tDrawOption = TString("pzsame");
  }
  else if(aErrType==kSys)
  {
    tColor = TColor::GetColorTransparent(aFitInfo.markerColor, 0.3);
    tGr->SetLineWidth(0);
    tDrawOption = TString("e2same");
  }
  else assert(0);

  tGr->SetMarkerStyle(aFitInfo.markerStyle);
  tGr->SetMarkerColor(tColor);
  tGr->SetFillColor(tColor);
  tGr->SetFillStyle(1000);
  tGr->SetLineColor(tColor);

  tGr->Draw(tDrawOption);
}

//---------------------------------------------------------------------------------------------------------------------------------
void DrawD0(TPad* aPad, FitInfo &aFitInfo, ErrorType aErrType=kStat, double aXOffset=0.5)
{
  aPad->cd();
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  TString tDrawOption;
  int tColor;

  TGraphAsymmErrors* tGr = GetD0(aFitInfo, aErrType, aXOffset);
  tGr->SetName(aFitInfo.descriptor + cErrorTypeTags[aErrType]);

  double tX=0., tY=0;
  tGr->GetPoint(0, tX, tY);
  if(tY != 0.)
  {
    if(aErrType==kStat)
    {
      tColor = aFitInfo.markerColor;
      tGr->SetLineWidth(1);
      tDrawOption = TString("pzsame");
    }
    else if(aErrType==kSys)
    {
      tColor = TColor::GetColorTransparent(aFitInfo.markerColor, 0.3);
      tGr->SetLineWidth(0);
      tDrawOption = TString("e2same");
    }
    else assert(0);

    tGr->SetMarkerStyle(aFitInfo.markerStyle);
    tGr->SetMarkerColor(tColor);
    tGr->SetFillColor(tColor);
    tGr->SetFillStyle(1000);
    tGr->SetLineColor(tColor);

    tGr->Draw(tDrawOption);
  }
}

//---------------------------------------------------------------------------------------------------------------------------------
void DrawRadiusvsLambda(TPad* aPad, FitInfo &aFitInfo, CentralityType aCentType=k0010, ErrorType aErrType=kStat)
{
  aPad->cd();
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  TString tDrawOption;
  int tColor;

  TGraphAsymmErrors* tGr = GetRadiusvsLambda(aFitInfo, aCentType, aErrType);
  tGr->SetName(aFitInfo.descriptor + cErrorTypeTags[aErrType]);
  if(aErrType==kStat)
  {
    tColor = aFitInfo.markerColor;
    tGr->SetLineWidth(1);
    tDrawOption = TString("pzsame");
  }
  else if(aErrType==kSys)
  {
    tColor = TColor::GetColorTransparent(aFitInfo.markerColor, 0.3);
    tGr->SetLineWidth(0);
    tDrawOption = TString("e2same");
  }
  else assert(0);

  tGr->SetMarkerStyle(aFitInfo.markerStyle);
  tGr->SetMarkerColor(tColor);
  tGr->SetFillColor(tColor);
  tGr->SetFillStyle(1000);
  tGr->SetLineColor(tColor);

  tGr->Draw(tDrawOption);
}

//---------------------------------------------------------------------------------------------------------------------------------
TGraphAsymmErrors* GetWeightedMeanReF0vsImF0(vector<FitInfo> &aFitInfoVec, IncludeResType aIncludeResType=kInclude10ResAnd3Res, IncludeD0Type aIncludeD0Type=kFreeAndFixedD0)
{
  td2dVec tReF0Vec;
  td2dVec tImF0Vec;

  for(unsigned int i=0; i<aFitInfoVec.size(); i++)
  {
    if(!IncludeFitInfoInMeanCalculation(aFitInfoVec[i], aIncludeResType, aIncludeD0Type)) continue;

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
    if(!IncludeFitInfoInMeanCalculation(aFitInfoVec[i], aIncludeResType, aIncludeD0Type)) continue;

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
    if(!IncludeFitInfoInMeanCalculation(aFitInfoVec[i], aIncludeResType, aIncludeD0Type)) continue;

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
TCanvas* DrawAllReF0vsImF0(AnalysisType aAnType, 
                           IncludeResType aIncludeResType=kInclude10ResAnd3Res, IncludeD0Type aIncludeD0Type=kFreeAndFixedD0,
                           IncludeRadiiType aIncludeRadiiType=kFreeAndFixedRadii, IncludeLambdaType aIncludeLambdaType=kFreeAndFixedLambda, 
                           ErrorType aErrType=kStatAndSys, bool bSaveImage=false)
{
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
    if(aErrType==kStatAndSys)
    {
      DrawReF0vsImF0((TPad*)tPadReF0vsImF0, aFitInfoVec[i], kSys);
      DrawReF0vsImF0((TPad*)tPadReF0vsImF0, aFitInfoVec[i], kStat);

      DrawD0((TPad*)tPadD0, aFitInfoVec[i], kSys);
      DrawD0((TPad*)tPadD0, aFitInfoVec[i], kStat);
    }
    else 
    {
      DrawReF0vsImF0((TPad*)tPadReF0vsImF0, aFitInfoVec[i], aErrType);
      DrawD0((TPad*)tPadD0, aFitInfoVec[i], aErrType);
    }

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

    gSystem->mkdir(gSaveLocationBase.Data());
    tSaveLocationFull_ReF0vsImF0 = gSaveLocationBase + TString::Format("%s/ReF0vsImF0%s.%s", cAnalysisBaseTags[aAnType], cIncludeResTypeTags[aIncludeResType], gSaveType.Data());
    tReturnCan->SaveAs(tSaveLocationFull_ReF0vsImF0);
  }

  return tReturnCan;
}


//TODO For use in CompareAllReF0vsImF0AcrossAnalyses!!!
//---------------------------------------------------------------------------------------------------------------------------------
void DrawAllReF0vsImF0(TPad* aPadReF0vsImF0, TPad* aPadD0, 
                       AnalysisType aAnType, 
                       IncludeResType aIncludeResType=kInclude10ResAnd3Res, IncludeD0Type aIncludeD0Type=kFreeAndFixedD0,
                       IncludeRadiiType aIncludeRadiiType=kFreeAndFixedRadii, IncludeLambdaType aIncludeLambdaType=kFreeAndFixedLambda, 
                       ErrorType aErrType=kStatAndSys, double aD0XOffset=0.5, double aD0XOffsetIncrement=0.)
{
  //------------------------
  vector<FitInfo> aFitInfoVec = GetFitInfoVec(aAnType, aIncludeResType, aIncludeD0Type, aIncludeRadiiType, aIncludeLambdaType);
  //------------------------

  TLatex* tTex = new TLatex();
  tTex->SetTextAlign(12);
  tTex->SetLineWidth(2);
  tTex->SetTextFont(42);
  tTex->SetTextSize(0.04);

  double tStartX = 0.30;
  double tStartY = 1.4;

  if(aAnType == kLamKchM) tStartX = -1.75;

  double tIncrementY = 0.08;

  int iD0Inc=0;
  TString tDescriptorFull, tDescriptor;
  int tDescriptorEnd = 0;
  for(unsigned int i=0; i<aFitInfoVec.size(); i++)
  {
    if(aErrType==kStatAndSys)
    {
      DrawReF0vsImF0((TPad*)aPadReF0vsImF0, aFitInfoVec[i], kSys);
      DrawReF0vsImF0((TPad*)aPadReF0vsImF0, aFitInfoVec[i], kStat);

      DrawD0((TPad*)aPadD0, aFitInfoVec[i], kSys, aD0XOffset+iD0Inc*aD0XOffsetIncrement);
      DrawD0((TPad*)aPadD0, aFitInfoVec[i], kStat, aD0XOffset+iD0Inc*aD0XOffsetIncrement);
    }
    else 
    {
      DrawReF0vsImF0((TPad*)aPadReF0vsImF0, aFitInfoVec[i], aErrType);
      DrawD0((TPad*)aPadD0, aFitInfoVec[i], aErrType, aD0XOffset+iD0Inc*aD0XOffsetIncrement);
    }
    if(aFitInfoVec[i].d0 != 0.) iD0Inc++;
  }
}


//---------------------------------------------------------------------------------------------------------------------------------
TCanvas* DrawAllRadiusvsLambda(AnalysisType aAnType, CentralityType aCentType=k0010, 
                               IncludeResType aIncludeResType=kInclude10ResAnd3Res, IncludeD0Type aIncludeD0Type=kFreeAndFixedD0, 
                               IncludeRadiiType aIncludeRadiiType=kFreeAndFixedRadii, IncludeLambdaType aIncludeLambdaType=kFreeAndFixedLambda, 
                               ErrorType aErrType=kStatAndSys, bool bSaveImage=false)
{
  TCanvas* tReturnCan = new TCanvas(TString::Format("tCanRadiusvsLambda_%s%s", cAnalysisBaseTags[aAnType], cCentralityTags[aCentType]), 
                                    TString::Format("tCanRadiusvsLambda_%s%s", cAnalysisBaseTags[aAnType], cCentralityTags[aCentType]));
  tReturnCan->cd();
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  SetupRadiusvsLambdaAxes((TPad*)tReturnCan);
  //------------------------
  vector<FitInfo> aFitInfoVec = GetFitInfoVec(aAnType, aIncludeResType, aIncludeD0Type, aIncludeRadiiType, aIncludeLambdaType);
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
    if(aErrType==kStatAndSys)
    {
      DrawRadiusvsLambda((TPad*)tReturnCan, aFitInfoVec[i], aCentType, kSys);
      DrawRadiusvsLambda((TPad*)tReturnCan, aFitInfoVec[i], aCentType, kStat);
    }
    else DrawRadiusvsLambda((TPad*)tReturnCan, aFitInfoVec[i], aCentType, aErrType);

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

    gSystem->mkdir(gSaveLocationBase.Data());
    tSaveLocationFull_RadiusvsLambda = gSaveLocationBase + TString::Format("%s/RadiusvsLambda%s%s.%s", cAnalysisBaseTags[aAnType], cCentralityTags[aCentType], cIncludeResTypeTags[aIncludeResType], gSaveType.Data());
    tReturnCan->SaveAs(tSaveLocationFull_RadiusvsLambda);
  }

  return tReturnCan;
}

//---------------------------------------------------------------------------------------------------------------------------------
void DrawRadiusvsLambdaAcrossAnalyses(TPad* aPad, int aMarkerStyle=20, double aMarkerSize=1., CentralityType aCentType=k0010, IncludeResType aIncludeResType=kInclude10ResAnd3Res, IncludeD0Type aIncludeD0Type=kFreeAndFixedD0, ErrorType aErrType=kStat)
{
  aPad->cd();
  //------------------------
  vector<FitInfo> aFitInfoVec_LamKchP = GetFitInfoVec(kLamKchP, aIncludeResType, aIncludeD0Type, kFreeRadiiOnly, kFreeLambdaOnly);
  vector<FitInfo> aFitInfoVec_LamKchM = GetFitInfoVec(kLamKchM, aIncludeResType, aIncludeD0Type, kFreeRadiiOnly, kFreeLambdaOnly);
  vector<FitInfo> aFitInfoVec_LamK0 = GetFitInfoVec(kLamK0, aIncludeResType, aIncludeD0Type, kFreeRadiiOnly, kFreeLambdaOnly);

  Color_t tColor_LamKchP = kRed+1;
  Color_t tColor_LamKchM = kBlue+1;
  Color_t tColor_LamK0 = kBlack;

  if(aErrType==kSys)
  {
    tColor_LamKchP = TColor::GetColorTransparent(tColor_LamKchP, 0.3);
    tColor_LamKchM = TColor::GetColorTransparent(tColor_LamKchM, 0.3);
    tColor_LamK0 = TColor::GetColorTransparent(tColor_LamK0, 0.3);
  }

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

    tGr_LamKchP = GetRadiusvsLambda(aFitInfoVec_LamKchP[0], aCentType, aErrType);
    tGr_LamKchM = GetRadiusvsLambda(aFitInfoVec_LamKchM[0], aCentType, aErrType);
    tGr_LamK0   = GetRadiusvsLambda(aFitInfoVec_LamK0[0], aCentType, aErrType);
  }

  TString tDrawOption = "epsame";

  tGr_LamKchP->SetMarkerColor(tColor_LamKchP);
  tGr_LamKchP->SetMarkerStyle(aMarkerStyle);
  tGr_LamKchP->SetMarkerSize(aMarkerSize);
  tGr_LamKchP->SetFillColor(tColor_LamKchP);
  tGr_LamKchP->SetFillStyle(1000);
  tGr_LamKchP->SetLineColor(tColor_LamKchP);


  tGr_LamKchM->SetMarkerColor(tColor_LamKchM);
  tGr_LamKchM->SetMarkerStyle(aMarkerStyle);
  tGr_LamKchM->SetMarkerSize(aMarkerSize);
  tGr_LamKchM->SetFillColor(tColor_LamKchM);
  tGr_LamKchM->SetFillStyle(1000);
  tGr_LamKchM->SetLineColor(tColor_LamKchM);


  tGr_LamK0->SetMarkerColor(tColor_LamK0);
  tGr_LamK0->SetMarkerStyle(aMarkerStyle);
  tGr_LamK0->SetMarkerSize(aMarkerSize);
  tGr_LamK0->SetFillColor(tColor_LamK0);
  tGr_LamK0->SetFillStyle(1000);
  tGr_LamK0->SetLineColor(tColor_LamK0);

  if(aErrType==kStat)
  {
    tDrawOption = ("pzsame");

    tGr_LamKchP->SetLineWidth(1);
    tGr_LamKchM->SetLineWidth(1);
    tGr_LamK0->SetLineWidth(1);
  }
  else if(aErrType==kSys)
  {
    tDrawOption = ("e2same");

    tGr_LamKchP->SetLineWidth(0);
    tGr_LamKchM->SetLineWidth(0);
    tGr_LamK0->SetLineWidth(0);
  }
  else assert(0);
  //------------------------
  tGr_LamKchP->Draw(tDrawOption);
  tGr_LamKchM->Draw(tDrawOption);
  tGr_LamK0->Draw(tDrawOption);

}




//---------------------------------------------------------------------------------------------------------------------------------
void DrawRadiusvsLambdaAcrossAnalysesQMResults(TPad* aPad, int aMarkerStyle=20, double aMarkerSize=1., CentralityType aCentType=k0010, ErrorType aErrType=kStat, IncludeD0Type tIncludeD0Type=kFreeD0Only)
{
  aPad->cd();
  //------------------------

  Color_t tColor_LamKchP = kRed+1;
  Color_t tColor_LamKchM = kBlue+1;
  Color_t tColor_LamK0 = kBlack;

  if(aErrType==kSys)
  {
    tColor_LamKchP = TColor::GetColorTransparent(tColor_LamKchP, 0.3);
    tColor_LamKchM = TColor::GetColorTransparent(tColor_LamKchM, 0.3);
    tColor_LamK0 = TColor::GetColorTransparent(tColor_LamK0, 0.3);
  }

  //------------------------

  TGraphAsymmErrors *tGr_LamKchP, *tGr_LamKchM, *tGr_LamK0;
  if(tIncludeD0Type==kFreeD0Only)
  {
    tGr_LamKchP = GetRadiusvsLambda(tFitInfoQM_LamKchP, aCentType, aErrType);
    tGr_LamKchM = GetRadiusvsLambda(tFitInfoQM_LamKchM, aCentType, aErrType);
    tGr_LamK0   = GetRadiusvsLambda(tFitInfoQM_LamK0, aCentType, aErrType);
  }
  if(tIncludeD0Type==kFixedD0Only)
  {
    tGr_LamKchP = GetRadiusvsLambda(tFitInfoQM_LamKchP_FixD0, aCentType, aErrType);
    tGr_LamKchM = GetRadiusvsLambda(tFitInfoQM_LamKchM_FixD0, aCentType, aErrType);
    tGr_LamK0   = GetRadiusvsLambda(tFitInfoQM_LamK0_FixD0, aCentType, aErrType);
  }
  //------------------------

  TString tDrawOption = "epsame";

  tGr_LamKchP->SetMarkerColor(tColor_LamKchP);
  tGr_LamKchP->SetMarkerStyle(aMarkerStyle);
  tGr_LamKchP->SetMarkerSize(aMarkerSize);
  tGr_LamKchP->SetFillColor(tColor_LamKchP);
  tGr_LamKchP->SetFillStyle(1000);
  tGr_LamKchP->SetLineColor(tColor_LamKchP);


  tGr_LamKchM->SetMarkerColor(tColor_LamKchM);
  tGr_LamKchM->SetMarkerStyle(aMarkerStyle);
  tGr_LamKchM->SetMarkerSize(aMarkerSize);
  tGr_LamKchM->SetFillColor(tColor_LamKchM);
  tGr_LamKchM->SetFillStyle(1000);
  tGr_LamKchM->SetLineColor(tColor_LamKchM);


  tGr_LamK0->SetMarkerColor(tColor_LamK0);
  tGr_LamK0->SetMarkerStyle(aMarkerStyle);
  tGr_LamK0->SetMarkerSize(aMarkerSize);
  tGr_LamK0->SetFillColor(tColor_LamK0);
  tGr_LamK0->SetFillStyle(1000);
  tGr_LamK0->SetLineColor(tColor_LamK0);

  if(aErrType==kStat)
  {
    tDrawOption = ("pzsame");

    tGr_LamKchP->SetLineWidth(1);
    tGr_LamKchM->SetLineWidth(1);
    tGr_LamK0->SetLineWidth(1);
  }
  else if(aErrType==kSys)
  {
    tDrawOption = ("e2same");

    tGr_LamKchP->SetLineWidth(0);
    tGr_LamKchM->SetLineWidth(0);
    tGr_LamK0->SetLineWidth(0);
  }
  else assert(0);
  //------------------------
  tGr_LamKchP->Draw(tDrawOption);
  tGr_LamKchM->Draw(tDrawOption);
  tGr_LamK0->Draw(tDrawOption);
}


//---------------------------------------------------------------------------------------------------------------------------------
TCanvas* CompareAllRadiusvsLambdaAcrossAnalyses(CentralityType aCentType, IncludeResType aIncludeResType=kInclude10ResAnd3Res, IncludeD0Type aIncludeD0Type=kFreeAndFixedD0, Plot10and3Type aPlot10and3Type=kPlot10and3SeparateAndAvg, bool aIncludeFreeFixedD0Avgs=true, ErrorType aErrType=kStatAndSys, bool bSaveImage=false)
{
  vector<bool> tIncludePlots = GetIncludePlotsVec(aIncludeResType, aIncludeD0Type, aPlot10and3Type, aIncludeFreeFixedD0Avgs);
  assert(tIncludePlots.size() == tDrawAcrossAnalysesInfoVec.size());
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

  int tNPlots = 0;
  for(unsigned int i=0; i<tIncludePlots.size(); i++) if(tIncludePlots[i]) tNPlots++;

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

  IncludeResType tIncResType;
  IncludeD0Type tIncD0Type;
  int tMarkerStyle;
  double tMarkerSize;
  TString tDescriptor;

  int iTex = 0;
  for(unsigned int i=0; i<tIncludePlots.size(); i++)
  {
    tIncResType = tDrawAcrossAnalysesInfoVec[i].incResType;
    tIncD0Type = tDrawAcrossAnalysesInfoVec[i].incD0Type;
    tMarkerStyle = tDrawAcrossAnalysesInfoVec[i].markerStyle;
    tMarkerSize = tDrawAcrossAnalysesInfoVec[i].markerSize;
    tDescriptor = tDrawAcrossAnalysesInfoVec[i].descriptor;

    if(tIncludePlots[i])
    {
      if(tIncResType==kIncludeNoRes) 
      {
        if(aErrType==kStatAndSys)
        {
          if(aIncludeD0Type==kFreeAndFixedD0)
          {
            DrawRadiusvsLambdaAcrossAnalysesQMResults((TPad*)tReturnCan, 20, tMarkerSize, aCentType, kSys, kFreeD0Only);
            DrawRadiusvsLambdaAcrossAnalysesQMResults((TPad*)tReturnCan, 20, tMarkerSize, aCentType, kStat, kFreeD0Only);

            DrawRadiusvsLambdaAcrossAnalysesQMResults((TPad*)tReturnCan, 24, tMarkerSize, aCentType, kSys, kFixedD0Only);
            DrawRadiusvsLambdaAcrossAnalysesQMResults((TPad*)tReturnCan, 24, tMarkerSize, aCentType, kStat, kFixedD0Only);
          }
          else
          {
            DrawRadiusvsLambdaAcrossAnalysesQMResults((TPad*)tReturnCan, tMarkerStyle, tMarkerSize, aCentType, kSys, aIncludeD0Type);
            DrawRadiusvsLambdaAcrossAnalysesQMResults((TPad*)tReturnCan, tMarkerStyle, tMarkerSize, aCentType, kStat, aIncludeD0Type);
          }
        }
        else 
        {
          if(aIncludeD0Type==kFreeAndFixedD0)
          {
            DrawRadiusvsLambdaAcrossAnalysesQMResults((TPad*)tReturnCan, 20, tMarkerSize, aCentType, aErrType, kFreeD0Only);  //Signifies QM results
            DrawRadiusvsLambdaAcrossAnalysesQMResults((TPad*)tReturnCan, 24, tMarkerSize, aCentType, aErrType, kFixedD0Only);  //Signifies QM results

          }
          else DrawRadiusvsLambdaAcrossAnalysesQMResults((TPad*)tReturnCan, tMarkerStyle, tMarkerSize, aCentType, aErrType, aIncludeD0Type);  //Signifies QM results
        }
      }
      else 
      {
        if(aErrType==kStatAndSys)
        {
          DrawRadiusvsLambdaAcrossAnalyses((TPad*)tReturnCan, tMarkerStyle, tMarkerSize, aCentType, tIncResType, tIncD0Type, kSys);
          DrawRadiusvsLambdaAcrossAnalyses((TPad*)tReturnCan, tMarkerStyle, tMarkerSize, aCentType, tIncResType, tIncD0Type, kStat);
        }
        else DrawRadiusvsLambdaAcrossAnalyses((TPad*)tReturnCan, tMarkerStyle, tMarkerSize, aCentType, tIncResType, tIncD0Type, aErrType);
      }

      if(tDescriptor.Contains("d_{0}")) tTex->DrawLatex(tStartX, tStartY-iTex*tIncrementY-0.1*tIncrementY, tDescriptor);
      else tTex->DrawLatex(tStartX, tStartY-iTex*tIncrementY, tDescriptor);
      tMarker->SetMarkerStyle(tMarkerStyle);
      tMarker->DrawMarker(tStartX-tIncrementX, tStartY-iTex*tIncrementY);
      iTex++;
    }
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
void DrawReF0vsImF0AcrossAnalyses(TPad* aPad, int aMarkerStyle=20, double aMarkerSize=1., IncludeResType aIncludeResType=kInclude10ResAnd3Res, IncludeD0Type aIncludeD0Type=kFreeAndFixedD0, ErrorType aErrType=kStat)
{
  aPad->cd();
  //------------------------
  vector<FitInfo> aFitInfoVec_LamKchP = GetFitInfoVec(kLamKchP, aIncludeResType, aIncludeD0Type, kFreeRadiiOnly, kFreeLambdaOnly);
  vector<FitInfo> aFitInfoVec_LamKchM = GetFitInfoVec(kLamKchM, aIncludeResType, aIncludeD0Type, kFreeRadiiOnly, kFreeLambdaOnly);
  vector<FitInfo> aFitInfoVec_LamK0 = GetFitInfoVec(kLamK0, aIncludeResType, aIncludeD0Type, kFreeRadiiOnly, kFreeLambdaOnly);

  Color_t tColor_LamKchP = kRed+1;
  Color_t tColor_LamKchM = kBlue+1;
  Color_t tColor_LamK0 = kBlack;

  if(aErrType==kSys)
  {
    tColor_LamKchP = TColor::GetColorTransparent(tColor_LamKchP, 0.3);
    tColor_LamKchM = TColor::GetColorTransparent(tColor_LamKchM, 0.3);
    tColor_LamK0 = TColor::GetColorTransparent(tColor_LamK0, 0.3);
  }

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

    tGr_LamKchP = GetReF0vsImF0(aFitInfoVec_LamKchP[0], aErrType);
    tGr_LamKchM = GetReF0vsImF0(aFitInfoVec_LamKchM[0], aErrType);
    tGr_LamK0   = GetReF0vsImF0(aFitInfoVec_LamK0[0], aErrType);
  }

  TString tDrawOption = "epsame";

  tGr_LamKchP->SetMarkerStyle(aMarkerStyle);
  tGr_LamKchP->SetMarkerSize(aMarkerSize);
  tGr_LamKchP->SetMarkerColor(tColor_LamKchP);
  tGr_LamKchP->SetFillColor(tColor_LamKchP);
  tGr_LamKchP->SetLineColor(tColor_LamKchP);
  tGr_LamKchP->SetFillStyle(1000);

  tGr_LamKchM->SetMarkerStyle(aMarkerStyle);
  tGr_LamKchM->SetMarkerSize(aMarkerSize);
  tGr_LamKchM->SetMarkerColor(tColor_LamKchM);
  tGr_LamKchM->SetFillColor(tColor_LamKchM);
  tGr_LamKchM->SetLineColor(tColor_LamKchM);
  tGr_LamKchM->SetFillStyle(1000);

  tGr_LamK0->SetMarkerStyle(aMarkerStyle);
  tGr_LamK0->SetMarkerSize(aMarkerSize);
  tGr_LamK0->SetMarkerColor(tColor_LamK0);
  tGr_LamK0->SetFillColor(tColor_LamK0);
  tGr_LamK0->SetLineColor(tColor_LamK0);
  tGr_LamK0->SetFillStyle(1000);

  if(aErrType==kStat)
  {
    tDrawOption = ("pzsame");

    tGr_LamKchP->SetLineWidth(1);
    tGr_LamKchM->SetLineWidth(1);
    tGr_LamK0->SetLineWidth(1);
  }
  else if(aErrType==kSys)
  {
    tDrawOption = ("e2same");

    tGr_LamKchP->SetLineWidth(0);
    tGr_LamKchM->SetLineWidth(0);
    tGr_LamK0->SetLineWidth(0);
  }
  else assert(0);
  
  //------------------------
  tGr_LamKchP->Draw(tDrawOption);
  tGr_LamKchM->Draw(tDrawOption);
  tGr_LamK0->Draw(tDrawOption);

  //------------------------------------------------------

}

//---------------------------------------------------------------------------------------------------------------------------------
void DrawReF0vsImF0AcrossAnalysesQMResults(TPad* aPad, int aMarkerStyle=20, double aMarkerSize=1., ErrorType aErrType=kStat, IncludeD0Type tIncludeD0Type=kFreeD0Only)
{
  aPad->cd();
  //------------------------
  Color_t tColor_LamKchP = kRed+1;
  Color_t tColor_LamKchM = kBlue+1;
  Color_t tColor_LamK0 = kBlack;

  if(aErrType==kSys)
  {
    tColor_LamKchP = TColor::GetColorTransparent(tColor_LamKchP, 0.3);
    tColor_LamKchM = TColor::GetColorTransparent(tColor_LamKchM, 0.3);
    tColor_LamK0 = TColor::GetColorTransparent(tColor_LamK0, 0.3);
  }

  //------------------------
  TGraphAsymmErrors *tGr_LamKchP, *tGr_LamKchM, *tGr_LamK0;
  if(tIncludeD0Type==kFreeD0Only)
  {
    tGr_LamKchP = GetReF0vsImF0(tFitInfoQM_LamKchP, aErrType);
    tGr_LamKchM = GetReF0vsImF0(tFitInfoQM_LamKchM, aErrType);
    tGr_LamK0   = GetReF0vsImF0(tFitInfoQM_LamK0, aErrType);
  }
  if(tIncludeD0Type==kFixedD0Only)
  {
    tGr_LamKchP = GetReF0vsImF0(tFitInfoQM_LamKchP_FixD0, aErrType);
    tGr_LamKchM = GetReF0vsImF0(tFitInfoQM_LamKchM_FixD0, aErrType);
    tGr_LamK0   = GetReF0vsImF0(tFitInfoQM_LamK0_FixD0, aErrType);
  }

  TString tDrawOption = "epsame";
  //------------------------
  tGr_LamKchP->SetMarkerColor(tColor_LamKchP);
  tGr_LamKchP->SetMarkerStyle(aMarkerStyle);
  tGr_LamKchP->SetMarkerSize(aMarkerSize);
  tGr_LamKchP->SetFillColor(tColor_LamKchP);
  tGr_LamKchP->SetFillStyle(1000);
  tGr_LamKchP->SetLineColor(tColor_LamKchP);


  tGr_LamKchM->SetMarkerColor(tColor_LamKchM);
  tGr_LamKchM->SetMarkerStyle(aMarkerStyle);
  tGr_LamKchM->SetMarkerSize(aMarkerSize);
  tGr_LamKchM->SetFillColor(tColor_LamKchM);
  tGr_LamKchM->SetFillStyle(1000);
  tGr_LamKchM->SetLineColor(tColor_LamKchM);


  tGr_LamK0->SetMarkerColor(tColor_LamK0);
  tGr_LamK0->SetMarkerStyle(aMarkerStyle);
  tGr_LamK0->SetMarkerSize(aMarkerSize);
  tGr_LamK0->SetFillColor(tColor_LamK0);
  tGr_LamK0->SetFillStyle(1000);
  tGr_LamK0->SetLineColor(tColor_LamK0);

  if(aErrType==kStat)
  {
    tDrawOption = ("pzsame");

    tGr_LamKchP->SetLineWidth(1);
    tGr_LamKchM->SetLineWidth(1);
    tGr_LamK0->SetLineWidth(1);
  }
  else if(aErrType==kSys)
  {
    tDrawOption = ("e2same");

    tGr_LamKchP->SetLineWidth(0);
    tGr_LamKchM->SetLineWidth(0);
    tGr_LamK0->SetLineWidth(0);
  }
  else assert(0);

  //------------------------
  tGr_LamKchP->Draw(tDrawOption);
  tGr_LamKchM->Draw(tDrawOption);
  tGr_LamK0->Draw(tDrawOption);

  //------------------------------------------------------

}


//---------------------------------------------------------------------------------------------------------------------------------
void DrawD0AcrossAnalyses(TPad* aPad, int aMarkerStyle=20, double aMarkerSize=1., IncludeResType aIncludeResType=kInclude10ResAnd3Res, IncludeD0Type aIncludeD0Type=kFreeAndFixedD0, ErrorType aErrType=kStat, double aXOffset=0.5)
{
  aPad->cd();
  //------------------------
  vector<FitInfo> aFitInfoVec_LamKchP = GetFitInfoVec(kLamKchP, aIncludeResType, aIncludeD0Type, kFreeRadiiOnly, kFreeLambdaOnly);
  vector<FitInfo> aFitInfoVec_LamKchM = GetFitInfoVec(kLamKchM, aIncludeResType, aIncludeD0Type, kFreeRadiiOnly, kFreeLambdaOnly);
  vector<FitInfo> aFitInfoVec_LamK0 = GetFitInfoVec(kLamK0, aIncludeResType, aIncludeD0Type, kFreeRadiiOnly, kFreeLambdaOnly);

  Color_t tColor_LamKchP = kRed+1;
  Color_t tColor_LamKchM = kBlue+1;
  Color_t tColor_LamK0 = kBlack;

  if(aErrType==kSys)
  {
    tColor_LamKchP = TColor::GetColorTransparent(tColor_LamKchP, 0.3);
    tColor_LamKchM = TColor::GetColorTransparent(tColor_LamKchM, 0.3);
    tColor_LamK0 = TColor::GetColorTransparent(tColor_LamK0, 0.3);
  }

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

    tGr_LamKchP = GetD0(aFitInfoVec_LamKchP[0], aErrType, aXOffset);
    tGr_LamKchM = GetD0(aFitInfoVec_LamKchM[0], aErrType, aXOffset);
    tGr_LamK0   = GetD0(aFitInfoVec_LamK0[0], aErrType, aXOffset);
  }

  double tX_LamKchP=0., tY_LamKchP=0.;
  double tX_LamKchM=0., tY_LamKchM=0.;
  double tX_LamK0=0., tY_LamK0=0.;

  tGr_LamKchP->GetPoint(0, tX_LamKchP, tY_LamKchP);
  tGr_LamKchM->GetPoint(0, tX_LamKchM, tY_LamKchM);
  tGr_LamK0->GetPoint(0, tX_LamK0, tY_LamK0);

  TString tDrawOption = "epsame";

  tGr_LamKchP->SetMarkerColor(tColor_LamKchP);
  tGr_LamKchP->SetMarkerStyle(aMarkerStyle);
  tGr_LamKchP->SetMarkerSize(aMarkerSize);
  tGr_LamKchP->SetFillColor(tColor_LamKchP);
  tGr_LamKchP->SetFillStyle(1000);
  tGr_LamKchP->SetLineColor(tColor_LamKchP);


  tGr_LamKchM->SetMarkerColor(tColor_LamKchM);
  tGr_LamKchM->SetMarkerStyle(aMarkerStyle);
  tGr_LamKchM->SetMarkerSize(aMarkerSize);
  tGr_LamKchM->SetFillColor(tColor_LamKchM);
  tGr_LamKchM->SetFillStyle(1000);
  tGr_LamKchM->SetLineColor(tColor_LamKchM);


  tGr_LamK0->SetMarkerColor(tColor_LamK0);
  tGr_LamK0->SetMarkerStyle(aMarkerStyle);
  tGr_LamK0->SetMarkerSize(aMarkerSize);
  tGr_LamK0->SetFillColor(tColor_LamK0);
  tGr_LamK0->SetFillStyle(1000);
  tGr_LamK0->SetLineColor(tColor_LamK0);

  if(aErrType==kStat)
  {
    tDrawOption = ("pzsame");

    tGr_LamKchP->SetLineWidth(1);
    tGr_LamKchM->SetLineWidth(1);
    tGr_LamK0->SetLineWidth(1);
  }
  else if(aErrType==kSys)
  {
    tDrawOption = ("e2same");

    tGr_LamKchP->SetLineWidth(0);
    tGr_LamKchM->SetLineWidth(0);
    tGr_LamK0->SetLineWidth(0);
  }
  else assert(0);
  //------------------------
  if(tY_LamKchP != 0.) tGr_LamKchP->Draw(tDrawOption);
  if(tY_LamKchM != 0.) tGr_LamKchM->Draw(tDrawOption);
  if(tY_LamK0 != 0.) tGr_LamK0->Draw(tDrawOption);

  //------------------------------------------------------

}

//---------------------------------------------------------------------------------------------------------------------------------
void DrawD0AcrossAnalysesQMResults(TPad* aPad, int aMarkerStyle=20, double aMarkerSize=1., ErrorType aErrType=kStat, IncludeD0Type tIncludeD0Type=kFreeD0Only, double aXOffset=0.5)
{
  aPad->cd();
  //------------------------
  Color_t tColor_LamKchP = kRed+1;
  Color_t tColor_LamKchM = kBlue+1;
  Color_t tColor_LamK0 = kBlack;

  if(aErrType==kSys)
  {
    tColor_LamKchP = TColor::GetColorTransparent(tColor_LamKchP, 0.3);
    tColor_LamKchM = TColor::GetColorTransparent(tColor_LamKchM, 0.3);
    tColor_LamK0 = TColor::GetColorTransparent(tColor_LamK0, 0.3);
  }

  //------------------------
  TGraphAsymmErrors *tGr_LamKchP, *tGr_LamKchM, *tGr_LamK0;
  if(tIncludeD0Type==kFreeD0Only)
  {
    tGr_LamKchP = GetD0(tFitInfoQM_LamKchP, aErrType, aXOffset);
    tGr_LamKchM = GetD0(tFitInfoQM_LamKchM, aErrType, aXOffset);
    tGr_LamK0   = GetD0(tFitInfoQM_LamK0, aErrType, aXOffset);
  }
  if(tIncludeD0Type==kFixedD0Only)
  {
    tGr_LamKchP = GetD0(tFitInfoQM_LamKchP_FixD0, aErrType, aXOffset);
    tGr_LamKchM = GetD0(tFitInfoQM_LamKchM_FixD0, aErrType, aXOffset);
    tGr_LamK0   = GetD0(tFitInfoQM_LamK0_FixD0, aErrType, aXOffset);
  }


  double tX_LamKchP=0., tY_LamKchP=0.;
  double tX_LamKchM=0., tY_LamKchM=0.;
  double tX_LamK0=0., tY_LamK0=0.;

  tGr_LamKchP->GetPoint(0, tX_LamKchP, tY_LamKchP);
  tGr_LamKchM->GetPoint(0, tX_LamKchM, tY_LamKchM);
  tGr_LamK0->GetPoint(0, tX_LamK0, tY_LamK0);


  TString tDrawOption = "epsame";
  //------------------------
  tGr_LamKchP->SetMarkerColor(tColor_LamKchP);
  tGr_LamKchP->SetMarkerStyle(aMarkerStyle);
  tGr_LamKchP->SetMarkerSize(aMarkerSize);
  tGr_LamKchP->SetFillColor(tColor_LamKchP);
  tGr_LamKchP->SetFillStyle(1000);
  tGr_LamKchP->SetLineColor(tColor_LamKchP);


  tGr_LamKchM->SetMarkerColor(tColor_LamKchM);
  tGr_LamKchM->SetMarkerStyle(aMarkerStyle);
  tGr_LamKchM->SetMarkerSize(aMarkerSize);
  tGr_LamKchM->SetFillColor(tColor_LamKchM);
  tGr_LamKchM->SetFillStyle(1000);
  tGr_LamKchM->SetLineColor(tColor_LamKchM);


  tGr_LamK0->SetMarkerColor(tColor_LamK0);
  tGr_LamK0->SetMarkerStyle(aMarkerStyle);
  tGr_LamK0->SetMarkerSize(aMarkerSize);
  tGr_LamK0->SetFillColor(tColor_LamK0);
  tGr_LamK0->SetFillStyle(1000);
  tGr_LamK0->SetLineColor(tColor_LamK0);

  if(aErrType==kStat)
  {
    tDrawOption = ("pzsame");

    tGr_LamKchP->SetLineWidth(1);
    tGr_LamKchM->SetLineWidth(1);
    tGr_LamK0->SetLineWidth(1);
  }
  else if(aErrType==kSys)
  {
    tDrawOption = ("e2same");

    tGr_LamKchP->SetLineWidth(0);
    tGr_LamKchM->SetLineWidth(0);
    tGr_LamK0->SetLineWidth(0);
  }
  else assert(0);
  //------------------------
  if(tY_LamKchP != 0.) tGr_LamKchP->Draw(tDrawOption);
  if(tY_LamKchM != 0.) tGr_LamKchM->Draw(tDrawOption);
  if(tY_LamK0 != 0.) tGr_LamK0->Draw(tDrawOption);
}


//---------------------------------------------------------------------------------------------------------------------------------
TCanvas* CompareAllReF0vsImF0AcrossAnalyses(IncludeResType aIncludeResType=kInclude10ResAnd3Res, IncludeD0Type aIncludeD0Type=kFreeAndFixedD0, Plot10and3Type aPlot10and3Type=kPlot10and3SeparateAndAvg, bool aIncludeFreeFixedD0Avgs=true, ErrorType aErrType=kStatAndSys, bool aDrawFixedRadii=false, bool aDrawPredictions=false, bool bSaveImage=false)
{
  vector<bool> tIncludePlots = GetIncludePlotsVec(aIncludeResType, aIncludeD0Type, aPlot10and3Type, aIncludeFreeFixedD0Avgs);
  assert(tIncludePlots.size() == tDrawAcrossAnalysesInfoVec.size());

  //----------------------------------------
  TString tCanBaseName =  "CompareAllReF0vsImF0AcrossAnalyses";
  TString tModifier = "";
  if(aIncludeD0Type==kFreeAndFixedD0 && aIncludeFreeFixedD0Avgs) tModifier = TString("_IncludeFreeFixedD0Avgs");

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

  int tNPlots = 0;
  for(unsigned int i=0; i<tIncludePlots.size(); i++) if(tIncludePlots[i]) tNPlots++;
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
  for(unsigned int i=0; i<tIncludePlots.size(); i++) if(tIncludePlots[i] && tDrawAcrossAnalysesInfoVec[i].incD0Type != kFixedD0Only) tND0Increments++;

  if(aDrawFixedRadii)
  {
    vector<FitInfo> aTempFitInfoVec = GetFitInfoVec(kLamKchP, aIncludeResType, aIncludeD0Type, kFixedRadiiOnly, kFreeLambdaOnly);
    for(unsigned int i=0; i<aTempFitInfoVec.size(); i++) if(aTempFitInfoVec[i].d0 != 0.) tND0Increments++;
  }
  tND0Increments +=1;  //To give some room at left and right of plot
  double tIncrementSize = 1./tND0Increments;

  //------------------------------------------------------

  IncludeResType tIncResType;
  IncludeD0Type tIncD0Type;
  int tMarkerStyle;
  double tMarkerSize;
  TString tDescriptor;

  int iTex = 0;
  int iD0Inc = 0;
  for(unsigned int i=0; i<tIncludePlots.size(); i++)
  {

    tIncResType = tDrawAcrossAnalysesInfoVec[i].incResType;
    tIncD0Type = tDrawAcrossAnalysesInfoVec[i].incD0Type;
    tMarkerStyle = tDrawAcrossAnalysesInfoVec[i].markerStyle;
    tMarkerSize = tDrawAcrossAnalysesInfoVec[i].markerSize;
    tDescriptor = tDrawAcrossAnalysesInfoVec[i].descriptor;


    if(tIncludePlots[i])
    {
      if(tIncResType==kIncludeNoRes)  //Signifies QM results
      {
        if(aErrType==kStatAndSys)
        {
          if(aIncludeD0Type==kFreeAndFixedD0)
          {
            DrawReF0vsImF0AcrossAnalysesQMResults((TPad*)tPadReF0vsImF0, 20, tMarkerSize, kSys, kFreeD0Only);
            DrawReF0vsImF0AcrossAnalysesQMResults((TPad*)tPadReF0vsImF0, 20, tMarkerSize, kStat, kFreeD0Only);

            DrawD0AcrossAnalysesQMResults((TPad*)tPadD0, 20, tMarkerSize, kSys, kFreeD0Only, (iD0Inc+1)*tIncrementSize);
            DrawD0AcrossAnalysesQMResults((TPad*)tPadD0, 20, tMarkerSize, kStat, kFreeD0Only, (iD0Inc+1)*tIncrementSize);

            //--------

            DrawReF0vsImF0AcrossAnalysesQMResults((TPad*)tPadReF0vsImF0, 24, tMarkerSize, kSys, kFixedD0Only);
            DrawReF0vsImF0AcrossAnalysesQMResults((TPad*)tPadReF0vsImF0, 24, tMarkerSize, kStat, kFixedD0Only);

            DrawD0AcrossAnalysesQMResults((TPad*)tPadD0, 24, tMarkerSize, kSys, kFixedD0Only, (iD0Inc+1)*tIncrementSize);
            DrawD0AcrossAnalysesQMResults((TPad*)tPadD0, 24, tMarkerSize, kStat, kFixedD0Only, (iD0Inc+1)*tIncrementSize);
          }
          else
          {
            DrawReF0vsImF0AcrossAnalysesQMResults((TPad*)tPadReF0vsImF0, tMarkerStyle, tMarkerSize, kSys, aIncludeD0Type);
            DrawReF0vsImF0AcrossAnalysesQMResults((TPad*)tPadReF0vsImF0, tMarkerStyle, tMarkerSize, kStat, aIncludeD0Type);

            DrawD0AcrossAnalysesQMResults((TPad*)tPadD0, tMarkerStyle, tMarkerSize, kSys, aIncludeD0Type, (iD0Inc+1)*tIncrementSize);
            DrawD0AcrossAnalysesQMResults((TPad*)tPadD0, tMarkerStyle, tMarkerSize, kStat, aIncludeD0Type, (iD0Inc+1)*tIncrementSize);
          }
        }
        else
        {
          if(aIncludeD0Type==kFreeAndFixedD0)
          {
            DrawReF0vsImF0AcrossAnalysesQMResults((TPad*)tPadReF0vsImF0, tMarkerStyle, tMarkerSize, aErrType, kFreeD0Only);
            DrawD0AcrossAnalysesQMResults((TPad*)tPadD0, tMarkerStyle, tMarkerSize, aErrType, kFreeD0Only, (iD0Inc+1)*tIncrementSize);

            //-----------------

            DrawReF0vsImF0AcrossAnalysesQMResults((TPad*)tPadReF0vsImF0, tMarkerStyle, tMarkerSize, aErrType, kFixedD0Only);
            DrawD0AcrossAnalysesQMResults((TPad*)tPadD0, tMarkerStyle, tMarkerSize, aErrType, kFixedD0Only, (iD0Inc+1)*tIncrementSize);
          }
          else
          {
            DrawReF0vsImF0AcrossAnalysesQMResults((TPad*)tPadReF0vsImF0, tMarkerStyle, tMarkerSize, aErrType, aIncludeD0Type);
            DrawD0AcrossAnalysesQMResults((TPad*)tPadD0, tMarkerStyle, tMarkerSize, aErrType, aIncludeD0Type, (iD0Inc+1)*tIncrementSize);
          }
        }
      }
      else 
      {
        if(aErrType==kStatAndSys)
        {
          DrawReF0vsImF0AcrossAnalyses((TPad*)tPadReF0vsImF0, tMarkerStyle, tMarkerSize, tIncResType, tIncD0Type, kSys);
          DrawReF0vsImF0AcrossAnalyses((TPad*)tPadReF0vsImF0, tMarkerStyle, tMarkerSize, tIncResType, tIncD0Type, kStat);

          DrawD0AcrossAnalyses((TPad*)tPadD0, tMarkerStyle, tMarkerSize, tIncResType, tIncD0Type, kSys, (iD0Inc+1)*tIncrementSize);
          DrawD0AcrossAnalyses((TPad*)tPadD0, tMarkerStyle, tMarkerSize, tIncResType, tIncD0Type, kStat, (iD0Inc+1)*tIncrementSize);
        }
        else
        {
          DrawReF0vsImF0AcrossAnalyses((TPad*)tPadReF0vsImF0, tMarkerStyle, tMarkerSize, tIncResType, tIncD0Type, aErrType);
          DrawD0AcrossAnalyses((TPad*)tPadD0, tMarkerStyle, tMarkerSize, tIncResType, tIncD0Type, aErrType, (iD0Inc+1)*tIncrementSize);
        }
      }

      tPadReF0vsImF0->cd();
      if(tDescriptor.Contains("d_{0}")) tTex->DrawLatex(tStartX, tStartY-iTex*tIncrementY-0.1*tIncrementY, tDescriptor);
      else tTex->DrawLatex(tStartX, tStartY-iTex*tIncrementY, tDescriptor);
      tMarker->SetMarkerStyle(tMarkerStyle);
      tMarker->DrawMarker(tStartX-tIncrementX, tStartY-iTex*tIncrementY);
      iTex++;
      if(tIncludePlots[i] && tDrawAcrossAnalysesInfoVec[i].incD0Type != kFixedD0Only) iD0Inc++;
    }
  }

  //------------------------------------------------------

  if(aDrawFixedRadii)
  {
    //TODO For now, since I need to finish this quickly, no option to average
    assert(aPlot10and3Type != kPlot10and3SeparateAndAvg);
    assert(!aIncludeFreeFixedD0Avgs);

    DrawAllReF0vsImF0(tPadReF0vsImF0, tPadD0,
                      kLamKchP, 
                      aIncludeResType, aIncludeD0Type,
                      kFixedRadiiOnly, kFreeLambdaOnly, 
                      kStat, (iD0Inc+1)*tIncrementSize, tIncrementSize);

    DrawAllReF0vsImF0(tPadReF0vsImF0, tPadD0,
                      kLamKchM, 
                      aIncludeResType, aIncludeD0Type,
                      kFixedRadiiOnly, kFreeLambdaOnly, 
                      kStat, (iD0Inc+1)*tIncrementSize, tIncrementSize);

    DrawAllReF0vsImF0(tPadReF0vsImF0, tPadD0,
                      kLamK0, 
                      aIncludeResType, aIncludeD0Type,
                      kFixedRadiiOnly, kFreeLambdaOnly, 
                      kStat, (iD0Inc+1)*tIncrementSize, tIncrementSize);
  }

  //------------------------------------------------------
  if(aDrawPredictions)
  {
    tPadReF0vsImF0->cd();

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

    TLegend* tLegPredictions = new TLegend(0.825, 0.725, 0.975, 0.875);
      tLegPredictions->SetLineWidth(0);
      tLegPredictions->AddEntry(tGr_0607100_Set1, "[1] Set 1", "p");
      tLegPredictions->AddEntry(tGr_0607100_Set2, "[1] Set 2", "p");
      tLegPredictions->AddEntry(tGr_PhysRevD_KLam, "[2] K#Lambda", "p");
      tLegPredictions->AddEntry(tGr_PhysRevD_AKLam, "[2] #bar{K}#Lambda", "p");
    tLegPredictions->Draw();
  }


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
    if(aDrawFixedRadii) tModifier = TString("_wFixedRadiiResults");
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

  bool bSaveFigures = true;
  CentralityType tCentType = kMB;

  IncludeResType tIncludeResType;
    tIncludeResType = kInclude10ResAnd3Res;
    //tIncludeResType = kInclude10ResOnly;
    //tIncludeResType = kInclude3ResOnly;

  IncludeD0Type tIncludeD0Type;
    //tIncludeD0Type = kFreeAndFixedD0;
    //tIncludeD0Type = kFreeD0Only;
    tIncludeD0Type = kFixedD0Only;

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



//-------------- Combinations for PWGCF 20180115 -----------------------------

/*
  tIncludeResType = kInclude10ResOnly;
  tIncludeD0Type = kFreeD0Only;
  tPlot10and3Type=kPlot10and3SeparateOnly;
  tIncludeFreeFixedD0Avgs=false;
*/

/*
  tIncludeResType = kInclude3ResOnly;
  tIncludeD0Type = kFreeD0Only;
  tPlot10and3Type=kPlot10and3SeparateOnly;
  tIncludeFreeFixedD0Avgs=false;
*/

/*
  tIncludeResType = kInclude10ResAnd3Res;
  tIncludeD0Type = kFreeD0Only;
  tPlot10and3Type=kPlot10and3SeparateOnly;
  tIncludeFreeFixedD0Avgs=false;
*/

/*
  tIncludeResType = kInclude10ResOnly;
  tIncludeD0Type = kFixedD0Only;
  tPlot10and3Type=kPlot10and3SeparateOnly;
  tIncludeFreeFixedD0Avgs=false;
*/

/*
  tIncludeResType = kInclude3ResOnly;
  tIncludeD0Type = kFixedD0Only;
  tPlot10and3Type=kPlot10and3SeparateOnly;
  tIncludeFreeFixedD0Avgs=false;
*/


//  NOT vital, possible only in Backup

/*
  tIncludeResType = kInclude10ResOnly;
  tIncludeD0Type = kFreeAndFixedD0;
  tPlot10and3Type=kPlot10and3SeparateOnly;
  tIncludeFreeFixedD0Avgs=false;
*/
/*
  tIncludeResType = kInclude3ResOnly;
  tIncludeD0Type = kFreeAndFixedD0;
  tPlot10and3Type=kPlot10and3SeparateOnly;
  tIncludeFreeFixedD0Avgs=false;
*/

/*
  tIncludeResType = kInclude10ResAnd3Res;
  tIncludeD0Type = kFreeD0Only;
  tPlot10and3Type=kPlot10and3SeparateOnly;
  tIncludeFreeFixedD0Avgs=false;
*/

/*
  tIncludeResType = kInclude10ResAnd3Res;
  tIncludeD0Type = kFreeAndFixedD0;
  tPlot10and3Type=kPlot10and3SeparateOnly;
  tIncludeFreeFixedD0Avgs=false;
*/

//-------------------------------------------------------------------------------
//------- Some common combinations ----------
/*
  //----- Draw all -----
  tIncludeResType = kInclude10ResAnd3Res;
  tIncludeD0Type = kFreeAndFixedD0;
  tPlot10and3Type=kPlot10and3SeparateAndAvg;
  tIncludeFreeFixedD0Avgs=true;
*/

/*
  //----- Draw free and fixed d0, for both 10, 3, and 10and3 -----
  //----- Formerly known as v2 -----
  tIncludeResType = kInclude10ResAnd3Res;
  tIncludeD0Type = kFreeAndFixedD0;
  tPlot10and3Type = kPlot10and3SeparateAndAvg;
  tIncludeFreeFixedD0Avgs = false;
*/


//-------------------------------------------------------------------------------
/*
  TCanvas* tCanReF0vsImF0_LamKchP;
  tCanReF0vsImF0_LamKchP = DrawAllReF0vsImF0(kLamKchP, tIncludeResType, tIncludeD0Type, tIncludeRadiiType, tIncludeLambdaType, tErrorType, bSaveFigures);

  //-------------
  TCanvas *tCanRadiusvsLambda_LamKchP1, *tCanRadiusvsLambda_LamKchP2, *tCanRadiusvsLambda_LamKchP3;
  if(tCentType != kMB)
  {
    tCanRadiusvsLambda_LamKchP1 = DrawAllRadiusvsLambda(kLamKchP, tCentType, tIncludeResType, tIncludeD0Type, tIncludeRadiiType, tIncludeLambdaType, tErrorType, bSaveFigures);
  }
  else
  {
    tCanRadiusvsLambda_LamKchP1 = DrawAllRadiusvsLambda(kLamKchP, k0010, tIncludeResType, tIncludeD0Type, tIncludeRadiiType, tIncludeLambdaType, tErrorType, bSaveFigures);
    tCanRadiusvsLambda_LamKchP2 = DrawAllRadiusvsLambda(kLamKchP, k1030, tIncludeResType, tIncludeD0Type, tIncludeRadiiType, tIncludeLambdaType, tErrorType, bSaveFigures);
    tCanRadiusvsLambda_LamKchP3 = DrawAllRadiusvsLambda(kLamKchP, k3050, tIncludeResType, tIncludeD0Type, tIncludeRadiiType, tIncludeLambdaType, tErrorType, bSaveFigures);
  }

//-------------------------------------------------------------------------------
  TCanvas* tCanReF0vsImF0_LamKchM;
  tCanReF0vsImF0_LamKchM = DrawAllReF0vsImF0(kLamKchM, tIncludeResType, tIncludeD0Type, tIncludeRadiiType, tIncludeLambdaType, tErrorType, bSaveFigures);

  //-------------
  TCanvas *tCanRadiusvsLambda_LamKchM1, *tCanRadiusvsLambda_LamKchM2, *tCanRadiusvsLambda_LamKchM3;
  if(tCentType != kMB)
  {
    tCanRadiusvsLambda_LamKchM1 = DrawAllRadiusvsLambda(kLamKchM, tCentType, tIncludeResType, tIncludeD0Type, tIncludeRadiiType, tIncludeLambdaType, tErrorType, bSaveFigures);
  }
  else
  {
    tCanRadiusvsLambda_LamKchM1 = DrawAllRadiusvsLambda(kLamKchM, k0010, tIncludeResType, tIncludeD0Type, tIncludeRadiiType, tIncludeLambdaType, tErrorType, bSaveFigures);
    tCanRadiusvsLambda_LamKchM2 = DrawAllRadiusvsLambda(kLamKchM, k1030, tIncludeResType, tIncludeD0Type, tIncludeRadiiType, tIncludeLambdaType, tErrorType, bSaveFigures);
    tCanRadiusvsLambda_LamKchM3 = DrawAllRadiusvsLambda(kLamKchM, k3050, tIncludeResType, tIncludeD0Type, tIncludeRadiiType, tIncludeLambdaType, tErrorType, bSaveFigures);
  }

//-------------------------------------------------------------------------------
  TCanvas* tCanReF0vsImF0_LamK0;
  tCanReF0vsImF0_LamK0 = DrawAllReF0vsImF0(kLamK0, tIncludeResType, tIncludeD0Type, tIncludeRadiiType, tIncludeLambdaType, tErrorType, bSaveFigures);

  //-------------
  TCanvas *tCanRadiusvsLambda_LamK01, *tCanRadiusvsLambda_LamK02, *tCanRadiusvsLambda_LamK03;
  if(tCentType != kMB)
  {
    tCanRadiusvsLambda_LamK01 = DrawAllRadiusvsLambda(kLamK0, tCentType, tIncludeResType, tIncludeD0Type, tIncludeRadiiType, tIncludeLambdaType, tErrorType, bSaveFigures);
  }
  else
  {
    tCanRadiusvsLambda_LamK01 = DrawAllRadiusvsLambda(kLamK0, k0010, tIncludeResType, tIncludeD0Type, tIncludeRadiiType, tIncludeLambdaType, tErrorType, bSaveFigures);
    tCanRadiusvsLambda_LamK02 = DrawAllRadiusvsLambda(kLamK0, k1030, tIncludeResType, tIncludeD0Type, tIncludeRadiiType, tIncludeLambdaType, tErrorType, bSaveFigures);
    tCanRadiusvsLambda_LamK03 = DrawAllRadiusvsLambda(kLamK0, k3050, tIncludeResType, tIncludeD0Type, tIncludeRadiiType, tIncludeLambdaType, tErrorType, bSaveFigures);
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








