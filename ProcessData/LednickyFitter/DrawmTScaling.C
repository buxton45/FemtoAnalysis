#include "DrawmTScaling.h"
#include "CompareFittingMethods.h"


//Note:  &aDataPointsWithErrors will have a collection of data points with associated errors
//       aDataPointsWithErrors.size() will give the number of data points to be plotted
//       aDataPointsWithErrors[i].size() will tell whether the graph should be of type TGraphErrors or TGraphAsymmErrors
//           aDataPointsWithErrors[i].size() == 4 ==> aDataPointsWithErrors[i] = [aX, aXerr, aY, aYerr]
//           aDataPointsWithErrors[i].size() == 6 ==> aDataPointsWithErrors[i] = [aX, aXerrLow, aXerrHigh, aY, aYerrLow, aYerrHigh]
//___________________________________________________________________________________
void SetDataPoints(TGraphAsymmErrors* aGraph, td2dVec &aDataPointsWithErrors, bool aSetErrors=true)
{
  unsigned int tNDataPoints = aDataPointsWithErrors.size();
  for(unsigned int i=0; i<tNDataPoints; i++)
  {
    if(aDataPointsWithErrors[i].size() == 4)
    {
      aGraph->SetPoint(i, aDataPointsWithErrors[i][0], aDataPointsWithErrors[i][2]);
      aGraph->SetPointError(i, aDataPointsWithErrors[i][1], aDataPointsWithErrors[i][1], aDataPointsWithErrors[i][3], aDataPointsWithErrors[i][3]);
    }
    else if(aDataPointsWithErrors[i].size() == 6)
    {
      aGraph->SetPoint(i, aDataPointsWithErrors[i][0], aDataPointsWithErrors[i][3]);
      aGraph->SetPointError(i, aDataPointsWithErrors[i][1], aDataPointsWithErrors[i][2], aDataPointsWithErrors[i][4], aDataPointsWithErrors[i][5]);
    }
    else assert(0);
  }
}

//___________________________________________________________________________________
void DrawPoints(TString aName, 
                td2dVec &aDataPointsWithSysErrors, td2dVec &aDataPointsWithStatErrors, 
                int aMarkerColorSys, int aMarkerColorStat, int aMarkerStyle, Size_t aMarkerSize, int aMarkerOutlineStyle=0)
{
  //First, make sure there are equal number systematic and statistical data points
  assert(aDataPointsWithSysErrors.size() == aDataPointsWithStatErrors.size());
  unsigned int tNDataPoints = aDataPointsWithSysErrors.size();

  //Next, make sure all points in each collection are of equal size
  //  i.e. make sure all points are set up for TGraphErrors or TGraphAsymmErrors
  for(unsigned int i=1; i<tNDataPoints; i++) assert(aDataPointsWithSysErrors[i-1].size() == aDataPointsWithSysErrors[i].size());
  for(unsigned int i=1; i<tNDataPoints; i++) assert(aDataPointsWithStatErrors[i-1].size() == aDataPointsWithStatErrors[i].size());

  //----------------------- Draw systematic error boxes first -------------------------------------------------
  TGraphAsymmErrors* tGr = new TGraphAsymmErrors(tNDataPoints);
  tGr->SetName(aName+TString("Sys"));
  tGr->SetFillColor(aMarkerColorSys);
  tGr->SetFillStyle(1000);
  tGr->SetLineColor(0);
  tGr->SetLineWidth(0);
  SetDataPoints(tGr, aDataPointsWithSysErrors);
  tGr->Draw("e2");


  //----------------------- Draw points with statistical errors -------------------------------------------------
  tGr = new TGraphAsymmErrors(tNDataPoints);
  tGr->SetName(aName+TString("Stat"));
  tGr->SetLineColor(aMarkerColorStat);
  tGr->SetMarkerColor(aMarkerColorStat);
  tGr->SetMarkerStyle(aMarkerStyle);
  tGr->SetMarkerSize(aMarkerSize);
  SetDataPoints(tGr, aDataPointsWithStatErrors);
  tGr->Draw("pz");


  //----------------------- If aMarkerOutlineStyle !=0, draw outline of points ----------------------------------
  if(aMarkerOutlineStyle !=0)
  {
    tGr = new TGraphAsymmErrors(tNDataPoints);
    tGr->SetName(aName+TString("Outline"));
    tGr->SetLineColor(1);
    tGr->SetMarkerColor(1);
    tGr->SetMarkerStyle(aMarkerOutlineStyle);
    tGr->SetMarkerSize(aMarkerSize);
    SetDataPoints(tGr, aDataPointsWithStatErrors, false);
    tGr->Draw("px");
  }

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
TGraphAsymmErrors* GetRadiusvsLambda(const FitInfo &aFitInfo, CentralityType aCentType=k0010)
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


//___________________________________________________________________________________
vector<double> GetRInfo(AnalysisType aAnType, CentralityType aCentType, bool aUseWeightedMean=false, bool bInclude10Res=true, bool bInclude3Res=true)
{
  vector<FitInfo> aFitInfoVec;
  if     (aAnType==kLamKchP) aFitInfoVec = tFitInfoVec_LamKchP;
  else if(aAnType==kLamKchM) aFitInfoVec = tFitInfoVec_LamKchM;
  else if(aAnType==kLamK0) aFitInfoVec = tFitInfoVec_LamK0;
  else assert(0);
  //--------------------------------

  TGraphAsymmErrors *tGr;
  if(aUseWeightedMean)
  {
    tGr = GetWeightedMeanRadiusvsLambda(aFitInfoVec, aCentType, bInclude10Res, bInclude3Res);
  }
  else
  {
    assert(!(bInclude10Res && bInclude3Res));
    if(bInclude10Res)
    {
      tGr = GetRadiusvsLambda(aFitInfoVec[0], aCentType);
    }
    else
    {
      tGr = GetRadiusvsLambda(aFitInfoVec[6], aCentType);
    }
  }

  double tRadius, tLambda;
  tGr->GetPoint(0, tRadius, tLambda);
  double tRadiusErr = tGr->GetErrorX(0);

  vector<double> tReturnVec{tRadius, tRadiusErr};
  return tReturnVec;
}

//___________________________________________________________________________________
vector<double> GetRInfoQM(AnalysisType aAnType, CentralityType aCentType)
{
  TGraphAsymmErrors *tGr;
  if     (aAnType==kLamKchP) tGr = GetRadiusvsLambda(tFitInfoQM_LamKchP, aCentType);
  else if(aAnType==kLamKchM) tGr = GetRadiusvsLambda(tFitInfoQM_LamKchM, aCentType);
  else if(aAnType==kLamK0) tGr = GetRadiusvsLambda(tFitInfoQM_LamK0, aCentType);
  else assert(0);
  //--------------------------------

  double tRadius, tLambda;
  tGr->GetPoint(0, tRadius, tLambda);
  double tRadiusErr = tGr->GetErrorX(0);

  vector<double> tReturnVec{tRadius, tRadiusErr};
  return tReturnVec;
}


//_________________________________________________________________________________________________________________________
//*************************************************************************************************************************
//_________________________________________________________________________________________________________________________

int main(int argc, char **argv) 
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything
//-----------------------------------------------------------------------------


//=========Macro generated from canvas: canmtcomb/canmtcomb
//=========  (Fri May  8 11:29:01 2015) by ROOT version5.34/26
  bool bRunAveragedKchPKchM = false;

  bool bUseMinvCalculation = true;
  bool bUseReducedMassCalculation = false;

  bool bMakeOthersTransparent = true;
  bool bOutlinePoints = true;

  bool bResultsWithResiduals = true;
  bool bSaveImage = false;



  //------------------------
  vector<bool> tInfoVec_10and3_WeightedMean{true, true, true};
  vector<bool> tInfoVec_10_WeightedMean{true, true, false};
  vector<bool> tInfoVec_3_WeightedMean{true, false, true};
  vector<bool> tInfoVec_10_AllFree{false, true, false};
  vector<bool> tInfoVec_3_AllFree{false, false, true};
  //------------------------
  bool bDrawQM = true;
  vector<bool> tInfoVec = tInfoVec_10and3_WeightedMean;



  assert(!(bUseMinvCalculation && bUseReducedMassCalculation));

  TString tSaveName = "./mTscaling";

  if(bRunAveragedKchPKchM) tSaveName += TString("Averaged");

  if(bUseMinvCalculation) tSaveName += TString("_MinvCalc");
  if(bUseReducedMassCalculation) tSaveName += TString("_RedMassCal");

  if(bOutlinePoints) tSaveName += TString("_OutlinedPoints");
  if(bMakeOthersTransparent) tSaveName += TString("_OthersTransparent");
  if(bResultsWithResiduals) tSaveName += TString("_WithResiduals");
  tSaveName += TString(".pdf");
  
  Int_t red = kRed;
  Int_t redT = TColor::GetColorTransparent(red,0.2);
  Int_t green = kGreen+2;
  Int_t greenT = TColor::GetColorTransparent(green,0.2);
  Int_t blue = kBlue;
  Int_t blueT = TColor::GetColorTransparent(blue,0.2);
  
  if(bMakeOthersTransparent)
  {
    red = TColor::GetColorTransparent(red,0.4);
    redT = TColor::GetColorTransparent(red,0.2);

    green = TColor::GetColorTransparent(green,0.4);
    greenT = TColor::GetColorTransparent(green,0.2);

    blue = TColor::GetColorTransparent(blue,0.4);
    blueT = TColor::GetColorTransparent(blue,0.2);
  }

  const Int_t myRed = kRed;
  const Int_t myGreen = kGreen+2;
  const Int_t myBlue = kBlue;

  const Int_t myRedT = TColor::GetColorTransparent(red,0.3);
  const Int_t myGreenT = TColor::GetColorTransparent(green,0.3);
  const Int_t myBlueT = TColor::GetColorTransparent(blue,0.3);

  const int tMarkerStyleLamK0 = 29;
  const int tMarkerStyleLamKchP = 21;
  const int tMarkerStyleLamKchM = 33;
  const int tMarkerStyleLamKchAvg = 20;

  int tMarkerStyleLamK0o = 30;
  int tMarkerStyleLamKchPo = 25;
  int tMarkerStyleLamKchMo = 27;
  int tMarkerStyleLamKchAvgo = 24;

  const Size_t tMarkerSize=1.6;

//----------------------- Set up canvas and axes --------------------------------------------
  TCanvas *canmtcomb = new TCanvas("canmtcomb", "canmtcomb",2626,901,700,500);
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);
  canmtcomb->Range(-0.1314815,-0.7522786,2.078395,10.43759);
  canmtcomb->SetFillColor(10);
  canmtcomb->SetBorderMode(0);
  canmtcomb->SetBorderSize(2);
  canmtcomb->SetLeftMargin(0.10);
  canmtcomb->SetRightMargin(0.04);
  canmtcomb->SetTopMargin(0.04);
  canmtcomb->SetBottomMargin(0.12);
  canmtcomb->SetFrameFillColor(0);
  canmtcomb->SetFrameBorderMode(0);
  canmtcomb->SetFrameBorderMode(0);
   
  TH1D *ramka = new TH1D("ramka","",100,0.2,1.99);
//  TH1D *ramka = new TH1D("ramka","",100,0.2,2.2);
  ramka->SetMinimum(1.15);
  ramka->SetMaximum(9.99);
  ramka->SetStats(0);

  ramka->GetXaxis()->SetTitle("#LT#it{m}_{T}#GT (GeV/#it{c}^{2})");
  ramka->GetXaxis()->SetNdivisions(8);
  ramka->GetXaxis()->SetLabelFont(42);
  ramka->GetXaxis()->SetLabelOffset(0.01);
  ramka->GetXaxis()->SetTitleSize(0.054);
  ramka->GetXaxis()->SetTitleOffset(0.94);
  ramka->GetXaxis()->SetTitleFont(42);
  ramka->GetYaxis()->SetTitle("#it{R}_{inv} (fm)");
  ramka->GetYaxis()->SetNdivisions(6);
  ramka->GetYaxis()->SetLabelFont(42);
  ramka->GetYaxis()->SetLabelOffset(0.01);
  ramka->GetYaxis()->SetTitleSize(0.054);
  ramka->GetYaxis()->SetTickLength(0.02);
  ramka->GetYaxis()->SetTitleOffset(0.8);
  ramka->GetYaxis()->SetTitleFont(42);
  ramka->GetZaxis()->SetLabelFont(42);
  ramka->GetZaxis()->SetLabelOffset(0.01);
  ramka->GetZaxis()->SetTitleSize(0.044);
  ramka->GetZaxis()->SetTitleOffset(0.9);
  ramka->GetZaxis()->SetTitleFont(42);
  ramka->Draw("");

//--- NOTE: Drawing of my points saved until last, so they are drawn on top -----------------
//---------------------------------------- Lambda-K0 ----------------------------------------
  double tLamK00010mT = 0.25*(1.583+1.585+1.584+1.583);  //m_avg
  if(bUseMinvCalculation) tLamK00010mT = 0.25*(1.603+1.604+1.603+1.602);  //m_inv
  if(bUseReducedMassCalculation) tLamK00010mT = 0.25*(1.395+1.396+1.396+1.395);  //m_red
  vector<double> RInfo_LamK0_0010;
  if(bDrawQM) RInfo_LamK0_0010 = GetRInfoQM(kLamK0, k0010);
  else RInfo_LamK0_0010 = GetRInfo(kLamK0, k0010, tInfoVec[0], tInfoVec[1], tInfoVec[2]);
  double tLamK00010R = RInfo_LamK0_0010[0];
  double tLamK00010Rerr = RInfo_LamK0_0010[1];
  double tLamK00010RerrSys = 0.329;
/*
  double tLamK00010R = 3.024;
  double tLamK00010Rerr = 0.541;
  double tLamK00010RerrSys = 0.329;
*/

  double tLamK01030mT = 0.25*(1.568+1.568+1.569+1.567);  //m_avg
  if(bUseMinvCalculation) tLamK01030mT = 0.25*(1.588+1.588+1.589+1.587);  //m_inv
  if(bUseReducedMassCalculation) tLamK01030mT = 0.25*(1.377+1.377+1.378+1.375);  //m_red
  vector<double> RInfo_LamK0_1030;
  if(bDrawQM) RInfo_LamK0_1030 = GetRInfoQM(kLamK0, k1030);
  else RInfo_LamK0_1030 = GetRInfo(kLamK0, k1030, tInfoVec[0], tInfoVec[1], tInfoVec[2]);
  double tLamK01030R = RInfo_LamK0_1030[0];
  double tLamK01030Rerr = RInfo_LamK0_1030[1];
  double tLamK01030RerrSys = 0.324;
/*
  double tLamK01030R = 2.270;
  double tLamK01030Rerr = 0.413;
  double tLamK01030RerrSys = 0.324;
*/

  double tLamK03050mT = 0.25*(1.528+1.526+1.528+1.525);  //m_avg
  if(bUseMinvCalculation) tLamK03050mT = 0.25*(1.548+1.546+1.549+1.546);  //m_inv
  if(bUseReducedMassCalculation) tLamK03050mT = 0.25*(1.331+1.328+1.331+1.327);  //m_red
  vector<double> RInfo_LamK0_3050;
  if(bDrawQM) RInfo_LamK0_3050 = GetRInfoQM(kLamK0, k3050);
  else RInfo_LamK0_3050 = GetRInfo(kLamK0, k3050, tInfoVec[0], tInfoVec[1], tInfoVec[2]);
  double tLamK03050R = RInfo_LamK0_3050[0];
  double tLamK03050Rerr = RInfo_LamK0_3050[1];
  double tLamK03050RerrSys = 0.280;
/*
  double tLamK03050R = 1.669;
  double tLamK03050Rerr = 0.307;
  double tLamK03050RerrSys = 0.280;
*/
/*
  if(bResultsWithResiduals)
  {
    tLamK00010R = 2.18;
    tLamK00010Rerr = 0.22;
    tLamK00010RerrSys = 0.33;

    tLamK01030R = 1.81;
    tLamK01030Rerr = 0.19;
    tLamK01030RerrSys = 0.32;

    tLamK03050R = 1.52;
    tLamK03050Rerr = 0.16;
    tLamK03050RerrSys = 0.28;
  }
*/

//---------------------------------------- Lambda-KchP --------------------------------------
  double tLamKchP0010mT = 0.25*(1.417+1.416+1.420+1.416); //m_avg
  if(bUseMinvCalculation) tLamKchP0010mT = 0.25*(1.439+1.438+1.442+1.437); //m_inv
  if(bUseReducedMassCalculation) tLamKchP0010mT = 0.25*(1.203+1.200+1.206+1.201); //m_red
  vector<double> RInfo_LamKchP_0010;
  if(bDrawQM) RInfo_LamKchP_0010 = GetRInfoQM(kLamKchP, k0010);
  else RInfo_LamKchP_0010 = GetRInfo(kLamKchP, k0010, tInfoVec[0], tInfoVec[1], tInfoVec[2]);
  double tLamKchP0010R = RInfo_LamKchP_0010[0];
  double tLamKchP0010Rerr = RInfo_LamKchP_0010[1];
  double tLamKchP0010RerrSys = 0.830;
/*
  double tLamKchP0010R = 4.045;
  double tLamKchP0010Rerr = 0.381;
  double tLamKchP0010RerrSys = 0.830;
*/

  double tLamKchP1030mT = 0.25*(1.405+1.401+1.409+1.402); //m_avg
  if(bUseMinvCalculation) tLamKchP1030mT = 0.25*(1.427+1.423+1.431+1.425); //m_inv
  if(bUseReducedMassCalculation) tLamKchP1030mT = 0.25*(1.188+1.182+1.192+1.184); //m_red
  vector<double> RInfo_LamKchP_1030;
  if(bDrawQM) RInfo_LamKchP_1030 = GetRInfoQM(kLamKchP, k1030);
  else RInfo_LamKchP_1030 = GetRInfo(kLamKchP, k1030, tInfoVec[0], tInfoVec[1], tInfoVec[2]);
  double tLamKchP1030R = RInfo_LamKchP_1030[0];
  double tLamKchP1030Rerr = RInfo_LamKchP_1030[1];
  double tLamKchP1030RerrSys = 0.663;
/*
  double tLamKchP1030R = 3.923;
  double tLamKchP1030Rerr = 0.454;
  double tLamKchP1030RerrSys = 0.663;
*/

  double tLamKchP3050mT = 0.25*(1.368+1.360+1.372+1.362); //m_avg
  if(bUseMinvCalculation) tLamKchP3050mT = 0.25*(1.390+1.382+1.395+1.385); //m_inv
  if(bUseReducedMassCalculation) tLamKchP3050mT = 0.25*(1.144+1.134+1.149+1.136); //m_red
  vector<double> RInfo_LamKchP_3050;
  if(bDrawQM) RInfo_LamKchP_3050 = GetRInfoQM(kLamKchP, k3050);
  else RInfo_LamKchP_3050 = GetRInfo(kLamKchP, k3050, tInfoVec[0], tInfoVec[1], tInfoVec[2]);
  double tLamKchP3050R = RInfo_LamKchP_3050[0];
  double tLamKchP3050Rerr = RInfo_LamKchP_3050[1];
  double tLamKchP3050RerrSys = 0.420;
/*
  double tLamKchP3050R = 3.717;
  double tLamKchP3050Rerr = 0.554;
  double tLamKchP3050RerrSys = 0.420;
*/
/*
  if(bResultsWithResiduals)
  {
    tLamKchP0010R = 4.97;
    tLamKchP0010Rerr = 1.01;
    tLamKchP0010RerrSys = 0.83;

    tLamKchP1030R = 4.76;
    tLamKchP1030Rerr = 1.01;
    tLamKchP1030RerrSys = 0.66;

    tLamKchP3050R = 3.55;
    tLamKchP3050Rerr = 0.52;
    tLamKchP3050RerrSys = 0.42;
  }
*/

//---------------------------------------- Lambda-KchM --------------------------------------
  double tLamKchM0010mT = 0.25*(1.419+1.417+1.420+1.419); //m_avg
  if(bUseMinvCalculation) tLamKchM0010mT = 0.25*(1.441+1.439+1.442+1.440); //m_inv
  if(bUseReducedMassCalculation) tLamKchM0010mT = 0.25*(1.204+1.202+1.205+1.204); //m_red
  vector<double> RInfo_LamKchM_0010;
  if(bDrawQM) RInfo_LamKchM_0010 = GetRInfoQM(kLamKchM, k0010);
  else RInfo_LamKchM_0010 = GetRInfo(kLamKchM, k0010, tInfoVec[0], tInfoVec[1], tInfoVec[2]);
  double tLamKchM0010R = RInfo_LamKchM_0010[0];
  double tLamKchM0010Rerr = RInfo_LamKchM_0010[1];
  double tLamKchM0010RerrSys = 1.375;
/*
  double tLamKchM0010R = 4.787;
  double tLamKchM0010Rerr = 0.788;
  double tLamKchM0010RerrSys = 1.375;
*/

  double tLamKchM1030mT = 0.25*(1.404+1.404+1.407+1.407); //m_avg
  if(bUseMinvCalculation) tLamKchM1030mT = 0.25*(1.426+1.426+1.428+1.429); //m_inv
  if(bUseReducedMassCalculation) tLamKchM1030mT = 0.25*(1.187+1.187+1.189+1.190); //m_red
  vector<double> RInfo_LamKchM_1030;
  if(bDrawQM) RInfo_LamKchM_1030 = GetRInfoQM(kLamKchM, k1030);
  else RInfo_LamKchM_1030 = GetRInfo(kLamKchM, k1030, tInfoVec[0], tInfoVec[1], tInfoVec[2]);
  double tLamKchM1030R = RInfo_LamKchM_1030[0];
  double tLamKchM1030Rerr = RInfo_LamKchM_1030[1];
  double tLamKchM1030RerrSys = 0.978;
/*
  double tLamKchM1030R = 4.001;
  double tLamKchM1030Rerr = 0.719;
  double tLamKchM1030RerrSys = 0.978;
*/

  double tLamKchM3050mT = 0.25*(1.364+1.367+1.366+1.370); //m_avg
  if(bUseMinvCalculation) tLamKchM3050mT = 0.25*(1.387+1.389+1.389+1.392); //m_inv
  if(bUseReducedMassCalculation) tLamKchM3050mT = 0.25*(1.139+1.143+1.141+1.146); //m_red
  vector<double> RInfo_LamKchM_3050;
  if(bDrawQM) RInfo_LamKchM_3050 = GetRInfoQM(kLamKchM, k3050);
  else RInfo_LamKchM_3050 = GetRInfo(kLamKchM, k3050, tInfoVec[0], tInfoVec[1], tInfoVec[2]);
  double tLamKchM3050R = RInfo_LamKchM_3050[0];
  double tLamKchM3050Rerr = RInfo_LamKchM_3050[1];
  double tLamKchM3050RerrSys = 0.457;
/*
  double tLamKchM3050R = 2.112;
  double tLamKchM3050Rerr = 0.517;
  double tLamKchM3050RerrSys = 0.457;
*/
/*
  if(bResultsWithResiduals)
  {
    tLamKchM0010R = 6.20;
    tLamKchM0010Rerr = 1.93;
    tLamKchM0010RerrSys = 1.38;

    tLamKchM1030R = 4.86;
    tLamKchM1030Rerr = 1.33;
    tLamKchM1030RerrSys = 0.98;

    tLamKchM3050R = 2.86;
    tLamKchM3050Rerr = 0.89;
    tLamKchM3050RerrSys = 0.46;
  }
*/

//-------------------------------- Average Lam-KchP and LamKchM -----------------------------
  double tLamKchAvg0010mT = 0.5*(tLamKchP0010mT+tLamKchM0010mT);
  double tLamKchAvg0010R = 0.5*(tLamKchP0010R+tLamKchM0010R);
  double tLamKchAvg0010Rerr = sqrt(pow(0.5*tLamKchP0010Rerr,2)+pow(0.5*tLamKchM0010Rerr,2));
  double tLamKchAvg0010RerrSys = sqrt(pow(0.5*tLamKchP0010RerrSys,2)+pow(0.5*tLamKchM0010RerrSys,2));

  double tLamKchAvg1030mT = 0.5*(tLamKchP1030mT+tLamKchM1030mT);
  double tLamKchAvg1030R = 0.5*(tLamKchP1030R+tLamKchM1030R);
  double tLamKchAvg1030Rerr = sqrt(pow(0.5*tLamKchP1030Rerr,2)+pow(0.5*tLamKchM1030Rerr,2));
  double tLamKchAvg1030RerrSys = sqrt(pow(0.5*tLamKchP1030RerrSys,2)+pow(0.5*tLamKchM1030RerrSys,2));

  double tLamKchAvg3050mT = 0.5*(tLamKchP3050mT+tLamKchM3050mT);
  double tLamKchAvg3050R = 0.5*(tLamKchP3050R+tLamKchM3050R);
  double tLamKchAvg3050Rerr = sqrt(pow(0.5*tLamKchP3050Rerr,2)+pow(0.5*tLamKchM3050Rerr,2));
  double tLamKchAvg3050RerrSys = sqrt(pow(0.5*tLamKchP3050RerrSys,2)+pow(0.5*tLamKchM3050RerrSys,2));


//---------------------------------- Draw Legend -------------------------------------------- 
  if(!bRunAveragedKchPKchM)
  {
    TLatex *   tex = new TLatex(1.375,9.4,"Pb-Pb #sqrt{#it{s}_{NN}} = 2.76 TeV");
    tex->SetTextFont(42);
    tex->SetTextSize(0.044);
    tex->SetLineWidth(2);
    tex->Draw();
//    tex->DrawLatex(0.85,9.4,"ALICE Preliminary");
    tex = new TLatex();
    tex->SetTextAlign(12);
    tex->SetTextFont(42);
    tex->SetTextSize(0.04);
    tex->SetLineWidth(2);


    // species text and markers
    TMarker *marker = new TMarker();
    marker->SetMarkerSize(tMarkerSize);
  
    tex->DrawLatex(1.5,8.8,"#pi^{#pm}#pi^{#pm}");
    marker->SetMarkerStyle(28);//pions
    marker->DrawMarker(1.66,8.8);
  
    tex->DrawLatex(1.5,8.2,"K^{#pm}K^{#pm}");
    marker->SetMarkerStyle(25);//Kch
    marker->DrawMarker(1.66,8.2);
  
    tex->DrawLatex(1.5,7.6,"K_{S}^{0}K_{S}^{0}");
    marker->SetMarkerStyle(27);//K0s
    marker->DrawMarker(1.66,7.6);
  
    tex->DrawLatex(1.5,7.0,"#bar{p}#bar{p}");
    marker->SetMarkerStyle(5);// antiprotons
    marker->DrawMarker(1.66,7.0);
  
    tex->DrawLatex(1.5,6.4,"pp");
    marker->SetMarkerStyle(24);//protons
    marker->DrawMarker(1.66,6.4);
  
    //------- Column 2 ----------------------
  
    tex->DrawLatex(1.8,8.2,"#LambdaK^{+}");
    marker->SetMarkerStyle(tMarkerStyleLamKchP); //LamK+
    marker->DrawMarker(1.95,8.2);
  
    tex->DrawLatex(1.8,7.6,"#LambdaK^{-}");
    marker->SetMarkerStyle(tMarkerStyleLamKchM); //LamK-
    marker->DrawMarker(1.95,7.6);
  
    tex->DrawLatex(1.8,7.0,"#LambdaK^{0}_{S}");
    marker->SetMarkerStyle(tMarkerStyleLamK0); //LamK0
    marker->DrawMarker(1.95,7.0);
  
  
  
    // centralities
    TLine line;
    line.SetLineWidth(2);
    line.SetLineColor(myRed);
    line.DrawLine(1.12,8.7,1.22,8.7);
    line.SetLineColor(myGreen);
    line.DrawLine(1.12,8.1,1.22,8.1);
    line.SetLineColor(myBlue);
    line.DrawLine(1.12,7.5,1.22,7.5);
    tex->DrawLatex(0.9,8.7,"0-10%");
    tex->DrawLatex(0.9,8.1,"10-30%");
    tex->DrawLatex(0.9,7.5,"30-50%");
  }

  else /*if(bRunAveragedKchPKchM)*/
  {
    TLatex *   tex = new TLatex(1.10,9.4,"ALICE Pb-Pb #sqrt{#it{s}_{NN}} = 2.76 TeV");
    tex->SetTextFont(42);
    tex->SetTextSize(0.044);
    tex->SetLineWidth(2);
    tex->Draw();
    tex = new TLatex();
    tex->SetTextAlign(12);
    tex->SetTextFont(42);
    tex->SetTextSize(0.04);
    tex->SetLineWidth(2);


    // species text and markers
    TMarker *marker = new TMarker();
    marker->SetMarkerSize(tMarkerSize);
  
    tex->DrawLatex(1.3,8.8,"#pi^{#pm}#pi^{#pm}");
    marker->SetMarkerStyle(28);//pions
    marker->DrawMarker(1.46,8.8);
  
    tex->DrawLatex(1.3,8.2,"K^{#pm}K^{#pm}");
    marker->SetMarkerStyle(25);//Kch
    marker->DrawMarker(1.46,8.2);
  
    tex->DrawLatex(1.3,7.6,"K_{S}^{0}K_{S}^{0}");
    marker->SetMarkerStyle(27);//K0s
    marker->DrawMarker(1.46,7.6);
  
    tex->DrawLatex(1.3,7.0,"#bar{p}#bar{p}");
    marker->SetMarkerStyle(5);// antiprotons
    marker->DrawMarker(1.46,7.0);
  
    tex->DrawLatex(1.3,6.4,"pp");
    marker->SetMarkerStyle(24);//protons
    marker->DrawMarker(1.46,6.4);
  
    //------- Column 2 ----------------------
  
     tex->DrawLatex(1.57,7.9,"#LT#LambdaK^{+}+#LambdaK^{-}#GT");
     marker->SetMarkerStyle(tMarkerStyleLamKchAvg); //AvgLamK+LamK-
     marker->DrawMarker(1.9,7.9);
  
    tex->DrawLatex(1.65,7.1,"#LambdaK^{0}_{S}");
    marker->SetMarkerStyle(tMarkerStyleLamK0); //LamK0
    marker->DrawMarker(1.9,7.0);
  
  
  
    // centralities
    TLine line;
    line.SetLineWidth(2);
    line.SetLineColor(myRed);
    line.DrawLine(1.07,8.7,1.17,8.7);
    line.SetLineColor(myGreen);
    line.DrawLine(1.07,8.1,1.17,8.1);
    line.SetLineColor(myBlue);
    line.DrawLine(1.07,7.5,1.17,7.5);
    tex->DrawLatex(0.85,8.7,"0-10%");
    tex->DrawLatex(0.85,8.1,"10-30%");
    tex->DrawLatex(0.85,7.5,"30-50%");
  }


//-------------------------------------------------------------------------------------------
//----------------------------- ALICE DATA --------------------------------------------------
//-------------------------------------------------------------------------------------------

//void DrawPoints(TString aName, 
//                td2dVec &aDataPointsWithSysErrors, td2dVec &aDataPointsWithStatErrors, 
//                int aMarkerColorSys, int aMarkerColorStat, int aMarkerStyle, Size_t aMarkerSize, int aMarkerOutlineStyle=0;)
//       aDataPointsWithErrors[i].size() will tell whether the graph should be of type TGraphErrors or TGraphAsymmErrors
//           aDataPointsWithErrors[i].size() == 4 ==> aDataPointsWithErrors[i] = [aX, aXerr, aY, aYerr]
//           aDataPointsWithErrors[i].size() == 6 ==> aDataPointsWithErrors[i] = [aX, aXerrLow, aXerrHigh, aY, aYerrLow, aYerrHigh]

//---------------------------- Protons ------------------------------------------------------
  //----- 0-10% -----
  DrawPoints("GraphPP0010",tPP0010Sys,tPP0010Stat,redT,red,tMarkerStylePP,tMarkerSize);

  //----- 10-30% -----
  DrawPoints("GraphPP1030",tPP1030Sys,tPP1030Stat,greenT,green,tMarkerStylePP,tMarkerSize);
  
  //----- 30-50% -----
  DrawPoints("GraphPP3050",tPP3050Sys,tPP3050Stat,blueT,blue,tMarkerStylePP,tMarkerSize);

  
//------------------------- Anti-Protons ----------------------------------------------------
  //----- 0-10% -----
  DrawPoints("GraphAPAP0010",tAPAP0010Sys,tAPAP0010Stat,redT,red,tMarkerStyleAPAP,tMarkerSize);
  
  //----- 10-30% -----
  DrawPoints("GraphAPAP1030",tAPAP1030Sys,tAPAP1030Stat,greenT,green,tMarkerStyleAPAP,tMarkerSize);

  //----- 30-50% -----
  DrawPoints("GraphAPAP3050",tAPAP3050Sys,tAPAP3050Stat,blueT,blue,tMarkerStyleAPAP,tMarkerSize);

  
//------------------------ K0s-K0s ----------------------------------------------------------  
  //----- 0-10% -----
  DrawPoints("GraphK0sK0s0010",tK0sK0s0010Sys,tK0sK0s0010Stat,redT,red,tMarkerStyleK0sK0s,tMarkerSize);

  //----- 10-30% -----   
  DrawPoints("GraphK0sK0s1030",tK0sK0s1030Sys,tK0sK0s1030Stat,greenT,green,tMarkerStyleK0sK0s,tMarkerSize);

  //----- 30-50% -----
  DrawPoints("GraphK0sK0s3050",tK0sK0s3050Sys,tK0sK0s3050Stat,blueT,blue,tMarkerStyleK0sK0s,tMarkerSize);


//----------------------------- Kch-Kch -----------------------------------------------------  
  //----- 0-10% -----
  DrawPoints("GraphKchKch0010",tKchKch0010Sys,tKchKch0010Stat,redT,red,tMarkerStyleKchKch,tMarkerSize);
  
  //----- 10-30% -----
  DrawPoints("GraphKchKch1030",tKchKch1030Sys,tKchKch1030Stat,greenT,green,tMarkerStyleKchKch,tMarkerSize);

  //----- 30-50% -----
  DrawPoints("GraphKchKch3050",tKchKch3050Sys,tKchKch3050Stat,blueT,blue,tMarkerStyleKchKch,tMarkerSize);

  
//------------------------------ Pi-Pi ------------------------------------------------------
  //----- 0-10% -----
  DrawPoints("GraphPiPi0010",tPiPi0010Sys,tPiPi0010Stat,redT,red,tMarkerStylePiPi,tMarkerSize);

  //----- 10-30% -----
  DrawPoints("GraphPiPi1030",tPiPi1030Sys,tPiPi1030Stat,greenT,green,tMarkerStylePiPi,tMarkerSize);

  //----- 30-50% -----
  DrawPoints("GraphPiPi3050",tPiPi3050Sys,tPiPi3050Stat,blueT,blue,tMarkerStylePiPi,tMarkerSize);

  
//-------------------------------------------------------------------------------------------
//-----------------------------------Draw my points -----------------------------------------
//-------------------------------------------------------------------------------------------

//---------------------------------------- Lambda-K0 ----------------------------------------
  if(!bOutlinePoints) tMarkerStyleLamK0o=0;
  //----- 0-10% Lambda-K0 -----
  td2dVec tLamK00010Sys = {{tLamK00010mT,0.015,tLamK00010R,tLamK00010RerrSys}};
  td2dVec tLamK00010Stat = {{tLamK00010mT,0.,tLamK00010R,tLamK00010Rerr}};
  DrawPoints("GraphLamK00010",tLamK00010Sys,tLamK00010Stat,myRedT,myRed,tMarkerStyleLamK0,tMarkerSize,tMarkerStyleLamK0o);


  //----- 10-30% Lambda-K0 -----
  td2dVec tLamK01030Sys = {{tLamK01030mT,0.015,tLamK01030R,tLamK01030RerrSys}};
  td2dVec tLamK01030Stat = {{tLamK01030mT,0.,tLamK01030R,tLamK01030Rerr}};
  DrawPoints("GraphLamK01030",tLamK01030Sys,tLamK01030Stat,myGreenT,myGreen,tMarkerStyleLamK0,tMarkerSize,tMarkerStyleLamK0o);


  //----- 30-50% Lambda-K0 -----
  td2dVec tLamK03050Sys = {{tLamK03050mT,0.015,tLamK03050R,tLamK03050RerrSys}};
  td2dVec tLamK03050Stat = {{tLamK03050mT,0.,tLamK03050R,tLamK03050Rerr}};
  DrawPoints("GraphLamK03050",tLamK03050Sys,tLamK03050Stat,myBlueT,myBlue,tMarkerStyleLamK0,tMarkerSize,tMarkerStyleLamK0o);



  if(!bRunAveragedKchPKchM)
  {
//---------------------------------------- Lambda-KchP --------------------------------------
    if(!bOutlinePoints) tMarkerStyleLamKchPo=0;
    //----- 0-10% Lambda-K0 -----
    td2dVec tLamKchP0010Sys = {{tLamKchP0010mT,0.015,tLamKchP0010R,tLamKchP0010RerrSys}};
    td2dVec tLamKchP0010Stat = {{tLamKchP0010mT,0.,tLamKchP0010R,tLamKchP0010Rerr}};
    DrawPoints("GraphLamKchP0010",tLamKchP0010Sys,tLamKchP0010Stat,myRedT,myRed,tMarkerStyleLamKchP,tMarkerSize,tMarkerStyleLamKchPo);


    //----- 10-30% Lambda-K0 -----
    td2dVec tLamKchP1030Sys = {{tLamKchP1030mT,0.015,tLamKchP1030R,tLamKchP1030RerrSys}};
    td2dVec tLamKchP1030Stat = {{tLamKchP1030mT,0.,tLamKchP1030R,tLamKchP1030Rerr}};
    DrawPoints("GraphLamKchP1030",tLamKchP1030Sys,tLamKchP1030Stat,myGreenT,myGreen,tMarkerStyleLamKchP,tMarkerSize,tMarkerStyleLamKchPo);


    //----- 30-50% Lambda-K0 -----
    td2dVec tLamKchP3050Sys = {{tLamKchP3050mT,0.015,tLamKchP3050R,tLamKchP3050RerrSys}};
    td2dVec tLamKchP3050Stat = {{tLamKchP3050mT,0.,tLamKchP3050R,tLamKchP3050Rerr}};
    DrawPoints("GraphLamKchP3050",tLamKchP3050Sys,tLamKchP3050Stat,myBlueT,myBlue,tMarkerStyleLamKchP,tMarkerSize,tMarkerStyleLamKchPo);

//---------------------------------------- Lambda-KchM --------------------------------------
    if(!bOutlinePoints) tMarkerStyleLamKchMo=0;
    //----- 0-10% Lambda-K0 -----
    td2dVec tLamKchM0010Sys = {{tLamKchM0010mT,0.015,tLamKchM0010R,tLamKchM0010RerrSys}};
    td2dVec tLamKchM0010Stat = {{tLamKchM0010mT,0.,tLamKchM0010R,tLamKchM0010Rerr}};
    DrawPoints("GraphLamKchM0010",tLamKchM0010Sys,tLamKchM0010Stat,myRedT,myRed,tMarkerStyleLamKchM,tMarkerSize,tMarkerStyleLamKchMo);


    //----- 10-30% Lambda-K0 -----
    td2dVec tLamKchM1030Sys = {{tLamKchM1030mT,0.015,tLamKchM1030R,tLamKchM1030RerrSys}};
    td2dVec tLamKchM1030Stat = {{tLamKchM1030mT,0.,tLamKchM1030R,tLamKchM1030Rerr}};
    DrawPoints("GraphLamKchM1030",tLamKchM1030Sys,tLamKchM1030Stat,myGreenT,myGreen,tMarkerStyleLamKchM,tMarkerSize,tMarkerStyleLamKchMo);


    //----- 30-50% Lambda-K0 -----
    td2dVec tLamKchM3050Sys = {{tLamKchM3050mT,0.015,tLamKchM3050R,tLamKchM3050RerrSys}};
    td2dVec tLamKchM3050Stat = {{tLamKchM3050mT,0.,tLamKchM3050R,tLamKchM3050Rerr}};
    DrawPoints("GraphLamKchM3050",tLamKchM3050Sys,tLamKchM3050Stat,myBlueT,myBlue,tMarkerStyleLamKchM,tMarkerSize,tMarkerStyleLamKchMo);
  }
  else /*if(bRunAveragedKchPKchM)*/
  {
//-------------------------------- Average Lam-KchP and LamKchM -----------------------------
    if(!bOutlinePoints) tMarkerStyleLamKchAvgo=0;
    //----- 0-10% Lambda-K0 -----
    td2dVec tLamKchAvg0010Sys = {{tLamKchAvg0010mT,0.015,tLamKchAvg0010R,tLamKchAvg0010RerrSys}};
    td2dVec tLamKchAvg0010Stat = {{tLamKchAvg0010mT,0.,tLamKchAvg0010R,tLamKchAvg0010Rerr}};
    DrawPoints("GraphLamKchAvg0010",tLamKchAvg0010Sys,tLamKchAvg0010Stat,myRedT,myRed,tMarkerStyleLamKchAvg,tMarkerSize,tMarkerStyleLamKchAvgo);


    //----- 10-30% Lambda-K0 -----
    td2dVec tLamKchAvg1030Sys = {{tLamKchAvg1030mT,0.015,tLamKchAvg1030R,tLamKchAvg1030RerrSys}};
    td2dVec tLamKchAvg1030Stat = {{tLamKchAvg1030mT,0.,tLamKchAvg1030R,tLamKchAvg1030Rerr}};
    DrawPoints("GraphLamKchAvg1030",tLamKchAvg1030Sys,tLamKchAvg1030Stat,myGreenT,myGreen,tMarkerStyleLamKchAvg,tMarkerSize,tMarkerStyleLamKchAvgo);


    //----- 30-50% Lambda-K0 -----
    td2dVec tLamKchAvg3050Sys = {{tLamKchAvg3050mT,0.015,tLamKchAvg3050R,tLamKchAvg3050RerrSys}};
    td2dVec tLamKchAvg3050Stat = {{tLamKchAvg3050mT,0.,tLamKchAvg3050R,tLamKchAvg3050Rerr}};
    DrawPoints("GraphLamKchAvg3050",tLamKchAvg3050Sys,tLamKchAvg3050Stat,myBlueT,myBlue,tMarkerStyleLamKchAvg,tMarkerSize,tMarkerStyleLamKchAvgo);
  }




//---------------------------- Save file ----------------------------------------------------
  if(bSaveImage) canmtcomb->SaveAs(tSaveName);


//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
