#include "DrawmTScaling.h"
#include "CompareFittingMethods.h"
#include "FitValuesLatexTableHelperWriter.h"
#include "FitValuesWriterwSysErrs.h"


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
                int aMarkerColorSys, int aMarkerColorStat, int aMarkerStyle, Size_t aMarkerSize, int aMarkerOutlineStyle=0, bool aDrawSys=true)
{
  //First, make sure there are equal number systematic and statistical data points
  assert(aDataPointsWithSysErrors.size() == aDataPointsWithStatErrors.size());
  unsigned int tNDataPoints = aDataPointsWithSysErrors.size();

  //Next, make sure all points in each collection are of equal size
  //  i.e. make sure all points are set up for TGraphErrors or TGraphAsymmErrors
  for(unsigned int i=1; i<tNDataPoints; i++) assert(aDataPointsWithSysErrors[i-1].size() == aDataPointsWithSysErrors[i].size());
  for(unsigned int i=1; i<tNDataPoints; i++) assert(aDataPointsWithStatErrors[i-1].size() == aDataPointsWithStatErrors[i].size());

  TGraphAsymmErrors* tGr;
  //----------------------- Draw systematic error boxes first -------------------------------------------------
  if(aDrawSys)
  {
    tGr = new TGraphAsymmErrors(tNDataPoints);
    tGr->SetName(aName+TString("Sys"));
    tGr->SetFillColor(aMarkerColorSys);
    tGr->SetFillStyle(1000);
    tGr->SetLineColor(0);
    tGr->SetLineWidth(0);
    SetDataPoints(tGr, aDataPointsWithSysErrors);
    tGr->Draw("e2");
  }

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



//___________________________________________________________________________________
vector<double> GetPlotRadiusInfo(TString aMasterFileLocation, TString aSystematicsFileLocation, TString aFitInfoTString, AnalysisType aAnType, CentralityType aCentType)
{
  FitParameter* tFitParam = FitValuesWriterwSysErrs::GetFitParameterSys(aMasterFileLocation, aSystematicsFileLocation, aFitInfoTString, aAnType, aCentType, kRadius);
  td1dVec tReturnRadiusInfo = td1dVec{tFitParam->GetFitValue(), tFitParam->GetFitValueError(), tFitParam->GetFitValueSysError()};
  return tReturnRadiusInfo;
}


//___________________________________________________________________________________
double CalculateWeightermT(td1dVec &aLamK0mTs, td1dVec &aLamK0Weights,
                           td1dVec &aLamKchPmTs, td1dVec &aLamKchPWeights,
                           td1dVec &aLamKchMmTs, td1dVec &aLamKchMWeights)
{
  assert(aLamK0mTs.size()==aLamK0Weights.size());
  assert(aLamKchPmTs.size()==aLamKchPWeights.size());
  assert(aLamKchMmTs.size()==aLamKchMWeights.size());

  assert(aLamK0mTs.size()==aLamKchPmTs.size());
  assert(aLamK0mTs.size()==aLamKchMmTs.size());

  unsigned int tNVals = aLamK0mTs.size();
  double tNum=0., tDen=0.;
  
  for(unsigned int i=0; i<tNVals; i++)
  {
    tNum += aLamK0mTs[i]*aLamK0Weights[i];
    tDen += aLamK0Weights[i];
  } 
  for(unsigned int i=0; i<tNVals; i++)
  {
    tNum += aLamKchPmTs[i]*aLamKchPWeights[i];
    tDen += aLamKchPWeights[i];
  } 
  for(unsigned int i=0; i<tNVals; i++)
  {
    tNum += aLamKchMmTs[i]*aLamKchMWeights[i];
    tDen += aLamKchMWeights[i];
  } 

  double tWeightedAvg = tNum/tDen;
  return tWeightedAvg;
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

  TString tResultsDate = "20180505";

  bool bSaveImage = false;
  bool bMakeOthersTransparent = true;
  bool bOutlinePoints = true;
  bool bDrawSysErrs = true;
  bool bStripResStamp = true;


  IncludeResidualsType tIncResType = kInclude3Residuals;
  ResPrimMaxDecayType tResPrimMaxDecayType = k10fm;

  TString tSaveFileType = "pdf";  //Needs to be pdf for systematics to be transparent!
  TString tSaveName = "mTscaling_MinvCalcv2";


//TRIPLE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  vector<NonFlatBgdFitType> tNonFlatBgdFitTypes{kLinear, kLinear,
                                                kPolynomial, kPolynomial, kPolynomial, kPolynomial};

  TString tFitInfoTString = 
                                                                 FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypes, 
                                                                                                      tIncResType, tResPrimMaxDecayType, 
                                                                                                      kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                      false, false, false, false, false, 
                                                                                                      true, false, false, true, 
                                                                                                      true, true);

  //For case of bDrawSysErrs = false, still need to grab any systematic error bars
  //Just easier to implement this way with pre existing functionality
  TString tResultsDate_Defualt = "20180505";
  TString tFitInfoTString_Default = 
                                                                 FitValuesWriter::BuildFitInfoTString(true, true, tNonFlatBgdFitTypes, 
                                                                                                      kInclude3Residuals, k10fm, 
                                                                                                      kUseXiDataAndCoulombOnlyInterp, false, 
                                                                                                      false, false, false, false, false, 
                                                                                                      true, false, false, true, 
                                                                                                      true, true);
  TString tSystematicsFileLocation = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_%s/%s/Systematics/FinalFitSystematics_wFitRangeSys%s.txt", tResultsDate_Defualt.Data(), tFitInfoTString_Default.Data(), tFitInfoTString_Default.Data());
  if(bDrawSysErrs)
  {
    tSystematicsFileLocation = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_%s/%s/Systematics/FinalFitSystematics_wFitRangeSys%s.txt", tResultsDate.Data(), tFitInfoTString.Data(), tFitInfoTString.Data());
  }
  TString tSaveDirBase = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_%s/%s/Comparisons/", tResultsDate.Data(), tFitInfoTString.Data());

  cout << "tFitInfoTString = " << tFitInfoTString << endl << endl;


  TString tMasterFileLocation_LamKch = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_%s/MasterFitResults_%s.txt", tResultsDate.Data(), tResultsDate.Data());
  TString tMasterFileLocation_LamK0 = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamK0_%s/MasterFitResults_%s.txt", tResultsDate.Data(), tResultsDate.Data());




  TString tCanNameMod, tLegInfo;
  if(tIncResType==kIncludeNoResiduals)
  {
    tCanNameMod = TString("_NoRes");
    tLegInfo = TString("No Res.");
  }
  else if(tIncResType==kInclude3Residuals)
  {
    tCanNameMod = TString("_3Res");
    tLegInfo = TString("3 Res.");
  }
  else if(tIncResType==kInclude10Residuals)
  {
    tCanNameMod = TString("_10Res");
    tLegInfo = TString("10 Res.");
  }
  else assert(0);

  //------------------------------------------------------

  if(bOutlinePoints) tSaveName += TString("_OutlinedPoints");
  if(bMakeOthersTransparent) tSaveName += TString("_OthersTransparent");

  tSaveName += tCanNameMod;
  if(bStripResStamp) tSaveName += TString("_NoResStamp");
  tSaveName += TString::Format(".%s", tSaveFileType.Data());
  
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


  const int tMarkerStyleLamK = 21;
  int tMarkerStyleLamKo = 25;
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
//---------------------------------------- mT values and weights ----------------------------------------
  vector<double> tLamK00010mT{1.603,1.604,1.603,1.602};  //m_inv
  vector<double> tLamK01030mT{1.588,1.588,1.589,1.587};  //m_inv
  vector<double> tLamK03050mT{1.548,1.546,1.549,1.546};  //m_inv

  vector<double> tLamK00010Weights{1.0, 1.0, 1.0, 1.0};
  vector<double> tLamK01030Weights{1.0, 1.0, 1.0, 1.0};
  vector<double> tLamK03050Weights{1.0, 1.0, 1.0, 1.0};

  //-----

  vector<double> tLamKchP0010mT{1.439,1.438,1.442,1.437}; //m_inv
  vector<double> tLamKchP1030mT{1.427,1.423,1.431,1.425}; //m_inv
  vector<double> tLamKchP3050mT{1.390,1.382,1.395,1.385}; //m_inv

  vector<double> tLamKchP0010Weights{1.0, 1.0, 1.0, 1.0};
  vector<double> tLamKchP1030Weights{1.0, 1.0, 1.0, 1.0};
  vector<double> tLamKchP3050Weights{1.0, 1.0, 1.0, 1.0};

  //-----

  vector<double> tLamKchM0010mT{1.441,1.439,1.442,1.440}; //m_inv
  vector<double> tLamKchM1030mT{1.426,1.426,1.428,1.429}; //m_inv
  vector<double> tLamKchM3050mT{1.387,1.389,1.389,1.392}; //m_inv

  vector<double> tLamKchM0010Weights{1.0, 1.0, 1.0, 1.0};
  vector<double> tLamKchM1030Weights{1.0, 1.0, 1.0, 1.0};
  vector<double> tLamKchM3050Weights{1.0, 1.0, 1.0, 1.0};






//---------------------------------------- LambdaK --------------------------------------
  double tLamK0010mT = CalculateWeightermT(tLamK00010mT, tLamK00010Weights,
                                           tLamKchP0010mT, tLamKchP0010Weights, 
                                           tLamKchM0010mT, tLamKchM0010Weights);
  vector<double> RInfo_LamK_0010;
  RInfo_LamK_0010 = GetPlotRadiusInfo(tMasterFileLocation_LamKch, tSystematicsFileLocation, tFitInfoTString, kLamKchP, k0010);
  double tLamK0010R = RInfo_LamK_0010[0];
  double tLamK0010Rerr = RInfo_LamK_0010[1];
  double tLamK0010RerrSys = RInfo_LamK_0010[2];

  double tLamK1030mT = CalculateWeightermT(tLamK01030mT, tLamK01030Weights,
                                           tLamKchP1030mT, tLamKchP1030Weights, 
                                           tLamKchM1030mT, tLamKchM1030Weights);
  vector<double> RInfo_LamK_1030;
  RInfo_LamK_1030 = GetPlotRadiusInfo(tMasterFileLocation_LamKch, tSystematicsFileLocation, tFitInfoTString, kLamKchP, k1030);
  double tLamK1030R = RInfo_LamK_1030[0];
  double tLamK1030Rerr = RInfo_LamK_1030[1];
  double tLamK1030RerrSys = RInfo_LamK_1030[2];

  double tLamK3050mT = CalculateWeightermT(tLamK03050mT, tLamK03050Weights,
                                           tLamKchP3050mT, tLamKchP3050Weights, 
                                           tLamKchM3050mT, tLamKchM3050Weights);
  vector<double> RInfo_LamK_3050;
  RInfo_LamK_3050 = GetPlotRadiusInfo(tMasterFileLocation_LamKch, tSystematicsFileLocation, tFitInfoTString, kLamKchP, k3050);
  double tLamK3050R = RInfo_LamK_3050[0];
  double tLamK3050Rerr = RInfo_LamK_3050[1];
  double tLamK3050RerrSys = RInfo_LamK_3050[2];



//------------------------------------------------------------------------------------------- 
//cout values so there's no confusion

cout << "RInfo_LamK_0010 = " << RInfo_LamK_0010[0] << " +- " << RInfo_LamK_0010[1] << " +- " << RInfo_LamK_0010[2] << endl;
cout << "RInfo_LamK_1030 = " << RInfo_LamK_1030[0] << " +- " << RInfo_LamK_1030[1] << " +- " << RInfo_LamK_1030[2] << endl;
cout << "RInfo_LamK_3050 = " << RInfo_LamK_3050[0] << " +- " << RInfo_LamK_3050[1] << " +- " << RInfo_LamK_3050[2] << endl;
cout << endl << endl;

//------------------------------------------------------------------------------------------- 

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
  DrawPoints("GraphPP0010",tPP0010Sys,tPP0010Stat,redT,red,20,tMarkerSize);

  //----- 10-30% -----
  DrawPoints("GraphPP1030",tPP1030Sys,tPP1030Stat,greenT,green,24,tMarkerSize);
  
  //----- 30-50% -----
  DrawPoints("GraphPP3050",tPP3050Sys,tPP3050Stat,blueT,blue,31,tMarkerSize);

  
//------------------------- Anti-Protons ----------------------------------------------------
  //----- 0-10% -----
  DrawPoints("GraphAPAP0010",tAPAP0010Sys,tAPAP0010Stat,redT,red,22,tMarkerSize);
  
  //----- 10-30% -----
  DrawPoints("GraphAPAP1030",tAPAP1030Sys,tAPAP1030Stat,greenT,green,26,tMarkerSize);

  //----- 30-50% -----
  DrawPoints("GraphAPAP3050",tAPAP3050Sys,tAPAP3050Stat,blueT,blue,33,tMarkerSize);

  
//------------------------ K0s-K0s ----------------------------------------------------------  
  //----- 0-10% -----
  DrawPoints("GraphK0sK0s0010",tK0sK0s0010Sys,tK0sK0s0010Stat,redT,red,23,tMarkerSize);

  //----- 10-30% -----   
  DrawPoints("GraphK0sK0s1030",tK0sK0s1030Sys,tK0sK0s1030Stat,greenT,green,32,tMarkerSize);

  //----- 30-50% -----
  DrawPoints("GraphK0sK0s3050",tK0sK0s3050Sys,tK0sK0s3050Stat,blueT,blue,27,tMarkerSize);


//----------------------------- Kch-Kch -----------------------------------------------------  
  //----- 0-10% -----
  DrawPoints("GraphKchKch0010",tKchKch0010Sys,tKchKch0010Stat,redT,red,21,tMarkerSize);
  
  //----- 10-30% -----
  DrawPoints("GraphKchKch1030",tKchKch1030Sys,tKchKch1030Stat,greenT,green,25,tMarkerSize);

  //----- 30-50% -----
  DrawPoints("GraphKchKch3050",tKchKch3050Sys,tKchKch3050Stat,blueT,blue,29,tMarkerSize);

  
//------------------------------ Pi-Pi ------------------------------------------------------
  //----- 0-10% -----
  DrawPoints("GraphPiPi0010",tPiPi0010Sys,tPiPi0010Stat,redT,red,34,tMarkerSize);

  //----- 10-30% -----
  DrawPoints("GraphPiPi1030",tPiPi1030Sys,tPiPi1030Stat,greenT,green,28,tMarkerSize);

  //----- 30-50% -----
  DrawPoints("GraphPiPi3050",tPiPi3050Sys,tPiPi3050Stat,blueT,blue,30,tMarkerSize);

  
//-------------------------------------------------------------------------------------------
//-----------------------------------Draw my points -----------------------------------------
//-------------------------------------------------------------------------------------------
//---------------------------------------- Lambda-K --------------------------------------
    if(!bOutlinePoints) tMarkerStyleLamKo=0;
    //----- 0-10% Lambda-K0 -----
    td2dVec tLamK0010Sys = {{tLamK0010mT,0.015,tLamK0010R,tLamK0010RerrSys}};
    td2dVec tLamK0010Stat = {{tLamK0010mT,0.,tLamK0010R,tLamK0010Rerr}};
    DrawPoints("GraphLamK0010",tLamK0010Sys,tLamK0010Stat,myRedT,myRed,tMarkerStyleLamK,tMarkerSize,tMarkerStyleLamKo,bDrawSysErrs);


    //----- 10-30% Lambda-K0 -----
    td2dVec tLamK1030Sys = {{tLamK1030mT,0.015,tLamK1030R,tLamK1030RerrSys}};
    td2dVec tLamK1030Stat = {{tLamK1030mT,0.,tLamK1030R,tLamK1030Rerr}};
    DrawPoints("GraphLamK1030",tLamK1030Sys,tLamK1030Stat,myGreenT,myGreen,tMarkerStyleLamK,tMarkerSize,tMarkerStyleLamKo,bDrawSysErrs);


    //----- 30-50% Lambda-K0 -----
    td2dVec tLamK3050Sys = {{tLamK3050mT,0.015,tLamK3050R,tLamK3050RerrSys}};
    td2dVec tLamK3050Stat = {{tLamK3050mT,0.,tLamK3050R,tLamK3050Rerr}};
    DrawPoints("GraphLamK3050",tLamK3050Sys,tLamK3050Stat,myBlueT,myBlue,tMarkerStyleLamK,tMarkerSize,tMarkerStyleLamKo,bDrawSysErrs);


//-------------------------------------------------------------------------------
  TLatex* tTex = new TLatex();
  tTex->SetTextAlign(12);
  tTex->SetLineWidth(2);
  tTex->SetTextFont(62);
  tTex->SetTextSize(0.06);

  TMarker *tMarker = new TMarker();
  tMarker->SetMarkerSize(2.0);
  tMarker->SetMarkerStyle(20);
  tMarker->SetMarkerColor(kBlack);

  if(!bStripResStamp) tTex->DrawLatex(0.35, 2., tLegInfo);
//  tMarker->SetMarkerStyle(tDrawingInfo.markerStyle);
//  tMarker->DrawMarker(0.3, 2.);



//---------------------------------- Draw Legend -------------------------------------------- 

    TLatex *   tex = new TLatex(1.175,9.4,"ALICE Pb-Pb #sqrt{#it{s}_{NN}} = 2.76 TeV");
    tex->SetTextFont(42);
    tex->SetTextSize(0.044);
    tex->SetLineWidth(2);
//    tex->Draw();
//    tex->DrawLatex(0.85,9.4,"ALICE Preliminary");
    tex = new TLatex();
    tex->SetTextAlign(12);
    tex->SetTextFont(42);
    tex->SetTextSize(0.04);
    tex->SetLineWidth(2);


    // species text and markers
    TMarker *marker = new TMarker();
    marker->SetMarkerSize(tMarkerSize);
  
    double tX1Text = 1.0;
    double tTextMarkerSep1 = 0.16;
    double tX2Text = 1.95;
    double tTextMarkerSep2 = 0.15;
    double tY1Text = 9.0;
    double tY1Sep = 0.6;

    double tShiftX = 0.;
    tShiftX = 0.1;
    tX1Text += tShiftX;

    tex->DrawLatex(tX1Text, tY1Text-0*tY1Sep, "#pi^{#pm}#pi^{#pm}");
    marker->SetMarkerStyle(28);//pions
    marker->DrawMarker(tX1Text+tTextMarkerSep1, tY1Text-0*tY1Sep);
  
    tex->DrawLatex(tX1Text, tY1Text-1*tY1Sep, "K^{#pm}K^{#pm}");
    marker->SetMarkerStyle(25);//Kch
    marker->DrawMarker(tX1Text+tTextMarkerSep1, tY1Text-1*tY1Sep);
  
    tex->DrawLatex(tX1Text, tY1Text-2*tY1Sep, "K_{S}^{0}K_{S}^{0}");
    marker->SetMarkerStyle(27);//K0s
    marker->DrawMarker(tX1Text+tTextMarkerSep1, tY1Text-2*tY1Sep);
  
    tex->DrawLatex(tX1Text, tY1Text-3*tY1Sep, "#bar{p}#bar{p}");
    marker->SetMarkerStyle(5);// antiprotons
    marker->DrawMarker(tX1Text+tTextMarkerSep1, tY1Text-3*tY1Sep);
  
    tex->DrawLatex(tX1Text, tY1Text-4*tY1Sep, "pp");
    marker->SetMarkerStyle(24);//protons
    marker->DrawMarker(tX1Text+tTextMarkerSep1, tY1Text-4*tY1Sep);



    tex->DrawLatex(tX1Text, tY1Text-5*tY1Sep, "#LambdaK");
    marker->SetMarkerStyle(tMarkerStyleLamK);
    marker->DrawMarker(tX1Text+tTextMarkerSep1, tY1Text-5*tY1Sep);

/*
    // centralities
    TLine line;
    line.SetLineWidth(2);
    line.SetLineColor(myRed);
    line.DrawLine(1.32+tShiftX, tY1Text-0.1-0*tY1Sep, 1.42+tShiftX, tY1Text-0.1-0*tY1Sep);
    line.SetLineColor(myGreen);
    line.DrawLine(1.32+tShiftX, tY1Text-0.1-1*tY1Sep, 1.42+tShiftX, tY1Text-0.1-1*tY1Sep);
    line.SetLineColor(myBlue);
    line.DrawLine(1.32+tShiftX, tY1Text-0.1-2*tY1Sep, 1.42+tShiftX, tY1Text-0.1-2*tY1Sep);

    tex->DrawLatex(1.1+tShiftX, tY1Text-0.1-0*tY1Sep, "0-10%");
    tex->DrawLatex(1.1+tShiftX, tY1Text-0.1-1*tY1Sep, "10-30%");
    tex->DrawLatex(1.1+tShiftX, tY1Text-0.1-2*tY1Sep, "30-50%");
*/

//---------------------------- Save file ----------------------------------------------------
  if(bSaveImage) 
  {
    canmtcomb->SaveAs(TString::Format("%s%s", tSaveDirBase.Data(), tSaveName.Data()));
  }

//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
