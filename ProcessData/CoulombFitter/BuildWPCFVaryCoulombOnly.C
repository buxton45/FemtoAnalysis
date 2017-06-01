#include "FitSharedAnalyses.h"
#include "CoulombFitter.h"
#include "TLegend.h"
#include "CanvasPartition.h"


#include "TColor.h"
#include <TStyle.h>
#include "TPaveText.h"
#include <TLatex.h>
#include "TGraphAsymmErrors.h"

#include "CanvasPartition.h"
class CanvasPartition;

CoulombFitter *myFitter = NULL;

//______________________________________________________________________________
void fcn(Int_t &npar, Double_t *gin, Double_t &f, Double_t *par, Int_t iflag)
{
  myFitter->CalculateChi2PML(npar,f,par);
//  myFitter->CalculateChi2PMLwMomResCorrection(npar,f,par);
//  myFitter->CalculateChi2(npar,f,par);
//  myFitter->CalculateFakeChi2(npar,f,par);
}


//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________


int main(int argc, char **argv)
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything

  ChronoTimer tFullTimer(kMin);
  tFullTimer.Start();
//-----------------------------------------------------------------------------

  //!!!!!!!!!!!!!!!! NOTE:  must set myFitter = to whichever LednickyFitter object I want to use
//-----------------------------------------------------------------------------
//Be sure to set the following...

  TString tFileDirectory = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cXicKch_20170505_ignoreOnFlyStatus/";
  TString tFileLocationBase = tFileDirectory + TString("Results_cXicKch_20170505_ignoreOnFlyStatus");
  bool bSaveImage = true;
  bool bStretchCanvas = true;

  AnalysisType tAnType, tConjType;
/*
  tAnType = kXiKchP;
  tConjType = kAXiKchM;
*/

  tAnType = kXiKchM;
  tConjType = kAXiKchP;


  //--Following for simulated curves---------
  int tNbinsK = 16;
  double tKMin = 0.;
  double tKMax = 0.16;
  double tBinSize = (tKMax-tKMin)/tNbinsK;
  //-----------------------------------------


//-----------------------------------------------------------------------------
  AnalysisRunType tAnalysisRunType = kTrain;
  int tNPartialAnalysis = 5;
  if(tAnalysisRunType==kTrain || tAnalysisRunType==kTrainSys) tNPartialAnalysis = 2;

  FitPairAnalysis* tPairAn = new FitPairAnalysis(tFileLocationBase,tAnType,k0010,tAnalysisRunType,tNPartialAnalysis);
  FitPairAnalysis* tConjAn = new FitPairAnalysis(tFileLocationBase,tConjType,k0010,tAnalysisRunType,tNPartialAnalysis);

  vector<FitPairAnalysis*> tVecOfPairAn;
  tVecOfPairAn.push_back(tPairAn);
  tVecOfPairAn.push_back(tConjAn);

  FitSharedAnalyses* tSharedAn = new FitSharedAnalyses(tVecOfPairAn);

  tSharedAn->RebinAnalyses(2);
  tSharedAn->CreateMinuitParameters();
//-----------------------------------------------------------------------------
  int tColor, tColorTransparent;
  if(tAnType==kXiKchP) tColor=kRed+1;
  else if(tAnType==kXiKchM) tColor=kBlue+1;
  else tColor=1;

  tColorTransparent = TColor::GetColorTransparent(tColor,0.2);

  TH1* tData = tPairAn->GetKStarCf();
  TH1* tDatawSysErrors = (TH1*)tSharedAn->GetFitPairAnalysis(0)->GetCfwSysErrors();
    tDatawSysErrors->SetFillColor(tColorTransparent);
    tDatawSysErrors->SetFillStyle(1000);
    tDatawSysErrors->SetLineColor(0);
    tDatawSysErrors->SetLineWidth(0);

  TH1* tConjData = tConjAn->GetKStarCf();
  TH1* tConjDatawSysErrors = (TH1*)tSharedAn->GetFitPairAnalysis(1)->GetCfwSysErrors();
    tConjDatawSysErrors->SetFillColor(tColorTransparent);
    tConjDatawSysErrors->SetFillStyle(1000);
    tConjDatawSysErrors->SetLineColor(0);
    tConjDatawSysErrors->SetLineWidth(0);
//-----------------------------------------------------------------------------
  CoulombFitter* tFitter = new CoulombFitter(tSharedAn,tKMax);
    tFitter->SetIncludeSingletAndTriplet(false);
    tFitter->SetApplyMomResCorrection(false);

  TString tFileLocationInterpHistos;
  if(tAnType==kAXiKchP || tAnType==kXiKchM) tFileLocationInterpHistos = "InterpHistsRepulsive";
  else if(tAnType==kXiKchP || tAnType==kAXiKchM) tFileLocationInterpHistos = "InterpHistsAttractive";
  tFitter->LoadInterpHistFile(tFileLocationInterpHistos);

  tFitter->SetUseRandomKStarVectors(true);
  tFitter->SetUseStaticPairs(true,50000,tBinSize);

  tFitter->GetFitSharedAnalyses()->GetMinuitObject()->SetFCN(fcn);
  myFitter = tFitter;
  //-------------------------------------------

  double tLambdaHighCurve = 0.9;
  double tRadiusHighCurve = 1.;

  double tLambdaLowCurve = 0.1;
  double tRadiusLowCurve = 10.;

  if(tAnType==kAXiKchP || tAnType==kXiKchM)
  {
    double tTmpLambda = tLambdaHighCurve;
    double tTmpRadius = tRadiusHighCurve;

    tLambdaHighCurve = tLambdaLowCurve;
    tRadiusHighCurve = tRadiusLowCurve;

    tLambdaLowCurve = tTmpLambda;
    tRadiusLowCurve = tTmpRadius;
  }

  TH1* tCoulombOnlyHistSampleHigh;
  TH1* tCoulombOnlyHistSampleLow;
  TGraphAsymmErrors* tCoulombOnlyGraph;

  tCoulombOnlyHistSampleHigh = tFitter->CreateFitHistogramSampleComplete("tCoulombOnlyHistSampleHigh", tAnType, tNbinsK, tKMin, tKMax, tLambdaHighCurve, tRadiusHighCurve, 0., 0., 0., 0., 0., 0., 1.0);
  tCoulombOnlyHistSampleLow = tFitter->CreateFitHistogramSampleComplete("tCoulombOnlyHistSampleLow", tAnType, tNbinsK, tKMin, tKMax, tLambdaLowCurve, tRadiusLowCurve, 0., 0., 0., 0., 0., 0., 1.0);
    
  assert(tCoulombOnlyHistSampleLow->GetBinWidth(1) == tCoulombOnlyHistSampleHigh->GetBinWidth(1));

  int tNPoints = tNbinsK;
  double tX[tNPoints], tXErrLow[tNPoints], tXErrHigh[tNPoints];
  double tY[tNPoints], tYErrLow[tNPoints], tYErrHigh[tNPoints];

  for(int i=1; i<=tNPoints; i++)
  {
    tX[i-1] = tCoulombOnlyHistSampleHigh->GetBinCenter(i); //TODO yes or no?
    tY[i-1] = tCoulombOnlyHistSampleHigh->GetBinContent(i);

    tXErrLow[i-1] = 0.;
    tXErrHigh[i-1] = 0.;

    tYErrLow[i-1] = TMath::Abs(tCoulombOnlyHistSampleHigh->GetBinContent(i) - tCoulombOnlyHistSampleLow->GetBinContent(i));
    tYErrHigh[i-1] = 0.;
  }

  tCoulombOnlyGraph = new TGraphAsymmErrors(tNPoints, tX, tY, tXErrLow, tXErrHigh, tYErrLow, tYErrHigh);
  tCoulombOnlyGraph->SetFillStyle(1000);
  int tColorCoulombOnly = TColor::GetColorTransparent(kMagenta,0.2);
  tCoulombOnlyGraph->SetFillColor(tColorCoulombOnly);


//-----------------------------------------------------------------------------

  TString tCanvasName = TString("canKStarCfswCoulombOnly");
  if(tAnType==kXiKchP) tCanvasName += TString("XiKchPwConj");
  else if(tAnType==kXiKchM) tCanvasName += TString("XiKchMwConj");
  else assert(0);

  tCanvasName += TString(cCentralityTags[k0010]);

  double tXLow=0, tXHigh=0, tYLow=0, tYHigh=0;
  tXLow = -0.02;
  tXHigh = 0.16;
  if(tAnType==kXiKchP)
  {
    tYLow = 0.92;
//    tYHigh = 2.52;
    tYHigh = 1.72;
  }
  else if(tAnType==kXiKchM)
  {
    tYLow = 0.28;
    tYHigh = 1.14;
  }
  else assert(0);

  int tNx=2, tNy=1;
  float tMarginLeft = 0.10;
  float tMarginRight = 0.0025;
  float tMarginBottom = 0.09;
  float tMarginTop = 0.0025;
  if(bStretchCanvas) tMarginBottom = 0.15;

  CanvasPartition* tCanPart = new CanvasPartition(tCanvasName, tNx, tNy, tXLow, tXHigh, tYLow, tYHigh, tMarginLeft, tMarginRight, tMarginBottom, tMarginTop);
  tCanPart->SetDrawOptStat(false);
  if(bStretchCanvas) tCanPart->GetCanvas()->SetCanvasSize(1400,500);

  //--------------------------------------------------------------
  double tMarkerSize = 0.5;
  float tLabelSize = 0.175, tLabelOffset=0.005;
  if(bStretchCanvas)
  {
    tMarkerSize *= 2;
    tLabelSize *= 1.5;
  }
  tCanPart->AddGraph(0, 0, tData, "", 20, tColor, tMarkerSize, "ex0", tLabelSize, tLabelOffset);
  tCanPart->AddGraph(0, 0, tDatawSysErrors, "", 20, tColorTransparent, tMarkerSize, "e2psame", tLabelSize, tLabelOffset);
  tCanPart->AddGraph(0, 0, tCoulombOnlyGraph, "", 20, tColorCoulombOnly, tMarkerSize, "3", tLabelSize, tLabelOffset);
  tCanPart->AddGraph(0, 0, tData, "", 20, tColor, tMarkerSize, "ex0same", tLabelSize, tLabelOffset);

  double tSystemInfoSize = 20;
  if(bStretchCanvas) tSystemInfoSize *= 2;
  TString tTextAnType = TString(cAnalysisRootTags[tAnType]);
  TString tTextCentrality = TString(cPrettyCentralityTags[k0010]);
  TString tCombinedText = tTextAnType + TString("  ") +  tTextCentrality;
  TPaveText* tCombined;
  if(tAnType==kXiKchP) tCombined = tCanPart->SetupTPaveText(tCombinedText,0,0,0.70,0.70,0.15,0.10,63,tSystemInfoSize);
  else if(tAnType==kXiKchM) tCombined = tCanPart->SetupTPaveText(tCombinedText,0,0,0.70,0.65,0.15,0.10,63,tSystemInfoSize);
  else assert(0);
  tCanPart->AddPadPaveText(tCombined,0,0);

  TString tBandInfo = "Coulomb Only Band";
  TString tBandTopInfo = TString::Format("Max: #lambda = %0.1f \t R = %0.1f fm", tLambdaHighCurve, tRadiusHighCurve);
  TString tBandBottomInfo = TString::Format("Min: #lambda = %0.1f \t R = %0.1f fm", tLambdaLowCurve, tRadiusLowCurve);
  TPaveText* tBand = tCanPart->SetupTPaveText(tBandInfo,0,0,0.60,0.35,0.15,0.20,63,(tSystemInfoSize/2));
  tBand->AddText(tBandTopInfo);
  tBand->AddText(tBandBottomInfo);
  tBand->SetTextAlign(12);
  tCanPart->AddPadPaveText(tBand,0,0);

  double tAliceXmin = 0.30, tAliceYmin = 0.91, tAliceWidth = 0.40, tAliceHeight = 0.08, tAliceFont = 43, tAliceSize = 17;
  if(bStretchCanvas)
  {
    tAliceSize *= 1.7;
    tAliceYmin = 0.89;
  }

  TString tTextAlicePrelim = TString("ALICE Preliminary");
  TPaveText* tAlicePrelim = tCanPart->SetupTPaveText(tTextAlicePrelim,0,0,tAliceXmin, tAliceYmin, tAliceWidth, tAliceHeight, tAliceFont, tAliceSize);
  tCanPart->AddPadPaveText(tAlicePrelim,0,0);

  //------------------------------------------------------------
  tCanPart->AddGraph(1, 0, tConjData, "", 20, tColor, tMarkerSize, "ex0", tLabelSize, tLabelOffset);
  tCanPart->AddGraph(1, 0, tConjDatawSysErrors, "", 20, tColorTransparent, tMarkerSize, "e2psame", tLabelSize, tLabelOffset);
  tCanPart->AddGraph(1, 0, tCoulombOnlyGraph, "", 20, tColorCoulombOnly, tMarkerSize, "3", tLabelSize, tLabelOffset);
  tCanPart->AddGraph(1, 0, tConjData, "", 20, tColor, tMarkerSize, "ex0same", tLabelSize, tLabelOffset);

  TString tTextConjType = TString(cAnalysisRootTags[tConjType]);
  TString tCombinedTextConj = tTextConjType + TString("  ") +  tTextCentrality;
  TPaveText* tCombinedConj;
  if(tAnType==kXiKchP) tCombinedConj = tCanPart->SetupTPaveText(tCombinedTextConj,1,0,0.70,0.70,0.15,0.10,63,tSystemInfoSize);
  else if(tAnType==kXiKchM) tCombinedConj = tCanPart->SetupTPaveText(tCombinedTextConj,1,0,0.70,0.65,0.15,0.10,63,tSystemInfoSize);
  else assert(0);
  tCanPart->AddPadPaveText(tCombinedConj,1,0);

  TString tTextSysInfo = TString("Pb-Pb #sqrt{#it{s}_{NN}} = 2.76 TeV");
  TPaveText* tSysInfo = tCanPart->SetupTPaveText(tTextSysInfo,1,0,tAliceXmin, tAliceYmin, tAliceWidth, tAliceHeight, tAliceFont, tAliceSize);
  tCanPart->AddPadPaveText(tSysInfo,1,0);

  //------------------------------------------------------------

  tCanPart->SetDrawUnityLine(true);
  tCanPart->DrawAll();
  if(!bStretchCanvas)
  {
    tCanPart->DrawXaxisTitle("#it{k}* (GeV/#it{c})", 43, 30, 0.315, 0.03);
    tCanPart->DrawYaxisTitle("#it{C}(#it{k}*)",43, 30, 0.05, 0.85);
  }
  else
  {
    tCanPart->DrawXaxisTitle("#it{k}* (GeV/#it{c})", 43, 50, 0.315, 0.03);
    tCanPart->DrawYaxisTitle("#it{C}(#it{k}*)",43, 50, 0.05, 0.75);
  }
  //-------------------------------------------------------------
  if(bSaveImage)
  {
    TString tSaveName = tFileDirectory;
    tSaveName += TString("WPCFCoulombOnlyCurves_") + cAnalysisBaseTags[tAnType] + TString(cCentralityTags[k0010]);
    if(bStretchCanvas) tSaveName += TString("_Stretch");
    tSaveName += TString(".pdf");
    tCanPart->GetCanvas()->SaveAs(tSaveName);
  }

  delete tFitter;

  delete tSharedAn;
  delete tPairAn;
  delete tConjAn;

//-------------------------------------------------------------------------------
  tFullTimer.Stop();
  cout << "Finished program: ";
  tFullTimer.PrintInterval();

  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.
  return 0;
}
