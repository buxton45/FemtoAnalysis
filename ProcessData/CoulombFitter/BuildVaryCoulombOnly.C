#include "FitSharedAnalyses.h"
#include "CoulombFitter.h"
#include "TLegend.h"
#include "CanvasPartition.h"


#include "TColor.h"
#include <TStyle.h>
#include "TPaveText.h"
#include <TLatex.h>
#include "TGraphAsymmErrors.h"

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
TCanvas* DrawKStarCfs(TH1* aData, TH1* aDatawSysErrors, AnalysisType aAnType=kXiKchP, CentralityType aCentType=k0010)
{
  TString tCanvasName = TString("canKStarCfwFits");
  if(aAnType==kXiKchP) tCanvasName += TString("XiKchP");
  else if(aAnType==kAXiKchM) tCanvasName += TString("AXiKchM");
  else if(aAnType==kXiKchM) tCanvasName += TString("XiKchM");
  else if(aAnType==kAXiKchP) tCanvasName += TString("AXiKchP");
  else assert(0);

  tCanvasName += TString(cCentralityTags[aCentType]);

  double tXLow=0, tXHigh=0, tYLow=0, tYHigh=0;
  tXLow = 0.0;
  tXHigh = 0.15;
  if(aAnType==kXiKchP || aAnType==kAXiKchM)
  {
    tYLow = 0.92;
    tYHigh = 2.5;
  }
  else if(aAnType==kXiKchM || aAnType==kAXiKchP)
  {
    tYLow = 0.38;
    tYHigh = 1.04;
  }
  else assert(0);

  TCanvas* tCan = new TCanvas(tCanvasName,tCanvasName);
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  int tColor;
  Int_t tColorTransparent;
  if(aAnType==kXiKchP || aAnType==kAXiKchM) tColor=kRed+1;
  else if(aAnType==kXiKchM || aAnType==kAXiKchP) tColor=kBlue+1;
  else tColor=1;

  tColorTransparent = TColor::GetColorTransparent(tColor,0.2);  //TODO transparency doesn't work with .eps!!!!!!!!!!

  aData->GetXaxis()->SetTitle("#it{k}* (GeV/#it{c})");
  aData->GetXaxis()->SetRangeUser(tXLow,tXHigh);
  aData->GetXaxis()->SetTitleSize(0.055);
  aData->GetXaxis()->SetTitleOffset(0.8);

  aData->GetYaxis()->SetTitle("#it{C}(#it{k}*)");
  aData->GetYaxis()->SetRangeUser(tYLow,tYHigh);
  aData->GetYaxis()->SetTitleSize(0.0575);
  aData->GetYaxis()->SetTitleOffset(0.8);

  aData->SetMarkerStyle(20);
  aData->SetMarkerColor(tColor);
  aData->SetLineColor(tColor);

  aDatawSysErrors->SetFillColor(tColorTransparent);
  aDatawSysErrors->SetMarkerColor(tColorTransparent);
  aDatawSysErrors->SetFillStyle(1000);
  aDatawSysErrors->SetLineColor(0);
  aDatawSysErrors->SetLineWidth(0);

  aData->Draw("ex0");
  aDatawSysErrors->Draw("e2psame");
  aData->Draw("ex0same");

  double tXaxisRangeLow;
  if(tXLow<0) tXaxisRangeLow = 0.;
  else tXaxisRangeLow = tXLow;
  TLine *tLine = new TLine(tXaxisRangeLow,1.,tXHigh,1.);
  tLine->SetLineColor(TColor::GetColorTransparent(kGray,0.75));
  tLine->Draw();

//  returnText->Draw();

  return tCan;
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

  TString tFileDirectory = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cXicKch_20170501/";
  TString tFileLocationBase = tFileDirectory + TString("Results_cXicKch_20170501");
  bool bSaveImage = true;
  bool bDrawErrorBands = true;

  AnalysisType tAnType;
  CentralityType tCentType;

  //tAnType = kXiKchP;
  //tAnType = kAXiKchM;

  tAnType = kXiKchM;
  //tAnType = kAXiKchP;

  tCentType = k0010;
  //tCentType = k1030;
  //tCentType = k3050;

  int tNbinsK = 15;
  double tKMin = 0.;
  double tKMax = 0.15;
  double tBinSize = (tKMax-tKMin)/tNbinsK;
  int tNPairsPerKStarBin = 50000;

  AnalysisRunType tAnalysisRunType = kTrain;
  int tNPartialAnalysis = 5;
  if(tAnalysisRunType==kTrain || tAnalysisRunType==kTrainSys) tNPartialAnalysis = 2;
   
  TString tAnBaseName = TString(cAnalysisBaseTags[tAnType]);
  TString tAnName0010 = tAnBaseName + TString(cCentralityTags[0]);


//-----------------------------------------------------------------------------

  FitPairAnalysis* tPairAn0010 = new FitPairAnalysis(tFileLocationBase,tAnType,k0010,tAnalysisRunType,tNPartialAnalysis);
  FitPairAnalysis* tPairAn1030 = new FitPairAnalysis(tFileLocationBase,tAnType,k1030,tAnalysisRunType,tNPartialAnalysis);
  FitPairAnalysis* tPairAn3050 = new FitPairAnalysis(tFileLocationBase,tAnType,k3050,tAnalysisRunType,tNPartialAnalysis);

  vector<FitPairAnalysis*> tVecOfPairAn;
  tVecOfPairAn.push_back(tPairAn0010);
  tVecOfPairAn.push_back(tPairAn1030);
  tVecOfPairAn.push_back(tPairAn3050);

  FitSharedAnalyses* tSharedAn = new FitSharedAnalyses(tVecOfPairAn);


  tSharedAn->RebinAnalyses(2);
  tSharedAn->CreateMinuitParameters();

  CoulombFitter* tFitter = new CoulombFitter(tSharedAn,tKMax);
    tFitter->SetIncludeSingletAndTriplet(false);
    tFitter->SetApplyMomResCorrection(false);


  TString tFileLocationInterpHistos;
  if(tAnType==kAXiKchP || tAnType==kXiKchM) tFileLocationInterpHistos = "InterpHistsRepulsive";
  else if(tAnType==kXiKchP || tAnType==kAXiKchM) tFileLocationInterpHistos = "InterpHistsAttractive";
  tFitter->LoadInterpHistFile(tFileLocationInterpHistos);

  //-------------------------------------------

  tFitter->SetUseRandomKStarVectors(true);
  tFitter->SetUseStaticPairs(true);
  tFitter->SetNPairsPerKStarBin(tNPairsPerKStarBin);
  tFitter->SetBinSizeKStar(tBinSize);
  tFitter->BuildPairSample4dVec();

  //-------------------------------------------

  tFitter->GetFitSharedAnalyses()->GetMinuitObject()->SetFCN(fcn);
  myFitter = tFitter;




//_______________________________________________________________________________________________________________________
  double tLambda, tRadius, tNorm;

  if(tAnType==kXiKchP || tAnType==kAXiKchM)
  {
    tLambda = 0.48;
    tRadius = 2.44;

    tNorm = 1.;
  }

  if(tAnType==kAXiKchP || tAnType==kXiKchM)
  {
    tLambda = 0.80;
    tRadius = 3.03;

    tNorm = 1.;
  }

  TString tName = cAnalysisRootTags[tAnType];

  TH1* tData;
  if(tCentType==k0010) tData = tPairAn0010->GetKStarCf();
  else if(tCentType==k1030) tData = tPairAn1030->GetKStarCf();
  else if(tCentType==k3050) tData = tPairAn3050->GetKStarCf();
  else assert(0);

  TH1* tDatawSysErrors = (TH1*)tSharedAn->GetFitPairAnalysis(0)->GetCfwSysErrors();

  TCanvas* tCan = DrawKStarCfs(tData, tDatawSysErrors, tAnType, tCentType);

  TH1* tCoulombOnlyHistSample;
  TH1* tCoulombOnlyHistSampleHigh;
  TH1* tCoulombOnlyHistSampleLow;
  TGraphAsymmErrors* tCoulombOnlyGraph;

  if(bDrawErrorBands)
  {
    tNbinsK = 15;
    tKMin = 0.;
    tKMax = 0.15;
    tBinSize = (tKMax-tKMin)/tNbinsK;

    CoulombFitter* tFitter2 = new CoulombFitter(tSharedAn,tKMax);
      tFitter2->SetIncludeSingletAndTriplet(false);
      tFitter2->SetApplyMomResCorrection(false);

    tFitter2->LoadInterpHistFile(tFileLocationInterpHistos);

    tFitter2->SetUseRandomKStarVectors(true);
    tFitter2->SetUseStaticPairs(true);
    tFitter2->SetNPairsPerKStarBin(tNPairsPerKStarBin);
    tFitter2->SetBinSizeKStar(tBinSize);
    tFitter2->BuildPairSample4dVec();

    tFitter2->GetFitSharedAnalyses()->GetMinuitObject()->SetFCN(fcn);
    myFitter = tFitter2;
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

    tCoulombOnlyHistSampleHigh = tFitter2->CreateFitHistogramSampleComplete("tCoulombOnlyHistSampleHigh", tAnType, tNbinsK, tKMin, tKMax, tLambdaHighCurve, tRadiusHighCurve, 0., 0., 0., 0., 0., 0., tNorm);
    tCoulombOnlyHistSampleLow = tFitter2->CreateFitHistogramSampleComplete("tCoulombOnlyHistSampleLow", tAnType, tNbinsK, tKMin, tKMax, tLambdaLowCurve, tRadiusLowCurve, 0., 0., 0., 0., 0., 0., tNorm);
    

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
    tCoulombOnlyGraph->SetFillColor(TColor::GetColorTransparent(kMagenta,0.2));

    tCoulombOnlyGraph->Draw("3");
    tCan->Update();

    delete tFitter2;
  }
  else
  {
    td1dVec tLambdas {0.1, 0.3, 0.5, 0.7, 0.9};
    td1dVec tRadii {1, 2, 3, 4, 5, 6, 8, 10};

    vector<int> tColors {kRed, kMagenta, kBlue, kGreen, kYellow};
    vector<int> tColorModifiers {3, 2, 1, 0, -4, -7, -9, -10};

    assert(tLambdas.size() == tColors.size());
    assert(tRadii.size() == tColorModifiers.size());

    int tCoulombBaseColor, tCoulombColor;

    for(unsigned int iLam=0; iLam<tLambdas.size(); iLam++)
    {
      tLambda = tLambdas[iLam];
      tCoulombBaseColor = tColors[iLam];
cout << "tLambda = " << tLambda << endl;
      for(unsigned int iRad=0; iRad<tRadii.size(); iRad++)
      {
        tRadius = tRadii[iRad];
        tCoulombColor = tCoulombBaseColor + tColorModifiers[iRad];
cout << "tRadius = " << tRadius << endl;

        tCoulombOnlyHistSample = tFitter->CreateFitHistogramSampleComplete("tCoulombOnlyHistSample", tAnType, tNbinsK, tKMin, tKMax, tLambda, tRadius, 0., 0., 0., 0., 0., 0., tNorm);
        tCoulombOnlyHistSample->SetLineColor(tCoulombColor);
        tCoulombOnlyHistSample->SetLineWidth(2);
        tCoulombOnlyHistSample->DrawCopy("lsame");

        tCan->Update();
      }

    }
    tCan->Update();
  }

  tDatawSysErrors->Draw("e2psame");
  tData->Draw("ex0same");

  //------------------------------------------------------------
  TString tTextAnType = TString(cAnalysisRootTags[tAnType]);
  TString tTextCentrality = TString(cPrettyCentralityTags[tCentType]);
  TString tCombinedText = tTextAnType + TString("  ") +  tTextCentrality;

  TLegend *tLeg;
  if(tAnType==kXiKchP || tAnType==kAXiKchM) tLeg = new TLegend(0.25,0.50,0.55,0.75);
  else if(tAnType==kXiKchM || tAnType==kAXiKchP) tLeg = new TLegend(0.25,0.35,0.55,0.60);
    tLeg->AddEntry(tData,tTextAnType,"p");
    tLeg->Draw();


  TLatex *tTex = new TLatex(0.09, 2.3, "Pb-Pb #sqrt{#it{s}_{NN}} = 2.76 TeV");
    tTex->SetTextFont(42);
    tTex->SetTextSize(0.05);
    tTex->SetLineWidth(2);
    tTex->Draw();
    tTex->DrawLatex(0.02,2.3,"ALICE Preliminary");
  //-------------------------------------------------------------
  if(bSaveImage)
  {
    TString tSaveName = tFileDirectory;
    tSaveName += cAnalysisBaseTags[tAnType] + TString(cCentralityTags[tCentType]) + TString("_DataVsCoulombOnly.pdf"); 
    tCan->SaveAs(tSaveName);
  }

  delete tFitter;

  delete tSharedAn;
  delete tPairAn0010;

//-------------------------------------------------------------------------------
  tFullTimer.Stop();
  cout << "Finished program: ";
  tFullTimer.PrintInterval();

  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.
  return 0;
}
