#include "FitSharedAnalyses.h"
#include "CoulombFitter.h"
#include "TLegend.h"
#include "CanvasPartition.h"


#include "TColor.h"
#include <TStyle.h>
#include "TPaveText.h"

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
TPaveText* CreateParamValuesText(AnalysisType tAnType, td1dVec &tParams, td1dVec &tParamErrors, bool bPrintErrors)
{
  TPaveText* tText;
  if(tAnType==kXiKchP || tAnType==kAXiKchM) tText = new TPaveText(0.60,0.35,0.85,0.75,"NDC");
  else if(tAnType==kXiKchM || tAnType==kAXiKchP) tText = new TPaveText(0.60,0.30,0.85,0.60,"NDC");
  else assert(0);

  tText->SetFillColor(0);
  tText->SetBorderSize(0);
  tText->SetTextAlign(22);
  tText->SetTextFont(63);
  tText->SetTextSize(20);

  if(!bPrintErrors)
  {
    tText->AddText(TString::Format("#lambda = %0.2f",tParams[0]));
    tText->AddText(TString::Format("R = %0.2f",tParams[1]));
    tText->AddText(TString::Format("Re[f0] = %0.2f",tParams[2]));
    tText->AddText(TString::Format("Im[f0] = %0.2f",tParams[3]));
    tText->AddText(TString::Format("d0 = %0.2f",tParams[4]));
  }
  else
  {
    tText->AddText(TString::Format("#lambda = %0.2f #pm %0.2f",tParams[0], tParamErrors[0]));
    tText->AddText(TString::Format("R = %0.2f #pm %0.2f",tParams[1], tParamErrors[1]));
    tText->AddText(TString::Format("Re[f0] = %0.2f #pm %0.2f",tParams[2], tParamErrors[2]));
    tText->AddText(TString::Format("Im[f0] = %0.2f #pm %0.2f",tParams[3], tParamErrors[3]));
    tText->AddText(TString::Format("d0 = %0.2f #pm %0.2f",tParams[4], tParamErrors[4]));
  }

  return tText;
}

//________________________________________________________________________________________________________________
TCanvas* DrawKStarCfswFits(TH1* aData, TH1* aDatawSysErrors, TH1* aFit, TH1* aCoulombOnlyFit, AnalysisType aAnType=kXiKchP, CentralityType aCentType=k0010, bool aDrawMultipleCoulomb=false)
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
    tYHigh = 1.7;
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

  TString tTextSysInfo = TString("Pb-Pb #sqrt{#it{s}_{NN}} = 2.76 TeV");
//  TString tTextAlicePrelim = TString("ALICE Preliminary");
  double tSysInfoXMin, tSysInfoXMax, tSysInfoYMin, tSysInfoYMax;
  if(aAnType==kXiKchP || aAnType==kAXiKchM)
  {
    tSysInfoXMin = 0.60;
    tSysInfoXMax = 0.89;

    tSysInfoYMin = 0.80;
    tSysInfoYMax = 0.875;
  }

  if(aAnType==kXiKchM || aAnType==kAXiKchP)
  {
    tSysInfoXMin = 0.60;
    tSysInfoXMax = 0.89;

    tSysInfoYMin = 0.70;
    tSysInfoYMax = 0.775;
  }

  TPaveText* returnText = new TPaveText(tSysInfoXMin,tSysInfoYMin,tSysInfoXMax,tSysInfoYMax, "NDC");
    returnText->SetFillColor(0);
    returnText->SetBorderSize(0);
    returnText->SetTextAlign(22);
    returnText->AddText(tTextSysInfo);

  int tColor;
  Int_t tColorTransparent;
  if(aAnType==kXiKchP || aAnType==kAXiKchM) tColor=kRed+1;
  else if(aAnType==kXiKchM || aAnType==kAXiKchP) tColor=kBlue+1;
  else tColor=1;

//  tColorTransparent = TColor::GetColorTransparent(tColor,0.2);  //TODO why doesn't this work?
  if(aAnType==kXiKchP || aAnType==kAXiKchM) tColorTransparent=kRed-10;
  else if(aAnType==kXiKchM || aAnType==kAXiKchP) tColorTransparent=kBlue-10;
  else tColor=1;

  aData->GetXaxis()->SetTitle("#it{k}* (GeV/#it{c})");
  aData->GetXaxis()->SetRangeUser(tXLow,tXHigh);
  aData->GetXaxis()->SetTitleSize(0.055);
  aData->GetXaxis()->SetTitleOffset(0.8);

  aData->GetYaxis()->SetTitle("#it{C}(#it{k}*)");
  aData->GetYaxis()->SetRangeUser(tYLow,tYHigh);
  aData->GetYaxis()->SetTitleSize(0.0575);
  aData->GetYaxis()->SetTitleOffset(0.8);

  aFit->SetMarkerStyle(22);
  aFit->SetMarkerColor(1);
  aFit->SetLineColor(1);
  aFit->SetLineStyle(1);
  aFit->SetLineWidth(2);

  aCoulombOnlyFit->SetMarkerStyle(21);
  aCoulombOnlyFit->SetMarkerColor(1);
  aCoulombOnlyFit->SetLineColor(1);
  aCoulombOnlyFit->SetLineStyle(7);

  aData->SetMarkerStyle(20);
  aData->SetMarkerColor(tColor);
  aData->SetLineColor(tColor);

  aDatawSysErrors->SetFillColor(tColorTransparent);
  aDatawSysErrors->SetMarkerColor(tColorTransparent);
  aDatawSysErrors->SetFillStyle(1000);
  aDatawSysErrors->SetLineColor(0);
  aDatawSysErrors->SetLineWidth(0);

  aData->Draw("ex0");
  if(!aDrawMultipleCoulomb) aFit->Draw("lsame");
  if(!aDrawMultipleCoulomb) aCoulombOnlyFit->Draw("lsame");
  aDatawSysErrors->Draw("e2psame");
  aData->Draw("ex0same");

  double tXaxisRangeLow;
  if(tXLow<0) tXaxisRangeLow = 0.;
  else tXaxisRangeLow = tXLow;
  TLine *tLine = new TLine(tXaxisRangeLow,1.,tXHigh,1.);
  tLine->SetLineColor(TColor::GetColorTransparent(kGray,0.75));
  tLine->Draw();

  returnText->Draw();

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
  bool bDrawMultipleCoulomb = false;
  bool bDrawMultipleLambda = false;

  AnalysisType tAnType;
  CentralityType tCentType;

  tAnType = kXiKchP;
  //tAnType = kAXiKchM;

  //tAnType = kXiKchM;
  //tAnType = kAXiKchP;

  tCentType = k0010;
  //tCentType = k1030;
  //tCentType = k3050;

  int tNbinsK = 15;
  double tKMin = 0.;
  double tKMax = 0.15;

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
  tFitter->SetUseStaticPairs(true,50000);

  //-------------------------------------------

  tFitter->GetFitSharedAnalyses()->GetMinuitObject()->SetFCN(fcn);
  myFitter = tFitter;




//_______________________________________________________________________________________________________________________
  double tLambda, tRadius, tReF0, tImF0, tD0, tNorm;
  double tLambdaErr, tRadiusErr, tReF0Err, tImF0Err, tD0Err;

  if(tAnType==kXiKchP || tAnType==kAXiKchM)
  {
    //18 April 2017 (0010 only)
    if(tAnType==kXiKchP)
    {
      tLambda = 0.30;
      tLambdaErr = 0.12;
    }
    else
    {
      tLambda = 0.20;
      tLambdaErr = 0.08;
    }
    tRadius = 3.04;
    tRadiusErr = 0.32;

    tReF0 = -0.01;
    tReF0Err = 0.1;

    tImF0 = 3.2;
    tImF0Err = 2.3;

    tD0 = 0.;
    tD0Err = 0.;

    tNorm = 1.;
  }

  if(tAnType==kAXiKchP || tAnType==kXiKchM)
  {
    //18 April 2017 (0010 only)
    if(tAnType==kAXiKchP)
    {
      tLambda = 0.31;
      tLambdaErr = 0.10;
    }
    else
    {
      tLambda = 0.23;
      tLambdaErr = 0.09;
    }
    tRadius = 4.12;
    tRadiusErr = 0.23;

    tReF0 = 1.33;
    tReF0Err = 0.89;

    tImF0 = 2.45;
    tImF0Err = 2.04;

    tD0 = 0.;
    tD0Err = 0.;

    tNorm = 1.;
  }
  vector<double> tFitParams {tLambda, tRadius, tReF0, tImF0, tD0};
  vector<double> tFitParamErrors {tLambdaErr, tRadiusErr, tReF0Err, tImF0Err, tD0Err};

  TString tName = cAnalysisRootTags[tAnType];

  TH1* tSampleHist1 = tFitter->CreateFitHistogramSampleComplete("SampleHist1", tAnType, tNbinsK, tKMin, tKMax, tLambda, tRadius, tReF0, tImF0, tD0, 0., 0., 0., tNorm);
    tSampleHist1->SetDirectory(0);
    tSampleHist1->SetTitle(tName);
    tSampleHist1->SetName(tName);

  TH1* tCoulombOnlyHist = tFitter->CreateFitHistogramSampleComplete("tCoulombOnlyHist", tAnType, tNbinsK, tKMin, tKMax, tLambda, tRadius, 0., 0., 0., 0., 0., 0., tNorm);
    tCoulombOnlyHist->SetDirectory(0);

  TH1* tData;
  if(tCentType==k0010) tData = tPairAn0010->GetKStarCf();
  else if(tCentType==k1030) tData = tPairAn1030->GetKStarCf();
  else if(tCentType==k3050) tData = tPairAn3050->GetKStarCf();
  else assert(0);

  TH1* tDatawSysErrors = (TH1*)tSharedAn->GetFitPairAnalysis(0)->GetCfwSysErrors();

  TCanvas* tCan = DrawKStarCfswFits(tData, tDatawSysErrors, tSampleHist1, tCoulombOnlyHist, tAnType, tCentType, (bDrawMultipleCoulomb||bDrawMultipleLambda));

  if(!bDrawMultipleCoulomb && !bDrawMultipleLambda)
  {
    TPaveText *tText = CreateParamValuesText(tAnType, tFitParams, tFitParamErrors, true);
    tText->Draw();
  }

  TH1* tCoulombOnlyHistSample;
  if(bDrawMultipleCoulomb)
  {
    td1dVec tRadii {1, 2, 3, 4, 5, 6, 8, 10};
    vector<int> tColors {kMagenta+3, kMagenta+2, kMagenta+1, kMagenta-0, kMagenta-4, kMagenta-7, kMagenta-9, kMagenta-10};
//    vector<int> tColors {kMagenta-10, kMagenta-9, kMagenta-7, kMagenta-4, kMagenta-0, kMagenta+1, kMagenta+2, kMagenta+3};
    assert(tRadii.size() == tColors.size());
    int tCoulombColor;
    tRadius = 0.;

    TLegend *tLeg2;
    if(tAnType==kXiKchP || tAnType==kAXiKchM) tLeg2 = new TLegend(0.60,0.25,0.85,0.75);
    else if(tAnType==kXiKchM || tAnType==kAXiKchP) tLeg2 = new TLegend(0.60,0.12,0.85,0.60);
    tLeg2->SetHeader("Coulomb Only");

    for(unsigned int i=0; i<tRadii.size(); i++)
    {
      tRadius = tRadii[i];
      tCoulombColor = tColors[i];
cout << "tRadius = " << tRadius << endl;
      tCoulombOnlyHistSample = tFitter->CreateFitHistogramSampleComplete("tCoulombOnlyHistSample", tAnType, tNbinsK, tKMin, tKMax, tLambda, tRadius, 0., 0., 0., 0., 0., 0., tNorm);
      tCoulombOnlyHistSample->SetLineColor(tCoulombColor);
      tCoulombOnlyHistSample->SetLineWidth(2);
      tCoulombOnlyHistSample->DrawCopy("lsame");

      tLeg2->AddEntry(tCoulombOnlyHistSample,TString::Format("R = %0.1f",tRadius),"l");

      tCan->Update();
    }
    tLeg2->Draw();
    tCan->Update();
  }

  else if(bDrawMultipleLambda)
  {
    tRadius = 3.;
    td1dVec tLambdas {0.1, 0.3, 0.5, 0.7, 0.9};
    vector<int> tColors {kMagenta+2, kMagenta+1, kMagenta-4, kMagenta-7, kMagenta-9};
    assert(tLambdas.size() == tColors.size());
    int tCoulombColor;
    tLambda = 0.;

    TLegend *tLeg2;
    if(tAnType==kXiKchP || tAnType==kAXiKchM) tLeg2 = new TLegend(0.60,0.25,0.85,0.75);
    else if(tAnType==kXiKchM || tAnType==kAXiKchP) tLeg2 = new TLegend(0.60,0.12,0.85,0.60);
    tLeg2->SetHeader("Coulomb Only");

    for(unsigned int i=0; i<tLambdas.size(); i++)
    {
      tLambda = tLambdas[i];
      tCoulombColor = tColors[i];
cout << "tLambda = " << tLambda << endl;
      tCoulombOnlyHistSample = tFitter->CreateFitHistogramSampleComplete("tCoulombOnlyHistSample", tAnType, tNbinsK, tKMin, tKMax, tLambda, tRadius, 0., 0., 0., 0., 0., 0., tNorm);
      tCoulombOnlyHistSample->SetLineColor(tCoulombColor);
      tCoulombOnlyHistSample->SetLineWidth(2);
      tCoulombOnlyHistSample->DrawCopy("lsame");

      tLeg2->AddEntry(tCoulombOnlyHistSample,TString::Format("#lambda = %0.1f",tLambda),"l");

      tCan->Update();
    }
    tLeg2->Draw();
    tCan->Update();
  }

  if(!bDrawMultipleCoulomb && !bDrawMultipleLambda) tSampleHist1->Draw("lsame"); //make sure data is on top
  tData->Draw("ex0same");

  //------------------------------------------------------------
  TString tTextAnType = TString(cAnalysisRootTags[tAnType]);
  TString tTextCentrality = TString(cPrettyCentralityTags[tCentType]);
  TString tCombinedText = tTextAnType + TString("  ") +  tTextCentrality;

  TLegend *tLeg;
  if(tAnType==kXiKchP || tAnType==kAXiKchM) tLeg = new TLegend(0.25,0.50,0.55,0.75);
  else if(tAnType==kXiKchM || tAnType==kAXiKchP) tLeg = new TLegend(0.25,0.35,0.55,0.60);
    tLeg->AddEntry(tData,tTextAnType,"p");
    if(!bDrawMultipleCoulomb && !bDrawMultipleLambda) tLeg->AddEntry(tSampleHist1,"Full Fit","l");
    if(!bDrawMultipleCoulomb && !bDrawMultipleLambda) tLeg->AddEntry(tCoulombOnlyHist, "Coulomb Only", "l");
    if(bDrawMultipleCoulomb) tLeg->AddEntry((TObject*)0, TString::Format("#lambda = %0.1f",tLambda), "");
    if(bDrawMultipleLambda) tLeg->AddEntry((TObject*)0, TString::Format("R = %0.1f",tRadius), "");
    tLeg->Draw();
  //-------------------------------------------------------------
  if(bSaveImage)
  {
    TString tSaveName = tFileDirectory;
    if(!bDrawMultipleCoulomb && !bDrawMultipleLambda) tSaveName += cAnalysisBaseTags[tAnType] + TString(cCentralityTags[tCentType]); 
    else if(bDrawMultipleCoulomb) tSaveName += TString("DataVsCoulombOnly_VaryR_") + TString(cAnalysisBaseTags[tAnType]) + TString(cCentralityTags[tCentType]);
    else if(bDrawMultipleLambda) tSaveName += TString("DataVsCoulombOnly_VaryLam_") + TString(cAnalysisBaseTags[tAnType]) + TString(cCentralityTags[tCentType]); 
    else assert(0);
    tSaveName += TString(".eps");
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
