#include "FitSharedAnalyses.h"
#include "CoulombFitter.h"
#include "TLegend.h"
#include "CanvasPartition.h"

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
TCanvas* DrawKStarCfswFits(TH1* aData, TH1* aFit, TH1* aCoulombOnlyFit, AnalysisType aAnType=kXiKchP, CentralityType aCentType=k0010, bool aSaveImage=false)
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
    tYLow = 0.94;
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

  TString tTextAnType = TString(cAnalysisRootTags[aAnType]);
  TString tTextCentrality = TString(cPrettyCentralityTags[aCentType]);
  TString tCombinedText = tTextAnType + TString("  ") +  tTextCentrality;

  int tColor;
  if(aAnType==kXiKchP || aAnType==kAXiKchM) tColor=kRed+1;
  else if(aAnType==kXiKchM || aAnType==kAXiKchP) tColor=kBlue+1;
  else tColor=1;



  aFit->GetXaxis()->SetTitle("#it{k}* (GeV/#it{c})");
  aFit->GetXaxis()->SetRangeUser(tXLow,tXHigh);
  aFit->GetXaxis()->SetTitleSize(0.055);
  aFit->GetXaxis()->SetTitleOffset(0.8);

  aFit->GetYaxis()->SetTitle("#it{C}(#it{k}*)");
  aFit->GetYaxis()->SetRangeUser(tYLow,tYHigh);
  aFit->GetYaxis()->SetTitleSize(0.0575);
  aFit->GetYaxis()->SetTitleOffset(0.8);

  aFit->SetMarkerStyle(22);
  aFit->SetMarkerColor(1);
  aFit->SetLineColor(1);
  aFit->SetLineStyle(1);

  aCoulombOnlyFit->SetMarkerStyle(21);
  aCoulombOnlyFit->SetMarkerColor(1);
  aCoulombOnlyFit->SetLineColor(1);
  aCoulombOnlyFit->SetLineStyle(7);

  aData->SetMarkerStyle(20);
  aData->SetMarkerColor(tColor);
  aData->SetLineColor(tColor);


  aFit->Draw("l");
  aCoulombOnlyFit->Draw("lsame");
  aData->Draw("psame");

  TLegend *tLeg = new TLegend(0.55,0.25,0.85,0.50);
    tLeg->AddEntry(aData,tTextAnType,"p");
    tLeg->AddEntry(aFit,"Full Fit","l");
    tLeg->AddEntry(aCoulombOnlyFit, "Coulomb Only", "l");
    tLeg->Draw();


  double tXaxisRangeLow;
  if(tXLow<0) tXaxisRangeLow = 0.;
  else tXaxisRangeLow = tXLow;
  TLine *tLine = new TLine(tXaxisRangeLow,1.,tXHigh,1.);
  tLine->SetLineColor(TColor::GetColorTransparent(kGray,0.75));
  tLine->Draw();

  returnText->Draw();

  if(aSaveImage)
  {
    TString tSaveName = cAnalysisBaseTags[aAnType] + TString(cCentralityTags[aCentType]) + TString(".eps"); 
    tCan->SaveAs(tSaveName);
  }

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

  vector<int> Share01(2);
    Share01[0] = 0;
    Share01[1] = 1;

  vector<int> Share23(2);
    Share23[0] = 2;
    Share23[1] = 3;

  vector<int> Share45(2);
    Share45[0] = 4;
    Share45[1] = 5;
//-----------------------------------------------------------------------------
//Be sure to set the following...

  TString tFileLocationBase = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cXicKch_20170406/Results_cXicKch_20170406";
  bool bSaveImage = true;

  AnalysisType tAnType;
  CentralityType tCentType;

  tAnType = kXiKchP;
  //tAnType = kAXiKchM;

  //tAnType = kXiKchM;
  //tAnType = kAXiKchP;

  tCentType = k0010;
  //tCentType = k1030;
  //tCentType = k3050;


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

//  CoulombFitter* tFitter = new CoulombFitter(tSharedAn,0.15);
  CoulombFitter* tFitter = new CoulombFitter(tSharedAn,0.30);
//  CoulombFitter* tFitter = new CoulombFitter(tSharedAn,0.02);
    tFitter->SetIncludeSingletAndTriplet(true);
    tFitter->SetApplyMomResCorrection(false);


  TString tFileLocationInterpHistos;
  if(tAnType==kAXiKchP || tAnType==kXiKchM) tFileLocationInterpHistos = "InterpHistsRepulsive";
  else if(tAnType==kXiKchP || tAnType==kAXiKchM) tFileLocationInterpHistos = "InterpHistsAttractive";
  tFitter->LoadInterpHistFile(tFileLocationInterpHistos);

  //-------------------------------------------

  tFitter->SetUseRandomKStarVectors(true);
  tFitter->SetUseStaticPairs(true,16384);

  //-------------------------------------------

  tFitter->GetFitSharedAnalyses()->GetMinuitObject()->SetFCN(fcn);
  myFitter = tFitter;




//_______________________________________________________________________________________________________________________

  int tNbinsK = 15;
  double tKMin = 0.;
  double tKMax = 0.15;

  double tLambda, tRadius, tReF0s, tImF0s, tD0s, tReF0t, tImF0t, tD0t, tNorm;

  if(tAnType==kXiKchP || tAnType==kAXiKchM)
  {
/*
    //18 April 2017 (0010 with 1030)
    tLambda = 0.398942;
    tRadius = 2.33678;

    if(tCentType==k1030)
    {
      tLambda = 0.1;
      tRadius = 1.56581;
    }

    tReF0s = -1.13176;
    tImF0s = 0.988682;
    tD0s = -5.;

    tReF0t = -0.17914;
    tImF0t = 0.0205195;
    tD0t = 2.48178;
    tNorm = 1.;
*/

    //18 April 2017 (0010 only)
    tLambda = 0.385577;
    tRadius = 2.13681;

    tReF0s = -0.782880;
    tImF0s = 0.448405;
    tD0s = -5.;

    tReF0t = -0.220497;
    tImF0t = 0.0327614;
    tD0t = 3.0;
    tNorm = 1.;

  }

  if(tAnType==kAXiKchP || tAnType==kXiKchM)
  {
/*
    //18 April 2017 (0010 with 1030)
    tLambda = 0.691397;
    tRadius = 4.11372;

    if(tCentType==k1030)
    {
      tLambda = 0.589762;
      tRadius = 2.41899;
    }

    tReF0s = -0.437582;
    tImF0s = -1.03858;
    tD0s = -2.19830;

    tReF0t = 0.126213;
    tImF0t = 0.743508;
    tD0t = -3.72636;
    tNorm = 1.;
*/

    //18 April 2017 (0010 only)
    tLambda = 0.740911;
    tRadius = 4.14400;

    tReF0s = -1.25864;
    tImF0s = -0.357157;
    tD0s = 2.94671;

    tReF0t = 0.450537;
    tImF0t = 0.224479;
    tD0t = -4.51343;
    tNorm = 1.;

  }

  TString tName = cAnalysisRootTags[tAnType];

  TH1* tSampleHist1 = tFitter->CreateFitHistogramSampleComplete("SampleHist1", tAnType, tNbinsK, tKMin, tKMax, tLambda, tRadius, tReF0s, tImF0s, tD0s, tReF0t, tImF0t, tD0t, tNorm);
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

  TCanvas* tCan = DrawKStarCfswFits(tData, tSampleHist1, tCoulombOnlyHist, tAnType, tCentType, bSaveImage);

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
