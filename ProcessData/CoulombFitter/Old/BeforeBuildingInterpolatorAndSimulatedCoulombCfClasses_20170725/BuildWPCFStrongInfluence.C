#include "FitSharedAnalyses.h"
#include "CoulombFitter.h"
#include "TLegend.h"
#include "CanvasPartition.h"

#include "TLatex.h"
#include "TColor.h"
#include <TStyle.h>
#include "TPaveText.h"
#include "TLegendEntry.h"

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
TPaveText* CreateParamValuesText(AnalysisType tAnType, td1dVec &tParams)
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

  tText->AddText(TString::Format("#lambda = %0.2f",tParams[0]));
  tText->AddText(TString::Format("R = %0.2f",tParams[1]));
  tText->AddText(TString::Format("Re[f0] = %0.2f",tParams[2]));
  tText->AddText(TString::Format("Im[f0] = %0.2f",tParams[3]));
  tText->AddText(TString::Format("d0 = %0.2f",tParams[4]));

  return tText;
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
  bool bScattMinusPlus = true;  //otherwise, PlusPlus
  bool bSaveImage = true;

  AnalysisType tAnType, tConjType;

  tAnType = kXiKchP;
  tConjType = kAXiKchM;

/*
  tAnType = kXiKchM;
  tConjType = kAXiKchP;
*/

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



//_______________________________________________________________________________________________________________________
  double tLambda, tRadius, tReF0, tImF0, tD0, tNorm;
  tLambda = 0.5;
  tRadius = 3.0;
  tReF0 = 0.5;
  tImF0 = 0.5;
  tD0 = 0.;
  tNorm = 1.;
  if(bScattMinusPlus) tReF0 *= -1.;
  vector<double> tFitParams {tLambda, tRadius, tReF0, tImF0, tD0};

  int tColor, tColorTransparent;
  if(tAnType==kXiKchP) tColor=kRed+1;
  else if(tAnType==kXiKchM) tColor=kBlue+1;
  else tColor=1;

  tColorTransparent = TColor::GetColorTransparent(tColor,0.2);

  TH1* tFullFit = tFitter->CreateFitHistogramSampleComplete("SampleHist1", tAnType, tNbinsK, tKMin, tKMax, tLambda, tRadius, tReF0, tImF0, tD0, 0., 0., 0., tNorm);
    tFullFit->SetDirectory(0);
  tFullFit->SetMarkerStyle(22);
  tFullFit->SetMarkerColor(tColor);
  tFullFit->SetLineColor(tColor);
  tFullFit->SetLineStyle(1);
  tFullFit->SetLineWidth(2);

  TH1* tCoulombOnlyFit = tFitter->CreateFitHistogramSampleComplete("tCoulombOnlyFit", tAnType, tNbinsK, tKMin, tKMax, tLambda, tRadius, 0., 0., 0., 0., 0., 0., tNorm);
    tCoulombOnlyFit->SetDirectory(0);
  tCoulombOnlyFit->SetMarkerStyle(21);
  tCoulombOnlyFit->SetMarkerColor(tColor);
  tCoulombOnlyFit->SetLineColor(tColor);
  tCoulombOnlyFit->SetLineStyle(7);

  TCanvas* tCan = new TCanvas("tCan","tCan");
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);
  tCan->cd();
  tCan->SetLeftMargin(0.13);
  tCan->SetRightMargin(0.0025);
  tCan->SetBottomMargin(0.13);
  tCan->SetTopMargin(0.020);

  double tXLow=0, tXHigh=0, tYLow=0, tYHigh=0;
  tXLow = 0.0;
  tXHigh = 0.15;
  if(tAnType==kXiKchP || tAnType==kAXiKchM)
  {
    tYLow = 0.92;
    tYHigh = 1.7;
  }
  else if(tAnType==kXiKchM || tAnType==kAXiKchP)
  {
    tYLow = 0.38;
    tYHigh = 1.09;
  }
  else assert(0);

  tFullFit->GetXaxis()->SetTitle("#it{k}* (GeV/#it{c})");
  tFullFit->GetXaxis()->SetRangeUser(tXLow,tXHigh);
  tFullFit->GetXaxis()->SetTitleSize(0.065);
  tFullFit->GetXaxis()->SetTitleOffset(0.9);
  tFullFit->GetXaxis()->SetLabelSize(0.04);
  tFullFit->GetXaxis()->SetLabelOffset(0.005);

  tFullFit->GetYaxis()->SetTitle("#it{C}(#it{k}*)");
  tFullFit->GetYaxis()->SetRangeUser(tYLow,tYHigh);
  tFullFit->GetYaxis()->SetTitleSize(0.070);
  tFullFit->GetYaxis()->SetTitleOffset(0.8);
  tFullFit->GetYaxis()->SetLabelSize(0.04);
  tFullFit->GetYaxis()->SetLabelOffset(0.005);

  tFullFit->Draw("l");
  tCoulombOnlyFit->Draw("lsame");

  double tXaxisRangeLow;
  if(tXLow<0) tXaxisRangeLow = 0.;
  else tXaxisRangeLow = tXLow;
  TLine *tLine = new TLine(tXaxisRangeLow,1.,tXHigh,1.);
  tLine->SetLineColor(TColor::GetColorTransparent(kGray,0.75));
  tLine->Draw();

  TPaveText *tText = CreateParamValuesText(tAnType, tFitParams);
  tText->Draw();
  //------------------------------------------------------------
  TString tTextAnType = TString(cAnalysisRootTags[tAnType]);
  TString tTextCentrality = TString(cPrettyCentralityTags[k0010]);
  TString tCombinedText = tTextAnType + TString("  ") +  tTextCentrality;

  TLegend *tLeg;
  if(tAnType==kXiKchP || tAnType==kAXiKchM) tLeg = new TLegend(0.25,0.50,0.55,0.75);
  else if(tAnType==kXiKchM || tAnType==kAXiKchP) tLeg = new TLegend(0.25,0.35,0.55,0.60);
  tLeg->SetHeader(TString::Format("%s Simulation",cAnalysisRootTags[tAnType]));
  tLeg->AddEntry(tFullFit,"Strong+Coulomb","l");
  tLeg->AddEntry(tCoulombOnlyFit, "Coulomb Only", "l");

  TLegendEntry* tHeader = (TLegendEntry*)tLeg->GetListOfPrimitives()->At(0);
  tHeader->SetTextAlign(22);

  tLeg->Draw();
  //-------------------------------------------------------------
  if(tAnType==kXiKchP || tAnType==kAXiKchM)
  {
    TLatex *tTex = new TLatex(0.0785, 1.64, "Pb-Pb #sqrt{#it{s}_{NN}} = 2.76 TeV");
      tTex->SetTextFont(42);
      tTex->SetTextSize(0.05);
      tTex->SetLineWidth(2);
      tTex->Draw();
      tTex->DrawLatex(0.015,1.64,"ALICE Preliminary");
  }
  if(tAnType==kAXiKchP || tAnType==kXiKchM)
  {
    TLatex *tTex = new TLatex(0.0785, 1.035, "Pb-Pb #sqrt{#it{s}_{NN}} = 2.76 TeV");
      tTex->SetTextFont(42);
      tTex->SetTextSize(0.05);
      tTex->SetLineWidth(2);
      tTex->Draw();
      tTex->DrawLatex(0.015,1.035,"ALICE Preliminary");
  }

  //-------------------------------------------------------------
  if(bSaveImage)
  {
    TString tSaveName = tFileDirectory;
    tSaveName += TString("WPCFStrongInfluence_") + cAnalysisBaseTags[tAnType] + TString(cCentralityTags[k0010]);
    if(bScattMinusPlus) tSaveName += TString("_ScattParamsMinusPlus");
    else tSaveName += TString("_ScattParamsPlusPlus");
    tSaveName += TString(".pdf");
    tCan->SaveAs(tSaveName);
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
