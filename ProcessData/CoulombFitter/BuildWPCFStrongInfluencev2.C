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
void CreateParamValuesText(AnalysisType tAnType, td1dVec &tParams1, td1dVec &tParams2)
{
  TPaveText *tText1, *tText2, *tText3;
  if(tAnType==kXiKchP || tAnType==kAXiKchM)
  {
/*
    tText1 = new TPaveText(0.70,0.65,0.85,0.80,"NDC");
    tText2 = new TPaveText(0.65,0.40,0.70,0.65,"NDC");
    tText3 = new TPaveText(0.85,0.40,0.90,0.65,"NDC");
*/
    tText1 = new TPaveText(0.625,0.50,0.775,0.65,"NDC");
    tText2 = new TPaveText(0.575,0.25,0.625,0.50,"NDC");
    tText3 = new TPaveText(0.775,0.25,0.825,0.50,"NDC");
  }
  else if(tAnType==kXiKchM || tAnType==kAXiKchP)
  {
    tText1 = new TPaveText(0.625,0.40,0.775,0.55,"NDC");
    tText2 = new TPaveText(0.575,0.15,0.625,0.40,"NDC");
    tText3 = new TPaveText(0.775,0.15,0.825,0.40,"NDC");
  }
  else assert(0);

  double tTextSize = 20;
  tText1->SetFillColor(0);
  tText1->SetBorderSize(0);
  tText1->SetTextAlign(22);
  tText1->SetTextFont(43);
  tText1->SetTextSize(tTextSize);

  tText2->SetFillColor(0);
  tText2->SetBorderSize(0);
  tText2->SetTextAlign(22);
  tText2->SetTextFont(43);
  tText2->SetTextSize(tTextSize);

  tText3->SetFillColor(0);
  tText3->SetBorderSize(0);
  tText3->SetTextAlign(22);
  tText3->SetTextFont(43);
  tText3->SetTextSize(tTextSize);

  tText1->AddText("Simulation Parameters");
  tText1->AddText(TString::Format("#lambda = %0.1f",tParams1[0]));
  tText1->AddText(TString::Format("R = %0.1f",tParams1[1]));


  tText2->AddText("Set 1");
  tText2->AddText(TString::Format("Re[f_{0}] = %0.1f",tParams1[2]));
  tText2->AddText(TString::Format("Im[f_{0}] = %0.1f",tParams1[3]));
  tText2->AddText(TString::Format("d_{0} = %0.1f",tParams1[4]));

  tText3->AddText("Set 2");
  tText3->AddText(TString::Format("Re[f_{0}] = %0.1f",tParams2[2]));
  tText3->AddText(TString::Format("Im[f_{0}] = %0.1f",tParams2[3]));
  tText3->AddText(TString::Format("d_{0} = %0.1f",tParams2[4]));

  TText* tLine;
  tLine = tText1->GetLine(0);
  tLine->SetTextFont(63);
  tLine = tText2->GetLine(0);
  tLine->SetTextFont(63);
  tLine = tText3->GetLine(0);
  tLine->SetTextFont(63);

  tText1->Draw();
  tText2->Draw();
  tText3->Draw();
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
  CoulombFitter* tFitter = new CoulombFitter(tSharedAn,tKMax);
    tFitter->SetIncludeSingletAndTriplet(false);
    tFitter->SetApplyMomResCorrection(false);

  TString tFileLocationInterpHistos;
  if(tAnType==kAXiKchP || tAnType==kXiKchM) tFileLocationInterpHistos = "InterpHistsRepulsive";
  else if(tAnType==kXiKchP || tAnType==kAXiKchM) tFileLocationInterpHistos = "InterpHistsAttractive";
  tFitter->LoadInterpHistFile(tFileLocationInterpHistos);

  tFitter->SetUseRandomKStarVectors(true);
  tFitter->SetUseStaticPairs(true);
  tFitter->SetNPairsPerKStarBin(50000);
  tFitter->SetBinSizeKStar(tBinSize);

  tFitter->GetFitSharedAnalyses()->GetMinuitObject()->SetFCN(fcn);
  myFitter = tFitter;
  //-------------------------------------------



//_______________________________________________________________________________________________________________________
  double tLambda, tRadius, tReF0, tReF02, tImF0, tD0, tNorm;
  tLambda = 0.5;
  tRadius = 3.0;
  tReF0 = 0.5;
  tReF02 = -1.*tReF0;
  tImF0 = 0.5;
  tD0 = 0.;
  tNorm = 1.;
  vector<double> tFitParams1 {tLambda, tRadius, tReF0, tImF0, tD0};
  vector<double> tFitParams2 {tLambda, tRadius, tReF02, tImF0, tD0};

  int tColor, tColorTransparent;
  if(tAnType==kXiKchP) tColor=kRed+1;
  else if(tAnType==kXiKchM) tColor=kBlue+1;
  else tColor=1;

  tColorTransparent = TColor::GetColorTransparent(tColor,0.2);

  TH1* tFullFit1 = tFitter->CreateFitHistogramSampleComplete("SampleHist1", tAnType, tNbinsK, tKMin, tKMax, tLambda, tRadius, tReF0, tImF0, tD0, 0., 0., 0., tNorm);
    tFullFit1->SetDirectory(0);
  tFullFit1->SetMarkerStyle(22);
  tFullFit1->SetMarkerColor(tColor);
  tFullFit1->SetLineColor(tColor);
  tFullFit1->SetLineStyle(7);
  tFullFit1->SetLineWidth(2);

  TH1* tFullFit2 = tFitter->CreateFitHistogramSampleComplete("SampleHist2", tAnType, tNbinsK, tKMin, tKMax, tLambda, tRadius, tReF02, tImF0, tD0, 0., 0., 0., tNorm);
    tFullFit2->SetDirectory(0);
  tFullFit2->SetMarkerStyle(22);
  tFullFit2->SetMarkerColor(tColor);
  tFullFit2->SetLineColor(tColor);
  tFullFit2->SetLineStyle(3);
  tFullFit2->SetLineWidth(2);

  TH1* tCoulombOnlyFit = tFitter->CreateFitHistogramSampleComplete("tCoulombOnlyFit", tAnType, tNbinsK, tKMin, tKMax, tLambda, tRadius, 0., 0., 0., 0., 0., 0., tNorm);
    tCoulombOnlyFit->SetDirectory(0);
  tCoulombOnlyFit->SetMarkerStyle(21);
  tCoulombOnlyFit->SetMarkerColor(tColor);
  tCoulombOnlyFit->SetLineColor(tColor);
  tCoulombOnlyFit->SetLineStyle(1);
  tCoulombOnlyFit->SetLineWidth(2);

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
    tYLow = 0.35;
    tYHigh = 1.06;
  }
  else assert(0);

  tFullFit1->GetXaxis()->SetTitle("#it{k}* (GeV/#it{c})");
  tFullFit1->GetXaxis()->SetRangeUser(tXLow,tXHigh);
  tFullFit1->GetXaxis()->SetTitleSize(0.065);
  tFullFit1->GetXaxis()->SetTitleOffset(0.9);
  tFullFit1->GetXaxis()->SetLabelSize(0.04);
  tFullFit1->GetXaxis()->SetLabelOffset(0.005);

  tFullFit1->GetYaxis()->SetTitle("#it{C}(#it{k}*)");
  tFullFit1->GetYaxis()->SetRangeUser(tYLow,tYHigh);
  tFullFit1->GetYaxis()->SetTitleSize(0.070);
  tFullFit1->GetYaxis()->SetTitleOffset(0.8);
  tFullFit1->GetYaxis()->SetLabelSize(0.04);
  tFullFit1->GetYaxis()->SetLabelOffset(0.005);

  tFullFit1->Draw("l");
  tFullFit2->Draw("lsame");
  tCoulombOnlyFit->Draw("lsame");

  double tXaxisRangeLow;
  if(tXLow<0) tXaxisRangeLow = 0.;
  else tXaxisRangeLow = tXLow;
  TLine *tLine = new TLine(tXaxisRangeLow,1.,tXHigh,1.);
  tLine->SetLineColor(TColor::GetColorTransparent(kGray,0.75));
  tLine->Draw();

  CreateParamValuesText(tAnType, tFitParams1, tFitParams2);
  //------------------------------------------------------------
  TLegend *tLeg;
  if(tAnType==kXiKchP || tAnType==kAXiKchM) tLeg = new TLegend(0.475,0.70,0.925,0.95);
  else if(tAnType==kXiKchM || tAnType==kAXiKchP) tLeg = new TLegend(0.475,0.575,0.925,0.825);
  tLeg->SetHeader(TString::Format("%s Simulation",cAnalysisRootTags[tAnType]));
  tLeg->AddEntry(tCoulombOnlyFit, "Coulomb Only", "l");
  tLeg->AddEntry(tFullFit1,"Strong+Coulomb {Set 1}","l");
  tLeg->AddEntry(tFullFit2,"Strong+Coulomb {Set 2}","l");


  TLegendEntry* tHeader = (TLegendEntry*)tLeg->GetListOfPrimitives()->At(0);
  tHeader->SetTextAlign(22);
  tHeader->SetTextFont(63);
  tHeader->SetTextSize(25);

  tLeg->Draw();
  //-------------------------------------------------------------
/*
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
*/
  //-------------------------------------------------------------
  if(bSaveImage)
  {
    TString tSaveName = tFileDirectory;
    tSaveName += TString("WPCFStrongInfluence_") + cAnalysisBaseTags[tAnType] + TString(cCentralityTags[k0010]);
    tSaveName += TString("_v2.pdf");
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
