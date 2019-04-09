/* CompareTwoAnalyses.C */
/* Originally CompareIgnoreOnFlyStatus.C, used to compare different settings of
   ignoreOnFlyStatus of V0s in analyses */

#include "FitGenerator.h"
class FitGenerator;

#include "CanvasPartition.h"
class CanvasPartition;


//________________________________________________________________________________________________________________
TCanvas* DrawAllKStarCfs(FitGenerator* aFG1, FitGenerator* aFG2, FitGenerator* aFG3, bool aZoom=false, TString aCanNameModifier="")
{
  TString tCanvasName = TString("canKStarCfsAll");
  if(aZoom) tCanvasName += TString("Zoom");
  tCanvasName += aCanNameModifier;

  int tNx=2, tNy=3;

  double tXLow = -0.02;
  double tXHigh = 0.99;
  if(aZoom) tXHigh = 0.329;

  double tYLow = 0.86;
  double tYHigh = 1.07;

  CanvasPartition* tCanPart = new CanvasPartition(tCanvasName,tNx,tNy,tXLow,tXHigh,tYLow,tYHigh,0.12,0.0025,0.13,0.0025);
  tCanPart->SetDrawOptStat(false);

  int tMarkerStyle1 = 20;
  int tMarkerStyle2 = 20;
  int tMarkerStyle3 = 24;

  int tMarkerColor1 = kRed+1;
  int tMarkerColor2 = kBlue+1;
  int tMarkerColor3 = kBlack;

  double tMarkerSize = 0.75;

  int tAnalysisNumber=0;
  for(int j=0; j<tNy; j++)
  {
    for(int i=0; i<tNx; i++)
    {
      tAnalysisNumber = j*tNx + i;

      TH1* tHist1 = (TH1*)aFG1->GetKStarCf(tAnalysisNumber);
        tHist1->SetMarkerStyle(tMarkerStyle1);
        tHist1->SetMarkerColor(tMarkerColor1);
        tHist1->SetLineColor(tMarkerColor1);

      TH1* tHist2 = (TH1*)aFG2->GetKStarCf(tAnalysisNumber);
        tHist2->SetMarkerStyle(tMarkerStyle2);
        tHist2->SetMarkerColor(tMarkerColor2);
        tHist2->SetLineColor(tMarkerColor2);

      TH1* tHist3 = (TH1*)aFG3->GetKStarCf(tAnalysisNumber);
        tHist3->SetMarkerStyle(tMarkerStyle3);
        tHist3->SetMarkerColor(tMarkerColor3);
        tHist3->SetLineColor(tMarkerColor3);



      tCanPart->AddGraph(i, j, tHist1, "", tMarkerStyle1, tMarkerColor1, tMarkerSize, "ex0");
      tCanPart->AddGraph(i, j, tHist2, "", tMarkerStyle2, tMarkerColor2, tMarkerSize, "ex0same");
      tCanPart->AddGraph(i, j, tHist3, "", tMarkerStyle3, tMarkerColor3, tMarkerSize, "ex0same");


      tCanPart->SetupTLegend("", i, j, 0.75, 0.15, 0.20, 0.35);
      tCanPart->AddLegendEntry(i, j, tHist1, cAnalysisRootTags[aFG1->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumber)->GetAnalysisType()], "p");
      tCanPart->AddLegendEntry(i, j, tHist2, cAnalysisRootTags[aFG2->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumber)->GetAnalysisType()], "p");
      tCanPart->AddLegendEntry(i, j, tHist3, cAnalysisRootTags[aFG3->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumber)->GetAnalysisType()], "p");


      CentralityType aCentType = aFG1->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumber)->GetCentralityType();
      TLatex* tTex = new TLatex(0.25, 1.05, cPrettyCentralityTags[aCentType]);

      tCanPart->AddPadPaveLatex(tTex, i, j);
    }
  }

  tCanPart->SetDrawUnityLine(true);
  tCanPart->DrawAll();
  tCanPart->DrawXaxisTitle("#it{k}* (GeV/#it{c})");
  tCanPart->DrawYaxisTitle("#it{C}(#it{k}*)",43,25,0.05,0.85);

  return tCanPart->GetCanvas();
}



//________________________________________________________________________________________________________________
TCanvas* DrawKStarCfs(FitGenerator* aFG1, FitGenerator* aFG2, FitGenerator* aFG3, bool aZoom=false, TString aCanNameModifier="")
{
  int tAnalysisNumber=0;
  CentralityType aCentType = aFG1->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumber)->GetCentralityType();

  TString tCanvasName = TString("canKStarCfs");
  if(aZoom) tCanvasName += TString("Zoom");
  tCanvasName += aCanNameModifier;
  tCanvasName += TString(cCentralityTags[aCentType]);

  double tXLow = -0.02;
  double tXHigh = 0.99;
  if(aZoom) tXHigh = 0.329;

  double tYLow = 0.86;
  double tYHigh = 1.07;
  TCanvas* tCan = new TCanvas(tCanvasName, tCanvasName);
  tCan->cd();
  gStyle->SetOptTitle(0);
  gStyle->SetOptStat(0);
    tCan->SetTopMargin(0.025);
    tCan->SetRightMargin(0.025);
    tCan->SetBottomMargin(0.15);
    tCan->SetLeftMargin(0.125);

  int tMarkerStyle1 = 20;
  int tMarkerStyle2 = 20;
  int tMarkerStyle3 = 20;

  int tMarkerColor1 = kRed+1;
  int tMarkerColor2 = kBlue+1;
  int tMarkerColor3 = kBlack;

  TH1* tHist1 = (TH1*)aFG1->GetKStarCf(tAnalysisNumber);
    tHist1->SetMarkerStyle(tMarkerStyle1);
    tHist1->SetMarkerColor(tMarkerColor1);
    tHist1->SetLineColor(tMarkerColor1);

  TH1* tHist2 = (TH1*)aFG2->GetKStarCf(tAnalysisNumber);
    tHist2->SetMarkerStyle(tMarkerStyle2);
    tHist2->SetMarkerColor(tMarkerColor2);
    tHist2->SetLineColor(tMarkerColor2);

  TH1* tHist3 = (TH1*)aFG3->GetKStarCf(tAnalysisNumber);
    tHist3->SetMarkerStyle(tMarkerStyle3);
    tHist3->SetMarkerColor(tMarkerColor3);
    tHist3->SetLineColor(tMarkerColor3);

  double tMarkerSize = 0.75;

  tHist1->GetXaxis()->SetRangeUser(tXLow, tXHigh);
  tHist1->GetYaxis()->SetRangeUser(tYLow, tYHigh);

  tHist1->GetXaxis()->SetTitle("#it{k}* (GeV/#it{c})");
    tHist1->GetXaxis()->SetTitleSize(0.065);
    tHist1->GetXaxis()->SetTitleOffset(0.96);
    tHist1->GetXaxis()->SetLabelSize(0.045);

  tHist1->GetYaxis()->SetTitle("#it{C}(#it{k}*)");
    tHist1->GetYaxis()->SetTitleSize(0.075);
    tHist1->GetYaxis()->SetTitleOffset(0.80);
    tHist1->GetYaxis()->SetLabelSize(0.045);

  tHist1->Draw();
  tHist2->Draw("same");
  tHist3->Draw("same");

  TLegend* tLeg1 = new TLegend(0.75, 0.20, 0.95, 0.55, "", "NDC");
  tLeg1->SetFillColor(0);
  tLeg1->SetBorderSize(0);
  tLeg1->SetTextAlign(22);
  tLeg1->SetTextSize(0.075);
    tLeg1->AddEntry(tHist1, cAnalysisRootTags[kLamKchP], "p");
    tLeg1->AddEntry(tHist2, cAnalysisRootTags[kLamKchM], "p");
    tLeg1->AddEntry(tHist3, cAnalysisRootTags[kLamK0], "p");
  tLeg1->Draw();

  TLatex* tTex = new TLatex();
  tTex->SetTextAlign(12);
  tTex->SetLineWidth(2);
  tTex->SetTextFont(42);
  tTex->SetTextSize(0.075);

  tTex->DrawLatex(0.25, 1.05, cPrettyCentralityTags[aCentType]);

  return tCan;
}


//________________________________________________________________________________________________________________
TCanvas* DrawKStarCfsFocusBackground(FitGenerator* aFG1, FitGenerator* aFG2, FitGenerator* aFG3, TString aCanNameModifier="")
{
  int tAnalysisNumber=4;
  CentralityType aCentType = aFG1->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumber)->GetCentralityType();

  int tRebin=4;

  TString tCanvasName = TString("canKStarCfsFocusBgd");
  tCanvasName += aCanNameModifier;
  tCanvasName += TString(cCentralityTags[aCentType]);
  tCanvasName += TString::Format("_Rebin%d", tRebin);

  double tXLow = -0.02;
  double tXHigh = 1.99;

  double tYLow = 0.945;
  double tYHigh = 1.05;
  TCanvas* tCan = new TCanvas(tCanvasName, tCanvasName);
  tCan->cd();
  gStyle->SetOptTitle(0);
  gStyle->SetOptStat(0);
    tCan->SetTopMargin(0.025);
    tCan->SetRightMargin(0.025);
    tCan->SetBottomMargin(0.15);
    tCan->SetLeftMargin(0.125);

  int tMarkerStyle1 = 20;
  int tMarkerStyle2 = 20;
  int tMarkerStyle3 = 20;

  int tMarkerColor1 = kRed+1;
  int tMarkerColor2 = kBlue+1;
  int tMarkerColor3 = kBlack;

  CfHeavy* tHist1Heavy = aFG1->GetKStarCfHeavy(tAnalysisNumber);
    tHist1Heavy->Rebin(tRebin);
  TH1* tHist1 = tHist1Heavy->GetHeavyCf();
    tHist1->SetMarkerStyle(tMarkerStyle1);
    tHist1->SetMarkerColor(tMarkerColor1);
    tHist1->SetLineColor(tMarkerColor1);

  CfHeavy* tHist2Heavy = aFG2->GetKStarCfHeavy(tAnalysisNumber);
    tHist2Heavy->Rebin(tRebin);
  TH1* tHist2 = tHist2Heavy->GetHeavyCf();
    tHist2->SetMarkerStyle(tMarkerStyle2);
    tHist2->SetMarkerColor(tMarkerColor2);
    tHist2->SetLineColor(tMarkerColor2);

  CfHeavy* tHist3Heavy = aFG3->GetKStarCfHeavy(tAnalysisNumber);
    tHist3Heavy->Rebin(tRebin);
  TH1* tHist3 = tHist3Heavy->GetHeavyCf();
    tHist3->SetMarkerStyle(tMarkerStyle3);
    tHist3->SetMarkerColor(tMarkerColor3);
    tHist3->SetLineColor(tMarkerColor3);

  double tMarkerSize = 0.75;

  tHist1->GetXaxis()->SetRangeUser(tXLow, tXHigh);
  tHist1->GetYaxis()->SetRangeUser(tYLow, tYHigh);

  tHist1->GetXaxis()->SetTitle("#it{k}* (GeV/#it{c})");
    tHist1->GetXaxis()->SetTitleSize(0.065);
    tHist1->GetXaxis()->SetTitleOffset(0.96);
    tHist1->GetXaxis()->SetLabelSize(0.045);

  tHist1->GetYaxis()->SetTitle("#it{C}(#it{k}*)");
    tHist1->GetYaxis()->SetTitleSize(0.075);
    tHist1->GetYaxis()->SetTitleOffset(0.80);
    tHist1->GetYaxis()->SetLabelSize(0.045);

  tHist1->Draw();
  tHist2->Draw("same");
  tHist3->Draw("same");

  TLegend* tLeg1 = new TLegend(0.50, 0.60, 0.70, 0.95, "", "NDC");
  tLeg1->SetFillColor(0);
  tLeg1->SetBorderSize(0);
  tLeg1->SetTextAlign(22);
  tLeg1->SetTextSize(0.075);
    tLeg1->AddEntry(tHist1, cAnalysisRootTags[kLamKchP], "p");
    tLeg1->AddEntry(tHist2, cAnalysisRootTags[kLamKchM], "p");
    tLeg1->AddEntry(tHist3, cAnalysisRootTags[kLamK0], "p");
  tLeg1->Draw();

  TLatex* tTex = new TLatex();
  tTex->SetTextAlign(12);
  tTex->SetLineWidth(2);
  tTex->SetTextFont(42);
  tTex->SetTextSize(0.075);

  tTex->DrawLatex(1.50, 1.04, cPrettyCentralityTags[aCentType]);

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

  ChronoTimer tFullTimer(kSec);
  tFullTimer.Start();
//-----------------------------------------------------------------------------
  TString tResultsDate1, tResultsDate2, tResultsDate3;

  tResultsDate1 = "20180505";
  tResultsDate2 = "20180505";
  tResultsDate3 = "20180505";


  AnalysisType tAnType1 = kLamKchP;
  AnalysisType tAnType2 = kLamKchM;
  AnalysisType tAnType3 = kLamK0;

  AnalysisRunType tAnRunType = kTrain;
  int tNPartialAnalysis = 2;
  CentralityType tCentType = kMB;  //TODO
  FitGeneratorType tGenType = kPairwConj;



  bool bUseStavCf1 = false;
  bool bUseStavCf2 = false;
  bool bUseStavCf3 = false;

  bool SaveImages = false;
  TString tSaveDir = "/home/jesse/Analysis/FemtoAnalysis/AnalysisNotes/7_ResultsAndDiscussion/Figures/";

  TString tDirectoryBase1 = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_%s/",tResultsDate1.Data());
  TString tFileLocationBase1 = TString::Format("%sResults_cLamcKch_%s",tDirectoryBase1.Data(),tResultsDate1.Data());
  TString tFileLocationBaseMC1 = TString::Format("%sResults_cLamcKchMC_%s",tDirectoryBase1.Data(),tResultsDate1.Data());

  TString tDirectoryBase2 = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_%s/",tResultsDate2.Data());
  TString tFileLocationBase2 = TString::Format("%sResults_cLamcKch_%s",tDirectoryBase2.Data(),tResultsDate2.Data());
  TString tFileLocationBaseMC2 = TString::Format("%sResults_cLamcKchMC_%s",tDirectoryBase2.Data(),tResultsDate2.Data());

  TString tDirectoryBase3 = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamK0_%s/",tResultsDate3.Data());
  TString tFileLocationBase3 = TString::Format("%sResults_cLamK0_%s",tDirectoryBase3.Data(),tResultsDate3.Data());
  TString tFileLocationBaseMC3 = TString::Format("%sResults_cLamK0MC_%s",tDirectoryBase3.Data(),tResultsDate3.Data());

  TString tSaveNameModifier = "";
  FitGenerator* tLamKchP1 = new FitGenerator(tFileLocationBase1, tFileLocationBaseMC1, tAnType1, tCentType, tAnRunType, tNPartialAnalysis, tGenType, false, false, "", bUseStavCf1);
  FitGenerator* tLamKchP2 = new FitGenerator(tFileLocationBase2, tFileLocationBaseMC2, tAnType2, tCentType, tAnRunType, tNPartialAnalysis, tGenType, false, false, "", bUseStavCf2);
  FitGenerator* tLamKchP3 = new FitGenerator(tFileLocationBase3, tFileLocationBaseMC3, tAnType3, tCentType, tAnRunType, tNPartialAnalysis, tGenType, false, false, "", bUseStavCf3);
  //-----------------------------------------------------------------------------
  bool bZoom = true;
  bool bDrawKStarCfs = false;
  bool bDrawAllKStarCfs = true;
  bool bDrawKStarCfsFocusBackground = false;
  //-----------------------------------------------------------------------------
  TString tCanNameModifier = TString("_LamKchPKchMK0");

  //-----------------------------------------------------------------------------
  if(bDrawKStarCfs)
  {
    TCanvas* tCan = DrawKStarCfs(tLamKchP1, tLamKchP2, tLamKchP3, bZoom, tCanNameModifier);
    if(SaveImages) tCan->SaveAs(TString::Format("%s%s.pdf", tSaveDir.Data(), tCan->GetName()));
  }
  if(bDrawAllKStarCfs)
  {
    TCanvas* tCanAll = DrawAllKStarCfs(tLamKchP1, tLamKchP2, tLamKchP3, bZoom, tCanNameModifier);
    if(SaveImages) tCanAll->SaveAs(TString::Format("%s%s.pdf", tSaveDir.Data(), tCanAll->GetName()));
  }
  if(bDrawKStarCfsFocusBackground)
  {
    TCanvas* tCanBgd = DrawKStarCfsFocusBackground(tLamKchP1, tLamKchP2, tLamKchP3, tCanNameModifier);
    if(SaveImages) tCanBgd->SaveAs(TString::Format("%s%s.pdf", tSaveDir.Data(), tCanBgd->GetName()));
  }
//-------------------------------------------------------------------------------
  tFullTimer.Stop();
  cout << "Finished program: ";
  tFullTimer.PrintInterval();

  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.
cout << "DONE" << endl;
  return 0;
}
