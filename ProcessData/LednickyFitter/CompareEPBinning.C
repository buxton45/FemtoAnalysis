#include "FitGenerator.h"
class FitGenerator;

//________________________________________________________________________________________________________________
TCanvas* CompareEPBinning(FitGenerator* aGenNoEPBin, FitGenerator* aGenEPBin8, FitGenerator* aGenEPBin16)
{
  AnalysisType tAnType = aGenNoEPBin->GetSharedAn()->GetFitPairAnalysis(0)->GetAnalysisType();

  TString tCanvasName = TString::Format("CompareEPBinning_%s", cAnalysisBaseTags[tAnType]);
  vector<CentralityType> tCentralityTypes = aGenNoEPBin->GetCentralityTypes();
  int tNAnalyses = aGenNoEPBin->GetNAnalyses();
  for(unsigned int i=0; i<tCentralityTypes.size(); i++) tCanvasName += TString(cCentralityTags[tCentralityTypes[i]]);


  int tNx=0, tNy=0;
  if(tNAnalyses == 6) {tNx=2; tNy=3;}
  else if(tNAnalyses == 4) {tNx=2; tNy=2;}
  else if(tNAnalyses == 3) {tNx=1; tNy=tNAnalyses;}
  else if(tNAnalyses == 2 || tNAnalyses==1) {tNx=tNAnalyses; tNy=1;}
  else assert(0);

  double tXLow = -0.02;
  double tXHigh = 1.99;
  double tYLow = 0.86;
  double tYHigh = 1.07;
  CanvasPartition* tCanPart = new CanvasPartition(tCanvasName,tNx,tNy,tXLow,tXHigh,tYLow,tYHigh,0.12,0.05,0.13,0.05);

  assert(tNx*tNy == tNAnalyses);
  int tAnalysisNumber=0;

  int tMarkerStyleNoBin = 20;
  int tMarkerStyleBin8 = 20;
  int tMarkerStyleBin16 = 24;

  double tMarkerSize = 0.5;

  int tColor;
  if     (tAnType==kLamK0 || tAnType==kALamK0)     tColor = kBlack;
  else if(tAnType==kLamKchP || tAnType==kALamKchM) tColor = kRed+1;
  else if(tAnType==kLamKchM || tAnType==kALamKchP) tColor = kBlue+1;
  else assert(0);

  int tColorBin8 = kCyan;
  int tColorBin16 = kMagenta;

  int tNx_Leg=0, tNy_Leg=0;

  TH1 *tCfNoEPBin, *tCfEPBin8, *tCfEPBin16;
  for(int j=0; j<tNy; j++)
  {
    for(int i=0; i<tNx; i++)
    {
      tAnalysisNumber = j*tNx + i;

      tCfNoEPBin = aGenNoEPBin->GetSharedAn()->GetKStarCfHeavy(tAnalysisNumber)->GetHeavyCfClone();
      tCfEPBin8 = aGenEPBin8->GetSharedAn()->GetKStarCfHeavy(tAnalysisNumber)->GetHeavyCfClone();
      tCfEPBin16 = aGenEPBin16->GetSharedAn()->GetKStarCfHeavy(tAnalysisNumber)->GetHeavyCfClone();
      //---------------------------------------------------------------------------------------------------------
      tCanPart->AddGraph(i, j, tCfNoEPBin, "", tMarkerStyleNoBin, tColor, tMarkerSize);
      tCanPart->AddGraph(i, j, tCfEPBin8, "", tMarkerStyleBin8, tColorBin8, tMarkerSize);
      tCanPart->AddGraph(i, j, tCfEPBin16, "", tMarkerStyleBin16, tColorBin16, tMarkerSize);
      if(i==tNx_Leg && j==tNy_Leg)
      {
        tCanPart->SetupTLegend("", i, j, 0.25, 0.05, 0.35, 0.50);
        tCanPart->AddLegendEntry(i, j, tCfNoEPBin, "No Binning", "p");
        tCanPart->AddLegendEntry(i, j, tCfEPBin8, "8 Bins", "p");
        tCanPart->AddLegendEntry(i, j, tCfEPBin16, "16 Bins", "p");
      }

      TString tTextAnType = TString(cAnalysisRootTags[aGenNoEPBin->GetSharedAn()->GetFitPairAnalysis(tAnalysisNumber)->GetAnalysisType()]);
      TString tTextCentrality = TString(cPrettyCentralityTags[aGenNoEPBin->GetSharedAn()->GetFitPairAnalysis(tAnalysisNumber)->GetCentralityType()]);
      TString tCombinedText = tTextAnType + TString("  ") +  tTextCentrality;
      TPaveText* tCombined = tCanPart->SetupTPaveText(tCombinedText,i,j,0.70,0.825,0.15,0.10,63,20);;
      tCanPart->AddPadPaveText(tCombined,i,j);
    }
  }

  tCanPart->SetDrawUnityLine(true);
  tCanPart->DrawAll();
  tCanPart->DrawXaxisTitle("#it{k}* (GeV/#it{c})");
  tCanPart->DrawYaxisTitle("#it{C}(#it{k}*)",43,25,0.05,0.75);


  return tCanPart->GetCanvas();
}

//________________________________________________________________________________________________________________
TCanvas* CompareEPBinning_SingleCentrality(FitGenerator* aGenNoEPBin, FitGenerator* aGenEPBin8, FitGenerator* aGenEPBin16, CentralityType aCentType)
{
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  //NOTE only designed for pair, not conj pair
  //To make for use with conjpair, just change how tAnalysisNumber is calculated and how tAnType is grabbed
  AnalysisType tAnType = aGenNoEPBin->GetSharedAn()->GetFitPairAnalysis(0)->GetAnalysisType();
  int tAnalysisNumber=2*(int)aCentType;

  TString tCanvasName = TString::Format("CompareEPBinning_%s%s", cAnalysisBaseTags[tAnType], cCentralityTags[aCentType]);
  int tNAnalyses = aGenNoEPBin->GetNAnalyses();


  double tXLow = 0.0;
  double tXHigh = 1.99;
  double tYLow = 0.86;
  double tYHigh = 1.07;

  TCanvas* tReturnCan = new TCanvas(tCanvasName, tCanvasName);
  tReturnCan->cd();

  int tMarkerStyleNoBin = 20;
  int tMarkerStyleBin8 = 20;
  int tMarkerStyleBin16 = 24;

  double tMarkerSize = 0.5;

  int tColor;
  if     (tAnType==kLamK0 || tAnType==kALamK0)     tColor = kBlack;
  else if(tAnType==kLamKchP || tAnType==kALamKchM) tColor = kRed+1;
  else if(tAnType==kLamKchM || tAnType==kALamKchP) tColor = kBlue+1;
  else assert(0);

  int tColorBin8 = kCyan;
  int tColorBin16 = kMagenta;

  TH1 *tCfNoEPBin, *tCfEPBin8, *tCfEPBin16;

  tCfNoEPBin = aGenNoEPBin->GetSharedAn()->GetKStarCfHeavy(tAnalysisNumber)->GetHeavyCfClone();
    tCfNoEPBin->SetMarkerStyle(tMarkerStyleNoBin);
    tCfNoEPBin->SetMarkerSize(tMarkerSize);
    tCfNoEPBin->SetMarkerColor(tColor);
    tCfNoEPBin->SetLineColor(tColor);

  tCfEPBin8 = aGenEPBin8->GetSharedAn()->GetKStarCfHeavy(tAnalysisNumber)->GetHeavyCfClone();
    tCfEPBin8->SetMarkerStyle(tMarkerStyleBin8);
    tCfEPBin8->SetMarkerSize(tMarkerSize);
    tCfEPBin8->SetMarkerColor(tColorBin8);
    tCfEPBin8->SetLineColor(tColorBin8);

  tCfEPBin16 = aGenEPBin16->GetSharedAn()->GetKStarCfHeavy(tAnalysisNumber)->GetHeavyCfClone();
    tCfEPBin16->SetMarkerStyle(tMarkerStyleBin16);
    tCfEPBin16->SetMarkerSize(tMarkerSize);
    tCfEPBin16->SetMarkerColor(tColorBin16);
    tCfEPBin16->SetLineColor(tColorBin16);

  //---------------------------------------------------------------------------------------------------------
  tCfNoEPBin->GetXaxis()->SetRangeUser(tXLow, tXHigh);
  tCfNoEPBin->GetYaxis()->SetRangeUser(tYLow, tYHigh);

  tCfNoEPBin->GetXaxis()->SetTitle("#it{k}* (GeV/#it{c})");
  tCfNoEPBin->GetYaxis()->SetTitle("#it{C}(#it{k}*)");

  tCfNoEPBin->Draw();
  tCfEPBin8->Draw("same");
  tCfEPBin16->Draw("same");
  //---------------------------------------------------------------------------------------------------------

  TLegend* tLeg = new TLegend(0.325, 0.15, 0.675, 0.45, "", "NDC");
    tLeg->SetFillColor(0);
    tLeg->SetBorderSize(0);
    tLeg->SetHeader("Event-Plane Binning", "C");
    tLeg->AddEntry(tCfNoEPBin, "No Binning", "p");
    tLeg->AddEntry(tCfEPBin8, "8 Bins", "p");
    tLeg->AddEntry(tCfEPBin16, "16 Bins", "p");
  tLeg->Draw();
  //---------------------------------------------------------------------------------------------------------



  TString tTextAnType = TString(cAnalysisRootTags[aGenNoEPBin->GetSharedAn()->GetFitPairAnalysis(tAnalysisNumber)->GetAnalysisType()]);
  TString tTextCentrality = TString(cPrettyCentralityTags[aGenNoEPBin->GetSharedAn()->GetFitPairAnalysis(tAnalysisNumber)->GetCentralityType()]);
  TString tCombinedText = tTextAnType + TString("  ") +  tTextCentrality;

  TPaveText* tText = new TPaveText(0.70, 0.75, 0.85, 0.85, "NDC");
    tText->SetFillColor(0);
    tText->SetBorderSize(0);
    tText->SetTextAlign(22);
    tText->SetTextFont(63);
    tText->SetTextSize(20);
  tText->AddText(tCombinedText);
  tText->Draw();


  double tXaxisRangeLow;
  if(tXLow<0) tXaxisRangeLow = 0.;
  else tXaxisRangeLow = tXLow;
  TLine *tLine = new TLine(tXaxisRangeLow,1.,tXHigh,1.);
  tLine->SetLineColor(TColor::GetColorTransparent(kGray,0.75));
  tLine->Draw();

  return tReturnCan;
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
  TString tResultsDate = "20180307";


  AnalysisRunType tAnRunType = kTrain;
  int tNPartialAnalysis = 2;
//  CentralityType tCentType = k0010;  //TODO
  CentralityType tCentType = kMB;
  FitGeneratorType tGenType = kPairwConj;

  bool bDrawAll = false;
  bool SaveImages = false;
  TString tSaveFileType = "pdf";
  TString tSaveDir = "/home/jesse/Analysis/Dissertation/6_Fitting/6.5_NonFlatBackground/AdditionalFigures/";

//-----------------------------------------------------------------------------
  TString tDirectoryBase_cLamcKch = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_%s/",tResultsDate.Data());
  TString tFileLocationBase_cLamcKch = TString::Format("%sResults_cLamcKch_%s",tDirectoryBase_cLamcKch.Data(),tResultsDate.Data());
  TString tFileLocationBaseMC_cLamcKch = TString::Format("%sResults_cLamcKchMC_%s",tDirectoryBase_cLamcKch.Data(),tResultsDate.Data());

  TString tDirectoryBase_cLamK0 = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamK0_%s/",tResultsDate.Data());
  TString tFileLocationBase_cLamK0 = TString::Format("%sResults_cLamK0_%s",tDirectoryBase_cLamK0.Data(),tResultsDate.Data());
  TString tFileLocationBaseMC_cLamK0 = TString::Format("%sResults_cLamK0MC_%s",tDirectoryBase_cLamK0.Data(),tResultsDate.Data());

//------------------------------------------------------------------------------

  TString tResultsDate_EPBin8 = "20180313_EPBinning8";

  TString tDirectoryBase_cLamcKch_EPBin8 = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_%s/",tResultsDate_EPBin8.Data());
  TString tFileLocationBase_cLamcKch_EPBin8 = TString::Format("%sResults_cLamcKch_%s",tDirectoryBase_cLamcKch_EPBin8.Data(),tResultsDate_EPBin8.Data());
  TString tFileLocationBaseMC_cLamcKch_EPBin8 = TString::Format("%sResults_cLamcKchMC_%s",tDirectoryBase_cLamcKch_EPBin8.Data(),tResultsDate_EPBin8.Data());

  TString tDirectoryBase_cLamK0_EPBin8 = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamK0_%s/",tResultsDate_EPBin8.Data());
  TString tFileLocationBase_cLamK0_EPBin8 = TString::Format("%sResults_cLamK0_%s",tDirectoryBase_cLamK0_EPBin8.Data(),tResultsDate_EPBin8.Data());
  TString tFileLocationBaseMC_cLamK0_EPBin8 = TString::Format("%sResults_cLamK0MC_%s",tDirectoryBase_cLamK0_EPBin8.Data(),tResultsDate_EPBin8.Data());

  //-----------------------------------------------------------------------------
  TString tResultsDate_EPBin16 = "20180314_EPBinning16";

  TString tDirectoryBase_cLamcKch_EPBin16 = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_%s/",tResultsDate_EPBin16.Data());
  TString tFileLocationBase_cLamcKch_EPBin16 = TString::Format("%sResults_cLamcKch_%s",tDirectoryBase_cLamcKch_EPBin16.Data(),tResultsDate_EPBin16.Data());
  TString tFileLocationBaseMC_cLamcKch_EPBin16 = TString::Format("%sResults_cLamcKchMC_%s",tDirectoryBase_cLamcKch_EPBin16.Data(),tResultsDate_EPBin16.Data());

  TString tDirectoryBase_cLamK0_EPBin16 = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamK0_%s/",tResultsDate_EPBin16.Data());
  TString tFileLocationBase_cLamK0_EPBin16 = TString::Format("%sResults_cLamK0_%s",tDirectoryBase_cLamK0_EPBin16.Data(),tResultsDate_EPBin16.Data());
  TString tFileLocationBaseMC_cLamK0_EPBin16 = TString::Format("%sResults_cLamK0MC_%s",tDirectoryBase_cLamK0_EPBin16.Data(),tResultsDate_EPBin16.Data());

  //-------------------------------------------------------------------------------

  if(bDrawAll)
  {
    FitGenerator* tLamKchP = new FitGenerator(tFileLocationBase_cLamcKch, tFileLocationBaseMC_cLamcKch, kLamKchP, tCentType, tAnRunType, tNPartialAnalysis, tGenType);
    FitGenerator* tLamKchM = new FitGenerator(tFileLocationBase_cLamcKch, tFileLocationBaseMC_cLamcKch, kLamKchM, tCentType, tAnRunType, tNPartialAnalysis, tGenType);
    FitGenerator* tLamK0 =   new FitGenerator(tFileLocationBase_cLamK0,   tFileLocationBaseMC_cLamK0,   kLamK0,   tCentType, tAnRunType, tNPartialAnalysis, tGenType);
    //-----
    FitGenerator* tLamKchP_EPBin8 = new FitGenerator(tFileLocationBase_cLamcKch_EPBin8,tFileLocationBaseMC_cLamcKch_EPBin8,kLamKchP, tCentType,tAnRunType,tNPartialAnalysis,tGenType);
    FitGenerator* tLamKchM_EPBin8 = new FitGenerator(tFileLocationBase_cLamcKch_EPBin8,tFileLocationBaseMC_cLamcKch_EPBin8,kLamKchM, tCentType,tAnRunType,tNPartialAnalysis,tGenType);
    FitGenerator* tLamK0_EPBin8 = new FitGenerator(tFileLocationBase_cLamK0_EPBin8,tFileLocationBaseMC_cLamK0_EPBin8,kLamK0, tCentType,tAnRunType,tNPartialAnalysis,tGenType);
    //-----
    FitGenerator* tLamKchP_EPBin16 = new FitGenerator(tFileLocationBase_cLamcKch_EPBin16,tFileLocationBaseMC_cLamcKch_EPBin16,kLamKchP, tCentType,tAnRunType,tNPartialAnalysis,tGenType);
    FitGenerator* tLamKchM_EPBin16 = new FitGenerator(tFileLocationBase_cLamcKch_EPBin16,tFileLocationBaseMC_cLamcKch_EPBin16,kLamKchM, tCentType,tAnRunType,tNPartialAnalysis,tGenType);
    FitGenerator* tLamK0_EPBin16 = new FitGenerator(tFileLocationBase_cLamK0_EPBin16,tFileLocationBaseMC_cLamK0_EPBin16,kLamK0, tCentType,tAnRunType,tNPartialAnalysis,tGenType);

    TCanvas* tCompareEPBinning_LamK0 = CompareEPBinning(tLamK0, tLamK0_EPBin8, tLamK0_EPBin16);
    TCanvas* tCompareEPBinning_LamKchP = CompareEPBinning(tLamKchP, tLamKchP_EPBin8, tLamKchP_EPBin16);
    TCanvas* tCompareEPBinning_LamKchM = CompareEPBinning(tLamKchM, tLamKchM_EPBin8, tLamKchM_EPBin16);
    if(SaveImages)
    {
      tCompareEPBinning_LamK0->SaveAs(tSaveDir + tCompareEPBinning_LamK0->GetName() + TString::Format(".%s", tSaveFileType.Data()));
      tCompareEPBinning_LamKchP->SaveAs(tSaveDir + tCompareEPBinning_LamKchP->GetName() + TString::Format(".%s", tSaveFileType.Data()));
      tCompareEPBinning_LamKchM->SaveAs(tSaveDir + tCompareEPBinning_LamKchM->GetName() + TString::Format(".%s", tSaveFileType.Data()));
    }
  }
  else
  {
    FitGenerator* tLamKchP = new FitGenerator(tFileLocationBase_cLamcKch, tFileLocationBaseMC_cLamcKch, kLamKchP, tCentType, tAnRunType, tNPartialAnalysis, tGenType);
    FitGenerator* tLamKchP_EPBin8 = new FitGenerator(tFileLocationBase_cLamcKch_EPBin8,tFileLocationBaseMC_cLamcKch_EPBin8,kLamKchP, tCentType,tAnRunType,tNPartialAnalysis,tGenType);
    FitGenerator* tLamKchP_EPBin16 = new FitGenerator(tFileLocationBase_cLamcKch_EPBin16,tFileLocationBaseMC_cLamcKch_EPBin16,kLamKchP, tCentType,tAnRunType,tNPartialAnalysis,tGenType);

    TCanvas* tCompareEPBinning_LamKchP_3050 = CompareEPBinning_SingleCentrality(tLamKchP, tLamKchP_EPBin8, tLamKchP_EPBin16, k3050);
    if(SaveImages) tCompareEPBinning_LamKchP_3050->SaveAs(tSaveDir + tCompareEPBinning_LamKchP_3050->GetName() + TString::Format(".%s", tSaveFileType.Data()));
  }

//------------------------------------------------------------------------------

  tFullTimer.Stop();
  cout << "Finished program: ";
  tFullTimer.PrintInterval();

  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.
cout << "DONE" << endl;
  return 0;
}
