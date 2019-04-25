/* From CompareThreeAnalyses.C */


#include "FitGenerator.h"
class FitGenerator;

#include "CanvasPartition.h"
class CanvasPartition;


//________________________________________________________________________________________________________________
TCanvas* DrawKStarCfs(FitGenerator* aFG1, FitGenerator* aFG2, FitGenerator* aFG3, bool aZoomX=false, bool aZoomY=false, TString aCanNameModifier="")
{
  AnalysisType aAnType = aFG1->GetFitSharedAnalyses()->GetFitPairAnalysis(0)->GetAnalysisType();
  AnalysisType aConjType = aFG1->GetFitSharedAnalyses()->GetFitPairAnalysis(1)->GetAnalysisType();

  //-------------------------
  int tNAnlyses = aFG1->GetFitSharedAnalyses()->GetNFitPairAnalysis();  //NOTE: this macro designed for 3 or 6 pair analyses!
  bool tConjIncluded = true;
  if(tNAnlyses%2 != 0) tConjIncluded=false;
  //-------------------------

  TString tCanvasName = TString("canKStarCfs");
  if(aZoomX) tCanvasName += TString("ZoomX");
  if(aZoomY) tCanvasName += TString("ZoomY");
  tCanvasName += TString(cAnalysisBaseTags[aAnType]);
  if(tConjIncluded) tCanvasName += TString("wConj");
  tCanvasName += aCanNameModifier;

  int tNx=2, tNy=3;
  if(!tConjIncluded) tNx=1;

  double tXLow = -0.02;
//  double tXHigh = 0.99;
  double tXHigh = aFG1->GetKStarCf(0)->GetXaxis()->GetBinUpEdge(aFG1->GetKStarCf(0)->GetNbinsX())-0.01;
  if(aZoomX) tXHigh = 0.329;

  double tYLow = 0.86;
  double tYHigh = 1.07;
  if(aZoomY)
  {
    tYLow = 0.951;
    tYHigh = 1.029;
  }
  CanvasPartition* tCanPart = new CanvasPartition(tCanvasName,tNx,tNy,tXLow,tXHigh,tYLow,tYHigh,0.12,0.05,0.13,0.05);
  if(!tConjIncluded) tCanPart->GetCanvas()->SetCanvasSize(350,500);

  int tAnalysisNumber=0;

  int tMarkerStyle1 = 20;
  int tMarkerStyle2 = 24;
  int tMarkerStyle3 = 20;

  int tMarkerColor1 = 1;
  int tMarkerColor2 = 1;
  int tMarkerColor3 = 1;


  double tMarkerSize = 0.5;
  if(aCanNameModifier.Contains("Rebin")) tMarkerSize = 0.75;

  if     (aAnType==kLamK0 || aAnType==kALamK0)     {tMarkerColor1 = kBlack; tMarkerColor2 = kGray+2; tMarkerColor3 = kGray+2;}
  else if(aAnType==kLamKchP || aAnType==kALamKchM) {tMarkerColor1 = kRed;   tMarkerColor2 = kRed+2;  tMarkerColor3 = kRed+2;}
  else if(aAnType==kLamKchM || aAnType==kALamKchP) {tMarkerColor1 = kBlue;  tMarkerColor2 = kBlue+2; tMarkerColor3 = kBlue+2;}
  else {tMarkerColor1=1; tMarkerColor2=1; tMarkerColor3=1;}

  tMarkerColor2 = kGreen+1;

/*
  if     (aAnType==kLamK0 || aAnType==kALamK0)     tMarkerColor1 = kBlack;
  else if(aAnType==kLamKchP || aAnType==kALamKchM) tMarkerColor1 = kRed;
  else if(aAnType==kLamKchM || aAnType==kALamKchP) tMarkerColor1 = kBlue;
  else tMarkerColor1=1;
  tMarkerColor2 = kCyan+1;
*/
//  tMarkerColor3 = kMagenta;
  tMarkerColor3 = kMagenta;

  for(int j=0; j<tNy; j++)
  {
    for(int i=0; i<tNx; i++)
    {
      tAnalysisNumber = j*tNx + i;
      //---------------------------------------------------------------------------------------------------------

      tCanPart->AddGraph(i,j,(TH1*)aFG1->GetKStarCf(tAnalysisNumber),"",tMarkerStyle1,tMarkerColor1,tMarkerSize);
      tCanPart->AddGraph(i,j,(TH1*)aFG2->GetKStarCf(tAnalysisNumber),"",tMarkerStyle2,tMarkerColor2,tMarkerSize);
      tCanPart->AddGraph(i,j,(TH1*)aFG3->GetKStarCf(tAnalysisNumber),"",tMarkerStyle3,tMarkerColor3,tMarkerSize);

      tCanPart->AddGraph(i,j,(TH1*)aFG2->GetKStarCf(tAnalysisNumber),"",tMarkerStyle2,tMarkerColor2,tMarkerSize);

      TString tTextAnType;
      if(tAnalysisNumber==0 || tAnalysisNumber==2 || tAnalysisNumber==4) tTextAnType = TString(cAnalysisRootTags[aAnType]);
      else if(tAnalysisNumber==1 || tAnalysisNumber==3 || tAnalysisNumber==5) tTextAnType = TString(cAnalysisRootTags[aConjType]);
      else assert(0);

      TPaveText* tAnTypeName = tCanPart->SetupTPaveText(tTextAnType,i,j,0.8,0.85);
      tCanPart->AddPadPaveText(tAnTypeName,i,j);

      CentralityType tCentType = aFG1->GetSharedAn()->GetFitPairAnalysis(tAnalysisNumber)->GetCentralityType();

      TString tTextCentrality = TString(cPrettyCentralityTags[tCentType]);
      TPaveText* tCentralityName = tCanPart->SetupTPaveText(tTextCentrality,i,j,0.10,0.85);
      tCanPart->AddPadPaveText(tCentralityName,i,j);

      if(i==0 && j==0)
      {
        if(aZoomY) tCanPart->SetupTLegend("", 0, 0, 0.20, 0.05, 0.60, 0.45);
        else       tCanPart->SetupTLegend("", 0, 0, 0.20, 0.05, 0.60, 0.50);
        tCanPart->AddLegendEntry(0, 0, (TH1*)tCanPart->GetGraphsInPad(0,0)->At(0), "Normal: Num/Den", "p");
        tCanPart->AddLegendEntry(0, 0, (TH1*)tCanPart->GetGraphsInPad(0,0)->At(1), "Correct Stav.", "p");
        tCanPart->AddLegendEntry(0, 0, (TH1*)tCanPart->GetGraphsInPad(0,0)->At(2), "Incorrect Stav.", "p");
      }
    }
  }

  tCanPart->SetDrawUnityLine(true);
  tCanPart->DrawAll();
  tCanPart->DrawXaxisTitle("#it{k}* (GeV/#it{c})");
  if(tConjIncluded) tCanPart->DrawYaxisTitle("#it{C}(#it{k}*)",43,25,0.05,0.75);
  else tCanPart->DrawYaxisTitle("#it{C}(#it{k}*)",43,25,0.075,0.875);

  return tCanPart->GetCanvas();
}



//________________________________________________________________________________________________________________
TCanvas* DrawKStarCfs_OnlyTwo(FitGenerator* aFG1, FitGenerator* aFG2, bool aZoomX=false, bool aZoomY=false, TString aCanNameModifier="")
{
  AnalysisType aAnType = aFG1->GetFitSharedAnalyses()->GetFitPairAnalysis(0)->GetAnalysisType();
  AnalysisType aConjType = aFG1->GetFitSharedAnalyses()->GetFitPairAnalysis(1)->GetAnalysisType();

  //-------------------------
  int tNAnlyses = aFG1->GetFitSharedAnalyses()->GetNFitPairAnalysis();  //NOTE: this macro designed for 3 or 6 pair analyses!
  bool tConjIncluded = true;
  if(tNAnlyses%2 != 0) tConjIncluded=false;
  //-------------------------

  TString tCanvasName = TString("canKStarCfs");
  if(aZoomX) tCanvasName += TString("ZoomX");
  if(aZoomY) tCanvasName += TString("ZoomY");
  tCanvasName += TString(cAnalysisBaseTags[aAnType]);
  if(tConjIncluded) tCanvasName += TString("wConj");
  tCanvasName += aCanNameModifier;

  int tNx=2, tNy=3;
  if(!tConjIncluded) tNx=1;

  double tXLow = -0.075;
//  double tXHigh = 0.99;
  double tXHigh = aFG1->GetKStarCf(0)->GetXaxis()->GetBinUpEdge(aFG1->GetKStarCf(0)->GetNbinsX()-2)-0.01;
  if(aZoomX) tXHigh = 0.329;

  double tYLow = 0.86;
  double tYHigh = 1.07;
  if(aZoomY)
  {
    tYLow = 0.951;
    tYHigh = 1.029;
  }
  CanvasPartition* tCanPart = new CanvasPartition(tCanvasName,tNx,tNy,tXLow,tXHigh,tYLow,tYHigh,0.12,0.05,0.13,0.05);
  if(!tConjIncluded) tCanPart->GetCanvas()->SetCanvasSize(350,500);

  int tAnalysisNumber=0;

  int tMarkerStyle1 = 20;
  int tMarkerStyle2 = 24;

  int tMarkerColor1 = 1;
  int tMarkerColor2 = 1;


  double tMarkerSize = 0.5;
  if(aCanNameModifier.Contains("Rebin")) tMarkerSize = 0.75;

  if     (aAnType==kLamK0 || aAnType==kALamK0)     {tMarkerColor1 = kBlack; tMarkerColor2 = kGray+2;}
  else if(aAnType==kLamKchP || aAnType==kALamKchM) {tMarkerColor1 = kRed;   tMarkerColor2 = kRed+2;}
  else if(aAnType==kLamKchM || aAnType==kALamKchP) {tMarkerColor1 = kBlue;  tMarkerColor2 = kBlue+2;}
  else {tMarkerColor1=1; tMarkerColor2=1;}

  tMarkerColor2 = kGreen+1;

/*
  if     (aAnType==kLamK0 || aAnType==kALamK0)     tMarkerColor1 = kBlack;
  else if(aAnType==kLamKchP || aAnType==kALamKchM) tMarkerColor1 = kRed;
  else if(aAnType==kLamKchM || aAnType==kALamKchP) tMarkerColor1 = kBlue;
  else tMarkerColor1=1;
  tMarkerColor2 = kCyan+1;
*/

  for(int j=0; j<tNy; j++)
  {
    for(int i=0; i<tNx; i++)
    {
      tAnalysisNumber = j*tNx + i;
      //---------------------------------------------------------------------------------------------------------

      tCanPart->AddGraph(i,j,(TH1*)aFG1->GetKStarCf(tAnalysisNumber),"",tMarkerStyle1,tMarkerColor1,tMarkerSize);
      tCanPart->AddGraph(i,j,(TH1*)aFG2->GetKStarCf(tAnalysisNumber),"",tMarkerStyle2,tMarkerColor2,tMarkerSize);

      tCanPart->AddGraph(i,j,(TH1*)aFG2->GetKStarCf(tAnalysisNumber),"",tMarkerStyle2,tMarkerColor2,tMarkerSize);

      TString tTextAnType;
      if(tAnalysisNumber==0 || tAnalysisNumber==2 || tAnalysisNumber==4) tTextAnType = TString(cAnalysisRootTags[aAnType]);
      else if(tAnalysisNumber==1 || tAnalysisNumber==3 || tAnalysisNumber==5) tTextAnType = TString(cAnalysisRootTags[aConjType]);
      else assert(0);

      TPaveText* tAnTypeName = tCanPart->SetupTPaveText(tTextAnType,i,j,0.8,0.85);
      tCanPart->AddPadPaveText(tAnTypeName,i,j);

      CentralityType tCentType = aFG1->GetSharedAn()->GetFitPairAnalysis(tAnalysisNumber)->GetCentralityType();

      TString tTextCentrality = TString(cPrettyCentralityTags[tCentType]);
      TPaveText* tCentralityName = tCanPart->SetupTPaveText(tTextCentrality,i,j,0.10,0.85);
      tCanPart->AddPadPaveText(tCentralityName,i,j);

      if(i==0 && j==0)
      {
        if(aZoomY) tCanPart->SetupTLegend("", 0, 0, 0.20, 0.05, 0.60, 0.45);
        else       tCanPart->SetupTLegend("", 0, 0, 0.20, 0.05, 0.60, 0.50);
        tCanPart->AddLegendEntry(0, 0, (TH1*)tCanPart->GetGraphsInPad(0,0)->At(0), "Normal: Num/Den", "p");
        tCanPart->AddLegendEntry(0, 0, (TH1*)tCanPart->GetGraphsInPad(0,0)->At(1), "Stavinskiy", "p");
      }

      if(i==1 && j==0)
      {
        TString tTextSysInfo = TString("ALICE Pb-Pb #sqrt{#it{s}_{NN}} = 2.76 TeV");
        TPaveText* tSysInfo = tCanPart->SetupTPaveText(tTextSysInfo,i,j,0.20,0.125,0.725,0.15,43,17);
        tCanPart->AddPadPaveText(tSysInfo,i,j);
      }
    }
  }

  tCanPart->SetDrawUnityLine(true);
  tCanPart->DrawAll();
  tCanPart->DrawXaxisTitle("#it{k}* (GeV/#it{c})");
  if(tConjIncluded) tCanPart->DrawYaxisTitle("#it{C}(#it{k}*)",43,25,0.05,0.75);
  else tCanPart->DrawYaxisTitle("#it{C}(#it{k}*)",43,25,0.075,0.875);

  return tCanPart->GetCanvas();
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
  tResultsDate3 = "20180416_IncorrectStav";

  int aRebin=2;
  bool aCustomRebin = true;
  if(aCustomRebin) aRebin=2;  //Just want it to be != 1, so markers will be resized

  AnalysisType tAnType = kLamKchP;
  AnalysisRunType tAnRunType = kTrain;
  int tNPartialAnalysis = 2;
  CentralityType tCentType = kMB;  //TODO
  FitGeneratorType tGenType = kPairwConj;

  AnalysisType tConjType;
  if(tAnType==kLamK0) {tConjType=kALamK0;}
  else if(tAnType==kLamKchP) {tConjType=kALamKchM;}
  else if(tAnType==kLamKchM) {tConjType=kALamKchP;}

  bool bUseStavCf1 = false;
  bool bUseStavCf2 = true;
  bool bUseStavCf3 = true;

  bool SaveImages = false;
  TString tSaveDir = "/home/jesse/Analysis/FemtoAnalysis/AnalysisNotes/4_CorrelationFunctions/Figures/AllThreeTogether/";
  TString tSaveDir_OnlyTwo = "/home/jesse/Analysis/FemtoAnalysis/AnalysisNotes/4_CorrelationFunctions/Figures/OnlyTwo/";
  TString tSaveFileType = "pdf";

  TString tGeneralAnTypeName;
  if(tAnType==kLamK0 || tAnType==kALamK0) tGeneralAnTypeName = "cLamK0";
  else if(tAnType==kLamKchP || tAnType==kALamKchM || tAnType==kLamKchM || tAnType==kALamKchP) tGeneralAnTypeName = "cLamcKch";
  else assert(0);


  TString tDirectoryBase1 = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_%s_%s/",tGeneralAnTypeName.Data(),tResultsDate1.Data());
  TString tFileLocationBase1 = TString::Format("%sResults_%s_%s",tDirectoryBase1.Data(),tGeneralAnTypeName.Data(),tResultsDate1.Data());
  TString tFileLocationBaseMC1 = TString::Format("%sResults_%sMC_%s",tDirectoryBase1.Data(),tGeneralAnTypeName.Data(),tResultsDate1.Data());

  TString tDirectoryBase2 = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_%s_%s/",tGeneralAnTypeName.Data(),tResultsDate2.Data());
  TString tFileLocationBase2 = TString::Format("%sResults_%s_%s",tDirectoryBase2.Data(),tGeneralAnTypeName.Data(),tResultsDate2.Data());
  TString tFileLocationBaseMC2 = TString::Format("%sResults_%sMC_%s",tDirectoryBase1.Data(),tGeneralAnTypeName.Data(),tResultsDate1.Data());

  TString tDirectoryBase3 = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_%s_%s/",tGeneralAnTypeName.Data(),tResultsDate3.Data());
  TString tFileLocationBase3 = TString::Format("%sResults_%s_%s",tDirectoryBase3.Data(),tGeneralAnTypeName.Data(),tResultsDate3.Data());
  TString tFileLocationBaseMC3 = TString::Format("%sResults_%sMC_%s",tDirectoryBase1.Data(),tGeneralAnTypeName.Data(),tResultsDate1.Data());

  TString tSaveNameModifier = "";
  FitGenerator* tLamKchP1 = new FitGenerator(tFileLocationBase1, tFileLocationBaseMC1, tAnType, tCentType, tAnRunType, tNPartialAnalysis, tGenType, false, false, "", bUseStavCf1);
  FitGenerator* tLamKchP2 = new FitGenerator(tFileLocationBase2, tFileLocationBaseMC2, tAnType, tCentType, tAnRunType, tNPartialAnalysis, tGenType, false, false, "", bUseStavCf2);
  FitGenerator* tLamKchP3 = new FitGenerator(tFileLocationBase3, tFileLocationBaseMC3, tAnType, tCentType, tAnRunType, tNPartialAnalysis, tGenType, false, false, "", bUseStavCf3);


  vector<double> tCustomBins{0.00, 0.02, 0.04, 0.06, 0.08,
                             0.10, 0.12, 0.14, 0.16, 0.18,
                             0.20, 0.22, 0.24, 0.26, 0.28,
                             0.30, 0.32, 0.34, 0.36, 0.38,
                             0.40, 0.42, 0.44, 0.46, 0.48,
                             0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
                             1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.45,
                             1.50, 1.55, 1.60, 1.65, 1.70, 1.75, 1.80, 1.85, 1.90, 1.95, 2.00};

  if(aCustomRebin)
  {
    for(int i=0; i<tLamKchP1->GetNAnalyses(); i++) tLamKchP1->GetKStarCfHeavy(i)->Rebin((int)tCustomBins.size()-1, tCustomBins);
    for(int i=0; i<tLamKchP2->GetNAnalyses(); i++) tLamKchP2->GetKStarCfHeavy(i)->Rebin((int)tCustomBins.size()-1, tCustomBins);
    for(int i=0; i<tLamKchP3->GetNAnalyses(); i++) tLamKchP3->GetKStarCfHeavy(i)->Rebin((int)tCustomBins.size()-1, tCustomBins);
  }
  else if(aRebin != 1)
  {
    for(int i=0; i<tLamKchP1->GetNAnalyses(); i++) tLamKchP1->GetKStarCfHeavy(i)->Rebin(aRebin);
    for(int i=0; i<tLamKchP2->GetNAnalyses(); i++) tLamKchP2->GetKStarCfHeavy(i)->Rebin(aRebin);
    for(int i=0; i<tLamKchP3->GetNAnalyses(); i++) tLamKchP3->GetKStarCfHeavy(i)->Rebin(aRebin);
  }
  //-----------------------------------------------------------------------------
  bool bZoomX = false;
  bool bZoomY = false;
  bool bDrawKStarCfs = true;
  bool bDrawKStarCfs_OnlyTwo = true;

  //-----------------------------------------------------------------------------
  vector<TString> tUseStavCfTags = {"", "StavCf"};
  TString tCanNameModifier = TString::Format("_%s%svs%s%svs%s%s", tResultsDate1.Data(), tUseStavCfTags[bUseStavCf1].Data(), 
                                                                  tResultsDate2.Data(), tUseStavCfTags[bUseStavCf2].Data(),
                                                                  tResultsDate3.Data(), tUseStavCfTags[bUseStavCf3].Data());

  if(aCustomRebin)        tCanNameModifier += TString("_CustomRebin");
  else if(aRebin != 1) tCanNameModifier += TString::Format("_Rebin%d", aRebin);

  if(bDrawKStarCfs)
  {
    TCanvas* tCan = DrawKStarCfs(tLamKchP1, tLamKchP2, tLamKchP3, bZoomX, bZoomY, tCanNameModifier);
    if(SaveImages) tCan->SaveAs(TString::Format("%s%s.%s", tSaveDir.Data(), tCan->GetName(), tSaveFileType.Data()));
  }

  //-----------------------------------------------------------------------------
  TString tCanNameModifier_OnlyTwo = TString::Format("_%s%svs%s%s", tResultsDate1.Data(), tUseStavCfTags[bUseStavCf1].Data(), 
                                                                    tResultsDate2.Data(), tUseStavCfTags[bUseStavCf2].Data());

  if(aCustomRebin)        tCanNameModifier_OnlyTwo += TString("_CustomRebin");
  else if(aRebin != 1) tCanNameModifier_OnlyTwo += TString::Format("_Rebin%d", aRebin);

  if(bDrawKStarCfs_OnlyTwo)
  {
    TCanvas* tCan_OnlyTwo = DrawKStarCfs_OnlyTwo(tLamKchP1, tLamKchP2, bZoomX, bZoomY, tCanNameModifier_OnlyTwo);
    if(SaveImages) tCan_OnlyTwo->SaveAs(TString::Format("%s%s.%s", tSaveDir_OnlyTwo.Data(), tCan_OnlyTwo->GetName(), tSaveFileType.Data()));
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
