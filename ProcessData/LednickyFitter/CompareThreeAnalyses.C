/* CompareTwoAnalyses.C */
/* Originally CompareIgnoreOnFlyStatus.C, used to compare different settings of
   ignoreOnFlyStatus of V0s in analyses */

#include "FitGenerator.h"
class FitGenerator;

#include "CanvasPartition.h"
class CanvasPartition;


//________________________________________________________________________________________________________________
TCanvas* DrawKStarCfs(FitGenerator* aFG1, FitGenerator* aFG2, FitGenerator* aFG3, bool aZoom=false, TString aCanNameModifier="")
{
  AnalysisType aAnType = aFG1->GetFitSharedAnalyses()->GetFitPairAnalysis(0)->GetAnalysisType();
  AnalysisType aConjType = aFG1->GetFitSharedAnalyses()->GetFitPairAnalysis(1)->GetAnalysisType();

  //-------------------------
  int tNAnlyses = aFG1->GetFitSharedAnalyses()->GetNFitPairAnalysis();  //NOTE: this macro designed for 3 or 6 pair analyses!
  bool tConjIncluded = true;
  if(tNAnlyses%2 != 0) tConjIncluded=false;
  //-------------------------

  TString tCanvasName = TString("canKStarCfs");
  if(aZoom) tCanvasName += TString("Zoom");
  tCanvasName += TString(cAnalysisBaseTags[aAnType]);
  if(tConjIncluded) tCanvasName += TString("wConj");
  tCanvasName += aCanNameModifier;

  int tNx=2, tNy=3;
  if(!tConjIncluded) tNx=1;

  double tXLow = -0.02;
//  double tXHigh = 0.99;
  double tXHigh = aFG1->GetKStarCf(0)->GetXaxis()->GetBinUpEdge(aFG1->GetKStarCf(0)->GetNbinsX())-0.01;
  if(aZoom) tXHigh = 0.32;

  double tYLow = 0.86;
  double tYHigh = 1.07;
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
/*
  if     (aAnType==kLamK0 || aAnType==kALamK0)     {tMarkerColor1 = kBlack; tMarkerColor2 = kGray+2; tMarkerColor3 = kGray+2;}
  else if(aAnType==kLamKchP || aAnType==kALamKchM) {tMarkerColor1 = kRed;   tMarkerColor2 = kRed+2;  tMarkerColor3 = kRed+2;}
  else if(aAnType==kLamKchM || aAnType==kALamKchP) {tMarkerColor1 = kBlue;  tMarkerColor2 = kBlue+2; tMarkerColor3 = kBlue+2;}
  else {tMarkerColor1=1; tMarkerColor2=1; tMarkerColor3=1;}
*/

  if     (aAnType==kLamK0 || aAnType==kALamK0)     tMarkerColor1 = kBlack;
  else if(aAnType==kLamKchP || aAnType==kALamKchM) tMarkerColor1 = kRed;
  else if(aAnType==kLamKchM || aAnType==kALamKchP) tMarkerColor1 = kBlue;
  else tMarkerColor1=1;

  tMarkerColor2 = kCyan+1;
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



      TString tTextAnType;
      if(tAnalysisNumber==0 || tAnalysisNumber==2 || tAnalysisNumber==4) tTextAnType = TString(cAnalysisRootTags[aAnType]);
      else if(tAnalysisNumber==1 || tAnalysisNumber==3 || tAnalysisNumber==5) tTextAnType = TString(cAnalysisRootTags[aConjType]);
      else assert(0);

      TPaveText* tAnTypeName = tCanPart->SetupTPaveText(tTextAnType,i,j,0.8,0.85);
      tCanPart->AddPadPaveText(tAnTypeName,i,j);

      CentralityType tCentType = aFG1->GetSharedAn()->GetFitPairAnalysis(tAnalysisNumber)->GetCentralityType();

      TString tTextCentrality = TString(cPrettyCentralityTags[tCentType]);
      TPaveText* tCentralityName = tCanPart->SetupTPaveText(tTextCentrality,i,j,0.05,0.85);
      tCanPart->AddPadPaveText(tCentralityName,i,j);

      if(i==0 && j==0)
      {
        tCanPart->SetupTLegend("", 0, 0, 0.35, 0.05, 0.50, 0.35);
        tCanPart->AddLegendEntry(0, 0, (TH1*)tCanPart->GetGraphsInPad(0,0)->At(0), "Normal: Num/Den", "p");
        tCanPart->AddLegendEntry(0, 0, (TH1*)tCanPart->GetGraphsInPad(0,0)->At(1), "Correct Stav.", "p");
        tCanPart->AddLegendEntry(0, 0, (TH1*)tCanPart->GetGraphsInPad(0,0)->At(2), "Incorrect Stav.", "p");
      }
    }
  }

  tCanPart->SetDrawUnityLine(true);
  tCanPart->DrawAll();
  tCanPart->DrawXaxisTitle("#it{k}* (GeV/#it{c})");
  if(tConjIncluded) tCanPart->DrawYaxisTitle("C(k*)",43,25,0.05,0.75);
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
  tResultsDate3 = "20180416";


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
  TString tSaveDir = "/home/jesse/Analysis/Presentations/AliFemto/20180627/Figures/";

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
  //-----------------------------------------------------------------------------
  bool bZoom = false;
  bool bDrawKStarCfs = true;

  //-----------------------------------------------------------------------------
  vector<TString> tUseStavCfTags = {"", "StavCf"};
  TString tCanNameModifier = TString::Format("_%s%svs%s%svs%s%s", tResultsDate1.Data(), tUseStavCfTags[bUseStavCf1].Data(), 
                                                                  tResultsDate2.Data(), tUseStavCfTags[bUseStavCf2].Data(),
                                                                  tResultsDate3.Data(), tUseStavCfTags[bUseStavCf3].Data());

  //-----------------------------------------------------------------------------
  if(bDrawKStarCfs)
  {
    TCanvas* tCan = DrawKStarCfs(tLamKchP1, tLamKchP2, tLamKchP3, bZoom, tCanNameModifier);
    if(SaveImages) tCan->SaveAs(TString::Format("%s%s.pdf", tSaveDir.Data(), tCan->GetName()));
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
