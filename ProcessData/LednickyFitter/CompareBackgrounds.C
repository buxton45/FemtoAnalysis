#include "FitGenerator.h"
class FitGenerator;


//________________________________________________________________________________________________________________
TCanvas* CompareLamKchAvgToLamK0(FitGenerator* aLamKchP, FitGenerator* aLamKchM, FitGenerator* aLamK0, bool aDrawIndividualKchAlso=false)
{
  TString tCanvasName = "CompareLamKchAvgToLamK0";
  if(aDrawIndividualKchAlso) tCanvasName += TString("_wIndivKch");

  vector<CentralityType> tCentralityTypes = aLamKchP->GetCentralityTypes();
  int tNAnalyses = aLamKchP->GetNAnalyses();
  for(unsigned int i=0; i<tCentralityTypes.size(); i++) tCanvasName += TString(cCentralityTags[tCentralityTypes[i]]);


  int tNx=0, tNy=0;
  if(tNAnalyses == 6) {tNx=2; tNy=3;}
  else if(tNAnalyses == 4) {tNx=2; tNy=2;}
  else if(tNAnalyses == 3) {tNx=1; tNy=tNAnalyses;}
  else if(tNAnalyses == 2 || tNAnalyses==1) {tNx=tNAnalyses; tNy=1;}
  else assert(0);

  double tXLow = -0.02;
  double tXHigh = 0.99;
  double tYLow = 0.86;
  double tYHigh = 1.07;
  CanvasPartition* tCanPart = new CanvasPartition(tCanvasName,tNx,tNy,tXLow,tXHigh,tYLow,tYHigh,0.12,0.05,0.13,0.05);

  assert(tNx*tNy == tNAnalyses);
  int tAnalysisNumber=0;

  int tMarkerStyle = 20;
  double tMarkerSizeLarge = 0.5;
  double tMarkerSizeSmall = 0.25;

  int tColorLamK0 = kBlack;
  int tColorLamKchP = kRed+1;
  int tColorLamKchM = kBlue+1;
  int tColorLamKchAvg = kMagenta+1;

  int tNx_Leg=0, tNy_Leg=0;

  TH1 *tCfLamKchP, *tCfLamKchM, *tCfLamK0;
  TH1 *tCfLamKchAvg;
  for(int j=0; j<tNy; j++)
  {
    for(int i=0; i<tNx; i++)
    {
      tAnalysisNumber = j*tNx + i;

      tCfLamKchP = aLamKchP->GetSharedAn()->GetKStarCfHeavy(tAnalysisNumber)->GetHeavyCfClone();
      tCfLamKchM = aLamKchM->GetSharedAn()->GetKStarCfHeavy(tAnalysisNumber)->GetHeavyCfClone();
      tCfLamK0 = aLamK0->GetSharedAn()->GetKStarCfHeavy(tAnalysisNumber)->GetHeavyCfClone();
      //---------------------------------------------------------------------------------------------------------
      tCfLamKchAvg = (TH1*)tCfLamKchP->Clone();
      tCfLamKchAvg->Add(tCfLamKchM);
      tCfLamKchAvg->Scale(0.5);

      tCanPart->AddGraph(i, j, tCfLamKchAvg, "", tMarkerStyle, tColorLamKchAvg, tMarkerSizeLarge);
      tCanPart->AddGraph(i, j, tCfLamK0, "", tMarkerStyle, tColorLamK0, tMarkerSizeLarge);
      if(i==tNx_Leg && j==tNy_Leg)
      {
        tCanPart->SetupTLegend("", i, j, 0.25, 0.05, 0.35, 0.50);
        tCanPart->AddLegendEntry(i, j, tCfLamK0, cAnalysisRootTags[kLamK0], "p");
        tCanPart->AddLegendEntry(i, j, tCfLamKchAvg, TString::Format("0.5*(%s+%s)", cAnalysisRootTags[kLamKchP], cAnalysisRootTags[kLamKchM]), "p");
      }

      if(aDrawIndividualKchAlso)
      {
        tCanPart->AddGraph(i, j, tCfLamKchP, "", tMarkerStyle, tColorLamKchP, tMarkerSizeSmall);
        tCanPart->AddGraph(i, j, tCfLamKchM, "", tMarkerStyle, tColorLamKchM, tMarkerSizeSmall);
        if(i==tNx_Leg && j==tNy_Leg)
        {
          tCanPart->AddLegendEntry(i, j, tCfLamKchP, cAnalysisRootTags[kLamKchP], "p");
          tCanPart->AddLegendEntry(i, j, tCfLamKchM, cAnalysisRootTags[kLamKchM], "p");
        }
      }
    }
  }

  tCanPart->SetDrawUnityLine(true);
  tCanPart->DrawAll();
  tCanPart->DrawXaxisTitle("k* (GeV/c)");
  tCanPart->DrawYaxisTitle("C(k*)",43,25,0.05,0.75);


  return tCanPart->GetCanvas();
}

//________________________________________________________________________________________________________________
TObjArray* DrawCfRatiosAndDiffs(FitGenerator* aGen1, FitGenerator* aGen2)
{
  AnalysisType tAnType1 = aGen1->GetSharedAn()->GetFitPairAnalysis(0)->GetAnalysisType();
  AnalysisType tAnType2 = aGen2->GetSharedAn()->GetFitPairAnalysis(0)->GetAnalysisType();
  TString tCanvasName1 = TString::Format("DrawCfRatios_%svs%s", cAnalysisBaseTags[tAnType1], cAnalysisBaseTags[tAnType2]);
  TString tCanvasName2 = TString::Format("DrawCfDiffs_%svs%s", cAnalysisBaseTags[tAnType1], cAnalysisBaseTags[tAnType2]);

  vector<CentralityType> tCentralityTypes = aGen1->GetCentralityTypes();
  int tNAnalyses = aGen1->GetNAnalyses();
  for(unsigned int i=0; i<tCentralityTypes.size(); i++) 
  {
    tCanvasName1 += TString(cCentralityTags[tCentralityTypes[i]]);
    tCanvasName2 += TString(cCentralityTags[tCentralityTypes[i]]);
  }

  int tNx=0, tNy=0;
  if(tNAnalyses == 6) {tNx=2; tNy=3;}
  else if(tNAnalyses == 4) {tNx=2; tNy=2;}
  else if(tNAnalyses == 3) {tNx=1; tNy=tNAnalyses;}
  else if(tNAnalyses == 2 || tNAnalyses==1) {tNx=tNAnalyses; tNy=1;}
  else assert(0);

  double tXLow = -0.02;
  double tXHigh = 0.99;

  double tYLow1 = 0.97;
  double tYHigh1 = 1.09;

  double tYLow2 = -0.09;
  double tYHigh2 = 0.19;

  CanvasPartition* tCanPart1 = new CanvasPartition(tCanvasName1,tNx,tNy,tXLow,tXHigh,tYLow1,tYHigh1,0.12,0.05,0.13,0.05);
  CanvasPartition* tCanPart2 = new CanvasPartition(tCanvasName2,tNx,tNy,tXLow,tXHigh,tYLow2,tYHigh2,0.12,0.05,0.13,0.05);

  assert(tNx*tNy == tNAnalyses);
  int tAnalysisNumber=0;

  int tMarkerStyle = 20;
  double tMarkerSize = 0.5;

  int tColor;

  if( ((tAnType1==kLamK0 || tAnType1==kALamK0) && (tAnType2==kLamKchP || tAnType2==kALamKchM)) ||
      ((tAnType2==kLamK0 || tAnType2==kALamK0) && (tAnType1==kLamKchP || tAnType1==kALamKchM)) )
  {
    tColor = kRed+2;
  }
  else if( ((tAnType1==kLamK0 || tAnType1==kALamK0) && (tAnType2==kLamKchM || tAnType2==kALamKchP)) ||
           ((tAnType2==kLamK0 || tAnType2==kALamK0) && (tAnType1==kLamKchM || tAnType1==kALamKchP)) )
  {
    tColor = kBlue+2;
  }
  else if( ((tAnType1==kLamKchP || tAnType1==kALamKchM) && (tAnType2==kLamKchM || tAnType2==kALamKchP)) ||
           ((tAnType2==kLamKchP || tAnType2==kALamKchM) && (tAnType1==kLamKchM || tAnType1==kALamKchP)) )
  {
    tColor = kMagenta+1;
  }
  else tColor = kYellow;


  int tNx_Leg=0, tNy_Leg=0;

  TString tTitle1 = TString::Format("#frac{C_{%s}(k*)}{C_{%s}(k*)}", cAnalysisRootTags[tAnType1], cAnalysisRootTags[tAnType2]);
  TString tTitle2 = TString::Format("C_{%s}(k*)-C_{%s}(k*)", cAnalysisRootTags[tAnType1], cAnalysisRootTags[tAnType2]);

  TH1 *tCf1, *tCf2;
  TH1 *tCfRatio, *tCfDiff;
  for(int j=0; j<tNy; j++)
  {
    for(int i=0; i<tNx; i++)
    {
      tAnalysisNumber = j*tNx + i;

      tCf1 = aGen1->GetSharedAn()->GetKStarCfHeavy(tAnalysisNumber)->GetHeavyCfClone();
      tCf2 = aGen2->GetSharedAn()->GetKStarCfHeavy(tAnalysisNumber)->GetHeavyCfClone();
      //---------------------------------------------------------------------------------------------------------
      tCfRatio = (TH1*)tCf1->Clone();
      tCfRatio->Divide(tCf2);

      tCfDiff = (TH1*)tCf1->Clone();
      tCfDiff->Add(tCf2, -1.0);

      tCanPart1->AddGraph(i, j, tCfRatio, "", tMarkerStyle, tColor, tMarkerSize);
      tCanPart2->AddGraph(i, j, tCfDiff, "", tMarkerStyle, tColor, tMarkerSize);
      if(i==tNx_Leg && j==tNy_Leg)
      {
        tCanPart1->SetupTLegend("", i, j, 0.25, 0.50, 0.35, 0.25);
        tCanPart1->AddLegendEntry(i, j, tCfRatio, tTitle1.Data(), "p");

        tCanPart2->SetupTLegend("", i, j, 0.25, 0.50, 0.60, 0.25);
        tCanPart2->AddLegendEntry(i, j, tCfDiff, tTitle2.Data(), "p");

      }
    }
  }

  TString tYaxisTitle1 = TString::Format("C_{%s}(k*)/C_{%s}(k*)", cAnalysisRootTags[tAnType1], cAnalysisRootTags[tAnType2]);
  tCanPart1->SetDrawUnityLine(true);
  tCanPart1->DrawAll();
  tCanPart1->DrawXaxisTitle("k* (GeV/#it{c})");
  tCanPart1->DrawYaxisTitle(tYaxisTitle1,43,20,0.075,0.65);

  TString tYaxisTitle2 = TString::Format("C_{%s}(k*)-C_{%s}(k*)", cAnalysisRootTags[tAnType1], cAnalysisRootTags[tAnType2]);
  tCanPart2->SetDrawUnityLine(true);
  tCanPart2->DrawAll();
  tCanPart2->DrawXaxisTitle("k* (GeV/#it{c})");
  tCanPart2->DrawYaxisTitle(tYaxisTitle2,43,20,0.075,0.65);


  TObjArray* tReturnArray = new TObjArray();
  tReturnArray->Add((TCanvas*)tCanPart1->GetCanvas());
  tReturnArray->Add((TCanvas*)tCanPart2->GetCanvas());
  return tReturnArray;
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
//  TString tResultsDate = "20161027";
//  TString tResultsDate = "20171220_onFlyStatusFalse";
  TString tResultsDate = "20171227";
//  TString tResultsDate = "20171227_LHC10h";
//  TString tResultsDate = "20180104_useIsProbableElectronMethodTrue";
//  TString tResultsDate = "20180104_useIsProbableElectronMethodFalse";

  AnalysisRunType tAnRunType = kTrain;
  int tNPartialAnalysis = 2;
//  CentralityType tCentType = k0010;  //TODO
  CentralityType tCentType = kMB;
  FitGeneratorType tGenType = kPairwConj;

  bool SaveImages = false;
  TString tSaveFileType = "eps";
  TString tSaveDir = "/home/jesse/Analysis/Presentations/GroupMeetings/20180308/Figures/";


//-----------------------------------------------------------------------------
  TString tDirectoryBase_cLamcKch = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_%s/",tResultsDate.Data());
  TString tFileLocationBase_cLamcKch = TString::Format("%sResults_cLamcKch_%s",tDirectoryBase_cLamcKch.Data(),tResultsDate.Data());
  TString tFileLocationBaseMC_cLamcKch = TString::Format("%sResults_cLamcKchMC_%s",tDirectoryBase_cLamcKch.Data(),tResultsDate.Data());

  TString tDirectoryBase_cLamK0 = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamK0_%s/",tResultsDate.Data());
  TString tFileLocationBase_cLamK0 = TString::Format("%sResults_cLamK0_%s",tDirectoryBase_cLamK0.Data(),tResultsDate.Data());
  TString tFileLocationBaseMC_cLamK0 = TString::Format("%sResults_cLamK0MC_%s",tDirectoryBase_cLamK0.Data(),tResultsDate.Data());
//-----------------------------------------------------------------------------

  FitGenerator* tLamKchP = new FitGenerator(tFileLocationBase_cLamcKch,tFileLocationBaseMC_cLamcKch,kLamKchP, tCentType,tAnRunType,tNPartialAnalysis,tGenType);
  FitGenerator* tLamKchM = new FitGenerator(tFileLocationBase_cLamcKch,tFileLocationBaseMC_cLamcKch,kLamKchM, tCentType,tAnRunType,tNPartialAnalysis,tGenType);
  FitGenerator* tLamK0 = new FitGenerator(tFileLocationBase_cLamK0,tFileLocationBaseMC_cLamK0,kLamK0, tCentType,tAnRunType,tNPartialAnalysis,tGenType);

//-------------------------------------------------------------------------------

  bool tDrawIndividualKchAlso = false;
  TCanvas* tCanCompareLamKchAvgToLamK0 = CompareLamKchAvgToLamK0(tLamKchP, tLamKchM, tLamK0, tDrawIndividualKchAlso);

  TObjArray* tDrawCfRatiosAndDiffs_LamKchM_LamKchP = DrawCfRatiosAndDiffs(tLamKchM, tLamKchP);
  TObjArray* tDrawCfRatiosAndDiffs_LamKchM_LamK0 = DrawCfRatiosAndDiffs(tLamKchM, tLamK0);
  TObjArray* tDrawCfRatiosAndDiffs_LamKchP_LamK0 = DrawCfRatiosAndDiffs(tLamKchP, tLamK0);

//-------------------------------------------------------------------------------

  if(SaveImages)
  {
    tCanCompareLamKchAvgToLamK0->SaveAs(tSaveDir + tCanCompareLamKchAvgToLamK0->GetName() + TString::Format(".%s", tSaveFileType.Data()));

    for(int i=0; i<tDrawCfRatiosAndDiffs_LamKchM_LamKchP->GetEntries(); i++)
    {
      ((TCanvas*)tDrawCfRatiosAndDiffs_LamKchM_LamKchP->At(i))->SaveAs(tSaveDir 
        + ((TCanvas*)tDrawCfRatiosAndDiffs_LamKchM_LamKchP->At(i))->GetName() + TString::Format(".%s", tSaveFileType.Data()));
    }

    for(int i=0; i<tDrawCfRatiosAndDiffs_LamKchM_LamK0->GetEntries(); i++)
    {
      ((TCanvas*)tDrawCfRatiosAndDiffs_LamKchM_LamK0->At(i))->SaveAs(tSaveDir 
        + ((TCanvas*)tDrawCfRatiosAndDiffs_LamKchM_LamK0->At(i))->GetName() + TString::Format(".%s", tSaveFileType.Data()));
    }

    for(int i=0; i<tDrawCfRatiosAndDiffs_LamKchP_LamK0->GetEntries(); i++)
    {
      ((TCanvas*)tDrawCfRatiosAndDiffs_LamKchP_LamK0->At(i))->SaveAs(tSaveDir 
        + ((TCanvas*)tDrawCfRatiosAndDiffs_LamKchP_LamK0->At(i))->GetName() + TString::Format(".%s", tSaveFileType.Data()));
    }


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
