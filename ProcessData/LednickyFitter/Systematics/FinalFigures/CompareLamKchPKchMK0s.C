/* CompareTwoAnalyses.C */
/* Originally CompareIgnoreOnFlyStatus.C, used to compare different settings of
   ignoreOnFlyStatus of V0s in analyses */

#include "FitGenerator.h"
class FitGenerator;

#include "CanvasPartition.h"
class CanvasPartition;

//________________________________________________________________________________________________________________
CfHeavy* CombineTwoHeavyCfs(CfHeavy *aCf1, CfHeavy* aCf2)
{
  TString aReturnCfName = TString::Format("%s_and_%s", aCf1->GetHeavyCfName().Data(), aCf2->GetHeavyCfName().Data());

  vector<CfLite*> tCfLiteColl1 = aCf1->GetCfLiteCollection();
  vector<CfLite*> tCfLiteColl2 = aCf2->GetCfLiteCollection();

  vector<CfLite*> tReturnCfLiteColl(0);
  for(int i=0; i<tCfLiteColl1.size(); i++) tReturnCfLiteColl.push_back(tCfLiteColl1[i]);
  for(int i=0; i<tCfLiteColl2.size(); i++) tReturnCfLiteColl.push_back(tCfLiteColl2[i]);

  CfHeavy* tReturnCf = new CfHeavy(aReturnCfName, aReturnCfName, tReturnCfLiteColl, 0.32, 0.40);
  return tReturnCf;
}


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
  int tMarkerStyle3 = 20;

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
      TH1* tHist2 = (TH1*)aFG2->GetKStarCf(tAnalysisNumber);
      TH1* tHist3 = (TH1*)aFG3->GetKStarCf(tAnalysisNumber);

      tCanPart->AddGraph(i, j, tHist1, "", tMarkerStyle1, tMarkerColor1, tMarkerSize, "ex0");
      tCanPart->AddGraph(i, j, tHist2, "", tMarkerStyle2, tMarkerColor2, tMarkerSize, "ex0same");
      tCanPart->AddGraph(i, j, tHist3, "", tMarkerStyle3, tMarkerColor3, tMarkerSize, "ex0same");


      tCanPart->SetupTLegend("", i, j, 0.75, 0.15, 0.20, 0.35);
      tCanPart->AddLegendEntry(i, j, tHist1, cAnalysisRootTags[aFG1->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumber)->GetAnalysisType()], "p");
      tCanPart->AddLegendEntry(i, j, tHist2, cAnalysisRootTags[aFG2->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumber)->GetAnalysisType()], "p");
      tCanPart->AddLegendEntry(i, j, tHist3, cAnalysisRootTags[aFG3->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumber)->GetAnalysisType()], "p");


      CentralityType aCentType = aFG1->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumber)->GetCentralityType();
      TPaveText* tText = tCanPart->SetupTPaveText(cPrettyCentralityTags[aCentType], i, j, 0.75, 0.80, 0.15, 0.10, 63, 15);
      tCanPart->AddPadPaveText(tText, i, j);
    }
  }

  tCanPart->SetDrawUnityLine(true);
  tCanPart->DrawAll();
  tCanPart->DrawXaxisTitle("#it{k}* (GeV/#it{c})");
  tCanPart->DrawYaxisTitle("#it{C}(#it{k}*)",43,25,0.05,0.85);

  return tCanPart->GetCanvas();
}

//________________________________________________________________________________________________________________
TCanvas* DrawAllKStarCfs_CombineConj(FitGenerator* aFG1, FitGenerator* aFG2, FitGenerator* aFG3, bool aZoom=false, TString aCanNameModifier="")
{
  TString tCanvasName = TString("canKStarCfsAll_CombineConj");
  if(aZoom) tCanvasName += TString("Zoom");
  tCanvasName += aCanNameModifier;

  int tNx=1, tNy=3;

  double tXLow = -0.02;
  double tXHigh = 0.99;
  if(aZoom) tXHigh = 0.329;

  double tYLow = 0.86;
  double tYHigh = 1.07;

  CanvasPartition* tCanPart = new CanvasPartition(tCanvasName,tNx,tNy,tXLow,tXHigh,tYLow,tYHigh,0.12,0.0025,0.05,0.0025);
  tCanPart->SetDrawOptStat(false);
  tCanPart->GetCanvas()->SetCanvasSize(700, 1500);

  int tMarkerStyle1 = 20;
  int tMarkerStyle2 = 20;
  int tMarkerStyle3 = 20;

  int tMarkerColor1 = kRed+1;
  int tMarkerColor2 = kBlue+1;
  int tMarkerColor3 = kBlack;

  double tMarkerSize = 0.75;

  int tAnalysisNumberA=0, tAnalysisNumberB=0;
  for(int j=0; j<tNy; j++)
  {

    tAnalysisNumberA = 2*j;
    tAnalysisNumberB = 2*j + 1;

    CfHeavy* tCfHeavy1a = aFG1->GetKStarCfHeavy(tAnalysisNumberA);
    CfHeavy* tCfHeavy1b = aFG1->GetKStarCfHeavy(tAnalysisNumberB);
    CfHeavy* tCfHeavy1 = CombineTwoHeavyCfs(tCfHeavy1a, tCfHeavy1b);
    TH1* tHist1 = tCfHeavy1->GetHeavyCfClone();

    CfHeavy* tCfHeavy2a = aFG2->GetKStarCfHeavy(tAnalysisNumberA);
    CfHeavy* tCfHeavy2b = aFG2->GetKStarCfHeavy(tAnalysisNumberB);
    CfHeavy* tCfHeavy2 = CombineTwoHeavyCfs(tCfHeavy2a, tCfHeavy2b);
    TH1* tHist2 = tCfHeavy2->GetHeavyCfClone();

    CfHeavy* tCfHeavy3a = aFG3->GetKStarCfHeavy(tAnalysisNumberA);
    CfHeavy* tCfHeavy3b = aFG3->GetKStarCfHeavy(tAnalysisNumberB);
    CfHeavy* tCfHeavy3 = CombineTwoHeavyCfs(tCfHeavy3a, tCfHeavy3b);
    TH1* tHist3 = tCfHeavy3->GetHeavyCfClone();



    tCanPart->AddGraph(0, j, tHist1, "", tMarkerStyle1, tMarkerColor1, tMarkerSize, "ex0");
    tCanPart->AddGraph(0, j, tHist2, "", tMarkerStyle2, tMarkerColor2, tMarkerSize, "ex0same");
    tCanPart->AddGraph(0, j, tHist3, "", tMarkerStyle3, tMarkerColor3, tMarkerSize, "ex0same");

    TString tText1 = TString::Format("%s & %s", cAnalysisRootTags[aFG1->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumberA)->GetAnalysisType()], 
                                                cAnalysisRootTags[aFG1->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumberB)->GetAnalysisType()]);

    TString tText2 = TString::Format("%s & %s", cAnalysisRootTags[aFG2->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumberA)->GetAnalysisType()], 
                                                cAnalysisRootTags[aFG2->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumberB)->GetAnalysisType()]);

    TString tText3 = TString::Format("%s & %s", cAnalysisRootTags[aFG3->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumberA)->GetAnalysisType()], 
                                                cAnalysisRootTags[aFG3->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumberB)->GetAnalysisType()]);


    tCanPart->SetupTLegend("", 0, j, 0.75, 0.15, 0.20, 0.35);
    tCanPart->AddLegendEntry(0, j, tHist1, tText1.Data(), "p");
    tCanPart->AddLegendEntry(0, j, tHist2, tText2.Data(), "p");
    tCanPart->AddLegendEntry(0, j, tHist3, tText3.Data(), "p");


    assert(aFG1->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumberA)->GetCentralityType() == aFG1->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumberB)->GetCentralityType());
    CentralityType aCentType = aFG1->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumberA)->GetCentralityType();
    TPaveText* tText = tCanPart->SetupTPaveText(cPrettyCentralityTags[aCentType], 0, j, 0.75, 0.80, 0.20, 0.15, 63, 25);
    tCanPart->AddPadPaveText(tText, 0, j);
  }

  tCanPart->SetDrawUnityLine(true);
  tCanPart->DrawAll();

  tCanPart->DrawXaxisTitle("#it{k}* (GeV/#it{c})");
  tCanPart->DrawYaxisTitle("#it{C}(#it{k}*)",43,25,0.05,0.90);

  return tCanPart->GetCanvas();
}


//________________________________________________________________________________________________________________
TCanvas* DrawAvgLamKchvsK0(FitGenerator* aFG1, FitGenerator* aFG2, FitGenerator* aFG3, bool aZoom=false, TString aCanNameModifier="")
{
  TString tCanvasName = TString("canAvgLamKchvsK0All");
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

  int tMarkerStyleAvg = 20;
  int tMarkerStyle3 = 20;

  int tMarkerColorAvg = kMagenta;
  int tMarkerColor3 = kBlack;

  double tMarkerSize = 0.75;

  int tAnalysisNumber=0;
  for(int j=0; j<tNy; j++)
  {
    for(int i=0; i<tNx; i++)
    {
      tAnalysisNumber = j*tNx + i;

      CfHeavy* tCfHeavy1 = aFG1->GetKStarCfHeavy(tAnalysisNumber);
      CfHeavy* tCfHeavy2 = aFG2->GetKStarCfHeavy(tAnalysisNumber);
      CfHeavy* tCfHeavyAvg = CombineTwoHeavyCfs(tCfHeavy1, tCfHeavy2);
      TH1* tHistAvg = tCfHeavyAvg->GetHeavyCfClone();

      TH1* tHist3 = (TH1*)aFG3->GetKStarCf(tAnalysisNumber);

      tCanPart->AddGraph(i, j, tHistAvg, "", tMarkerStyleAvg, tMarkerColorAvg, tMarkerSize, "ex0");
      tCanPart->AddGraph(i, j, tHist3, "", tMarkerStyle3, tMarkerColor3, tMarkerSize, "ex0same");

      TString tTextAvg = TString::Format("%s + %s", cAnalysisRootTags[aFG1->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumber)->GetAnalysisType()], 
                                                    cAnalysisRootTags[aFG2->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumber)->GetAnalysisType()]);

      tCanPart->SetupTLegend("", i, j, 0.75, 0.15, 0.20, 0.35);
      tCanPart->AddLegendEntry(i, j, tHistAvg, tTextAvg.Data(), "p");
      tCanPart->AddLegendEntry(i, j, tHist3, cAnalysisRootTags[aFG3->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumber)->GetAnalysisType()], "p");


      CentralityType aCentType = aFG1->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumber)->GetCentralityType();
      TPaveText* tText = tCanPart->SetupTPaveText(cPrettyCentralityTags[aCentType], i, j, 0.75, 0.80, 0.15, 0.10, 63, 15);
      tCanPart->AddPadPaveText(tText, i, j);
    }
  }

  tCanPart->SetDrawUnityLine(true);
  tCanPart->DrawAll();
  tCanPart->DrawXaxisTitle("#it{k}* (GeV/#it{c})");
  tCanPart->DrawYaxisTitle("#it{C}(#it{k}*)",43,25,0.05,0.85);

  return tCanPart->GetCanvas();
}

//________________________________________________________________________________________________________________
TCanvas* DrawAvgLamKchvsK0_CombineConj(FitGenerator* aFG1, FitGenerator* aFG2, FitGenerator* aFG3, bool aZoom=false, TString aCanNameModifier="")
{
  TString tCanvasName = TString("canAvgLamKchvsK0All_CombineConj");
  if(aZoom) tCanvasName += TString("Zoom");
  tCanvasName += aCanNameModifier;

  int tNx=1, tNy=3;

  double tXLow = -0.02;
  double tXHigh = 0.99;
  if(aZoom) tXHigh = 0.329;

  double tYLow = 0.86;
  double tYHigh = 1.07;

  CanvasPartition* tCanPart = new CanvasPartition(tCanvasName,tNx,tNy,tXLow,tXHigh,tYLow,tYHigh,0.12,0.0025,0.05,0.0025);
  tCanPart->SetDrawOptStat(false);
  tCanPart->GetCanvas()->SetCanvasSize(700, 1500);

  int tMarkerStyleAvg = 20;
  int tMarkerStyle3 = 20;

  int tMarkerColorAvg = kMagenta;
  int tMarkerColor3 = kBlack;

  double tMarkerSize = 0.75;

  int tAnalysisNumberA=0, tAnalysisNumberB=0;
  for(int j=0; j<tNy; j++)
  {

    tAnalysisNumberA = 2*j;
    tAnalysisNumberB = 2*j + 1;

    CfHeavy* tCfHeavy1a = aFG1->GetKStarCfHeavy(tAnalysisNumberA);
    CfHeavy* tCfHeavy1b = aFG1->GetKStarCfHeavy(tAnalysisNumberB);
    CfHeavy* tCfHeavy1 = CombineTwoHeavyCfs(tCfHeavy1a, tCfHeavy1b);

    CfHeavy* tCfHeavy2a = aFG2->GetKStarCfHeavy(tAnalysisNumberA);
    CfHeavy* tCfHeavy2b = aFG2->GetKStarCfHeavy(tAnalysisNumberB);
    CfHeavy* tCfHeavy2 = CombineTwoHeavyCfs(tCfHeavy2a, tCfHeavy2b);

    CfHeavy* tCfHeavyAvg = CombineTwoHeavyCfs(tCfHeavy1, tCfHeavy2);
    TH1* tHistAvg = tCfHeavyAvg->GetHeavyCfClone();


    CfHeavy* tCfHeavy3a = aFG3->GetKStarCfHeavy(tAnalysisNumberA);
    CfHeavy* tCfHeavy3b = aFG3->GetKStarCfHeavy(tAnalysisNumberB);
    CfHeavy* tCfHeavy3 = CombineTwoHeavyCfs(tCfHeavy3a, tCfHeavy3b);
    TH1* tHist3 = tCfHeavy3->GetHeavyCfClone();

    tCanPart->AddGraph(0, j, tHistAvg, "", tMarkerStyleAvg, tMarkerColorAvg, tMarkerSize, "ex0");
    tCanPart->AddGraph(0, j, tHist3, "", tMarkerStyle3, tMarkerColor3, tMarkerSize, "ex0same");

    TString tText1 = TString::Format("%s & %s", cAnalysisRootTags[aFG1->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumberA)->GetAnalysisType()], 
                                                cAnalysisRootTags[aFG1->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumberB)->GetAnalysisType()]);

    TString tText2 = TString::Format("%s & %s", cAnalysisRootTags[aFG2->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumberA)->GetAnalysisType()], 
                                                cAnalysisRootTags[aFG2->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumberB)->GetAnalysisType()]);

    TString tTextAvg = TString::Format("(%s) + (%s)", tText1.Data(), tText2.Data());

    TString tText3 = TString::Format("%s & %s", cAnalysisRootTags[aFG3->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumberA)->GetAnalysisType()], 
                                                cAnalysisRootTags[aFG3->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumberB)->GetAnalysisType()]);


    tCanPart->SetupTLegend("", 0, j, 0.50, 0.15, 0.45, 0.35);
    tCanPart->AddLegendEntry(0, j, tHistAvg, tTextAvg.Data(), "p");
    tCanPart->AddLegendEntry(0, j, tHist3, tText3.Data(), "p");


    assert(aFG1->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumberA)->GetCentralityType() == aFG1->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumberB)->GetCentralityType());
    CentralityType aCentType = aFG1->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumberA)->GetCentralityType();
    TPaveText* tText = tCanPart->SetupTPaveText(cPrettyCentralityTags[aCentType], 0, j, 0.75, 0.80, 0.20, 0.15, 63, 25);
    tCanPart->AddPadPaveText(tText, 0, j);


  }

  tCanPart->SetDrawUnityLine(true);
  tCanPart->DrawAll();
  tCanPart->DrawXaxisTitle("#it{k}* (GeV/#it{c})");
  tCanPart->DrawYaxisTitle("#it{C}(#it{k}*)",43,25,0.05,0.90);

  return tCanPart->GetCanvas();
}

//________________________________________________________________________________________________________________
TCanvas* DrawAllKStarCfsAndAvgLamKchvsK0_CombineConj(FitGenerator* aFG1, FitGenerator* aFG2, FitGenerator* aFG3, bool aZoom=false, TString aCanNameModifier="")
{
  TString tCanvasName = TString("canKStarCfsAllAndAvgLamKchvsK0_CombineConj");
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
  int tMarkerStyle3 = 20;
  int tMarkerStyleAvg = 20;

  int tMarkerColor1 = kRed+1;
  int tMarkerColor2 = kBlue+1;
  int tMarkerColor3 = kBlack;
  int tMarkerColorAvg = kMagenta;

  double tMarkerSize = 0.75;


  //------------------------------------------------------------------------------------------------------
  int tAnalysisNumberA=0, tAnalysisNumberB=0;
  for(int j=0; j<tNy; j++)
  {

    tAnalysisNumberA = 2*j;
    tAnalysisNumberB = 2*j + 1;

    CfHeavy* tCfHeavy1a = aFG1->GetKStarCfHeavy(tAnalysisNumberA);
    CfHeavy* tCfHeavy1b = aFG1->GetKStarCfHeavy(tAnalysisNumberB);
    CfHeavy* tCfHeavy1 = CombineTwoHeavyCfs(tCfHeavy1a, tCfHeavy1b);
    TH1* tHist1 = tCfHeavy1->GetHeavyCfClone();

    CfHeavy* tCfHeavy2a = aFG2->GetKStarCfHeavy(tAnalysisNumberA);
    CfHeavy* tCfHeavy2b = aFG2->GetKStarCfHeavy(tAnalysisNumberB);
    CfHeavy* tCfHeavy2 = CombineTwoHeavyCfs(tCfHeavy2a, tCfHeavy2b);
    TH1* tHist2 = tCfHeavy2->GetHeavyCfClone();

    CfHeavy* tCfHeavy3a = aFG3->GetKStarCfHeavy(tAnalysisNumberA);
    CfHeavy* tCfHeavy3b = aFG3->GetKStarCfHeavy(tAnalysisNumberB);
    CfHeavy* tCfHeavy3 = CombineTwoHeavyCfs(tCfHeavy3a, tCfHeavy3b);
    TH1* tHist3 = tCfHeavy3->GetHeavyCfClone();



    tCanPart->AddGraph(0, j, tHist1, "", tMarkerStyle1, tMarkerColor1, tMarkerSize, "ex0");
    tCanPart->AddGraph(0, j, tHist2, "", tMarkerStyle2, tMarkerColor2, tMarkerSize, "ex0same");
    tCanPart->AddGraph(0, j, tHist3, "", tMarkerStyle3, tMarkerColor3, tMarkerSize, "ex0same");

    TString tText1 = TString::Format("%s & %s", cAnalysisRootTags[aFG1->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumberA)->GetAnalysisType()], 
                                                cAnalysisRootTags[aFG1->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumberB)->GetAnalysisType()]);

    TString tText2 = TString::Format("%s & %s", cAnalysisRootTags[aFG2->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumberA)->GetAnalysisType()], 
                                                cAnalysisRootTags[aFG2->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumberB)->GetAnalysisType()]);

    TString tText3 = TString::Format("%s & %s", cAnalysisRootTags[aFG3->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumberA)->GetAnalysisType()], 
                                                cAnalysisRootTags[aFG3->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumberB)->GetAnalysisType()]);


    tCanPart->SetupTLegend("", 0, j, 0.75, 0.15, 0.20, 0.35);
    tCanPart->AddLegendEntry(0, j, tHist1, tText1.Data(), "p");
    tCanPart->AddLegendEntry(0, j, tHist2, tText2.Data(), "p");
    tCanPart->AddLegendEntry(0, j, tHist3, tText3.Data(), "p");


    assert(aFG1->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumberA)->GetCentralityType() == aFG1->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumberB)->GetCentralityType());
    CentralityType aCentType = aFG1->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumberA)->GetCentralityType();
    TPaveText* tText = tCanPart->SetupTPaveText(cPrettyCentralityTags[aCentType], 0, j, 0.75, 0.80, 0.15, 0.10, 63, 15);
    tCanPart->AddPadPaveText(tText, 0, j);
  }


  //------------------------------------------------------------------------------------------------------
  for(int j=0; j<tNy; j++)
  {

    tAnalysisNumberA = 2*j;
    tAnalysisNumberB = 2*j + 1;

    CfHeavy* tCfHeavy1a = aFG1->GetKStarCfHeavy(tAnalysisNumberA);
    CfHeavy* tCfHeavy1b = aFG1->GetKStarCfHeavy(tAnalysisNumberB);
    CfHeavy* tCfHeavy1 = CombineTwoHeavyCfs(tCfHeavy1a, tCfHeavy1b);

    CfHeavy* tCfHeavy2a = aFG2->GetKStarCfHeavy(tAnalysisNumberA);
    CfHeavy* tCfHeavy2b = aFG2->GetKStarCfHeavy(tAnalysisNumberB);
    CfHeavy* tCfHeavy2 = CombineTwoHeavyCfs(tCfHeavy2a, tCfHeavy2b);

    CfHeavy* tCfHeavyAvg = CombineTwoHeavyCfs(tCfHeavy1, tCfHeavy2);
    TH1* tHistAvg = tCfHeavyAvg->GetHeavyCfClone();


    CfHeavy* tCfHeavy3a = aFG3->GetKStarCfHeavy(tAnalysisNumberA);
    CfHeavy* tCfHeavy3b = aFG3->GetKStarCfHeavy(tAnalysisNumberB);
    CfHeavy* tCfHeavy3 = CombineTwoHeavyCfs(tCfHeavy3a, tCfHeavy3b);
    TH1* tHist3 = tCfHeavy3->GetHeavyCfClone();

    tCanPart->AddGraph(1, j, tHistAvg, "", tMarkerStyleAvg, tMarkerColorAvg, tMarkerSize, "ex0");
    tCanPart->AddGraph(1, j, tHist3, "", tMarkerStyle3, tMarkerColor3, tMarkerSize, "ex0same");

    TString tText1 = TString::Format("%s & %s", cAnalysisRootTags[aFG1->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumberA)->GetAnalysisType()], 
                                                cAnalysisRootTags[aFG1->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumberB)->GetAnalysisType()]);

    TString tText2 = TString::Format("%s & %s", cAnalysisRootTags[aFG2->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumberA)->GetAnalysisType()], 
                                                cAnalysisRootTags[aFG2->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumberB)->GetAnalysisType()]);

    TString tTextAvg = TString::Format("(%s) + (%s)", tText1.Data(), tText2.Data());

    TString tText3 = TString::Format("%s & %s", cAnalysisRootTags[aFG3->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumberA)->GetAnalysisType()], 
                                                cAnalysisRootTags[aFG3->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumberB)->GetAnalysisType()]);


    tCanPart->SetupTLegend("", 1, j, 0.40, 0.15, 0.55, 0.35);
    tCanPart->AddLegendEntry(1, j, tHistAvg, tTextAvg.Data(), "p");
    tCanPart->AddLegendEntry(1, j, tHist3, tText3.Data(), "p");


    assert(aFG1->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumberA)->GetCentralityType() == aFG1->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumberB)->GetCentralityType());
    CentralityType aCentType = aFG1->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumberA)->GetCentralityType();
    TPaveText* tText = tCanPart->SetupTPaveText(cPrettyCentralityTags[aCentType], 1, j, 0.75, 0.80, 0.15, 0.10, 63, 15);
    tCanPart->AddPadPaveText(tText, 1, j);

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

  bool bDrawAllKStarCfs = false;
  bool bDrawAllKStarCfs_CombineConj = false;

  bool bDrawAvgLamKchvsK0 = false;
  bool bDrawAvgLamKchvsK0_CombineConj = false;

  bool bDrawAllKStarCfsAndAvgLamKchvsK0_CombineConj = true;

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
  if(bDrawAllKStarCfs_CombineConj)
  {
    TCanvas* tCanAll_CombineConj = DrawAllKStarCfs_CombineConj(tLamKchP1, tLamKchP2, tLamKchP3, bZoom, tCanNameModifier);
    if(SaveImages) tCanAll_CombineConj->SaveAs(TString::Format("%s%s.pdf", tSaveDir.Data(), tCanAll_CombineConj->GetName()));
  }

  if(bDrawAvgLamKchvsK0)
  {
    TCanvas* tCanAvgLamKchvsK0 = DrawAvgLamKchvsK0(tLamKchP1, tLamKchP2, tLamKchP3, bZoom, tCanNameModifier);
    if(SaveImages) tCanAvgLamKchvsK0->SaveAs(TString::Format("%s%s.pdf", tSaveDir.Data(), tCanAvgLamKchvsK0->GetName()));
  }
  if(bDrawAvgLamKchvsK0_CombineConj)
  {
    TCanvas* tCanDrawAvgLamKchvsK0_CombineConj = DrawAvgLamKchvsK0_CombineConj(tLamKchP1, tLamKchP2, tLamKchP3, bZoom, tCanNameModifier);
    if(SaveImages) tCanDrawAvgLamKchvsK0_CombineConj->SaveAs(TString::Format("%s%s.pdf", tSaveDir.Data(), tCanDrawAvgLamKchvsK0_CombineConj->GetName()));
  }


  if(bDrawAllKStarCfsAndAvgLamKchvsK0_CombineConj)
  {
    TCanvas* tCanDrawAllKStarCfsAndAvgLamKchvsK0_CombineConj = DrawAllKStarCfsAndAvgLamKchvsK0_CombineConj(tLamKchP1, tLamKchP2, tLamKchP3, bZoom, tCanNameModifier);
    if(SaveImages) tCanDrawAllKStarCfsAndAvgLamKchvsK0_CombineConj->SaveAs(TString::Format("%s%s.pdf", tSaveDir.Data(), tCanDrawAllKStarCfsAndAvgLamKchvsK0_CombineConj->GetName()));
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
