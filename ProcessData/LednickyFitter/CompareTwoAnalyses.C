/* CompareTwoAnalyses.C */
/* Originally CompareIgnoreOnFlyStatus.C, used to compare different settings of
   ignoreOnFlyStatus of V0s in analyses */

#include "FitGenerator.h"
class FitGenerator;

#include "CanvasPartition.h"
class CanvasPartition;


//________________________________________________________________________________________________________________
TCanvas* DrawKStarCfs(FitGenerator* aFG1, FitGenerator* aFG2, bool aZoom=false, TString aCanNameModifier="")
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

  double tYLow = 0.71;
  double tYHigh = 1.09;
  CanvasPartition* tCanPart = new CanvasPartition(tCanvasName,tNx,tNy,tXLow,tXHigh,tYLow,tYHigh,0.12,0.05,0.13,0.05);
  if(!tConjIncluded) tCanPart->GetCanvas()->SetCanvasSize(350,500);

  int tAnalysisNumber=0;

  int tMarkerStyle1 = 20;
  int tMarkerStyle2 = 25;

  int tMarkerColor1 = 1;
  int tMarkerColor2 = 1;


  double tMarkerSize = 0.5;

  if(aAnType==kLamK0 || aAnType==kALamK0) {tMarkerColor1 = kBlack; tMarkerColor2 = kGray+2;}
  else if(aAnType==kLamKchP || aAnType==kALamKchM) {tMarkerColor1 = kRed; tMarkerColor2 = kRed+2;}
  else if(aAnType==kLamKchM || aAnType==kALamKchP) {tMarkerColor1 = kBlue; tMarkerColor2 = kBlue+2;}
  else {tMarkerColor1=1; tMarkerColor2=1;}

  for(int j=0; j<tNy; j++)
  {
    for(int i=0; i<tNx; i++)
    {
      tAnalysisNumber = j*tNx + i;
      //---------------------------------------------------------------------------------------------------------

      tCanPart->AddGraph(i,j,(TH1*)aFG1->GetKStarCf(tAnalysisNumber),"",tMarkerStyle1,tMarkerColor1,tMarkerSize);
      tCanPart->AddGraph(i,j,(TH1*)aFG2->GetKStarCf(tAnalysisNumber),"",tMarkerStyle2,tMarkerColor2,tMarkerSize);

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
    }
  }

  tCanPart->SetDrawUnityLine(true);
  tCanPart->DrawAll();
  tCanPart->DrawXaxisTitle("k* (GeV/c)");
  if(tConjIncluded) tCanPart->DrawYaxisTitle("C(k*)",43,25,0.05,0.75);
  else tCanPart->DrawYaxisTitle("C(k*)",43,25,0.075,0.875);

  return tCanPart->GetCanvas();
}



//________________________________________________________________________________________________________________
TCanvas* DrawKStarCfRatios(FitGenerator* aFG1, FitGenerator* aFG2, bool aZoom=false)
{
  AnalysisType aAnType = aFG1->GetFitSharedAnalyses()->GetFitPairAnalysis(0)->GetAnalysisType();
  AnalysisType aConjType = aFG1->GetFitSharedAnalyses()->GetFitPairAnalysis(1)->GetAnalysisType();

  TString tCanvasName = TString("canKStarCfsRatios");
  if(aZoom) tCanvasName += TString("Zoom");
  tCanvasName += TString(cAnalysisBaseTags[aAnType]) + TString("wConj");

  int tNx=2, tNy=3;

  double tXLow = -0.02;
//  double tXHigh = 0.32;
  double tXHigh = aFG1->GetKStarCf(0)->GetXaxis()->GetBinUpEdge(aFG1->GetKStarCf(0)->GetNbinsX());
  if(aZoom) tXHigh = 0.32;

  double tYLow = 0.88;
  double tYHigh = 1.05;
  CanvasPartition* tCanPart = new CanvasPartition(tCanvasName,tNx,tNy,tXLow,tXHigh,tYLow,tYHigh,0.12,0.05,0.13,0.05);

  int tAnalysisNumber=0;

  int tMarkerStyle1 = 20;
  int tMarkerColor1 = 1;
  double tMarkerSize = 0.5;

  if(aAnType==kLamK0 || aAnType==kALamK0) tMarkerColor1 = kBlack;
  else if(aAnType==kLamKchP || aAnType==kALamKchM) tMarkerColor1 = kRed;
  else if(aAnType==kLamKchM || aAnType==kALamKchP) tMarkerColor1 = kBlue;
  else tMarkerColor1=1;

  TH1* tRatio;
  TString tRatioName;
  for(int j=0; j<tNy; j++)
  {
    for(int i=0; i<tNx; i++)
    {
      tAnalysisNumber = j*tNx + i;
      //---------------------------------------------------------------------------------------------------------
      tRatioName = TString::Format("KStarCfRatios%s%d", cAnalysisBaseTags[aAnType], tAnalysisNumber);
      tRatio = (TH1*)aFG1->GetKStarCf(tAnalysisNumber)->Clone(tRatioName);
      tRatio->Divide((TH1*)aFG2->GetKStarCf(tAnalysisNumber));
      tCanPart->AddGraph(i,j,(TH1*)tRatio->Clone(tRatioName),"",tMarkerStyle1,tMarkerColor1,tMarkerSize);


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
    }
  }

  tCanPart->SetDrawUnityLine(true);
  tCanPart->DrawAll();
  tCanPart->DrawXaxisTitle("k* (GeV/c)");
  tCanPart->DrawYaxisTitle("C_{1}(k*)/C_{2}(k*)",43,25,0.05,0.75);

  return tCanPart->GetCanvas();
}

//________________________________________________________________________________________________________________
TCanvas* DrawNumDenRatiosPartAn(bool aDrawNum, FitGenerator* aFG1, FitGenerator* aFG2)
{
  AnalysisType aAnType = aFG1->GetFitSharedAnalyses()->GetFitPairAnalysis(0)->GetAnalysisType();
  AnalysisType aConjType = aFG1->GetFitSharedAnalyses()->GetFitPairAnalysis(1)->GetAnalysisType();

  TString tCanvasName;
  if(aDrawNum) tCanvasName = TString("canNumRatiosPartAn");
  else tCanvasName = TString("canDenRatiosPartAn");
  tCanvasName += TString(cAnalysisBaseTags[aAnType]) + TString("wConj");

  int tNx=2, tNy=6;

  double tXLow = -0.02;
  double tXHigh = 0.98;
  double tYLow = 0.52;
  double tYHigh = 1.02;
  CanvasPartition* tCanPart = new CanvasPartition(tCanvasName,tNx,tNy,tXLow,tXHigh,tYLow,tYHigh,0.12,0.05,0.13,0.05);
  tCanPart->GetCanvas()->SetCanvasSize(700,1500);

  int tMarkerStyle1 = 20;
  int tMarkerColor1 = 1;
  double tMarkerSize = 0.5;

  if(aAnType==kLamK0 || aAnType==kALamK0) tMarkerColor1 = kBlack;
  else if(aAnType==kLamKchP || aAnType==kALamKchM) tMarkerColor1 = kRed;
  else if(aAnType==kLamKchM || aAnType==kALamKchP) tMarkerColor1 = kBlue;
  else tMarkerColor1=1;

  TH1* tRatio;
  TString tRatioName;
  int tAnalysisNumber=0;
  int tPartialAnNumber = 0;
  FitPartialAnalysis *tPartAn1, *tPartAn2;
  for(int j=0; j<tNy; j++)
  {
    for(int i=0; i<tNx; i++)
    {
      tPartialAnNumber = i;
      //---------------------------------------------------------------------------------------------------------
      tPartAn1 = aFG1->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumber)->GetFitPartialAnalysis(tPartialAnNumber);
      tPartAn2 = aFG2->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumber)->GetFitPartialAnalysis(tPartialAnNumber);
      //---------------------------------------------------------------------------------------------------------
      if(aDrawNum) 
      {
        tRatioName = TString::Format("NumRatios%s_%d_%d", cAnalysisBaseTags[aAnType], tAnalysisNumber, tPartialAnNumber);
        tRatio = (TH1*)tPartAn1->GetKStarCfLite()->Num()->Clone(tRatioName);
        tRatio->Divide((TH1*)tPartAn2->GetKStarCfLite()->Num());
      }
      else 
      {
        tRatioName = TString::Format("DenRatios%s_%d_%d", cAnalysisBaseTags[aAnType], tAnalysisNumber, tPartialAnNumber);
        tRatio = (TH1*)tPartAn1->GetKStarCfLite()->Den()->Clone(tRatioName);
        tRatio->Divide((TH1*)tPartAn2->GetKStarCfLite()->Den());
      }
      tCanPart->AddGraph(i,j,(TH1*)tRatio->Clone(tRatioName),"",tMarkerStyle1,tMarkerColor1,tMarkerSize);

      assert(tPartAn1->GetAnalysisType() == tPartAn2->GetAnalysisType());
      assert(tPartAn1->GetBFieldType() == tPartAn2->GetBFieldType());
      assert(tPartAn1->GetCentralityType() == tPartAn2->GetCentralityType());

      TString tTextAnType = TString::Format("%s (%s)", cAnalysisRootTags[tPartAn1->GetAnalysisType()], cBFieldTags[tPartAn1->GetBFieldType()]);

      TPaveText* tAnTypeName = tCanPart->SetupTPaveText(tTextAnType,i,j,0.6,0.85);
      tCanPart->AddPadPaveText(tAnTypeName,i,j);

      TString tTextCentrality = TString(cPrettyCentralityTags[tPartAn1->GetCentralityType()]);
      TPaveText* tCentralityName = tCanPart->SetupTPaveText(tTextCentrality,i,j,0.05,0.85);
      tCanPart->AddPadPaveText(tCentralityName,i,j);

      tAnalysisNumber += i;
    }
  }

  tCanPart->SetDrawUnityLine(true);
  tCanPart->DrawAll();
  tCanPart->DrawXaxisTitle("k* (GeV/c)");
  if(aDrawNum) tCanPart->DrawYaxisTitle("Num1/Num2",43,25,0.05,0.75);
  else tCanPart->DrawYaxisTitle("Den1/Den2",43,25,0.05,0.75);

  return tCanPart->GetCanvas();
}

//________________________________________________________________________________________________________________
TCanvas* DrawNumDenRatiosAn(bool aDrawNum, FitGenerator* aFG1, FitGenerator* aFG2, bool aNormalize=false)
{
  AnalysisType aAnType = aFG1->GetFitSharedAnalyses()->GetFitPairAnalysis(0)->GetAnalysisType();
  AnalysisType aConjType = aFG1->GetFitSharedAnalyses()->GetFitPairAnalysis(1)->GetAnalysisType();

  TString tCanvasName;
  if(aDrawNum) tCanvasName = TString("canNumRatios");
  else tCanvasName = TString("canDenRatios");
  tCanvasName += TString(cAnalysisBaseTags[aAnType]) + TString("wConj");

  int tNx=2, tNy=3;

  double tXLow = -0.02;
  double tXHigh = 0.98;
  double tYLow = 0.52;
  double tYHigh = 1.02;
  CanvasPartition* tCanPart = new CanvasPartition(tCanvasName,tNx,tNy,tXLow,tXHigh,tYLow,tYHigh,0.12,0.05,0.13,0.05);
//  tCanPart->GetCanvas()->SetCanvasSize(700,1500);

  int tMarkerStyle1 = 20;
  int tMarkerColor1 = 1;
  double tMarkerSize = 0.5;

  if(aAnType==kLamK0 || aAnType==kALamK0) tMarkerColor1 = kBlack;
  else if(aAnType==kLamKchP || aAnType==kALamKchM) tMarkerColor1 = kRed;
  else if(aAnType==kLamKchM || aAnType==kALamKchP) tMarkerColor1 = kBlue;
  else tMarkerColor1=1;

  TH1* tRatio;
  TString tRatioName;
  int tAnalysisNumber=0;
  FitPairAnalysis *tAn1, *tAn2;
  for(int j=0; j<tNy; j++)
  {
    for(int i=0; i<tNx; i++)
    {
      tAnalysisNumber = j*tNx + i;
      //---------------------------------------------------------------------------------------------------------
      tAn1 = aFG1->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumber);
      tAn2 = aFG2->GetFitSharedAnalyses()->GetFitPairAnalysis(tAnalysisNumber);
      //---------------------------------------------------------------------------------------------------------
      if(aDrawNum) tRatioName = TString::Format("NumRatios%s_%d", cAnalysisBaseTags[aAnType], tAnalysisNumber);
      else tRatioName = TString::Format("DenRatios%s_%d", cAnalysisBaseTags[aAnType], tAnalysisNumber);

      tRatio = (TH1*)tAn1->GetKStarCfHeavy()->GetSimplyAddedNumDen(tRatioName, aDrawNum);
      tRatio->Divide((TH1*)tAn2->GetKStarCfHeavy()->GetSimplyAddedNumDen(tRatioName, aDrawNum));
      if(aNormalize) tRatio->Scale(((TH1*)tAn2->GetKStarCfHeavy()->GetSimplyAddedNumDen(tRatioName, aDrawNum))->Integral()/((TH1*)tAn1->GetKStarCfHeavy()->GetSimplyAddedNumDen(tRatioName, aDrawNum))->Integral());
      tCanPart->AddGraph(i,j,(TH1*)tRatio->Clone(tRatioName),"",tMarkerStyle1,tMarkerColor1,tMarkerSize);

      assert(tAn1->GetAnalysisType() == tAn2->GetAnalysisType());
      assert(tAn1->GetCentralityType() == tAn2->GetCentralityType());

      TString tTextAnType = TString::Format("%s", cAnalysisRootTags[tAn1->GetAnalysisType()]);

      TPaveText* tAnTypeName = tCanPart->SetupTPaveText(tTextAnType,i,j,0.6,0.85);
      tCanPart->AddPadPaveText(tAnTypeName,i,j);

      TString tTextCentrality = TString(cPrettyCentralityTags[tAn1->GetCentralityType()]);
      TPaveText* tCentralityName = tCanPart->SetupTPaveText(tTextCentrality,i,j,0.05,0.85);
      tCanPart->AddPadPaveText(tCentralityName,i,j);

    }
  }

  tCanPart->SetDrawUnityLine(true);
  tCanPart->DrawAll();
  tCanPart->DrawXaxisTitle("k* (GeV/c)");
  if(aDrawNum) tCanPart->DrawYaxisTitle("Num1/Num2",43,25,0.05,0.75);
  else tCanPart->DrawYaxisTitle("Den1/Den2",43,25,0.05,0.75);

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
  TString tResultsDate1, tResultsDate2;

//  tResultsDate1 = "20161027";
//  tResultsDate2 = "20171227";

  tResultsDate1 = "20180416";
  tResultsDate2 = "20180416";

//  tResultsDate1 = "20170505_ignoreOnFlyStatus";
//  tResultsDate2 = "20171220_onFlyStatusFalse";

//  tResultsDate1 = "20171227";
//  tResultsDate2 = "20171227_LHC10h";

  AnalysisType tAnType = kLamKchP;
  AnalysisRunType tAnRunType = kTrain;
  int tNPartialAnalysis = 2;
  CentralityType tCentType = kMB;  //TODO
  FitGeneratorType tGenType = kPairwConj;

  AnalysisType tConjType;
  if(tAnType==kLamK0) {tConjType=kALamK0;}
  else if(tAnType==kLamKchP) {tConjType=kALamKchM;}
  else if(tAnType==kLamKchM) {tConjType=kALamKchP;}

  bool bUseNumRotPar2InsteadOfDen1 = false;
  bool bUseNumRotPar2InsteadOfDen2 = true;

  bool SaveImages = false;
  TString tSaveDir = "/home/jesse/Analysis/Presentations/GroupMeetings/20180607/Figures/";

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

  TString tSaveNameModifier = "";
  FitGenerator* tLamKchP1 = new FitGenerator(tFileLocationBase1, tFileLocationBaseMC1, tAnType, tCentType, tAnRunType, tNPartialAnalysis, tGenType, false, false, "", bUseNumRotPar2InsteadOfDen1);
  FitGenerator* tLamKchP2 = new FitGenerator(tFileLocationBase2, tFileLocationBaseMC2, tAnType, tCentType, tAnRunType, tNPartialAnalysis, tGenType, false, false, "", bUseNumRotPar2InsteadOfDen2);
  //-----------------------------------------------------------------------------
  bool bZoom = false;
  bool bDrawKStarCfs = true;
  bool bDrawKStarCfRatios = false;

  bool bDrawNumDenRatiosPartAn = false;
  bool bDrawNumDenRatiosAn = false;
  //-----------------------------------------------------------------------------
  vector<TString> tUseNumRot2Tags = {"", "NumRotPar2"};
  TString tCanNameModifier = TString::Format("_%s%svs%s%s", tResultsDate1.Data(), tUseNumRot2Tags[bUseNumRotPar2InsteadOfDen1].Data(), 
                                                            tResultsDate2.Data(), tUseNumRot2Tags[bUseNumRotPar2InsteadOfDen2].Data());

  //-----------------------------------------------------------------------------
  if(bDrawKStarCfs)
  {
    TCanvas* tCan = DrawKStarCfs(tLamKchP1, tLamKchP2, bZoom, tCanNameModifier);
    if(SaveImages) tCan->SaveAs(TString::Format("%s%s.eps", tSaveDir.Data(), tCan->GetName()));
  }

  if(bDrawKStarCfRatios)
  {
    TCanvas* tRatioCan = DrawKStarCfRatios(tLamKchP1, tLamKchP2, bZoom);
    if(SaveImages) tRatioCan->SaveAs(TString::Format("%s%s.eps", tSaveDir.Data(), tRatioCan->GetName()));
  }

  if(bDrawNumDenRatiosPartAn)
  {
    TCanvas* tRatioNumPartAn = DrawNumDenRatiosPartAn(true, tLamKchP1, tLamKchP2);
    TCanvas* tRatioDenPartAn = DrawNumDenRatiosPartAn(false, tLamKchP1, tLamKchP2);
    if(SaveImages)
    {
      tRatioNumPartAn->SaveAs(TString::Format("%s%s.eps", tSaveDir.Data(), tRatioNumPartAn->GetName()));
      tRatioDenPartAn->SaveAs(TString::Format("%s%s.eps", tSaveDir.Data(), tRatioDenPartAn->GetName()));
    }
  }

  if(bDrawNumDenRatiosAn)
  {
    TCanvas* tRatioNumAn = DrawNumDenRatiosAn(true, tLamKchP1, tLamKchP2);
    TCanvas* tRatioDenAn = DrawNumDenRatiosAn(false, tLamKchP1, tLamKchP2);
    if(SaveImages)
    {
      tRatioNumAn->SaveAs(TString::Format("%s%s.eps", tSaveDir.Data(), tRatioNumAn->GetName()));
      tRatioDenAn->SaveAs(TString::Format("%s%s.eps", tSaveDir.Data(), tRatioDenAn->GetName()));
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
