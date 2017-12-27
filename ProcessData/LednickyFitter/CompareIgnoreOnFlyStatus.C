#include "FitGenerator.h"
class FitGenerator;

#include "CanvasPartition.h"
class CanvasPartition;


//________________________________________________________________________________________________________________
TCanvas* DrawKStarCfs(vector<vector<TH1*> > &aHistos, AnalysisType aAnType, AnalysisType aConjType)
{
  TString tCanvasName = TString("canKStarCfs");
  tCanvasName += TString(cAnalysisBaseTags[aAnType]) + TString("wConj");

  assert(aHistos.size()==6);
  int tNx=2, tNy=3;


  double tXLow = -0.02;
  double tXHigh = 0.99;
  double tYLow = 0.71;
  double tYHigh = 1.09;
  CanvasPartition* tCanPart = new CanvasPartition(tCanvasName,tNx,tNy,tXLow,tXHigh,tYLow,tYHigh,0.12,0.05,0.13,0.05);

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

      tCanPart->AddGraph(i,j,aHistos[tAnalysisNumber][0],"",tMarkerStyle1,tMarkerColor1,tMarkerSize);
      tCanPart->AddGraph(i,j,aHistos[tAnalysisNumber][1],"",tMarkerStyle2,tMarkerColor2,tMarkerSize);

      TString tTextAnType;
      if(tAnalysisNumber==0 || tAnalysisNumber==2 || tAnalysisNumber==4) tTextAnType = TString(cAnalysisRootTags[aAnType]);
      else if(tAnalysisNumber==1 || tAnalysisNumber==3 || tAnalysisNumber==5) tTextAnType = TString(cAnalysisRootTags[aConjType]);
      else assert(0);

      TPaveText* tAnTypeName = tCanPart->SetupTPaveText(tTextAnType,i,j,0.8,0.85);
      tCanPart->AddPadPaveText(tAnTypeName,i,j);

      CentralityType tCentType;
      if(tAnalysisNumber==0 || tAnalysisNumber==1) tCentType = k0010;
      else if(tAnalysisNumber==2 || tAnalysisNumber==3) tCentType = k0010;
      else if(tAnalysisNumber==4 || tAnalysisNumber==5) tCentType = k0010;
      else assert(0);
      TString tTextCentrality = TString(cPrettyCentralityTags[tCentType]);
      TPaveText* tCentralityName = tCanPart->SetupTPaveText(tTextCentrality,i,j,0.05,0.85);
      tCanPart->AddPadPaveText(tCentralityName,i,j);
    }
  }

  tCanPart->SetDrawUnityLine(true);
  tCanPart->DrawAll();
  tCanPart->DrawXaxisTitle("k* (GeV/c)");
  tCanPart->DrawYaxisTitle("C(k*)",43,25,0.05,0.75);

  return tCanPart->GetCanvas();
}

//________________________________________________________________________________________________________________
TCanvas* DrawRatios(vector<vector<TH1*> > &aHistos, AnalysisType aAnType, AnalysisType aConjType)
{
  TString tCanvasName = TString("canKStarCfsRatios");
  tCanvasName += TString(cAnalysisBaseTags[aAnType]) + TString("wConj");

  assert(aHistos.size()==6);
  int tNx=2, tNy=3;


  double tXLow = -0.02;
  double tXHigh = 0.32;
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
      tRatio = (TH1*)aHistos[tAnalysisNumber][0]->Clone(tRatioName);
      tRatio->Divide((TH1*)aHistos[tAnalysisNumber][1]);
      tCanPart->AddGraph(i,j,(TH1*)tRatio->Clone(tRatioName),"",tMarkerStyle1,tMarkerColor1,tMarkerSize);


      TString tTextAnType;
      if(tAnalysisNumber==0 || tAnalysisNumber==2 || tAnalysisNumber==4) tTextAnType = TString(cAnalysisRootTags[aAnType]);
      else if(tAnalysisNumber==1 || tAnalysisNumber==3 || tAnalysisNumber==5) tTextAnType = TString(cAnalysisRootTags[aConjType]);
      else assert(0);

      TPaveText* tAnTypeName = tCanPart->SetupTPaveText(tTextAnType,i,j,0.8,0.85);
      tCanPart->AddPadPaveText(tAnTypeName,i,j);

      CentralityType tCentType;
      if(tAnalysisNumber==0 || tAnalysisNumber==1) tCentType = k0010;
      else if(tAnalysisNumber==2 || tAnalysisNumber==3) tCentType = k0010;
      else if(tAnalysisNumber==4 || tAnalysisNumber==5) tCentType = k0010;
      else assert(0);
      TString tTextCentrality = TString(cPrettyCentralityTags[tCentType]);
      TPaveText* tCentralityName = tCanPart->SetupTPaveText(tTextCentrality,i,j,0.05,0.85);
      tCanPart->AddPadPaveText(tCentralityName,i,j);
    }
  }

  tCanPart->SetDrawUnityLine(true);
  tCanPart->DrawAll();
  tCanPart->DrawXaxisTitle("k* (GeV/c)");
  tCanPart->DrawYaxisTitle("C(k*)",43,25,0.05,0.75);

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
  TString tResultsDate1 = "20161027";
//  TString tResultsDate2 = "20170505_ignoreOnFlyStatus";
  TString tResultsDate2 = "20171220_onFlyStatusFalse";

  AnalysisType tAnType = kLamKchP;
  AnalysisRunType tAnRunType = kTrain;
  int tNPartialAnalysis = 2;
  CentralityType tCentType = kMB;  //TODO
  FitGeneratorType tGenType = kPairwConj;

  AnalysisType tConjType;
  if(tAnType==kLamK0) {tConjType=kALamK0;}
  else if(tAnType==kLamKchP) {tConjType=kALamKchM;}
  else if(tAnType==kLamKchM) {tConjType=kALamKchP;}

  bool SaveImages = false;

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
  FitGenerator* tLamKchP1 = new FitGenerator(tFileLocationBase1,tFileLocationBaseMC1,tAnType, tCentType,tAnRunType,tNPartialAnalysis,tGenType);
  FitGenerator* tLamKchP2 = new FitGenerator(tFileLocationBase2,tFileLocationBaseMC2,tAnType, tCentType,tAnRunType,tNPartialAnalysis,tGenType);


/*
  TH1* tKStarCf1a = tLamKchP1->GetKStarCf(0);
    tKStarCf1a->SetMarkerStyle(20);
    tKStarCf1a->SetMarkerColor(2);
    tKStarCf1a->SetMarkerSize(1.);

  TH1* tKStarCf1b = tLamKchP1->GetKStarCf(1);
    tKStarCf1b->SetMarkerStyle(20);
    tKStarCf1b->SetMarkerColor(2);
    tKStarCf1b->SetMarkerSize(1.);

  TH1* tKStarCf2a = tLamKchP2->GetKStarCf(0);
    tKStarCf2a->SetMarkerStyle(25);
    tKStarCf2a->SetMarkerColor(2);
    tKStarCf2a->SetMarkerSize(1.);

  TH1* tKStarCf2b = tLamKchP2->GetKStarCf(1);
    tKStarCf2b->SetMarkerStyle(25);
    tKStarCf2b->SetMarkerColor(2);
    tKStarCf2b->SetMarkerSize(1.);


  TCanvas *tCan = new TCanvas("tCan", "tCan");
  tCan->Divide(2,1);
  gStyle->SetOptStat(0);

  tCan->cd(1);
  tKStarCf1a->Draw();
  tKStarCf2a->Draw("same");

  tCan->cd(2);
  tKStarCf1b->Draw();
  tKStarCf2b->Draw("same");
*/
  //---------------------------------------------------
  TH1* tCfPair0010_1 = tLamKchP1->GetKStarCf(0);
  TH1* tCfConj0010_1 = tLamKchP1->GetKStarCf(1);

  TH1* tCfPair1030_1 = tLamKchP1->GetKStarCf(2);
  TH1* tCfConj1030_1 = tLamKchP1->GetKStarCf(3);

  TH1* tCfPair3050_1 = tLamKchP1->GetKStarCf(4);
  TH1* tCfConj3050_1 = tLamKchP1->GetKStarCf(5);
  //---------------------------------------------------
  TH1* tCfPair0010_2 = tLamKchP2->GetKStarCf(0);
  TH1* tCfConj0010_2 = tLamKchP2->GetKStarCf(1);

  TH1* tCfPair1030_2 = tLamKchP2->GetKStarCf(2);
  TH1* tCfConj1030_2 = tLamKchP2->GetKStarCf(3);

  TH1* tCfPair3050_2 = tLamKchP2->GetKStarCf(4);
  TH1* tCfConj3050_2 = tLamKchP2->GetKStarCf(5);


  vector<vector<TH1*> > tHistos {{tCfPair0010_1,tCfPair0010_2}, {tCfConj0010_1,tCfConj0010_2}, {tCfPair1030_1,tCfPair1030_2}, {tCfConj1030_1,tCfConj1030_2}, {tCfPair3050_1,tCfPair3050_2}, {tCfConj3050_1,tCfConj3050_2} };

  TCanvas* tCan = DrawKStarCfs(tHistos, tAnType, tConjType);
  TCanvas* tRatioCan = DrawRatios(tHistos, tAnType, tConjType);


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
