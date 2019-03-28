#include <iostream>
#include <iomanip>
#include <math.h>
#include <vector>
#include <assert.h>

#include "TF1.h"
#include "TH1F.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TApplication.h"
#include "TPaveText.h"
#include "TStyle.h"
#include "TAxis.h"
#include "TGraph.h"
#include "TMath.h"
#include "TList.h"

#include "TApplication.h"

using std::cout;
using std::endl;
using std::vector;

#include "PartialAnalysis.h"
class PartialAnalysis;

#include "Analysis.h"
class Analysis;

#include "CanvasPartition.h"

int main(int argc, char **argv) 
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything
//-----------------------------------------------------------------------------


  //TODO TODO TODO TODO TODO pdfviewer and LaTex are being super annoying, and not displaying Delta correctly.  Workaround is to save file as
  //TODO TODO TODO TODO TODO .eps and convert with epstopdf to pdf
  bool bSave = false;
  TString tSaveFileType = "pdf";
  TString tSaveDirectoryBase = "/home/jesse/Analysis/FemtoAnalysis/AnalysisNotes/3_DataSelection/Figures/";

  CentralityType tCentType = k0010;

  TString FileLocationBaseLamKch = "~/Analysis/FemtoAnalysis/Results/Results_cLamcKch_20180423_NoAvgSepCut/Results_cLamcKch_20180423_NoAvgSepCut";
  Analysis* LamKchP = new Analysis(FileLocationBaseLamKch, kLamKchP, tCentType);
  Analysis* ALamKchM = new Analysis(FileLocationBaseLamKch, kALamKchM, tCentType);
  Analysis* LamKchM = new Analysis(FileLocationBaseLamKch, kLamKchM, tCentType);
  Analysis* ALamKchP = new Analysis(FileLocationBaseLamKch, kALamKchP, tCentType);

  TString FileLocationBaseLamK0 = "~/Analysis/FemtoAnalysis/Results/Results_cLamK0_20180423_NoAvgSepCut/Results_cLamK0_20180423_NoAvgSepCut";
  Analysis* LamK0 = new Analysis(FileLocationBaseLamK0, kLamK0, tCentType);
  Analysis* ALamK0 = new Analysis(FileLocationBaseLamK0, kALamK0, tCentType);

//-----------------------------------------------------------------------------

  double aMinNorm = 14.99;
  double aMaxNorm = 19.99;
  int aRebin = 2;

  LamKchP->BuildAllAvgSepHeavyCfs(aMinNorm, aMaxNorm, aRebin);
  ALamKchM->BuildAllAvgSepHeavyCfs(aMinNorm, aMaxNorm, aRebin);
  LamKchM->BuildAllAvgSepHeavyCfs(aMinNorm, aMaxNorm, aRebin);
  ALamKchP->BuildAllAvgSepHeavyCfs(aMinNorm, aMaxNorm, aRebin);

  LamK0->BuildAllAvgSepHeavyCfs(aMinNorm, aMaxNorm, aRebin);
  ALamK0->BuildAllAvgSepHeavyCfs(aMinNorm, aMaxNorm, aRebin);

//-----------------------------------------------------------------------------
//*****************************************************************************
//---------------- LamKch -----------------------------------------------------
  int tNx=2;
  int tNy=4;
  double tXLow = -0.5;
  double tXHigh = 14.;
  double tYLow = -0.225;
  double tYHigh = 3.5;

  TString tCanvasName_LamKch = "LamKchAvgSepCfs";
  CanvasPartition* tCanPart_LamKch = new CanvasPartition(tCanvasName_LamKch,tNx,tNy,tXLow,tXHigh,tYLow,tYHigh,0.12,0.0025,0.125,0.0025);
  tCanPart_LamKch->SetDrawOptStat(false);
//  tCanPart_LamKch->GetCanvas()->SetCanvasSize(2100, 2000);

  TH1D* tDummy = new TH1D("tDummy", "tDummy", 1, tXLow, tXHigh);
    tDummy->GetXaxis()->SetRangeUser(tXLow, tXHigh);
    tDummy->GetYaxis()->SetRangeUser(tYLow, tYHigh);

  int tMarkerStyle = 20;
  int tMarkerColor = kBlack;
  double tMarkerSize = 0.5;

  //All default values (except XMin), so text matches (but doesn't overlap) what is added in AddGraph call
  double tTextXMin1 = 0.15;
  double tTextWidth1 = 0.15;

  double tTextXMin2 = 0.55;
  double tTextWidth2 = 0.35;

  double tTextFont1 = 63;
  double tTextFont2 = 43;

  double tTextYMin = 0.75;

  double tTextHeight = 0.20;
  double tTextSize=15;

  int aNx, aNy;

  //----------------------------------------------------------------------------------------
  TH1* tCfLamKchP_TrackPos = LamKchP->GetAvgSepHeavyCf(kTrackPos)->GetHeavyCfClone();
  TH1* tCfLamKchP_TrackNeg = LamKchP->GetAvgSepHeavyCf(kTrackNeg)->GetHeavyCfClone();


  aNx=0, aNy=0;
  tCanPart_LamKch->AddGraph(aNx, aNy, (TH1D*)tDummy->Clone(), "", tMarkerStyle, tMarkerColor, tMarkerSize, "AXIS");
  tCanPart_LamKch->AddGraph(aNx, aNy, tCfLamKchP_TrackPos, "", tMarkerStyle, tMarkerColor, tMarkerSize, "ex0same");
    tCanPart_LamKch->AddPadPaveText(tCanPart_LamKch->SetupTPaveText(cAnalysisRootTags[kLamKchP], aNx, aNy, tTextXMin1, tTextYMin, tTextWidth1, tTextHeight, tTextFont1, tTextSize), aNx, aNy);
    tCanPart_LamKch->AddPadPaveText(tCanPart_LamKch->SetupTPaveText(LamKchP->GetDaughtersHistoTitle(kTrackPos), aNx, aNy, tTextXMin2, tTextYMin, tTextWidth2, tTextHeight, tTextFont2, tTextSize), aNx, aNy);

  aNx=1, aNy=0;
  tCanPart_LamKch->AddGraph(aNx, aNy, (TH1D*)tDummy->Clone(), "", tMarkerStyle, tMarkerColor, tMarkerSize, "AXIS");
  tCanPart_LamKch->AddGraph(aNx, aNy, tCfLamKchP_TrackNeg, "", tMarkerStyle, tMarkerColor, tMarkerSize, "ex0same");
    tCanPart_LamKch->AddPadPaveText(tCanPart_LamKch->SetupTPaveText(cAnalysisRootTags[kLamKchP], aNx, aNy, tTextXMin1, tTextYMin, tTextWidth1, tTextHeight, tTextFont1, tTextSize), aNx, aNy);
    tCanPart_LamKch->AddPadPaveText(tCanPart_LamKch->SetupTPaveText(LamKchP->GetDaughtersHistoTitle(kTrackNeg), aNx, aNy, tTextXMin2, tTextYMin, tTextWidth2, tTextHeight, tTextFont2, tTextSize), aNx, aNy);





  //----------------------------------------------------------------------------------------
  TH1* tCfALamKchM_TrackPos = ALamKchM->GetAvgSepHeavyCf(kTrackPos)->GetHeavyCfClone();
  TH1* tCfALamKchM_TrackNeg = ALamKchM->GetAvgSepHeavyCf(kTrackNeg)->GetHeavyCfClone();


  aNx=0, aNy=1;
  tCanPart_LamKch->AddGraph(aNx, aNy, (TH1D*)tDummy->Clone(), "", tMarkerStyle, tMarkerColor, tMarkerSize, "AXIS");
  tCanPart_LamKch->AddGraph(aNx, aNy, tCfALamKchM_TrackPos, "", tMarkerStyle, tMarkerColor, tMarkerSize, "ex0same");
    tCanPart_LamKch->AddPadPaveText(tCanPart_LamKch->SetupTPaveText(cAnalysisRootTags[kALamKchM], aNx, aNy, tTextXMin1, tTextYMin, tTextWidth1, tTextHeight, tTextFont1, tTextSize), aNx, aNy);
    tCanPart_LamKch->AddPadPaveText(tCanPart_LamKch->SetupTPaveText(ALamKchM->GetDaughtersHistoTitle(kTrackPos), aNx, aNy, tTextXMin2, tTextYMin, tTextWidth2, tTextHeight, tTextFont2, tTextSize), aNx, aNy);

  aNx=1, aNy=1;
  tCanPart_LamKch->AddGraph(aNx, aNy, (TH1D*)tDummy->Clone(), "", tMarkerStyle, tMarkerColor, tMarkerSize, "AXIS");
  tCanPart_LamKch->AddGraph(aNx, aNy, tCfALamKchM_TrackNeg, "", tMarkerStyle, tMarkerColor, tMarkerSize, "ex0same");
    tCanPart_LamKch->AddPadPaveText(tCanPart_LamKch->SetupTPaveText(cAnalysisRootTags[kALamKchM], aNx, aNy, tTextXMin1, tTextYMin, tTextWidth1, tTextHeight, tTextFont1, tTextSize), aNx, aNy);
    tCanPart_LamKch->AddPadPaveText(tCanPart_LamKch->SetupTPaveText(ALamKchM->GetDaughtersHistoTitle(kTrackNeg), aNx, aNy, tTextXMin2, tTextYMin, tTextWidth2, tTextHeight, tTextFont2, tTextSize), aNx, aNy);







  //----------------------------------------------------------------------------------------
  TH1* tCfLamKchM_TrackPos = LamKchM->GetAvgSepHeavyCf(kTrackPos)->GetHeavyCfClone();
  TH1* tCfLamKchM_TrackNeg = LamKchM->GetAvgSepHeavyCf(kTrackNeg)->GetHeavyCfClone();


  aNx=0, aNy=2;
  tCanPart_LamKch->AddGraph(aNx, aNy, (TH1D*)tDummy->Clone(), "", tMarkerStyle, tMarkerColor, tMarkerSize, "AXIS");
  tCanPart_LamKch->AddGraph(aNx, aNy, tCfLamKchM_TrackPos, "", tMarkerStyle, tMarkerColor, tMarkerSize, "ex0same");
    tCanPart_LamKch->AddPadPaveText(tCanPart_LamKch->SetupTPaveText(cAnalysisRootTags[kLamKchM], aNx, aNy, tTextXMin1, tTextYMin, tTextWidth1, tTextHeight, tTextFont1, tTextSize), aNx, aNy);
    tCanPart_LamKch->AddPadPaveText(tCanPart_LamKch->SetupTPaveText(LamKchM->GetDaughtersHistoTitle(kTrackPos), aNx, aNy, tTextXMin2, tTextYMin, tTextWidth2, tTextHeight, tTextFont2, tTextSize), aNx, aNy);

  aNx=1, aNy=2;
  tCanPart_LamKch->AddGraph(aNx, aNy, (TH1D*)tDummy->Clone(), "", tMarkerStyle, tMarkerColor, tMarkerSize, "AXIS");
  tCanPart_LamKch->AddGraph(aNx, aNy, tCfLamKchM_TrackNeg, "", tMarkerStyle, tMarkerColor, tMarkerSize, "ex0same");
    tCanPart_LamKch->AddPadPaveText(tCanPart_LamKch->SetupTPaveText(cAnalysisRootTags[kLamKchM], aNx, aNy, tTextXMin1, tTextYMin, tTextWidth1, tTextHeight, tTextFont1, tTextSize), aNx, aNy);
    tCanPart_LamKch->AddPadPaveText(tCanPart_LamKch->SetupTPaveText(LamKchM->GetDaughtersHistoTitle(kTrackNeg), aNx, aNy, tTextXMin2, tTextYMin, tTextWidth2, tTextHeight, tTextFont2, tTextSize), aNx, aNy);







  //----------------------------------------------------------------------------------------
  TH1* tCfALamKchP_TrackPos = ALamKchP->GetAvgSepHeavyCf(kTrackPos)->GetHeavyCfClone();
  TH1* tCfALamKchP_TrackNeg = ALamKchP->GetAvgSepHeavyCf(kTrackNeg)->GetHeavyCfClone();


  aNx=0, aNy=3;
  tCanPart_LamKch->AddGraph(aNx, aNy, (TH1D*)tDummy->Clone(), "", tMarkerStyle, tMarkerColor, tMarkerSize, "AXIS");
  tCanPart_LamKch->AddGraph(aNx, aNy, tCfALamKchP_TrackPos, "", tMarkerStyle, tMarkerColor, tMarkerSize, "ex0same");
    tCanPart_LamKch->AddPadPaveText(tCanPart_LamKch->SetupTPaveText(cAnalysisRootTags[kALamKchP], aNx, aNy, tTextXMin1, tTextYMin, tTextWidth1, tTextHeight, tTextFont1, tTextSize), aNx, aNy);
    tCanPart_LamKch->AddPadPaveText(tCanPart_LamKch->SetupTPaveText(ALamKchP->GetDaughtersHistoTitle(kTrackPos), aNx, aNy, tTextXMin2, tTextYMin, tTextWidth2, tTextHeight, tTextFont2, tTextSize), aNx, aNy);

  aNx=1, aNy=3;
  tCanPart_LamKch->AddGraph(aNx, aNy, (TH1D*)tDummy->Clone(), "", tMarkerStyle, tMarkerColor, tMarkerSize, "AXIS");
  tCanPart_LamKch->AddGraph(aNx, aNy, tCfALamKchP_TrackNeg, "", tMarkerStyle, tMarkerColor, tMarkerSize, "ex0same");
    tCanPart_LamKch->AddPadPaveText(tCanPart_LamKch->SetupTPaveText(cAnalysisRootTags[kALamKchP], aNx, aNy, tTextXMin1, tTextYMin, tTextWidth1, tTextHeight, tTextFont1, tTextSize), aNx, aNy);
    tCanPart_LamKch->AddPadPaveText(tCanPart_LamKch->SetupTPaveText(ALamKchP->GetDaughtersHistoTitle(kTrackNeg), aNx, aNy, tTextXMin2, tTextYMin, tTextWidth2, tTextHeight, tTextFont2, tTextSize), aNx, aNy);






  //----------------------------------
  tCanPart_LamKch->SetDrawUnityLine(true);
  tCanPart_LamKch->DrawAll();
  tCanPart_LamKch->DrawXaxisTitle("Avg. Sep. #bar{#bf{#Deltar}} (cm)", 43, 25, 0.70, 0.015); //Note, changing xaxis low (=0.315) does nothing
  tCanPart_LamKch->DrawYaxisTitle("#it{C}(#bar{#bf{#Deltar}})", 43, 35, 0.06, 0.80);

  if(bSave) tCanPart_LamKch->GetCanvas()->SaveAs(TString::Format("%s%s.%s", tSaveDirectoryBase.Data(), tCanPart_LamKch->GetCanvas()->GetName(), tSaveFileType.Data()));


//-----------------------------------------------------------------------------
//*****************************************************************************
//---------------- LamK0 ------------------------------------------------------
  tNx=4;
  tNy=2;

  TString tCanvasName_LamK0 = "LamK0AvgSepCfs";
  CanvasPartition* tCanPart_LamK0 = new CanvasPartition(tCanvasName_LamK0,tNx,tNy,tXLow,tXHigh,tYLow,tYHigh,0.12,0.0025,0.125,0.0025);
  tCanPart_LamK0->SetDrawOptStat(false);
//  tCanPart_LamK0->GetCanvas()->SetCanvasSize(2100, 2000);

  tTextXMin1 = 0.10;
  tTextWidth1 = 0.15;

  tTextXMin2 = 0.40;
  tTextWidth2 = 0.55;

  //----------------------------------------------------------------------------------------
  TH1* tCfLamK0_PosPos = LamK0->GetAvgSepHeavyCf(kPosPos)->GetHeavyCfClone();
  TH1* tCfLamK0_PosNeg = LamK0->GetAvgSepHeavyCf(kPosNeg)->GetHeavyCfClone();
  TH1* tCfLamK0_NegPos = LamK0->GetAvgSepHeavyCf(kNegPos)->GetHeavyCfClone();
  TH1* tCfLamK0_NegNeg = LamK0->GetAvgSepHeavyCf(kNegNeg)->GetHeavyCfClone();


  aNx=0, aNy=0;
  tCanPart_LamK0->AddGraph(aNx, aNy, (TH1D*)tDummy->Clone(), "", tMarkerStyle, tMarkerColor, tMarkerSize, "AXIS");
  tCanPart_LamK0->AddGraph(aNx, aNy, tCfLamK0_PosPos, "", tMarkerStyle, tMarkerColor, tMarkerSize, "ex0same");
    tCanPart_LamK0->AddPadPaveText(tCanPart_LamK0->SetupTPaveText(cAnalysisRootTags[kLamK0], aNx, aNy, tTextXMin1, tTextYMin, tTextWidth1, tTextHeight, tTextFont1, tTextSize), aNx, aNy);
    tCanPart_LamK0->AddPadPaveText(tCanPart_LamK0->SetupTPaveText(LamK0->GetDaughtersHistoTitle(kPosPos), aNx, aNy, tTextXMin2, tTextYMin, tTextWidth2, tTextHeight, tTextFont2, tTextSize), aNx, aNy);

  aNx=1, aNy=0;
  tCanPart_LamK0->AddGraph(aNx, aNy, (TH1D*)tDummy->Clone(), "", tMarkerStyle, tMarkerColor, tMarkerSize, "AXIS");
  tCanPart_LamK0->AddGraph(aNx, aNy, tCfLamK0_PosNeg, "", tMarkerStyle, tMarkerColor, tMarkerSize, "ex0same");
    tCanPart_LamK0->AddPadPaveText(tCanPart_LamK0->SetupTPaveText(cAnalysisRootTags[kLamK0], aNx, aNy, tTextXMin1, tTextYMin, tTextWidth1, tTextHeight, tTextFont1, tTextSize), aNx, aNy);
    tCanPart_LamK0->AddPadPaveText(tCanPart_LamK0->SetupTPaveText(LamK0->GetDaughtersHistoTitle(kPosNeg), aNx, aNy, tTextXMin2, tTextYMin, tTextWidth2, tTextHeight, tTextFont2, tTextSize), aNx, aNy);

  aNx=2, aNy=0;
  tCanPart_LamK0->AddGraph(aNx, aNy, (TH1D*)tDummy->Clone(), "", tMarkerStyle, tMarkerColor, tMarkerSize, "AXIS");
  tCanPart_LamK0->AddGraph(aNx, aNy, tCfLamK0_NegPos, "", tMarkerStyle, tMarkerColor, tMarkerSize, "ex0same");
    tCanPart_LamK0->AddPadPaveText(tCanPart_LamK0->SetupTPaveText(cAnalysisRootTags[kLamK0], aNx, aNy, tTextXMin1, tTextYMin, tTextWidth1, tTextHeight, tTextFont1, tTextSize), aNx, aNy);
    tCanPart_LamK0->AddPadPaveText(tCanPart_LamK0->SetupTPaveText(LamK0->GetDaughtersHistoTitle(kNegPos), aNx, aNy, tTextXMin2, tTextYMin, tTextWidth2, tTextHeight, tTextFont2, tTextSize), aNx, aNy);

  aNx=3, aNy=0;
  tCanPart_LamK0->AddGraph(aNx, aNy, (TH1D*)tDummy->Clone(), "", tMarkerStyle, tMarkerColor, tMarkerSize, "AXIS");
  tCanPart_LamK0->AddGraph(aNx, aNy, tCfLamK0_NegNeg, "", tMarkerStyle, tMarkerColor, tMarkerSize, "ex0same");
    tCanPart_LamK0->AddPadPaveText(tCanPart_LamK0->SetupTPaveText(cAnalysisRootTags[kLamK0], aNx, aNy, tTextXMin1, tTextYMin, tTextWidth1, tTextHeight, tTextFont1, tTextSize), aNx, aNy);
    tCanPart_LamK0->AddPadPaveText(tCanPart_LamK0->SetupTPaveText(LamK0->GetDaughtersHistoTitle(kNegNeg), aNx, aNy, tTextXMin2, tTextYMin, tTextWidth2, tTextHeight, tTextFont2, tTextSize), aNx, aNy);



  //----------------------------------------------------------------------------------------
  TH1* tCfALamK0_PosPos = ALamK0->GetAvgSepHeavyCf(kPosPos)->GetHeavyCfClone();
  TH1* tCfALamK0_PosNeg = ALamK0->GetAvgSepHeavyCf(kPosNeg)->GetHeavyCfClone();
  TH1* tCfALamK0_NegPos = ALamK0->GetAvgSepHeavyCf(kNegPos)->GetHeavyCfClone();
  TH1* tCfALamK0_NegNeg = ALamK0->GetAvgSepHeavyCf(kNegNeg)->GetHeavyCfClone();


  aNx=0, aNy=1;
  tCanPart_LamK0->AddGraph(aNx, aNy, (TH1D*)tDummy->Clone(), "", tMarkerStyle, tMarkerColor, tMarkerSize, "AXIS");
  tCanPart_LamK0->AddGraph(aNx, aNy, tCfALamK0_PosPos, "", tMarkerStyle, tMarkerColor, tMarkerSize, "ex0same");
    tCanPart_LamK0->AddPadPaveText(tCanPart_LamK0->SetupTPaveText(cAnalysisRootTags[kALamK0], aNx, aNy, tTextXMin1, tTextYMin, tTextWidth1, tTextHeight, tTextFont1, tTextSize), aNx, aNy);
    tCanPart_LamK0->AddPadPaveText(tCanPart_LamK0->SetupTPaveText(ALamK0->GetDaughtersHistoTitle(kPosPos), aNx, aNy, tTextXMin2, tTextYMin, tTextWidth2, tTextHeight, tTextFont2, tTextSize), aNx, aNy);

  aNx=1, aNy=1;
  tCanPart_LamK0->AddGraph(aNx, aNy, (TH1D*)tDummy->Clone(), "", tMarkerStyle, tMarkerColor, tMarkerSize, "AXIS");
  tCanPart_LamK0->AddGraph(aNx, aNy, tCfALamK0_PosNeg, "", tMarkerStyle, tMarkerColor, tMarkerSize, "ex0same");
    tCanPart_LamK0->AddPadPaveText(tCanPart_LamK0->SetupTPaveText(cAnalysisRootTags[kALamK0], aNx, aNy, tTextXMin1, tTextYMin, tTextWidth1, tTextHeight, tTextFont1, tTextSize), aNx, aNy);
    tCanPart_LamK0->AddPadPaveText(tCanPart_LamK0->SetupTPaveText(ALamK0->GetDaughtersHistoTitle(kPosNeg), aNx, aNy, tTextXMin2, tTextYMin, tTextWidth2, tTextHeight, tTextFont2, tTextSize), aNx, aNy);

  aNx=2, aNy=1;
  tCanPart_LamK0->AddGraph(aNx, aNy, (TH1D*)tDummy->Clone(), "", tMarkerStyle, tMarkerColor, tMarkerSize, "AXIS");
  tCanPart_LamK0->AddGraph(aNx, aNy, tCfALamK0_NegPos, "", tMarkerStyle, tMarkerColor, tMarkerSize, "ex0same");
    tCanPart_LamK0->AddPadPaveText(tCanPart_LamK0->SetupTPaveText(cAnalysisRootTags[kALamK0], aNx, aNy, tTextXMin1, tTextYMin, tTextWidth1, tTextHeight, tTextFont1, tTextSize), aNx, aNy);
    tCanPart_LamK0->AddPadPaveText(tCanPart_LamK0->SetupTPaveText(ALamK0->GetDaughtersHistoTitle(kNegPos), aNx, aNy, tTextXMin2, tTextYMin, tTextWidth2, tTextHeight, tTextFont2, tTextSize), aNx, aNy);

  aNx=3, aNy=1;
  tCanPart_LamK0->AddGraph(aNx, aNy, (TH1D*)tDummy->Clone(), "", tMarkerStyle, tMarkerColor, tMarkerSize, "AXIS");
  tCanPart_LamK0->AddGraph(aNx, aNy, tCfALamK0_NegNeg, "", tMarkerStyle, tMarkerColor, tMarkerSize, "ex0same");
    tCanPart_LamK0->AddPadPaveText(tCanPart_LamK0->SetupTPaveText(cAnalysisRootTags[kALamK0], aNx, aNy, tTextXMin1, tTextYMin, tTextWidth1, tTextHeight, tTextFont1, tTextSize), aNx, aNy);
    tCanPart_LamK0->AddPadPaveText(tCanPart_LamK0->SetupTPaveText(ALamK0->GetDaughtersHistoTitle(kNegNeg), aNx, aNy, tTextXMin2, tTextYMin, tTextWidth2, tTextHeight, tTextFont2, tTextSize), aNx, aNy);



  //----------------------------------
  tCanPart_LamK0->SetDrawUnityLine(true);
  tCanPart_LamK0->DrawAll();
  tCanPart_LamK0->DrawXaxisTitle("Avg. Sep. #bar{#bf{#Deltar}} (cm)", 43, 25, 0.70, 0.015); //Note, changing xaxis low (=0.315) does nothing
  tCanPart_LamK0->DrawYaxisTitle("#it{C}(#bar{#bf{#Deltar}})", 43, 35, 0.06, 0.80);

  if(bSave) tCanPart_LamK0->GetCanvas()->SaveAs(TString::Format("%s%s.%s", tSaveDirectoryBase.Data(), tCanPart_LamK0->GetCanvas()->GetName(), tSaveFileType.Data()));


//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.



  return 0;
}
