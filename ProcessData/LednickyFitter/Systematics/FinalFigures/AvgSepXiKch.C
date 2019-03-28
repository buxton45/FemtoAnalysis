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

  //TODO TODO TODO TODO TODO If using ~/Analysis/FemtoAnalysis/Results/Results_cXicKch_20170425_maxDcaXi0.3/Results_cXicKch_20170425_maxDcaXi0.3 file
  //TODO TODO TODO TODO TODO must set cAvgSepCfBaseTagsDen[7] = "DenV0TrackBacAvgSepCf_", not "DenXiTrackBacAvgSepCf_"
  //TODO TODO TODO TODO TODO Only naming is wrong, not figure.  In more current versions, naming is correct

  //TODO TODO TODO TODO TODO Also, pdfviewer and LaTex are being super annoying, and not displaying Delta correctly.  Workaround is to save file as
  //TODO TODO TODO TODO TODO .eps and convert with epstopdf to pdf

  bool bSave = false;
  TString tSaveFileType = "pdf";
  TString tSaveDirectoryBase = "/home/jesse/Analysis/FemtoAnalysis/AnalysisNotes/3_DataSelection/Figures/";

  CentralityType tCentType = k0010;

  TString FileLocationBase = "~/Analysis/FemtoAnalysis/Results/Results_cXicKch_20170425_maxDcaXi0.3/Results_cXicKch_20170425_maxDcaXi0.3";
  Analysis* XiKchP = new Analysis(FileLocationBase, kXiKchP, tCentType);
  Analysis* AXiKchP = new Analysis(FileLocationBase, kAXiKchP, tCentType);
  Analysis* XiKchM = new Analysis(FileLocationBase, kXiKchM, tCentType);
  Analysis* AXiKchM = new Analysis(FileLocationBase, kAXiKchM, tCentType);




//-----------------------------------------------------------------------------

  double aMinNorm = 14.99;
  double aMaxNorm = 19.99;
  int aRebin = 2;

  XiKchP->BuildAllAvgSepHeavyCfs(aMinNorm, aMaxNorm, aRebin);
  AXiKchP->BuildAllAvgSepHeavyCfs(aMinNorm, aMaxNorm, aRebin);
  XiKchM->BuildAllAvgSepHeavyCfs(aMinNorm, aMaxNorm, aRebin);
  AXiKchM->BuildAllAvgSepHeavyCfs(aMinNorm, aMaxNorm, aRebin);

//-----------------------------------------------------------------------------
  int tNx=3;
  int tNy=4;
  double tXLow = -0.5;
  double tXHigh = 14.;
  double tYLow = -0.125;
  double tYHigh = 2.25;

  TString tCanvasName = "XiKchAvgSepCfs";
  CanvasPartition* tCanPart = new CanvasPartition(tCanvasName,tNx,tNy,tXLow,tXHigh,tYLow,tYHigh,0.12,0.0025,0.125,0.0025);
  tCanPart->SetDrawOptStat(false);
//  tCanPart->GetCanvas()->SetCanvasSize(2100, 2000);

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
  TH1* tCfXiKchP_TrackPos = XiKchP->GetAvgSepHeavyCf(kTrackPos)->GetHeavyCfClone();
  TH1* tCfXiKchP_TrackNeg = XiKchP->GetAvgSepHeavyCf(kTrackNeg)->GetHeavyCfClone();
  TH1* tCfXiKchP_TrackBac = XiKchP->GetAvgSepHeavyCf(kTrackBac)->GetHeavyCfClone();

  aNx=0, aNy=0;
  tCanPart->AddGraph(aNx, aNy, (TH1D*)tDummy->Clone(), "", tMarkerStyle, tMarkerColor, tMarkerSize, "AXIS");
  tCanPart->AddGraph(aNx, aNy, tCfXiKchP_TrackPos, "", tMarkerStyle, tMarkerColor, tMarkerSize, "ex0same");
    tCanPart->AddPadPaveText(tCanPart->SetupTPaveText(cAnalysisRootTags[kXiKchP], aNx, aNy, tTextXMin1, tTextYMin, tTextWidth1, tTextHeight, tTextFont1, tTextSize), aNx, aNy);
    tCanPart->AddPadPaveText(tCanPart->SetupTPaveText("p(#Lambda(#Xi-)) - K+", aNx, aNy, tTextXMin2, tTextYMin, tTextWidth2, tTextHeight, tTextFont2, tTextSize), aNx, aNy);

  aNx=1, aNy=0;
  tCanPart->AddGraph(aNx, aNy, (TH1D*)tDummy->Clone(), "", tMarkerStyle, tMarkerColor, tMarkerSize, "AXIS");
  tCanPart->AddGraph(aNx, aNy, tCfXiKchP_TrackNeg, "", tMarkerStyle, tMarkerColor, tMarkerSize, "ex0same");
    tCanPart->AddPadPaveText(tCanPart->SetupTPaveText(cAnalysisRootTags[kXiKchP], aNx, aNy, tTextXMin1, tTextYMin, tTextWidth1, tTextHeight, tTextFont1, tTextSize), aNx, aNy);
    tCanPart->AddPadPaveText(tCanPart->SetupTPaveText("#pi-(#Lambda(#Xi-)) - K+", aNx, aNy, tTextXMin2, tTextYMin, tTextWidth2, tTextHeight, tTextFont2, tTextSize), aNx, aNy);

  aNx=2, aNy=0;
  tCanPart->AddGraph(aNx, aNy, (TH1D*)tDummy->Clone(), "", tMarkerStyle, tMarkerColor, tMarkerSize, "AXIS");
  tCanPart->AddGraph(aNx, aNy, tCfXiKchP_TrackBac, "", tMarkerStyle, tMarkerColor, tMarkerSize, "ex0same");
    tCanPart->AddPadPaveText(tCanPart->SetupTPaveText(cAnalysisRootTags[kXiKchP], aNx, aNy, tTextXMin1, tTextYMin, tTextWidth1, tTextHeight, tTextFont1, tTextSize), aNx, aNy);
    tCanPart->AddPadPaveText(tCanPart->SetupTPaveText(XiKchP->GetDaughtersHistoTitle(kTrackBac, true), aNx, aNy, tTextXMin2, tTextYMin, tTextWidth2, tTextHeight, tTextFont2, tTextSize), aNx, aNy);



  //----------------------------------------------------------------------------------------
  TH1* tCfAXiKchM_TrackPos = AXiKchM->GetAvgSepHeavyCf(kTrackPos)->GetHeavyCfClone();
  TH1* tCfAXiKchM_TrackNeg = AXiKchM->GetAvgSepHeavyCf(kTrackNeg)->GetHeavyCfClone();
  TH1* tCfAXiKchM_TrackBac = AXiKchM->GetAvgSepHeavyCf(kTrackBac)->GetHeavyCfClone();

  aNx=0, aNy=1;
  tCanPart->AddGraph(aNx, aNy, (TH1D*)tDummy->Clone(), "", tMarkerStyle, tMarkerColor, tMarkerSize, "AXIS");
  tCanPart->AddGraph(aNx, aNy, tCfAXiKchM_TrackPos, "", tMarkerStyle, tMarkerColor, tMarkerSize, "ex0same");
    tCanPart->AddPadPaveText(tCanPart->SetupTPaveText(cAnalysisRootTags[kAXiKchM], aNx, aNy, tTextXMin1, tTextYMin, tTextWidth1, tTextHeight, tTextFont1, tTextSize), aNx, aNy);
    tCanPart->AddPadPaveText(tCanPart->SetupTPaveText("#pi+(#bar{#Lambda}(#bar{#Xi}+)) - K-", aNx, aNy, tTextXMin2, tTextYMin, tTextWidth2, tTextHeight, tTextFont2, tTextSize), aNx, aNy);

  aNx=1, aNy=1;
  tCanPart->AddGraph(aNx, aNy, (TH1D*)tDummy->Clone(), "", tMarkerStyle, tMarkerColor, tMarkerSize, "AXIS");
  tCanPart->AddGraph(aNx, aNy, tCfAXiKchM_TrackNeg, "", tMarkerStyle, tMarkerColor, tMarkerSize, "ex0same");
    tCanPart->AddPadPaveText(tCanPart->SetupTPaveText(cAnalysisRootTags[kAXiKchM], aNx, aNy, tTextXMin1, tTextYMin, tTextWidth1, tTextHeight, tTextFont1, tTextSize), aNx, aNy);
    tCanPart->AddPadPaveText(tCanPart->SetupTPaveText("#bar{p}(#bar{#Lambda}(#bar{#Xi}+)) - K-", aNx, aNy, tTextXMin2, tTextYMin, tTextWidth2, tTextHeight, tTextFont2, tTextSize), aNx, aNy);

  aNx=2, aNy=1;
  tCanPart->AddGraph(aNx, aNy, (TH1D*)tDummy->Clone(), "", tMarkerStyle, tMarkerColor, tMarkerSize, "AXIS");
  tCanPart->AddGraph(aNx, aNy, tCfAXiKchM_TrackBac, "", tMarkerStyle, tMarkerColor, tMarkerSize, "ex0same");
    tCanPart->AddPadPaveText(tCanPart->SetupTPaveText(cAnalysisRootTags[kAXiKchM], aNx, aNy, tTextXMin1, tTextYMin, tTextWidth1, tTextHeight, tTextFont1, tTextSize), aNx, aNy);
    tCanPart->AddPadPaveText(tCanPart->SetupTPaveText(AXiKchM->GetDaughtersHistoTitle(kTrackBac, true), aNx, aNy, tTextXMin2, tTextYMin, tTextWidth2, tTextHeight, tTextFont2, tTextSize), aNx, aNy);





  //----------------------------------------------------------------------------------------
  TH1* tCfXiKchM_TrackPos = XiKchM->GetAvgSepHeavyCf(kTrackPos)->GetHeavyCfClone();
  TH1* tCfXiKchM_TrackNeg = XiKchM->GetAvgSepHeavyCf(kTrackNeg)->GetHeavyCfClone();
  TH1* tCfXiKchM_TrackBac = XiKchM->GetAvgSepHeavyCf(kTrackBac)->GetHeavyCfClone();

  aNx=0, aNy=2;
  tCanPart->AddGraph(aNx, aNy, (TH1D*)tDummy->Clone(), "", tMarkerStyle, tMarkerColor, tMarkerSize, "AXIS");
  tCanPart->AddGraph(aNx, aNy, tCfXiKchM_TrackPos, "", tMarkerStyle, tMarkerColor, tMarkerSize, "ex0same");
    tCanPart->AddPadPaveText(tCanPart->SetupTPaveText(cAnalysisRootTags[kXiKchM], aNx, aNy, tTextXMin1, tTextYMin, tTextWidth1, tTextHeight, tTextFont1, tTextSize), aNx, aNy);
    tCanPart->AddPadPaveText(tCanPart->SetupTPaveText("p(#Lambda(#Xi-)) - K-", aNx, aNy, tTextXMin2, tTextYMin, tTextWidth2, tTextHeight, tTextFont2, tTextSize), aNx, aNy);

  aNx=1, aNy=2;
  tCanPart->AddGraph(aNx, aNy, (TH1D*)tDummy->Clone(), "", tMarkerStyle, tMarkerColor, tMarkerSize, "AXIS");
  tCanPart->AddGraph(aNx, aNy, tCfXiKchM_TrackNeg, "", tMarkerStyle, tMarkerColor, tMarkerSize, "ex0same");
    tCanPart->AddPadPaveText(tCanPart->SetupTPaveText(cAnalysisRootTags[kXiKchM], aNx, aNy, tTextXMin1, tTextYMin, tTextWidth1, tTextHeight, tTextFont1, tTextSize), aNx, aNy);
    tCanPart->AddPadPaveText(tCanPart->SetupTPaveText("#pi-(#Lambda(#Xi-)) - K-", aNx, aNy, tTextXMin2, tTextYMin, tTextWidth2, tTextHeight, tTextFont2, tTextSize), aNx, aNy);

  aNx=2, aNy=2;
  tCanPart->AddGraph(aNx, aNy, (TH1D*)tDummy->Clone(), "", tMarkerStyle, tMarkerColor, tMarkerSize, "AXIS");
  tCanPart->AddGraph(aNx, aNy, tCfXiKchM_TrackBac, "", tMarkerStyle, tMarkerColor, tMarkerSize, "ex0same");
    tCanPart->AddPadPaveText(tCanPart->SetupTPaveText(cAnalysisRootTags[kXiKchM], aNx, aNy, tTextXMin1, tTextYMin, tTextWidth1, tTextHeight, tTextFont1, tTextSize), aNx, aNy);
    tCanPart->AddPadPaveText(tCanPart->SetupTPaveText(XiKchM->GetDaughtersHistoTitle(kTrackBac, true), aNx, aNy, tTextXMin2, tTextYMin, tTextWidth2, tTextHeight, tTextFont2, tTextSize), aNx, aNy);





  //----------------------------------------------------------------------------------------
  TH1* tCfAXiKchP_TrackPos = AXiKchP->GetAvgSepHeavyCf(kTrackPos)->GetHeavyCfClone();
  TH1* tCfAXiKchP_TrackNeg = AXiKchP->GetAvgSepHeavyCf(kTrackNeg)->GetHeavyCfClone();
  TH1* tCfAXiKchP_TrackBac = AXiKchP->GetAvgSepHeavyCf(kTrackBac)->GetHeavyCfClone();

  aNx=0, aNy=3;
  tCanPart->AddGraph(aNx, aNy, (TH1D*)tDummy->Clone(), "", tMarkerStyle, tMarkerColor, tMarkerSize, "AXIS");
  tCanPart->AddGraph(aNx, aNy, tCfAXiKchP_TrackPos, "", tMarkerStyle, tMarkerColor, tMarkerSize, "ex0same");
    tCanPart->AddPadPaveText(tCanPart->SetupTPaveText(cAnalysisRootTags[kAXiKchP], aNx, aNy, tTextXMin1, tTextYMin, tTextWidth1, tTextHeight, tTextFont1, tTextSize), aNx, aNy);
    tCanPart->AddPadPaveText(tCanPart->SetupTPaveText("#pi+(#bar{#Lambda}(#bar{#Xi}+)) - K+", aNx, aNy, tTextXMin2, tTextYMin, tTextWidth2, tTextHeight, tTextFont2, tTextSize), aNx, aNy);

  aNx=1, aNy=3;
  tCanPart->AddGraph(aNx, aNy, (TH1D*)tDummy->Clone(), "", tMarkerStyle, tMarkerColor, tMarkerSize, "AXIS");
  tCanPart->AddGraph(aNx, aNy, tCfAXiKchP_TrackNeg, "", tMarkerStyle, tMarkerColor, tMarkerSize, "ex0same");
    tCanPart->AddPadPaveText(tCanPart->SetupTPaveText(cAnalysisRootTags[kAXiKchP], aNx, aNy, tTextXMin1, tTextYMin, tTextWidth1, tTextHeight, tTextFont1, tTextSize), aNx, aNy);
    tCanPart->AddPadPaveText(tCanPart->SetupTPaveText("#bar{p}(#bar{#Lambda}(#bar{#Xi}+)) - K+", aNx, aNy, tTextXMin2, tTextYMin, tTextWidth2, tTextHeight, tTextFont2, tTextSize), aNx, aNy);

  aNx=2, aNy=3;
  tCanPart->AddGraph(aNx, aNy, (TH1D*)tDummy->Clone(), "", tMarkerStyle, tMarkerColor, tMarkerSize, "AXIS");
  tCanPart->AddGraph(aNx, aNy, tCfAXiKchP_TrackBac, "", tMarkerStyle, tMarkerColor, tMarkerSize, "ex0same");
    tCanPart->AddPadPaveText(tCanPart->SetupTPaveText(cAnalysisRootTags[kAXiKchP], aNx, aNy, tTextXMin1, tTextYMin, tTextWidth1, tTextHeight, tTextFont1, tTextSize), aNx, aNy);
    tCanPart->AddPadPaveText(tCanPart->SetupTPaveText(AXiKchP->GetDaughtersHistoTitle(kTrackBac, true), aNx, aNy, tTextXMin2, tTextYMin, tTextWidth2, tTextHeight, tTextFont2, tTextSize), aNx, aNy);




  //----------------------------------
  tCanPart->SetDrawUnityLine(true);
  tCanPart->DrawAll();
  tCanPart->DrawXaxisTitle("Avg. Sep. #bar{#bf{#Deltar}} (cm)", 43, 25, 0.70, 0.015); //Note, changing xaxis low (=0.315) does nothing
  tCanPart->DrawYaxisTitle("#it{C}(#bar{#bf{#Deltar}})", 43, 35, 0.06, 0.80);

  if(bSave) tCanPart->GetCanvas()->SaveAs(TString::Format("%s%s.%s", tSaveDirectoryBase.Data(), tCanPart->GetCanvas()->GetName(), tSaveFileType.Data()));

//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.



  return 0;
}
