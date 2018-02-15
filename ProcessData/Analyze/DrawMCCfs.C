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

//  TString tResultsDate = "20161027";
  TString tResultsDate = "20170213";
//  TString tResultsDate = "20171227";

  AnalysisType tAnType = kLamK0;
  AnalysisRunType tAnRunType = kTrain;
  int tNPartialAnalysis = 2;

  AnalysisType tConjAnType;
  if(tAnType==kLamK0) {tConjAnType=kALamK0;}
  else if(tAnType==kLamKchP) {tConjAnType=kALamKchM;}
  else if(tAnType==kLamKchM) {tConjAnType=kALamKchP;}

//-----------------------------------------------------------------------------
  bool bSaveFigures = false;

//-----------------------------------------------------------------------------

  TString tGeneralAnTypeName;
  if(tAnType==kLamK0 || tAnType==kALamK0) tGeneralAnTypeName = "cLamK0";
  else if(tAnType==kLamKchP || tAnType==kALamKchM || tAnType==kLamKchM || tAnType==kALamKchP) tGeneralAnTypeName = "cLamcKch";
  else assert(0);


  TString tDirectoryBase = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_%s_%s/",tGeneralAnTypeName.Data(),tResultsDate.Data());
  TString tFileLocationBaseMCa = TString::Format("%sResults_%sMCa_%s",tDirectoryBase.Data(),tGeneralAnTypeName.Data(),tResultsDate.Data());
  TString tFileLocationBaseMCb = TString::Format("%sResults_%sMCb_%s",tDirectoryBase.Data(),tGeneralAnTypeName.Data(),tResultsDate.Data());

  TString tSaveDirectoryBase = "/home/jesse/Analysis/FemtoAnalysis/AnalysisNotes/Comments/Laura/20180117/Figures/";
//  TString tSaveDirectoryBase = tDirectoryBase;


  //-----MC
  Analysis* LamKMC_0010 = new Analysis(tFileLocationBaseMCa,tAnType,k0010);
  Analysis* ALamKMC_0010 = new Analysis(tFileLocationBaseMCa,tConjAnType,k0010);

  Analysis* LamKMC_1030 = new Analysis(tFileLocationBaseMCb,tAnType,k1030);
  Analysis* ALamKMC_1030 = new Analysis(tFileLocationBaseMCb,tConjAnType,k1030);

  Analysis* LamKMC_3050 = new Analysis(tFileLocationBaseMCb,tAnType,k3050);
  Analysis* ALamKMC_3050 = new Analysis(tFileLocationBaseMCb,tConjAnType,k3050);

  //-------------------------------------------------------------------
  int tRebin = 4;

  LamKMC_0010->BuildKStarHeavyCf();
    LamKMC_0010->GetKStarHeavyCf()->Rebin(tRebin);
  ALamKMC_0010->BuildKStarHeavyCf();
    ALamKMC_0010->GetKStarHeavyCf()->Rebin(tRebin);

  LamKMC_1030->BuildKStarHeavyCf();
    LamKMC_1030->GetKStarHeavyCf()->Rebin(tRebin);
  ALamKMC_1030->BuildKStarHeavyCf();
    ALamKMC_1030->GetKStarHeavyCf()->Rebin(tRebin);

  LamKMC_3050->BuildKStarHeavyCf();
    LamKMC_3050->GetKStarHeavyCf()->Rebin(tRebin);
  ALamKMC_3050->BuildKStarHeavyCf();
    ALamKMC_3050->GetKStarHeavyCf()->Rebin(tRebin);


  //-------------------------------------------------------------------
  bool bZoomROP = false;

  TString tCanvasName = TString::Format("canMCKStarCfs_%s", cAnalysisBaseTags[tAnType]);

  int tNx = 2;
  int tNy = 3;

  double tXLow = -0.02;
  double tXHigh = 0.99;
  double tYLow = 0.71;
  double tYHigh = 1.09;
  if(bZoomROP)
  {
    tXLow = -0.02;
    tXHigh = 0.329;
    tYLow = 0.86;
    tYHigh = 1.07;
  }

  CanvasPartition* tCanPart = new CanvasPartition(tCanvasName,tNx,tNy,tXLow,tXHigh,tYLow,tYHigh,0.12,0.0025,0.13,0.0025);
  tCanPart->SetDrawOptStat(false);

  int tColor;
  if(tAnType==kLamK0 || tAnType==kALamK0) tColor=kBlack;
  else if(tAnType==kLamKchP || tAnType==kALamKchM) tColor=kRed+1;
  else if(tAnType==kLamKchM || tAnType==kALamKchP) tColor=kBlue+1;
  else tColor=1;

  TString tCombinedText;
  TPaveText* tCombined;

  tCanPart->AddGraph(0, 0, (TH1*)LamKMC_0010->GetKStarHeavyCf()->GetHeavyCf(), "", 20, tColor, 0.5, "ex0");
    tCombinedText = TString::Format("%s  %s (MC)", cAnalysisRootTags[tAnType], cPrettyCentralityTags[k0010]);
    tCombined = tCanPart->SetupTPaveText(tCombinedText, 0, 0, 0.65,0.825,0.15,0.10,63,20);
    tCanPart->AddPadPaveText(tCombined, 0, 0);
  tCanPart->AddGraph(1, 0, (TH1*)ALamKMC_0010->GetKStarHeavyCf()->GetHeavyCf(), "", 20, tColor, 0.5, "ex0");
    tCombinedText = TString::Format("%s  %s (MC)", cAnalysisRootTags[tConjAnType], cPrettyCentralityTags[k0010]);
    tCombined = tCanPart->SetupTPaveText(tCombinedText, 1, 0, 0.65,0.825,0.15,0.10,63,20);
    tCanPart->AddPadPaveText(tCombined, 1, 0);

  tCanPart->AddGraph(0, 1, (TH1*)LamKMC_1030->GetKStarHeavyCf()->GetHeavyCf(), "", 20, tColor, 0.5, "ex0");
    tCombinedText = TString::Format("%s  %s (MC)", cAnalysisRootTags[tAnType], cPrettyCentralityTags[k1030]);
    tCombined = tCanPart->SetupTPaveText(tCombinedText, 0, 1, 0.65,0.825,0.15,0.10,63,20);
    tCanPart->AddPadPaveText(tCombined, 0, 1);
  tCanPart->AddGraph(1, 1, (TH1*)ALamKMC_1030->GetKStarHeavyCf()->GetHeavyCf(), "", 20, tColor, 0.5, "ex0");
    tCombinedText = TString::Format("%s  %s (MC)", cAnalysisRootTags[tConjAnType], cPrettyCentralityTags[k1030]);
    tCombined = tCanPart->SetupTPaveText(tCombinedText, 1, 1, 0.65,0.825,0.15,0.10,63,20);
    tCanPart->AddPadPaveText(tCombined, 1, 1);

  tCanPart->AddGraph(0, 2, (TH1*)LamKMC_3050->GetKStarHeavyCf()->GetHeavyCf(), "", 20, tColor, 0.5, "ex0");
    tCombinedText = TString::Format("%s  %s (MC)", cAnalysisRootTags[tAnType], cPrettyCentralityTags[k3050]);
    tCombined = tCanPart->SetupTPaveText(tCombinedText, 0, 2, 0.65,0.825,0.15,0.10,63,20);
    tCanPart->AddPadPaveText(tCombined, 0, 2);
  tCanPart->AddGraph(1, 2, (TH1*)ALamKMC_3050->GetKStarHeavyCf()->GetHeavyCf(), "", 20, tColor, 0.5, "ex0");
    tCombinedText = TString::Format("%s  %s (MC)", cAnalysisRootTags[tConjAnType], cPrettyCentralityTags[k3050]);
    tCombined = tCanPart->SetupTPaveText(tCombinedText, 1, 2, 0.65,0.825,0.15,0.10,63,20);
    tCanPart->AddPadPaveText(tCombined, 1, 2);


  tCanPart->SetDrawUnityLine(true);
  tCanPart->DrawAll();
  tCanPart->DrawXaxisTitle("#it{k}* (GeV/#it{c})");
  tCanPart->DrawYaxisTitle("#it{C}(#it{k}*)",43,25,0.05,0.85);

  //------------------------------------------------------------

  if(bSaveFigures)
  {
    tCanPart->GetCanvas()->SaveAs(TString::Format("%s%s.eps", tSaveDirectoryBase.Data(), tCanPart->GetCanvas()->GetName()));
  }

//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
