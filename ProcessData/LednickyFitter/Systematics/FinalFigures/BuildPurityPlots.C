//Taken from /home/jesse/Analysis/FemtoAnalysis/ProcessData/Analyze/BuildcLamK0Analyses.C

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


#include "Analysis.h"
class Analysis;

int main(int argc, char **argv) 
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything
//-----------------------------------------------------------------------------
  TString tResultsDate = "20180505";

  AnalysisType tAnType = kLamK0;
  AnalysisType tConjAnType = kALamK0;
  AnalysisRunType tAnRunType = kTrain;
  int tNPartialAnalysis = 2;
  CentralityType tCentType = k0010;  //TODO

//-----------------------------------------------------------------------------
  bool bSaveFigures = false;
  TString tSaveFileType = "pdf";

  bool bPrintPurity = true;
//-----------------------------------------------------------------------------

  TString tGeneralAnTypeName;
  if(tAnType==kLamK0 || tAnType==kALamK0) tGeneralAnTypeName = "cLamK0";
  else if(tAnType==kLamKchP || tAnType==kALamKchM || tAnType==kLamKchM || tAnType==kALamKchP) tGeneralAnTypeName = "cLamcKch";
  else assert(0);


  TString tDirectoryBase = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_%s_%s/",tGeneralAnTypeName.Data(),tResultsDate.Data());
  TString tFileLocationBase = TString::Format("%sResults_%s_%s",tDirectoryBase.Data(),tGeneralAnTypeName.Data(),tResultsDate.Data());

  TString tSaveDirectoryBase = "/home/jesse/Analysis/FemtoAnalysis/AnalysisNotes/3_DataSelection/Figures/";

  //-----Data
  Analysis* LamK0 = new Analysis(tFileLocationBase,tAnType,tCentType);
  Analysis* ALamK0 = new Analysis(tFileLocationBase,tConjAnType,tCentType);

  //-------------------------------------------------------------------
  vector<TString> tPrintPurText{"", "_wPrintPurity"};

  LamK0->BuildPurityCollection();
  ALamK0->BuildPurityCollection();

  TCanvas* canPurity = new TCanvas("canPurity","canPurity");
  canPurity->Divide(2,1);

  LamK0->DrawAllPurityHistos((TPad*)canPurity->cd(1), bPrintPurity);
  ALamK0->DrawAllPurityHistos((TPad*)canPurity->cd(2), bPrintPurity);

  TCanvas* canPurityLam = new TCanvas("canPurityLam", "canPurityLam");
  LamK0->DrawPurityHisto(0, canPurityLam, bPrintPurity);

  TCanvas* canPurityK0 = new TCanvas("canPurityK0", "canPurityK0");
  LamK0->DrawPurityHisto(1, canPurityK0, bPrintPurity);

  TCanvas* canPurityALam = new TCanvas("canPurityALam", "canPurityALam");
  ALamK0->DrawPurityHisto(0, canPurityALam, bPrintPurity);

  if(bSaveFigures)
  {
    TString aName = TString::Format("cLamK0Purity%s.%s", tPrintPurText[bPrintPurity].Data(), tSaveFileType.Data());
    canPurity->SaveAs(tSaveDirectoryBase+aName);

/*
    TString aName2 = TString::Format("LamPurity%s_LamK0.%s", tPrintPurText[bPrintPurity].Data(), tSaveFileType.Data());
    canPurity->cd(1)->cd(1)->SaveAs(tSaveDirectoryBase+aName2);

    TString aName3 = TString::Format("K0Purity%s_LamK0.%s", tPrintPurText[bPrintPurity].Data(), tSaveFileType.Data());
    canPurity->cd(1)->cd(2)->SaveAs(tSaveDirectoryBase+aName3);
*/

    TString aNameLam = TString::Format("LamPurity%s_LamK0.%s", tPrintPurText[bPrintPurity].Data(), tSaveFileType.Data());
    canPurityLam->SaveAs(tSaveDirectoryBase+aNameLam);

    TString aNameK0  = TString::Format("K0Purity%s_LamK0.%s", tPrintPurText[bPrintPurity].Data(), tSaveFileType.Data());
    canPurityK0->SaveAs(tSaveDirectoryBase+aNameK0);

    TString aNameALam = TString::Format("ALamPurity%s_ALamK0.%s", tPrintPurText[bPrintPurity].Data(), tSaveFileType.Data());
    canPurityALam->SaveAs(tSaveDirectoryBase+aNameALam);
  }



//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
