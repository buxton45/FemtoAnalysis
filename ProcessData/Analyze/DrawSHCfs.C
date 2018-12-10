#include <iostream>
#include <iomanip>

#include "TApplication.h"

using std::cout;
using std::endl;
using std::vector;

#include "PartialAnalysis.h"
class PartialAnalysis;

#include "Analysis.h"
class Analysis;

int main(int argc, char **argv) 
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything
//-----------------------------------------------------------------------------

  //--Rarely change---------------------
  AnalysisRunType tAnRunType = kTrain;
  int tNPartialAnalysis = 2;
  CentralityType tCentType = k0010;
  //------------------------------------
  TString tResultsDate = "20181205";
  AnalysisType tAnType = kLamKchP;

  bool bSaveFigures = false;

  int tl = 1;
  int tm = 1;
  bool tRealComponent=true;

  double tMinNorm=0.32;
  double tMaxNorm=0.40;
  int tRebin=2;

//-----------------------------------------------------------------------------

  TString tGeneralAnTypeName;
  if(tAnType==kLamK0 || tAnType==kALamK0) tGeneralAnTypeName = "cLamK0";
  else if(tAnType==kLamKchP || tAnType==kALamKchM || tAnType==kLamKchM || tAnType==kALamKchP) tGeneralAnTypeName = "cLamcKch";
  else assert(0);


  TString tDirectoryBase = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_%s_%s/",tGeneralAnTypeName.Data(),tResultsDate.Data());
  TString tFileLocationBase = TString::Format("%sResults_%s_%s",tDirectoryBase.Data(),tGeneralAnTypeName.Data(),tResultsDate.Data());

//  TString tSaveDirectoryBase = TString::Format("/home/jesse/Analysis/Presentations/AliFemto/20180627/Figures/Fits/%s/", cAnalysisBaseTags[tAnType]);
  TString tSaveDirectoryBase = tDirectoryBase;

//-----------------------------------------------------------------------------

  Analysis* tAnaly = new Analysis(tFileLocationBase, tAnType, tCentType, tAnRunType, 2, "", false);
  TH1D* tSHCf = (TH1D*)tAnaly->GetSHCf(tl, tm, tRealComponent, tMinNorm, tMaxNorm, tRebin);
  tSHCf->GetXaxis()->SetRangeUser(0., 0.5);

  TCanvas* tCan = new TCanvas("tCan", "tCan");
  tCan->cd();

  tSHCf->Draw();


//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
