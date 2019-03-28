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

  TString tDirectoryBaseLamKch = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_%s/",tResultsDate.Data());
  TString tFileLocationBaseLamKch = TString::Format("%sResults_cLamcKch_%s",tDirectoryBaseLamKch.Data(),tResultsDate.Data());

  TString tDirectoryBaseLamK0 = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamK0_%s/",tResultsDate.Data());
  TString tFileLocationBaseLamK0 = TString::Format("%sResults_cLamK0_%s",tDirectoryBaseLamK0.Data(),tResultsDate.Data());

  //-----Data
  Analysis* LamK0 = new Analysis(tFileLocationBaseLamK0, kLamK0, tCentType);
  Analysis* ALamK0 = new Analysis(tFileLocationBaseLamK0, kALamK0, tCentType);

  Analysis* LamKchP = new Analysis(tFileLocationBaseLamKch, kLamKchP, tCentType);
  Analysis* ALamKchM = new Analysis(tFileLocationBaseLamKch, kALamKchM, tCentType);

  Analysis* LamKchM = new Analysis(tFileLocationBaseLamKch, kLamKchM, tCentType);
  Analysis* ALamKchP = new Analysis(tFileLocationBaseLamKch, kALamKchP, tCentType);

  //-------------------------------------------------------------------
 cout << "Centrality = " << cPrettyCentralityTags[tCentType] << endl;

 cout << "LamK0->GetNParticles(0) = " << LamK0->GetNParticles(0) << endl;
 cout << "LamK0->GetNParticles(1) = " << LamK0->GetNParticles(1) << endl << endl;

 cout << "ALamK0->GetNParticles(0) = " << ALamK0->GetNParticles(0) << endl;
 cout << "ALamK0->GetNParticles(1) = " << ALamK0->GetNParticles(1) << endl << endl;

 cout << "LamKchP->GetNParticles(0) = " << LamKchP->GetNParticles(0) << endl;
 cout << "LamKchP->GetNParticles(1) = " << LamKchP->GetNParticles(1) << endl << endl;

 cout << "ALamKchM->GetNParticles(0) = " << ALamKchM->GetNParticles(0) << endl;
 cout << "ALamKchM->GetNParticles(1) = " << ALamKchM->GetNParticles(1) << endl << endl;

 cout << "LamKchM->GetNParticles(0) = " << LamKchM->GetNParticles(0) << endl;
 cout << "LamKchM->GetNParticles(1) = " << LamKchM->GetNParticles(1) << endl << endl;

 cout << "ALamKchP->GetNParticles(0) = " << ALamKchP->GetNParticles(0) << endl;
 cout << "ALamKchP->GetNParticles(1) = " << ALamKchP->GetNParticles(1) << endl << endl;


//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
