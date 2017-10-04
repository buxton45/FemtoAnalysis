#include <iostream>
#include <iomanip>
#include <math.h>
#include <vector>
#include <assert.h>

#include "ThermEventsCollection.h"
#include "TApplication.h"
#include "TStyle.h"
#include "TPaveText.h"
#include "TLegend.h"

#include "PIDMapping.h"
#include "ThermCommon.h"



//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________

int main(int argc, char **argv) 
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything
//-----------------------------------------------------------------------------

/*
  TString tDirectory = "~/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b2/";
  TString tFileLocationCfs = tDirectory + "CorrelationFunctions_5MixedEvNum.root";
*/

  TString tDirectory = "/home/jesse/Analysis/ReducedTherminator2Events/test/";
  TString tFileLocationCfs = tDirectory + "testCorrelationFunctions_5MixedEvNum.root";

  AnalysisType tAnType = kLamKchP;

  TH3D* tNum3d = Get3dHisto(tFileLocationCfs, TString::Format("Num3d%s", cAnalysisBaseTags[tAnType]));
  int tIndexLambda = GetParticleIndexInPidInfo(kPDGLam);
  int tIndexKchP = GetParticleIndexInPidInfo(kPDGKchP);
cout << "tIndexLambda = " << tIndexLambda << endl;
cout << "tIndexKchP = " << tIndexKchP << endl;
  TH1D* tNumPrimOnlyProject = tNum3d->ProjectionZ("tNumPrimOnlyProject", tIndexLambda+1, tIndexLambda+1, tIndexKchP+1, tIndexKchP+1);
    tNumPrimOnlyProject->SetMarkerStyle(20);
    tNumPrimOnlyProject->SetMarkerColor(1);

  TH1D* tNumPrimOnly = Get1dHisto(tFileLocationCfs, TString::Format("NumPrimaryOnly%s", cAnalysisBaseTags[tAnType]));
    tNumPrimOnly->SetMarkerStyle(20);
    tNumPrimOnly->SetMarkerColor(2);

  TCanvas *tCan = new TCanvas("tCan", "tCan");
  tCan->cd();
  tNumPrimOnlyProject->Draw();
  tNumPrimOnly->Draw("same");

  TCanvas *tCan2 = new TCanvas("tCan2", "tCan2");
  tCan2->Divide(2,1);
  tCan2->cd(1);
  tNumPrimOnlyProject->Draw();
  tCan2->cd(2);
  tNumPrimOnly->Draw();
//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}


