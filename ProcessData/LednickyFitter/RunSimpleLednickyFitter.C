#include "SimpleLednickyFitter.h"
class SimpleLednickyFitter;

SimpleLednickyFitter *myFitter = NULL;

//______________________________________________________________________________
void fcn(Int_t &npar, Double_t *gin, Double_t &f, Double_t *par, Int_t iflag)
{
  myFitter->SimpleLednickyFitter::CalculateFitFunction(npar,f,par);
}

//______________________________________________________________________________
CfLite* GetDataCfLite(AnalysisType tAnType)
{
  TString tResultsDate = "20161027";
  double aMinNorm=0.32, aMaxNorm=0.40;

  TString tGeneralAnTypeName;
  if(tAnType==kLamK0 || tAnType==kALamK0) tGeneralAnTypeName = "cLamK0";
  else if(tAnType==kLamKchP || tAnType==kALamKchM || tAnType==kLamKchM || tAnType==kALamKchP) tGeneralAnTypeName = "cLamcKch";
  else assert(0);


  TString tDirectoryBase = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_%s_%s/",tGeneralAnTypeName.Data(),tResultsDate.Data());
  TString tFileLocationBase = TString::Format("%sResults_%s_%s",tDirectoryBase.Data(),tGeneralAnTypeName.Data(),tResultsDate.Data());

  TString tFileLocationFemtoMinus = tFileLocationBase + cBFieldTags[kFemtoMinus] + TString(".root");
  TString tFileLocationFemtoPlus = tFileLocationBase + cBFieldTags[kFemtoPlus] + TString(".root");

  FitPartialAnalysis* tPartAnMinus = new FitPartialAnalysis(tFileLocationFemtoMinus, TString::Format("AnMinus_%s", cAnalysisBaseTags[tAnType]), tAnType, k0010, kFemtoMinus);
  FitPartialAnalysis* tPartAnPlus = new FitPartialAnalysis(tFileLocationFemtoPlus, TString::Format("AnPlus_%s", cAnalysisBaseTags[tAnType]), tAnType, k0010, kFemtoPlus);

  CfLite* tCfLiteMinus = tPartAnMinus->GetKStarCfLite();
  CfLite* tCfLitePlus = tPartAnPlus->GetKStarCfLite();

  TH1D* tNumMinus = (TH1D*) tCfLiteMinus->Num();
  TH1D* tDenMinus = (TH1D*) tCfLiteMinus->Den();

  TH1D* tNumPlus = (TH1D*) tCfLitePlus->Num();
  TH1D* tDenPlus = (TH1D*) tCfLitePlus->Den();

  TString tNumTotName = TString::Format("NumTot%s", cAnalysisBaseTags[tAnType]);
  TH1D* tNumTot = (TH1D*)tNumMinus->Clone(tNumTotName.Data());
  tNumMinus->Add(tNumPlus);

  TString tDenTotName = TString::Format("DenTot%s", cAnalysisBaseTags[tAnType]);
  TH1D* tDenTot = (TH1D*)tDenMinus->Clone(tDenTotName.Data());
  tDenMinus->Add(tDenPlus);

  TString tCfTotName = TString::Format("CfTot%s", cAnalysisBaseTags[tAnType]);
  CfLite* tReturnCfLite = new CfLite(tCfTotName, tCfTotName, tNumTot, tDenTot, aMinNorm, aMaxNorm);

  return tReturnCfLite;
}


//______________________________________________________________________________
int main(int argc, char **argv) 
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything

  ChronoTimer tFullTimer(kSec);
  tFullTimer.Start();
//-----------------------------------------------------------------------------
  AnalysisType tAnType = kLamKchP;
  bool bSaveFigures = false;

  TString tDirectory = "~/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b2/";
  TString tFileLocationCfs = tDirectory + "CorrelationFunctions_10MixedEvNum";

//  TString tFileNameModifier = "";
  TString tFileNameModifier = "_WeightParentsInteraction";
//  TString tFileNameModifier = "_WeightParentsInteraction_NoCharged";

  tFileLocationCfs += tFileNameModifier;
  tFileLocationCfs += TString(".root");

  TString tSaveLocationBase = "/home/jesse/Analysis/Presentations/GroupMeetings/20171012/Figures/";

//-------------------------------------------------------------------------------
  TString tBaseName = "Full";
  //TString tBaseName = "PrimaryOnly";
  //TString tBaseName = "PrimaryAndShortDecays";
  //TString tBaseName = "WithoutSigmaSt";
  //TString tBaseName = "SigmaStOnly";

  SimpleLednickyFitter *tSLFitter = new SimpleLednickyFitter(tAnType, tFileLocationCfs, tBaseName);

/*
  CfLite* tDataCfLite = GetDataCfLite(tAnType);
  SimpleLednickyFitter *tSLFitter = new SimpleLednickyFitter(tAnType, tDataCfLite);
*/
/*
  vector<double> tSimParams(6);
    tSimParams[0] = 0.5;
    tSimParams[1] = 5.0;
    tSimParams[2] = -0.5;
    tSimParams[3] = 0.5;
    tSimParams[4] = 0.;
    tSimParams[5] = 1.;
  SimpleLednickyFitter *tSLFitter = new SimpleLednickyFitter(tAnType, tSimParams);
*/
  tSLFitter->GetMinuitObject()->SetFCN(fcn);
  myFitter = tSLFitter;

  tSLFitter->DoFit();


  TCanvas* tCanCfWithFit = new TCanvas("tCanCfWithFit", "tCanCfWithFit");
  tSLFitter->DrawCfWithFit((TPad*)tCanCfWithFit);

//  TCanvas* tCan2 = new TCanvas("tCan2", "tCan2");
//  tSLFitter->DrawCfNumDen((TPad*)tCan2);

  if(bSaveFigures) tCanCfWithFit->SaveAs(tSaveLocationBase + TString::Format("%s/", cAnalysisBaseTags[tAnType]) + TString(tCanCfWithFit->GetName()) + tBaseName + tFileNameModifier + TString(".eps"));

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
