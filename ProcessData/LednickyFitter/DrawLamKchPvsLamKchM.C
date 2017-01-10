#include "FitGenerator.h"
class FitGenerator;

int main(int argc, char **argv) 
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything

  ChronoTimer tFullTimer(kSec);
  tFullTimer.Start();
//-----------------------------------------------------------------------------
  TString tResultsDate = "20161027";

  AnalysisRunType tAnRunType = kTrain;
  int tNPartialAnalysis = 2;
  CentralityType tCentType = k0010;
  bool SaveImages = false;

  TString tDirectoryBase = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_%s/",tResultsDate.Data());
  TString tFileLocationBase = TString::Format("%sResults_cLamcKch_%s",tDirectoryBase.Data(),tResultsDate.Data());
  TString tFileLocationBaseMC = TString::Format("%sResults_cLamcKchMC_%s",tDirectoryBase.Data(),tResultsDate.Data());

//-------------------------------------------------------------------------------

  FitPairAnalysis* tLamKchP = new FitPairAnalysis(tFileLocationBase,tFileLocationBaseMC,kLamKchP,tCentType,tAnRunType,tNPartialAnalysis);
  FitPairAnalysis* tALamKchM = new FitPairAnalysis(tFileLocationBase,tFileLocationBaseMC,kALamKchM,tCentType,tAnRunType,tNPartialAnalysis);

  FitPairAnalysis* tLamKchM = new FitPairAnalysis(tFileLocationBase,tFileLocationBaseMC,kLamKchM,tCentType,tAnRunType,tNPartialAnalysis);
  FitPairAnalysis* tALamKchP = new FitPairAnalysis(tFileLocationBase,tFileLocationBaseMC,kALamKchP,tCentType,tAnRunType,tNPartialAnalysis);

//-------------------------------------------------------------------------------
  TString tCanvasName = TString("canLamKchPvsLamKchM0010");
  int tNx=2, tNy=1;


  double tXLow = -0.02;
  double tXHigh = 0.49;
  double tYLow = 0.85;
  double tYHigh = 1.03;
  CanvasPartition* tCanPart = new CanvasPartition(tCanvasName,tNx,tNy,tXLow,tXHigh,tYLow,tYHigh,0.12,0.05,0.13,0.05);

  int tMarkerStyle = 20;
  double tMarkerSize = 0.5;

  TH1* tCf_LamKchP = tLamKchP->GetKStarCfHeavy()->GetHeavyCfClone();
  TH1* tSysErrs_LamKchP = tLamKchP->GetCfwSysErrors();
    tSysErrs_LamKchP->SetFillStyle(0);

  TH1* tCf_ALamKchM = tALamKchM->GetKStarCfHeavy()->GetHeavyCfClone();
  TH1* tSysErrs_ALamKchM = tALamKchM->GetCfwSysErrors();
    tSysErrs_ALamKchM->SetFillStyle(0);

  TH1* tCf_LamKchM = tLamKchM->GetKStarCfHeavy()->GetHeavyCfClone();
  TH1* tSysErrs_LamKchM = tLamKchM->GetCfwSysErrors();
    tSysErrs_LamKchM->SetFillStyle(0);

  TH1* tCf_ALamKchP = tALamKchP->GetKStarCfHeavy()->GetHeavyCfClone();
  TH1* tSysErrs_ALamKchP = tALamKchP->GetCfwSysErrors();
    tSysErrs_ALamKchP->SetFillStyle(0);

//-------------------------------------------------------------------------------

  tCanPart->AddGraph(0,0,tCf_LamKchP,"",tMarkerStyle,2,tMarkerSize);
  tCanPart->AddGraph(0,0,tSysErrs_LamKchP,"",tMarkerStyle,2,tMarkerSize,"e2psame");

  tCanPart->AddGraph(0,0,tCf_LamKchM,"",tMarkerStyle,4,tMarkerSize);
  tCanPart->AddGraph(0,0,tSysErrs_LamKchM,"",tMarkerStyle,4,tMarkerSize,"e2psame");
/*
  TString tText1a = TString("#LambdaK+ vs #LambdaK-");
  TPaveText* tPaveText1a = tCanPart->SetupTPaveText(tText1a,0,0,0.75,0.85);
  tCanPart->AddPadPaveText(tPaveText1a,0,0);
*/
  TString tText1b = TString("0-10%");
  TPaveText* tPaveText1b = tCanPart->SetupTPaveText(tText1b,0,0,0.05,0.85);
  tCanPart->AddPadPaveText(tPaveText1b,0,0);

//-------------------------------------------------------------------------------

  tCanPart->AddGraph(1,0,tCf_ALamKchM,"",tMarkerStyle,2,tMarkerSize);
  tCanPart->AddGraph(1,0,tSysErrs_ALamKchM,"",tMarkerStyle,2,tMarkerSize,"e2psame");

  tCanPart->AddGraph(1,0,tCf_ALamKchP,"",tMarkerStyle,4,tMarkerSize);
  tCanPart->AddGraph(1,0,tSysErrs_ALamKchP,"",tMarkerStyle,4,tMarkerSize,"e2psame");
/*
  TString tText2a = TString("#bar{#Lambda}K- vs #bar{#Lambda}K+");
  TPaveText* tPaveText2a = tCanPart->SetupTPaveText(tText2a,1,0,0.75,0.85);
  tCanPart->AddPadPaveText(tPaveText2a,1,0);
*/
  TString tText2b = TString("0-10%");
  TPaveText* tPaveText2b = tCanPart->SetupTPaveText(tText2b,1,0,0.05,0.85);
  tCanPart->AddPadPaveText(tPaveText2b,1,0);

//-------------------------------------------------------------------------------


  tCanPart->SetDrawUnityLine(true);
  tCanPart->DrawAll();
  tCanPart->DrawXaxisTitle("k* (GeV/c)");
  tCanPart->DrawYaxisTitle("C(k*)",43,25,0.05,0.75);


  if(SaveImages) tCanPart->GetCanvas()->SaveAs(tDirectoryBase+tCanvasName+TString(".pdf"));
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
