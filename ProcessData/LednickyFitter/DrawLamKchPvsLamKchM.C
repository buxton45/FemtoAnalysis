#include "FitGenerator.h"
#include "TMarker.h"
#include "TArrow.h"
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
  bool SaveImages = true;

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
  double tXHigh = 0.29;
  double tYLow = 0.86;
  double tYHigh = 1.03;
  CanvasPartition* tCanPart = new CanvasPartition(tCanvasName,tNx,tNy,tXLow,tXHigh,tYLow,tYHigh,0.10,0.02,0.13,0.02);
  tCanPart->GetCanvas()->SetCanvasSize(1400,500);

  int tMarkerStyle = 20;
  double tMarkerSize = 1.0;

  int tColorLamKchP = kRed+1;
  int tColorLamKchM = kBlue+1;

  int tColorLamKchPTransparent = TColor::GetColorTransparent(tColorLamKchP,0.2);
  int tColorLamKchMTransparent = TColor::GetColorTransparent(tColorLamKchM,0.2);

  TH1* tCf_LamKchP = tLamKchP->GetKStarCfHeavy()->GetHeavyCfClone();
  TH1* tSysErrs_LamKchP = tLamKchP->GetCfwSysErrors();
    tSysErrs_LamKchP->SetFillColor(tColorLamKchPTransparent);
    tSysErrs_LamKchP->SetFillStyle(1000);
    tSysErrs_LamKchP->SetLineColor(0);
    tSysErrs_LamKchP->SetLineWidth(0);

  TH1* tCf_ALamKchM = tALamKchM->GetKStarCfHeavy()->GetHeavyCfClone();
  TH1* tSysErrs_ALamKchM = tALamKchM->GetCfwSysErrors();
    tSysErrs_ALamKchM->SetFillColor(tColorLamKchPTransparent);
    tSysErrs_ALamKchM->SetFillStyle(1000);
    tSysErrs_ALamKchM->SetLineColor(0);
    tSysErrs_ALamKchM->SetLineWidth(0);

  TH1* tCf_LamKchM = tLamKchM->GetKStarCfHeavy()->GetHeavyCfClone();
  TH1* tSysErrs_LamKchM = tLamKchM->GetCfwSysErrors();
    tSysErrs_LamKchM->SetFillColor(tColorLamKchMTransparent);
    tSysErrs_LamKchM->SetFillStyle(1000);
    tSysErrs_LamKchM->SetLineColor(0);
    tSysErrs_LamKchM->SetLineWidth(0);

  TH1* tCf_ALamKchP = tALamKchP->GetKStarCfHeavy()->GetHeavyCfClone();
  TH1* tSysErrs_ALamKchP = tALamKchP->GetCfwSysErrors();
    tSysErrs_ALamKchP->SetFillColor(tColorLamKchMTransparent);
    tSysErrs_ALamKchP->SetFillStyle(1000);
    tSysErrs_ALamKchP->SetLineColor(0);
    tSysErrs_ALamKchP->SetLineWidth(0);

//-------------------------------------------------------------------------------

  tCanPart->AddGraph(0,0,tCf_LamKchP,"",tMarkerStyle,tColorLamKchP,tMarkerSize,"ex0");
  tCanPart->AddGraph(0,0,tSysErrs_LamKchP,"",tMarkerStyle,tColorLamKchPTransparent,tMarkerSize,"e2psame");

  tCanPart->AddGraph(0,0,tCf_LamKchM,"",tMarkerStyle,tColorLamKchM,tMarkerSize,"ex0same");
  tCanPart->AddGraph(0,0,tSysErrs_LamKchM,"",tMarkerStyle,tColorLamKchMTransparent,tMarkerSize,"e2psame");
/*
  TString tText1a = TString("#LambdaK+ vs #LambdaK-");
  TPaveText* tPaveText1a = tCanPart->SetupTPaveText(tText1a,0,0,0.75,0.85);
  tCanPart->AddPadPaveText(tPaveText1a,0,0);
*/
  TString tText1b = TString("0-10%");
  TPaveText* tPaveText1b = tCanPart->SetupTPaveText(tText1b,0,0,0.085,0.885,0.075,0.075,63,35);
  tCanPart->AddPadPaveText(tPaveText1b,0,0);

//-------------------------------------------------------------------------------

  tCanPart->AddGraph(1,0,tCf_ALamKchM,"",tMarkerStyle,tColorLamKchP,tMarkerSize,"ex0");
  tCanPart->AddGraph(1,0,tSysErrs_ALamKchM,"",tMarkerStyle,tColorLamKchPTransparent,tMarkerSize,"e2psame");

  tCanPart->AddGraph(1,0,tCf_ALamKchP,"",tMarkerStyle,tColorLamKchM,tMarkerSize,"ex0same");
  tCanPart->AddGraph(1,0,tSysErrs_ALamKchP,"",tMarkerStyle,tColorLamKchMTransparent,tMarkerSize,"e2psame");
/*
  TString tText2a = TString("#bar{#Lambda}K- vs #bar{#Lambda}K+");
  TPaveText* tPaveText2a = tCanPart->SetupTPaveText(tText2a,1,0,0.75,0.85);
  tCanPart->AddPadPaveText(tPaveText2a,1,0);
*/
  TString tText2b = TString("0-10%");
  TPaveText* tPaveText2b = tCanPart->SetupTPaveText(tText2b,1,0,0.085,0.885,0.075,0.075,63,35);
  tCanPart->AddPadPaveText(tPaveText2b,1,0);

//-------------------------------------------------------------------------------


  tCanPart->SetDrawUnityLine(true);
  tCanPart->DrawAll();
  tCanPart->DrawXaxisTitle("#it{k}* (GeV/#it{c})",43,35,0.315,0.03);
  tCanPart->DrawYaxisTitle("C(#it{k}*)",43,45,0.05,0.75);

//-------------------------------------------------------------------------------
  TPad* tPadLeft = tCanPart->GetPad(0,0);
  TPad* tPadRight = tCanPart->GetPad(1,0);

  TArrow* ar1 = new TArrow(0.225,1.0075,0.26,1.015,0.005,"<|");
    ar1->SetLineWidth(2);

  TLatex* tex = new TLatex();
  TMarker *marker = new TMarker();
    marker->SetMarkerStyle(tMarkerStyle);
    marker->SetMarkerSize(1.5);

  tex->SetTextFont(42);
  tex->SetTextSize(0.088);
  tex->SetLineWidth(2);

  //-------------------------------------
  tPadLeft->cd();
  tex->DrawLatex(0.22,0.905,"#LambdaK+");
  marker->SetMarkerColor(tColorLamKchP);
  marker->DrawMarker(0.265,0.910);

  tex->DrawLatex(0.22,0.88,"#LambdaK-");
  marker->SetMarkerColor(tColorLamKchM);
  marker->DrawMarker(0.265,0.885);

  ar1->Draw();
  tex->DrawLatex(0.265,1.01,"#Omega");

  tex->DrawLatex(0.075,0.96, "ALICE Preliminary");
  tex->DrawLatex(0.075,0.94, "Pb-Pb #sqrt{#it{s}_{NN}} = 2.76 TeV");

  //-------------------------------------
  tPadRight->cd();
  tex->DrawLatex(0.22,0.905,"#bar{#Lambda}K-");
  marker->SetMarkerColor(tColorLamKchP);
  marker->DrawMarker(0.265,0.910);

  tex->DrawLatex(0.22,0.88,"#bar{#Lambda}K+");
  marker->SetMarkerColor(tColorLamKchM);
  marker->DrawMarker(0.265,0.885);

  ar1->Draw();
  tex->DrawLatex(0.265,1.01,"#bar{#Omega}");

//-------------------------------------------------------------------------------
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
