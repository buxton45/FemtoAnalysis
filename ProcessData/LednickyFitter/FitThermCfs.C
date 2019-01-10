#include <iostream>
#include <iomanip>
#include <math.h>
#include <vector>
#include <assert.h>

#include "TApplication.h"
#include "TStyle.h"
#include "TPaveText.h"
#include "TLegend.h"
#include "TH3.h"

#include "ThermCf.h"
#include "FitPartialAnalysis.h"

bool gRejectPoints=false;
double tRejectOmegaLow = 0.19;
double tRejectOmegaHigh = 0.24;

//________________________________________________________________________________________________________________
TH1* GetThermHist1d(TString aFileLocation, TString aHistName)
{
  TFile *tFile = TFile::Open(aFileLocation);

  TH1 *tReturnHist = (TH1*)tFile->Get(aHistName);
  TH1 *tReturnHistClone = (TH1*)tReturnHist->Clone();
  tReturnHistClone->SetDirectory(0);

  tFile->Close();
  delete tFile;

  return tReturnHistClone;
}

//________________________________________________________________________________________________________________
TH3* GetThermHist3d(TString aFileLocation, TString aHistName)
{
  TFile *tFile = TFile::Open(aFileLocation);

  TH3 *tReturnHist = (TH3*)tFile->Get(aHistName);
  TH3 *tReturnHistClone = (TH3*)tReturnHist->Clone();
  tReturnHistClone->SetDirectory(0);

  tFile->Close();
  delete tFile;

  return tReturnHistClone;
}

//________________________________________________________________________________________________________________
double LednickyEqWithNorm(double *x, double *par)
{
  if(gRejectPoints && x[0]>tRejectOmegaLow && x[0]<tRejectOmegaHigh)
  {
    TF1::RejectPoint();
    return 0;
  }

  double tUnNormCf = FitPartialAnalysis::LednickyEq(x, par);
  double tNormCf = par[5]*tUnNormCf;
  return tNormCf;
}

//________________________________________________________________________________________________________________
double FitFunctionGaussian(double *x, double *par)
{
  //4 parameters
  return par[0]*exp(-0.5*(pow((x[0]-par[1])/(sqrt(2)*par[2]),2.0))) + par[3];
}

//________________________________________________________________________________________________________________
TF1* FitwGauss(TH1* aHist, double aMinFit=0., double aMaxFit=50.)
{
  TString tFitName = TString::Format("%s_FitGauss", aHist->GetName());
//  TF1* tReturnFunction = new TF1(tFitName, BackgroundFitter::FitFunctionGaussian, aMinFit, aMaxFit, 4);  //No sqrt(2) with sigma
  TF1* tReturnFunction = new TF1(tFitName, FitFunctionGaussian, -50., 50., 4);

  double tMaxVal = aHist->GetMaximum();
  double tMaxPos = aHist->GetBinCenter(aHist->GetMaximumBin());
  int tApproxSigBin = aHist->FindLastBinAbove(tMaxVal/2.);
  double tApproxSig = aHist->GetBinCenter(tApproxSigBin);

  tReturnFunction->SetParameter(0, tMaxVal);
  tReturnFunction->SetParLimits(0, 0., 1.5*tMaxVal);

  tReturnFunction->SetParameter(1, tMaxPos);
//  tReturnFunction->SetParLimits(1, 0., 50.);
//  tReturnFunction->FixParameter(1, 0.);

  tReturnFunction->SetParameter(2, tApproxSig);
  tReturnFunction->SetParLimits(2, 0., 50.);

  tReturnFunction->FixParameter(3, 0.);

  aHist->Fit(tFitName, "0", "", aMinFit, aMaxFit);
  return tReturnFunction;
}


//________________________________________________________________________________________________________________
void DrawHistwGaussFit(TPad* aPad, TH1* aHist, double aGaussFitMin, double aGaussFitMax, TString aMuName="#mu_{Out}", TString aSigmaName="R_{Out}")
{
  TF1* tGaussFit = FitwGauss(aHist, aGaussFitMin, aGaussFitMax);

  aPad->cd();
  aHist->Draw();
  tGaussFit->Draw("same");

  //----- Draw lines to show fit range -----
  TLine* tLineMin = new TLine(aGaussFitMin, 0., aGaussFitMin, 0.25*aHist->GetMaximum());
  TLine* tLineMax = new TLine(aGaussFitMax, 0., aGaussFitMax, 0.25*aHist->GetMaximum());

  tLineMin->SetLineColor(TColor::GetColorTransparent(kRed,0.75));
  tLineMin->SetLineStyle(2);
  tLineMin->Draw();

  tLineMax->SetLineColor(TColor::GetColorTransparent(kRed,0.75));
  tLineMax->SetLineStyle(2);
  tLineMax->Draw();
  //----------------------------------------

  TPaveText* tText = new TPaveText(0.15, 0.50, 0.40, 0.85, "NDC");
    tText->SetFillColor(0);
    tText->SetBorderSize(0);
    tText->SetTextColor(kRed);
/*
    tText->AddText("[0]*exp#left(- #frac{#left(x-[1]#right)^{2}}{4[2]^{2}}#right)");
    tText->AddText(TString::Format("[1] = %0.3e", tGaussFit->GetParameter(1)));
    tText->AddText(TString::Format("[2] = %0.3e", tGaussFit->GetParameter(2)));
*/
    tText->AddText(TString::Format("N*exp#left(- #frac{#left(x-%s#right)^{2}}{4%s^{2}}#right)", aMuName.Data(), aSigmaName.Data()));
    tText->AddText("");
    tText->AddText(TString::Format("%s = %0.3e", aMuName.Data(), tGaussFit->GetParameter(1)));
    tText->AddText(TString::Format("%s = %0.3e",   aSigmaName.Data(), tGaussFit->GetParameter(2)));
    tText->Draw();

}

//________________________________________________________________________________________________________________
void Draw1DSourceProjwFit(TPad* aPad, TH3* a3DoslHist, TString aComponent, double aGaussFitMin=-20., double aGaussFitMax=20., double aProjLow=-100, double aProjHigh=-100)
{
  assert(aComponent.EqualTo("Out") || aComponent.EqualTo("Side") || aComponent.EqualTo("Long"));

  int tBinProjLow, tBinProjHigh;
  if(aProjLow==-100. && aProjHigh==-100.)
  {
    tBinProjLow=-1;
    tBinProjHigh=-1;
  }
  else
  {
    tBinProjLow = a3DoslHist->GetXaxis()->FindBin(aProjLow);
    tBinProjHigh = a3DoslHist->GetXaxis()->FindBin(aProjHigh);
  }

  //-----------------------------------------------------------

  TH1D* t1DSource;
  if(aComponent.EqualTo("Out"))
  {
    t1DSource = a3DoslHist->ProjectionX("Out", tBinProjLow, tBinProjHigh, tBinProjLow, tBinProjHigh);
      t1DSource->SetTitle("PairSource_Out");
      t1DSource->GetXaxis()->SetTitle("R_{Out}(fm)");
  }
  else if(aComponent.EqualTo("Side"))
  {
    t1DSource = a3DoslHist->ProjectionY("Side", tBinProjLow, tBinProjHigh, tBinProjLow, tBinProjHigh);
      t1DSource->SetTitle("PairSource_Side");
      t1DSource->GetXaxis()->SetTitle("R_{Side}(fm)");
  }
  else if(aComponent.EqualTo("Long"))
  {
    t1DSource = a3DoslHist->ProjectionZ("Long", tBinProjLow, tBinProjHigh, tBinProjLow, tBinProjHigh);
    t1DSource->SetTitle("PairSource_Long");
    t1DSource->GetXaxis()->SetTitle("R_{Long}(fm)");
  }
  else assert(0);

  t1DSource->SetMarkerStyle(20);
  t1DSource->SetMarkerSize(0.75);
  t1DSource->SetMarkerColor(kBlack);

  //-----------------------------------------------------------
  TString tMuName = TString::Format("#mu_{%s}", aComponent.Data());
  TString tSigmaName = TString::Format("R_{%s}", aComponent.Data());
  DrawHistwGaussFit(aPad, t1DSource, aGaussFitMin, aGaussFitMax, tMuName, tSigmaName);
}

//________________________________________________________________________________________________________________
void Draw1DCfwFit(TPad* aPad, AnalysisType aAnType, TH1* aThermCf, double aFitMax=0.3, bool aFixLambda=false)
{
  double tRef0, tImf0, td0;
  if(aAnType == kLamKchP || aAnType == kKchPKchP || aAnType == kK0K0 || aAnType == kLamLam)
  {
    tRef0 = -1.16;
    tImf0 = 0.51;
    td0 = 1.08;
  }
  else if(aAnType == kLamKchM)
  {
    tRef0 = 0.41;
    tImf0 = 0.47;
    td0 = -4.89;
  }
  else if(aAnType == kLamK0)
  {
    tRef0 = -0.41;
    tImf0 = 0.20;
    td0 = 2.08;
  }
  else assert(0);

  int tNFitParams = 5;
  TString tFitName = TString::Format("tFitFcn_%s", cAnalysisBaseTags[aAnType]);
  TF1* tFitFcn = new TF1(tFitName, LednickyEqWithNorm,0.,0.5,tNFitParams+1);
    if(aFixLambda) tFitFcn->FixParameter(0, 1.);
    else           tFitFcn->SetParameter(0, 1.);

    tFitFcn->SetParameter(1, 5.);

    tFitFcn->FixParameter(2, tRef0);
    tFitFcn->FixParameter(3, tImf0);
    tFitFcn->FixParameter(4, td0);

    tFitFcn->SetParameter(5, 1.);

  aThermCf->Fit(tFitName, "0", "", 0.0, aFitMax);

  //-----------------------------------------------------------
  aThermCf->GetXaxis()->SetRangeUser(0., 0.5);
  aThermCf->GetYaxis()->SetRangeUser(0.80, 1.02);

  aPad->cd();

  aThermCf->Draw();
  tFitFcn->Draw("same");

  TPaveText* tText = new TPaveText(0.50, 0.15, 0.85, 0.60, "NDC");
    tText->SetFillColor(0);
    tText->SetBorderSize(0);
    tText->SetTextColor(kBlack);
    tText->SetTextAlign(32);
    tText->AddText("Lednicky Fit");
    tText->AddText(TString::Format("#lambda = %0.3f", tFitFcn->GetParameter(0)));
    tText->AddText(TString::Format("R = %0.3f", tFitFcn->GetParameter(1)));
    tText->AddText(TString::Format("Re[f0] = %0.3f", tFitFcn->GetParameter(2)));
    tText->AddText(TString::Format("Im[f0] = %0.3f", tFitFcn->GetParameter(3)));
    tText->AddText(TString::Format("d0 = %0.3f", tFitFcn->GetParameter(4)));
    tText->AddText(TString::Format("Norm = %0.3f", tFitFcn->GetParameter(5)));

    tText->Draw();
}



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
  AnalysisType tAnType = kLamKchP;
  if(tAnType==kLamKchM) gRejectPoints=true;

  bool bCombineConjugates = true;
  bool bDrawDeltaT = true;
  bool bDraw2DHists = false;
  bool bSaveFigures = false;
  TString tSaveDir = "/home/jesse/Analysis/Presentations/GroupMeetings/20190108/Figures/";

  int tRebin=1;
  double tMinNorm = /*0.80*//*0.80*/0.32;
  double tMaxNorm = /*0.99*//*0.99*/0.40;

  int tImpactParam = 2;
  TString aCfDescriptor = "Full";
//  TString aCfDescriptor = "PrimaryOnly";

//  TString tFileNameBase = "CorrelationFunctions_wOtherPairs";
//  TString tFileNameBase = "CorrelationFunctions_wOtherPairs_DrawRStarFromGaussian_LamKchPMuOut3_LamKchMMuOut6";
//  TString tFileNameBase = "CorrelationFunctions_wOtherPairs_DrawRStarFromGaussian_cLamcKchMuOut6_cLamK0MuOut1";

//  TString tFileNameBase = "CorrelationFunctions_wOtherPairs_DrawRStarFromGaussian_BuildCfYlm_cLamcKchMuOut3_cLamK0MuOut3_KchPKchPR538";
  TString tFileNameBase = "CorrelationFunctions_wOtherPairs_BuildCfYlm";
//  TString tFileNameBase = "CorrelationFunctions_DrawRStarFromGaussian_BuildCfYlm_cLamcKchMuOut1_cLamK0MuOut1";
//  TString tFileNameBase = "CorrelationFunctions_DrawRStarFromGaussian_BuildCfYlm_cLamcKchMuOut6_cLamK0MuOut6";


  TString tFileNameModifier = "";
//  TString tFileNameModifier = "_WeightParentsInteraction";
//  TString tFileNameModifier = "_WeightParentsInteraction_OnlyWeightLongDecayParents";
//  TString tFileNameModifier = "_WeightParentsInteraction_NoCharged";

  //--------------------------------------------

  TString tFileName = TString::Format("%s%s.root", tFileNameBase.Data(), tFileNameModifier.Data());

  TString tFileDir = TString::Format("/home/jesse/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b%d/", tImpactParam);
  TString tFileLocation = TString::Format("%s%s", tFileDir.Data(), tFileName.Data());

  TString t3dDescriptor;
  if(aCfDescriptor.EqualTo("Full")) t3dDescriptor = TString("");
  else t3dDescriptor=aCfDescriptor;

  TString tHistName3d = TString::Format("PairSource3d_osl%s%s", t3dDescriptor.Data(), cAnalysisBaseTags[tAnType]);
//  TString tHistName3d = TString::Format("TrueRosl%s%s", t3dDescriptor.Data(), cAnalysisBaseTags[tAnType]);
//  TString tHistName3d = TString::Format("SimpleRosl%s%s", t3dDescriptor.Data(), cAnalysisBaseTags[tAnType]);

  if(tAnType==kKchPKchP || tAnType==kK0K0 || tAnType==kLamLam) bCombineConjugates = false;
  if(tFileNameBase.Contains("PairOnly")) bCombineConjugates = false;

  //--------------------------------------------

  //ThermCf already knows the default directory, so it only needs the name of the file, not the complete path
  TH1* tThermCf = ThermCf::GetThermCf(tFileName, aCfDescriptor, tAnType, tImpactParam, bCombineConjugates, kMe, tRebin, tMinNorm, tMaxNorm);
  TH3* tTest3d = GetThermHist3d(tFileLocation, tHistName3d);

  double tGaussFitMin = -20.;
  double tGaussFitMax = 20.;

  double tProjLow = -100.;
  double tProjHigh = -100.;

  bool tFixLambdaInFit = true;
  double tKStarFitMax = 0.3;

  TString tCanCfwSourceName = TString::Format("CanCfwSource_%s_%s", aCfDescriptor.Data(), cAnalysisBaseTags[tAnType]);
  TCanvas* tCanCfwSource = new TCanvas(tCanCfwSourceName, tCanCfwSourceName);
  tCanCfwSource->Divide(2,2);


  Draw1DCfwFit((TPad*)tCanCfwSource->cd(1), tAnType, tThermCf, tKStarFitMax, tFixLambdaInFit);

  Draw1DSourceProjwFit((TPad*)tCanCfwSource->cd(2), tTest3d, "Out", tGaussFitMin, tGaussFitMax, tProjLow, tProjHigh);
  Draw1DSourceProjwFit((TPad*)tCanCfwSource->cd(3), tTest3d, "Side", tGaussFitMin, tGaussFitMax, tProjLow, tProjHigh);
  Draw1DSourceProjwFit((TPad*)tCanCfwSource->cd(4), tTest3d, "Long", tGaussFitMin, tGaussFitMax, tProjLow, tProjHigh);

//  if(bSaveFigures) tCanCfwSource->SaveAs(TString::Format("%s%s_%s_%s.eps", tSaveDir.Data(), tFileNameBase.Data(), aCfDescriptor.Data(), cAnalysisBaseTags[tAnType]));
  if(bSaveFigures) tCanCfwSource->SaveAs(TString::Format("%s%s_FromFile%s.eps", tSaveDir.Data(), tCanCfwSourceName.Data(), tFileNameBase.Data()));
  //-------------------------------------------------------------------------------

  if(tFileNameBase.Contains("DrawRStarFromGaussian")) bDrawDeltaT=false;
  if(bDrawDeltaT)
  {
    TString tHistNameDeltaT;
    if(aCfDescriptor.EqualTo("Full")) tHistNameDeltaT = TString::Format("PairDeltaT_inPRF%s", cAnalysisBaseTags[tAnType]);
    else                              tHistNameDeltaT = TString::Format("PairDeltaT_inPRF%s%s", aCfDescriptor.Data(), cAnalysisBaseTags[tAnType]);

    TH1* tDeltaTHist = GetThermHist1d(tFileLocation, tHistNameDeltaT);

    TString tCanDeltaTName = TString::Format("CanDeltaT_%s_%s", aCfDescriptor.Data(), cAnalysisBaseTags[tAnType]);
    TCanvas* tCanDeltaT = new TCanvas(tCanDeltaTName, tCanDeltaTName); 
    tCanDeltaT->cd();

    tDeltaTHist->GetXaxis()->SetTitle("#Deltat_{PRF}");

    TString tMuName = "#mu_{#Deltat}";
    TString tSigmaName = "#Deltat";

    DrawHistwGaussFit((TPad*)tCanDeltaT->cd(), tDeltaTHist, -20., 20., tMuName, tSigmaName);
    if(bSaveFigures) tCanDeltaT->SaveAs(TString::Format("%s%s_FromFile%s.eps", tSaveDir.Data(), tCanDeltaTName.Data(), tFileNameBase.Data()));
  }

  //-------------------------------------------------------------------------------

  if(bDraw2DHists)
  {
    TH2D* tSourceSO = (TH2D*)tTest3d->Project3D("yx");
      tSourceSO->SetTitle("Side(y) vs. Out(x)");
      tSourceSO->GetYaxis()->SetTitle("Side");
      tSourceSO->GetXaxis()->SetTitle("Out");

    TH2D* tSourceLO = (TH2D*)tTest3d->Project3D("zx");
      tSourceLO->SetTitle("Long(y) vs. Out(x)");
      tSourceLO->GetYaxis()->SetTitle("Long");
      tSourceLO->GetXaxis()->SetTitle("Out");


    TH2D* tSourceLS = (TH2D*)tTest3d->Project3D("zy");
      tSourceLS->SetTitle("Long(y) vs. Side(x)");
      tSourceLS->GetYaxis()->SetTitle("Long");
      tSourceLS->GetXaxis()->SetTitle("Side");

    TCanvas* tCan2DHists = new TCanvas("tCan2DHists", "tCan2DHists", 500, 1500);
    tCan2DHists->Divide(1,3);

    tCan2DHists->cd(1);
    tSourceSO->Draw("colz");

    tCan2DHists->cd(2);
    tSourceLO->Draw("colz");

    tCan2DHists->cd(3);
    tSourceLS->Draw("colz");
  }



//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
