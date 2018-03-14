#include <iostream>
#include <iomanip>
#include <math.h>
#include <vector>
#include <assert.h>

#include "TApplication.h"
#include "TStyle.h"
#include "TPaveText.h"
#include "TLegend.h"
#include "TF1.h"

#include "Therm3dCf.h"
class Therm3dCf;

//________________________________________________________________________________________________________________
void SetStyleAndColor(TH1* aHist, int aMarkerStyle, int aColor)
{
  aHist->SetLineColor(aColor);
  aHist->SetMarkerColor(aColor);
  aHist->SetMarkerStyle(aMarkerStyle);
  aHist->SetMarkerSize(0.5);
}

bool gRejectOmega=false;
//________________________________________________________________________________________________________________
double FitFunctionPolynomial(double *x, double *par)
{
  if(gRejectOmega && x[0]>0.19 && x[0]<0.23)
  {
    TF1::RejectPoint();
    return 0;
  }
  return par[0] + par[1]*x[0] + par[2]*pow(x[0],2) + par[3]*pow(x[0],3) + par[4]*pow(x[0],4) + par[5]*pow(x[0],5) + par[6]*pow(x[0],6);
}


//________________________________________________________________________________________________________________
TF1* FitBackground(TH1D* aBgdOnlyCf, int aPower=6, double aMinBgdFit=0., double aMaxBgdFit=2.)
{
  assert(aPower <= 6);  //Up to 6th order polynomial

  TString tFitName = "BgdFit";
  TF1* tBgdFit = new TF1(tFitName, FitFunctionPolynomial, 0., 2., 7);

  tBgdFit->SetParameter(0, 1.);
  tBgdFit->SetParameter(1, 0.);
  tBgdFit->SetParameter(2, 0.);
  tBgdFit->SetParameter(3, 0.);
  tBgdFit->SetParameter(4, 0.);
  tBgdFit->SetParameter(5, 0.);
  tBgdFit->SetParameter(6, 0.);

  if(aPower<6)
  {
    for(int i=6; i>aPower; i--)
    {
cout << "tBgdFit->FixParameter(" << i << ", 0.);" << endl;
      tBgdFit->FixParameter(i, 0.);
    }
  }

  aBgdOnlyCf->Fit(tFitName, "0q", "", aMinBgdFit, aMaxBgdFit);
  return tBgdFit;
}

//________________________________________________________________________________________________________________
void Draw1vs2vs3(TPad* aPad, AnalysisType aAnType, TH1D* aCf1, TH1D* aCf2, TH1D* aCf3, TString aDescriptor1, TString aDescriptor2, TString aDescriptor3, TString aOverallDescriptor, bool aFitBgd=true)
{
  aPad->cd();
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  if(aFitBgd) aDescriptor3 += TString(" (w. Fit)");
  //---------------------------------------------------------------
  aCf1->GetXaxis()->SetTitle("#it{k}* (GeV/#it{c})");
  aCf1->GetYaxis()->SetTitle("#it{C}(#it{k}*)");

  aCf1->GetXaxis()->SetRangeUser(0.,2.0);
  if(aOverallDescriptor.Contains("Cfs")) aCf1->GetYaxis()->SetRangeUser(0.86, 1.07);

  aCf1->Draw();
  aCf2->Draw("same");
  aCf3->Draw("same");

  //---------------------------------------------------------------

  TLegend* tLeg = new TLegend(0.60, 0.15, 0.85, 0.40);
    tLeg->SetFillColor(0);
    tLeg->SetBorderSize(0);
    tLeg->SetTextAlign(22);
  tLeg->SetHeader(TString::Format("%s %s Cfs", cAnalysisRootTags[aAnType], aOverallDescriptor.Data()));

  tLeg->AddEntry(aCf1, aDescriptor1.Data());
  tLeg->AddEntry(aCf2, aDescriptor2.Data());
  tLeg->AddEntry(aCf3, aDescriptor3.Data());

  //---------------------------------------------------------------

  if(aFitBgd)
  {
    int tPower = 6;
    TF1* tBgdFit = FitBackground(aCf3, tPower);
    TF1* tBgdFitDraw;
    if(aAnType==kLamKchM || aAnType==kALamKchP) //Want to draw fit function without gaping gap where Omega peak omitted
    {
      gRejectOmega=false;
      tBgdFitDraw = new TF1(tBgdFit->GetName()+TString("_Draw"), FitFunctionPolynomial, 0., 2., 7);
      for(int i=0; i<tBgdFit->GetNpar(); i++) tBgdFitDraw->SetParameter(i, tBgdFit->GetParameter(i));
    }
    else tBgdFitDraw = tBgdFit;

    tBgdFit->SetLineColor(aCf3->GetLineColor());
    tBgdFit->Draw("lsame");

    cout << endl << endl << "Bgd Chi2 = " << tBgdFit->GetChisquare() << endl;
    cout << "Background parameters:" << endl;
    for(int i=0; i<tBgdFit->GetNpar(); i++) cout << "Par[" << i << "] = " << tBgdFit->GetParameter(i) << endl;
  }


  //---------------------------------------------------------------
  if(aOverallDescriptor.Contains("Cfs")) 
  {
    TH1D* tCfDiff = (TH1D*)aCf2->Clone();
    tCfDiff->Add(aCf3, -1.);
    TH1D* tUnity = (TH1D*)aCf2->Clone();
    for(int i=1; i<=tUnity->GetNbinsX(); i++) tUnity->SetBinContent(i, 1.);
    tCfDiff->Add(tUnity, 1.);
    SetStyleAndColor(tCfDiff, 20, kGreen);
    tCfDiff->Draw("same");

    TH1D* tCfRatio = (TH1D*)aCf2->Clone();
    tCfRatio->Divide(aCf3);
    SetStyleAndColor(tCfRatio, 20, kMagenta);
    tCfRatio->Draw("same");

    tLeg->AddEntry(tCfDiff, "1+Diff");
    tLeg->AddEntry(tCfRatio, "Ratio");
  }
  //---------------------------------------------------------------

  aCf1->Draw("same");
  tLeg->Draw();

  if(aOverallDescriptor.Contains("Cfs")) 
  {
    TLine* tLine = new TLine(0, 1, 2, 1);
    tLine->SetLineColor(14);
    tLine->Draw();
  }
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

  AnalysisType tAnType = kLamKchM;
  if(tAnType==kLamKchM || tAnType==kALamKchP) gRejectOmega=true;
  else gRejectOmega=false;

  bool bDrawNumsAndDens = false;

  bool bSaveFigures = false;
  int tRebin=2;
  double tMinNorm = 0.32;
  double tMaxNorm = 0.40;

  int tImpactParam = 8;

  TString tDirectory = TString::Format("/home/jesse/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b%d/", tImpactParam);

  TString tFileLocationCfs1 = TString::Format("%sCorrelationFunctions.root", tDirectory.Data());
  TString tFileLocationCfs2 = TString::Format("%sCorrelationFunctions_RandomEPs.root", tDirectory.Data());
  TString tFileLocationCfs3 = TString::Format("%sCorrelationFunctions_RandomEPs_NumWeight1.root", tDirectory.Data());

  TString tSaveLocationBase = "/home/jesse/Analysis/Presentations/GroupMeetings/20171102/Figures/";

  TString tDescriptor1 = "Cf w. no Bgd";
  TString tDescriptor2 = "Cf w. Bgd";
  TString tDescriptor3 = "Bgd";
  TString tOverallDescriptorA = "Cfs";
  TString tOverallDescriptorB = "Nums";
  TString tOverallDescriptorC = "Dens";

  int tMarkerStyle1 = 20;
  int tMarkerStyle2 = 24;
  int tMarkerStyle3 = 26;

  int tColor1 = 1;
  int tColor2 = 2;
  int tColor3 = 4;

  //--------------------------------------------

  Therm3dCf *t3dCf1 = new Therm3dCf(tAnType, tFileLocationCfs1, tRebin);
  t3dCf1->SetNormalizationRegion(tMinNorm, tMaxNorm);
  TH1D* tCf1 = t3dCf1->GetFullCf(tMarkerStyle1, tColor1);

  Therm3dCf *t3dCf2 = new Therm3dCf(tAnType, tFileLocationCfs2, tRebin);
  t3dCf2->SetNormalizationRegion(tMinNorm, tMaxNorm);
  TH1D* tCf2 = t3dCf2->GetFullCf(tMarkerStyle2, tColor2);

  Therm3dCf *t3dCf3 = new Therm3dCf(tAnType, tFileLocationCfs3, tRebin);
  t3dCf3->SetNormalizationRegion(tMinNorm, tMaxNorm);
  TH1D* tCf3 = t3dCf3->GetFullCf(tMarkerStyle3, tColor3);


//-------------------------------------------------------------------------------

  TCanvas* tCanCfs = new TCanvas("CompareBgds_Cfs", "CompareBgds_Cfs");
  Draw1vs2vs3((TPad*)tCanCfs, tAnType, tCf1, tCf2, tCf3, tDescriptor1, tDescriptor2, tDescriptor3, tOverallDescriptorA);

  if(bSaveFigures)
  {
    TString tSaveFileBase = tSaveLocationBase + TString::Format("%s/", cAnalysisBaseTags[tAnType]);

    TString tSaveName_Cfs = TString::Format("%s%s_%s.eps", tSaveFileBase.Data(), tCanCfs->GetName(), cAnalysisBaseTags[tAnType]);
    tCanCfs->SaveAs(tSaveName_Cfs);
  }


//-------------------------------------------------------------------------------
  if(bDrawNumsAndDens)
  {
    TH1D* tNum1 = t3dCf1->GetFullNum();
      SetStyleAndColor(tNum1, tMarkerStyle1, tColor1);
    TH1D* tDen1 = t3dCf1->GetFullDen();
      SetStyleAndColor(tDen1, tMarkerStyle1, tColor1);

    TH1D* tNum2 = t3dCf2->GetFullNum();
      SetStyleAndColor(tNum2, tMarkerStyle2, tColor2);
    TH1D* tDen2 = t3dCf2->GetFullDen();
      SetStyleAndColor(tDen2, tMarkerStyle2, tColor2);

    TH1D* tNum3 = t3dCf3->GetFullNum();
      SetStyleAndColor(tNum3, tMarkerStyle3, tColor3);
    TH1D* tDen3 = t3dCf3->GetFullDen();
      SetStyleAndColor(tDen3, tMarkerStyle3, tColor3);

    TCanvas* tCanNums = new TCanvas("CompareBgds_Nums", "CompareBgds_Nums");
    Draw1vs2vs3((TPad*)tCanNums, tAnType, tNum1, tNum2, tNum3, tDescriptor1, tDescriptor2, tDescriptor3, tOverallDescriptorB);

    TCanvas* tCanDens = new TCanvas("CompareBgds_Dens", "CompareBgds_Dens");
    Draw1vs2vs3((TPad*)tCanDens, tAnType, tDen1, tDen2, tDen3, tDescriptor1, tDescriptor2, tDescriptor3, tOverallDescriptorC);

    if(bSaveFigures)
    {
      TString tSaveFileBase = tSaveLocationBase + TString::Format("%s/", cAnalysisBaseTags[tAnType]);

      TString tSaveName_Nums = TString::Format("%s%s_%s.eps", tSaveFileBase.Data(), tCanNums->GetName(), cAnalysisBaseTags[tAnType]);
      tCanNums->SaveAs(tSaveName_Nums);

      TString tSaveName_Dens = TString::Format("%s%s_%s.eps", tSaveFileBase.Data(), tCanDens->GetName(), cAnalysisBaseTags[tAnType]);
      tCanDens->SaveAs(tSaveName_Dens);

    }
  }




//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
