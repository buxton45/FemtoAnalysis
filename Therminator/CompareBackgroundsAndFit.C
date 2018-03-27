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
#include "TLatex.h"

#include "Therm3dCf.h"
class Therm3dCf;

#include "CfHeavy.h"
#include "CompareBackgrounds.h"


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
void PrintFitParams(TPad* aPad, TF1* aFit, double aTextSize=0.035)
{
  aPad->cd();

  TLatex* tTex = new TLatex();
  tTex->SetTextAlign(12);
  tTex->SetLineWidth(2);
  tTex->SetTextFont(42);
  tTex->SetTextSize(aTextSize);

  double tPar0, tPar1, tPar2, tPar3, tPar4, tPar5, tPar6;
  tPar0 = aFit->GetParameter(0);
  tPar1 = aFit->GetParameter(1);
  tPar2 = aFit->GetParameter(2);
  tPar3 = aFit->GetParameter(3);
  tPar4 = aFit->GetParameter(4);
  tPar5 = aFit->GetParameter(5);
  tPar6 = aFit->GetParameter(6);

  tTex->DrawLatex(0.15, 0.92, TString::Format("Bgd = %0.3f + %0.3fx + %0.3fx^{2} + ..." , tPar0, tPar1, tPar2));
  tTex->DrawLatex(0.30, 0.91, TString::Format("... + %0.3fx^{3} + %0.3fx^{4} + ..." , tPar3, tPar4));
  tTex->DrawLatex(0.30, 0.90, TString::Format("... + %0.3fx^{5} + %0.3fx^{6} + ..." , tPar5, tPar6));
}

//________________________________________________________________________________________________________________
void PrintInfo(TPad* aPad, AnalysisType aAnType, double aTextSize=0.04)
{
  aPad->cd();

  TLatex* tTex = new TLatex();
  tTex->SetTextAlign(12);
  tTex->SetLineWidth(2);
  tTex->SetTextFont(42);
  tTex->SetTextSize(aTextSize);

  tTex->DrawLatex(0.2, 1.05, "THERMINATOR");
  tTex->DrawLatex(1.4, 1.05, cAnalysisRootTags[aAnType]);
}
//________________________________________________________________________________________________________________
void PrintInfo(TPad* aPad, TString aOverallDescriptor, double aTextSize=0.04)
{
  aPad->cd();

  TLatex* tTex = new TLatex();
  tTex->SetTextAlign(12);
  tTex->SetLineWidth(2);
  tTex->SetTextFont(42);
  tTex->SetTextSize(aTextSize);

  tTex->DrawLatex(0.2, 1.05, "THERMINATOR");
  tTex->DrawLatex(1.2, 1.05, aOverallDescriptor);
}


//________________________________________________________________________________________________________________
TH1D* CombineConjugates(AnalysisType aAnType, TString aFileLocation, int aRebin, double aMinNorm, double aMaxNorm, int aMarkerStyle, int aColor)
{
  AnalysisType tConjAnType;
  if     (aAnType==kLamK0)    {tConjAnType=kALamK0;}
  else if(aAnType==kALamK0)   {tConjAnType=kLamK0;}
  else if(aAnType==kLamKchP)  {tConjAnType=kALamKchM;}
  else if(aAnType==kALamKchM) {tConjAnType=kLamKchP;}
  else if(aAnType==kLamKchM)  {tConjAnType=kALamKchP;}
  else if(aAnType==kALamKchP) {tConjAnType=kLamKchM;}
  else assert(0);

  //--------------------------------

  Therm3dCf *t3dCf1 = new Therm3dCf(aAnType, aFileLocation);
  t3dCf1->SetNormalizationRegion(aMinNorm, aMaxNorm);
  TH1D* tNum1 = t3dCf1->GetFullNum();
  TH1D* tDen1 = t3dCf1->GetFullDen();

  Therm3dCf *t3dCf2 = new Therm3dCf(tConjAnType, aFileLocation);
  t3dCf2->SetNormalizationRegion(aMinNorm, aMaxNorm);
  TH1D* tNum2 = t3dCf2->GetFullNum();
  TH1D* tDen2 = t3dCf2->GetFullDen();

  //--------------------------------

  CfLite* tCfLite1 = new CfLite(TString::Format("CfLite_%s", cAnalysisBaseTags[aAnType]), 
                                TString::Format("CfLite_%s", cAnalysisBaseTags[aAnType]),
                                tNum1, tDen1, aMinNorm, aMaxNorm);

  CfLite* tCfLite2 = new CfLite(TString::Format("CfLite_%s", cAnalysisBaseTags[tConjAnType]), 
                                TString::Format("CfLite_%s", cAnalysisBaseTags[tConjAnType]),
                                tNum2, tDen2, aMinNorm, aMaxNorm);

  //--------------------------------
  vector<CfLite*> tCfLiteCollection{tCfLite1, tCfLite2};
  CfHeavy* tCfHeavy = new CfHeavy(TString::Format("CfLite_%s_%s", cAnalysisBaseTags[aAnType], cAnalysisBaseTags[tConjAnType]),
                                  TString::Format("CfLite_%s_%s", cAnalysisBaseTags[aAnType], cAnalysisBaseTags[tConjAnType]), 
                                  tCfLiteCollection, aMinNorm, aMaxNorm);
  tCfHeavy->Rebin(aRebin);
  //--------------------------------
  TH1D* tReturnCf = (TH1D*)tCfHeavy->GetHeavyCfClone();
  SetStyleAndColor(tReturnCf, aMarkerStyle, aColor);

  return tReturnCf;
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
  aCf1->GetYaxis()->SetRangeUser(0.86, 1.07);

  aCf1->Draw();
  aCf2->Draw("same");
  aCf3->Draw("same");

  //---------------------------------------------------------------

  TLegend* tLeg = new TLegend(0.60, 0.15, 0.85, 0.40);
    tLeg->SetFillColor(0);
    tLeg->SetBorderSize(0);
    tLeg->SetTextAlign(22);

  tLeg->AddEntry(aCf1, aDescriptor1.Data());
  tLeg->AddEntry(aCf2, aDescriptor2.Data());
  tLeg->AddEntry(aCf3, aDescriptor3.Data());

  //---------------------------------------------------------------

  TF1 *tBgdFit, *tBgdFitDraw;
  if(aFitBgd)
  {
    int tPower = 6;
    tBgdFit = FitBackground(aCf3, tPower);
    if(aAnType==kLamKchM || aAnType==kALamKchP) //Want to draw fit function without gaping gap where Omega peak omitted
    {
      gRejectOmega=false;
      tBgdFitDraw = new TF1(tBgdFit->GetName()+TString("_Draw"), FitFunctionPolynomial, 0., 2., 7);
      for(int i=0; i<tBgdFit->GetNpar(); i++) tBgdFitDraw->SetParameter(i, tBgdFit->GetParameter(i));
    }
    else tBgdFitDraw = tBgdFit;

    tBgdFitDraw->SetLineColor(aCf3->GetLineColor());
    tBgdFitDraw->Draw("lsame");

    cout << endl << endl << "Bgd Chi2 = " << tBgdFit->GetChisquare() << endl;
    cout << "Background parameters:" << endl;
    for(int i=0; i<tBgdFit->GetNpar(); i++) cout << "Par[" << i << "] = " << tBgdFit->GetParameter(i) << endl;
    cout << "Single Line: " << endl;
    for(int i=0; i<tBgdFit->GetNpar(); i++) cout << tBgdFit->GetParameter(i) << ", ";
    cout << endl;
  }


  //---------------------------------------------------------------
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

  tLeg->AddEntry(tCfDiff, "1+Diff (B-C)");
  tLeg->AddEntry(tCfRatio, "Ratio (B/C)");

  //---------------------------------------------------------------

  aCf1->Draw("same");
  tLeg->Draw();


  TLine* tLine = new TLine(0, 1, 2, 1);
  tLine->SetLineColor(14);
  tLine->Draw();

  PrintInfo(aPad, aOverallDescriptor, 0.04);
  if(aFitBgd) PrintFitParams(aPad, tBgdFitDraw, 0.035);
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
  if(tAnType==kLamKchM || tAnType==kALamKchP) gRejectOmega=true;
  else gRejectOmega=false;

  bool bCombineConjugates = true;
  bool bCombineImpactParams = true;

  bool bSaveFigures = false;
  int tRebin=2;
  double tMinNorm = 0.32;
  double tMaxNorm = 0.40;

  int tImpactParam = 8;
  //-------------------------------------------------
  CentralityType tCentType=kMB;
  if(tImpactParam==3) tCentType=k0010;
  else if(tImpactParam==5 || tImpactParam==7) tCentType=k1030;
  else if(tImpactParam==8 || tImpactParam==9) tCentType=k3050;
  else assert(0);
  //-------------------------------------------------
  AnalysisType tConjAnType;
  if     (tAnType==kLamK0)    {tConjAnType=kALamK0;}
  else if(tAnType==kALamK0)   {tConjAnType=kLamK0;}
  else if(tAnType==kLamKchP)  {tConjAnType=kALamKchM;}
  else if(tAnType==kALamKchM) {tConjAnType=kLamKchP;}
  else if(tAnType==kLamKchM)  {tConjAnType=kALamKchP;}
  else if(tAnType==kALamKchP) {tConjAnType=kLamKchM;}
  else assert(0);
  //-------------------------------------------------

  TString tFileNameCfs1 = "CorrelationFunctions.root";
  TString tFileNameCfs2 = "CorrelationFunctions_RandomEPs.root";
  TString tFileNameCfs3 = "CorrelationFunctions_RandomEPs_NumWeight1.root";

  TString tSaveDir = "/home/jesse/Analysis/FemtoAnalysis/AnalysisNotes/Comments/Laura/20180117/Figures/";

  TString tDescriptor1 = "Cf w/o Bgd (A)";
  TString tDescriptor2 = "Cf w. Bgd (B)";
  TString tDescriptor3 = "Bgd (C)";

  TString tOverallDescriptor = cAnalysisRootTags[tAnType];
  if(bCombineConjugates) tOverallDescriptor += TString::Format(" & %s", cAnalysisRootTags[tConjAnType]);
  if(!bCombineImpactParams) tOverallDescriptor += TString::Format(" (b=%d)", tImpactParam);
  else tOverallDescriptor += TString::Format(" (%s)", cPrettyCentralityTags[tCentType]);

  int tMarkerStyle1 = 20;
  int tMarkerStyle2 = 24;
  int tMarkerStyle3 = 26;

  int tColor1 = 1;
  int tColor2 = 2;
  int tColor3 = 4;

  //--------------------------------------------
  TH1D *tCf1, *tCf2, *tCf3;
  if(!bCombineImpactParams)
  {
    tCf1 = (TH1D*)GetTHERMCf(tFileNameCfs1, tAnType, tImpactParam, bCombineConjugates, tRebin, tMinNorm, tMaxNorm, tMarkerStyle1, tColor1);
    tCf2 = (TH1D*)GetTHERMCf(tFileNameCfs2, tAnType, tImpactParam, bCombineConjugates, tRebin, tMinNorm, tMaxNorm, tMarkerStyle2, tColor2);
    tCf3 = (TH1D*)GetTHERMCf(tFileNameCfs3, tAnType, tImpactParam, bCombineConjugates, tRebin, tMinNorm, tMaxNorm, tMarkerStyle3, tColor3);
  }
  else
  {
    tCf1 = (TH1D*)GetCombinedTHERMCfs(tFileNameCfs1, tAnType, tCentType, bCombineConjugates, tRebin, tMinNorm, tMaxNorm, tMarkerStyle1, tColor1);
    tCf2 = (TH1D*)GetCombinedTHERMCfs(tFileNameCfs2, tAnType, tCentType, bCombineConjugates, tRebin, tMinNorm, tMaxNorm, tMarkerStyle2, tColor2);
    tCf3 = (TH1D*)GetCombinedTHERMCfs(tFileNameCfs3, tAnType, tCentType, bCombineConjugates, tRebin, tMinNorm, tMaxNorm, tMarkerStyle3, tColor3);
  }
//-------------------------------------------------------------------------------
  TString tCanCfsName = TString::Format("CompareBgds_Cfs_%s", cAnalysisBaseTags[tAnType]);
  if(bCombineConjugates) tCanCfsName += TString("wConj");
  if(!bCombineImpactParams) tCanCfsName += TString::Format("_b%d", tImpactParam);
  else tCanCfsName += TString(cCentralityTags[tCentType]);

  TCanvas* tCanCfs = new TCanvas(tCanCfsName, tCanCfsName);
  Draw1vs2vs3((TPad*)tCanCfs, tAnType, tCf1, tCf2, tCf3, tDescriptor1, tDescriptor2, tDescriptor3, tOverallDescriptor);

  if(bSaveFigures)
  {
    TString tSaveFileBase = tSaveDir + TString::Format("%s/", cAnalysisBaseTags[tAnType]);

    TString tSaveName_Cfs = TString::Format("%s%s_%s.eps", tSaveFileBase.Data(), tCanCfs->GetName(), cAnalysisBaseTags[tAnType]);
    tCanCfs->SaveAs(tSaveName_Cfs);
  }

//-------------------------------------------------------------------------------




//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
