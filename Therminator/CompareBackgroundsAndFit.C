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
TF1* FitBackground(TH1D* aBgdOnlyCf, int aPower=6, double aMinBgdFit=0., double aMaxBgdFit=3.)
{
  assert(aPower <= 6);  //Up to 6th order polynomial

  TString tFitName = "BgdFit";
  TF1* tBgdFit = new TF1(tFitName, FitFunctionPolynomial, 0., aMaxBgdFit, 7);

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
CentralityType GetCentralityType(int aImpactParam)
{
  CentralityType tCentType=kMB;
  if(aImpactParam==3) tCentType=k0010;
  else if(aImpactParam==5 || aImpactParam==7) tCentType=k1030;
  else if(aImpactParam==8 || aImpactParam==9) tCentType=k3050;
  else assert(0);

  return tCentType;
}

//________________________________________________________________________________________________________________
AnalysisType GetConjAnType(AnalysisType aAnType)
{
  AnalysisType tConjAnType;
  if     (aAnType==kLamK0)    {tConjAnType=kALamK0;}
  else if(aAnType==kALamK0)   {tConjAnType=kLamK0;}
  else if(aAnType==kLamKchP)  {tConjAnType=kALamKchM;}
  else if(aAnType==kALamKchM) {tConjAnType=kLamKchP;}
  else if(aAnType==kLamKchM)  {tConjAnType=kALamKchP;}
  else if(aAnType==kALamKchP) {tConjAnType=kLamKchM;}
  else assert(0);

  return tConjAnType;
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
    for(int i=0; i<tBgdFit->GetNpar(); i++) cout << TString::Format("% 11.8f, ", tBgdFit->GetParameter(i));
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
TCanvas* CompareCfWithAndWithoutBgd(TString aCfDescriptor, AnalysisType aAnType, int aImpactParam, bool aCombineConjugates, bool aCombineImpactParams, bool aUseAdamEvents, int aRebin, double aMinNorm, double aMaxNorm)
{
  if(aAnType==kLamKchM || aAnType==kALamKchP) gRejectOmega=true;
  else gRejectOmega=false;
  //-------------------------------------------------
  CentralityType tCentType=kMB;
  if(aCombineImpactParams) tCentType = GetCentralityType(aImpactParam);
  //-------------------------------------------------
  AnalysisType tConjAnType = GetConjAnType(aAnType);
  //-------------------------------------------------
  TString tFileNameCfs1 = "CorrelationFunctions.root";
  TString tFileNameCfs2 = "CorrelationFunctions_RandomEPs.root";
  TString tFileNameCfs3 = "CorrelationFunctions_RandomEPs_NumWeight1.root";

  TString tDescriptor1 = "Cf w/o Bgd (A)";
  TString tDescriptor2 = "Cf w. Bgd (B)";
  TString tDescriptor3 = "Bgd (C)";

  TString tOverallDescriptor = cAnalysisRootTags[aAnType];
  if(aCombineConjugates) tOverallDescriptor += TString::Format(" & %s", cAnalysisRootTags[tConjAnType]);
  if(!aCombineImpactParams) tOverallDescriptor += TString::Format(" (b=%d)", aImpactParam);
  else tOverallDescriptor += TString::Format(" (%s)", cPrettyCentralityTags[tCentType]);

  int tMarkerStyle1 = 20;
  int tMarkerStyle2 = 24;
  int tMarkerStyle3 = 26;

  int tColor1 = 1;
  int tColor2 = 2;
  int tColor3 = 4;

  //--------------------------------------------
  TH1D *tCf1, *tCf2, *tCf3;
  if(!aCombineImpactParams)
  {
    tCf1 = (TH1D*)GetTHERMCf(tFileNameCfs1, aCfDescriptor, aAnType, aImpactParam, aCombineConjugates, aUseAdamEvents, aRebin, aMinNorm, aMaxNorm, tMarkerStyle1, tColor1);
    tCf2 = (TH1D*)GetTHERMCf(tFileNameCfs2, aCfDescriptor, aAnType, aImpactParam, aCombineConjugates, aUseAdamEvents, aRebin, aMinNorm, aMaxNorm, tMarkerStyle2, tColor2);
    tCf3 = (TH1D*)GetTHERMCf(tFileNameCfs3, aCfDescriptor, aAnType, aImpactParam, aCombineConjugates, aUseAdamEvents, aRebin, aMinNorm, aMaxNorm, tMarkerStyle3, tColor3);
  }
  else
  {
    tCf1 = (TH1D*)GetCombinedTHERMCfs(tFileNameCfs1, aCfDescriptor, aAnType, tCentType, aCombineConjugates, aUseAdamEvents, aRebin, aMinNorm, aMaxNorm, tMarkerStyle1, tColor1);
    tCf2 = (TH1D*)GetCombinedTHERMCfs(tFileNameCfs2, aCfDescriptor, aAnType, tCentType, aCombineConjugates, aUseAdamEvents, aRebin, aMinNorm, aMaxNorm, tMarkerStyle2, tColor2);
    tCf3 = (TH1D*)GetCombinedTHERMCfs(tFileNameCfs3, aCfDescriptor, aAnType, tCentType, aCombineConjugates, aUseAdamEvents, aRebin, aMinNorm, aMaxNorm, tMarkerStyle3, tColor3);
  }
//-------------------------------------------------------------------------------
  TString tCanCfsName;
  tCanCfsName = TString::Format("CompareBgds_%s_%s", aCfDescriptor.Data(), cAnalysisBaseTags[aAnType]);

  if(aCombineConjugates) tCanCfsName += TString("wConj");
  if(!aCombineImpactParams) tCanCfsName += TString::Format("_b%d", aImpactParam);
  else tCanCfsName += TString(cCentralityTags[tCentType]);

  TCanvas* tCanCfs = new TCanvas(tCanCfsName, tCanCfsName);
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);
  Draw1vs2vs3((TPad*)tCanCfs, aAnType, tCf1, tCf2, tCf3, tDescriptor1, tDescriptor2, tDescriptor3, tOverallDescriptor);

  return tCanCfs;
}


//________________________________________________________________________________________________________________
TCanvas* DrawBgdwFit(TString aCfDescriptor, TString aFileNameCfs, AnalysisType aAnType, int aImpactParam, bool aCombineConjugates, bool aCombineImpactParams, bool aUseAdamEvents, int aRebin, double aMinNorm, double aMaxNorm)
{
  if(aAnType==kLamKchM || aAnType==kALamKchP) gRejectOmega=true;
  else gRejectOmega=false;

  //-------------------------------------------------
  CentralityType tCentType=kMB;
  if(aCombineImpactParams) tCentType = GetCentralityType(aImpactParam);
  //-------------------------------------------------
  AnalysisType tConjAnType = GetConjAnType(aAnType);
  //-------------------------------------------------
  TString tDescriptor = "Bgd (w. Fit)";

  TString tOverallDescriptor = cAnalysisRootTags[aAnType];
  if(aCombineConjugates) tOverallDescriptor += TString::Format(" & %s", cAnalysisRootTags[tConjAnType]);
  if(!aCombineImpactParams) tOverallDescriptor += TString::Format(" (b=%d)", aImpactParam);
  else tOverallDescriptor += TString::Format(" (%s)", cPrettyCentralityTags[tCentType]);

  int tMarkerStyle = 26;
  int tColor = 4;

  //--------------------------------------------
  TH1D *tCf;
  if(!aCombineImpactParams) tCf = (TH1D*)GetTHERMCf(aFileNameCfs, aCfDescriptor, aAnType, aImpactParam, aCombineConjugates, aUseAdamEvents, aRebin, aMinNorm, aMaxNorm, tMarkerStyle, tColor);
  else tCf = (TH1D*)GetCombinedTHERMCfs(aFileNameCfs, aCfDescriptor, aAnType, tCentType, aCombineConjugates, aUseAdamEvents, aRebin, aMinNorm, aMaxNorm, tMarkerStyle, tColor);
//-------------------------------------------------------------------------------
  TString tCanBgdwFitName = "BgdwFitOnly";

  if(aFileNameCfs.Contains("_RandomEPs_NumWeight1")) tCanBgdwFitName += TString("_RandomEPs_NumWeight1");
  else if(aFileNameCfs.Contains("_RandomEPs")) tCanBgdwFitName += TString("_RandomEPs");
  else tCanBgdwFitName += TString("");

  tCanBgdwFitName += TString::Format("_%s_", aCfDescriptor.Data());
  tCanBgdwFitName += TString(cAnalysisBaseTags[aAnType]);

  if(aCombineConjugates) tCanBgdwFitName += TString("wConj");
  if(!aCombineImpactParams) tCanBgdwFitName += TString::Format("_b%d", aImpactParam);
  else tCanBgdwFitName += TString(cCentralityTags[tCentType]);

//-------------------------------------------------------------------------------

  TCanvas* tCanBgdwFit = new TCanvas(tCanBgdwFitName, tCanBgdwFitName);
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);
  //---------------------------------------------------------------
  tCf->GetXaxis()->SetTitle("#it{k}* (GeV/#it{c})");
  tCf->GetYaxis()->SetTitle("#it{C}(#it{k}*)");

  tCf->GetXaxis()->SetRangeUser(0.,3.0);
  tCf->GetYaxis()->SetRangeUser(0.86, 1.07);

  tCf->Draw();
  //---------------------------------------------------------------
  TLegend* tLeg = new TLegend(0.60, 0.15, 0.85, 0.40);
    tLeg->SetFillColor(0);
    tLeg->SetBorderSize(0);
    tLeg->SetTextAlign(22);

  tLeg->AddEntry(tCf, tDescriptor.Data());
  //---------------------------------------------------------------
  TF1 *tBgdFit, *tBgdFitDraw;
  int tPower = 6;
  tBgdFit = FitBackground(tCf, tPower);
  if(aAnType==kLamKchM || aAnType==kALamKchP) //Want to draw fit function without gaping gap where Omega peak omitted
  {
    gRejectOmega=false;
    tBgdFitDraw = new TF1(tBgdFit->GetName()+TString("_Draw"), FitFunctionPolynomial, 0., 3., 7);
    for(int i=0; i<tBgdFit->GetNpar(); i++) tBgdFitDraw->SetParameter(i, tBgdFit->GetParameter(i));
  }
  else tBgdFitDraw = tBgdFit;

  tBgdFitDraw->SetLineColor(tCf->GetLineColor());
  tBgdFitDraw->Draw("lsame");

  cout << endl << endl << "Bgd Chi2 = " << tBgdFit->GetChisquare() << endl;
  cout << "Background parameters:" << endl;
  for(int i=0; i<tBgdFit->GetNpar(); i++) cout << "Par[" << i << "] = " << tBgdFit->GetParameter(i) << endl;
  cout << "Single Line: " << endl;
  for(int i=0; i<tBgdFit->GetNpar(); i++) cout << TString::Format("% 11.8f, ", tBgdFit->GetParameter(i));
  cout << endl;

  tLeg->Draw();

  TLine* tLine = new TLine(0, 1, 2, 1);
  tLine->SetLineColor(14);
  tLine->Draw();

  PrintInfo((TPad*)tCanBgdwFit, tOverallDescriptor, 0.04);
  PrintFitParams((TPad*)tCanBgdwFit, tBgdFitDraw, 0.035);

  //---------------------------------------------------------------

  return tCanBgdwFit;
}

//________________________________________________________________________________________________________________
TCanvas* DrawLamKchPMBgdwFit(TString aCfDescriptor, TString aFileNameCfs, int aImpactParam, bool aUseAdamEvents, int aRebin, double aMinNorm, double aMaxNorm)
{
  gRejectOmega=true;
  //-------------------------------------------------
  CentralityType tCentType = GetCentralityType(aImpactParam);
  //-------------------------------------------------
  TString tDescriptor = "Bgd (w. Fit)";

  TString tOverallDescriptor = TString::Format("(#bar{#Lambda})#LambdaK^{#pm} (%s)", cPrettyCentralityTags[tCentType]);

  int tMarkerStyle = 26;
  int tColor = 4;

  //--------------------------------------------
  TH1D *tCf = (TH1D*)GetLamKchPMCombinedTHERMCfs(aFileNameCfs, aCfDescriptor, tCentType, aUseAdamEvents, aRebin, aMinNorm, aMaxNorm, tMarkerStyle, tColor);
//-------------------------------------------------------------------------------
  TString tCanBgdwFitName = "LamKchPMBgdwFitOnly";

  if(aFileNameCfs.Contains("_RandomEPs_NumWeight1")) tCanBgdwFitName += TString("_RandomEPs_NumWeight1");
  else if(aFileNameCfs.Contains("_RandomEPs")) tCanBgdwFitName += TString("_RandomEPs");
  else tCanBgdwFitName += TString("");

  tCanBgdwFitName += TString::Format("_%s", aCfDescriptor.Data());
  tCanBgdwFitName += TString(cCentralityTags[tCentType]);

//-------------------------------------------------------------------------------

  TCanvas* tCanBgdwFit = new TCanvas(tCanBgdwFitName, tCanBgdwFitName);
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);
  //---------------------------------------------------------------
  tCf->GetXaxis()->SetTitle("#it{k}* (GeV/#it{c})");
  tCf->GetYaxis()->SetTitle("#it{C}(#it{k}*)");

  tCf->GetXaxis()->SetRangeUser(0.,3.0);
  tCf->GetYaxis()->SetRangeUser(0.86, 1.07);

  tCf->Draw();
  //---------------------------------------------------------------
  TLegend* tLeg = new TLegend(0.60, 0.15, 0.85, 0.40);
    tLeg->SetFillColor(0);
    tLeg->SetBorderSize(0);
    tLeg->SetTextAlign(22);

  tLeg->AddEntry(tCf, tDescriptor.Data());
  //---------------------------------------------------------------
  TF1 *tBgdFit, *tBgdFitDraw;
  int tPower = 6;
  tBgdFit = FitBackground(tCf, tPower);

  gRejectOmega=false;
  tBgdFitDraw = new TF1(tBgdFit->GetName()+TString("_Draw"), FitFunctionPolynomial, 0., 3., 7);
  for(int i=0; i<tBgdFit->GetNpar(); i++) tBgdFitDraw->SetParameter(i, tBgdFit->GetParameter(i));

  tBgdFitDraw->SetLineColor(tCf->GetLineColor());
  tBgdFitDraw->Draw("lsame");

  cout << endl << endl << "Bgd Chi2 = " << tBgdFit->GetChisquare() << endl;
  cout << "Background parameters:" << endl;
  for(int i=0; i<tBgdFit->GetNpar(); i++) cout << "Par[" << i << "] = " << tBgdFit->GetParameter(i) << endl;
  cout << "Single Line: " << endl;
  for(int i=0; i<tBgdFit->GetNpar(); i++) cout << TString::Format("% 11.8f, ", tBgdFit->GetParameter(i));
  cout << endl;

  tLeg->Draw();

  TLine* tLine = new TLine(0, 1, 2, 1);
  tLine->SetLineColor(14);
  tLine->Draw();

  PrintInfo((TPad*)tCanBgdwFit, tOverallDescriptor, 0.04);
  PrintFitParams((TPad*)tCanBgdwFit, tBgdFitDraw, 0.035);

  //---------------------------------------------------------------

  return tCanBgdwFit;
}

//________________________________________________________________________________________________________________
TCanvas* CompareAnalyses(TString aCfDescriptor, TString aFileNameCfs, int aImpactParam, bool aCombineConjugates, bool aCombineImpactParams, bool aUseAdamEvents, int aRebin, double aMinNorm, double aMaxNorm)
{
  CentralityType tCentType=kMB;
  if(aCombineImpactParams) tCentType = GetCentralityType(aImpactParam);
  //-------------------------------------------------

  TString tDescriptor1 = cAnalysisRootTags[kLamK0];
  TString tDescriptor2 = cAnalysisRootTags[kLamKchP];
  TString tDescriptor3 = cAnalysisRootTags[kLamKchM];

  if(aCombineConjugates)
  {
    tDescriptor1 += TString::Format(" & %s", cAnalysisRootTags[kALamK0]);
    tDescriptor2 += TString::Format(" & %s", cAnalysisRootTags[kALamKchM]);
    tDescriptor3 += TString::Format(" & %s", cAnalysisRootTags[kALamKchP]);
  }

  TString tOverallDescriptor;
  if(!aCombineImpactParams) tOverallDescriptor = TString::Format("b=%d", aImpactParam);
  else tOverallDescriptor = TString::Format("%s", cPrettyCentralityTags[tCentType]);

  int tMarkerStyle1 = 20;
  int tMarkerStyle2 = 20;
  int tMarkerStyle3 = 20;

  int tColor1 = kBlack;
  int tColor2 = kRed+1;
  int tColor3 = kBlue+1;

  //--------------------------------------------
  TH1D *tCf1, *tCf2, *tCf3;
  if(!aCombineImpactParams)
  {
    tCf1 = (TH1D*)GetTHERMCf(aFileNameCfs, aCfDescriptor, kLamK0, aImpactParam, aCombineConjugates, aUseAdamEvents, aRebin, aMinNorm, aMaxNorm, tMarkerStyle1, tColor1);
    tCf2 = (TH1D*)GetTHERMCf(aFileNameCfs, aCfDescriptor, kLamKchP, aImpactParam, aCombineConjugates, aUseAdamEvents, aRebin, aMinNorm, aMaxNorm, tMarkerStyle2, tColor2);
    tCf3 = (TH1D*)GetTHERMCf(aFileNameCfs, aCfDescriptor, kLamKchM, aImpactParam, aCombineConjugates, aUseAdamEvents, aRebin, aMinNorm, aMaxNorm, tMarkerStyle3, tColor3);
  }
  else
  {
    tCf1 = (TH1D*)GetCombinedTHERMCfs(aFileNameCfs, aCfDescriptor, kLamK0, tCentType, aCombineConjugates, aUseAdamEvents, aRebin, aMinNorm, aMaxNorm, tMarkerStyle1, tColor1);
    tCf2 = (TH1D*)GetCombinedTHERMCfs(aFileNameCfs, aCfDescriptor, kLamKchP, tCentType, aCombineConjugates, aUseAdamEvents, aRebin, aMinNorm, aMaxNorm, tMarkerStyle2, tColor2);
    tCf3 = (TH1D*)GetCombinedTHERMCfs(aFileNameCfs, aCfDescriptor, kLamKchM, tCentType, aCombineConjugates, aUseAdamEvents, aRebin, aMinNorm, aMaxNorm, tMarkerStyle3, tColor3);
  }
//-------------------------------------------------------------------------------
  TString tCanCfsName;
  tCanCfsName = "CompareAnalyses";

  if(aFileNameCfs.Contains("_RandomEPs_NumWeight1")) tCanCfsName += TString("_RandomEPs_NumWeight1");
  else if(aFileNameCfs.Contains("_RandomEPs")) tCanCfsName += TString("_RandomEPs");
  else tCanCfsName += TString("");

  tCanCfsName += TString::Format("_%s_", aCfDescriptor.Data());

  if(aCombineConjugates) tCanCfsName += TString("wConj");
  if(!aCombineImpactParams) tCanCfsName += TString::Format("_b%d", aImpactParam);
  else tCanCfsName += TString(cCentralityTags[tCentType]);

  TCanvas* tCanCfs = new TCanvas(tCanCfsName, tCanCfsName);
  tCanCfs->cd();
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);
  //---------------------------------------------------------------
  tCf1->GetXaxis()->SetTitle("#it{k}* (GeV/#it{c})");
  tCf1->GetYaxis()->SetTitle("#it{C}(#it{k}*)");

  tCf1->GetXaxis()->SetRangeUser(0.,2.0);
  tCf1->GetYaxis()->SetRangeUser(0.86, 1.07);

  tCf1->Draw();
  tCf2->Draw("same");
  tCf3->Draw("same");
  //---------------------------------------------------------------

  TLegend* tLeg = new TLegend(0.60, 0.15, 0.85, 0.40);
    tLeg->SetFillColor(0);
    tLeg->SetBorderSize(0);
    tLeg->SetTextAlign(22);

  tLeg->AddEntry(tCf1, tDescriptor1.Data());
  tLeg->AddEntry(tCf2, tDescriptor2.Data());
  tLeg->AddEntry(tCf3, tDescriptor3.Data());

  tLeg->Draw();
  //---------------------------------------------------------------

  TLine* tLine = new TLine(0, 1, 2, 1);
  tLine->SetLineColor(14);
  tLine->Draw();

  PrintInfo((TPad*)tCanCfs, tOverallDescriptor, 0.04);

  return tCanCfs;
}

//________________________________________________________________________________________________________________
TCanvas* CompareToAdam(TString aCfDescriptor, TString aFileNameCfs, AnalysisType aAnType, int aImpactParam, bool aCombineConjugates, bool aCombineImpactParams, int aRebin, double aMinNorm, double aMaxNorm)
{
  CentralityType tCentType=kMB;
  if(aCombineImpactParams) tCentType = GetCentralityType(aImpactParam);
  //-------------------------------------------------

  TString tDescriptor1 = "Me";
  TString tDescriptor2 = "Adam";

  TString tOverallDescriptor = cAnalysisRootTags[aAnType];
  if(aCombineConjugates) tOverallDescriptor += TString::Format(" & %s", cAnalysisRootTags[GetConjAnType(aAnType)]);
  if(!aCombineImpactParams) tOverallDescriptor += TString::Format(" (b=%d)", aImpactParam);
  else tOverallDescriptor += TString::Format(" (%s)", cPrettyCentralityTags[tCentType]);

  int tMarkerStyle1 = 20;
  int tMarkerStyle2 = 20;

  int tColor1 = kBlack;
  int tColor2 = kRed+1;

  //--------------------------------------------
  TH1D *tCf1, *tCf2;
  if(!aCombineImpactParams)
  {
    tCf1 = (TH1D*)GetTHERMCf(aFileNameCfs, aCfDescriptor, aAnType, aImpactParam, aCombineConjugates, false, aRebin, aMinNorm, aMaxNorm, tMarkerStyle1, tColor1);
    tCf2 = (TH1D*)GetTHERMCf(aFileNameCfs, aCfDescriptor, aAnType, aImpactParam, aCombineConjugates, true, aRebin, aMinNorm, aMaxNorm, tMarkerStyle2, tColor2);
  }
  else
  {
    tCf1 = (TH1D*)GetCombinedTHERMCfs(aFileNameCfs, aCfDescriptor, aAnType, tCentType, aCombineConjugates, false, aRebin, aMinNorm, aMaxNorm, tMarkerStyle1, tColor1);
    tCf2 = (TH1D*)GetCombinedTHERMCfs(aFileNameCfs, aCfDescriptor, aAnType, tCentType, aCombineConjugates, true, aRebin, aMinNorm, aMaxNorm, tMarkerStyle2, tColor2);
  }
//-------------------------------------------------------------------------------
  TString tCanCfsName;
  tCanCfsName = TString::Format("CompareToAdam_%s", cAnalysisRootTags[aAnType]);
  if(aCombineConjugates) tCanCfsName += TString("wConj");

  if(!aCombineImpactParams) tCanCfsName += TString::Format("_b%d", aImpactParam);
  else tCanCfsName += TString(cCentralityTags[tCentType]);

  if(aFileNameCfs.Contains("_RandomEPs_NumWeight1")) tCanCfsName += TString("_RandomEPs_NumWeight1");
  else if(aFileNameCfs.Contains("_RandomEPs")) tCanCfsName += TString("_RandomEPs");
  else tCanCfsName += TString("");

  tCanCfsName += TString::Format("_%s_", aCfDescriptor.Data());

  TCanvas* tCanCfs = new TCanvas(tCanCfsName, tCanCfsName);
  tCanCfs->cd();
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);
  //---------------------------------------------------------------
  tCf1->GetXaxis()->SetTitle("#it{k}* (GeV/#it{c})");
  tCf1->GetYaxis()->SetTitle("#it{C}(#it{k}*)");

  tCf1->GetXaxis()->SetRangeUser(0.,2.0);
  tCf1->GetYaxis()->SetRangeUser(0.86, 1.07);

  tCf1->Draw();
  tCf2->Draw("same");
  //---------------------------------------------------------------

  TLegend* tLeg = new TLegend(0.60, 0.15, 0.85, 0.40);
    tLeg->SetFillColor(0);
    tLeg->SetBorderSize(0);
    tLeg->SetTextAlign(22);

  tLeg->AddEntry(tCf1, tDescriptor1.Data());
  tLeg->AddEntry(tCf2, tDescriptor2.Data());

  tLeg->Draw();
  //---------------------------------------------------------------

  TLine* tLine = new TLine(0, 1, 2, 1);
  tLine->SetLineColor(14);
  tLine->Draw();

  PrintInfo((TPad*)tCanCfs, tOverallDescriptor, 0.04);

  return tCanCfs;
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

  bool bUseAdamEvents = false;

  bool bCompareWithAndWithoutBgd = false;
  bool bDrawBgdwFitOnly = true;
  bool bDrawLamKchPMBgdwFitOnly = false;
  bool bCompareAnalyses = false;
  bool bCompareToAdam = true;

  bool bSaveFigures = false;
  int tRebin=2;
  double tMinNorm = 0.32;
  double tMaxNorm = 0.40;

  int tImpactParam = 8;

  TString tCfDescriptor = "Full";
//  TString tCfDescriptor = "PrimaryOnly";

  TString tSingleFileName = "CorrelationFunctions_RandomEPs_NumWeight1.root";

  TString tSaveDir = "/home/jesse/Analysis/FemtoAnalysis/AnalysisNotes/Comments/Laura/20180117/Figures/";
  TString tSaveFileBase = tSaveDir + TString::Format("%s/", cAnalysisBaseTags[tAnType]);


  TCanvas *tCanCfs, *tCanBgdwFit, *tCanCompareAnalyses, *tCanCompareToAdam, *tCanLamKchPMBgdwFit;

  if(bCompareWithAndWithoutBgd)
  {
    tCanCfs = CompareCfWithAndWithoutBgd(tCfDescriptor, tAnType, tImpactParam, bCombineConjugates, bCombineImpactParams, bUseAdamEvents, tRebin, tMinNorm, tMaxNorm);
    if(bSaveFigures) tCanCfs->SaveAs(TString::Format("%s%s.eps", tSaveFileBase.Data(), tCanCfs->GetName()));
  }

  if(bDrawBgdwFitOnly)
  {
    tCanBgdwFit = DrawBgdwFit(tCfDescriptor, tSingleFileName, tAnType, tImpactParam, bCombineConjugates, bCombineImpactParams, bUseAdamEvents, tRebin, tMinNorm, tMaxNorm);
    if(bSaveFigures) tCanBgdwFit->SaveAs(TString::Format("%s%s.eps", tSaveFileBase.Data(), tCanBgdwFit->GetName()));
  }

  if(bCompareAnalyses)
  {
    tCanCompareAnalyses = CompareAnalyses(tCfDescriptor, tSingleFileName, tImpactParam, bCombineConjugates, bCombineImpactParams, bUseAdamEvents, tRebin, tMinNorm, tMaxNorm);
    if(bSaveFigures) tCanCompareAnalyses->SaveAs(TString::Format("%s%s.eps", tSaveFileBase.Data(), tCanCompareAnalyses->GetName()));
  }

  if(bCompareToAdam)
  {
    tCanCompareToAdam = CompareToAdam(tCfDescriptor, tSingleFileName, tAnType, tImpactParam, bCombineConjugates, bCombineImpactParams, tRebin, tMinNorm, tMaxNorm);
    if(bSaveFigures) tCanCompareToAdam->SaveAs(TString::Format("%s%s.eps", tSaveFileBase.Data(), tCanCompareToAdam->GetName()));
  }


  if(bDrawLamKchPMBgdwFitOnly)
  {
    tCanLamKchPMBgdwFit = DrawLamKchPMBgdwFit(tCfDescriptor, tSingleFileName, tImpactParam, bUseAdamEvents, tRebin, tMinNorm, tMaxNorm);
    if(bSaveFigures) tCanLamKchPMBgdwFit->SaveAs(TString::Format("%s%s.eps", tSaveFileBase.Data(), tCanLamKchPMBgdwFit->GetName()));
  }

//-------------------------------------------------------------------------------




//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
