//Taken from /home/jesse/Analysis/FemtoAnalysis/Therminator/CompareBackgroundsAndFit.C

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
#include "TLegendEntry.h"

#include "ThermCf.h"
class ThermCf;

#include "SuperpositionFitBgd.h"
class SuperpositionFitBgd;

#include "CanvasPartition.h"
#include "FitGeneratorAndDraw.h"
#include "HistInfoPrinter.h"

//GLOBAL!!!!!!!!!!!!!!!
SuperpositionFitBgd *GlobalSupBgdFitter = NULL;

//______________________________________________________________________________
void GlobalSupBgdFCN(Int_t &npar, Double_t *gin, Double_t &f, Double_t *par, Int_t iflag)
{
  GlobalSupBgdFitter->CalculateBgdFitFunction(npar,f,par);
}




bool gRejectOmegaTherm=false;
//________________________________________________________________________________________________________________
double FitFunctionPolynomial(double *x, double *par)
{
  if(gRejectOmegaTherm && x[0]>0.19 && x[0]<0.23)
  {
    TF1::RejectPoint();
    return 0;
  }
  return par[0] + par[1]*x[0] + par[2]*pow(x[0],2) + par[3]*pow(x[0],3) + par[4]*pow(x[0],4) + par[5]*pow(x[0],5) + par[6]*pow(x[0],6);
}

//________________________________________________________________________________________________________________
TF1* FitBackground(TH1* aBgdOnlyCf, int aPower=6, double aMinBgdFit=0., double aMaxBgdFit=3.)
{
  cout << endl << endl << "Fitting: " << aBgdOnlyCf->GetName() << " with FitBackground" << endl;

  assert(aPower <= 6);  //Up to 6th order polynomial

  TString tFitName = TString::Format("BgdFit_%s", aBgdOnlyCf->GetName());
  TF1* tBgdFit = new TF1(tFitName, FitFunctionPolynomial, 0., 3., 7);

  tBgdFit->SetParameter(0, 1.);
  tBgdFit->FixParameter(1, 0.);
  tBgdFit->SetParameter(2, 0.);
  tBgdFit->SetParameter(3, 0.);
  tBgdFit->SetParameter(4, 0.);
  tBgdFit->SetParameter(5, 0.);
  tBgdFit->SetParameter(6, 0.);

//  tBgdFit->SetParLimits(0, 1.0, 1.1);

  //If I want the fit to be symmetric about the y-axis
//  tBgdFit->FixParameter(1, 0.);
//  tBgdFit->FixParameter(3, 0.);
//  tBgdFit->FixParameter(5, 0.);

  if(aPower<6)
  {
    for(int i=6; i>aPower; i--)
    {
      cout << "tBgdFit->FixParameter(" << i << ", 0.);" << endl;
      tBgdFit->FixParameter(i, 0.);
    }
  }

  aBgdOnlyCf->Fit(tFitName, "0q", "", aMinBgdFit, aMaxBgdFit);
  //-------------------------------------------------------------------
  cout << "Chi2 = " << tBgdFit->GetChisquare() << endl;
  cout << "Parameters:" << endl;
  for(int i=0; i<tBgdFit->GetNpar(); i++) cout << "Par[" << i << "] = " << tBgdFit->GetParameter(i) << endl;
  cout << "Single Line: " << endl;
  for(int i=0; i<tBgdFit->GetNpar(); i++) cout << TString::Format("% 11.8f, ", tBgdFit->GetParameter(i));
  cout << endl << endl;
  //-------------------------------------------------------------------
  return tBgdFit;
}

//________________________________________________________________________________________________________________
TF1* TempFit(TH1* aBgdOnlyCf, double aMinBgdFit=0., double aMaxBgdFit=3.)
{
  cout << endl << endl << "Fitting: " << aBgdOnlyCf->GetName() << " with FitBackground" << endl;

  TString tFitName = TString::Format("BgdFit_%s", aBgdOnlyCf->GetName());
  TF1* tBgdFit = new TF1(tFitName, FitFunctionPolynomial, 0., 3., 7);

  tBgdFit->SetParameter(0, 1.);
  tBgdFit->SetParLimits(0, 0., 1.);

  tBgdFit->FixParameter(1, 0.);
  tBgdFit->FixParameter(2, 0.);
  tBgdFit->FixParameter(3, 0.);
  tBgdFit->FixParameter(4, 0.);
  tBgdFit->FixParameter(5, 0.);
  tBgdFit->FixParameter(6, 0.);

  aBgdOnlyCf->Fit(tFitName, "0q", "", aMinBgdFit, aMaxBgdFit);
  //-------------------------------------------------------------------
  cout << "Chi2 = " << tBgdFit->GetChisquare() << endl;
  cout << "Parameters:" << endl;
  for(int i=0; i<tBgdFit->GetNpar(); i++) cout << "Par[" << i << "] = " << tBgdFit->GetParameter(i) << endl;
  cout << "Single Line: " << endl;
  for(int i=0; i<tBgdFit->GetNpar(); i++) cout << TString::Format("% 11.8f, ", tBgdFit->GetParameter(i));
  cout << endl << endl;
  //-------------------------------------------------------------------
  return tBgdFit;
}

//________________________________________________________________________________________________________________
double FitFunctionPolynomialwNorm(double *x, double *par)
{
  if(gRejectOmegaTherm && x[0]>0.19 && x[0]<0.23)
  {
    TF1::RejectPoint();
    return 0;
  }
  return par[7]*FitFunctionPolynomial(x, par);
}

//________________________________________________________________________________________________________________
TF1* FitBackgroundwNorm(TF1* aThermBgdFit, TH1D* aBgdOnlyCf, int aPower=6, double aMinBgdFit=0., double aMaxBgdFit=3.)
{
//  cout << endl << endl << "Fitting: " << aBgdOnlyCf->GetName() << " with FitBackgroundwNorm" << endl;

  assert(aPower <= 6);  //Up to 6th order polynomial

  TString tFitName = TString::Format("BgdFitwNorm_%s", aBgdOnlyCf->GetName());
  TF1* tBgdFit = new TF1(tFitName, FitFunctionPolynomialwNorm, 0., 3., 8);

  tBgdFit->FixParameter(0, aThermBgdFit->GetParameter(0));
  tBgdFit->FixParameter(1, aThermBgdFit->GetParameter(1));
  tBgdFit->FixParameter(2, aThermBgdFit->GetParameter(2));
  tBgdFit->FixParameter(3, aThermBgdFit->GetParameter(3));
  tBgdFit->FixParameter(4, aThermBgdFit->GetParameter(4));
  tBgdFit->FixParameter(5, aThermBgdFit->GetParameter(5));
  tBgdFit->FixParameter(6, aThermBgdFit->GetParameter(6));

  tBgdFit->SetParameter(7, 1.);
/*
  if(aPower<6)
  {
    for(int i=6; i>aPower; i--)
    {
      cout << "tBgdFitwNorm->FixParameter(" << i << ", 0.);" << endl;
      tBgdFit->FixParameter(i, 0.);
    }
  }
*/
  aBgdOnlyCf->Fit(tFitName, "0q", "", aMinBgdFit, aMaxBgdFit);
  //-------------------------------------------------------------------
/*
  cout << "Chi2 = " << tBgdFit->GetChisquare() << endl;
  cout << "Parameters:" << endl;
  for(int i=0; i<tBgdFit->GetNpar(); i++) cout << "Par[" << i << "] = " << tBgdFit->GetParameter(i) << endl;
  cout << "Single Line: " << endl;
  for(int i=0; i<tBgdFit->GetNpar(); i++) cout << TString::Format("% 11.8f, ", tBgdFit->GetParameter(i));
  cout << endl << endl;
*/
  //-------------------------------------------------------------------
  return tBgdFit;
}

//________________________________________________________________________________________________________________
double FitFunctionPolynomialwNormAndOffset(double *x, double *par)
{
  if(gRejectOmegaTherm && x[0]>0.19 && x[0]<0.23)
  {
    TF1::RejectPoint();
    return 0;
  }
  return par[7]*FitFunctionPolynomial(x, par) + par[8];
}

//________________________________________________________________________________________________________________
TF1* FitBackgroundwNormAndOffset(TF1* aThermBgdFit, TH1D* aBgdOnlyCf, int aPower=6, double aMinBgdFit=0., double aMaxBgdFit=3.)
{
//  cout << endl << endl << "Fitting: " << aBgdOnlyCf->GetName() << " with FitBackgroundwNorm" << endl;

  assert(aPower <= 6);  //Up to 6th order polynomial

  TString tFitName = TString::Format("BgdFitwNormAndOffset_%s", aBgdOnlyCf->GetName());
  TF1* tBgdFit = new TF1(tFitName, FitFunctionPolynomialwNormAndOffset, 0., 3., 9);

  tBgdFit->FixParameter(0, aThermBgdFit->GetParameter(0));
  tBgdFit->FixParameter(1, aThermBgdFit->GetParameter(1));
  tBgdFit->FixParameter(2, aThermBgdFit->GetParameter(2));
  tBgdFit->FixParameter(3, aThermBgdFit->GetParameter(3));
  tBgdFit->FixParameter(4, aThermBgdFit->GetParameter(4));
  tBgdFit->FixParameter(5, aThermBgdFit->GetParameter(5));
  tBgdFit->FixParameter(6, aThermBgdFit->GetParameter(6));

  tBgdFit->SetParameter(7, 1.);
  tBgdFit->SetParameter(8, 0.);

  aBgdOnlyCf->Fit(tFitName, "0q", "", aMinBgdFit, aMaxBgdFit);

  return tBgdFit;
}

//________________________________________________________________________________________________________________
void PrintFitParams(TPad* aPad, TF1* aFit, double aTextSize=0.035, int aColor=800)
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

  tTex->DrawLatex(0.15, 0.92, TString::Format("#color[%i]{Bgd} = %0.3f + %0.3fx + %0.3fx^{2}" , aColor, tPar0, tPar1, tPar2));
  tTex->DrawLatex(0.30, 0.91, TString::Format("+ %0.3fx^{3} + %0.3fx^{4}" , tPar3, tPar4));
  tTex->DrawLatex(0.30, 0.90, TString::Format("+ %0.3fx^{5} + %0.3fx^{6}" , tPar5, tPar6));
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

  //For CompareBackgroundReductionMethods when using ArtificialV3Signal-1
  //tTex->DrawLatex(0.3, 3.05, "THERMINATOR");
  //tTex->DrawLatex(1.2, 3.05, aOverallDescriptor);

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
int GetColor(AnalysisType aAnType)
{
  int tReturnColor;

  if(aAnType==kLamK0 || aAnType==kALamK0) tReturnColor = kBlack;
  else if(aAnType==kLamKchP || aAnType==kALamKchM) tReturnColor = kRed+1;
  else if(aAnType==kLamKchM || aAnType==kALamKchP) tReturnColor = kBlue+1;
  else assert(0);

  return tReturnColor;
}

//________________________________________________________________________________________________________________
void ScaleCustomRebinnedCf(double aOriginalBinWidth, TH1* aCf, const td1dVec &aCustomBins)
{
  assert(aCf->GetNbinsX() == (int)(aCustomBins.size()-1));
  double tInvScale;
  for(unsigned int i=1; i<aCustomBins.size(); i++)
  {
    tInvScale = (aCustomBins[i]-aCustomBins[i-1])/aOriginalBinWidth;
    aCf->SetBinContent(i, aCf->GetBinContent(i)/tInvScale);
    aCf->SetBinError(i, aCf->GetBinError(i)/tInvScale);
  }
}



//________________________________________________________________________________________________________________
TH1D* GetQuickData(AnalysisType aAnType, CentralityType aCentType, bool aCombineConjugates, TString aResultsData="20180505", int aRebin=1)
{
  TString tFileName;
  if(aRebin==1) tFileName = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Therminator/QuickData/QuickDataCfs_%s.root", aResultsData.Data());
  else          tFileName = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Therminator/QuickData/QuickDataCfs_%s_Rebin%d.root", aResultsData.Data(), aRebin);
  TFile tFile(tFileName);

  TString tReturnHistName;
  if(!aCombineConjugates) tReturnHistName = TString::Format("KStarHeavyCf_%s%s", cAnalysisBaseTags[aAnType], cCentralityTags[aCentType]);
  else tReturnHistName = TString::Format("KStarHeavyCf_%s%s%s", cAnalysisBaseTags[aAnType], cAnalysisBaseTags[GetConjAnType(aAnType)], cCentralityTags[aCentType]);
  TH1D* tReturnHist = (TH1D*)tFile.Get(tReturnHistName);
  tReturnHist->SetDirectory(0);

  return tReturnHist;
}


//________________________________________________________________________________________________________________
TObjArray* GetSlowDataWithSysErrs(AnalysisType aAnType, CentralityType aCentType, bool aCombineConjugates, TString aResultsData, td1dVec &aCustomBins)
{
  assert(aAnType==kLamK0 || aAnType==kLamKchP || aAnType==kLamKchM);

  TString tDirectoryBase_LamK, tFileLocationBase_LamK, tFileLocationBaseMC_LamK;

  if(aAnType==kLamK0 || aAnType==kALamK0)
  {
    tDirectoryBase_LamK = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamK0_%s/",aResultsData.Data());
    tFileLocationBase_LamK = TString::Format("%sResults_cLamK0_%s",tDirectoryBase_LamK.Data(),aResultsData.Data());
    tFileLocationBaseMC_LamK = TString::Format("%sResults_cLamK0MC_%s",tDirectoryBase_LamK.Data(),aResultsData.Data());
  }
  else if(aAnType==kLamKchP || aAnType==kALamKchM
        ||aAnType==kLamKchM || aAnType==kALamKchP)
  {
    tDirectoryBase_LamK = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_%s/",aResultsData.Data());
    tFileLocationBase_LamK = TString::Format("%sResults_cLamcKch_%s",tDirectoryBase_LamK.Data(),aResultsData.Data());
    tFileLocationBaseMC_LamK = TString::Format("%sResults_cLamcKchMC_%s",tDirectoryBase_LamK.Data(),aResultsData.Data());
  }
  else assert(0);

  FitGeneratorAndDraw* tFG = new FitGeneratorAndDraw(tFileLocationBase_LamK, tFileLocationBaseMC_LamK, aAnType, kMB);

  CfHeavy* tCfHeavyAn   = tFG->GetKStarCfHeavy(2*aCentType);
  CfHeavy* tCfHeavyConj = tFG->GetKStarCfHeavy(2*aCentType+1);

  TH1* tCfwSysErrsAn   = tFG->GetSharedAn()->GetFitPairAnalysis(2*aCentType)->GetCfwSysErrors();
  TH1* tCfwSysErrsConj = tFG->GetSharedAn()->GetFitPairAnalysis(2*aCentType+1)->GetCfwSysErrors();

  if(aCustomBins.size() != 1)
  {
    tCfHeavyAn->Rebin((int)aCustomBins.size()-1, aCustomBins);
    tCfHeavyConj->Rebin((int)aCustomBins.size()-1, aCustomBins);

    if(!tCfwSysErrsAn->GetSumw2N()) tCfwSysErrsAn->Sumw2();
    if(!tCfwSysErrsConj->GetSumw2N()) tCfwSysErrsConj->Sumw2();

    double tOGBinWidthAn = tCfwSysErrsAn->GetBinWidth(1);
    tCfwSysErrsAn = tCfwSysErrsAn->Rebin((int)aCustomBins.size()-1, TString::Format("%s_CustomRebin", tCfwSysErrsAn->GetName()), aCustomBins.data());
    ScaleCustomRebinnedCf(tOGBinWidthAn, tCfwSysErrsAn, aCustomBins);

    double tOGBinWidthConj = tCfwSysErrsConj->GetBinWidth(1);
    tCfwSysErrsConj = tCfwSysErrsConj->Rebin((int)aCustomBins.size()-1, TString::Format("%s_CustomRebin", tCfwSysErrsConj->GetName()), aCustomBins.data());
    ScaleCustomRebinnedCf(tOGBinWidthConj, tCfwSysErrsConj, aCustomBins);
  }
  else if(aCustomBins[0] != 1)
  {
    tCfHeavyAn->Rebin(aCustomBins[0]);
    tCfHeavyConj->Rebin(aCustomBins[0]);

    if(!tCfwSysErrsAn->GetSumw2N()) tCfwSysErrsAn->Sumw2();
    if(!tCfwSysErrsConj->GetSumw2N()) tCfwSysErrsConj->Sumw2();
    tCfwSysErrsAn->Rebin(aCustomBins[0]);
      tCfwSysErrsAn->Scale(1./aCustomBins[0]);
    tCfwSysErrsConj->Rebin(aCustomBins[0]);
      tCfwSysErrsConj->Scale(1./aCustomBins[0]);
  }

  if(aCustomBins.size() != 1 || aCustomBins[0] != 1)
  {
    //Technically not rebinned correctly, so data probably slight differ
    assert(tCfwSysErrsAn->GetNbinsX() == tCfHeavyAn->GetHeavyCf()->GetNbinsX());
    assert(tCfwSysErrsConj->GetNbinsX() == tCfHeavyConj->GetHeavyCf()->GetNbinsX());
    assert(tCfwSysErrsAn->GetNbinsX() == tCfwSysErrsConj->GetNbinsX());

    for(int i=1; i<=tCfwSysErrsAn->GetNbinsX(); i++)
    {
      assert(tCfwSysErrsAn->GetBinWidth(i) == tCfHeavyAn->GetHeavyCf()->GetBinWidth(i));
      assert(tCfwSysErrsConj->GetBinWidth(i) == tCfHeavyConj->GetHeavyCf()->GetBinWidth(i));
      assert(tCfwSysErrsAn->GetBinWidth(i) == tCfwSysErrsConj->GetBinWidth(i));

      double tFracDiffAn = (tCfwSysErrsAn->GetBinContent(i) - tCfHeavyAn->GetHeavyCf()->GetBinContent(i))/tCfHeavyAn->GetHeavyCf()->GetBinContent(i);
      assert(fabs(tFracDiffAn) < 0.05);
      double tFracDiffConj = (tCfwSysErrsConj->GetBinContent(i) - tCfHeavyConj->GetHeavyCf()->GetBinContent(i))/tCfHeavyConj->GetHeavyCf()->GetBinContent(i);
      assert(fabs(tFracDiffConj) < 0.05);

      tCfwSysErrsAn->SetBinContent(i, tCfHeavyAn->GetHeavyCf()->GetBinContent(i));
      tCfwSysErrsConj->SetBinContent(i, tCfHeavyConj->GetHeavyCf()->GetBinContent(i));
    }
  }

  TH1 *tReturnHistStat, *tReturnHistSys;
  if(!aCombineConjugates)
  {
    tReturnHistStat = (TH1D*)tCfHeavyAn->GetHeavyCfClone();
    tReturnHistSys  = tCfwSysErrsAn;
  }
  else
  {
    CfHeavy* tCfHeavyCombined = FitGeneratorAndDraw::CombineTwoHeavyCfs(tCfHeavyAn, tCfHeavyConj);
    tReturnHistStat = tCfHeavyCombined->GetHeavyCfClone();

    tReturnHistSys = FitGeneratorAndDraw::CombineTwoHists(tCfwSysErrsAn, tCfwSysErrsConj, tCfHeavyAn->GetTotalNumScale(), tCfHeavyConj->GetTotalNumScale());
  }

  TObjArray* tReturnArray = new TObjArray();
  tReturnArray->Add(tReturnHistStat);
  tReturnArray->Add(tReturnHistSys);

  return tReturnArray;
}

/*
//________________________________________________________________________________________________________________
TObjArray* GetSlowDataWithSysErrs(AnalysisType aAnType, CentralityType aCentType, bool aCombineConjugates, TString aResultsData="20190319", int aRebin=1)
{
  assert(aAnType==kLamK0 || aAnType==kLamKchP || aAnType==kLamKchM);

  TString tDirectoryBase_LamK, tFileLocationBase_LamK, tFileLocationBaseMC_LamK;

  if(aAnType==kLamK0 || aAnType==kALamK0)
  {
    tDirectoryBase_LamK = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamK0_%s/",aResultsData.Data());
    tFileLocationBase_LamK = TString::Format("%sResults_cLamK0_%s",tDirectoryBase_LamK.Data(),aResultsData.Data());
    tFileLocationBaseMC_LamK = TString::Format("%sResults_cLamK0MC_%s",tDirectoryBase_LamK.Data(),aResultsData.Data());
  }
  else if(aAnType==kLamKchP || aAnType==kALamKchM
        ||aAnType==kLamKchM || aAnType==kALamKchP)
  {
    tDirectoryBase_LamK = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_%s/",aResultsData.Data());
    tFileLocationBase_LamK = TString::Format("%sResults_cLamcKch_%s",tDirectoryBase_LamK.Data(),aResultsData.Data());
    tFileLocationBaseMC_LamK = TString::Format("%sResults_cLamcKchMC_%s",tDirectoryBase_LamK.Data(),aResultsData.Data());
  }
  else assert(0);

  FitGeneratorAndDraw* tFG = new FitGeneratorAndDraw(tFileLocationBase_LamK, tFileLocationBaseMC_LamK, aAnType, kMB);

  CfHeavy* tCfHeavyAn   = tFG->GetKStarCfHeavy(2*aCentType);
  CfHeavy* tCfHeavyConj = tFG->GetKStarCfHeavy(2*aCentType+1);

  TH1* tCfwSysErrsAn   = tFG->GetSharedAn()->GetFitPairAnalysis(2*aCentType)->GetCfwSysErrors();
  TH1* tCfwSysErrsConj = tFG->GetSharedAn()->GetFitPairAnalysis(2*aCentType+1)->GetCfwSysErrors();

  if(aRebin != 1)
  {
    tCfHeavyAn->Rebin(aRebin);
    tCfHeavyConj->Rebin(aRebin);

    if(!tCfwSysErrsAn->GetSumw2N()) tCfwSysErrsAn->Sumw2();
    if(!tCfwSysErrsConj->GetSumw2N()) tCfwSysErrsConj->Sumw2();
    tCfwSysErrsAn->Rebin(aRebin);
      tCfwSysErrsAn->Scale(1./aRebin);
    tCfwSysErrsConj->Rebin(aRebin);
      tCfwSysErrsConj->Scale(1./aRebin);

    //Technically not rebinned correctly, so data probably slight differ
    assert(tCfwSysErrsAn->GetNbinsX() == tCfHeavyAn->GetHeavyCf()->GetNbinsX());
    for(int i=1; i<=tCfwSysErrsAn->GetNbinsX(); i++)
    {
      double tFracDiffAn = (tCfwSysErrsAn->GetBinContent(i) - tCfHeavyAn->GetHeavyCf()->GetBinContent(i))/tCfHeavyAn->GetHeavyCf()->GetBinContent(i);
      assert(fabs(tFracDiffAn) < 0.05);
      double tFracDiffConj = (tCfwSysErrsConj->GetBinContent(i) - tCfHeavyConj->GetHeavyCf()->GetBinContent(i))/tCfHeavyConj->GetHeavyCf()->GetBinContent(i);
      assert(fabs(tFracDiffConj) < 0.05);

      tCfwSysErrsAn->SetBinContent(i, tCfHeavyAn->GetHeavyCf()->GetBinContent(i));
      tCfwSysErrsConj->SetBinContent(i, tCfHeavyConj->GetHeavyCf()->GetBinContent(i));
    }
  }

  TH1 *tReturnHistStat, *tReturnHistSys;
  if(!aCombineConjugates)
  {
    tReturnHistStat = (TH1D*)tCfHeavyAn->GetHeavyCfClone();
    tReturnHistSys  = tCfwSysErrsAn;
  }
  else
  {
    CfHeavy* tCfHeavyCombined = FitGeneratorAndDraw::CombineTwoHeavyCfs(tCfHeavyAn, tCfHeavyConj);
    tReturnHistStat = tCfHeavyCombined->GetHeavyCfClone();

    tReturnHistSys = FitGeneratorAndDraw::CombineTwoHists(tCfwSysErrsAn, tCfwSysErrsConj, tCfHeavyAn->GetTotalNumScale(), tCfHeavyConj->GetTotalNumScale());
  }

  TObjArray* tReturnArray = new TObjArray();
  tReturnArray->Add(tReturnHistStat);
  tReturnArray->Add(tReturnHistSys);

  return tReturnArray;
}
*/

//________________________________________________________________________________________________________________
TObjArray* GetSlowDataWithSysErrs(AnalysisType aAnType, CentralityType aCentType, bool aCombineConjugates, TString aResultsData="20190319", int aRebin=1)
{
  td1dVec tRebinVec{aRebin};
  TObjArray* tReturnArray = GetSlowDataWithSysErrs(aAnType, aCentType, aCombineConjugates, aResultsData, tRebinVec);
  return tReturnArray;
}

//________________________________________________________________________________________________________________
double GetChi2Value(TH1D* aData, TF1* aFit, double aMin, double aMax)
{
  double tChi=0., tChi2=0.;
  double tBinCenter=0., tFitVal=0.;

  double tMinBin = aData->FindBin(aMin);
  double tMaxBin = aData->FindBin(aMax);

  for(int i=tMinBin; i<tMaxBin; i++)
  {
    tBinCenter = aData->GetBinCenter(i);
    tFitVal = aFit->Eval(tBinCenter);

    tChi = (aData->GetBinContent(i)-tFitVal)/aData->GetBinError(i);
    tChi2 += tChi*tChi;
  }
  return tChi2;
}


//________________________________________________________________________________________________________________
void Draw1vs2vs3(TPad* aPad, AnalysisType aAnType, TH1* aCf1, TH1* aCf2, TH1* aCf3, TString aDescriptor1, TString aDescriptor2, TString aDescriptor3, TString aOverallDescriptor, bool aFitBgd=true, double aMaxFit=3.0)
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
    cout << "**************************************************" << endl;
    cout << "Fitting call from: Draw1vs2vs3" << endl;
    int tPower = 6;
    tBgdFit = FitBackground(aCf3, tPower, 0., aMaxFit);
    cout << "**************************************************" << endl;
    if(aAnType==kLamKchM || aAnType==kALamKchP) //Want to draw fit function without gaping gap where Omega peak omitted
    {
      gRejectOmegaTherm=false;
      tBgdFitDraw = new TF1(tBgdFit->GetName()+TString("_Draw"), FitFunctionPolynomial, 0., 2., 7);
      for(int i=0; i<tBgdFit->GetNpar(); i++) tBgdFitDraw->SetParameter(i, tBgdFit->GetParameter(i));
    }
    else tBgdFitDraw = tBgdFit;

    tBgdFitDraw->SetLineColor(aCf3->GetLineColor());
    tBgdFitDraw->Draw("lsame");
  }


  //---------------------------------------------------------------
  TH1D* tCfRatio = (TH1D*)aCf2->Clone();
  tCfRatio->Divide(aCf3);
  ThermCf::SetStyleAndColor(tCfRatio, 20, kCyan);
  tCfRatio->Draw("same");

  TH1D* tCfDiff = (TH1D*)aCf2->Clone();
  tCfDiff->Add(aCf3, -1.);
  TH1D* tUnity = (TH1D*)aCf2->Clone();
  for(int i=1; i<=tUnity->GetNbinsX(); i++) tUnity->SetBinContent(i, 1.);
  tCfDiff->Add(tUnity, 1.);
  ThermCf::SetStyleAndColor(tCfDiff, 24, kMagenta);
  tCfDiff->Draw("same");

  tLeg->AddEntry(tCfRatio, "Ratio (B/C)");
  tLeg->AddEntry(tCfDiff, "1+Diff (B-C)");

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
TCanvas* CompareCfWithAndWithoutBgd(TString aCfDescriptor, AnalysisType aAnType, int aImpactParam, bool aCombineConjugates, bool aCombineImpactParams, ThermEventsType aEventsType, int aRebin, double aMinNorm, double aMaxNorm, bool aUseStavCf=false)
{
  if(aAnType==kLamKchM || aAnType==kALamKchP) gRejectOmegaTherm=true;
  else gRejectOmegaTherm=false;
  if(aCfDescriptor.EqualTo("PrimaryOnly") || aCfDescriptor.EqualTo("PrimaryAndShortDecays")) gRejectOmegaTherm=false;  //In this case, Omega peak not present
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
  int tMarkerStyle2 = 21;
  int tMarkerStyle3 = 26;

  int tColor1 = kBlack;
  int tColor2 = kGreen+1;
  int tColor3 = kOrange;

  //--------------------------------------------

  ThermCf* tThermCf1 = new ThermCf(tFileNameCfs1, aCfDescriptor, aAnType, GetCentralityType(aImpactParam), aCombineConjugates, aEventsType, aRebin, aMinNorm, aMaxNorm, aUseStavCf);
  ThermCf* tThermCf2 = new ThermCf(tFileNameCfs2, aCfDescriptor, aAnType, GetCentralityType(aImpactParam), aCombineConjugates, aEventsType, aRebin, aMinNorm, aMaxNorm, aUseStavCf);
  ThermCf* tThermCf3 = new ThermCf(tFileNameCfs3, aCfDescriptor, aAnType, GetCentralityType(aImpactParam), aCombineConjugates, aEventsType, aRebin, aMinNorm, aMaxNorm, aUseStavCf);

  if(!aCombineImpactParams)
  {
    tThermCf1->SetSpecificImpactParam(aImpactParam, aCombineImpactParams);
    tThermCf2->SetSpecificImpactParam(aImpactParam, aCombineImpactParams);
    tThermCf3->SetSpecificImpactParam(aImpactParam, aCombineImpactParams);
  }
  TH1* tCf1 = tThermCf1->GetThermCf(tMarkerStyle1, tColor1, 0.75);
  TH1* tCf2 = tThermCf2->GetThermCf(tMarkerStyle2, tColor2, 0.75);
  TH1* tCf3 = tThermCf3->GetThermCf(tMarkerStyle3, tColor3, 0.75);

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
TF1* GetTHERMBgdFit(int aPower, TString aCfDescriptor, TString aFileNameCfs, AnalysisType aAnType, int aImpactParam, bool aCombineConjugates, bool aCombineImpactParams, ThermEventsType aEventsType, int aRebin, double aMinNorm, double aMaxNorm, double aMaxBgdFit=3.0, bool aUseStavCf=false)
{
  if(aAnType==kLamKchM || aAnType==kALamKchP) gRejectOmegaTherm=true;
  else gRejectOmegaTherm=false;
  if(aCfDescriptor.EqualTo("PrimaryOnly") || aCfDescriptor.EqualTo("PrimaryAndShortDecays")) gRejectOmegaTherm=false;  //In this case, Omega peak not present
  //-------------------------------------------------
  CentralityType tCentType=kMB;
  if(aCombineImpactParams) tCentType = GetCentralityType(aImpactParam);
  //-------------------------------------------------
  AnalysisType tConjAnType = GetConjAnType(aAnType);
  //-------------------------------------------------
  int tMarkerStyle = 26;
  int tColor = kOrange;

  //--------------------------------------------
  ThermCf* tThermCf = new ThermCf(aFileNameCfs, aCfDescriptor, aAnType, GetCentralityType(aImpactParam), aCombineConjugates, aEventsType, aRebin, aMinNorm, aMaxNorm, aUseStavCf);
  if(!aCombineImpactParams) tThermCf->SetSpecificImpactParam(aImpactParam, aCombineImpactParams);
  TH1* tCf = tThermCf->GetThermCf(tMarkerStyle, tColor, 0.75);
  //-------------------------------------------------------------------------------
  cout << "**************************************************" << endl;
  cout << "Fitting call(1) from: GetTHERMBgdFit" << endl;
  TF1 *tBgdFit, *tBgdFitDraw;
  tBgdFit = FitBackground(tCf, aPower, 0., aMaxBgdFit);
  cout << "**************************************************" << endl;
  if(aAnType==kLamKchM || aAnType==kALamKchP) //Want to draw fit function without gaping gap where Omega peak omitted
  {
    gRejectOmegaTherm=false;
    tBgdFitDraw = new TF1(tBgdFit->GetName()+TString("_Draw"), FitFunctionPolynomial, 0., 3., 7);
    for(int i=0; i<tBgdFit->GetNpar(); i++) tBgdFitDraw->SetParameter(i, tBgdFit->GetParameter(i));
  }
  else tBgdFitDraw = tBgdFit;

  tBgdFitDraw->SetLineColor(tCf->GetLineColor());
  tBgdFitDraw->SetRange(0., aMaxBgdFit);


  return tBgdFitDraw;
}


//________________________________________________________________________________________________________________
TCanvas* DrawBgdwFit(TString aCfDescriptor, TString aFileNameCfs, AnalysisType aAnType, int aImpactParam, bool aCombineConjugates, bool aCombineImpactParams, ThermEventsType aEventsType, int aRebin, double aMinNorm, double aMaxNorm, double aMaxBgdFit=3.0, bool aUseStavCf=false)
{
  if(aAnType==kLamKchM || aAnType==kALamKchP) gRejectOmegaTherm=true;
  else gRejectOmegaTherm=false;
  if(aCfDescriptor.EqualTo("PrimaryOnly") || aCfDescriptor.EqualTo("PrimaryAndShortDecays")) gRejectOmegaTherm=false;  //In this case, Omega peak not present
  //-------------------------------------------------
  CentralityType tCentType=kMB;
  if(aCombineImpactParams) tCentType = GetCentralityType(aImpactParam);
  //-------------------------------------------------
  AnalysisType tConjAnType = GetConjAnType(aAnType);
  //-------------------------------------------------
  TString tDescriptor = "THERM. Bgd (w. Fit)";

  TString tOverallDescriptor = cAnalysisRootTags[aAnType];
  if(aCombineConjugates) tOverallDescriptor += TString::Format(" & %s", cAnalysisRootTags[tConjAnType]);
  if(!aCombineImpactParams) tOverallDescriptor += TString::Format(" (b=%d)", aImpactParam);
  else tOverallDescriptor += TString::Format(" (%s)", cPrettyCentralityTags[tCentType]);

  int tMarkerStyle = 26;
  int tColor = kGreen+1;

  //--------------------------------------------
  ThermCf* tThermCf = new ThermCf(aFileNameCfs, aCfDescriptor, aAnType, GetCentralityType(aImpactParam), aCombineConjugates, aEventsType, aRebin, aMinNorm, aMaxNorm, aUseStavCf);
  if(!aCombineImpactParams) tThermCf->SetSpecificImpactParam(aImpactParam, aCombineImpactParams);
  TH1* tCf = tThermCf->GetThermCf(tMarkerStyle, tColor, 0.75);
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

  tCf->GetXaxis()->SetRangeUser(0.,aMaxBgdFit);
  tCf->GetYaxis()->SetRangeUser(0.86, 1.07);

  tCf->Draw();
  //---------------------------------------------------------------
  cout << "**************************************************" << endl;
  cout << "Fitting call(1) from: DrawBgdwFit" << endl;
  TF1 *tBgdFit, *tBgdFitDraw;
  int tPower = 6;
  tBgdFit = FitBackground(tCf, tPower, 0., aMaxBgdFit);
  cout << "**************************************************" << endl;
  if(aAnType==kLamKchM || aAnType==kALamKchP) //Want to draw fit function without gaping gap where Omega peak omitted
  {
    gRejectOmegaTherm=false;
    tBgdFitDraw = new TF1(tBgdFit->GetName()+TString("_Draw"), FitFunctionPolynomial, 0., 3., 7);
    for(int i=0; i<tBgdFit->GetNpar(); i++) tBgdFitDraw->SetParameter(i, tBgdFit->GetParameter(i));
  }
  else tBgdFitDraw = tBgdFit;

  tBgdFitDraw->SetLineColor(tCf->GetLineColor());
  tBgdFitDraw->SetRange(0., aMaxBgdFit);
  tBgdFitDraw->Draw("lsame");

  TLine* tLine = new TLine(0, 1, 2, 1);
  tLine->SetLineColor(14);
  tLine->Draw();

  PrintInfo((TPad*)tCanBgdwFit, tOverallDescriptor, 0.04);
  PrintFitParams((TPad*)tCanBgdwFit, tBgdFitDraw, 0.035, tColor);

  //---------------------------------------------------------------
  TLegend* tLeg = new TLegend(0.55, 0.15, 0.90, 0.30);
    tLeg->SetFillColor(0);
    tLeg->SetBorderSize(0);
    tLeg->SetTextAlign(22);

  tLeg->AddEntry(tCf, tDescriptor.Data());

  //---------------------------------------------------------------

  CentralityType tCentTypeData = GetCentralityType(aImpactParam);

  TH1D* tData = GetQuickData(aAnType, tCentTypeData, aCombineConjugates);
  ThermCf::SetStyleAndColor(tData, 20, GetColor(aAnType));

//  cout << "**************************************************" << endl;
//  cout << "Fitting call(2) from: DrawBgdwFit" << endl;
  TF1 *tBgdFitData, *tBgdFitDataDraw;
//  tBgdFitData = FitBackgroundwNorm(tBgdFit, tData, tPower, 0.6, 0.9);
  tBgdFitData = FitBackgroundwNormAndOffset(tBgdFit, tData, tPower, 0.32, 0.80);
//  cout << "**************************************************" << endl;
  if(aAnType==kLamKchM || aAnType==kALamKchP) //Want to draw fit function without gaping gap where Omega peak omitted
  {
    gRejectOmegaTherm=false;
//    tBgdFitDataDraw = new TF1(tBgdFitData->GetName()+TString("_Draw"), FitFunctionPolynomialwNorm, 0., 3., 8);
    tBgdFitDataDraw = new TF1(tBgdFitData->GetName()+TString("_Draw"), FitFunctionPolynomialwNormAndOffset, 0., 3., 9);
    for(int i=0; i<tBgdFitData->GetNpar(); i++) tBgdFitDataDraw->SetParameter(i, tBgdFitData->GetParameter(i));
  }
  else tBgdFitDataDraw = tBgdFitData;

  tBgdFitDataDraw->SetLineColor(tData->GetLineColor());
  tBgdFitDataDraw->SetRange(0., aMaxBgdFit);

  tData->Draw("same");
  tBgdFitDataDraw->Draw("lsame");

  tCf->Draw("same");
  tBgdFitDraw->Draw("lsame");

  tLeg->AddEntry(tData, TString::Format("Data (%s)", cPrettyCentralityTags[tCentTypeData]));
  //-------------------------------------
  TLatex* tTex = new TLatex();
  tTex->SetTextAlign(12);
  tTex->SetLineWidth(2);
  tTex->SetTextFont(42);
  tTex->SetTextSize(0.035);

  tTex->DrawLatex(0.15, 0.88, TString::Format("Bgd = %0.3f*#color[%d]{Bgd}" , tBgdFitData->GetParameter(7), tColor));
  //---------------------------------------------------------------

  tLeg->Draw();
  return tCanBgdwFit;
}

//________________________________________________________________________________________________________________
TF1* GetLamKchPMBgdFit(TString aCfDescriptor, TH1* aCf, double aMaxBgdFit=3.0)
{
  gRejectOmegaTherm=true;
  if(aCfDescriptor.EqualTo("PrimaryOnly") || aCfDescriptor.EqualTo("PrimaryAndShortDecays")) gRejectOmegaTherm=false;  //In this case, Omega peak not present
  //-------------------------------------------------
  cout << "**************************************************" << endl;
  cout << "Fitting call from: GetLamKchPMBgdFit" << endl;
  TF1 *tBgdFit;
  int tPower = 6;
  tBgdFit = FitBackground(aCf, tPower, 0., aMaxBgdFit);
  cout << "**************************************************" << endl;

  gRejectOmegaTherm=false;
  //---------------------------------------------------------------

  return tBgdFit;
}

//________________________________________________________________________________________________________________
TF1* GetLamKchPMBgdFit(TString aCfDescriptor, TString aFileNameCfs, int aImpactParam, ThermEventsType aEventsType, int aRebin, double aMinNorm, double aMaxNorm, double aMaxBgdFit=3.0, bool aUseStavCf=false)
{
  gRejectOmegaTherm=true;
  if(aCfDescriptor.EqualTo("PrimaryOnly") || aCfDescriptor.EqualTo("PrimaryAndShortDecays")) gRejectOmegaTherm=false;  //In this case, Omega peak not present
  //-------------------------------------------------
  CentralityType tCentType = GetCentralityType(aImpactParam);
  //-------------------------------------------------

  //--------------------------------------------
  TH1 *tCf = ThermCf::GetLamKchPMCombinedThermCfs(aFileNameCfs, aCfDescriptor, tCentType, aEventsType, aRebin, aMinNorm, aMaxNorm, 20, kBlack, aUseStavCf);
  //---------------------------------------------------------------
  cout << "**************************************************" << endl;
  cout << "Fitting call from: GetLamKchPMBgdFit" << endl;
  TF1 *tBgdFit;
  int tPower = 6;
  tBgdFit = FitBackground(tCf, tPower, 0., aMaxBgdFit);
  cout << "**************************************************" << endl;

  gRejectOmegaTherm=false;
  //---------------------------------------------------------------

  return tBgdFit;
}

//________________________________________________________________________________________________________________
void BuildBgdwFitPanel(CanvasPartition* aCanPart, int tColumn, int tRow, TString aCfDescriptor, TString aFileNameCfs, AnalysisType aAnType, int aImpactParam, bool aCombineConjugates, bool aCombineImpactParams, ThermEventsType aEventsType, int aRebin, double aMinNorm, double aMaxNorm, double aMaxBgdFit=3.0, bool aAvgLamKchPMFit=false, bool aUseStavCf=false, bool aZoomY=false, bool aShiftText=false)
{
  if(aAnType==kLamKchM || aAnType==kALamKchP) gRejectOmegaTherm=true;
  else gRejectOmegaTherm=false;
  if(aCfDescriptor.EqualTo("PrimaryOnly") || aCfDescriptor.EqualTo("PrimaryAndShortDecays")) gRejectOmegaTherm=false;  //In this case, Omega peak not present
  //-------------------------------------------------
  CentralityType tCentType=kMB;
  if(aCombineImpactParams) tCentType = GetCentralityType(aImpactParam);
  //-------------------------------------------------
  AnalysisType tConjAnType = GetConjAnType(aAnType);
  //-------------------------------------------------
//  TString tDescriptor = "THERM. Bgd (w. Fit)";
  TString tDescriptor = "THERM. Bgd.";

  TString tOverallDescriptor = cAnalysisRootTags[aAnType];
  if(aCombineConjugates) tOverallDescriptor += TString::Format(" & %s", cAnalysisRootTags[tConjAnType]);
  if(!aCombineImpactParams) tOverallDescriptor += TString::Format(" (b=%d)", aImpactParam);
  else tOverallDescriptor += TString::Format(" (%s)", cPrettyCentralityTags[tCentType]);

  int tMarkerStyle = 26;
  int tColor = kGreen+1;

  //--------------------------------------------
  ThermCf* tThermCf = new ThermCf(aFileNameCfs, aCfDescriptor, aAnType, GetCentralityType(aImpactParam), aCombineConjugates, aEventsType, aRebin, aMinNorm, aMaxNorm, aUseStavCf);
  if(!aCombineImpactParams) tThermCf->SetSpecificImpactParam(aImpactParam, aCombineImpactParams);
  TH1* tCf = tThermCf->GetThermCf(tMarkerStyle, tColor, 0.75);
//-------------------------------------------------------------------------------
  cout << "**************************************************" << endl;
  cout << "Fitting call(1) from: DrawBgdwFit" << endl;
  TF1 *tBgdFit, *tBgdFitDraw;
  int tPower = 6;
  
  if((aAnType==kLamKchP || aAnType==kALamKchM || aAnType==kLamKchM || aAnType==kALamKchP) && aAvgLamKchPMFit)
  {
    tCf = ThermCf::GetLamKchPMCombinedThermCfs(aFileNameCfs, aCfDescriptor, tCentType, aEventsType, aRebin, aMinNorm, aMaxNorm, 20, kBlack, aUseStavCf);
    tBgdFit = GetLamKchPMBgdFit(aCfDescriptor, aFileNameCfs, aImpactParam, aEventsType, aRebin, aMinNorm, aMaxNorm, aMaxBgdFit, aUseStavCf);
  }
  else tBgdFit = FitBackground(tCf, tPower, 0., aMaxBgdFit);
  cout << "**************************************************" << endl;
  if(aAnType==kLamKchM || aAnType==kALamKchP) //Want to draw fit function without gaping gap where Omega peak omitted
  {
    gRejectOmegaTherm=false;
    tBgdFitDraw = new TF1(tBgdFit->GetName()+TString("_Draw"), FitFunctionPolynomial, 0., 3., 7);
    for(int i=0; i<tBgdFit->GetNpar(); i++) tBgdFitDraw->SetParameter(i, tBgdFit->GetParameter(i));
  }
  else tBgdFitDraw = tBgdFit;

  tBgdFitDraw->SetLineColor(tCf->GetLineColor());
  tBgdFitDraw->SetLineWidth(2.0);
  tBgdFitDraw->SetLineStyle(2);
  tBgdFitDraw->SetRange(0., aMaxBgdFit);

  //---------------------------------------------------------------


  CentralityType tCentTypeData = GetCentralityType(aImpactParam);
/*
  TH1D* tData = GetQuickData(aAnType, tCentTypeData, aCombineConjugates, "20180505", aRebin);
  ThermCf::SetStyleAndColor(tData, 20, GetColor(aAnType));
  TH1D* tDataSys = (TH1D*)tData->Clone("tDataSys");
*/

  TObjArray* tDataStatAndSys = GetSlowDataWithSysErrs(aAnType, tCentTypeData, aCombineConjugates, "20190319", aRebin);
  TH1D* tData = (TH1D*)tDataStatAndSys->At(0);
    ThermCf::SetStyleAndColor(tData, 20, GetColor(aAnType));
  TH1D* tDataSys = (TH1D*)tDataStatAndSys->At(1);


  //Set proper attributes for drawing systematics
  ThermCf::SetStyleAndColor(tDataSys, 20, TColor::GetColorTransparent(GetColor(aAnType), 0.2));
  tDataSys->SetFillColor(TColor::GetColorTransparent(GetColor(aAnType), 0.2));
  tDataSys->SetFillStyle(1000);
  tDataSys->SetLineColor(0);
  tDataSys->SetLineWidth(0);


//  cout << "**************************************************" << endl;
//  cout << "Fitting call(2) from: DrawBgdwFit" << endl;
  TF1 *tBgdFitData, *tBgdFitDataDraw;
//  tBgdFitData = FitBackgroundwNorm(tBgdFit, tData, tPower, 0.6, 0.9);
  tBgdFitData = FitBackgroundwNormAndOffset(tBgdFit, tData, tPower, 0.32, 0.80);
//  cout << "**************************************************" << endl;
  if(aAnType==kLamKchM || aAnType==kALamKchP) //Want to draw fit function without gaping gap where Omega peak omitted
  {
    gRejectOmegaTherm=false;
//    tBgdFitDataDraw = new TF1(tBgdFitData->GetName()+TString("_Draw"), FitFunctionPolynomialwNorm, 0., 3., 8);
    tBgdFitDataDraw = new TF1(tBgdFitData->GetName()+TString("_Draw"), FitFunctionPolynomialwNormAndOffset, 0., 3., 9);
    for(int i=0; i<tBgdFitData->GetNpar(); i++) tBgdFitDataDraw->SetParameter(i, tBgdFitData->GetParameter(i));
  }
  else tBgdFitDataDraw = tBgdFitData;

  tBgdFitDataDraw->SetLineColor(tData->GetLineColor());
  tBgdFitDataDraw->SetLineWidth(2.0);
  tBgdFitDataDraw->SetRange(0., aMaxBgdFit);

  //---------------------------------------------------------------------------------------------------------

  double tMarkerSize = 2.0;
  aCanPart->AddGraph(tColumn, tRow, tCf, "", tMarkerStyle, tColor, tMarkerSize, "ex0");  //ex0 suppresses the error along x
  aCanPart->AddGraph(tColumn, tRow, tBgdFitDraw, "", 20, tColor, tMarkerSize, "same l");
  aCanPart->AddGraph(tColumn, tRow, tDataSys, "", 20, TColor::GetColorTransparent(GetColor(aAnType), 0.2), tMarkerSize, "e2psame");
  aCanPart->AddGraph(tColumn, tRow, tData, "", 20, GetColor(aAnType), tMarkerSize, "ex0same");
  aCanPart->AddGraph(tColumn, tRow, tBgdFitDataDraw, "", 20, GetColor(aAnType), tMarkerSize, "same l");

  aCanPart->AddGraph(tColumn, tRow, tCf, "", tMarkerStyle, tColor, tMarkerSize, "ex0same");  //draw again so on top
  aCanPart->AddGraph(tColumn, tRow, tBgdFitDataDraw, "", 20, GetColor(aAnType), tMarkerSize, "same l");
  aCanPart->AddGraph(tColumn, tRow, tBgdFitDraw, "", 20, tColor, tMarkerSize, "same l");
  //---------------------------------------------------------------------------------------------------------
  TString tSysInfoTString;
  if(aCombineConjugates) tSysInfoTString = TString::Format("%s#scale[0.5]{ }#oplus#scale[0.5]{ }%s,  %s", cAnalysisRootTags[aAnType], cAnalysisRootTags[aAnType+1], cPrettyCentralityTags[tCentTypeData]);
  else                   tSysInfoTString = TString::Format("%s,  %s", cAnalysisRootTags[aAnType], cPrettyCentralityTags[tCentTypeData]);
  TPaveText* tSysInfoPaveText;
  tSysInfoPaveText = aCanPart->SetupTPaveText(tSysInfoTString, tColumn, tRow, 0.175, 0.80, 0.80, 0.195, 63, 55, 22, true);
  aCanPart->AddPadPaveText(tSysInfoPaveText, tColumn, tRow);

  //---------------------------------------------------------------------------------------------------------
  if(!aZoomY)
  {
    double tPar0, tPar1, tPar2, tPar3, tPar4, tPar5, tPar6;
    tPar0 = tBgdFitDraw->GetParameter(0);
    tPar1 = tBgdFitDraw->GetParameter(1);
    tPar2 = tBgdFitDraw->GetParameter(2);
    tPar3 = tBgdFitDraw->GetParameter(3);
    tPar4 = tBgdFitDraw->GetParameter(4);
    tPar5 = tBgdFitDraw->GetParameter(5);
    tPar6 = tBgdFitDraw->GetParameter(6);

    double tScaleX = aCanPart->GetXScaleFactor(tColumn, tRow);
    double tScaleY = aCanPart->GetYScaleFactor(tColumn, tRow);
    double tTexSize = 0.035*(tScaleY/tScaleX);
    TLatex* tTex1 = new TLatex(0.15, 0.92, TString::Format("#color[%i]{Bgd} = %0.3f + %0.3fx + %0.3fx^{2} + ...", tColor, tPar0, tPar1, tPar2));
    TLatex* tTex2 = new TLatex(0.30, 0.91, TString::Format("... + %0.3fx^{3} + %0.3fx^{4} + ..." , tPar3, tPar4));
    TLatex* tTex3 = new TLatex(0.30, 0.90, TString::Format("... + %0.3fx^{5} + %0.3fx^{6} + ..." , tPar5, tPar6));

    tTex1->SetTextAlign(12);
    tTex1->SetLineWidth(2);
    tTex1->SetTextFont(42);
    tTex1->SetTextSize(tTexSize);

    tTex2->SetTextAlign(12);
    tTex2->SetLineWidth(2);
    tTex2->SetTextFont(42);
    tTex2->SetTextSize(tTexSize);

    tTex3->SetTextAlign(12);
    tTex3->SetLineWidth(2);
    tTex3->SetTextFont(42);
    tTex3->SetTextSize(tTexSize);

    aCanPart->AddPadPaveLatex(tTex1, tColumn, tRow);
    aCanPart->AddPadPaveLatex(tTex2, tColumn, tRow);
    aCanPart->AddPadPaveLatex(tTex3, tColumn, tRow);
  }
  //---------------------------------------------------------------------------------------------------------
  if(tRow==0)
  {
    if(aShiftText)
    {
      if(!aZoomY) aCanPart->SetupTLegend("", tColumn, tRow, 0.425, 0.15, 0.35, 0.15, 1, true);
      else        aCanPart->SetupTLegend("", tColumn, tRow, 0.30, 0.495, 0.575, 0.30, 1, true);
    }
    else
    {
      if(!aZoomY) aCanPart->SetupTLegend("", tColumn, tRow, 0.55, 0.15, 0.35, 0.15, 1, true);
      else        aCanPart->SetupTLegend("", tColumn, tRow, 0.30, 0.495, 0.65, 0.325, 1, true);
    }
    aCanPart->AddLegendEntry(tColumn, tRow, tCf, tDescriptor.Data(), "p");
    aCanPart->AddLegendEntry(tColumn, tRow, tData, "ALICE", "p");
  }

  if(tRow==(aCanPart->GetNy()-1) && tColumn==0)
  {
    TString tAliceInfo = TString("ALICE Preliminary");
    TPaveText* tAliceInfoText = aCanPart->SetupTPaveText(tAliceInfo, tColumn, tRow, 0.0, 0.025, 1.0, 0.20, 43, 55, 22, true);
    aCanPart->AddPadPaveText(tAliceInfoText, tColumn, tRow);
  }
  if(tRow==(aCanPart->GetNy()-1) && tColumn==1)
  {
    TString tAliceInfo = TString("Pb-Pb #sqrt{#it{s}_{NN}} = 2.76 TeV");
    TPaveText* tAliceInfoText = aCanPart->SetupTPaveText(tAliceInfo, tColumn, tRow, 0.0, 0.025, 1.0, 0.20, 43, 55, 22, true);
    aCanPart->AddPadPaveText(tAliceInfoText, tColumn, tRow);
  }

}
/*
//________________________________________________________________________________________________________________
void BuildBgdwFitPanel(CanvasPartition* aCanPart, int tColumn, int tRow, TString aCfDescriptor, TString aFileNameCfs, AnalysisType aAnType, int aImpactParam, bool aCombineConjugates, bool aCombineImpactParams, ThermEventsType aEventsType, td1dVec &aCustomBins, double aMinNorm, double aMaxNorm, double aMaxBgdFit=3.0, bool aAvgLamKchPMFit=false, bool aUseStavCf=false, bool aZoomY=false, bool aShiftText=false)
{
  if(aAnType==kLamKchM || aAnType==kALamKchP) gRejectOmegaTherm=true;
  else gRejectOmegaTherm=false;
  if(aCfDescriptor.EqualTo("PrimaryOnly") || aCfDescriptor.EqualTo("PrimaryAndShortDecays")) gRejectOmegaTherm=false;  //In this case, Omega peak not present
  //-------------------------------------------------
  CentralityType tCentType=kMB;
  if(aCombineImpactParams) tCentType = GetCentralityType(aImpactParam);
  //-------------------------------------------------
  AnalysisType tConjAnType = GetConjAnType(aAnType);
  //-------------------------------------------------
//  TString tDescriptor = "THERM. Bgd (w. Fit)";
  TString tDescriptor = "THERM. Bgd.";

  TString tOverallDescriptor = cAnalysisRootTags[aAnType];
  if(aCombineConjugates) tOverallDescriptor += TString::Format(" & %s", cAnalysisRootTags[tConjAnType]);
  if(!aCombineImpactParams) tOverallDescriptor += TString::Format(" (b=%d)", aImpactParam);
  else tOverallDescriptor += TString::Format(" (%s)", cPrettyCentralityTags[tCentType]);

  int tMarkerStyle = 26;
  int tColor = kGreen+1;

  //--------------------------------------------
  //aCustomBins will only go out to 2 GeV/c for the experimental data, but the THERM data go out to 3 GeV/c
  assert(aCustomBins[aCustomBins.size()-1]==2.00);
  td1dVec aCustomBinsLong(aCustomBins.size());
  for(int i=0; i<aCustomBins.size(); i++) aCustomBinsLong[i]=aCustomBins[i];
  for(int i=1; i<=10; i++) aCustomBinsLong.push_back(aCustomBins[aCustomBins.size()-1] + 0.10*i);
  assert(aCustomBinsLong[aCustomBinsLong.size()-1]==3.00);

  //--------------------------------------------
  ThermCf* tThermCf = new ThermCf(aFileNameCfs, aCfDescriptor, aAnType, GetCentralityType(aImpactParam), aCombineConjugates, aEventsType, 1, aMinNorm, aMaxNorm, aUseStavCf);
  if(!aCombineImpactParams) tThermCf->SetSpecificImpactParam(aImpactParam, aCombineImpactParams);
  CfHeavy* tThermCfHeavy = tThermCf->GetThermCfHeavy();
  tThermCfHeavy->Rebin((int)aCustomBinsLong.size()-1, aCustomBinsLong);
  TH1* tCf = tThermCfHeavy->GetHeavyCfClone();
//-------------------------------------------------------------------------------
  cout << "**************************************************" << endl;
  cout << "Fitting call(1) from: DrawBgdwFit" << endl;
  TF1 *tBgdFit, *tBgdFitDraw;
  int tPower = 6;
  
  if((aAnType==kLamKchP || aAnType==kALamKchM || aAnType==kLamKchM || aAnType==kALamKchP) && aAvgLamKchPMFit)
  {
    tThermCfHeavy = ThermCf::GetLamKchPMCombinedThermCfsHeavy(aFileNameCfs, aCfDescriptor, tCentType, aEventsType, 1, aMinNorm, aMaxNorm, aUseStavCf);
    tThermCfHeavy->Rebin((int)aCustomBinsLong.size()-1, aCustomBinsLong);
    tCf = tThermCfHeavy->GetHeavyCfClone();
    tBgdFit = GetLamKchPMBgdFit(aCfDescriptor, tCf, aMaxBgdFit);
  }
  else tBgdFit = FitBackground(tCf, tPower, 0., aMaxBgdFit);
  cout << "**************************************************" << endl;
  if(aAnType==kLamKchM || aAnType==kALamKchP) //Want to draw fit function without gaping gap where Omega peak omitted
  {
    gRejectOmegaTherm=false;
    tBgdFitDraw = new TF1(tBgdFit->GetName()+TString("_Draw"), FitFunctionPolynomial, 0., 3., 7);
    for(int i=0; i<tBgdFit->GetNpar(); i++) tBgdFitDraw->SetParameter(i, tBgdFit->GetParameter(i));
  }
  else tBgdFitDraw = tBgdFit;

  tBgdFitDraw->SetLineColor(tCf->GetLineColor());
  tBgdFitDraw->SetLineWidth(2.0);
  tBgdFitDraw->SetLineStyle(2);
  tBgdFitDraw->SetRange(0., aMaxBgdFit);

  //---------------------------------------------------------------


  CentralityType tCentTypeData = GetCentralityType(aImpactParam);

//  TH1D* tData = GetQuickData(aAnType, tCentTypeData, aCombineConjugates, "20180505", 1);
//  ThermCf::SetStyleAndColor(tData, 20, GetColor(aAnType));
//  TH1D* tDataSys = (TH1D*)tData->Clone("tDataSys");

//  double tOGBinWidth = tData->GetBinWidth(1);
//  tData = (TH1D*)tData->Rebin((int)aCustomBins.size()-1, TString::Format("%s_CustomRebin", tData->GetName()), aCustomBins.data());
//  ScaleCustomRebinnedCf(tOGBinWidth, tData, aCustomBins);

//  double tOGBinWidthSys = tDataSys->GetBinWidth(1);
//  tDataSys = (TH1D*)tDataSys->Rebin((int)aCustomBins.size()-1, TString::Format("%s_CustomRebin", tDataSys->GetName()), aCustomBins.data());
//  ScaleCustomRebinnedCf(tOGBinWidthSys, tDataSys, aCustomBins);


  TObjArray* tDataStatAndSys = GetSlowDataWithSysErrs(aAnType, tCentTypeData, aCombineConjugates, "20190319", aCustomBins);
  TH1D* tData = (TH1D*)tDataStatAndSys->At(0);
    ThermCf::SetStyleAndColor(tData, 20, GetColor(aAnType));
  TH1D* tDataSys = (TH1D*)tDataStatAndSys->At(1);


  //Set proper attributes for drawing systematics
  ThermCf::SetStyleAndColor(tDataSys, 20, TColor::GetColorTransparent(GetColor(aAnType), 0.2));
  tDataSys->SetFillColor(TColor::GetColorTransparent(GetColor(aAnType), 0.2));
  tDataSys->SetFillStyle(1000);
  tDataSys->SetLineColor(0);
  tDataSys->SetLineWidth(0);


//  cout << "**************************************************" << endl;
//  cout << "Fitting call(2) from: DrawBgdwFit" << endl;
  TF1 *tBgdFitData, *tBgdFitDataDraw;
//  tBgdFitData = FitBackgroundwNorm(tBgdFit, tData, tPower, 0.6, 0.9);
  tBgdFitData = FitBackgroundwNormAndOffset(tBgdFit, tData, tPower, 0.32, 0.80);
//  cout << "**************************************************" << endl;
  if(aAnType==kLamKchM || aAnType==kALamKchP) //Want to draw fit function without gaping gap where Omega peak omitted
  {
    gRejectOmegaTherm=false;
//    tBgdFitDataDraw = new TF1(tBgdFitData->GetName()+TString("_Draw"), FitFunctionPolynomialwNorm, 0., 3., 8);
    tBgdFitDataDraw = new TF1(tBgdFitData->GetName()+TString("_Draw"), FitFunctionPolynomialwNormAndOffset, 0., 3., 9);
    for(int i=0; i<tBgdFitData->GetNpar(); i++) tBgdFitDataDraw->SetParameter(i, tBgdFitData->GetParameter(i));
  }
  else tBgdFitDataDraw = tBgdFitData;

  tBgdFitDataDraw->SetLineColor(tData->GetLineColor());
  tBgdFitDataDraw->SetLineWidth(2.0);
  tBgdFitDataDraw->SetRange(0., aMaxBgdFit);

  //---------------------------------------------------------------------------------------------------------

  double tMarkerSize = 2.0;
  aCanPart->AddGraph(tColumn, tRow, tCf, "", tMarkerStyle, tColor, tMarkerSize, "ex0");  //ex0 suppresses the error along x
  aCanPart->AddGraph(tColumn, tRow, tBgdFitDraw, "", 20, tColor, tMarkerSize, "same l");
  aCanPart->AddGraph(tColumn, tRow, tDataSys, "", 20, TColor::GetColorTransparent(GetColor(aAnType), 0.2), tMarkerSize, "e2psame");
  aCanPart->AddGraph(tColumn, tRow, tData, "", 20, GetColor(aAnType), tMarkerSize, "ex0same");
  aCanPart->AddGraph(tColumn, tRow, tBgdFitDataDraw, "", 20, GetColor(aAnType), tMarkerSize, "same l");

  aCanPart->AddGraph(tColumn, tRow, tCf, "", tMarkerStyle, tColor, tMarkerSize, "ex0same");  //draw again so on top
  aCanPart->AddGraph(tColumn, tRow, tBgdFitDataDraw, "", 20, GetColor(aAnType), tMarkerSize, "same l");
  aCanPart->AddGraph(tColumn, tRow, tBgdFitDraw, "", 20, tColor, tMarkerSize, "same l");
  //---------------------------------------------------------------------------------------------------------
  TString tSysInfoTString;
  if(aCombineConjugates) tSysInfoTString = TString::Format("%s#scale[0.5]{ }#oplus#scale[0.5]{ }%s,  %s", cAnalysisRootTags[aAnType], cAnalysisRootTags[aAnType+1], cPrettyCentralityTags[tCentTypeData]);
  else                   tSysInfoTString = TString::Format("%s,  %s", cAnalysisRootTags[aAnType], cPrettyCentralityTags[tCentTypeData]);
  TPaveText* tSysInfoPaveText;
  tSysInfoPaveText = aCanPart->SetupTPaveText(tSysInfoTString, tColumn, tRow, 0.175, 0.80, 0.80, 0.195, 63, 55, 22, true);
  aCanPart->AddPadPaveText(tSysInfoPaveText, tColumn, tRow);

  //---------------------------------------------------------------------------------------------------------
  if(!aZoomY)
  {
    double tPar0, tPar1, tPar2, tPar3, tPar4, tPar5, tPar6;
    tPar0 = tBgdFitDraw->GetParameter(0);
    tPar1 = tBgdFitDraw->GetParameter(1);
    tPar2 = tBgdFitDraw->GetParameter(2);
    tPar3 = tBgdFitDraw->GetParameter(3);
    tPar4 = tBgdFitDraw->GetParameter(4);
    tPar5 = tBgdFitDraw->GetParameter(5);
    tPar6 = tBgdFitDraw->GetParameter(6);

    double tScaleX = aCanPart->GetXScaleFactor(tColumn, tRow);
    double tScaleY = aCanPart->GetYScaleFactor(tColumn, tRow);
    double tTexSize = 0.035*(tScaleY/tScaleX);
    TLatex* tTex1 = new TLatex(0.15, 0.92, TString::Format("#color[%i]{Bgd} = %0.3f + %0.3fx + %0.3fx^{2} + ...", tColor, tPar0, tPar1, tPar2));
    TLatex* tTex2 = new TLatex(0.30, 0.91, TString::Format("... + %0.3fx^{3} + %0.3fx^{4} + ..." , tPar3, tPar4));
    TLatex* tTex3 = new TLatex(0.30, 0.90, TString::Format("... + %0.3fx^{5} + %0.3fx^{6} + ..." , tPar5, tPar6));

    tTex1->SetTextAlign(12);
    tTex1->SetLineWidth(2);
    tTex1->SetTextFont(42);
    tTex1->SetTextSize(tTexSize);

    tTex2->SetTextAlign(12);
    tTex2->SetLineWidth(2);
    tTex2->SetTextFont(42);
    tTex2->SetTextSize(tTexSize);

    tTex3->SetTextAlign(12);
    tTex3->SetLineWidth(2);
    tTex3->SetTextFont(42);
    tTex3->SetTextSize(tTexSize);

    aCanPart->AddPadPaveLatex(tTex1, tColumn, tRow);
    aCanPart->AddPadPaveLatex(tTex2, tColumn, tRow);
    aCanPart->AddPadPaveLatex(tTex3, tColumn, tRow);
  }
  //---------------------------------------------------------------------------------------------------------
  if(tRow==0)
  {
    if(aShiftText)
    {
      if(!aZoomY) aCanPart->SetupTLegend("", tColumn, tRow, 0.425, 0.15, 0.35, 0.15, 1, true);
      else        aCanPart->SetupTLegend("", tColumn, tRow, 0.30, 0.495, 0.575, 0.30, 1, true);
    }
    else
    {
      if(!aZoomY) aCanPart->SetupTLegend("", tColumn, tRow, 0.55, 0.15, 0.35, 0.15, 1, true);
      else        aCanPart->SetupTLegend("", tColumn, tRow, 0.30, 0.495, 0.65, 0.325, 1, true);
    }
    aCanPart->AddLegendEntry(tColumn, tRow, tCf, tDescriptor.Data(), "p");
    aCanPart->AddLegendEntry(tColumn, tRow, tData, "ALICE", "p");
  }

  if(tRow==(aCanPart->GetNy()-1) && tColumn==0)
  {
    TString tAliceInfo = TString("ALICE Preliminary");
    TPaveText* tAliceInfoText = aCanPart->SetupTPaveText(tAliceInfo, tColumn, tRow, 0.0, 0.025, 1.0, 0.20, 43, 55, 22, true);
    aCanPart->AddPadPaveText(tAliceInfoText, tColumn, tRow);
  }
  if(tRow==(aCanPart->GetNy()-1) && tColumn==1)
  {
    TString tAliceInfo = TString("Pb-Pb #sqrt{#it{s}_{NN}} = 2.76 TeV");
    TPaveText* tAliceInfoText = aCanPart->SetupTPaveText(tAliceInfo, tColumn, tRow, 0.0, 0.025, 1.0, 0.20, 43, 55, 22, true);
    aCanPart->AddPadPaveText(tAliceInfoText, tColumn, tRow);
  }

}
*/

//________________________________________________________________________________________________________________
void BuildBgdwFitPanel(CanvasPartition* aCanPart, int tColumn, int tRow, TString aCfDescriptor, TString aFileNameCfs, AnalysisType aAnType, int aImpactParam, bool aCombineConjugates, bool aCombineImpactParams, ThermEventsType aEventsType, td1dVec &aCustomBins, double aMinNorm, double aMaxNorm, double aMaxBgdFit=3.0, bool aAvgLamKchPMFit=false, bool aUseStavCf=false, bool aZoomY=false, bool aShiftText=false)
{
  if(aAnType==kLamKchM || aAnType==kALamKchP) gRejectOmegaTherm=true;
  else gRejectOmegaTherm=false;
  if(aCfDescriptor.EqualTo("PrimaryOnly") || aCfDescriptor.EqualTo("PrimaryAndShortDecays")) gRejectOmegaTherm=false;  //In this case, Omega peak not present
  //-------------------------------------------------
  CentralityType tCentType=kMB;
  if(aCombineImpactParams) tCentType = GetCentralityType(aImpactParam);
  //-------------------------------------------------
  AnalysisType tConjAnType = GetConjAnType(aAnType);
  //-------------------------------------------------
//  TString tDescriptor = "THERM. Bgd (w. Fit)";
  TString tDescriptor = "THERM. Bgd.";

  TString tOverallDescriptor = cAnalysisRootTags[aAnType];
  if(aCombineConjugates) tOverallDescriptor += TString::Format(" & %s", cAnalysisRootTags[tConjAnType]);
  if(!aCombineImpactParams) tOverallDescriptor += TString::Format(" (b=%d)", aImpactParam);
  else tOverallDescriptor += TString::Format(" (%s)", cPrettyCentralityTags[tCentType]);

  int tMarkerStyle = 25;
  int tColor = kGreen+1;

  //--------------------------------------------
  //aCustomBins will only go out to 2 GeV/c for the experimental data, but the THERM data go out to 3 GeV/c
  assert(aCustomBins[aCustomBins.size()-1]==2.00);
  td1dVec aCustomBinsLong(aCustomBins.size());
  for(int i=0; i<aCustomBins.size(); i++) aCustomBinsLong[i]=aCustomBins[i];
  for(int i=1; i<=10; i++) aCustomBinsLong.push_back(aCustomBins[aCustomBins.size()-1] + 0.10*i);
  
  //If LamK0 and PrimaryAndShortDecays, combine the first two bins because the error bars are really big
  if((aAnType==kLamK0 || aAnType==kALamK0) && aCfDescriptor.EqualTo("PrimaryAndShortDecays")) aCustomBinsLong.erase(aCustomBinsLong.begin()+1);
  
  assert(aCustomBinsLong[aCustomBinsLong.size()-1]==3.00);

  //--------------------------------------------
  ThermCf* tThermCf = new ThermCf(aFileNameCfs, aCfDescriptor, aAnType, GetCentralityType(aImpactParam), aCombineConjugates, aEventsType, 1, aMinNorm, aMaxNorm, aUseStavCf);
  if(!aCombineImpactParams) tThermCf->SetSpecificImpactParam(aImpactParam, aCombineImpactParams);
  CfHeavy* tThermCfHeavy = tThermCf->GetThermCfHeavy();
  tThermCfHeavy->Rebin((int)aCustomBinsLong.size()-1, aCustomBinsLong);
  TH1* tCf = tThermCfHeavy->GetHeavyCfClone();
//-------------------------------------------------------------------------------
  cout << "**************************************************" << endl;
  cout << "Fitting call(1) from: DrawBgdwFit" << endl;
  TF1 *tBgdFit, *tBgdFitDraw;
  int tPower = 6;
  
  if((aAnType==kLamKchP || aAnType==kALamKchM || aAnType==kLamKchM || aAnType==kALamKchP) && aAvgLamKchPMFit)
  {
    tThermCfHeavy = ThermCf::GetLamKchPMCombinedThermCfsHeavy(aFileNameCfs, aCfDescriptor, tCentType, aEventsType, 1, aMinNorm, aMaxNorm, aUseStavCf);
    tThermCfHeavy->Rebin((int)aCustomBinsLong.size()-1, aCustomBinsLong);
    tCf = tThermCfHeavy->GetHeavyCfClone();
    tBgdFit = GetLamKchPMBgdFit(aCfDescriptor, tCf, aMaxBgdFit);
  }
  else tBgdFit = FitBackground(tCf, tPower, 0., aMaxBgdFit);
  cout << "**************************************************" << endl;
  if(aAnType==kLamKchM || aAnType==kALamKchP) //Want to draw fit function without gaping gap where Omega peak omitted
  {
    gRejectOmegaTherm=false;
    tBgdFitDraw = new TF1(tBgdFit->GetName()+TString("_Draw"), FitFunctionPolynomial, 0., 3., 7);
    for(int i=0; i<tBgdFit->GetNpar(); i++) tBgdFitDraw->SetParameter(i, tBgdFit->GetParameter(i));
  }
  else tBgdFitDraw = tBgdFit;

  tBgdFitDraw->SetLineColor(tCf->GetLineColor());
  tBgdFitDraw->SetLineWidth(2.0);
  tBgdFitDraw->SetLineStyle(2);
  tBgdFitDraw->SetRange(0., aMaxBgdFit);

  //---------------------------------------------------------------


  CentralityType tCentTypeData = GetCentralityType(aImpactParam);
/*
  TH1D* tData = GetQuickData(aAnType, tCentTypeData, aCombineConjugates, "20180505", 1);
  ThermCf::SetStyleAndColor(tData, 20, GetColor(aAnType));
  TH1D* tDataSys = (TH1D*)tData->Clone("tDataSys");

  double tOGBinWidth = tData->GetBinWidth(1);
  tData = (TH1D*)tData->Rebin((int)aCustomBins.size()-1, TString::Format("%s_CustomRebin", tData->GetName()), aCustomBins.data());
  ScaleCustomRebinnedCf(tOGBinWidth, tData, aCustomBins);

  double tOGBinWidthSys = tDataSys->GetBinWidth(1);
  tDataSys = (TH1D*)tDataSys->Rebin((int)aCustomBins.size()-1, TString::Format("%s_CustomRebin", tDataSys->GetName()), aCustomBins.data());
  ScaleCustomRebinnedCf(tOGBinWidthSys, tDataSys, aCustomBins);
*/

  TObjArray* tDataStatAndSys = GetSlowDataWithSysErrs(aAnType, tCentTypeData, aCombineConjugates, "20190319", aCustomBins);
  TH1D* tData = (TH1D*)tDataStatAndSys->At(0);
    ThermCf::SetStyleAndColor(tData, 20, GetColor(aAnType));
  TH1D* tDataSys = (TH1D*)tDataStatAndSys->At(1);

  //Set proper attributes for drawing systematics
  ThermCf::SetStyleAndColor(tDataSys, 20, TColor::GetColorTransparent(GetColor(aAnType), 0.2));
  tDataSys->SetFillColor(TColor::GetColorTransparent(GetColor(aAnType), 0.2));
  tDataSys->SetFillStyle(1000);
  tDataSys->SetLineColor(0);
  tDataSys->SetLineWidth(0);


//  cout << "**************************************************" << endl;
//  cout << "Fitting call(2) from: DrawBgdwFit" << endl;
  TF1 *tBgdFitData, *tBgdFitDataDraw;
//  tBgdFitData = FitBackgroundwNorm(tBgdFit, tData, tPower, 0.6, 0.9);
  tBgdFitData = FitBackgroundwNormAndOffset(tBgdFit, tData, tPower, 0.32, 0.80);
//  cout << "**************************************************" << endl;
  if(aAnType==kLamKchM || aAnType==kALamKchP) //Want to draw fit function without gaping gap where Omega peak omitted
  {
    gRejectOmegaTherm=false;
//    tBgdFitDataDraw = new TF1(tBgdFitData->GetName()+TString("_Draw"), FitFunctionPolynomialwNorm, 0., 3., 8);
    tBgdFitDataDraw = new TF1(tBgdFitData->GetName()+TString("_Draw"), FitFunctionPolynomialwNormAndOffset, 0., 3., 9);
    for(int i=0; i<tBgdFitData->GetNpar(); i++) tBgdFitDataDraw->SetParameter(i, tBgdFitData->GetParameter(i));
  }
  else tBgdFitDataDraw = tBgdFitData;

  tBgdFitDataDraw->SetLineColor(tData->GetLineColor());
  tBgdFitDataDraw->SetLineWidth(2.0);
  tBgdFitDataDraw->SetRange(0., aMaxBgdFit);

  //---------------------------------------------------------------------------------------------------------

  double tMarkerSize = 2.0;
  aCanPart->AddGraph(tColumn, tRow, tCf, "", tMarkerStyle, tColor, tMarkerSize, "ex0");  //ex0 suppresses the error along x
  aCanPart->AddGraph(tColumn, tRow, tBgdFitDraw, "", 20, tColor, tMarkerSize, "same l");
  aCanPart->AddGraph(tColumn, tRow, tDataSys, "", 20, TColor::GetColorTransparent(GetColor(aAnType), 0.2), tMarkerSize, "e2psame");
  aCanPart->AddGraph(tColumn, tRow, tData, "", 20, GetColor(aAnType), tMarkerSize, "ex0same");
  aCanPart->AddGraph(tColumn, tRow, tBgdFitDataDraw, "", 20, GetColor(aAnType), tMarkerSize, "same l");

  aCanPart->AddGraph(tColumn, tRow, tCf, "", tMarkerStyle, tColor, tMarkerSize, "ex0same");  //draw again so on top
  aCanPart->AddGraph(tColumn, tRow, tBgdFitDataDraw, "", 20, GetColor(aAnType), tMarkerSize, "same l");
  aCanPart->AddGraph(tColumn, tRow, tBgdFitDraw, "", 20, tColor, tMarkerSize, "same l");
  //---------------------------------------------------------------------------------------------------------

  TString tSysInfoTString;
  if(aCombineConjugates) tSysInfoTString = TString::Format("%s#scale[0.5]{ }#oplus#scale[0.5]{ }%s   %s", cAnalysisRootTags[aAnType], cAnalysisRootTags[aAnType+1], cPrettyCentralityTags[tCentTypeData]);
  else                   tSysInfoTString = TString::Format("%s,  %s", cAnalysisRootTags[aAnType], cPrettyCentralityTags[tCentTypeData]);
  TPaveText* tSysInfoPaveText;
  tSysInfoPaveText = aCanPart->SetupTPaveText(tSysInfoTString, tColumn, tRow, 0.185, 0.735, 0.80, 0.195, 43, 55, 31, true);
  aCanPart->AddPadPaveText(tSysInfoPaveText, tColumn, tRow);


  //---------------------------------------------------------------------------------------------------------
  if(!aZoomY)
  {
    double tPar0, tPar1, tPar2, tPar3, tPar4, tPar5, tPar6;
    tPar0 = tBgdFitDraw->GetParameter(0);
    tPar1 = tBgdFitDraw->GetParameter(1);
    tPar2 = tBgdFitDraw->GetParameter(2);
    tPar3 = tBgdFitDraw->GetParameter(3);
    tPar4 = tBgdFitDraw->GetParameter(4);
    tPar5 = tBgdFitDraw->GetParameter(5);
    tPar6 = tBgdFitDraw->GetParameter(6);

    double tScaleX = aCanPart->GetXScaleFactor(tColumn, tRow);
    double tScaleY = aCanPart->GetYScaleFactor(tColumn, tRow);
    double tTexSize = 0.035*(tScaleY/tScaleX);
    TLatex* tTex1 = new TLatex(0.15, 0.92, TString::Format("#color[%i]{Bgd} = %0.3f + %0.3fx + %0.3fx^{2} + ...", tColor, tPar0, tPar1, tPar2));
    TLatex* tTex2 = new TLatex(0.30, 0.91, TString::Format("... + %0.3fx^{3} + %0.3fx^{4} + ..." , tPar3, tPar4));
    TLatex* tTex3 = new TLatex(0.30, 0.90, TString::Format("... + %0.3fx^{5} + %0.3fx^{6} + ..." , tPar5, tPar6));

    tTex1->SetTextAlign(12);
    tTex1->SetLineWidth(2);
    tTex1->SetTextFont(42);
    tTex1->SetTextSize(tTexSize);

    tTex2->SetTextAlign(12);
    tTex2->SetLineWidth(2);
    tTex2->SetTextFont(42);
    tTex2->SetTextSize(tTexSize);

    tTex3->SetTextAlign(12);
    tTex3->SetLineWidth(2);
    tTex3->SetTextFont(42);
    tTex3->SetTextSize(tTexSize);

    aCanPart->AddPadPaveLatex(tTex1, tColumn, tRow);
    aCanPart->AddPadPaveLatex(tTex2, tColumn, tRow);
    aCanPart->AddPadPaveLatex(tTex3, tColumn, tRow);
  }
  //---------------------------------------------------------------------------------------------------------
  if(tRow==0)
  {
    int tNColumns=1;
    if(aShiftText)
    {
      if(!aZoomY) aCanPart->SetupTLegend("", tColumn, tRow, 0.425, 0.15, 0.35, 0.15, tNColumns, true);
      else        aCanPart->SetupTLegend("", tColumn, tRow, 0.30, 0.495, 0.575, 0.30, tNColumns, true);
    }
    else
    {
      if(!aZoomY) aCanPart->SetupTLegend("", tColumn, tRow, 0.55, 0.15, 0.35, 0.15, tNColumns, true);
//      else        aCanPart->SetupTLegend("", tColumn, tRow, 0.30, 0.475, 0.65, 0.375, tNColumns, true);
      else        aCanPart->SetupTLegend("", tColumn, tRow, 0.24, 0.35, 0.70, 0.45, tNColumns, true);  //MOST USED OPTION
    }
/*
    aCanPart->AddLegendEntry(tColumn, tRow, tData, "ALICE, stat. errors", "PE");
    aCanPart->AddLegendEntry(tColumn, tRow, tDataSys, "syst. errors", "F");
    aCanPart->AddLegendEntry(tColumn, tRow, tBgdFitDataDraw, "ALICE Bgd. Fit", "L");   
    aCanPart->AddLegendEntry(tColumn, tRow, tCf, tDescriptor.Data(), "p");
    aCanPart->AddLegendEntry(tColumn, tRow, tBgdFitDraw, "THERM. Bgd. Fit", "L"); 
*/
    aCanPart->AddLegendEntry(tColumn, tRow, tData, "ALICE", "P");
    aCanPart->AddLegendEntry(tColumn, tRow, tBgdFitDataDraw, "Scaled Bgd. Fit", "L");   
    aCanPart->AddLegendEntry(tColumn, tRow, tCf, tDescriptor.Data(), "p");
    aCanPart->AddLegendEntry(tColumn, tRow, tBgdFitDraw, "THERM. Bgd. Fit", "L"); 
    
    TObjArray* tLegArray = ((TObjArray*)aCanPart->GetPadLegends()->At(tColumn + tRow*3));
    ((TLegend*)tLegArray->At(0))->SetTextAlign(12);
               
  }

  if(tRow==(aCanPart->GetNy()-1) && tColumn==0)
  {
    TString tAliceInfo = TString("ALICE");
    TPaveText* tAliceInfoText = aCanPart->SetupTPaveText(tAliceInfo, tColumn, tRow, 0.0, 0.135, 1.0, 0.20, 43, 55, 22, true);
    aCanPart->AddPadPaveText(tAliceInfoText, tColumn, tRow);
  }
  if(tRow==(aCanPart->GetNy()-1) && tColumn==1)
  {
    TString tAliceInfo = TString("Pb-Pb #sqrt{#it{s}_{NN}} = 2.76 TeV");
    TPaveText* tAliceInfoText = aCanPart->SetupTPaveText(tAliceInfo, tColumn, tRow, 0.0, 0.135, 1.0, 0.20, 43, 55, 22, true);
    aCanPart->AddPadPaveText(tAliceInfoText, tColumn, tRow);
  }
  
  //---------------------------------------------------------------------------------------------------------
  // Label panels with letters for PRC
  vector<TString> tLetters{"(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)"};
  int tLettIdx = tColumn + 3*tRow;
  TString tLetter = tLetters[tLettIdx];
  TPaveText* tLetterPT = aCanPart->SetupTPaveText(tLetter, tColumn, tRow, 0.75, 0.060, 0.20, 0.20, 63, 50, 33, true);
  aCanPart->AddPadPaveText(tLetterPT, tColumn, tRow);
  

/*
  FILE* tOutput = stdout;
  double tXAxisHigh = aCanPart->GetAxesRanges()[1];
  HistInfoPrinter::PrintHistInfowStatAndSystYAML(tData, tDataSys, tOutput, 0., tXAxisHigh);
  HistInfoPrinter::PrintHistInfoYAML(tCf, tOutput, 0., tXAxisHigh);  
*/

}



//________________________________________________________________________________________________________________
TCanvas* DrawBgdwFit_AllCentv2(TString aCfDescriptor, TString aFileNameCfs, AnalysisType aAnType, bool aCombineConjugates, ThermEventsType aEventsType, int aRebin, double aMinNorm, double aMaxNorm, double aMaxBgdFit=3.0, bool aAvgLamKchPMFit=false, bool aUseStavCf=false)
{
  bool tCombineImpactParams = true; //Should always be true here
//-------------------------------------------------------------------------------
  TString tCanBgdwFitName = "BgdwFitOnly";

  if(aFileNameCfs.Contains("_RandomEPs_NumWeight1")) tCanBgdwFitName += TString("_RandomEPs_NumWeight1");
  else if(aFileNameCfs.Contains("_RandomEPs")) tCanBgdwFitName += TString("_RandomEPs");
  else tCanBgdwFitName += TString("");

  tCanBgdwFitName += TString::Format("_%s_", aCfDescriptor.Data());
  tCanBgdwFitName += TString(cAnalysisBaseTags[aAnType]);

  if(aCombineConjugates) tCanBgdwFitName += TString("wConj");
  tCanBgdwFitName += TString("_1030_3050");

  if(aAvgLamKchPMFit) tCanBgdwFitName += TString("_AvgLamKchPMFit");

//-------------------------------------------------------------------------------
  int tNx=1;
  int tNy=3;
  double tXLow = 0.;
//  double tXHigh = aMaxBgdFit;
  double tXHigh = 2.0;
  double tYLow = 0.86;
  double tYHigh = 1.07;

  CanvasPartition* tCanPart = new CanvasPartition(tCanBgdwFitName,tNx,tNy,tXLow,tXHigh,tYLow,tYHigh,0.15,0.0025,0.075,0.0025);
  tCanPart->SetDrawOptStat(false);
  tCanPart->GetCanvas()->SetCanvasSize(700, 1500);

  BuildBgdwFitPanel(tCanPart, 0, 0, aCfDescriptor, aFileNameCfs, aAnType, 3, aCombineConjugates, tCombineImpactParams, aEventsType, aRebin, aMinNorm, aMaxNorm, aMaxBgdFit, aAvgLamKchPMFit, aUseStavCf, false);
  BuildBgdwFitPanel(tCanPart, 0, 1, aCfDescriptor, aFileNameCfs, aAnType, 5, aCombineConjugates, tCombineImpactParams, aEventsType, aRebin, aMinNorm, aMaxNorm, aMaxBgdFit, aAvgLamKchPMFit, aUseStavCf, false);
  BuildBgdwFitPanel(tCanPart, 0, 2, aCfDescriptor, aFileNameCfs, aAnType, 8, aCombineConjugates, tCombineImpactParams, aEventsType, aRebin, aMinNorm, aMaxNorm, aMaxBgdFit, aAvgLamKchPMFit, aUseStavCf, false);

  //----- Increase label size on axes
  for(int j=0; j<tNy; j++)
  {
    for(int i=0; i<tNx; i++)
    {
      TH1* tTempHist = (TH1*)(tCanPart->GetGraphsInPad(i,j)->At(0));
      tTempHist->GetXaxis()->SetLabelSize(2.0*tTempHist->GetXaxis()->GetLabelSize());
      tTempHist->GetYaxis()->SetLabelSize(2.0*tTempHist->GetYaxis()->GetLabelSize());
    }

  }


  //-----------------
  tCanPart->SetDrawUnityLine(true);
  tCanPart->DrawAll();
  tCanPart->DrawXaxisTitle("#it{k}* (GeV/#it{c})", 43, 35, 0.75, 0.02); //Note, changing xaxis low (=0.315) does nothing
  tCanPart->DrawYaxisTitle("#it{C}(#it{k}*)", 43, 40, 0.06, 0.925);

  return tCanPart->GetCanvas();
}



//________________________________________________________________________________________________________________
TCanvas* DrawBgdwFit_AllCentAllAnv2(TString aCfDescriptor, TString aFileNameCfs, bool aCombineConjugates, ThermEventsType aEventsType, int aRebin, double aMinNorm, double aMaxNorm, double aMaxBgdFit=3.0, bool aAvgLamKchPMFit=false, bool aUseStavCf=false)
{
  bool tCombineImpactParams = true; //Should always be true here
  bool aZoomY=true;
//-------------------------------------------------------------------------------
  TString tCanBgdwFitName = "BgdwFitOnly";

  if(aFileNameCfs.Contains("_RandomEPs_NumWeight1")) tCanBgdwFitName += TString("_RandomEPs_NumWeight1");
  else if(aFileNameCfs.Contains("_RandomEPs")) tCanBgdwFitName += TString("_RandomEPs");
  else tCanBgdwFitName += TString("");

  tCanBgdwFitName += TString::Format("_%s_", aCfDescriptor.Data());
  tCanBgdwFitName += TString("AllAn");

  if(aCombineConjugates) tCanBgdwFitName += TString("wConj");
  tCanBgdwFitName += TString("_1030_3050");

  if(aAvgLamKchPMFit) tCanBgdwFitName += TString("_AvgLamKchPMFit");

//-------------------------------------------------------------------------------
  int tNx=3;
  int tNy=3;
  double tXLow = -0.04;
//  double tXHigh = aMaxBgdFit-0.02;
  double tXHigh = 1.68;
  double tYLow = 0.9525;
  double tYHigh = 1.01999; 

  CanvasPartition* tCanPart = new CanvasPartition(tCanBgdwFitName,tNx,tNy,tXLow,tXHigh,tYLow,tYHigh,0.11,0.0025,0.10,0.0025);
  tCanPart->SetDrawOptStat(false);
  tCanPart->GetCanvas()->SetCanvasSize(2100, 1500);
  tCanPart->SetAllTicks(1,1);

  BuildBgdwFitPanel(tCanPart, 0, 0, aCfDescriptor, aFileNameCfs, kLamKchP, 3, aCombineConjugates, tCombineImpactParams, aEventsType, aRebin, aMinNorm, aMaxNorm, aMaxBgdFit, aAvgLamKchPMFit, aUseStavCf, aZoomY);
  BuildBgdwFitPanel(tCanPart, 0, 1, aCfDescriptor, aFileNameCfs, kLamKchP, 5, aCombineConjugates, tCombineImpactParams, aEventsType, aRebin, aMinNorm, aMaxNorm, aMaxBgdFit, aAvgLamKchPMFit, aUseStavCf, aZoomY);
  BuildBgdwFitPanel(tCanPart, 0, 2, aCfDescriptor, aFileNameCfs, kLamKchP, 8, aCombineConjugates, tCombineImpactParams, aEventsType, aRebin, aMinNorm, aMaxNorm, aMaxBgdFit, aAvgLamKchPMFit, aUseStavCf, aZoomY);

  BuildBgdwFitPanel(tCanPart, 1, 0, aCfDescriptor, aFileNameCfs, kLamKchM, 3, aCombineConjugates, tCombineImpactParams, aEventsType, aRebin, aMinNorm, aMaxNorm, aMaxBgdFit, aAvgLamKchPMFit, aUseStavCf, aZoomY);
  BuildBgdwFitPanel(tCanPart, 1, 1, aCfDescriptor, aFileNameCfs, kLamKchM, 5, aCombineConjugates, tCombineImpactParams, aEventsType, aRebin, aMinNorm, aMaxNorm, aMaxBgdFit, aAvgLamKchPMFit, aUseStavCf, aZoomY);
  BuildBgdwFitPanel(tCanPart, 1, 2, aCfDescriptor, aFileNameCfs, kLamKchM, 8, aCombineConjugates, tCombineImpactParams, aEventsType, aRebin, aMinNorm, aMaxNorm, aMaxBgdFit, aAvgLamKchPMFit, aUseStavCf, aZoomY);

  BuildBgdwFitPanel(tCanPart, 2, 0, aCfDescriptor, aFileNameCfs, kLamK0, 3, aCombineConjugates, tCombineImpactParams, aEventsType, aRebin, aMinNorm, aMaxNorm, aMaxBgdFit, aAvgLamKchPMFit, aUseStavCf, aZoomY);
  BuildBgdwFitPanel(tCanPart, 2, 1, aCfDescriptor, aFileNameCfs, kLamK0, 5, aCombineConjugates, tCombineImpactParams, aEventsType, aRebin, aMinNorm, aMaxNorm, aMaxBgdFit, aAvgLamKchPMFit, aUseStavCf, aZoomY);
  BuildBgdwFitPanel(tCanPart, 2, 2, aCfDescriptor, aFileNameCfs, kLamK0, 8, aCombineConjugates, tCombineImpactParams, aEventsType, aRebin, aMinNorm, aMaxNorm, aMaxBgdFit, aAvgLamKchPMFit, aUseStavCf, aZoomY);

  //----- Increase label size on axes
  for(int j=0; j<tNy; j++)
  {
    for(int i=0; i<tNx; i++)
    {
      TH1* tTempHist = (TH1*)(tCanPart->GetGraphsInPad(i,j)->At(0));

      tTempHist->GetXaxis()->SetLabelSize(3.25*tTempHist->GetXaxis()->GetLabelSize());
      tTempHist->GetXaxis()->SetLabelOffset(2.5*tTempHist->GetXaxis()->GetLabelOffset());

      tTempHist->GetYaxis()->SetLabelSize(3.25*tTempHist->GetYaxis()->GetLabelSize());
      tTempHist->GetYaxis()->SetLabelOffset(5.0*tTempHist->GetYaxis()->GetLabelOffset());

      if(j==0) tTempHist->GetYaxis()->SetRangeUser(0.9875, 1.015);
      if(j==1) tTempHist->GetYaxis()->SetRangeUser(0.975, 1.018);
    }

  }

  //-----------------
  tCanPart->SetDrawUnityLine(true);
  tCanPart->DrawAll();
  tCanPart->DrawXaxisTitle("#it{k}* (GeV/#it{c})", 43, 75, 0.825, 0.01); //Note, changing xaxis low (=0.315) does nothing
  tCanPart->DrawYaxisTitle("#it{C}(#it{k}*)", 43, 75, 0.045, 0.85);

  return tCanPart->GetCanvas();
}

//________________________________________________________________________________________________________________
TCanvas* DrawBgdwFit_AllCentAllAnv2(TString aCfDescriptor, TString aFileNameCfs, bool aCombineConjugates, ThermEventsType aEventsType, td1dVec &aCustomBins, double aMinNorm, double aMaxNorm, double aMaxBgdFit=3.0, bool aAvgLamKchPMFit=false, bool aUseStavCf=false)
{
  bool tCombineImpactParams = true; //Should always be true here
  bool aZoomY=true;
//-------------------------------------------------------------------------------
  TString tCanBgdwFitName = "BgdwFitOnly";

  if(aFileNameCfs.Contains("_RandomEPs_NumWeight1")) tCanBgdwFitName += TString("_RandomEPs_NumWeight1");
  else if(aFileNameCfs.Contains("_RandomEPs")) tCanBgdwFitName += TString("_RandomEPs");
  else tCanBgdwFitName += TString("");

  tCanBgdwFitName += TString::Format("_%s_", aCfDescriptor.Data());
  tCanBgdwFitName += TString("AllAn");

  if(aCombineConjugates) tCanBgdwFitName += TString("wConj");
  tCanBgdwFitName += TString("_1030_3050");

  if(aAvgLamKchPMFit) tCanBgdwFitName += TString("_AvgLamKchPMFit");

  tCanBgdwFitName += TString("_CustomRebin");

//-------------------------------------------------------------------------------
  int tNx=3;
  int tNy=3;
  double tXLow = -0.04;
//  double tXHigh = aMaxBgdFit-0.02;
  double tXHigh = 1.68;
  //double tYLow = 0.9525;
  //double tYLow = 0.945;
  double tYLow = 0.928;
  double tYHigh = 1.01999;

  CanvasPartition* tCanPart = new CanvasPartition(tCanBgdwFitName,tNx,tNy,tXLow,tXHigh,tYLow,tYHigh,0.11,0.0025,0.10,0.0025);
  tCanPart->SetDrawOptStat(false);
  tCanPart->GetCanvas()->SetCanvasSize(2100, 1500);
  tCanPart->SetAllTicks(1,1);  

  BuildBgdwFitPanel(tCanPart, 0, 0, aCfDescriptor, aFileNameCfs, kLamKchP, 3, aCombineConjugates, tCombineImpactParams, aEventsType, aCustomBins, aMinNorm, aMaxNorm, aMaxBgdFit, aAvgLamKchPMFit, aUseStavCf, aZoomY);
  BuildBgdwFitPanel(tCanPart, 0, 1, aCfDescriptor, aFileNameCfs, kLamKchP, 5, aCombineConjugates, tCombineImpactParams, aEventsType, aCustomBins, aMinNorm, aMaxNorm, aMaxBgdFit, aAvgLamKchPMFit, aUseStavCf, aZoomY);
  BuildBgdwFitPanel(tCanPart, 0, 2, aCfDescriptor, aFileNameCfs, kLamKchP, 8, aCombineConjugates, tCombineImpactParams, aEventsType, aCustomBins, aMinNorm, aMaxNorm, aMaxBgdFit, aAvgLamKchPMFit, aUseStavCf, aZoomY);

  BuildBgdwFitPanel(tCanPart, 1, 0, aCfDescriptor, aFileNameCfs, kLamKchM, 3, aCombineConjugates, tCombineImpactParams, aEventsType, aCustomBins, aMinNorm, aMaxNorm, aMaxBgdFit, aAvgLamKchPMFit, aUseStavCf, aZoomY);
  BuildBgdwFitPanel(tCanPart, 1, 1, aCfDescriptor, aFileNameCfs, kLamKchM, 5, aCombineConjugates, tCombineImpactParams, aEventsType, aCustomBins, aMinNorm, aMaxNorm, aMaxBgdFit, aAvgLamKchPMFit, aUseStavCf, aZoomY);
  BuildBgdwFitPanel(tCanPart, 1, 2, aCfDescriptor, aFileNameCfs, kLamKchM, 8, aCombineConjugates, tCombineImpactParams, aEventsType, aCustomBins, aMinNorm, aMaxNorm, aMaxBgdFit, aAvgLamKchPMFit, aUseStavCf, aZoomY);

  BuildBgdwFitPanel(tCanPart, 2, 0, aCfDescriptor, aFileNameCfs, kLamK0, 3, aCombineConjugates, tCombineImpactParams, aEventsType, aCustomBins, aMinNorm, aMaxNorm, aMaxBgdFit, aAvgLamKchPMFit, aUseStavCf, aZoomY);
  BuildBgdwFitPanel(tCanPart, 2, 1, aCfDescriptor, aFileNameCfs, kLamK0, 5, aCombineConjugates, tCombineImpactParams, aEventsType, aCustomBins, aMinNorm, aMaxNorm, aMaxBgdFit, aAvgLamKchPMFit, aUseStavCf, aZoomY);
  BuildBgdwFitPanel(tCanPart, 2, 2, aCfDescriptor, aFileNameCfs, kLamK0, 8, aCombineConjugates, tCombineImpactParams, aEventsType, aCustomBins, aMinNorm, aMaxNorm, aMaxBgdFit, aAvgLamKchPMFit, aUseStavCf, aZoomY);

  //----- Increase label size on axes
  double tLabelScaleX = 3.5;
  double tLabelScaleY = 3.5;
  
  double tLabelOffsetScaleX = 2.0;
  double tLabelOffsetScaleY = 2.0;
  for(int j=0; j<tNy; j++)
  {
    for(int i=0; i<tNx; i++)
    {
      TH1* tTempHist = (TH1*)(tCanPart->GetGraphsInPad(i,j)->At(0));

      tTempHist->GetXaxis()->SetLabelSize(tLabelScaleX*tTempHist->GetXaxis()->GetLabelSize());
      tTempHist->GetXaxis()->SetLabelOffset(tLabelOffsetScaleX*tTempHist->GetXaxis()->GetLabelOffset());

      tTempHist->GetYaxis()->SetLabelSize(tLabelScaleY*tTempHist->GetYaxis()->GetLabelSize());
      tTempHist->GetYaxis()->SetLabelOffset(tLabelOffsetScaleY*tTempHist->GetYaxis()->GetLabelOffset());

      if(j==0) tTempHist->GetYaxis()->SetRangeUser(0.9825, 1.015);
      if(j==1) tTempHist->GetYaxis()->SetRangeUser(0.9725, 1.018);
      
      if(aCfDescriptor.EqualTo("Full"))
      {
        if(j==0) 
        {
          tTempHist->GetYaxis()->SetRangeUser(0.9825, 1.0375);
          //((TH1*)tCanPart->GetGraphsInPad(0,0)->At(0))->GetYaxis()->SetNdivisions(510);
        }
        if(j==1) tTempHist->GetYaxis()->SetRangeUser(0.9725, 1.02499);
        if(j==2) tTempHist->GetYaxis()->SetRangeUser(0.9475, 1.039);
      }            
      
      if(aCfDescriptor.EqualTo("PrimaryAndShortDecays"))
      {
        if(j==0) tTempHist->GetYaxis()->SetRangeUser(0.9825, 1.0325);
        if(j==1) tTempHist->GetYaxis()->SetRangeUser(0.9725, 1.01999);
        if(j==2) tTempHist->GetYaxis()->SetRangeUser(0.9475, 1.034999);
      }      
    }

  }

  //-----------------
  tCanPart->SetDrawUnityLine(true);
  tCanPart->DrawAll();
  tCanPart->DrawXaxisTitle("#it{k}* (GeV/#it{c})", 43, 75, 0.825, 0.01); //Note, changing xaxis low (=0.315) does nothing
  tCanPart->DrawYaxisTitle("#it{C}(#it{k}*)", 43, 75, 0.045, 0.85);

  return tCanPart->GetCanvas();
}




//________________________________________________________________________________________________________________
TCanvas* DrawBgdwFit_SingleCentAllAnv2(CentralityType aCentType, TString aCfDescriptor, TString aFileNameCfs, bool aCombineConjugates, ThermEventsType aEventsType, int aRebin, double aMinNorm, double aMaxNorm, double aMaxBgdFit=3.0, bool aAvgLamKchPMFit=false, bool aUseStavCf=false)
{
  bool tCombineImpactParams = true; //Should always be true here
  bool aZoomY=true;
//-------------------------------------------------------------------------------
  TString tCanBgdwFitName = "BgdwFitOnly";

  if(aFileNameCfs.Contains("_RandomEPs_NumWeight1")) tCanBgdwFitName += TString("_RandomEPs_NumWeight1");
  else if(aFileNameCfs.Contains("_RandomEPs")) tCanBgdwFitName += TString("_RandomEPs");
  else tCanBgdwFitName += TString("");

  tCanBgdwFitName += TString::Format("_%s_", aCfDescriptor.Data());
  tCanBgdwFitName += TString("AllAn");

  if(aCombineConjugates) tCanBgdwFitName += TString("wConj");
  tCanBgdwFitName += TString(cCentralityTags[aCentType]);

  if(aAvgLamKchPMFit) tCanBgdwFitName += TString("_AvgLamKchPMFit");

//-------------------------------------------------------------------------------
  int tNx=3;
  int tNy=1;
  double tXLow = -0.04;
//  double tXHigh = aMaxBgdFit-0.02;
  double tXHigh = 1.68;
  double tYLow = 0.955;
  double tYHigh = 1.01999;

  CanvasPartition* tCanPart = new CanvasPartition(tCanBgdwFitName,tNx,tNy,tXLow,tXHigh,tYLow,tYHigh,0.075,0.0025,0.15,0.0025);
  tCanPart->SetDrawOptStat(false);
  tCanPart->GetCanvas()->SetCanvasSize(2100, 500);

  int tImpParam = 0;
  if(aCentType==k0010) tImpParam=3;
  else if(aCentType==k1030) tImpParam = 5;
  else if(aCentType==k3050) tImpParam = 8;
  else assert(0);

  BuildBgdwFitPanel(tCanPart, 0, 0, aCfDescriptor, aFileNameCfs, kLamKchP, tImpParam, aCombineConjugates, tCombineImpactParams, aEventsType, aRebin, aMinNorm, aMaxNorm, aMaxBgdFit, aAvgLamKchPMFit, aUseStavCf, aZoomY, true);

  BuildBgdwFitPanel(tCanPart, 1, 0, aCfDescriptor, aFileNameCfs, kLamKchM, tImpParam, aCombineConjugates, tCombineImpactParams, aEventsType, aRebin, aMinNorm, aMaxNorm, aMaxBgdFit, aAvgLamKchPMFit, aUseStavCf, aZoomY, true);

  BuildBgdwFitPanel(tCanPart, 2, 0, aCfDescriptor, aFileNameCfs, kLamK0, tImpParam, aCombineConjugates, tCombineImpactParams, aEventsType, aRebin, aMinNorm, aMaxNorm, aMaxBgdFit, aAvgLamKchPMFit, aUseStavCf, aZoomY, true);

  //----- Increase label size on axes
  for(int j=0; j<tNy; j++)
  {
    for(int i=0; i<tNx; i++)
    {
      TH1* tTempHist = (TH1*)(tCanPart->GetGraphsInPad(i,j)->At(0));

      tTempHist->GetXaxis()->SetLabelSize(2.1*tTempHist->GetXaxis()->GetLabelSize());
      tTempHist->GetXaxis()->SetLabelOffset(0.25*tTempHist->GetXaxis()->GetLabelOffset());

      tTempHist->GetYaxis()->SetLabelSize(2.25*tTempHist->GetYaxis()->GetLabelSize());
      tTempHist->GetYaxis()->SetLabelOffset(2.5*tTempHist->GetYaxis()->GetLabelOffset());

      if(aCentType==k0010) tTempHist->GetYaxis()->SetRangeUser(0.9865, 1.015);
      if(aCentType==k1030) tTempHist->GetYaxis()->SetRangeUser(0.975, 1.015);
    }

  }

  //------------------
  tCanPart->SetDrawUnityLine(false);
  tCanPart->DrawAll();
  tCanPart->DrawXaxisTitle("#it{k}* (GeV/#it{c})", 43, 60, 0.825, 0.01); //Note, changing xaxis low (=0.315) does nothing
  tCanPart->DrawYaxisTitle("#it{C}(#it{k}*)", 43, 75, 0.035, 0.65);

  return tCanPart->GetCanvas();
}







//________________________________________________________________________________________________________________
TCanvas* DrawBgdwFit_AllCent(TString aCfDescriptor, TString aFileNameCfs, AnalysisType aAnType, bool aCombineConjugates, ThermEventsType aEventsType, int aRebin, double aMinNorm, double aMaxNorm, double aMaxBgdFit=3.0, bool aUseStavCf=false)
{
  TCanvas *tReturnCan, *tCan_0010, *tCan_1030, *tCan_3050;

  bool tCombineImpactParams = true; //Should always be true here
  tCan_0010 = DrawBgdwFit(aCfDescriptor, aFileNameCfs, aAnType, 3, aCombineConjugates, tCombineImpactParams, aEventsType, aRebin, aMinNorm, aMaxNorm, aMaxBgdFit, aUseStavCf);
  tCan_1030 = DrawBgdwFit(aCfDescriptor, aFileNameCfs, aAnType, 5, aCombineConjugates, tCombineImpactParams, aEventsType, aRebin, aMinNorm, aMaxNorm, aMaxBgdFit, aUseStavCf);
  tCan_3050 = DrawBgdwFit(aCfDescriptor, aFileNameCfs, aAnType, 8, aCombineConjugates, tCombineImpactParams, aEventsType, aRebin, aMinNorm, aMaxNorm, aMaxBgdFit, aUseStavCf);
  //-----------------
  TString tCanName = tCan_0010->GetName();
  tCanName += TString("_1030_3050");

  tReturnCan = new TCanvas(tCanName, tCanName, 700, 1500);
  tReturnCan->Divide(1, 3, 0.01, 0.001);
  tReturnCan->cd(1);
  tCan_0010->DrawClonePad();
  tReturnCan->cd(2);
  tCan_1030->DrawClonePad();
  tReturnCan->cd(3);
  tCan_3050->DrawClonePad();
  //-----------------
  delete tCan_0010;
  delete tCan_1030;
  delete tCan_3050;
  //-----------------
  return tReturnCan;
}

//________________________________________________________________________________________________________________
TCanvas* DrawLamKchPMBgdwFit(TString aCfDescriptor, TString aFileNameCfs, int aImpactParam, ThermEventsType aEventsType, int aRebin, double aMinNorm, double aMaxNorm, double aMaxBgdFit=3.0, bool aUseStavCf=false)
{
  gRejectOmegaTherm=true;
  if(aCfDescriptor.EqualTo("PrimaryOnly") || aCfDescriptor.EqualTo("PrimaryAndShortDecays")) gRejectOmegaTherm=false;  //In this case, Omega peak not present
  //-------------------------------------------------
  CentralityType tCentType = GetCentralityType(aImpactParam);
  //-------------------------------------------------
  TString tDescriptor = "Bgd (w. Fit)";

  TString tOverallDescriptor = TString::Format("(#bar{#Lambda})#LambdaK^{#pm} (%s)", cPrettyCentralityTags[tCentType]);

  int tMarkerStyle = 26;
  int tColor = kOrange;

  //--------------------------------------------
  TH1 *tCf = ThermCf::GetLamKchPMCombinedThermCfs(aFileNameCfs, aCfDescriptor, tCentType, aEventsType, aRebin, aMinNorm, aMaxNorm, tMarkerStyle, tColor, aUseStavCf);
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
  cout << "**************************************************" << endl;
  cout << "Fitting call from: DrawLamKchPMBgdwFit" << endl;
  TF1 *tBgdFit, *tBgdFitDraw;
  int tPower = 6;
  tBgdFit = FitBackground(tCf, tPower, 0., aMaxBgdFit);
  cout << "**************************************************" << endl;

  gRejectOmegaTherm=false;
  tBgdFitDraw = new TF1(tBgdFit->GetName()+TString("_Draw"), FitFunctionPolynomial, 0., 3., 7);
  for(int i=0; i<tBgdFit->GetNpar(); i++) tBgdFitDraw->SetParameter(i, tBgdFit->GetParameter(i));

  tBgdFitDraw->SetLineColor(tCf->GetLineColor());
  tBgdFitDraw->SetRange(0., aMaxBgdFit);
  tBgdFitDraw->Draw("lsame");

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
TCanvas* DrawLamKchPMBgdwFit_AllCent(TString aCfDescriptor, TString aFileNameCfs, ThermEventsType aEventsType, int aRebin, double aMinNorm, double aMaxNorm, double aMaxBgdFit=3.0, bool aUseStavCf=false)
{
  TCanvas *tReturnCan, *tCan_0010, *tCan_1030, *tCan_3050;

  tCan_0010 = DrawLamKchPMBgdwFit(aCfDescriptor, aFileNameCfs, 3, aEventsType, aRebin, aMinNorm, aMaxNorm, aMaxBgdFit, aUseStavCf);
  tCan_1030 = DrawLamKchPMBgdwFit(aCfDescriptor, aFileNameCfs, 5, aEventsType, aRebin, aMinNorm, aMaxNorm, aMaxBgdFit, aUseStavCf);
  tCan_3050 = DrawLamKchPMBgdwFit(aCfDescriptor, aFileNameCfs, 8, aEventsType, aRebin, aMinNorm, aMaxNorm, aMaxBgdFit, aUseStavCf);
  //-----------------
  TString tCanName = tCan_0010->GetName();
  tCanName += TString("_1030_3050");

  tReturnCan = new TCanvas(tCanName, tCanName, 700, 1500);
  tReturnCan->Divide(1, 3, 0.01, 0.001);
  tReturnCan->cd(1);
  tCan_0010->DrawClonePad();
  tReturnCan->cd(2);
  tCan_1030->DrawClonePad();
  tReturnCan->cd(3);
  tCan_3050->DrawClonePad();
  //-----------------
  delete tCan_0010;
  delete tCan_1030;
  delete tCan_3050;
  //-----------------
  return tReturnCan;
}

//________________________________________________________________________________________________________________
TCanvas* CompareAnalyses(TString aCfDescriptor, TString aFileNameCfs, int aImpactParam, bool aCombineConjugates, bool aCombineImpactParams, ThermEventsType aEventsType, int aRebin, double aMinNorm, double aMaxNorm, bool aUseStavCf=false)
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

  ThermCf* tThermCf1 = new ThermCf(aFileNameCfs, aCfDescriptor, kLamK0, GetCentralityType(aImpactParam), aCombineConjugates, aEventsType, aRebin, aMinNorm, aMaxNorm, aUseStavCf);
  ThermCf* tThermCf2 = new ThermCf(aFileNameCfs, aCfDescriptor, kLamKchP, GetCentralityType(aImpactParam), aCombineConjugates, aEventsType, aRebin, aMinNorm, aMaxNorm, aUseStavCf);
  ThermCf* tThermCf3 = new ThermCf(aFileNameCfs, aCfDescriptor, kLamKchM, GetCentralityType(aImpactParam), aCombineConjugates, aEventsType, aRebin, aMinNorm, aMaxNorm, aUseStavCf);

  if(!aCombineImpactParams)
  {
    tThermCf1->SetSpecificImpactParam(aImpactParam, aCombineImpactParams);
    tThermCf2->SetSpecificImpactParam(aImpactParam, aCombineImpactParams);
    tThermCf3->SetSpecificImpactParam(aImpactParam, aCombineImpactParams);
  }
  TH1* tCf1 = tThermCf1->GetThermCf(tMarkerStyle1, tColor1, 0.75);
  TH1* tCf2 = tThermCf2->GetThermCf(tMarkerStyle2, tColor2, 0.75);
  TH1* tCf3 = tThermCf3->GetThermCf(tMarkerStyle3, tColor3, 0.75);
//-------------------------------------------------------------------------------
  TString tCanCfsName;
  tCanCfsName = TString::Format("CompareAnalyses_%s", aCfDescriptor.Data());

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
TCanvas* CompareAnalyses_AllCent(TString aCfDescriptor, TString aFileNameCfs, bool aCombineConjugates, ThermEventsType aEventsType, int aRebin, double aMinNorm, double aMaxNorm, bool aUseStavCf=false)
{
  TCanvas *tReturnCan, *tCan_0010, *tCan_1030, *tCan_3050;

  bool tCombineImpactParams = true; //Should always be true here
  tCan_0010 = CompareAnalyses(aCfDescriptor, aFileNameCfs, 3, aCombineConjugates, tCombineImpactParams, aEventsType, aRebin, aMinNorm, aMaxNorm, aUseStavCf);
  tCan_1030 = CompareAnalyses(aCfDescriptor, aFileNameCfs, 5, aCombineConjugates, tCombineImpactParams, aEventsType, aRebin, aMinNorm, aMaxNorm, aUseStavCf);
  tCan_3050 = CompareAnalyses(aCfDescriptor, aFileNameCfs, 8, aCombineConjugates, tCombineImpactParams, aEventsType, aRebin, aMinNorm, aMaxNorm, aUseStavCf);
  //-----------------
  TString tCanName = tCan_0010->GetName();
  tCanName += TString("_1030_3050");

  tReturnCan = new TCanvas(tCanName, tCanName, 700, 1500);
  tReturnCan->Divide(1,3, 0.01, 0.001);
  tReturnCan->cd(1);
  tCan_0010->DrawClonePad();
  tReturnCan->cd(2);
  tCan_1030->DrawClonePad();
  tReturnCan->cd(3);
  tCan_3050->DrawClonePad();
  //-----------------
  delete tCan_0010;
  delete tCan_1030;
  delete tCan_3050;
  //-----------------
  return tReturnCan;

}






//________________________________________________________________________________________________________________
TCanvas* DrawDataVsTherm(TString aCfDescriptor, TString aFileNameCfs, AnalysisType aAnType, int aImpactParam, bool aCombineConjugates, bool aCombineImpactParams, ThermEventsType aEventsType, int aRebin, double aMinNorm, double aMaxNorm, bool aUseStavCf=false)
{
  //-------------------------------------------------
  CentralityType tCentType=kMB;
  if(aCombineImpactParams) tCentType = GetCentralityType(aImpactParam);
  //-------------------------------------------------
  CentralityType tCentTypeData = GetCentralityType(aImpactParam);
  //-------------------------------------------------
  AnalysisType tConjAnType = GetConjAnType(aAnType);
  //-------------------------------------------------

  TString tOverallDescriptor = cAnalysisRootTags[aAnType];
  if(aCombineConjugates) tOverallDescriptor += TString::Format(" & %s", cAnalysisRootTags[tConjAnType]);
  if(!aCombineImpactParams) tOverallDescriptor += TString::Format(" (b=%d)", aImpactParam);
  else tOverallDescriptor += TString::Format(" (%s)", cPrettyCentralityTags[tCentType]);

  int tMarkerStyle = 26;
  int tColor = kOrange;

  //--------------------------------------------
  TH1D* tData = GetQuickData(aAnType, tCentTypeData, aCombineConjugates);
  ThermCf::SetStyleAndColor(tData, 20, GetColor(aAnType));
  //--------------------------------------------
  ThermCf* tThermCfLong = new ThermCf(aFileNameCfs, aCfDescriptor, aAnType, GetCentralityType(aImpactParam), aCombineConjugates, aEventsType, aRebin, aMinNorm, aMaxNorm, aUseStavCf);
  if(!aCombineImpactParams) tThermCfLong->SetSpecificImpactParam(aImpactParam, aCombineImpactParams);
  TH1* tCfLong = tThermCfLong->GetThermCf(tMarkerStyle, tColor, 0.75);
  //--------------------------------------------
  if(tCfLong->GetBinWidth(1) != tData->GetBinWidth(1))
  {
    cout << "tCfLong->GetBinWidth(1) != tData->GetBinWidth(1)!!!!!! CRASH" << endl;
    cout << "\t tCfLong->GetBinWidth(1) = " << tCfLong->GetBinWidth(1) << endl;
    cout << "\t tData->GetBinWidth(1) = " << tData->GetBinWidth(1) << endl;
    assert(0);
  }

  TH1D* tCf;
  if(tCfLong->GetNbinsX() != tData->GetNbinsX())  //To divide (later, for tRatio), tData and tCf need to have same number of bins, and same bin size
  {
    tCf = (TH1D*)tData->Clone(tCfLong->GetName());
    for(int i=1; i<=tCf->GetNbinsX(); i++)
    {
      tCf->SetBinContent(i, tCfLong->GetBinContent(i));
      tCf->SetBinError(i, tCfLong->GetBinError(i));
    }
    assert(tCf->GetNbinsX()==tData->GetNbinsX());
    assert(tCf->GetBinWidth(1)==tData->GetBinWidth(1));
    ThermCf::SetStyleAndColor(tCf, tMarkerStyle, tColor);
  }
  else tCf = (TH1D*)tCfLong->Clone();
//-------------------------------------------------------------------------------
  TString tCanDataVsThemName = "DataVsTherm";

  if(aFileNameCfs.Contains("_RandomEPs_NumWeight1")) tCanDataVsThemName += TString("_RandomEPs_NumWeight1");
  else if(aFileNameCfs.Contains("_RandomEPs")) tCanDataVsThemName += TString("_RandomEPs");
  else tCanDataVsThemName += TString("");

  tCanDataVsThemName += TString::Format("_%s_", aCfDescriptor.Data());
  tCanDataVsThemName += TString(cAnalysisBaseTags[aAnType]);

  if(aCombineConjugates) tCanDataVsThemName += TString("wConj");
  if(!aCombineImpactParams) tCanDataVsThemName += TString::Format("_b%d", aImpactParam);
  else tCanDataVsThemName += TString(cCentralityTags[tCentType]);

//-------------------------------------------------------------------------------

  TCanvas* tCanDataVsThem = new TCanvas(tCanDataVsThemName, tCanDataVsThemName);
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);
  //---------------------------------------------------------------
  tData->GetXaxis()->SetTitle("#it{k}* (GeV/#it{c})");
  tData->GetYaxis()->SetTitle("#it{C}(#it{k}*)");

  tData->GetXaxis()->SetRangeUser(0.,3.0);
  tData->GetYaxis()->SetRangeUser(0.86, 1.07);

  //---------------------------------------------------------------

  TH1D* tRatio;

  tRatio = (TH1D*)tData->Clone();
  tRatio->Divide(tCf);
  ThermCf::SetStyleAndColor(tRatio, 24, kMagenta);

  //---------------------------------------------------------------

  tData->Draw();
  tCf->Draw("same");
  tRatio->Draw("same");
  //---------------------------------------------------------------
  if(aAnType==kLamKchM || aAnType==kALamKchP) gRejectOmegaTherm=true;
  else gRejectOmegaTherm=false;
  if(aCfDescriptor.EqualTo("PrimaryOnly") || aCfDescriptor.EqualTo("PrimaryAndShortDecays")) gRejectOmegaTherm=false;  //In this case, Omega peak not present

  cout << "**************************************************" << endl;
  cout << "Fitting call from: DrawDataVsTherm" << endl;
  TF1 *tRatioFit, *tRatioFitDraw;
  int tPower = 6;
  tRatioFit = FitBackground(tRatio, tPower, 0., 3.);
  cout << "**************************************************" << endl;
  if(aAnType==kLamKchM || aAnType==kALamKchP) //Want to draw fit function without gaping gap where Omega peak omitted
  {
    gRejectOmegaTherm=false;
    tRatioFitDraw = new TF1(tRatioFit->GetName()+TString("_Draw"), FitFunctionPolynomial, 0., 3., 7);
    for(int i=0; i<tRatioFit->GetNpar(); i++) tRatioFitDraw->SetParameter(i, tRatioFit->GetParameter(i));
  }
  else tRatioFitDraw = tRatioFit;

  tRatioFitDraw->SetLineColor(tRatio->GetLineColor());
  tRatioFitDraw->Draw("lsame");



  TLegend* tLeg = new TLegend(0.60, 0.15, 0.85, 0.40);
    tLeg->SetFillColor(0);
    tLeg->SetBorderSize(0);
    tLeg->SetTextAlign(22);

  TString tThermDescriptor = "Therm";
  if(!aCombineImpactParams) tThermDescriptor += TString::Format(" (b=%d)", aImpactParam);
  else tCanDataVsThemName += TString::Format(" (%s)", cPrettyCentralityTags[tCentType]);

  tLeg->AddEntry(tData, TString::Format("Data (%s)", cPrettyCentralityTags[tCentTypeData]));
  tLeg->AddEntry(tCf, tThermDescriptor);
  tLeg->AddEntry(tRatio, "Ratio (Data/Therm)");

  tLeg->Draw();
  //---------------------------------------------------------------

  TLine* tLine = new TLine(0, 1, 2, 1);
  tLine->SetLineColor(14);
  tLine->Draw();

  PrintInfo((TPad*)tCanDataVsThem, tOverallDescriptor, 0.04);

  //---------------------------------------------------------------

  TH1D* tData2 = GetQuickData(aAnType, tCentTypeData, aCombineConjugates, "20180505");
  ThermCf::SetStyleAndColor(tData2, 24, GetColor(aAnType));
  tData2->Draw("same");
  //---------------------------------------------------------------


  return tCanDataVsThem;
}



//________________________________________________________________________________________________________________
TCanvas* DrawDataVsTherm_AllCent(TString aCfDescriptor, TString aFileNameCfs, AnalysisType aAnType, bool aCombineConjugates, ThermEventsType aEventsType, int aRebin, double aMinNorm, double aMaxNorm, bool aUseStavCf=false)
{
  TCanvas *tReturnCan, *tCan_0010, *tCan_1030, *tCan_3050;

  bool tCombineImpactParams = true; //Should always be true here
  tCan_0010 = DrawDataVsTherm(aCfDescriptor, aFileNameCfs, aAnType, 3, aCombineConjugates, tCombineImpactParams, aEventsType, aRebin, aMinNorm, aMaxNorm, aUseStavCf);
  tCan_1030 = DrawDataVsTherm(aCfDescriptor, aFileNameCfs, aAnType, 5, aCombineConjugates, tCombineImpactParams, aEventsType, aRebin, aMinNorm, aMaxNorm, aUseStavCf);
  tCan_3050 = DrawDataVsTherm(aCfDescriptor, aFileNameCfs, aAnType, 8, aCombineConjugates, tCombineImpactParams, aEventsType, aRebin, aMinNorm, aMaxNorm, aUseStavCf);
  //-----------------
  TString tCanName = tCan_0010->GetName();
  tCanName += TString("_1030_3050");

  tReturnCan = new TCanvas(tCanName, tCanName, 700, 1500);
  tReturnCan->Divide(1,3, 0.01, 0.001);
  tReturnCan->cd(1);
  tCan_0010->DrawClonePad();
  tReturnCan->cd(2);
  tCan_1030->DrawClonePad();
  tReturnCan->cd(3);
  tCan_3050->DrawClonePad();
  //-----------------
  delete tCan_0010;
  delete tCan_1030;
  delete tCan_3050;
  //-----------------
  return tReturnCan;
}


//________________________________________________________________________________________________________________
TCanvas* CompareBackgroundReductionMethods(TString aCfDescriptor, AnalysisType aAnType, int aImpactParam, bool aCombineConjugates, bool aCombineImpactParams, ThermEventsType aEventsType, int aRebin, double aMinNorm, double aMaxNorm, bool aNumWeight1, bool aDrawData, bool bVerbose)
{
  CentralityType tCentType=kMB;
  if(aCombineImpactParams) tCentType = GetCentralityType(aImpactParam);
  //-------------------------------------------------
  AnalysisType tConjAnType = GetConjAnType(aAnType);
  //-------------------------------------------------

  TString tOverallDescriptor = cAnalysisRootTags[aAnType];
  if(aCombineConjugates) tOverallDescriptor += TString::Format(" & %s", cAnalysisRootTags[tConjAnType]);
  if(!aCombineImpactParams) tOverallDescriptor += TString::Format(" (b=%d)", aImpactParam);
  else tOverallDescriptor += TString::Format(" (%s)", cPrettyCentralityTags[tCentType]);

  //--------------------------------------------
  TString tFileName_AlignEPs, tFileName_RandomEPs;
  if(aNumWeight1)
  {
    tFileName_AlignEPs = "CorrelationFunctions_NumWeight1.root";
    tFileName_RandomEPs = "CorrelationFunctions_RandomEPs_NumWeight1.root";
  }
  else
  {
    tFileName_AlignEPs = "CorrelationFunctions.root";
    tFileName_RandomEPs = "CorrelationFunctions_RandomEPs.root";
  }
  //--------------------------------------------

  int tMarkerStyle_AlignEPs                = 20;
  int tColor_AlignEPs                      = kBlack;

  int tMarkerStyle_AlignEPs_UseStavCf  = 20;
  int tColor_AlignEPs_UseStavCf        = kCyan;

  int tMarkerStyle_RandomEPs               = 20;
  int tColor_RandomEPs                     = kGreen+1;

  int tMarkerStyle_RandomEPs_UseStavCf = 24;
  int tColor_RandomEPs_UseStavCf       = kMagenta;
  //--------------------------------------------

  ThermCf* tThermCf_AlignEPs                = new ThermCf(tFileName_AlignEPs, 
                                                          aCfDescriptor, aAnType, GetCentralityType(aImpactParam), aCombineConjugates, aEventsType, aRebin, aMinNorm, aMaxNorm, false);
  ThermCf* tThermCf_AlignEPs_UseStavCf  = new ThermCf(tFileName_AlignEPs, 
                                                          aCfDescriptor, aAnType, GetCentralityType(aImpactParam), aCombineConjugates, aEventsType, aRebin, aMinNorm, aMaxNorm, true);

  ThermCf* tThermCf_RandomEPs               = new ThermCf(tFileName_RandomEPs, 
                                                          aCfDescriptor, aAnType, GetCentralityType(aImpactParam), aCombineConjugates, aEventsType, aRebin, aMinNorm, aMaxNorm, false);
  ThermCf* tThermCf_RandomEPs_UseStavCf = new ThermCf(tFileName_RandomEPs, 
                                                          aCfDescriptor, aAnType, GetCentralityType(aImpactParam), aCombineConjugates, aEventsType, aRebin, aMinNorm, aMaxNorm, true);

  if(!aCombineImpactParams)
  {
    tThermCf_AlignEPs->SetSpecificImpactParam(aImpactParam, aCombineImpactParams);
    tThermCf_AlignEPs_UseStavCf->SetSpecificImpactParam(aImpactParam, aCombineImpactParams);
    tThermCf_RandomEPs->SetSpecificImpactParam(aImpactParam, aCombineImpactParams);
    tThermCf_RandomEPs_UseStavCf->SetSpecificImpactParam(aImpactParam, aCombineImpactParams);
  }

  TH1* tCf_AlignEPs = tThermCf_AlignEPs->GetThermCf(tMarkerStyle_AlignEPs, tColor_AlignEPs, 0.75);
  TH1* tCf_AlignEPs_UseStavCf = tThermCf_AlignEPs_UseStavCf->GetThermCf(tMarkerStyle_AlignEPs_UseStavCf, tColor_AlignEPs_UseStavCf, 0.75);
  TH1* tCf_RandomEPs = tThermCf_RandomEPs->GetThermCf(tMarkerStyle_RandomEPs, tColor_RandomEPs, 0.75);
  TH1* tCf_RandomEPs_UseStavCf = tThermCf_RandomEPs_UseStavCf->GetThermCf(tMarkerStyle_RandomEPs_UseStavCf, tColor_RandomEPs_UseStavCf, 0.75);

  if(bVerbose)
  {
    for(int i=1; i<=tCf_AlignEPs_UseStavCf->GetNbinsX(); i++)
    {
      cout << "tCf_AlignEPs_UseStavCf->GetBinContent(" << i << ") = " << tCf_AlignEPs_UseStavCf->GetBinContent(i) << endl;
      cout << "tCf_RandomEPs_UseStavCf->GetBinContent(" << i << ") = " << tCf_RandomEPs_UseStavCf->GetBinContent(i) << endl;
      cout << "difference = " << tCf_AlignEPs_UseStavCf->GetBinContent(i) - tCf_RandomEPs_UseStavCf->GetBinContent(i) << endl;
      cout << "----------" << endl;
      cout << "tCf_AlignEPs_UseStavCf->GetBinError(" << i << ") = " << tCf_AlignEPs_UseStavCf->GetBinError(i) << endl;
      cout << "tCf_RandomEPs_UseStavCf->GetBinError(" << i << ") = " << tCf_RandomEPs_UseStavCf->GetBinError(i) << endl;
      cout << "difference = " << tCf_AlignEPs_UseStavCf->GetBinError(i) - tCf_RandomEPs_UseStavCf->GetBinError(i) << endl;
      cout << endl << endl;
    }
  }

//-------------------------------------------------------------------------------
  TString tCanName = "CompareBackgroundReductionMethods";

  tCanName += TString::Format("_%s_", aCfDescriptor.Data());
  tCanName += TString(cAnalysisBaseTags[aAnType]);

  if(aCombineConjugates) tCanName += TString("wConj");
  if(!aCombineImpactParams) tCanName += TString::Format("_b%d", aImpactParam);
  else tCanName += TString(cCentralityTags[tCentType]);

  if(aNumWeight1) tCanName += TString("_NumWeight1");
  if(aDrawData) tCanName += TString("_wData");
//-------------------------------------------------------------------------------

  TCanvas* tCan = new TCanvas(tCanName, tCanName);
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);
  //---------------------------------------------------------------
  tCf_AlignEPs->GetXaxis()->SetTitle("#it{k}* (GeV/#it{c})");
  tCf_AlignEPs->GetYaxis()->SetTitle("#it{C}(#it{k}*)");

//  tCf_AlignEPs->GetXaxis()->SetRangeUser(0.,3.0);
  tCf_AlignEPs->GetXaxis()->SetRangeUser(0.,2.0);
  tCf_AlignEPs->GetYaxis()->SetRangeUser(0.86, 1.07);
//  tCf_AlignEPs->GetYaxis()->SetRangeUser(0.71, 3.31);  //when using ArtificialV3Signal-1

  tCf_AlignEPs->Draw();
  tCf_AlignEPs_UseStavCf->Draw("same");
  tCf_RandomEPs->Draw("same");
  tCf_RandomEPs_UseStavCf->Draw("same");
  //---------------------------------------------------------------

  TLine* tLine = new TLine(0, 1, 2, 1);
  tLine->SetLineColor(14);
  tLine->Draw();

  PrintInfo((TPad*)tCan, tOverallDescriptor, 0.04);

  //---------------------------------------------------------------
  TLegend* tLeg = new TLegend(0.55, 0.15, 0.90, 0.30);
//  TLegend* tLeg = new TLegend(0.55, 0.40, 0.90, 0.55);  //when using ArtificialV3Signal-1
    tLeg->SetFillColor(0);
    tLeg->SetBorderSize(0);
    tLeg->SetTextAlign(22);

  tLeg->AddEntry(tCf_AlignEPs, "Aligns EPs");
  tLeg->AddEntry(tCf_RandomEPs, "Random EPs");
  tLeg->AddEntry(tCf_RandomEPs_UseStavCf, "Random EPs, Stav.");
  tLeg->AddEntry(tCf_AlignEPs_UseStavCf, "Aligns EPs, Stav.");

  //---------------------------------------------------------------

  if(aDrawData)
  {
    CentralityType tCentTypeData = GetCentralityType(aImpactParam);

    TH1D* tData = GetQuickData(aAnType, tCentTypeData, aCombineConjugates, "20180505");
    ThermCf::SetStyleAndColor(tData, 21, GetColor(aAnType));

    TH1D* tData_UseStavCf = GetQuickData(aAnType, tCentTypeData, aCombineConjugates, "20180505_StavCfs");
    ThermCf::SetStyleAndColor(tData_UseStavCf, 22, GetColor(aAnType));

    tData->Draw("same");
    tData_UseStavCf->Draw("same");

    tLeg->AddEntry(tData, TString::Format("Data (%s)", cPrettyCentralityTags[tCentTypeData]));
    tLeg->AddEntry(tData_UseStavCf, "Data, Rotate Par2");
  }

  //---------------------------------------------------------------

  tLeg->Draw();
  return tCan;
}

//________________________________________________________________________________________________________________
TCanvas* CompareBackgroundReductionMethods_AllCent(TString aCfDescriptor, AnalysisType aAnType, bool aCombineConjugates, ThermEventsType aEventsType, int aRebin, double aMinNorm, double aMaxNorm, bool aNumWeight1, bool aDrawData, bool bVerbose)
{
  TCanvas *tReturnCan, *tCan_0010, *tCan_1030, *tCan_3050;

  bool tCombineImpactParams = true; //Should always be true here
  tCan_0010 = CompareBackgroundReductionMethods(aCfDescriptor, aAnType, 3, aCombineConjugates, tCombineImpactParams, aEventsType, aRebin, aMinNorm, aMaxNorm, aNumWeight1, aDrawData, bVerbose);
  tCan_1030 = CompareBackgroundReductionMethods(aCfDescriptor, aAnType, 5, aCombineConjugates, tCombineImpactParams, aEventsType, aRebin, aMinNorm, aMaxNorm, aNumWeight1, aDrawData, bVerbose);
  tCan_3050 = CompareBackgroundReductionMethods(aCfDescriptor, aAnType, 8, aCombineConjugates, tCombineImpactParams, aEventsType, aRebin, aMinNorm, aMaxNorm, aNumWeight1, aDrawData, bVerbose);
  //-----------------
  TString tCanName = tCan_0010->GetName();
  tCanName += TString("_1030_3050");

  tReturnCan = new TCanvas(tCanName, tCanName, 700, 1500);
  tReturnCan->Divide(1,3, 0.01, 0.001);
  tReturnCan->cd(1);
  tCan_0010->DrawClonePad();
  tReturnCan->cd(2);
  tCan_1030->DrawClonePad();
  tReturnCan->cd(3);
  tCan_3050->DrawClonePad();
  //-----------------
  delete tCan_0010;
  delete tCan_1030;
  delete tCan_3050;
  //-----------------
  return tReturnCan;
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
  if(tAnType==kLamKchM || tAnType==kALamKchP) gRejectOmegaTherm=true;
  else gRejectOmegaTherm=false;

  bool bCombineConjugates = true;
  bool bCombineImpactParams = true;

  ThermEventsType tEventsType = kMeAndAdam;  //kMe, kAdam, kMeAndAdam

  bool bCompareWithAndWithoutBgd = false;
  bool bDrawBgdwFitOnly = true;


  bool bDrawLamKchPMBgdwFitOnly = false;  //Consider maybe to slim down total Fig 4
  bool bCompareAnalyses = false;

  bool bDrawDataVsTherm = false;

  bool bCompareBackgroundReductionMethods = false;
    bool bNumWeight1 = true; 
    bool bDrawData = false;
    bool bVerbose = false;
  if(bCompareBackgroundReductionMethods) tEventsType = kMe;  //TODO for now, I haven't run over Adam's results

  bool bUseStavCf=false;

  bool bDrawAllCentralities = false;

  bool bSaveFigures = false;
  TString tSaveFileType = "pdf";

  int tRebin=2;  //NOTE: tRebin for bDrawBgdwFitOnly in block below
  double tMinNorm = 0.32;
  double tMaxNorm = 0.40;
  double tMaxBgdFit = 2.0;

  int tImpactParam = 9;

  TString tCfDescriptor = "Full";
//  TString tCfDescriptor = "PrimaryOnly";
//  TString tCfDescriptor = "PrimaryAndShortDecays";
  if(bCompareBackgroundReductionMethods) tCfDescriptor = TString("Full");

  if(tCfDescriptor.EqualTo("PrimaryOnly") || tCfDescriptor.EqualTo("PrimaryAndShortDecays")) gRejectOmegaTherm=false;  //In this case, Omega peak not present

  TString tSingleFileName = "CorrelationFunctions_RandomEPs_NumWeight1.root";

  TString tSaveDir = "/home/jesse/Analysis/FemtoAnalysis/AnalysisNotes/5_Fitting/5.5_NonFlatBackground/Figures/";
  TString tSaveFileBase = tSaveDir + TString::Format("%s/", cAnalysisBaseTags[tAnType]);


  TCanvas *tCanCfs, *tCanBgdwFit, *tCanBgdwFitv2, *tCanBgdwFitAllCentAllAnv2, *tCanCompareAnalyses, *tCanLamKchPMBgdwFit, *tCanDataVsTherm, *tCanCompBgdRedMethods, *tCanBgdwFitSingleCentAllAn0010, *tCanBgdwFitSingleCentAllAn1030, *tCanBgdwFitSingleCentAllAn3050;

  if(bCompareWithAndWithoutBgd)
  {
    tCanCfs = CompareCfWithAndWithoutBgd(tCfDescriptor, tAnType, tImpactParam, bCombineConjugates, bCombineImpactParams, tEventsType, tRebin, tMinNorm, tMaxNorm, bUseStavCf);
    if(bSaveFigures) tCanCfs->SaveAs(TString::Format("%s%s.%s", tSaveDir.Data(), tCanCfs->GetName(), tSaveFileType.Data()));
  }

  if(bDrawBgdwFitOnly)
  {
    tRebin=4;
    bool aAvgLamKchPMFit = false;
/*
    if(bDrawAllCentralities) tCanBgdwFit = DrawBgdwFit_AllCent(tCfDescriptor, tSingleFileName, tAnType, bCombineConjugates, tEventsType, tRebin, tMinNorm, tMaxNorm, tMaxBgdFit, bUseStavCf);
    else tCanBgdwFit = DrawBgdwFit(tCfDescriptor, tSingleFileName, tAnType, tImpactParam, bCombineConjugates, bCombineImpactParams, tEventsType, tRebin, tMinNorm, tMaxNorm, tMaxBgdFit, bUseStavCf);

    tCanBgdwFitv2 = DrawBgdwFit_AllCentv2(tCfDescriptor, tSingleFileName, tAnType, bCombineConjugates, tEventsType, tRebin, tMinNorm, tMaxNorm, tMaxBgdFit, aAvgLamKchPMFit, bUseStavCf);
*/

    //------------------------------------------
//    tCanBgdwFitAllCentAllAnv2 = DrawBgdwFit_AllCentAllAnv2(tCfDescriptor, tSingleFileName, bCombineConjugates, tEventsType, tRebin, tMinNorm, tMaxNorm, tMaxBgdFit, aAvgLamKchPMFit, bUseStavCf);

    vector<double> tCustomBins{0.00, 0.04, 0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32, 0.36, 0.40, 
                               0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.40, 1.50, 
                               1.60, 1.70, 1.80, 1.90, 2.00};

    tCanBgdwFitAllCentAllAnv2 = DrawBgdwFit_AllCentAllAnv2(tCfDescriptor, tSingleFileName, bCombineConjugates, tEventsType, tCustomBins, tMinNorm, tMaxNorm, tMaxBgdFit, aAvgLamKchPMFit, bUseStavCf);

    //------------------------------------------
/*
    tCanBgdwFitSingleCentAllAn0010 = DrawBgdwFit_SingleCentAllAnv2(k0010, tCfDescriptor, tSingleFileName, bCombineConjugates, tEventsType, tRebin, tMinNorm, tMaxNorm, tMaxBgdFit, aAvgLamKchPMFit, bUseStavCf);

    tCanBgdwFitSingleCentAllAn1030 = DrawBgdwFit_SingleCentAllAnv2(k1030, tCfDescriptor, tSingleFileName, bCombineConjugates, tEventsType, tRebin, tMinNorm, tMaxNorm, tMaxBgdFit, aAvgLamKchPMFit, bUseStavCf);

    tCanBgdwFitSingleCentAllAn3050 = DrawBgdwFit_SingleCentAllAnv2(k3050, tCfDescriptor, tSingleFileName, bCombineConjugates, tEventsType, tRebin, tMinNorm, tMaxNorm, tMaxBgdFit, aAvgLamKchPMFit, bUseStavCf);
*/

    if(bSaveFigures)
    {
/*    
      tCanBgdwFit->SaveAs(TString::Format("%s%s.%s", tSaveDir.Data(), tCanBgdwFit->GetName(), tSaveFileType.Data()));
      tCanBgdwFitv2->SaveAs(TString::Format("%s%s.%s", tSaveDir.Data(), tCanBgdwFitv2->GetName(), tSaveFileType.Data()));
*/      
      tCanBgdwFitAllCentAllAnv2->SaveAs(TString::Format("%s%s.%s", tSaveDir.Data(), tCanBgdwFitAllCentAllAnv2->GetName(), tSaveFileType.Data()));
/*
      tCanBgdwFitSingleCentAllAn0010->SaveAs(TString::Format("%s%s.%s", tSaveDir.Data(), tCanBgdwFitSingleCentAllAn0010->GetName(), tSaveFileType.Data()));
      tCanBgdwFitSingleCentAllAn1030->SaveAs(TString::Format("%s%s.%s", tSaveDir.Data(), tCanBgdwFitSingleCentAllAn1030->GetName(), tSaveFileType.Data()));
      tCanBgdwFitSingleCentAllAn3050->SaveAs(TString::Format("%s%s.%s", tSaveDir.Data(), tCanBgdwFitSingleCentAllAn3050->GetName(), tSaveFileType.Data()));
*/      
    }
  }



  if(bCompareAnalyses)
  {
    if(bDrawAllCentralities) tCanCompareAnalyses = CompareAnalyses_AllCent(tCfDescriptor, tSingleFileName, bCombineConjugates, tEventsType, tRebin, tMinNorm, tMaxNorm, bUseStavCf);
    else tCanCompareAnalyses = CompareAnalyses(tCfDescriptor, tSingleFileName, tImpactParam, bCombineConjugates, bCombineImpactParams, tEventsType, tRebin, tMinNorm, tMaxNorm, bUseStavCf);
    if(bSaveFigures) tCanCompareAnalyses->SaveAs(TString::Format("%s%s.%s", tSaveDir.Data(), tCanCompareAnalyses->GetName(), tSaveFileType.Data()));
  }



  if(bDrawLamKchPMBgdwFitOnly)
  {
    if(bDrawAllCentralities) tCanLamKchPMBgdwFit = DrawLamKchPMBgdwFit_AllCent(tCfDescriptor, tSingleFileName, tEventsType, tRebin, tMinNorm, tMaxNorm, tMaxBgdFit, bUseStavCf);
    else tCanLamKchPMBgdwFit = DrawLamKchPMBgdwFit(tCfDescriptor, tSingleFileName, tImpactParam, tEventsType, tRebin, tMinNorm, tMaxNorm, tMaxBgdFit, bUseStavCf);
    if(bSaveFigures) tCanLamKchPMBgdwFit->SaveAs(TString::Format("%s%s.%s", tSaveFileBase.Data(), tCanLamKchPMBgdwFit->GetName(), tSaveFileType.Data()));
  }

  if(bDrawDataVsTherm)
  {
    if(bDrawAllCentralities) tCanDataVsTherm = DrawDataVsTherm_AllCent(tCfDescriptor, tSingleFileName, tAnType, bCombineConjugates, tEventsType, 1, tMinNorm, tMaxNorm, bUseStavCf);
    else tCanDataVsTherm = DrawDataVsTherm(tCfDescriptor, tSingleFileName, tAnType, tImpactParam, bCombineConjugates, bCombineImpactParams, tEventsType, 1, tMinNorm, tMaxNorm, bUseStavCf);
    if(bSaveFigures) tCanDataVsTherm->SaveAs(TString::Format("%s%s.%s", tSaveFileBase.Data(), tCanDataVsTherm->GetName(), tSaveFileType.Data()));
  }

  if(bCompareBackgroundReductionMethods)
  {
    if(bDrawAllCentralities) tCanCompBgdRedMethods = CompareBackgroundReductionMethods_AllCent(tCfDescriptor, tAnType, bCombineConjugates, tEventsType, tRebin, tMinNorm, tMaxNorm, bNumWeight1, bDrawData, bVerbose);
    else tCanCompBgdRedMethods = CompareBackgroundReductionMethods(tCfDescriptor, tAnType, tImpactParam, bCombineConjugates, bCombineImpactParams, tEventsType, tRebin, tMinNorm, tMaxNorm, bNumWeight1, bDrawData, bVerbose);
    if(bSaveFigures) tCanCompBgdRedMethods->SaveAs(TString::Format("%s%s.%s", tSaveFileBase.Data(), tCanCompBgdRedMethods->GetName(), tSaveFileType.Data()));

  }

//-------------------------------------------------------------------------------




//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
