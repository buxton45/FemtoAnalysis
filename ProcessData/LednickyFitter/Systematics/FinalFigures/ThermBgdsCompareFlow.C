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

#include "ThermCf.h"
class ThermCf;

#include "CanvasPartition.h"


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
void PrintInfo(TPad* aPad, TString aOverallDescriptor, double aTextSize=0.04, double aX1=0.2, double aY1=1.05, double aX2=1.2, double aY2=1.05)
{
  aPad->cd();

  TLatex* tTex = new TLatex();
  tTex->SetTextAlign(12);
  tTex->SetLineWidth(2);
  tTex->SetTextFont(42);
  tTex->SetTextSize(aTextSize);

  tTex->DrawLatex(aX1, aY1, "THERMINATOR");
  tTex->DrawLatex(aX2, aY2, aOverallDescriptor);

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
void Draw1vs2vs3(TPad* aPad, AnalysisType aAnType, TH1* aCf1, TH1* aCf2, TH1* aCf3, TString aDescriptor1, TString aDescriptor2, TString aDescriptor3, TString aOverallDescriptor)
{
  aPad->cd();
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);


  //---------------------------------------------------------------
  aCf1->GetXaxis()->SetTitle("#it{k}* (GeV/#it{c})");
    aCf1->GetXaxis()->SetTitleSize(0.075);
    aCf1->GetXaxis()->SetTitleOffset(0.85);
    aCf1->GetXaxis()->SetLabelSize(0.05);

  aCf1->GetYaxis()->SetTitle("#it{C}(#it{k}*)");
    aCf1->GetYaxis()->SetTitleSize(0.085);
    aCf1->GetYaxis()->SetTitleOffset(0.70);
    aCf1->GetYaxis()->SetLabelSize(0.05);

  double tMaxDrawX = 3.0;
  aCf1->GetXaxis()->SetRangeUser(0.,tMaxDrawX);
  aCf1->GetYaxis()->SetRangeUser(0.65, 1.55);

  aCf1->Draw();
  aCf2->Draw("same");
  aCf3->Draw("same");

  //---------------------------------------------------------------

  TLegend* tLeg = new TLegend(0.60, 0.70, 0.85, 0.95);
    tLeg->SetFillColor(0);
    tLeg->SetBorderSize(0);
    tLeg->SetTextAlign(22);

  tLeg->AddEntry(aCf1, aDescriptor1.Data());
  tLeg->AddEntry(aCf2, aDescriptor2.Data());
  tLeg->AddEntry(aCf3, aDescriptor3.Data());

  //---------------------------------------------------------------

  aCf1->Draw("same");
  tLeg->Draw();


  TLine* tLine = new TLine(0, 1, tMaxDrawX, 1);
  tLine->SetLineColor(14);
  tLine->Draw();

  double tInfX1 = 0.4;
  double tInfY1 = 1.47;

  double tInfX2 = 0.4;
  double tInfY2 = 1.37;

  PrintInfo(aPad, aOverallDescriptor, 0.04, tInfX1, tInfY1, tInfX2, tInfY2);
}

//________________________________________________________________________________________________________________
TCanvas* CompareCfWithAndWithoutBgd(TString aCfDescriptor, AnalysisType aAnType, int aImpactParam, bool aCombineConjugates, bool aCombineImpactParams, ThermEventsType aEventsType, int aRebin, double aMinNorm, double aMaxNorm, bool aUseStavCf=false)
{
  //-------------------------------------------------
  CentralityType tCentType=kMB;
  if(aCombineImpactParams) tCentType = GetCentralityType(aImpactParam);
  //-------------------------------------------------
  AnalysisType tConjAnType = GetConjAnType(aAnType);
  //-------------------------------------------------
  TString tFileNameCfs1 = "CorrelationFunctions_RandomEPs_ArtificialV2Signal-1_NumWeight1.root";
  TString tFileNameCfs2 = "CorrelationFunctions_RandomEPs_ArtificialV3Signal-1_NumWeight1.root";
  TString tFileNameCfs3 = "CorrelationFunctions_RandomEPs_KillFlowSignals_NumWeight1.root";

  TString tDescriptor1 = "v_{2}=0.5, v_{3}=0.0";
  TString tDescriptor2 = "v_{2}=0.0, v_{3}=0.5";
  TString tDescriptor3 = "v_{2}=0.0, v_{3}=0.0";

  TString tOverallDescriptor = cAnalysisRootTags[aAnType];
  if(aCombineConjugates) tOverallDescriptor += TString::Format(" & %s", cAnalysisRootTags[tConjAnType]);
  if(!aCombineImpactParams) tOverallDescriptor += TString::Format(" (b=%d fm)", aImpactParam);
  else tOverallDescriptor += TString::Format(" (%s)", cPrettyCentralityTags[tCentType]);

  int tMarkerStyle1 = 20;
  int tMarkerStyle2 = 21;
  int tMarkerStyle3 = 26;

  int tColor1 = kBlack;
  int tColor2 = kGreen+1;
  int tColor3 = kOrange;

  //--------------------------------------------

  ThermCf* tThermCf1 = new ThermCf(tFileNameCfs1, aCfDescriptor, aAnType, GetCentralityType(aImpactParam), aCombineConjugates, aEventsType, aRebin, aMinNorm, aMaxNorm, aUseStavCf, aCombineImpactParams, aImpactParam);
  ThermCf* tThermCf2 = new ThermCf(tFileNameCfs2, aCfDescriptor, aAnType, GetCentralityType(aImpactParam), aCombineConjugates, aEventsType, aRebin, aMinNorm, aMaxNorm, aUseStavCf, aCombineImpactParams, aImpactParam);
  ThermCf* tThermCf3 = new ThermCf(tFileNameCfs3, aCfDescriptor, aAnType, GetCentralityType(aImpactParam), aCombineConjugates, aEventsType, aRebin, aMinNorm, aMaxNorm, aUseStavCf, aCombineImpactParams, aImpactParam);

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
  tCanCfsName = TString::Format("CompareFlowBgds_%s_%s", aCfDescriptor.Data(), cAnalysisBaseTags[aAnType]);

  if(aCombineConjugates) tCanCfsName += TString("wConj");
  if(!aCombineImpactParams) tCanCfsName += TString::Format("_b%d", aImpactParam);
  else tCanCfsName += TString(cCentralityTags[tCentType]);

  TCanvas* tCanCfs = new TCanvas(tCanCfsName, tCanCfsName);
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);
  tCanCfs->SetTopMargin(0.025);
  tCanCfs->SetRightMargin(0.025);
  tCanCfs->SetBottomMargin(0.15);
  tCanCfs->SetLeftMargin(0.125);
  Draw1vs2vs3((TPad*)tCanCfs, aAnType, tCf1, tCf2, tCf3, tDescriptor1, tDescriptor2, tDescriptor3, tOverallDescriptor);

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
  bool bCombineConjugates = true;
  bool bCombineImpactParams = false;

  ThermEventsType tEventsType = kMe;  //kMe, kAdam, kMeAndAdam

  bool bSaveFigures = false;
  TString tSaveFileType = "pdf";

  int tRebin=3; 
  double tMinNorm = 0.32;
  double tMaxNorm = 0.40;
  double tMaxBgdFit = 2.0;

  int tImpactParam = 5;

  TString tCfDescriptor = "Full";

  TString tSaveDir = "/home/jesse/Analysis/FemtoAnalysis/AnalysisNotes/5_Fitting/5.5_NonFlatBackground/Figures/";


  TCanvas *tCanCfs = CompareCfWithAndWithoutBgd(tCfDescriptor, tAnType, tImpactParam, bCombineConjugates, bCombineImpactParams, tEventsType, tRebin, tMinNorm, tMaxNorm, false);
  if(bSaveFigures) tCanCfs->SaveAs(TString::Format("%s%s.%s", tSaveDir.Data(), tCanCfs->GetName(), tSaveFileType.Data()));
 
//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
