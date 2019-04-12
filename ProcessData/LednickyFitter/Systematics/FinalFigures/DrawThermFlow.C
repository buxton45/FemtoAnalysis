//From file in Therminator directory of same name

#include <iostream>
#include <iomanip>
#include <math.h>
#include <vector>
#include <assert.h>

#include "TFile.h"
#include "TApplication.h"
#include "TStyle.h"
#include "TPaveText.h"
#include "TLegend.h"
#include "TF1.h"
#include "TLatex.h"
#include "TGraph.h"
#include "TGraphErrors.h"
#include "TCanvas.h"
#include "TAxis.h"
#include "TLegend.h"
#include "TLegendEntry.h"

#include "PIDMapping.h"

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
TGraphErrors* GetFlowGraph(TString aFileLocation, int aPID, bool aV2=true)
{
  TFile tFile(aFileLocation);
  TString tGraphName;
  if(aV2) tGraphName = TString("v2_");
  else    tGraphName = TString("v3_");

  if(aPID==0) tGraphName += TString("UnIdent");
  else        tGraphName += GetParticleNamev2(aPID);
  TGraphErrors* tReturnGraph = (TGraphErrors*)tFile.Get(tGraphName);

  int tColor = 1;
  if(aPID==321) tColor = kRed;
  else if(aPID==311) tColor = kYellow-3;
  else if(aPID==3122) tColor = kGreen+1;

  int tMarkerStyle = 20;
  if(!aV2) tMarkerStyle = 34;

  TString tParticleName;
  if(aPID==0) tParticleName = TString("UnIdent");
  else tParticleName = GetParticleNamev2(aPID);

  TString tFlowTitle = TString("v_{2}");
  if(!aV2) tFlowTitle = TString("v_{3}");

  tReturnGraph->SetMarkerSize(1.0);
  tReturnGraph->SetMarkerColor(tColor);
  tReturnGraph->SetMarkerStyle(tMarkerStyle);
  tReturnGraph->SetLineColor(tColor);
  tReturnGraph->SetTitle(TString::Format("%s vs. p_{T} (%s)", tFlowTitle.Data(), tParticleName.Data()));
  tReturnGraph->SetName(TString::Format("%s_%s", tFlowTitle.Data(), tParticleName.Data()));

  return tReturnGraph;
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


  bool bSaveFigures = false;
  TString tSaveFileType = "pdf";

  int tPIDUnIdent = 0;
  int tPIDK0s = 311;
  int tPIDKch = 321;
  int tPIDLam = 3122;

  int tImpactParam = 8;
  bool bDrawUnIdentOnlyV2 = false;
  bool bDrawV3 = true;
  bool bDrawUnIdentOnlyV3 = true;

  if(!bDrawV3) bDrawUnIdentOnlyV3=false;  //Just so _UnIdentOnlyV3 tag won't be included in save name

  vector<TString> tUnIdentOnlyTagV2 = {"", "_UnIdentOnlyV2"};
  vector<TString> tUnIdentOnlyTagV3 = {"", "_UnIdentOnlyV3"};

  TString tFileName = TString("FlowGraphs_RandomEPs");
  TString tFileLocation = TString::Format("/home/jesse/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b%d/%s.root", tImpactParam, tFileName.Data());
  if(!bDrawV3) tFileName += TString("_V2Only");

  TString tSaveDir = "/home/jesse/Analysis/FemtoAnalysis/AnalysisNotes/5_Fitting/5.5_NonFlatBackground/Figures/";

  //-------------------------------------------------------------------------------------

  TGraphErrors* tGraphv2_UnIdent = GetFlowGraph(tFileLocation, tPIDUnIdent, true);
  TGraphErrors* tGraphv2_K0s = GetFlowGraph(tFileLocation, tPIDK0s, true);
  TGraphErrors* tGraphv2_Kch = GetFlowGraph(tFileLocation, tPIDKch, true);
  TGraphErrors* tGraphv2_Lam = GetFlowGraph(tFileLocation, tPIDLam, true);

  TGraphErrors* tGraphv3_UnIdent = GetFlowGraph(tFileLocation, tPIDUnIdent, false);
  TGraphErrors* tGraphv3_K0s = GetFlowGraph(tFileLocation, tPIDK0s, false);
  TGraphErrors* tGraphv3_Kch = GetFlowGraph(tFileLocation, tPIDKch, false);
  TGraphErrors* tGraphv3_Lam = GetFlowGraph(tFileLocation, tPIDLam, false);

  double tXLow = 0.;
  double tXHigh = 3.;
  double tYLow = -0.04;
  double tYHigh = 0.42;
  tGraphv2_UnIdent->GetXaxis()->SetLimits(tXLow, tXHigh);
  tGraphv2_UnIdent->GetYaxis()->SetRangeUser(tYLow, tYHigh);

  tGraphv2_UnIdent->GetXaxis()->SetTitle("#LT#it{p}_{T}#GT (GeV/#it{c})");
    tGraphv2_UnIdent->GetXaxis()->SetTitleSize(0.065);
    tGraphv2_UnIdent->GetXaxis()->SetTitleOffset(0.96);
    tGraphv2_UnIdent->GetXaxis()->SetLabelSize(0.045);

  if(bDrawV3) tGraphv2_UnIdent->GetYaxis()->SetTitle("#it{v}_{n}{EP}");
  else        tGraphv2_UnIdent->GetYaxis()->SetTitle("#it{v}_{2}{EP}");
    tGraphv2_UnIdent->GetYaxis()->SetTitleSize(0.075);
    tGraphv2_UnIdent->GetYaxis()->SetTitleOffset(0.80);
    tGraphv2_UnIdent->GetYaxis()->SetLabelSize(0.045);

  TCanvas* tFlowCan = new TCanvas(tFileName, tFileName);
  tFlowCan->cd();
  gStyle->SetOptTitle(0);
    tFlowCan->SetTopMargin(0.025);
    tFlowCan->SetRightMargin(0.025);
    tFlowCan->SetBottomMargin(0.15);
    tFlowCan->SetLeftMargin(0.125);


  tGraphv2_UnIdent->Draw("AP");
  if(!bDrawUnIdentOnlyV2)
  {
    tGraphv2_K0s->Draw("Psame");
    tGraphv2_Kch->Draw("Psame");
    tGraphv2_Lam->Draw("Psame");
    tGraphv2_UnIdent->Draw("Psame");
  }

  if(bDrawV3)
  {
    tGraphv3_UnIdent->Draw("Psame");  
    if(!bDrawUnIdentOnlyV3)
    {
      tGraphv3_K0s->Draw("Psame");
      tGraphv3_Kch->Draw("Psame");
      tGraphv3_Lam->Draw("Psame");
      tGraphv3_UnIdent->Draw("Psame");
    }
  }

  //------------------------
  TLegend* tLeg1;
  if(bDrawUnIdentOnlyV2) tLeg1 = new TLegend(0.175, 0.77, 0.335, 0.90, "#it{v}_{2}", "NDC");
  else                   tLeg1 = new TLegend(0.175, 0.575, 0.335, 0.90, "#it{v}_{2}", "NDC");
  tLeg1->SetFillColor(0);
  //tLeg1->SetBorderSize(0);
  tLeg1->SetTextAlign(22);
  tLeg1->SetTextSize(0.045);
    tLeg1->AddEntry(tGraphv2_UnIdent, "Unident.", "p");
    if(!bDrawUnIdentOnlyV2)
    {
      tLeg1->AddEntry(tGraphv2_K0s, GetParticleNamev2(tPIDK0s), "p");
      tLeg1->AddEntry(tGraphv2_Kch, "K^{#pm}", "p");
      tLeg1->AddEntry(tGraphv2_Lam, "#Lambda+#bar{#Lambda}", "p");
    }
  TLegendEntry *tHeader1 = (TLegendEntry*)tLeg1->GetListOfPrimitives()->First();
    tHeader1->SetTextSize(0.065);
    tHeader1->SetTextFont(62);
  tLeg1->Draw();

  //------------------------
  if(bDrawV3)
  {
    TLegend* tLeg2;
    if(bDrawUnIdentOnlyV3) tLeg2 = new TLegend(0.335, 0.77, 0.495, 0.90, "#it{v}_{3}", "NDC");
    else                   tLeg2 = new TLegend(0.335, 0.575, 0.495, 0.90, "#it{v}_{3}", "NDC");
    tLeg2->SetFillColor(0);
    //tLeg2->SetBorderSize(0);
    tLeg2->SetTextAlign(22);
    tLeg2->SetTextSize(0.045);
      tLeg2->AddEntry(tGraphv3_UnIdent, "Unident.", "p");
      if(!bDrawUnIdentOnlyV3)
      {
        tLeg2->AddEntry(tGraphv3_K0s, GetParticleNamev2(tPIDK0s), "p");
        tLeg2->AddEntry(tGraphv3_Kch, "K^{#pm}", "p");
        tLeg2->AddEntry(tGraphv3_Lam, "#Lambda+#bar{#Lambda}", "p");
      }
    TLegendEntry *tHeader2 = (TLegendEntry*)tLeg2->GetListOfPrimitives()->First();
      tHeader2->SetTextSize(0.065);
      tHeader2->SetTextFont(62);
    tLeg2->Draw();
  }
  //---------------------------------------
  TString tOverallDescriptor = TString::Format("b=%d fm", tImpactParam);

  double tInfX1 = 1.5;
  double tInfY1 = 0.37;

  double tInfX2 = 1.5;
  double tInfY2 = 0.33;

  PrintInfo((TPad*)tFlowCan, tOverallDescriptor, 0.05, tInfX1, tInfY1, tInfX2, tInfY2);

  //---------------------------------------

  if(bSaveFigures) tFlowCan->SaveAs(TString::Format("%s%s_b%d%s%s.%s", tSaveDir.Data(), tFileName.Data(), tImpactParam, tUnIdentOnlyTagV2[bDrawUnIdentOnlyV2].Data(), tUnIdentOnlyTagV3[bDrawUnIdentOnlyV3].Data(), tSaveFileType.Data()));

//-------------------------------------------------------------------------------




//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
