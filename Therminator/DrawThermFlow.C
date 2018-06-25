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

#include "PIDMapping.h"

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
  else if(aPID==3122) tColor = kBlue+1;

  int tMarkerStyle = 20;
  if(!aV2) tMarkerStyle = 24;

  TString tParticleName;
  if(aPID==0) tParticleName = TString("UnIdent");
  else tParticleName = GetParticleNamev2(aPID);

  TString tFlowTitle = TString("v_{2}");
  if(!aV2) tFlowTitle = TString("v_{3}");

  tReturnGraph->SetMarkerSize(0.75);
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

  int tPIDUnIdent = 0;
  int tPIDK0s = 311;
  int tPIDKch = 321;
  int tPIDLam = 3122;

  int tImpactParam = 8;
  int tV3InclusionProb1 = 25;
  bool bDrawUnIdentOnly = false;

  vector<TString> tUnIdentOnlyTag = {"", "_UnIdentOnly"};

  TString tFileName = TString::Format("FlowGraphs_ArtificialV3Signal%d", tV3InclusionProb1);
//  TString tFileName = TString("FlowGraphs_RandomEPs");
  TString tFileLocation = TString::Format("/home/jesse/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b%d/%s.root", tImpactParam, tFileName.Data());

  TString tSaveDir = "/home/jesse/Analysis/Presentations/AliFemto/20180627/Figures/Flow/";

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
  double tYLow = -0.05;
  double tYHigh = 0.55;
  tGraphv2_UnIdent->GetXaxis()->SetLimits(tXLow, tXHigh);
  tGraphv2_UnIdent->GetYaxis()->SetRangeUser(tYLow, tYHigh);


  TCanvas* tFlowCan = new TCanvas(tFileName, tFileName);
  tFlowCan->cd();
  gStyle->SetOptTitle(0);

  tGraphv2_UnIdent->Draw("AP");
  if(!bDrawUnIdentOnly)
  {
    tGraphv2_K0s->Draw("Psame");
    tGraphv2_Kch->Draw("Psame");
    tGraphv2_Lam->Draw("Psame");
  }

  tGraphv3_UnIdent->Draw("Psame");  
  if(!bDrawUnIdentOnly)
  {
    tGraphv3_K0s->Draw("Psame");
    tGraphv3_Kch->Draw("Psame");
    tGraphv3_Lam->Draw("Psame");
  }

  //------------------------
  TLegend* tLeg1 = new TLegend(0.15, 0.60, 0.35, 0.80, "v_{2}", "NDC");
  tLeg1->SetFillColor(0);
  //tLeg1->SetBorderSize(0);
  tLeg1->SetTextAlign(22);
    tLeg1->AddEntry(tGraphv2_UnIdent, "UnIdent", "p");
    if(!bDrawUnIdentOnly)
    {
      tLeg1->AddEntry(tGraphv2_K0s, GetParticleNamev2(tPIDK0s), "p");
      tLeg1->AddEntry(tGraphv2_Kch, GetParticleNamev2(tPIDKch), "p");
      tLeg1->AddEntry(tGraphv2_Lam, GetParticleNamev2(tPIDLam), "p");
    }
  tLeg1->Draw();

  //------------------------

  TLegend* tLeg2 = new TLegend(0.35, 0.60, 0.55, 0.80, "v_{3}", "NDC");
  tLeg2->SetFillColor(0);
  //tLeg2->SetBorderSize(0);
  tLeg2->SetTextAlign(22);
    tLeg2->AddEntry(tGraphv3_UnIdent, "UnIdent", "p");
    if(!bDrawUnIdentOnly)
    {
      tLeg2->AddEntry(tGraphv3_K0s, GetParticleNamev2(tPIDK0s), "p");
      tLeg2->AddEntry(tGraphv3_Kch, GetParticleNamev2(tPIDKch), "p");
      tLeg2->AddEntry(tGraphv3_Lam, GetParticleNamev2(tPIDLam), "p");
    }
  tLeg2->Draw();

  //---------------------------------------

  if(bSaveFigures) tFlowCan->SaveAs(TString::Format("%s%s_b%d%s.eps", tSaveDir.Data(), tFileName.Data(), tImpactParam, tUnIdentOnlyTag[bDrawUnIdentOnly].Data()));

//-------------------------------------------------------------------------------




//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
