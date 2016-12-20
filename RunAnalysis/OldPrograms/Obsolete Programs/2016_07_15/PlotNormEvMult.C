#include "UsefulMacros.C"

void PlotNormEvMult()
{
  TString File = "Resultsgrid_cLamK0_CentBins_Bp1NEW2.root";

  TH1F *NormEvMult_All = GetHisto(File,"LamK0","NormEvMult_EvPass");
    NormEvMult_All->SetLineColor(1);

  TH1F *NormEvMult_0010 = GetHisto(File,"LamK0_0010","NormEvMult_EvPass");
    NormEvMult_0010->SetLineColor(2);

  TH1F *NormEvMult_1030 = GetHisto(File,"LamK0_1030","NormEvMult_EvPass");
    NormEvMult_1030->SetLineColor(4);

  TH1F *NormEvMult_3050 = GetHisto(File,"LamK0_3050","NormEvMult_EvPass");
    NormEvMult_3050->SetLineColor(6);

  TCanvas* aCanvas = new TCanvas("aCanvas","Plotting Canvas");
  aCanvas->Divide(2,2);
  gStyle->SetOptStat(0);

  aCanvas->cd(1);
  NormEvMult_All->GetXaxis()->SetRange(0,1000);
  NormEvMult_All->GetYaxis()->SetRangeUser(0,22000);
  NormEvMult_All->SetTitle("Centrality x 10");
  NormEvMult_All->Draw();

  aCanvas->cd(2);
  NormEvMult_0010->GetXaxis()->SetRange(0,1000);
  NormEvMult_0010->GetYaxis()->SetRangeUser(0,22000);
  NormEvMult_0010->SetTitle("Centrality x 10");
  NormEvMult_0010->Draw();

  aCanvas->cd(3);
  NormEvMult_1030->GetXaxis()->SetRange(0,1000);
  NormEvMult_1030->GetYaxis()->SetRangeUser(0,22000);
  NormEvMult_1030->SetTitle("Centrality x 10");
  NormEvMult_1030->Draw();

  aCanvas->cd(4);
  NormEvMult_3050->GetXaxis()->SetRange(0,1000);
  NormEvMult_3050->GetYaxis()->SetRangeUser(0,22000);
  NormEvMult_3050->SetTitle("Centrality x 10");
  NormEvMult_3050->Draw();


  aCanvas->SaveAs("~/Analysis/Presentations/AliFemto/2015.04.01/PlotNormEvMult_NEW2.pdf");
}
