#include "UsefulMacros.C"

void CompareEventTriggers()
{
  TString File_kMB = "Resultsgrid_cLamK0_CentBins_Bp1NEW_kMB.root";
  TString File_kSC = "Resultsgrid_cLamK0_CentBins_Bp1NEW_kSC.root";
  TString File_kC = "Resultsgrid_cLamK0_CentBins_Bp1NEW_kC.root";

  TH1F *NormEvMult_kMB = GetHisto(File_kMB,"LamK0","NormEvMult_EvPass");
    NormEvMult_kMB->SetLineColor(15);
  TH1F *NormEvMult_kSC = GetHisto(File_kSC,"LamK0","NormEvMult_EvPass");
    NormEvMult_kSC->SetLineColor(12);
  TH1F *NormEvMult_kC = GetHisto(File_kC,"LamK0","NormEvMult_EvPass");
    NormEvMult_kC->SetLineColor(1);

  NormEvMult_kC->GetXaxis()->SetRange(0,1000);

  NormEvMult_kC->Draw();
  NormEvMult_kSC->Draw("same");
  NormEvMult_kMB->Draw("same");

  //--------------------------------------------------------------------
  TString File_kMB2 = "Resultsgrid_cLamK0_CentBins_Bp1NEW2_kMB.root";
  TString File_kSC2 = "Resultsgrid_cLamK0_CentBins_Bp1NEW2_kSC.root";
  TString File_kC2 = "Resultsgrid_cLamK0_CentBins_Bp1NEW2_kC.root";

  TH1F *NormEvMult_kMB2 = GetHisto(File_kMB2,"LamK0","NormEvMult_EvPass");
    NormEvMult_kMB2->SetLineColor(kRed-7);
  TH1F *NormEvMult_kSC2 = GetHisto(File_kSC2,"LamK0","NormEvMult_EvPass");
    NormEvMult_kSC2->SetLineColor(kRed-4);
  TH1F *NormEvMult_kC2 = GetHisto(File_kC2,"LamK0","NormEvMult_EvPass");
    NormEvMult_kC2->SetLineColor(kRed);

  NormEvMult_kC2->Draw("same");
  NormEvMult_kSC2->Draw("same");
  NormEvMult_kMB2->Draw("same");

  //--------------------------------------------------------------------
  TString File_kMB3 = "Resultsgrid_cLamK0_CentBins_Bp1NEW3_kMB.root";
  TString File_kSC3 = "Resultsgrid_cLamK0_CentBins_Bp1NEW3_kSC.root";
  TString File_kC3 = "Resultsgrid_cLamK0_CentBins_Bp1NEW3_kC.root";

  TH1F *NormEvMult_kMB3 = GetHisto(File_kMB3,"LamK0","NormEvMult_EvPass");
    NormEvMult_kMB3->SetLineColor(kGreen+1);
  TH1F *NormEvMult_kSC3 = GetHisto(File_kSC3,"LamK0","NormEvMult_EvPass");
    NormEvMult_kSC3->SetLineColor(kGreen+2);
  TH1F *NormEvMult_kC3 = GetHisto(File_kC3,"LamK0","NormEvMult_EvPass");
    NormEvMult_kC3->SetLineColor(kGreen+3);

  NormEvMult_kC3->Draw("same");
  NormEvMult_kSC3->Draw("same");
  NormEvMult_kMB3->Draw("same");















  c1->SaveAs("CompareEventTriggers.pdf");
}
