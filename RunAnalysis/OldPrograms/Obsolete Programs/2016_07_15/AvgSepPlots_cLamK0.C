#include "buildAllcLamK03.cxx"

void AvgSepPlots_cLamK0()
{

  //----- 0010 --------------------------------------------------------------------------------------
  vector<TString> VectorOfFileNames_0010;
    VectorOfFileNames_0010.push_back("Resultsgrid_cLamK0_CentBins_Bp1NEW.root");
    VectorOfFileNames_0010.push_back("Resultsgrid_cLamK0_CentBins_Bp2NEW.root");
    VectorOfFileNames_0010.push_back("Resultsgrid_cLamK0_CentBins_Bm1NEW.root");
    VectorOfFileNames_0010.push_back("Resultsgrid_cLamK0_CentBins_Bm2NEW.root");
    VectorOfFileNames_0010.push_back("Resultsgrid_cLamK0_CentBins_Bm3NEW.root");


  buildAllcLamK03 *cLamK0Analysis3_0010 = new buildAllcLamK03(VectorOfFileNames_0010,"LamK0_0010","ALamK0_0010");
  cLamK0Analysis3_0010->BuildAvgSepCollections();

  TH1F *AvgSepCf_PosPos_LamK0 = cLamK0Analysis3_0010->GetAvgSepCf_PosPos_LamK0_Tot();
  TH1F *AvgSepCf_NegNeg_LamK0 = cLamK0Analysis3_0010->GetAvgSepCf_NegNeg_LamK0_Tot();
  TH1F *AvgSepCf_PosPos_ALamK0 = cLamK0Analysis3_0010->GetAvgSepCf_PosPos_ALamK0_Tot();
  TH1F *AvgSepCf_NegNeg_ALamK0 = cLamK0Analysis3_0010->GetAvgSepCf_NegNeg_ALamK0_Tot();

  //--A horizontal line at y=1 to help guide the eye
  TLine *line = new TLine(0,1,20,1);
  line->SetLineColor(14);

  TCanvas* aCanvas = new TCanvas("aCanvas","Plotting Canvas");
  aCanvas->Divide(2,2);
  gStyle->SetOptStat(0);

  aCanvas->cd(1);
  AvgSepCf_PosPos_LamK0->GetYaxis()->SetRangeUser(-0.5,5.);
  AvgSepCf_PosPos_LamK0->SetTitle("p(#Lambda) - #pi^{+}(K^{0})");
  AvgSepCf_PosPos_LamK0->Draw();
  line->Draw();

  aCanvas->cd(2);
  AvgSepCf_NegNeg_LamK0->GetYaxis()->SetRangeUser(-0.5,5.);
  AvgSepCf_NegNeg_LamK0->SetTitle("#pi^{-}(#Lambda) - #pi^{-}(K^{0})");
  AvgSepCf_NegNeg_LamK0->Draw();
  line->Draw();

  aCanvas->cd(3);
  AvgSepCf_PosPos_ALamK0->GetYaxis()->SetRangeUser(-0.5,5.);
  AvgSepCf_PosPos_ALamK0->SetTitle("#pi^{+}(#bar{#Lambda}) - #pi^{+}(K^{0})");
  AvgSepCf_PosPos_ALamK0->Draw();
  line->Draw();

  aCanvas->cd(4);
  AvgSepCf_NegNeg_ALamK0->GetYaxis()->SetRangeUser(-0.5,5.);
  AvgSepCf_NegNeg_ALamK0->SetTitle("#bar{p}(#bar{#Lambda}) - #pi^{-}(K^{0})");
  AvgSepCf_NegNeg_ALamK0->Draw();
  line->Draw();

  aCanvas->SaveAs("~/Analysis/K0Lam/NEW/2015_03_06/cLamcK0/AvgSepCFs/AvgSepCFs.pdf");

}
