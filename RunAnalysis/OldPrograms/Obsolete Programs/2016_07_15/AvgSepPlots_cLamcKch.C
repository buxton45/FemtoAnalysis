#include "buildAllcLamcKch3.cxx"

void AvgSepPlots_cLamcKch()
{
  vector<TString> VectorOfFileNames;
    VectorOfFileNames.push_back("Resultsgrid_cLamcKch_CentBins_Bp1NEW.root");
    VectorOfFileNames.push_back("Resultsgrid_cLamcKch_CentBins_Bp2NEW.root");
    VectorOfFileNames.push_back("Resultsgrid_cLamcKch_CentBins_Bm1NEW.root");
    VectorOfFileNames.push_back("Resultsgrid_cLamcKch_CentBins_Bm2NEW.root");
    VectorOfFileNames.push_back("Resultsgrid_cLamcKch_CentBins_Bm3NEW.root");


  //---------------------------------------------------------------------------------------------------------------------------

  buildAllcLamcKch3 *cLamcKchAnalysis3_0010 = new buildAllcLamcKch3(VectorOfFileNames,"LamKchP_0010","LamKchM_0010","ALamKchP_0010","ALamKchM_0010");

  cLamcKchAnalysis3_0010->BuildAvgSepCollections();

  TH1F *AvgSepCf_TrackPos_LamKchP = cLamcKchAnalysis3_0010->GetAvgSepCf_TrackPos_LamKchP_Tot();
  TH1F *AvgSepCf_TrackNeg_LamKchM = cLamcKchAnalysis3_0010->GetAvgSepCf_TrackNeg_LamKchM_Tot();
  TH1F *AvgSepCf_TrackPos_ALamKchP = cLamcKchAnalysis3_0010->GetAvgSepCf_TrackPos_ALamKchP_Tot();
  TH1F *AvgSepCf_TrackNeg_ALamKchM = cLamcKchAnalysis3_0010->GetAvgSepCf_TrackNeg_ALamKchM_Tot();

  //--A horizontal line at y=1 to help guide the eye
  TLine *line = new TLine(0,1,20,1);
  line->SetLineColor(14);

  TCanvas* aCanvas = new TCanvas("aCanvas","Plotting Canvas");
  aCanvas->Divide(2,2);
  gStyle->SetOptStat(0);

  aCanvas->cd(1);
  AvgSepCf_TrackPos_LamKchP->GetYaxis()->SetRangeUser(-0.5,5.);
  AvgSepCf_TrackPos_LamKchP->SetTitle("p(#Lambda) - K+");
  AvgSepCf_TrackPos_LamKchP->Draw();
  line->Draw();

  aCanvas->cd(2);
  AvgSepCf_TrackNeg_LamKchM->GetYaxis()->SetRangeUser(-0.5,5.);
  AvgSepCf_TrackNeg_LamKchM->SetTitle("#pi^{-}(#Lambda) - K-");
  AvgSepCf_TrackNeg_LamKchM->Draw();
  line->Draw();

  aCanvas->cd(3);
  AvgSepCf_TrackPos_ALamKchP->GetYaxis()->SetRangeUser(-0.5,5.);
  AvgSepCf_TrackPos_ALamKchP->SetTitle("#pi^{+}(#bar{#Lambda}) - K+");
  AvgSepCf_TrackPos_ALamKchP->Draw();
  line->Draw();

  aCanvas->cd(4);
  AvgSepCf_TrackNeg_ALamKchM->GetYaxis()->SetRangeUser(-0.5,5.);
  AvgSepCf_TrackNeg_ALamKchM->SetTitle("#bar{p}(#bar{#Lambda}) - K-");
  AvgSepCf_TrackNeg_ALamKchM->Draw();
  line->Draw();


  aCanvas->SaveAs("~/Analysis/K0Lam/NEW/2015_03_06/cLamcKch/AvgSepCFs/AvgSepCFs.pdf");


}
