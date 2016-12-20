#include "UsefulMacros.C"
//_________________________________________________________________________________________
void buildAvgSepCorr()
{
  const Int_t MinNormBin = 150;
  const Int_t MaxNormBin = 200;
  //_____________________________________BP1________________________________
  TString File_Bp1 = "Resultsgrid_cLamK0_CentBins_Bp1NEW.root";
  //-------------Lam-K0------------------
  //----- ++ ----
  TH1F *NumPosPosAvgSepCF_LamK0 = GetHisto(File_Bp1, "LamK0", "NumPosPosAvgSepCF_LamK0");
    NumPosPosAvgSepCF_LamK0->SetLineColor(1);
  TH1F *DenPosPosAvgSepCF_LamK0 = GetHisto(File_Bp1, "LamK0", "DenPosPosAvgSepCF_LamK0");
    DenPosPosAvgSepCF_LamK0->SetLineColor(1);
  TH1F *PosPosAvgSepCF_LamK0 = buildCF("PosPosAvgSepCF_LamK0","Lam-K0 (B+ 1)",NumPosPosAvgSepCF_LamK0,DenPosPosAvgSepCF_LamK0,MinNormBin,MaxNormBin);
    PosPosAvgSepCF_LamK0->SetLineColor(1);

  //----- +- ----
  TH1F *NumPosNegAvgSepCF_LamK0 = GetHisto(File_Bp1, "LamK0", "NumPosNegAvgSepCF_LamK0");
    NumPosNegAvgSepCF_LamK0->SetLineColor(1);
  TH1F *DenPosNegAvgSepCF_LamK0 = GetHisto(File_Bp1, "LamK0", "DenPosNegAvgSepCF_LamK0");
    DenPosNegAvgSepCF_LamK0->SetLineColor(1);
  TH1F *PosNegAvgSepCF_LamK0 = buildCF("PosNegAvgSepCF_LamK0","Lam-K0 (B+ 1)",NumPosNegAvgSepCF_LamK0,DenPosNegAvgSepCF_LamK0,MinNormBin,MaxNormBin);
    PosNegAvgSepCF_LamK0->SetLineColor(1);

  //----- -+ ----
  TH1F *NumNegPosAvgSepCF_LamK0 = GetHisto(File_Bp1, "LamK0", "NumNegPosAvgSepCF_LamK0");
    NumNegPosAvgSepCF_LamK0->SetLineColor(1);
  TH1F *DenNegPosAvgSepCF_LamK0 = GetHisto(File_Bp1, "LamK0", "DenNegPosAvgSepCF_LamK0");
    DenNegPosAvgSepCF_LamK0->SetLineColor(1);
  TH1F *NegPosAvgSepCF_LamK0 = buildCF("NegPosAvgSepCF_LamK0","Lam-K0 (B+ 1)",NumNegPosAvgSepCF_LamK0,DenNegPosAvgSepCF_LamK0,MinNormBin,MaxNormBin);
    NegPosAvgSepCF_LamK0->SetLineColor(1);

  //----- -+ ----
  TH1F *NumNegNegAvgSepCF_LamK0 = GetHisto(File_Bp1, "LamK0", "NumNegNegAvgSepCF_LamK0");
    NumNegNegAvgSepCF_LamK0->SetLineColor(1);
  TH1F *DenNegNegAvgSepCF_LamK0 = GetHisto(File_Bp1, "LamK0", "DenNegNegAvgSepCF_LamK0");
    DenNegNegAvgSepCF_LamK0->SetLineColor(1);
  TH1F *NegNegAvgSepCF_LamK0 = buildCF("NegNegAvgSepCF_LamK0","Lam-K0 (B+ 1)",NumNegNegAvgSepCF_LamK0,DenNegNegAvgSepCF_LamK0,MinNormBin,MaxNormBin);
    NegNegAvgSepCF_LamK0->SetLineColor(1);



  //-------------ALam-K0------------------
  //----- ++ ----
  TH1F *NumPosPosAvgSepCF_ALamK0 = GetHisto(File_Bp1, "ALamK0", "NumPosPosAvgSepCF_ALamK0");
    NumPosPosAvgSepCF_ALamK0->SetLineColor(1);
  TH1F *DenPosPosAvgSepCF_ALamK0 = GetHisto(File_Bp1, "ALamK0", "DenPosPosAvgSepCF_ALamK0");
    DenPosPosAvgSepCF_ALamK0->SetLineColor(1);
  TH1F *PosPosAvgSepCF_ALamK0 = buildCF("PosPosAvgSepCF_ALamK0","ALam-K0 (B+ 1)",NumPosPosAvgSepCF_ALamK0,DenPosPosAvgSepCF_ALamK0,MinNormBin,MaxNormBin);
    PosPosAvgSepCF_ALamK0->SetLineColor(1);

  //----- +- ----
  TH1F *NumPosNegAvgSepCF_ALamK0 = GetHisto(File_Bp1, "ALamK0", "NumPosNegAvgSepCF_ALamK0");
    NumPosNegAvgSepCF_ALamK0->SetLineColor(1);
  TH1F *DenPosNegAvgSepCF_ALamK0 = GetHisto(File_Bp1, "ALamK0", "DenPosNegAvgSepCF_ALamK0");
    DenPosNegAvgSepCF_ALamK0->SetLineColor(1);
  TH1F *PosNegAvgSepCF_ALamK0 = buildCF("PosNegAvgSepCF_ALamK0","ALam-K0 (B+ 1)",NumPosNegAvgSepCF_ALamK0,DenPosNegAvgSepCF_ALamK0,MinNormBin,MaxNormBin);
    PosNegAvgSepCF_ALamK0->SetLineColor(1);

  //----- -+ ----
  TH1F *NumNegPosAvgSepCF_ALamK0 = GetHisto(File_Bp1, "ALamK0", "NumNegPosAvgSepCF_ALamK0");
    NumNegPosAvgSepCF_ALamK0->SetLineColor(1);
  TH1F *DenNegPosAvgSepCF_ALamK0 = GetHisto(File_Bp1, "ALamK0", "DenNegPosAvgSepCF_ALamK0");
    DenNegPosAvgSepCF_ALamK0->SetLineColor(1);
  TH1F *NegPosAvgSepCF_ALamK0 = buildCF("NegPosAvgSepCF_ALamK0","ALam-K0 (B+ 1)",NumNegPosAvgSepCF_ALamK0,DenNegPosAvgSepCF_ALamK0,MinNormBin,MaxNormBin);
    NegPosAvgSepCF_ALamK0->SetLineColor(1);

  //----- -+ ----
  TH1F *NumNegNegAvgSepCF_ALamK0 = GetHisto(File_Bp1, "ALamK0", "NumNegNegAvgSepCF_ALamK0");
    NumNegNegAvgSepCF_ALamK0->SetLineColor(1);
  TH1F *DenNegNegAvgSepCF_ALamK0 = GetHisto(File_Bp1, "ALamK0", "DenNegNegAvgSepCF_ALamK0");
    DenNegNegAvgSepCF_ALamK0->SetLineColor(1);
  TH1F *NegNegAvgSepCF_ALamK0 = buildCF("NegNegAvgSepCF_ALamK0","ALam-K0 (B+ 1)",NumNegNegAvgSepCF_ALamK0,DenNegNegAvgSepCF_ALamK0,MinNormBin,MaxNormBin);
    NegNegAvgSepCF_ALamK0->SetLineColor(1);


  TLine *line = new TLine(0,1,20,1);
  line->SetLineColor(14);

  TCanvas* c1 = new TCanvas("c1","Plotting Canvas");
  c1->Divide(2,2);

  c1->cd(1);
  PosPosAvgSepCF_LamK0->GetYaxis()->SetRangeUser(-0.5,5.);
  PosPosAvgSepCF_LamK0->Draw();
  line->Draw();

  c1->cd(2);
  PosNegAvgSepCF_LamK0->GetYaxis()->SetRangeUser(-0.5,5.);
  PosNegAvgSepCF_LamK0->Draw();
  line->Draw();

  c1->cd(3);
  NegPosAvgSepCF_LamK0->GetYaxis()->SetRangeUser(-0.5,5.);
  NegPosAvgSepCF_LamK0->Draw();
  line->Draw();

  c1->cd(4);
  NegNegAvgSepCF_LamK0->GetYaxis()->SetRangeUser(-0.5,5.);
  NegNegAvgSepCF_LamK0->Draw();
  line->Draw();

  TCanvas* c2 = new TCanvas("c2","Plotting Canvas");
  c2->Divide(2,2);

  c2->cd(1);
  PosPosAvgSepCF_ALamK0->GetYaxis()->SetRangeUser(-0.5,5.);
  PosPosAvgSepCF_ALamK0->Draw();
  line->Draw();

  c2->cd(2);
  PosNegAvgSepCF_ALamK0->GetYaxis()->SetRangeUser(-0.5,5.);
  PosNegAvgSepCF_ALamK0->Draw();
  line->Draw();

  c2->cd(3);
  NegPosAvgSepCF_ALamK0->GetYaxis()->SetRangeUser(-0.5,5.);
  NegPosAvgSepCF_ALamK0->Draw();
  line->Draw();

  c2->cd(4);
  NegNegAvgSepCF_ALamK0->GetYaxis()->SetRangeUser(-0.5,5.);
  NegNegAvgSepCF_ALamK0->Draw();
  line->Draw();

}
