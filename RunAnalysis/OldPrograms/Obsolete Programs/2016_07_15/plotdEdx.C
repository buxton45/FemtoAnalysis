//_________________________________________________________________________________________
TH1F* GetHisto(TString FileName, TString ListName, TString ArrayName, TString HistoName)
{
  TFile f1(FileName);
  TList *femtolist = (TList*)f1.Get(ListName);
  if(ArrayName)
    {
      TObjArray *array = (TObjArray*)femtolist->FindObject(ArrayName);
      TH1F *ReturnHisto = (TH1F*)array->FindObject(HistoName);
    }
  else
    {
      TH1F *ReturnHisto = (TH1F*)femtolist->FindObject(HistoName);
    }
  return ReturnHisto;
}

//_________________________________________________________________________________________
void plotdEdx()
{
  TString File_Bp1 = "Resultsgrid_cLamK0_Bp1.root";

  TH1F* fdEdxPosDaughter_Lam_Bp1 = GetHisto(File_Bp1, "femtolist", "LamK0", "fdEdxPosDaughter_Lam_Pass");
  TH1F* fdEdxNegDaughter_Lam_Bp1 = GetHisto(File_Bp1, "femtolist", "LamK0", "fdEdxNegDaughter_Lam_Pass");


  TCanvas* c1 = new TCanvas("c1","Plotting Canvas");
  gStyle->SetOptStat(0);

  fdEdxPosDaughter_Lam_Bp1->SetTitle("dEdx of #Lambda daughters");
  fdEdxPosDaughter_Lam_Bp1->Draw("CONT");
  fdEdxNegDaughter_Lam_Bp1->Draw("CONT same");

  c1->SaveAs("plotdEdx_Bp1.pdf");

}
