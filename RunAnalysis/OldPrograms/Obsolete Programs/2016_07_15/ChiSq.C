TH1F* GetHisto(char* FileName, char* ListName, char* ArrayName, char* HistoName)
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
void Normalize(TH1F* fHisto, Int_t fMinNormBin, Int_t fMaxNormBin)
{
  float scale = fHisto->Integral(fMinNormBin,fMaxNormBin);
  fHisto->Scale(1./scale);
}
//_________________________________________________________________________________________
TH1F *BuildCorrFctn(char* name, TH1F* fNum, TH1F* fDen)
{
  TH1F* fCorrFctn = fNum->Clone(name);
  fCorrFctn->Divide(fDen);
  return fCorrFctn;
}
//_________________________________________________________________________________________
void ChiSq()
{
  const Int_t MinNormBin = 60;
  const Int_t MaxNormBin = 75;
  //-----------------------------
  TH1F *NumK0Lam1 = GetHisto("Resultsgrid_cLamK0_ss_NEW2.root", "femtolist", "K0Lam", "NumK0LamKStarcf");
    Normalize(NumK0Lam1,MinNormBin,MaxNormBin);
  TH1F *DenK0Lam1 = GetHisto("Resultsgrid_cLamK0_ss_NEW2.root", "femtolist", "K0Lam", "DenK0LamKStarcf");
    Normalize(DenK0Lam1,MinNormBin,MaxNormBin);
  TH1F *CfK0Lam1 = BuildCorrFctn("K0Lam1", NumK0Lam1, DenK0Lam1);
  NumK0Lam1->SetLineColor(1);
  DenK0Lam1->SetLineColor(1);
  CfK0Lam1->SetLineColor(1);

  TH1F *NumK0ALam1 = GetHisto("Resultsgrid_cLamK0_ss_NEW2.root", "femtolist", "K0ALam", "NumK0ALamKStarcf");
    Normalize(NumK0ALam1,MinNormBin,MaxNormBin);
  TH1F *DenK0ALam1 = GetHisto("Resultsgrid_cLamK0_ss_NEW2.root", "femtolist", "K0ALam", "DenK0ALamKStarcf");
    Normalize(DenK0ALam1,MinNormBin,MaxNormBin);
  TH1F *CfK0ALam1 = BuildCorrFctn("K0ALam1", NumK0ALam1, DenK0ALam1);
  NumK0ALam1->SetLineColor(1);
  DenK0ALam1->SetLineColor(1);
  CfK0ALam1->SetLineColor(1);

  //-----------------------------
  TH1F *NumK0Lam2 = GetHisto("TEST.root", "femtolist", "LamK0", "NumLamK0KStarCF1");
    Normalize(NumK0Lam2,MinNormBin,MaxNormBin);
  TH1F *DenK0Lam2 = GetHisto("TEST.root", "femtolist", "LamK0", "DenLamK0KStarCF1");
    Normalize(DenK0Lam2,MinNormBin,MaxNormBin);
  TH1F *CfK0Lam2 = BuildCorrFctn("K0Lam2", NumK0Lam2, DenK0Lam2);
  NumK0Lam2->SetLineColor(2);
  DenK0Lam2->SetLineColor(2);
  CfK0Lam2->SetLineColor(2);

  TH1F *NumK0ALam2 = GetHisto("TEST.root", "femtolist", "ALamK0", "NumALamK0KStarCF2");
    Normalize(NumK0ALam2,MinNormBin,MaxNormBin);
  TH1F *DenK0ALam2 = GetHisto("TEST.root", "femtolist", "ALamK0", "DenALamK0KStarCF2");
    Normalize(DenK0ALam2,MinNormBin,MaxNormBin);
  TH1F *CfK0ALam2 = BuildCorrFctn("K0ALam2", NumK0ALam2, DenK0ALam2);
  NumK0ALam2->SetLineColor(2);
  DenK0ALam2->SetLineColor(2);
  CfK0ALam2->SetLineColor(2);

  //-----------------------------


  TCanvas* c1 = new TCanvas("c1","Plotting Canvas");
  c1->Divide(2,3);
  //---------------------------LamK0---------------------------------------------
  //--------Numerators--------------------------
  c1->cd(1);
  Double_t p12Num_K0Lam = NumK0Lam1->Chi2Test(NumK0Lam2,"WW");
  NumK0Lam1->Draw();
  NumK0Lam2->Draw("same");
  //--------Denominators--------------------------
  c1->cd(3);
  Double_t p12Den_K0Lam = DenK0Lam1->Chi2Test(DenK0Lam2,"WW");
  DenK0Lam1->Draw();
  DenK0Lam2->Draw("same");
  //--------Correlation Functions--------------------------
  c1->cd(5);
  Double_t p12Cf_K0Lam = CfK0Lam1->Chi2Test(CfK0Lam2,"WW");
  CfK0Lam1->Draw();
  CfK0Lam2->Draw("same");
  //-------------------------------------------------------
  cout << "p12Num_K0Lam = " << p12Num_K0Lam << endl;
  cout << "p12Den_K0Lam = " << p12Den_K0Lam << endl;
  cout << "p12Cf_K0Lam = " << p12Cf_K0Lam << endl;

  //---------------------------ALamK0---------------------------------------------
  //--------Numerators--------------------------
  c1->cd(2);
  Double_t p12Num_K0ALam = NumK0ALam1->Chi2Test(NumK0ALam2,"WW");
  NumK0ALam1->Draw();
  NumK0ALam2->Draw("same");
  //--------Denominators--------------------------
  c1->cd(4);
  Double_t p12Den_K0ALam = DenK0ALam1->Chi2Test(DenK0ALam2,"WW");
  DenK0ALam1->Draw();
  DenK0ALam2->Draw("same");
  //--------Correlation Functions--------------------------
  c1->cd(6);
  Double_t p12Cf_K0ALam = CfK0ALam1->Chi2Test(CfK0ALam2,"WW");
  CfK0ALam1->Draw();
  CfK0ALam2->Draw("same");
  //-------------------------------------------------------
  cout << "p12Num_K0ALam = " << p12Num_K0ALam << endl;
  cout << "p12Den_K0ALam = " << p12Den_K0ALam << endl;
  cout << "p12Cf_K0ALam = " << p12Cf_K0ALam << endl;

}
