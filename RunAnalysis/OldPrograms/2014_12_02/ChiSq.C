{
  const Int_t min_normbin = 60;
  const Int_t max_normbin = 75;
//----------------------------------------------------------
  TFile f1("Analysis1Resultsgrid_K0Lam_Bp1_Pur.root");
  TList *femtolist = (TList*)f1.Get("femtolist");
  TObjArray *K0Lam = (TObjArray*)femtolist->FindObject("K0Lam");
  TObjArray *K0ALam = (TObjArray*)femtolist->FindObject("K0ALam");

  TH1D *NumK0LamKStarcf1 = (TH1D*)K0Lam->FindObject("NumK0LamKStarcf");
  TH1D *DenK0LamKStarcf1 = (TH1D*)K0Lam->FindObject("DenK0LamKStarcf");

  TH1D *NumK0ALamKStarcf1 = (TH1D*)K0Lam->FindObject("NumK0ALamKStarcf");
  TH1D *DenK0ALamKStarcf1 = (TH1D*)K0Lam->FindObject("DenK0ALamKStarcf");

  Double_t myKStar_numScale1 = NumK0LamKStarcf1->Integral(min_normbin,max_normbin);
  NumK0LamKStarcf1->Scale(1./myKStar_numScale1);
  Double_t myKStar_denScale1 = DenK0LamKStarcf1->Integral(min_normbin,max_normbin);
  DenK0LamKStarcf1->Scale(1./myKStar_denScale1);

  TH1D *myKStarcf1 = NumK0LamKStarcf1->Clone("myKStarcf1");
  myKStarcf1->Divide(DenK0LamKStarcf1);
  myKStarcf1->SetTitle("K0-Lam");
  myKStarcf1->SetMarkerColor(1);
  myKStarcf1->SetMarkerSize(1.);
  myKStarcf1->SetLineColor(1);

//----------------------------------------------------------
  TFile f2("Analysis1Resultsgrid_K0Lam_v3_Bp1.root");
  TList *femtolist2 = (TList*)f2.Get("femtolist");

  TH1D *NummyKStarcf2 = (TH1D*)femtolist2->FindObject("NummyKStarcf");
  TH1D *DenmyKStarcf2 = (TH1D*)femtolist2->FindObject("DenmyKStarcf");

  Double_t myKStar_numScale2 = NummyKStarcf2->Integral(min_normbin,max_normbin);
  NummyKStarcf2->Scale(1./myKStar_numScale2);
  Double_t myKStar_denScale2 = DenmyKStarcf2->Integral(min_normbin,max_normbin);
  DenmyKStarcf2->Scale(1./myKStar_denScale2);

  TH1D *myKStarcf2 = NummyKStarcf2->Clone("myKStarcf2");
  myKStarcf2->Divide(DenmyKStarcf2);
  myKStarcf2->SetTitle("K0-Lam");
  myKStarcf2->SetMarkerColor(2);
  myKStarcf2->SetMarkerSize(2.);
  myKStarcf2->SetLineColor(2);
//----------------------------------------------------------
  myKStarcf1->Scale(100.);
  myKStarcf2->Scale(100.);
//  myKStarcf2->AddBinContent(70,100);  //check to make sure chi2 test working
  Double_t pvalue = myKStarcf1->Chi2Test(myKStarcf2,"WW");
  cout << "The p value = " << pvalue << endl;



//----------------------------------------------------------------------

  TCanvas* c1 = new TCanvas("c1","Plotting Canvas");
  myKStarcf1->Draw();
  myKStarcf2->Draw("same");



}
