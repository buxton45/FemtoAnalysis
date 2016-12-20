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
void DrawNorm(TH1F* Histo, short color)
{
//  double scale = Histo->Integral(0,Histo->GetNbinsX()+1);
  //cout << scale << endl;
//  Histo->Scale(1./scale);
  Histo->SetMarkerColor(color);
  Histo->SetMarkerSize(0.75);
  //Histo->SetMarkerStyle(20);
  Histo->SetLineColor(color);
  Histo->DrawNormalized("epsame");
  Histo->DrawNormalized("same lhist");
}

//_________________________________________________________________________________________
int FindFirstNonZeroBin(TH1F* Histo)
{
  for(int i=1;i<Histo->GetNbinsX();i++)
    {
      if(Histo->GetBinContent(i) != 0) break;
    }
  return i;
}

//_________________________________________________________________________________________
int FindLastNonZeroBin(TH1F* Histo)
{
  for(int i=Histo->GetNbinsX(); i>0; i--)
    {
      if(Histo->GetBinContent(i) !=0) break;
    }
  return i;
}

//_________________________________________________________________________________________
void compCuts2()
{
  TCanvas* c1 = new TCanvas("c1","Plotting Canvas");
  c1->Divide(1,2);
  gStyle->SetOptStat(0);

  //-------------------------------------------------------
  c1->cd(1);
  TH1F* K0Pass_LamMass_1 = GetHisto("Resultsgrid_cLamK0_ss_OLD_NoMisID.root","femtolist","K0Lam","LambdaMass_K01_Pass");
  int FirstNonZero = FindFirstNonZeroBin(K0Pass_LamMass_1);
  //int LastNonZero = FindLastNonZeroBin(K0Pass_LamMass_1);
  int LastNonZero = K0Pass_LamMass_1->FindBin(1.80);
  TAxis *xax1 = K0Pass_LamMass_1->GetXaxis();
    xax1->SetRange(FirstNonZero,LastNonZero);
    xax1->SetTitle("m_{p#pi-} (GeV/c^{2})");
    xax1->SetTitleSize(0.055);
    xax1->SetTitleOffset(0.75);
    xax1->SetLabelSize(0.05);
  TAxis *yax1 = K0Pass_LamMass_1->GetYaxis();
    yax1->SetTitle("dN/dm_{p#pi-}");
    yax1->SetTitleSize(0.06);
    yax1->SetTitleOffset(0.8);
  K0Pass_LamMass_1->SetTitle("Assume #Lambda hypothesis (m_{p#pi-}) for particles passing K^{0} cuts");
  DrawNorm(K0Pass_LamMass_1,1);

  TH1F* K0Pass_LamMass_2 = GetHisto("Resultsgrid_cLamK0_ss_NEW2_NoMisID.root","femtolist","K0Lam","LambdaMass_K01_Pass");
  DrawNorm(K0Pass_LamMass_2,2);
/*
  TH1F* K0Pass_LamMass_3 = GetHisto("Resultsgrid_cLamK0_ss_OLD.root","femtolist","K0Lam","LambdaMass_K01_Pass");
  DrawNorm(K0Pass_LamMass_3,3);
*/
  TH1F* K0Pass_LamMass_4 = GetHisto("Resultsgrid_cLamK0_ss_NEW2.root","femtolist","K0Lam","LambdaMass_K01_Pass");
  DrawNorm(K0Pass_LamMass_4,4);
/*
  TH1F* K0Pass_LamMass_5 = GetHisto("Resultsgrid_cLamK0_Bp1_NEW.root","femtolist","K0Lam","LambdaMass_K01_Pass");
  DrawNorm(K0Pass_LamMass_5,5);

  TH1F* K0Pass_LamMass_6 = GetHisto("Resultsgrid_cLamK0_ss_NEW2c.root","femtolist","K0Lam","LambdaMass_K01_Pass");
  DrawNorm(K0Pass_LamMass_6,6);
*/

  leg1 = new TLegend(0.60,0.40,0.89,0.80);
  leg1->SetFillColor(0);
  leg1->AddEntry(K0Pass_LamMass_1, "No MisID & TPC only #pi daughter", "lp");
  leg1->AddEntry(K0Pass_LamMass_2, "No MisID & TPC/TOF #pi", "lp");
  //leg1->AddEntry(K0Pass_LamMass_3, "MisID & TPC only #pi", "lp");
  leg1->AddEntry(K0Pass_LamMass_4, "MisID & TPC/TOF #pi", "lp");
  //leg1->AddEntry(K0Pass_LamMass_5, "Complex MisID & TPC only #pi", "lp");
  //leg1->AddEntry(K0Pass_LamMass_6, "Complex MisID & TPC/TOF #pi", "lp");
  leg1->Draw();

  //-------------------------------------------------------
  c1->cd(2);
  int FirstZoom = K0Pass_LamMass_1->FindBin(1.09);
  int LastZoom = K0Pass_LamMass_1->FindBin(1.16);
  K0Pass_LamMass_1->SetTitle("");
  K0Pass_LamMass_1->GetXaxis()->SetRange(FirstZoom,LastZoom);
  K0Pass_LamMass_1->GetYaxis()->SetRangeUser(0.,K0Pass_LamMass_1->GetMaximum());
  DrawNorm(K0Pass_LamMass_1,1);
  DrawNorm(K0Pass_LamMass_2,2);
  //DrawNorm(K0Pass_LamMass_3,3);
  DrawNorm(K0Pass_LamMass_4,4);
  //DrawNorm(K0Pass_LamMass_5,5);
  //DrawNorm(K0Pass_LamMass_6,6);

/*
  //-------------------------------------------------------
  c1->cd(3);
  DrawNorm(K0Pass_LamMass_1,1);
  DrawNorm(K0Pass_LamMass_2,2);

  leg3 = new TLegend(0.60,0.12,0.89,0.32);
  leg3->SetFillColor(0);
  leg3->AddEntry(K0Pass_LamMass_1, "No MisID & TPC only #pi daughter", "lp");
  leg3->AddEntry(K0Pass_LamMass_2, "No MisID & TPC/TOF #pi", "lp");
  leg3->Draw();
  //-------------------------------------------------------
  c1->cd(4);
  K0Pass_LamMass_3->GetXaxis()->SetRange(FirstZoom,LastZoom);
  K0Pass_LamMass_3->GetYaxis()->SetRangeUser(0.,K0Pass_LamMass_3->GetMaximum());
  DrawNorm(K0Pass_LamMass_3,3);
  DrawNorm(K0Pass_LamMass_4,4);

  leg4 = new TLegend(0.60,0.12,0.89,0.32);
  leg4->SetFillColor(0);
  leg4->AddEntry(K0Pass_LamMass_3, "MisID & TPC only #pi", "lp");
  leg4->AddEntry(K0Pass_LamMass_4, "MisID & TPC/TOF #pi", "lp");
  leg4->Draw();
  //-------------------------------------------------------
  c1->cd(5);
  K0Pass_LamMass_5->GetXaxis()->SetRange(FirstZoom,LastZoom);
  K0Pass_LamMass_5->GetYaxis()->SetRangeUser(0.,K0Pass_LamMass_5->GetMaximum());
  DrawNorm(K0Pass_LamMass_5,5);
  DrawNorm(K0Pass_LamMass_6,6);

  leg5 = new TLegend(0.60,0.12,0.89,0.32);
  leg5->SetFillColor(0);
  leg5->AddEntry(K0Pass_LamMass_5, "Complex MisID & TPC only #pi", "lp");
  leg5->AddEntry(K0Pass_LamMass_6, "Complex MisID & TPC/TOF #pi", "lp");
  leg5->Draw();
*/
  c1->SaveAs("compCuts2.pdf");

}
