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
void compCuts3()
{
  TCanvas* c1 = new TCanvas("c1","Plotting Canvas");
  //c1->Divide(1,2);
  gStyle->SetOptStat(0);

  //-------------------------------------------------------
  c1->cd(1);
  TH1F* LamPass_K0Mass_1 = GetHisto("Resultsgrid_cLamK0_ss_OLD_NoMisID.root","femtolist","K0Lam","K0ShortMass_Lam_Pass");
  int FirstNonZero = FindFirstNonZeroBin(LamPass_K0Mass_1);
  int LastNonZero = FindLastNonZeroBin(LamPass_K0Mass_1);
  LamPass_K0Mass_1->GetXaxis()->SetRange(FirstNonZero,LastNonZero);
  DrawNorm(LamPass_K0Mass_1,1);

  TH1F* LamPass_K0Mass_2 = GetHisto("Resultsgrid_cLamK0_ss_NEW2_NoMisID.root","femtolist","K0Lam","K0ShortMass_Lam_Pass");
  DrawNorm(LamPass_K0Mass_2,2);
/*
  TH1F* LamPass_K0Mass_3 = GetHisto("Resultsgrid_cLamK0_ss_OLD.root","femtolist","K0Lam","K0ShortMass_Lam_Pass");
  DrawNorm(LamPass_K0Mass_3,3);
*/
  TH1F* LamPass_K0Mass_4 = GetHisto("Resultsgrid_cLamK0_ss_NEW2.root","femtolist","K0Lam","K0ShortMass_Lam_Pass");
  DrawNorm(LamPass_K0Mass_4,4);
/*
  TH1F* LamPass_K0Mass_5 = GetHisto("Resultsgrid_cLamK0_Bp1_NEW.root","femtolist","K0Lam","K0ShortMass_Lam_Pass");
  DrawNorm(LamPass_K0Mass_5,5);

  TH1F* LamPass_K0Mass_6 = GetHisto("Resultsgrid_cLamK0_ss_NEW2c.root","femtolist","K0Lam","K0ShortMass_Lam_Pass");
  DrawNorm(LamPass_K0Mass_6,6);
*/

  leg1 = new TLegend(0.35,0.20,0.65,0.50);
  leg1->SetFillColor(0);
  leg1->AddEntry(LamPass_K0Mass_1, "No MisID & TPC only #pi daughter", "lp");
  leg1->AddEntry(LamPass_K0Mass_2, "No MisID & TPC/TOF #pi", "lp");
  //leg1->AddEntry(LamPass_K0Mass_3, "MisID & TPC only #pi", "lp");
  leg1->AddEntry(LamPass_K0Mass_4, "MisID & TPC/TOF #pi", "lp");
  //leg1->AddEntry(LamPass_K0Mass_5, "Complex MisID & TPC only #pi", "lp");
  //leg1->AddEntry(LamPass_K0Mass_6, "Complex MisID & TPC/TOF #pi", "lp");
  leg1->Draw();

/*
  //-------------------------------------------------------
  c1->cd(2);
  int FirstZoom = LamPass_K0Mass_1->FindBin(0.45);
  int LastZoom = LamPass_K0Mass_1->FindBin(0.55);
  LamPass_K0Mass_1->GetXaxis()->SetRange(FirstZoom,LastZoom);
  LamPass_K0Mass_1->GetYaxis()->SetRangeUser(0.,LamPass_K0Mass_1->GetMaximum());
  DrawNorm(LamPass_K0Mass_1,1);
  DrawNorm(LamPass_K0Mass_2,2);
  //DrawNorm(LamPass_K0Mass_3,3);
  DrawNorm(LamPass_K0Mass_4,4);
  //DrawNorm(LamPass_K0Mass_5,5);
  //DrawNorm(LamPass_K0Mass_6,6);


  //-------------------------------------------------------
  c1->cd(3);
  DrawNorm(LamPass_K0Mass_1,1);
  DrawNorm(LamPass_K0Mass_2,2);

  leg3 = new TLegend(0.60,0.12,0.89,0.32);
  leg3->SetFillColor(0);
  leg3->AddEntry(LamPass_K0Mass_1, "No MisID & TPC only #pi daughter", "lp");
  leg3->AddEntry(LamPass_K0Mass_2, "No MisID & TPC/TOF #pi", "lp");
  leg3->Draw();
  //-------------------------------------------------------
  c1->cd(4);
  LamPass_K0Mass_3->GetXaxis()->SetRange(FirstZoom,LastZoom);
  LamPass_K0Mass_3->GetYaxis()->SetRangeUser(0.,LamPass_K0Mass_3->GetMaximum());
  DrawNorm(LamPass_K0Mass_3,3);
  DrawNorm(LamPass_K0Mass_4,4);

  leg4 = new TLegend(0.60,0.12,0.89,0.32);
  leg4->SetFillColor(0);
  leg4->AddEntry(LamPass_K0Mass_3, "MisID & TPC only #pi", "lp");
  leg4->AddEntry(LamPass_K0Mass_4, "MisID & TPC/TOF #pi", "lp");
  leg4->Draw();
  //-------------------------------------------------------
  c1->cd(5);
  LamPass_K0Mass_5->GetXaxis()->SetRange(FirstZoom,LastZoom);
  LamPass_K0Mass_5->GetYaxis()->SetRangeUser(0.,LamPass_K0Mass_5->GetMaximum());
  DrawNorm(LamPass_K0Mass_5,5);
  DrawNorm(LamPass_K0Mass_6,6);

  leg5 = new TLegend(0.60,0.12,0.89,0.32);
  leg5->SetFillColor(0);
  leg5->AddEntry(LamPass_K0Mass_5, "Complex MisID & TPC only #pi", "lp");
  leg5->AddEntry(LamPass_K0Mass_6, "Complex MisID & TPC/TOF #pi", "lp");
  leg5->Draw();
*/
  c1->SaveAs("compCuts3.pdf");

}
