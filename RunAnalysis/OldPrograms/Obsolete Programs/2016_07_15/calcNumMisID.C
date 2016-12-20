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
void calcNumMisID()
{
  TH1F *K0Short1 = GetHisto("Resultsgrid_cLamK0_Bp1_NEW.root","femtolist","K0Lam","LambdaMass_K01_Pass");
    Double_t NumK0Short1 = K0Short1->Integral(0,K0Short1->GetNbinsX()+1);
  TH1F *MisIDK0Short1 = GetHisto("Resultsgrid_cLamK0_Bp1_NEW.root","femtolist","K0Lam","MisIDK0Short1");
    Double_t NumMisIDK0Short1 = MisIDK0Short1->Integral(0,MisIDK0Short1->GetNbinsX()+1);
  cout << "NumK0Short1 = " << NumK0Short1 << endl;
  cout << "NumMisIDK0Short1 = " << NumMisIDK0Short1 << endl;
  cout << "NumMisIDK0Short1 / NumK0Short1 = " << NumMisIDK0Short1/NumK0Short1 << endl << endl;

  TH1F *Lambdas = GetHisto("Resultsgrid_cLamK0_Bp1_NEW.root","femtolist","K0Lam","LambdaMass_Lam_Pass");
    Double_t NumLambdas = Lambdas->Integral(0,Lambdas->GetNbinsX()+1);
  TH1F *MisIDLambdas = GetHisto("Resultsgrid_cLamK0_Bp1_NEW.root","femtolist","K0Lam","MisIDLambdas");
    Double_t NumMisIDLambdas = MisIDLambdas->Integral(0,MisIDLambdas->GetNbinsX()+1);
  cout << "NumLambdas = " << NumLambdas << endl;
  cout << "NumMisIDLambdas = " << NumMisIDLambdas << endl;
  cout << "NumMisIDLambdas / NumLambdas = " << NumMisIDLambdas/NumLambdas << endl << endl;

  TH1F *K0Short2 = GetHisto("Resultsgrid_cLamK0_Bp1_NEW.root","femtolist","K0ALam","LambdaMass_K02_Pass");
    Double_t NumK0Short2 = K0Short2->Integral(0,K0Short2->GetNbinsX()+1);
  TH1F *MisIDK0Short2 = GetHisto("Resultsgrid_cLamK0_Bp1_NEW.root","femtolist","K0ALam","MisIDK0Short2");
    Double_t NumMisIDK0Short2 = MisIDK0Short2->Integral(0,MisIDK0Short2->GetNbinsX()+1);
  cout << "NumK0Short2 = " << NumK0Short2 << endl;
  cout << "NumMisIDK0Short2 = " << NumMisIDK0Short2 << endl;
  cout << "NumMisIDK0Short2 / NumK0Short2 = " << NumMisIDK0Short2/NumK0Short2 << endl << endl;

  TH1F *AntiLambdas = GetHisto("Resultsgrid_cLamK0_Bp1_NEW.root","femtolist","K0ALam","LambdaMass_ALam_Pass");
    Double_t NumAntiLambdas = AntiLambdas->Integral(0,AntiLambdas->GetNbinsX()+1);
  TH1F *MisIDAntiLambdas = GetHisto("Resultsgrid_cLamK0_Bp1_NEW.root","femtolist","K0ALam","MisIDAntiLambdas");
    Double_t NumMisIDAntiLambdas = MisIDAntiLambdas->Integral(0,MisIDAntiLambdas->GetNbinsX()+1);
  cout << "NumAntiLambdas = " << NumAntiLambdas << endl;
  cout << "NumMisIDAntiLambdas = " << NumMisIDAntiLambdas << endl;
  cout << "NumMisIDAntiLambdas / NumAntiLambdas = " << NumMisIDAntiLambdas/NumAntiLambdas << endl << endl;

}
