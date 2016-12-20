

#include "TH2F.h"

using namespace std;

//_________________________________________________________________________________________
TH2F* GetHistoClone(TString FileName, TString ArrayName, TString HistoName, TString CloneName)
{
  TFile f1(FileName);
  TList *femtolist = (TList*)f1.Get("femtolist");
  if(ArrayName)
    {
      TObjArray *array = (TObjArray*)femtolist->FindObject(ArrayName);
      array->SetName("LamK0Bp1");
      cout << "Array name: " << array->GetName() << endl;
      TH2F *ReturnHisto = (TH2F*)array->FindObject(HistoName);
    }
  else
    {
      TH2F *ReturnHisto = (TH2F*)femtolist->FindObject(HistoName);
    }
  TH2F *ReturnHistoCopy = (TH2F*)ReturnHisto->Clone(CloneName);
  ReturnHistoCopy->SetDirectory(0);
  return ReturnHistoCopy;
}





//_________________________________________________________________________________________
void test2()
{

  const Int_t MinNormBin = 60;
  const Int_t MaxNormBin = 75;

  TString FileName = "Resultsgrid_cLamcKch_CentBins_Bp1NEW2.root";

  TH2F *Num_LamKchP = GetHistoClone(FileName,"LamKchP", "NumTrackPosSepCFs_LamKchP", "Num_LamKchP");
  TH2F *Den_LamKchP = GetHistoClone(FileName,"LamKchP", "DenTrackPosSepCFs_LamKchP", "Den_LamKchP");

  cout << Num_LamKchP->GetNbinsX() << endl;
  cout << Num_LamKchP->GetNbinsY() << endl;
  cout << Num_LamKchP->GetMaximum() << endl;

  //Num_LamKchP->Draw("lego");

  TH1D *Num_LamKchP_bin1 = Num_LamKchP->ProjectionY("Num_LamKchP_bin1",1,1);
  cout << Num_LamKchP_bin1->GetMaximum() << endl;
  //Num_LamKchP_bin1->Draw();

  TH1F *Num_LamKchP_bin1Clone = Num_LamKchP_bin1->Clone();
  cout << Num_LamKchP_bin1Clone->GetMaximum() << endl;
  //Num_LamKchP_bin1Clone->Draw();

  TString aName = "aName";
  int aInt = 5;
  aName+=aInt;
  cout << aName << endl;

  TH1F *TestHisto = Num_LamKchP_bin1Clone->Clone();
  TestHisto->SetBit(TH1::kIsAverage);
  Num_LamKchP_bin1Clone->SetBit(TH1::kIsAverage);
  TestHisto->Add(Num_LamKchP_bin1Clone);
  TestHisto->SetDirectory(0);
  TestHisto->Draw();

}
