#include "UsefulMacros.C"
#include <vector>

  typedef vector<TH1F*> aVec;

//_________________________________________________________________________________________
TH1F* GetHistoClone(TString FileName, TString ArrayName, TString HistoName, TString CloneName)
{
  TFile f1(FileName);
  TList *femtolist = (TList*)f1.Get("femtolist");
  if(ArrayName)
    {
      TObjArray *array = (TObjArray*)femtolist->FindObject(ArrayName);
      array->SetName("LamK0Bp1");
      cout << "Array name: " << array->GetName() << endl;
      TH1F *ReturnHisto = (TH1F*)array->FindObject(HistoName);
    }
  else
    {
      TH1F *ReturnHisto = (TH1F*)femtolist->FindObject(HistoName);
    }
  TH1F *ReturnHistoCopy = (TH1F*)ReturnHisto->Clone(CloneName);
  ReturnHistoCopy->SetDirectory(0);
  return ReturnHistoCopy;
}
//_________________________________________________________________________________________
void ReadOutArrayOfNames(TString *aArrayOfNames)
{
  for(int i=0; i<5; i++)
  {
    cout << aArrayOfNames[i] << endl;
  }
}

//_________________________________________________________________________________________
void DrawInCanvas(TCanvas *aCanvas, TH1F* aHistogram)
{
  aCanvas->cd();
  gStyle->SetOptStat(0);
  aCanvas->Divide(2,2);
  aCanvas->cd(1);
  aHistogram->Draw();
}




//_________________________________________________________________________________________
void test()
{

  const Int_t MinNormBin = 60;
  const Int_t MaxNormBin = 75;

  TString FileName = "Resultsgrid_cLamK0_CentBins_Bp1NEW.root";

  TH1F *Num_LamK0 = GetHisto(FileName,"LamK0", "NumKStarCF_LamK0");
  TH1F *Den_LamK0 = GetHisto(FileName,"LamK0", "DenKStarCF_LamK0");
  TH1F *Cf_LamK0 = buildCF("Cf_LamK0","Lam-K0",Num_LamK0,Den_LamK0,MinNormBin,MaxNormBin);
  Cf_LamK0->Draw();


  aVec aVectorOfHistograms;
  //vector<TH1F*> aVectorOfHistograms;

  aVectorOfHistograms.push_back(Num_LamK0);
  aVectorOfHistograms.push_back(Den_LamK0);
  aVectorOfHistograms.push_back(Cf_LamK0);
  aVectorOfHistograms[2]->SetLineColor(2);
  aVectorOfHistograms[2]->Draw("same");

  TH1F *hist1 = new TH1F("hist1", "My first histo", 100, 2, 200);
    hist1->Fill(50);
    hist1->Fill(120,2);
    hist1->Fill(130,2);
    //hist1->Draw();

  new TH1F("hist2", "My second histo", 100, 2, 200);
    hist2->Fill(10,3);
    hist2->Fill(70,2);
    hist2->Fill(110,3);
    //hist2->Draw();
    hist2->Sumw2();

  TH1F *hist2Clone = new TH1F();
  hist2Clone = (TH1F*)hist2->Clone("hist2Clone");
  //hist2Clone->Sumw2();

  TH1F *tmp1 = GetHistoClone(FileName,"LamK0","NumKStarCF_LamK0","Clone");
  //tmp1->Draw();

  cout << Num_LamK0->GetName() << endl;
  cout << tmp1->GetName() << endl;

  TList *myList = new TList();
  myList->Add(Num_LamK0);
  myList->Add(tmp1);
  myList->ls();

  TH1F* blah = new TH1F();

/*
  TString str1 = "String1";
  cout << str1 << endl;
  TString str2 = "String2";
  bool tester = kTRUE;
  if(tester){ str1+="_"; str1+=str2;}
  cout << str1 << endl;
*/

  TString ArrayOfNames[5];
    ArrayOfNames[0] = "Name0";
    ArrayOfNames[1] = "Name1";
    ArrayOfNames[2] = "Name2";
    ArrayOfNames[3] = "Name3";
    ArrayOfNames[4] = "Name4";

/*
  for(int i=0; i<5; i++)
  {
    cout << ArrayOfNames[i] << endl;
  }
*/

  ReadOutArrayOfNames(ArrayOfNames);

  int ArraySize = sizeof(ArrayOfNames)/sizeof(TString);
  cout << "Array Size: " << ArraySize << endl;

  vector<TString> VecOfNames;
    VecOfNames.push_back("vName0");
    VecOfNames.push_back("vName1");
    VecOfNames.push_back("vName2");
    VecOfNames.push_back("vName3");
    VecOfNames.push_back("vName4");

  cout << VecOfNames.size() << endl;
  //cout << VecOfNames[0] << endl;
  //cout << VecOfNames[1] << endl;
  //cout << VecOfNames[2] << endl;
  //cout << VecOfNames[3] << endl;
  //cout << VecOfNames[4] << endl;

  TString Name0, Name1, Name2, Name3, Name4;

  for(int i=0; i<VecOfNames.size(); i++)
  {
    if(VecOfNames[i].Contains("vName0")) {Name0 = VecOfNames[i];}
    else if(VecOfNames[i].Contains("vName1")) {Name1 = VecOfNames[i];}
    else if(VecOfNames[i].Contains("vName2")) {Name2 = VecOfNames[i];}
    else if(VecOfNames[i].Contains("vName3")) {Name3 = VecOfNames[i];}
    else if(VecOfNames[i].Contains("vName4")) {Name4 = VecOfNames[i];}
  }

  cout << Name0 << endl;
  cout << Name1 << endl;
  cout << Name2 << endl;
  cout << Name3 << endl;
  cout << Name4 << endl;

  TString first = "first";
  TString second = "_"+first;
  cout << second << endl;


  TCanvas* c1 = new TCanvas("c1","Plotting Canvas1");
  TCanvas* c2 = new TCanvas("c2","Plotting Canvas2");

/*
  c1->cd();
  hist1->Draw();
  c2->cd();
  hist2->Draw();
*/

/*
  DrawInCanvas(c1,hist2);
  DrawInCanvas(c2,hist2Clone);
*/


/*
  const char null[80] = {'\0'};

  TString NullString1 = "";
  TString NullString2 = null;

  cout << "Is NullString1 really empty?" << endl;
  if(!NullString1.IsNull()){cout << "NO!" << endl;}
  if(NullString1.IsNull()){cout << "YES!!!" << endl;}

  cout << "Is NullString2 really empty?" << endl;
  if(!NullString2.IsNull()){cout << "NO!" << endl;}
  if(NullString2.IsNull()){cout << "YES!!!" << endl;}
 */


}
