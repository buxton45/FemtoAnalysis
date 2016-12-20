#include "UsefulMacros.C"
#include "TMath.h"


void GetMisID()
{
  TCanvas* aCan1 = new TCanvas("aCan1","aCan1");
  gPad->SetLogy();

  TCanvas* aCan2 = new TCanvas("aCan2","aCan2");
  gPad->SetLogy();

  TString aPair;
    aPair = "LamKchP";
    //aPair = "LamKchM";
    //aPair = "ALamKchP";
    //aPair = "ALamKchM";

  TString aDirectoryName = aPair + "_0010";

  TString aHistoName1 = "PId_Lam_Pass";
  //TString aHistoName = "PId_ALam_Pass";

  TString aHistoName2 = "PId_KchP_Pass";
  //TString aHistoName2 = "PId_KchM_Pass";

  TString DataFile1 = "~/Analysis/K0Lam/Results_cLamcKch_AsRcMC_20160224/Results_cLamcKch_AsRcMC_20160224_Bp1.root";
  TH1F* tHisto1 = GetHisto(DataFile1,aDirectoryName,aHistoName1);
  TH1F* tHisto2 = GetHisto(DataFile1,aDirectoryName,aHistoName2);

//-----------------------------------
  aCan1->cd();
  tHisto1->Draw();

  cout << "Particle1" << endl;
  cout << "->GetEntries() = " << tHisto1->GetEntries() << endl << endl;

  for(int i=1; i<=tHisto1->GetNbinsX(); i++)
  {
    if(tHisto1->GetBinContent(i) > 0)
    {
      cout << "\t\t Bin = " << i << endl;
      cout << "\t\t PDG Pid = " << (i-1) << endl;
      cout << "\t\t Counts = " << tHisto1->GetBinContent(i) << endl << endl;
    }
  }

//-----------------------------------
  aCan2->cd();
  tHisto2->Draw();

  cout << "Particle2" << endl;
  cout << "->GetEntries() = " << tHisto2->GetEntries() << endl << endl;

  for(int i=1; i<=tHisto2->GetNbinsX(); i++)
  {
    if(tHisto2->GetBinContent(i) > 0)
    {
      cout << "\t\t Bin = " << i << endl;
      cout << "\t\t PDG Pid = " << (i-1) << endl;
      cout << "\t\t Counts = " << tHisto2->GetBinContent(i) << endl << endl;
    }
  }


}
