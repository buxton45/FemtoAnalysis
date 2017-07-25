#include "TTree.h"
#include "TNtuple.h"

void FillKStarNtuple(float aKStarMin, float aKStarMax, TNtuple* aNtuple, float aKStarMag, float aKStarOut, float aKStarSide, float aKStarLong)
{
  if(aKStarMag > aKStarMin && aKStarMag < aKStarMax) aNtuple->Fill(aKStarMag,aKStarOut,aKStarSide,aKStarLong);
}



void TTreeTest()
{

  TString tFileName = "~/Analysis/K0Lam/Results_cXicKch_20160414/Results_cXicKch_20160414_Bp2.root";
  TFile tFile(tFileName);
  TList *tFemtoList = (TList*)tFile.Get("femtolist");
  TObjArray *tArray = (TObjArray*)tFemtoList->FindObject("XiKchP_0010")->Clone();
    tArray->SetOwner();
  TNtuple *tNtuple = (TNtuple*)tArray->FindObject("PairKStarKStarCf_XiKchP");

  float tKStarMag, tKStarOut, tKStarSide, tKStarLong;

  tNtuple->SetBranchAddress("KStarMag", &tKStarMag);
  tNtuple->SetBranchAddress("KStarOut", &tKStarOut);
  tNtuple->SetBranchAddress("KStarSide", &tKStarSide);
  tNtuple->SetBranchAddress("KStarLong", &tKStarLong);

  float tKStarMin = 0.10;
  float tKStarMax = 0.15;

  TFile *newFile = new TFile("small.root","recreate");


  char tArgument [40];
  sprintf(tArgument, "KStarMag>%lf && KStarMag<%lf",tKStarMin,tKStarMax);
  TNtuple* tKStarNtuple = tNtuple->CopyTree(tArgument);

//  TNtuple* tKStarNtuple = tNtuple->CopyTree("KStarMag>0.10 && KStarMag<0.15");

/*
  TNtuple* tKStarNtuple = new TNtuple("tKStarNtuple", "tKStarNtuple", "KMag:KOut:KSide:KLong");
  tKStarNtuple->SetDirectory(0);

cout << "tNtuple->GetEntries() = " << tNtuple->GetEntries() << endl;

  for(int i=0; i<tNtuple->GetEntries(); i++)
  {
cout << "i = " << i << endl;
    tNtuple->GetEntry(i);
    FillKStarNtuple(tKStarMin, tKStarMax, tKStarNtuple, tKStarMag, tKStarOut, tKStarSide, tKStarLong);
  }
*/
  cout << "tKStarNtuple->GetEntries() = " << tKStarNtuple->GetEntries() << endl;


}
