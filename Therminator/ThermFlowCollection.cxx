/* ThermFlowCollection.cxx */

#include "ThermFlowCollection.h"

#ifdef __ROOT__
ClassImp(ThermFlowCollection)
#endif


//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________


//________________________________________________________________________________________________________________
ThermFlowCollection::ThermFlowCollection(int aNpTBins, double apTBinSize) :
  fAnUnIdent(nullptr),
  fAnKch(nullptr),
  fAnK0s(nullptr),
  fAnLam(nullptr),

  fSaveFileName("FlowGraphs")
{
  fAnUnIdent = new ThermFlowAnalysis(aNpTBins, apTBinSize, 0);
  fAnKch = new ThermFlowAnalysis(aNpTBins, apTBinSize, 321);
  fAnK0s = new ThermFlowAnalysis(aNpTBins, apTBinSize, 311);
  fAnLam = new ThermFlowAnalysis(aNpTBins, apTBinSize, 3122);
}



//________________________________________________________________________________________________________________
ThermFlowCollection::~ThermFlowCollection()
{
/*no-op*/
}


//________________________________________________________________________________________________________________
void ThermFlowCollection::BuildVnEPIngredients(ThermEvent &aEvent)
{
  fAnUnIdent->BuildVnEPIngredients(aEvent);
  fAnKch->BuildVnEPIngredients(aEvent);
  fAnK0s->BuildVnEPIngredients(aEvent);
  fAnLam->BuildVnEPIngredients(aEvent);
}


//________________________________________________________________________________________________________________
void ThermFlowCollection::DrawAllFlowHarmonics()
{
  TObjArray* tGraphsUnIdent = fAnUnIdent->GetVnGraphs();
  TGraphErrors* tGrUnIdent_v2 = (TGraphErrors*)tGraphsUnIdent->At(0);
  TGraphErrors* tGrUnIdent_v3 = (TGraphErrors*)tGraphsUnIdent->At(1);

  TObjArray* tGraphsKch = fAnKch->GetVnGraphs();
  TGraphErrors* tGrKch_v2 = (TGraphErrors*)tGraphsKch->At(0);
  TGraphErrors* tGrKch_v3 = (TGraphErrors*)tGraphsKch->At(1);

  TObjArray* tGraphsK0s = fAnK0s->GetVnGraphs();
  TGraphErrors* tGrK0s_v2 = (TGraphErrors*)tGraphsK0s->At(0);
  TGraphErrors* tGrK0s_v3 = (TGraphErrors*)tGraphsK0s->At(1);

  TObjArray* tGraphsLam = fAnLam->GetVnGraphs();
  TGraphErrors* tGrLam_v2 = (TGraphErrors*)tGraphsLam->At(0);
  TGraphErrors* tGrLam_v3 = (TGraphErrors*)tGraphsLam->At(1);

  //--------------------------------------
  TCanvas* tCanAllVn = new TCanvas("tCanAllVn", "tCanAllVn");

  tGrUnIdent_v2->Draw("AP");
  tGrUnIdent_v3->Draw("Psame");

  tGrKch_v2->Draw("Psame");
  tGrKch_v3->Draw("Psame");

  tGrK0s_v2->Draw("Psame");
  tGrK0s_v3->Draw("Psame");

  tGrLam_v2->Draw("Psame");
  tGrLam_v3->Draw("Psame");

  tCanAllVn->SaveAs(TString::Format("%s.pdf", fSaveFileName.Data()));
}

//________________________________________________________________________________________________________________
void ThermFlowCollection::SaveAllGraphs()
{
  TFile* tSaveFile = new TFile(TString::Format("%s.root", fSaveFileName.Data()), "RECREATE");

  fAnUnIdent->SaveGraphs(tSaveFile);
  fAnKch->SaveGraphs(tSaveFile);
  fAnK0s->SaveGraphs(tSaveFile);
  fAnLam->SaveGraphs(tSaveFile);

  tSaveFile->Close();
}


//________________________________________________________________________________________________________________
void ThermFlowCollection::Finalize()
{
  fAnUnIdent->Finalize();
  fAnKch->Finalize();
  fAnK0s->Finalize();
  fAnLam->Finalize();

  SaveAllGraphs();
  DrawAllFlowHarmonics();
}

