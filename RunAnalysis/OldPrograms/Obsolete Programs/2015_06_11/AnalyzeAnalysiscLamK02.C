#include "buildAllcLamK02.cxx"

void AnalyzeAnalysiscLamK02()
{
  vector<TString> VectorOfFileNames;
    VectorOfFileNames.push_back("Resultsgrid_cLamK0_CentBins_Bp1NEW.root");
    VectorOfFileNames.push_back("Resultsgrid_cLamK0_CentBins_Bp2NEW.root");
    VectorOfFileNames.push_back("Resultsgrid_cLamK0_CentBins_Bm1NEW.root");
    VectorOfFileNames.push_back("Resultsgrid_cLamK0_CentBins_Bm2NEW.root");
    VectorOfFileNames.push_back("Resultsgrid_cLamK0_CentBins_Bm3NEW.root");


  buildAllcLamK02 *cLamK0Analysis2 = new buildAllcLamK02(VectorOfFileNames,"LamK0_0010","ALamK0_0010");
  cLamK0Analysis2->BuildCFCollections();
  cLamK0Analysis2->BuildAvgSepCollections();
  cLamK0Analysis2->BuildPurityCollections();


  TCanvas* c1 = new TCanvas("c1","Plotting Canvas1");
  cLamK0Analysis2->DrawFinalCFs(c1);


  TCanvas* c2 = new TCanvas("c2","Plotting Canvas2");
  TCanvas* c3 = new TCanvas("c3","Plotting Canvas3");
  cLamK0Analysis2->DrawFinalAvgSepCFs(c2,c3);

  TCanvas* c4 = new TCanvas("c4","Plotting Canvas4");
  cLamK0Analysis2->DrawFinalPurity(c4);



//-------------Second go round to break things

  vector<TString> VectorOfFileNames2;
    VectorOfFileNames2.push_back("Resultsgrid_cLamK0_CentBins_Bp1NEW.root");
    VectorOfFileNames2.push_back("Resultsgrid_cLamK0_CentBins_Bp2NEW.root");
    VectorOfFileNames2.push_back("Resultsgrid_cLamK0_CentBins_Bm1NEW.root");
    VectorOfFileNames2.push_back("Resultsgrid_cLamK0_CentBins_Bm2NEW.root");
    VectorOfFileNames2.push_back("Resultsgrid_cLamK0_CentBins_Bm3NEW.root");


  buildAllcLamK02 *cLamK0Analysis22 = new buildAllcLamK02(VectorOfFileNames2,"LamK0","ALamK0");
/*
  cLamK0Analysis22->BuildCFCollections();
  cLamK0Analysis22->BuildAvgSepCollections();
  cLamK0Analysis22->BuildPurityCollections();


  TCanvas* c12 = new TCanvas("c12","Plotting Canvas12");
  cLamK0Analysis22->DrawFinalCFs(c12);


  TCanvas* c22 = new TCanvas("c22","Plotting Canvas22");
  TCanvas* c32 = new TCanvas("c32","Plotting Canvas32");
  cLamK0Analysis22->DrawFinalAvgSepCFs(c22,c32);

  TCanvas* c42 = new TCanvas("c42","Plotting Canvas42");
  cLamK0Analysis22->DrawFinalPurity(c42);
*/
}
