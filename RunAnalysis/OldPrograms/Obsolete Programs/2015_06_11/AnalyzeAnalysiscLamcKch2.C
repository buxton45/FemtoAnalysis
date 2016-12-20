#include "buildAllcLamcKch2.cxx"

void AnalyzeAnalysiscLamcKch2()
{
  vector<TString> VectorOfFileNames;
    VectorOfFileNames.push_back("Resultsgrid_cLamcKch_CentBins_Bp1NEW.root");
    VectorOfFileNames.push_back("Resultsgrid_cLamcKch_CentBins_Bp2NEW.root");
    VectorOfFileNames.push_back("Resultsgrid_cLamcKch_CentBins_Bm1NEW.root");
    VectorOfFileNames.push_back("Resultsgrid_cLamcKch_CentBins_Bm2NEW.root");
    VectorOfFileNames.push_back("Resultsgrid_cLamcKch_CentBins_Bm3NEW.root");

  buildAllcLamcKch2 *cLamcKchAnalysis2 = new buildAllcLamcKch2(VectorOfFileNames,"LamKchP","LamKchM","ALamKchP","ALamKchM");



  cLamcKchAnalysis2->BuildCFCollections();
  //cLamcKchAnalysis2->BuildAvgSepCollections();
  //cLamcKchAnalysis2->BuildPurityCollections();
/*
  TCanvas* c1 = new TCanvas("c1","Plotting Canvas1");
  cLamcKchAnalysis2->DrawFinalCFs(c1);


  TCanvas* c2 = new TCanvas("c2","Plotting Canvas2");
  TCanvas* c3 = new TCanvas("c3","Plotting Canvas3");
  TCanvas* c4 = new TCanvas("c4","Plotting Canvas4");
  TCanvas* c5 = new TCanvas("c5","Plotting Canvas5");
  cLamcKchAnalysis2->DrawFinalAvgSepCFs(c2,c3,c4,c5);


  TCanvas* c6 = new TCanvas("c6","Plotting Canvas6");
  cLamcKchAnalysis2->DrawFinalPurity(c6);
*/
}
