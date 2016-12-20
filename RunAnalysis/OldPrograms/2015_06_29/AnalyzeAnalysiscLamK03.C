#include "buildAllcLamK03.cxx"

void AnalyzeAnalysiscLamK03()
{
  vector<TString> VectorOfFileNames;
    VectorOfFileNames.push_back("Resultsgrid_cLamK0_CentBins_Bp1NEWAS.root");
    VectorOfFileNames.push_back("Resultsgrid_cLamK0_CentBins_Bp2NEWAS.root");
    VectorOfFileNames.push_back("Resultsgrid_cLamK0_CentBins_Bm1NEWAS.root");
    VectorOfFileNames.push_back("Resultsgrid_cLamK0_CentBins_Bm2NEWAS.root");
    VectorOfFileNames.push_back("Resultsgrid_cLamK0_CentBins_Bm3NEWAS.root");

  TString aSaveDirectory = "~/Analysis/K0Lam/NEW2/cLamK0/";  //note:  This directory must already be created

  //-------------------------------------------------------------------------------------------
  buildAllcLamK03 *cLamK0Analysis3 = new buildAllcLamK03(VectorOfFileNames,"LamK0_0010","ALamK0_0010");
  cLamK0Analysis3->BuildCFCollections();
  cLamK0Analysis3->BuildAvgSepCollections();
  cLamK0Analysis3->BuildPurityCollections();
  //cLamK0Analysis3->BuildSepCollections(5);

  TCanvas* canvasCfs = new TCanvas("canvasCfs","KStar Cfs");
  cLamK0Analysis3->DrawFinalCFs(canvasCfs);
  //canvasCfs->SaveAs(aSaveDirectory+"TEST.pdf");
  //canvasCfs->SaveAs("~/Analysis/K0Lam/NEWASmc/2015_04_08/cLamK0_0010_CFs.pdf");

  TCanvas* canvasAvgSepLamK0 = new TCanvas("canvasAvgSepLamK0","AvgSep LamK0");
  TCanvas* canvasAvgSepALamK0 = new TCanvas("canvasAvgSepALamK0","AvgSep ALamK0");
  cLamK0Analysis3->DrawFinalAvgSepCFs(canvasAvgSepLamK0,canvasAvgSepALamK0);
  //canvasAvgSepLamK0->SaveAs("~/Analysis/K0Lam/NEWASmc/2015_04_08/LamK0_0010_AvgSepCFs.pdf");
  //canvasAvgSepALamK0->SaveAs("~/Analysis/K0Lam/NEWASmc/2015_04_08/ALamK0_0010_AvgSepCFs.pdf");

  TCanvas* canvasPurity = new TCanvas("canvasPurity","Purity");
  cLamK0Analysis3->DrawFinalPurity(canvasPurity);
  //canvasPurity->SaveAs("~/Analysis/K0Lam/NEWASmc/2015_04_08/cLamK0_0010_Purity.pdf");

/*
  TCanvas* canvasLamK0LikeSigns = new TCanvas("canvasLamK0LikeSigns","SepCfs LamK0 Like Signs");
  TCanvas* canvasLamK0UnlikeSigns = new TCanvas("canvasLamK0UnlikeSigns","SepCfs LamK0 Unlike Signs");
  TCanvas* canvasALamK0LikeSigns = new TCanvas("canvasALamK0LikeSigns","SepCfs ALamK0 Like Signs");
  TCanvas* canvasALamK0UnlikeSigns = new TCanvas("canvasALamK0UnlikeSigns","SepCfs ALamK0 Unlike Signs");
  cLamK0Analysis3->DrawFinalSepCFs(canvasLamK0LikeSigns,canvasLamK0UnlikeSigns,canvasALamK0LikeSigns,canvasALamK0UnlikeSigns);
  //cLamK0Analysis3->DrawAvgOfFinalSepCFs(canvasLamK0LikeSigns,canvasLamK0UnlikeSigns,canvasALamK0LikeSigns,canvasALamK0UnlikeSigns);
*/

}
