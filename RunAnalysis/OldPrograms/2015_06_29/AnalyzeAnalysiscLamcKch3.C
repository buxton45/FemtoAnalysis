#include "buildAllcLamcKch3.cxx"

void AnalyzeAnalysiscLamcKch3()
{
  vector<TString> VectorOfFileNames;

    VectorOfFileNames.push_back("Resultsgrid_cLamcKch_CentBins_Bp1NEW2.root");
    VectorOfFileNames.push_back("Resultsgrid_cLamcKch_CentBins_Bp2NEW2.root");
    VectorOfFileNames.push_back("Resultsgrid_cLamcKch_CentBins_Bm1NEW2.root");
    VectorOfFileNames.push_back("Resultsgrid_cLamcKch_CentBins_Bm2NEW2.root");
    VectorOfFileNames.push_back("Resultsgrid_cLamcKch_CentBins_Bm3NEW2.root");

  //---------------------------------------------------------------------------------------------------------------------------
/*
  buildAllcLamcKch3 *cLamcKchAnalysis3_mb = new buildAllcLamcKch3(VectorOfFileNames,"LamKchP","LamKchM","ALamKchP","ALamKchM");

  cLamcKchAnalysis3_mb->BuildCFCollections();
  cLamcKchAnalysis3_mb->BuildAvgSepCollections();
  cLamcKchAnalysis3_mb->BuildPurityCollections();

  TCanvas* c1_mb = new TCanvas("c1_mb","Plotting Canvas1_mb",1400,500);
  cLamcKchAnalysis3_mb->DrawFinalCFs(c1_mb);
  c1_mb->SaveAs("~/Analysis/K0Lam/NEWASmc/2015_04_08/cLamcKch_mb_CFs.pdf");


  TCanvas* c2_mb = new TCanvas("c2_mb","Plotting Canvas2_mb");
  TCanvas* c3_mb = new TCanvas("c3_mb","Plotting Canvas3_mb");
  TCanvas* c4_mb = new TCanvas("c4_mb","Plotting Canvas4_mb");
  TCanvas* c5_mb = new TCanvas("c5_mb","Plotting Canvas5_mb");
  cLamcKchAnalysis3_mb->DrawFinalAvgSepCFs(c2_mb,c3_mb,c4_mb,c5_mb);
  c2_mb->SaveAs("~/Analysis/K0Lam/NEWASmc/2015_04_08/LamKchP_mb_AvgSepCFs.pdf");
  c3_mb->SaveAs("~/Analysis/K0Lam/NEWASmc/2015_04_08/LamKchM_mb_AvgSepCFs.pdf");
  c4_mb->SaveAs("~/Analysis/K0Lam/NEWASmc/2015_04_08/ALamKchP_mb_AvgSepCFs.pdf");
  c5_mb->SaveAs("~/Analysis/K0Lam/NEWASmc/2015_04_08/ALamKchM_mb_AvgSepCFs.pdf");


  TCanvas* c6_mb = new TCanvas("c6_mb","Plotting Canvas6_mb");
  cLamcKchAnalysis3_mb->DrawFinalPurity(c6_mb);
  c6_mb->SaveAs("~/Analysis/K0Lam/NEWASmc/2015_04_08/cLamcKch_mb_Purity.pdf");
*/
  //---------------------------------------------------------------------------------------------------------------------------

  buildAllcLamcKch3 *cLamcKchAnalysis3_0010 = new buildAllcLamcKch3(VectorOfFileNames,"LamKchP_0010","LamKchM_0010","ALamKchP_0010","ALamKchM_0010");

  cLamcKchAnalysis3_0010->BuildCFCollections();
  cLamcKchAnalysis3_0010->BuildAvgSepCollections();
  cLamcKchAnalysis3_0010->BuildPurityCollections();
  cLamcKchAnalysis3_0010->BuildSepCollections(5);

  TCanvas* c1_0010 = new TCanvas("c1_0010","Plotting Canvas1_0010",1400,500);
  cLamcKchAnalysis3_0010->DrawFinalCFs(c1_0010);
  //c1_0010->SaveAs("~/Analysis/K0Lam/NEWASmc/2015_04_08/cLamcKch_0010_CFs.pdf");

  TCanvas* c2_0010 = new TCanvas("c2_0010","Plotting Canvas2_0010");
  TCanvas* c3_0010 = new TCanvas("c3_0010","Plotting Canvas3_0010");
  TCanvas* c4_0010 = new TCanvas("c4_0010","Plotting Canvas4_0010");
  TCanvas* c5_0010 = new TCanvas("c5_0010","Plotting Canvas5_0010");
  cLamcKchAnalysis3_0010->DrawFinalAvgSepCFs(c2_0010,c3_0010,c4_0010,c5_0010);
  //c2_0010->SaveAs("~/Analysis/K0Lam/NEW2/2015_05_10/LamKchP_0010_AvgSepCFs.pdf");
  //c3_0010->SaveAs("~/Analysis/K0Lam/NEW2/2015_05_10/LamKchM_0010_AvgSepCFs.pdf");
  //c4_0010->SaveAs("~/Analysis/K0Lam/NEW2/2015_05_10/ALamKchP_0010_AvgSepCFs.pdf");
  //c5_0010->SaveAs("~/Analysis/K0Lam/NEW2/2015_05_10/ALamKchM_0010_AvgSepCFs.pdf");

  TCanvas* c6_0010 = new TCanvas("c6_0010","Plotting Canvas6_0010");
  cLamcKchAnalysis3_0010->DrawFinalPurity(c6_0010);
  //c6_0010->SaveAs("~/Analysis/K0Lam/NEWASmc/2015_04_08/cLamcKch_0010_Purity.pdf");

  TCanvas* c7_0010 = new TCanvas("c7_0010","Plotting Canvas7_0010");
  TCanvas* c8_0010 = new TCanvas("c8_0010","Plotting Canvas8_0010");
  TCanvas* c9_0010 = new TCanvas("c9_0010","Plotting Canvas9_0010");
  TCanvas* c10_0010 = new TCanvas("c10_0010","Plotting Canvas10_0010");
  cLamcKchAnalysis3_0010->DrawFinalSepCFs(c7_0010,c8_0010,c9_0010,c10_0010);
  //cLamcKchAnalysis3_0010->DrawAvgOfFinalSepCFs(c7_0010,c8_0010,c9_0010,c10_0010);

  //c7_0010->SaveAs("~/Analysis/K0Lam/NEW2/2015_05_10/LamKchP_0010_SepCFs.pdf");
  //c8_0010->SaveAs("~/Analysis/K0Lam/NEW2/2015_05_10/LamKchM_0010_SepCFs.pdf");
  //c9_0010->SaveAs("~/Analysis/K0Lam/NEW2/2015_05_10/ALamKchP_0010_SepCFs.pdf");
  //c10_0010->SaveAs("~/Analysis/K0Lam/NEW2/2015_05_10/ALamKchM_0010_SepCFs.pdf");


/*
  TFile *myFile_0010 = new TFile("Resultsgrid_cLamcKch_CentBins_NEWASmc_0010.root", "RECREATE");
  cLamcKchAnalysis3_0010->SaveAll(myFile_0010);
  myFile_0010->Close();
*/

  //---------------------------------------------------------------------------------------------------------------------------
/*
  buildAllcLamcKch3 *cLamcKchAnalysis3_1030 = new buildAllcLamcKch3(VectorOfFileNames,"LamKchP_1030","LamKchM_1030","ALamKchP_1030","ALamKchM_1030");

  cLamcKchAnalysis3_1030->BuildCFCollections();
  cLamcKchAnalysis3_1030->BuildAvgSepCollections();
  cLamcKchAnalysis3_1030->BuildPurityCollections();

  TCanvas* c1_1030 = new TCanvas("c1_1030","Plotting Canvas1_1030",1400,500);
  cLamcKchAnalysis3_1030->DrawFinalCFs(c1_1030);
  c1_1030->SaveAs("~/Analysis/K0Lam/NEWASmc/2015_04_08/cLamcKch_1030_CFs.pdf");


  TCanvas* c2_1030 = new TCanvas("c2_1030","Plotting Canvas2_1030");
  TCanvas* c3_1030 = new TCanvas("c3_1030","Plotting Canvas3_1030");
  TCanvas* c4_1030 = new TCanvas("c4_1030","Plotting Canvas4_1030");
  TCanvas* c5_1030 = new TCanvas("c5_1030","Plotting Canvas5_1030");
  cLamcKchAnalysis3_1030->DrawFinalAvgSepCFs(c2_1030,c3_1030,c4_1030,c5_1030);
  c2_1030->SaveAs("~/Analysis/K0Lam/NEWASmc/2015_04_08/LamKchP_1030_AvgSepCFs.pdf");
  c3_1030->SaveAs("~/Analysis/K0Lam/NEWASmc/2015_04_08/LamKchM_1030_AvgSepCFs.pdf");
  c4_1030->SaveAs("~/Analysis/K0Lam/NEWASmc/2015_04_08/ALamKchP_1030_AvgSepCFs.pdf");
  c5_1030->SaveAs("~/Analysis/K0Lam/NEWASmc/2015_04_08/ALamKchM_1030_AvgSepCFs.pdf");


  TCanvas* c6_1030 = new TCanvas("c6_1030","Plotting Canvas6_1030");
  cLamcKchAnalysis3_1030->DrawFinalPurity(c6_1030);
  c6_1030->SaveAs("~/Analysis/K0Lam/NEWASmc/2015_04_08/cLamcKch_1030_Purity.pdf");
*/
  //---------------------------------------------------------------------------------------------------------------------------
/*
  buildAllcLamcKch3 *cLamcKchAnalysis3_3050 = new buildAllcLamcKch3(VectorOfFileNames,"LamKchP_3050","LamKchM_3050","ALamKchP_3050","ALamKchM_3050");

  cLamcKchAnalysis3_3050->BuildCFCollections();
  cLamcKchAnalysis3_3050->BuildAvgSepCollections();
  cLamcKchAnalysis3_3050->BuildPurityCollections();

  TCanvas* c1_3050 = new TCanvas("c1_3050","Plotting Canvas1_3050",1400,500);
  cLamcKchAnalysis3_3050->DrawFinalCFs(c1_3050);
  c1_3050->SaveAs("~/Analysis/K0Lam/NEWASmc/2015_04_08/cLamcKch_3050_CFs.pdf");


  TCanvas* c2_3050 = new TCanvas("c2_3050","Plotting Canvas2_3050");
  TCanvas* c3_3050 = new TCanvas("c3_3050","Plotting Canvas3_3050");
  TCanvas* c4_3050 = new TCanvas("c4_3050","Plotting Canvas4_3050");
  TCanvas* c5_3050 = new TCanvas("c5_3050","Plotting Canvas5_3050");
  cLamcKchAnalysis3_3050->DrawFinalAvgSepCFs(c2_3050,c3_3050,c4_3050,c5_3050);
  c2_3050->SaveAs("~/Analysis/K0Lam/NEWASmc/2015_04_08/LamKchP_3050_AvgSepCFs.pdf");
  c3_3050->SaveAs("~/Analysis/K0Lam/NEWASmc/2015_04_08/LamKchM_3050_AvgSepCFs.pdf");
  c4_3050->SaveAs("~/Analysis/K0Lam/NEWASmc/2015_04_08/ALamKchP_3050_AvgSepCFs.pdf");
  c5_3050->SaveAs("~/Analysis/K0Lam/NEWASmc/2015_04_08/ALamKchM_3050_AvgSepCFs.pdf");


  TCanvas* c6_3050 = new TCanvas("c6_3050","Plotting Canvas6_3050");
  cLamcKchAnalysis3_3050->DrawFinalPurity(c6_3050);
  c6_3050->SaveAs("~/Analysis/K0Lam/NEWASmc/2015_04_08/cLamcKch_3050_Purity.pdf");
*/
}
