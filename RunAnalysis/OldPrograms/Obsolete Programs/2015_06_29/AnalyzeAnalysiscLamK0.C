#include "buildAllcLamK0.cxx"

void AnalyzeAnalysiscLamK0()
{
  vector<TString> VectorOfFileNames;
    VectorOfFileNames.push_back("Resultsgrid_cLamK0_CentBins_Bp1NEW3AS.root");
    VectorOfFileNames.push_back("Resultsgrid_cLamK0_CentBins_Bp2NEW3AS.root");
    VectorOfFileNames.push_back("Resultsgrid_cLamK0_CentBins_Bm1NEW3AS.root");
    VectorOfFileNames.push_back("Resultsgrid_cLamK0_CentBins_Bm2NEW3AS.root");
    VectorOfFileNames.push_back("Resultsgrid_cLamK0_CentBins_Bm3NEW3AS.root");

  TString aSaveDirectory = "~/Analysis/K0Lam/NEW3AS/cLamK0/";  //note:  This directory must already be created
  TString aSaveRootFileName = "Resultsgrid_cLamK0_CentBins_NEW3AS_0010.root";
  TString aCentralityTag = "_0010";

  bool SavePDFs = false;
  bool SaveRootFile = false;

  //-------------------------------------------------------------------------------------------
  buildAllcLamK0 *cLamK0Analysis = new buildAllcLamK0(VectorOfFileNames,"LamK0_0010","ALamK0_0010");
  cLamK0Analysis->BuildCFCollections();
  cLamK0Analysis->BuildAvgSepCollections();
  cLamK0Analysis->BuildPurityCollections();
  cLamK0Analysis->BuildSepCollections(5);

  //-----KStarCfs
  TCanvas* canvasCfs = new TCanvas("canvasCfs","KStar Cfs");
  cLamK0Analysis->DrawFinalCFs(canvasCfs);
  if(SavePDFs) canvasCfs->SaveAs(aSaveDirectory+"Cfs/cLamK0_Cfs"+aCentralityTag+".pdf");

  //-----AvgSepCfs
  TCanvas* canvasAvgSepLamK0 = new TCanvas("canvasAvgSepLamK0","AvgSep LamK0");
  TCanvas* canvasAvgSepALamK0 = new TCanvas("canvasAvgSepALamK0","AvgSep ALamK0");
  cLamK0Analysis->DrawFinalAvgSepCFs(canvasAvgSepLamK0,canvasAvgSepALamK0);
  if(SavePDFs)
  {
    canvasAvgSepLamK0->SaveAs(aSaveDirectory+"AvgSepCfs/LamK0_AvgSepCfs"+aCentralityTag+".pdf");
    canvasAvgSepALamK0->SaveAs(aSaveDirectory+"AvgSepCfs/ALamK0_AvgSepCfs"+aCentralityTag+".pdf");
  }

  //-----Purity
  TCanvas* canvasPurity = new TCanvas("canvasPurity","Purity");
  cLamK0Analysis->DrawFinalPurity(canvasPurity);
  if(SavePDFs) canvasPurity->SaveAs(aSaveDirectory+"Purity/cLamK0_Purity"+aCentralityTag+".pdf");

  //-----SepCfs
  TCanvas* canvasSepLamK0LikeSigns = new TCanvas("canvasSepLamK0LikeSigns","SepCfs LamK0 Like Signs");
  TCanvas* canvasSepLamK0UnlikeSigns = new TCanvas("canvasSepLamK0UnlikeSigns","SepCfs LamK0 Unlike Signs");
  TCanvas* canvasSepALamK0LikeSigns = new TCanvas("canvasSepALamK0LikeSigns","SepCfs ALamK0 Like Signs");
  TCanvas* canvasSepALamK0UnlikeSigns = new TCanvas("canvasSepALamK0UnlikeSigns","SepCfs ALamK0 Unlike Signs");
  cLamK0Analysis->DrawFinalSepCFs(canvasSepLamK0LikeSigns,canvasSepLamK0UnlikeSigns,canvasSepALamK0LikeSigns,canvasSepALamK0UnlikeSigns);
  if(SavePDFs)
  {
    canvasSepLamK0LikeSigns->SaveAs(aSaveDirectory+"SepCfs/LamK0_SepCfsLikeSigns"+aCentralityTag+".pdf");
    canvasSepLamK0UnlikeSigns->SaveAs(aSaveDirectory+"SepCfs/LamK0_SepCfsUnlikeSigns"+aCentralityTag+".pdf");
    canvasSepALamK0LikeSigns->SaveAs(aSaveDirectory+"SepCfs/ALamK0_SepCfsLikeSigns"+aCentralityTag+".pdf");
    canvasSepALamK0UnlikeSigns->SaveAs(aSaveDirectory+"SepCfs/ALamK0_SepCfsUnlikeSigns"+aCentralityTag+".pdf");
  }

  //-----Avg of SepCfs
  TCanvas* canvasAvgOfSepsLamK0LikeSigns = new TCanvas("canvasAvgOfSepsLamK0LikeSigns","Avg of SepCfs LamK0 Like Signs");
  TCanvas* canvasAvgOfSepsLamK0UnlikeSigns = new TCanvas("canvasAvgOfSepsLamK0UnlikeSigns","Avg of SepCfs LamK0 Unlike Signs");
  TCanvas* canvasAvgOfSepsALamK0LikeSigns = new TCanvas("canvasAvgOfSepsALamK0LikeSigns","Avg of SepCfs ALamK0 Like Signs");
  TCanvas* canvasAvgOfSepsALamK0UnlikeSigns = new TCanvas("canvasAvgOfSepsALamK0UnlikeSigns","Avg of SepCfs ALamK0 Unlike Signs");
  cLamK0Analysis->DrawAvgOfFinalSepCFs(canvasAvgOfSepsLamK0LikeSigns,canvasAvgOfSepsLamK0UnlikeSigns,canvasAvgOfSepsALamK0LikeSigns,canvasAvgOfSepsALamK0UnlikeSigns);
  if(SavePDFs)
  {
    canvasAvgOfSepsLamK0LikeSigns->SaveAs(aSaveDirectory+"SepCfs/LamK0_AvgOfSepCfsLikeSigns"+aCentralityTag+".pdf");
    canvasAvgOfSepsLamK0UnlikeSigns->SaveAs(aSaveDirectory+"SepCfs/LamK0_AvgOfSepCfsUnlikeSigns"+aCentralityTag+".pdf");
    canvasAvgOfSepsALamK0LikeSigns->SaveAs(aSaveDirectory+"SepCfs/ALamK0_AvgOfSepCfsLikeSigns"+aCentralityTag+".pdf");
    canvasAvgOfSepsALamK0UnlikeSigns->SaveAs(aSaveDirectory+"SepCfs/ALamK0_AvgOfSepCfsUnlikeSigns"+aCentralityTag+".pdf");
  }

  //-----ROOT file to save
  if(SaveRootFile)
  {
    TFile *myFile = new TFile(aSaveDirectory+aSaveRootFileName, "RECREATE");
    cLamK0Analysis->SaveAll(myFile);
    myFile->Close();
  }

}
