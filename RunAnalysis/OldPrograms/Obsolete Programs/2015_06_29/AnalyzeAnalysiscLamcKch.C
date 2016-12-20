#include "buildAllcLamcKch.cxx"

void AnalyzeAnalysiscLamcKch()
{
  vector<TString> VectorOfFileNames;

    VectorOfFileNames.push_back("Resultsgrid_cLamcKch_CentBins_Bp1NEW2.root");
    VectorOfFileNames.push_back("Resultsgrid_cLamcKch_CentBins_Bp2NEW2.root");
    VectorOfFileNames.push_back("Resultsgrid_cLamcKch_CentBins_Bm1NEW2.root");
    VectorOfFileNames.push_back("Resultsgrid_cLamcKch_CentBins_Bm2NEW2.root");
    VectorOfFileNames.push_back("Resultsgrid_cLamcKch_CentBins_Bm3NEW2.root");

  TString aSaveDirectory = "~/Analysis/K0Lam/NEW2/cLamcKch/";  //note:  This directory must already be created
  TString aSaveRootFileName = "Resultsgrid_cLamcKch_CentBins_NEW2_0010.root";
  TString aCentralityTag = "_0010";

  bool SavePDFs = true;
  bool SaveRootFile = true;

  //---------------------------------------------------------------------------------------------------------------------------

  buildAllcLamcKch *cLamcKchAnalysis = new buildAllcLamcKch(VectorOfFileNames,"LamKchP_0010","LamKchM_0010","ALamKchP_0010","ALamKchM_0010");
  cLamcKchAnalysis->BuildCFCollections();
  cLamcKchAnalysis->BuildAvgSepCollections();
  cLamcKchAnalysis->BuildPurityCollections();
  cLamcKchAnalysis->BuildSepCollections(5);

  //-----KStarCfs
  TCanvas* canvasCfs = new TCanvas("canvasCfs","KStar Cfs",1400,500);
  cLamcKchAnalysis->DrawFinalCFs(canvasCfs);
  if(SavePDFs) canvasCfs->SaveAs(aSaveDirectory+"Cfs/cLamcKch_Cfs"+aCentralityTag+".pdf");

  //-----AvgSepCfs
  TCanvas* canvasAvgSepLamKchP = new TCanvas("canvasAvgSepLamKchP","AvgSep LamKchP");
  TCanvas* canvasAvgSepLamKchM = new TCanvas("canvasAvgSepLamKchM","AvgSep LamKchM");
  TCanvas* canvasAvgSepALamKchP = new TCanvas("canvasAvgSepALamKchP","AvgSep ALamKchP");
  TCanvas* canvasAvgSepALamKchM = new TCanvas("canvasAvgSepALamKchM","AvgSep ALamKchM");
  cLamcKchAnalysis->DrawFinalAvgSepCFs(canvasAvgSepLamKchP,canvasAvgSepLamKchM,canvasAvgSepALamKchP,canvasAvgSepALamKchM);
  if(SavePDFs)
  {
    canvasAvgSepLamKchP->SaveAs(aSaveDirectory+"AvgSepCfs/LamKchP_AvgSepCfs"+aCentralityTag+".pdf");
    canvasAvgSepLamKchM->SaveAs(aSaveDirectory+"AvgSepCfs/LamKchM_AvgSepCfs"+aCentralityTag+".pdf");
    canvasAvgSepALamKchP->SaveAs(aSaveDirectory+"AvgSepCfs/ALamKchP_AvgSepCfs"+aCentralityTag+".pdf");
    canvasAvgSepALamKchM->SaveAs(aSaveDirectory+"AvgSepCfs/ALamKchM_AvgSepCfs"+aCentralityTag+".pdf");
  }

  //-----Purity
  TCanvas* canvasPurity = new TCanvas("canvasPurity","Purity");
  cLamcKchAnalysis->DrawFinalPurity(canvasPurity);
  if(SavePDFs) canvasPurity->SaveAs(aSaveDirectory+"Purity/cLamcKch_Purity"+aCentralityTag+".pdf");

  //-----SepCfs
  TCanvas* canvasSepLamKchP = new TCanvas("canvasSepLamKchP","SepCfs LamKchP");
  TCanvas* canvasSepLamKchM = new TCanvas("canvasSepLamKchM","SepCfs LamKchM");
  TCanvas* canvasSepALamKchP = new TCanvas("canvasSepALamKchP","SepCfs ALamKchP");
  TCanvas* canvasSepALamKchM = new TCanvas("canvasSepALamKchM","SepCfs ALamKchM");
  cLamcKchAnalysis->DrawFinalSepCFs(canvasSepLamKchP,canvasSepLamKchM,canvasSepALamKchP,canvasSepALamKchM);
  if(SavePDFs)
  {
    canvasSepLamKchP->SaveAs(aSaveDirectory+"SepCfs/LamKchP_SepCfs"+aCentralityTag+".pdf");
    canvasSepLamKchM->SaveAs(aSaveDirectory+"SepCfs/LamKchM_SepCfs"+aCentralityTag+".pdf");
    canvasSepALamKchP->SaveAs(aSaveDirectory+"SepCfs/ALamKchP_SepCfs"+aCentralityTag+".pdf");
    canvasSepALamKchM->SaveAs(aSaveDirectory+"SepCfs/ALamKchM_SepCfs"+aCentralityTag+".pdf");
  }

  //-----Avg of SepCfs
  TCanvas* canvasAvgOfSepsLamKchP = new TCanvas("canvasAvgOfSepsLamKchP","Avg of SepCfs LamKchP");
  TCanvas* canvasAvgOfSepsLamKchM = new TCanvas("canvasAvgOfSepsLamKchM","Avg of SepCfs LamKchM");
  TCanvas* canvasAvgOfSepsALamKchP = new TCanvas("canvasAvgOfSepsALamKchP","Avg of SepCfs ALamKchP");
  TCanvas* canvasAvgOfSepsALamKchM = new TCanvas("canvasAvgOfSepsALamKchM","Avg of SepCfs ALamKchM");
  cLamcKchAnalysis->DrawAvgOfFinalSepCFs(canvasAvgOfSepsLamKchP,canvasAvgOfSepsLamKchM,canvasAvgOfSepsALamKchP,canvasAvgOfSepsALamKchM);
  if(SavePDFs)
  {
    canvasAvgOfSepsLamKchP->SaveAs(aSaveDirectory+"SepCfs/LamKchP_AvgOfSepCfs"+aCentralityTag+".pdf");
    canvasAvgOfSepsLamKchM->SaveAs(aSaveDirectory+"SepCfs/LamKchM_AvgOfSepCfs"+aCentralityTag+".pdf");
    canvasAvgOfSepsALamKchP->SaveAs(aSaveDirectory+"SepCfs/ALamKchP_AvgOfSepCfs"+aCentralityTag+".pdf");
    canvasAvgOfSepsALamKchM->SaveAs(aSaveDirectory+"SepCfs/ALamKchM_AvgOfSepCfs"+aCentralityTag+".pdf");
  }

  //-----ROOT file to save
  if(SaveRootFile)
  {
    TFile *myFile = new TFile(aSaveDirectory+aSaveRootFileName, "RECREATE");
    cLamcKchAnalysis->SaveAll(myFile);
    myFile->Close();
  }
}
