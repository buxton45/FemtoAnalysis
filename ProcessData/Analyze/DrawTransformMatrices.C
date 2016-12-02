//________________________________________________________________________________________________________________
TH2D* Get2dHisto(TString FileName, TString HistoName)
{
  TFile f1(FileName);
  TH2D* ReturnHisto = (TH2D*)f1.Get(HistoName);
  TH2D *ReturnHistoClone = (TH2D*)ReturnHisto->Clone();
  ReturnHistoClone->SetDirectory(0);

  return ReturnHistoClone;
}

//________________________________________________________________________________________________________________
TCanvas* DrawTransform(TH2D* aMatrix, TString aMotherName, TString aDaughterName, TString aPairPartnerName, bool aDrawLogZ=false)
{
  TString tMatrixTitle = aMotherName + TString(" To ") + aDaughterName + TString(" TransformMatrix ")
                         + TString("(") + aDaughterName + aPairPartnerName + TString(")");
  aMatrix->SetTitle(tMatrixTitle);

  TString tCanvasName = TString("can") + tMatrixTitle;
  TCanvas* tReturnCan = new TCanvas(tCanvasName,tCanvasName);
  if(aDrawLogZ) tReturnCan->SetLogz();


  TString tXName = TString("k*_{") + aDaughterName + aPairPartnerName + TString("}(GeV/c)");
  TString tYName = TString("k*_{") + aMotherName + aPairPartnerName + TString("}(GeV/c)");


  aMatrix->GetXaxis()->SetTitle(tXName);
    aMatrix->GetXaxis()->SetTitleSize(0.04);
    aMatrix->GetXaxis()->SetTitleOffset(1.1);

  aMatrix->GetYaxis()->SetTitle(tYName);
    aMatrix->GetYaxis()->SetTitleSize(0.04);
    aMatrix->GetYaxis()->SetTitleOffset(1.2);

  aMatrix->GetZaxis()->SetLabelSize(0.02);
  aMatrix->GetZaxis()->SetLabelOffset(0.004);

  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);
  tReturnCan->cd();
  aMatrix->Draw("colz");


  //-------------------------
  TString tBoxText = aMotherName + aPairPartnerName + TString(" To ") + aDaughterName + aPairPartnerName;
  double tTextXmin = 0.15;
  double tTextXmax = 0.35;
  double tTextYmin = 0.75;
  double tTextYmax = 0.85;
  TPaveText* tText = new TPaveText(tTextXmin,tTextYmin,tTextXmax,tTextYmax,"NDC");
    tText->SetFillColor(0);
    tText->SetBorderSize(0);
    tText->AddText(tBoxText);
  tText->Draw();

  return tReturnCan;
}

//________________________________________________________________________________________________________________
void DrawTransformMatrices()
{
  TString tNameKchP = "K^{+}";
  TString tNameKchM = "K^{-}";

  TString tNameLam = "#Lambda";
  TString tNameALam = "#bar{#Lambda}";

  TString tNameSig = "#Sigma";
  TString tNameASig = "#bar{#Sigma}";

  TString tNameXiC = "#Xi^{ch}";
  TString tNameAXiC = "#bar{#Xi}^{ch}";

  TString tNameXi0 = "#Xi^{0}";
  TString tNameAXi0 = "#bar{#Xi}^{0}";

  TString tNameOmega = "#Omega";
  TString tNameAOmega = "#bar{#Omega}";

  //--------------------------------------------------
  bool bDrawLogZ = false;

  TString tFileName = "/home/jesse/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b2/TransformMatrices_Mix5.root";
  TString tSaveLocationBase = "/home/jesse/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b2/Figures/";

  TString SigToLamName = "fSigToLamKchPTransform";
  TString XiCToLamName = "fXiCToLamKchPTransform";
  TString Xi0ToLamName = "fXi0ToLamKchPTransform";
  TString OmegaToLamName = "fOmegaToLamKchPTransform";
  TString ASigToALamName = "fASigToALamKchPTransform";
  TString AXiCToALamName = "fAXiCToALamKchPTransform";
  TString AXi0ToALamName = "fAXi0ToALamKchPTransform";
  TString AOmegaToALamName = "fAOmegaToALamKchPTransform";

  TH2D* SigToLam = Get2dHisto(tFileName,SigToLamName);
  TH2D* XiCToLam = Get2dHisto(tFileName,XiCToLamName);
  TH2D* Xi0ToLam = Get2dHisto(tFileName,Xi0ToLamName);
  TH2D* OmegaToLam = Get2dHisto(tFileName,OmegaToLamName);
  TH2D* ASigToALam = Get2dHisto(tFileName,ASigToALamName);
  TH2D* AXiCToALam = Get2dHisto(tFileName,AXiCToALamName);
  TH2D* AXi0ToALam = Get2dHisto(tFileName,AXi0ToALamName);
  TH2D* AOmegaToALam = Get2dHisto(tFileName,AOmegaToALamName);

  TCanvas* tCanSigToLam = DrawTransform(SigToLam, tNameSig, tNameLam, tNameKchP, bDrawLogZ);
    tCanSigToLam->SaveAs(tSaveLocationBase + SigToLamName+ TString(".pdf"));
  TCanvas* tCanXiCToLam = DrawTransform(XiCToLam, tNameXiC, tNameLam, tNameKchP, bDrawLogZ);
    tCanXiCToLam->SaveAs(tSaveLocationBase + XiCToLamName+ TString(".pdf"));
  TCanvas* tCanXi0ToLam = DrawTransform(Xi0ToLam, tNameXi0, tNameLam, tNameKchP, bDrawLogZ);
    tCanXi0ToLam->SaveAs(tSaveLocationBase + Xi0ToLamName+ TString(".pdf"));
  TCanvas* tCanOmegaToLam = DrawTransform(OmegaToLam, tNameOmega, tNameLam, tNameKchP, bDrawLogZ);
    tCanOmegaToLam->SaveAs(tSaveLocationBase + OmegaToLamName+ TString(".pdf"));
  TCanvas* tCanASigToALam = DrawTransform(ASigToALam, tNameASig, tNameALam, tNameKchP, bDrawLogZ);
    tCanASigToALam->SaveAs(tSaveLocationBase + ASigToALamName+ TString(".pdf"));
  TCanvas* tCanAXiCToALam = DrawTransform(AXiCToALam, tNameAXiC, tNameALam, tNameKchP, bDrawLogZ);
    tCanAXiCToALam->SaveAs(tSaveLocationBase + AXiCToALamName+ TString(".pdf"));
  TCanvas* tCanAXi0ToALam = DrawTransform(AXi0ToALam, tNameAXi0, tNameALam, tNameKchP, bDrawLogZ);
    tCanAXi0ToALam->SaveAs(tSaveLocationBase + AXi0ToALamName+ TString(".pdf"));
  TCanvas* tCanAOmegaToALam = DrawTransform(AOmegaToALam, tNameAOmega, tNameALam, tNameKchP, bDrawLogZ);
    tCanAOmegaToALam->SaveAs(tSaveLocationBase + AOmegaToALamName+ TString(".pdf"));



}
