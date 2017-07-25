/*CanvasPartition.cxx			*/

#include "CanvasPartition.h"

#ifdef __ROOT__
ClassImp(CanvasPartition)
#endif



//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________



//________________________________________________________________________________________________________________
CanvasPartition::CanvasPartition(TString aCanvasName, int aNx, int aNy, double aXRangeLow, double aXRangeHigh, double aYRangeLow, double aYRangeHigh, float aMarginLeft, float aMarginRight, float aMarginBottom, float aMarginTop) :
  fDrawUnityLine(false),
  fDrawOptStat(true),
  fNx(aNx),
  fNy(aNy),

  fXaxisRangeLow(aXRangeLow),
  fXaxisRangeHigh(aXRangeHigh),
  fYaxisRangeLow(aYRangeLow),
  fYaxisRangeHigh(aYRangeHigh),

  fMarginLeft(aMarginLeft),
  fMarginRight(aMarginRight),
  fMarginBottom(aMarginBottom),
  fMarginTop(aMarginTop),

  fCanvas(0),

  fGraphs(0),
  fGraphsDrawOptions(aNx*aNy),
  fPadPaveTexts(0),
  fPadLegends(0),

  fPadArray(0),

  fXScaleFactors(0),
  fYScaleFactors(0)

{
  fCanvas = new TCanvas(aCanvasName,aCanvasName);
  fPadArray = BuildPartition(fCanvas,aNx,aNy,aMarginLeft,aMarginRight,aMarginBottom,aMarginTop);
  
  fXScaleFactors = GetScaleFactors(kXaxis,fPadArray,fNx,fNy);
  fYScaleFactors = GetScaleFactors(kYaxis,fPadArray,fNx,fNy);

  fGraphs = new TObjArray();
    fGraphs->SetName("GraphsCollection");
  fPadPaveTexts = new TObjArray();
    fPadPaveTexts->SetName("PaveTextsCollection");
  fPadLegends = new TObjArray();
    fPadLegends->SetName("LegendsCollection");

  for(int i=0; i<aNx*aNy; i++)
  {
    TObjArray* tTemp = new TObjArray();
    tTemp->SetName(TString::Format("Graphs_%d",i));
    fGraphs->Add(tTemp);

    TObjArray* tTemp2 = new TObjArray();
    tTemp2->SetName(TString::Format("PaveTexts_%d",i));
    fPadPaveTexts->Add(tTemp2);

    TObjArray* tTemp3 = new TObjArray();
    tTemp3->SetName(TString::Format("Legend_%d",i));
    fPadLegends->Add(tTemp3);
  }

}



//________________________________________________________________________________________________________________
CanvasPartition::~CanvasPartition()
{

}


//________________________________________________________________________________________________________________
//Adapted from $ROOTSYS/tutorials/graphics/canvas2.C
td2dTPadVec CanvasPartition::BuildPartition(TCanvas *aCanvas,const Int_t Nx,const Int_t Ny,
                                  Float_t lMargin, Float_t rMargin,
                                  Float_t bMargin, Float_t tMargin)
{
   //Array of pads to be returned
    td2dTPadVec returnPadArray(0);

   if (!aCanvas) return returnPadArray;

   // Setup Pad layout:
   Float_t vSpacing = 0.0;
   Float_t vStep  = (1.- bMargin - tMargin - (Ny-1) * vSpacing) / Ny;

   Float_t hSpacing = 0.0;
   Float_t hStep  = (1.- lMargin - rMargin - (Nx-1) * hSpacing) / Nx;

   Float_t vposd,vposu,vmard,vmaru,vfactor;
   Float_t hposl,hposr,hmarl,hmarr,hfactor;

   for (Int_t i=0;i<Nx;i++) {
      td1dTPadVec tTempPadVec(0);

      if (i==0) {
         hposl = 0.0;
         hposr = lMargin + hStep;
         hfactor = hposr-hposl;
         hmarl = lMargin / hfactor;
         hmarr = 0.0;
      } else if (i == Nx-1) {
         hposl = hposr + hSpacing;
         hposr = hposl + hStep + rMargin;
         hfactor = hposr-hposl;
         hmarl = 0.0;
         hmarr = rMargin / (hposr-hposl);
      } else {
         hposl = hposr + hSpacing;
         hposr = hposl + hStep;
         hfactor = hposr-hposl;
         hmarl = 0.0;
         hmarr = 0.0;
      }

      for (Int_t j=0;j<Ny;j++) {
/*
         //Old numbering scheme, bottom to top
         if (j==0) {
            vposd = 0.0;
            vposu = bMargin + vStep;
            vfactor = vposu-vposd;
            vmard = bMargin / vfactor;
            vmaru = 0.0;
         } else if (j == Ny-1) {
            vposd = vposu + vSpacing;
            vposu = vposd + vStep + tMargin;
            vfactor = vposu-vposd;
            vmard = 0.0;
            vmaru = tMargin / (vposu-vposd);
         } else {
            vposd = vposu + vSpacing;
            vposu = vposd + vStep;
            vfactor = vposu-vposd;
            vmard = 0.0;
            vmaru = 0.0;
         }
*/
         //New numbering scheme, top to bottom
         if (j==0) {
            vposu = 1.0;
            vposd = vposu-vStep-tMargin;
            vfactor = vposu-vposd;
            vmard = 0.0;
            if(Ny==1) vmard = 0.06;  //TODO make more general
            vmaru = tMargin/vfactor;

         } else if (j == Ny-1) {
            vposu = vposd-vSpacing;
            vposd = vposu-vStep-bMargin;

            //vposd should = 0 here, but for sometimes it is slightly off, probably due to machine precision
            //If vposd goes negative, this causes an error, so I need to following
            if(abs(vposd) < 0.00001) vposd = 0.;
            else assert(0);

            vfactor = vposu-vposd;
            vmard = bMargin/vfactor;
            vmaru = 0.0;
         } else {
            vposu = vposd-vSpacing;
            vposd = vposu-vStep;
            vfactor = vposu-vposd;
            vmard = 0.0;
            vmaru = 0.0;
         }


         aCanvas->cd(0);

         char name[16];
         sprintf(name,"pad_%i_%i",i,j);
         TPad *pad = (TPad*) gROOT->FindObject(name);
         if (pad) delete pad;
         pad = new TPad(name,"",hposl,vposd,hposr,vposu);
         pad->SetLeftMargin(hmarl);
         pad->SetRightMargin(hmarr);
         pad->SetBottomMargin(vmard);
         pad->SetTopMargin(vmaru);

         pad->SetFrameBorderMode(0);
         pad->SetBorderMode(0);
         pad->SetBorderSize(0);

	 pad->SetFillStyle(4000);
	 pad->SetFrameFillStyle(4000);
	 //pad->SetTicks(1,1);
         pad->Draw();
/*
         pad->Update();
         pad->cd();
         pad->DrawFrame(fXaxisRangeLow,fYaxisRangeLow,fXaxisRangeHigh,fYaxisRangeHigh);
*/
         tTempPadVec.push_back(pad);
      }
      returnPadArray.push_back(tTempPadVec);
   }
  return returnPadArray;
}


//________________________________________________________________________________________________________________
float** CanvasPartition::GetScaleFactors(AxisType aAxisType, td2dTPadVec &fPadArray, int Nx, int Ny)
{
  float** returnScaleFactors = 0;
  returnScaleFactors = new float*[Nx];

  for(int i=0; i<Nx; i++)
  {
    returnScaleFactors[i] = new float[Ny];
    for(int j=0; j<Ny; j++)
    {
      if(aAxisType == kXaxis) returnScaleFactors[i][j] = fPadArray[0][0]->GetAbsWNDC()/fPadArray[i][j]->GetAbsWNDC();
      else if(aAxisType == kYaxis) returnScaleFactors[i][j] = fPadArray[0][0]->GetAbsHNDC()/fPadArray[i][j]->GetAbsHNDC();
      else
      {
        cout << "ERROR: CanvasPartition::GetScaleFactors: Invalid aAxisType = " << aAxisType << endl;
        assert(0);
      }
    }
  }

  return returnScaleFactors;
}

//________________________________________________________________________________________________________________
void CanvasPartition::SetupOptStat(int aNx, int aNy, double aStatX, double aStatY, double aStatW, double aStatH)
{
  float tLeftMargin = fPadArray[aNx][aNy]->GetLeftMargin();
  float tRightMargin = fPadArray[aNx][aNy]->GetRightMargin();
  float tTopMargin = fPadArray[aNx][aNy]->GetTopMargin();
  float tBottomMargin = fPadArray[aNx][aNy]->GetBottomMargin();

  float tReNormalizedWidth = 1. - (tLeftMargin+tRightMargin);
  float tReNormalizedHeight = 1. - (tTopMargin+tBottomMargin);

  //------------------------------------

  double tNormalizedTextXmin = tLeftMargin + aStatX*tReNormalizedWidth;
  double tNormalizedTextYmin = tBottomMargin + aStatY*tReNormalizedHeight;

  tReNormalizedWidth *= aStatW;
  tReNormalizedHeight *= aStatH;
  //------------------------------------

  gStyle->SetOptFit();
  gStyle->SetStatH(tReNormalizedHeight);
  gStyle->SetStatW(tReNormalizedWidth);

  gStyle->SetStatX(tNormalizedTextXmin);
  gStyle->SetStatY(tNormalizedTextYmin);
}


//________________________________________________________________________________________________________________
TPaveText* CanvasPartition::SetupTPaveText(TString aText, int aNx, int aNy, double aTextXmin, double aTextYmin, double aTextWidth, double aTextHeight, double aTextFont, double aTextSize)
{
  float tLeftMargin = fPadArray[aNx][aNy]->GetLeftMargin();
  float tRightMargin = fPadArray[aNx][aNy]->GetRightMargin();
  float tTopMargin = fPadArray[aNx][aNy]->GetTopMargin();
  float tBottomMargin = fPadArray[aNx][aNy]->GetBottomMargin();

  float tReNormalizedWidth = 1. - (tLeftMargin+tRightMargin);
  float tReNormalizedHeight = 1. - (tTopMargin+tBottomMargin);

  //------------------------------------

  double tNormalizedTextXmin = tLeftMargin + aTextXmin*tReNormalizedWidth;
  double tNormalizedTextYmin = tBottomMargin + aTextYmin*tReNormalizedHeight;

  double tNormalizedTextXmax = tNormalizedTextXmin + aTextWidth*tReNormalizedWidth;
  double tNormalizedTextYmax = tNormalizedTextYmin + aTextHeight*tReNormalizedHeight;

  //------------------------------------

  TPaveText* returnText = new TPaveText(tNormalizedTextXmin,tNormalizedTextYmin,tNormalizedTextXmax,tNormalizedTextYmax,"NDC");
    returnText->SetFillColor(0);
    returnText->SetBorderSize(0);
    returnText->SetTextAlign(22);
    returnText->SetTextFont(aTextFont);
    returnText->SetTextSize(aTextSize);
    if(!aText.IsNull()) returnText->AddText(aText);

  return returnText;
}

//________________________________________________________________________________________________________________
void CanvasPartition::AddPadPaveText(TPaveText* aText, int aNx, int aNy)
{
  int tPosition = aNx + aNy*fNx;
  ((TObjArray*)fPadPaveTexts->At(tPosition))->Add(aText);
}

//________________________________________________________________________________________________________________
void CanvasPartition::SetupTLegend(TString aHeader, int aNx, int aNy, double aTextXmin, double aTextYmin, double aTextWidth, double aTextHeight)
{
  float tLeftMargin = fPadArray[aNx][aNy]->GetLeftMargin();
  float tRightMargin = fPadArray[aNx][aNy]->GetRightMargin();
  float tTopMargin = fPadArray[aNx][aNy]->GetTopMargin();
  float tBottomMargin = fPadArray[aNx][aNy]->GetBottomMargin();

  float tReNormalizedWidth = 1. - (tLeftMargin+tRightMargin);
  float tReNormalizedHeight = 1. - (tTopMargin+tBottomMargin);

  //------------------------------------

  double tNormalizedTextXmin = tLeftMargin + aTextXmin*tReNormalizedWidth;
  double tNormalizedTextYmin = tBottomMargin + aTextYmin*tReNormalizedHeight;

  double tNormalizedTextXmax = tNormalizedTextXmin + aTextWidth*tReNormalizedWidth;
  double tNormalizedTextYmax = tNormalizedTextYmin + aTextHeight*tReNormalizedHeight;

  //------------------------------------

  TLegend* tLeg = new TLegend(tNormalizedTextXmin,tNormalizedTextYmin,tNormalizedTextXmax,tNormalizedTextYmax,"NDC");
    tLeg->SetFillColor(0);
    tLeg->SetBorderSize(0);
    tLeg->SetTextAlign(22);
    if(!aHeader.IsNull()) tLeg->SetHeader(aHeader);

  //------------------------------------
  int tPosition = aNx + aNy*fNx;
  ((TObjArray*)fPadLegends->At(tPosition))->Add(tLeg);
}

//________________________________________________________________________________________________________________
void CanvasPartition::AddLegendEntry(int aNx, int aNy, const TObject *tObj, const char *label, Option_t *option, int tLegNumInPad)
{
  int tPosition = aNx + aNy*fNx;
  TObjArray* tLegArray = ((TObjArray*)fPadLegends->At(tPosition));
  ((TLegend*)tLegArray->At(tLegNumInPad))->AddEntry(tObj, label, option);
}



//________________________________________________________________________________________________________________
void CanvasPartition::DrawInPad(int aNx, int aNy)
{
  fPadArray[aNx][aNy]->cd();
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  if(fDrawOptStat) SetupOptStat(aNx,aNy,0.85,0.60,0.20,0.25);

  int tPosition = aNx + aNy*fNx;
  TObjArray* tGraphsToDraw = ((TObjArray*)fGraphs->At(tPosition));
  assert(tGraphsToDraw->GetEntries() == (int)fGraphsDrawOptions[tPosition].size());

  TIter tNextGraph(tGraphsToDraw);
  TObject *tGraphObj = NULL;

  //One cannot use SetRangeUser to set xrange below (or, presumably, above) the limits of the histogram
  //Therefore, I build this tTrash histogram below and use it to draw the correct axes.
  //  Something similar could have be done on the individual pads themselves in BuildPartition (above),
  //  but this method makes it easier for me to alter the attributes of the axes.
  //NOTE: The histograms MUST be drawn with the option "sames" (NOT "same") for the fit parameters to still be drawn
/*
  TH1F* tTrash = new TH1F("","",1,fXaxisRangeLow,fXaxisRangeHigh);
    SetupAxis(kXaxis,tTrash,fXScaleFactors[aNx][aNy],fYScaleFactors[aNx][aNy]);
    SetupAxis(kYaxis,tTrash,fXScaleFactors[aNx][aNy],fYScaleFactors[aNx][aNy]);
    tTrash->DrawCopy("AXIS");
  delete tTrash;
*/

  int tCounter = 0;
  while(tGraphObj = tNextGraph())
  {
    if(tCounter==0) tGraphObj->Draw("AXIS"+fGraphsDrawOptions[tPosition][tCounter]);
    if(fGraphsDrawOptions[tPosition][tCounter] == TString("lsame")) ((TH1*)tGraphObj)->GetXaxis()->SetRange(1,((TH1*)tGraphObj)->GetNbinsX());  //TODO work-around so stupid 
                                                                                                                                                //underflow is not drawn
    tGraphObj->Draw(fGraphsDrawOptions[tPosition][tCounter]);
    tCounter++;
  }

  TObjArray* tPaveTexts = (TObjArray*)fPadPaveTexts->At(tPosition);
  for(int i=0; i<tPaveTexts->GetEntries(); i++) tPaveTexts->At(i)->Draw();

  TObjArray* tLegends = (TObjArray*)fPadLegends->At(tPosition);
  for(int i=0; i<tLegends->GetEntries(); i++) tLegends->At(i)->Draw();

  if(fDrawUnityLine)
  {
    double tXaxisRangeLow;
    if(fXaxisRangeLow<0) tXaxisRangeLow = 0.;
    else tXaxisRangeLow = fXaxisRangeLow;
    TLine *tLine = new TLine(tXaxisRangeLow,1.,fXaxisRangeHigh,1.);
    tLine->SetLineColor(TColor::GetColorTransparent(kGray,0.75));
    tLine->Draw();
  }

  tGraphsToDraw->At(0)->Draw("ex0same");
}

//________________________________________________________________________________________________________________
void CanvasPartition::DrawAll()
{
  for(int i=0; i<fNx; i++)
  {
    for(int j=0; j<fNy; j++)
    {
      DrawInPad(i,j);
    }
  }
}



//________________________________________________________________________________________________________________
void CanvasPartition::DrawXaxisTitle(TString aTitle, int aTextFont, int aTextSize, double aXLow, double aYLow)
{
  fCanvas->cd(0);

  TLatex *tXax = new TLatex(aXLow,aYLow,aTitle);
  tXax->SetTextFont(aTextFont);
  tXax->SetTextSize(aTextSize);
  tXax->SetTextAlign(10);
//  tXax->Draw();
  tXax->DrawLatex(0.9-tXax->GetXsize(),aYLow,aTitle);
}

//________________________________________________________________________________________________________________
void CanvasPartition::DrawYaxisTitle(TString aTitle, int aTextFont, int aTextSize, double aXLow, double aYLow)
{
  fCanvas->cd(0);

  TLatex *tYax = new TLatex(aXLow,aYLow,aTitle);
  tYax->SetTextFont(aTextFont);
  tYax->SetTextSize(aTextSize);
  tYax->SetTextAngle(90);
  tYax->SetTextAlign(10);
  tYax->Draw();
}



