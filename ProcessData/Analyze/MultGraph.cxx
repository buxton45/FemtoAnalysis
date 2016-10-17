/*MultGraph.cxx			*/

#include "MultGraph.h"

#ifdef __ROOT__
ClassImp(MultGraph)
#endif



//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________



//________________________________________________________________________________________________________________
MultGraph::MultGraph(int aNx, int aNy) :
  fNx(aNx),
  fNy(aNy)

{
  fGraphs = new TObjArray();
  fGraphs->SetName("GraphsCollection");

  for(int i=0; i<aNx*aNy; i++)
  {
    TObjArray* tTemp = new TObjArray();
    tTemp->SetName(TString::Format("Graphs_%d",i));
  }







}



//________________________________________________________________________________________________________________
MultGraph::~MultGraph()
{

}


//________________________________________________________________________________________________________________
TAxis* MultGraph::SetupAxis(AxisType aAxisType, TGraphAsymmErrors* fGraph, float fXscale, float fYscale)
{
  TAxis* tReturnAxis;

  if(aAxisType == kXaxis) tReturnAxis = fGraph->GetXaxis();
  else if(aAxisType == kYaxis) tReturnAxis = fGraph->GetYaxis();
  else 
  {
    cout << "ERROR: MultGraph::SetupAxis: Invalid aAxisType = " << aAxisType << endl;
    assert(0);
  }

  tReturnAxis->SetTitle("");
  tReturnAxis->SetTitleSize(0.);
  tReturnAxis->SetTitleOffset(0.);

  tReturnAxis->SetLabelFont(43);
  tReturnAxis->SetLabelSize(17);
  tReturnAxis->SetLabelOffset(0.01);

  if(aAxisType == kXaxis)
  {
    tReturnAxis->SetNdivisions(510);
    tReturnAxis->SetTickLength(0.04*fYscale/fXscale);
  }
  else
  {
    tReturnAxis->SetNdivisions(505);
    tReturnAxis->SetTickLength(0.04*fXscale/fYscale);
  }

  return tReturnAxis;
}


//________________________________________________________________________________________________________________
//Adapted from $ROOTSYS/tutorials/graphics/canvas2.C
TPad** MultGraph::CanvasPartition(TCanvas *aCanvas,const Int_t Nx = 2,const Int_t Ny = 2,
                                  Float_t lMargin = 0.15, Float_t rMargin = 0.05,
                                  Float_t bMargin = 0.15, Float_t tMargin = 0.05)
{
   if (!aCanvas) return;

   //Array of pads to be returned
   TPad** returnPadArray = 0;
   returnPadArray = new TPad*[Nx];

   // Setup Pad layout:
   Float_t vSpacing = 0.0;
   Float_t vStep  = (1.- bMargin - tMargin - (Ny-1) * vSpacing) / Ny;

   Float_t hSpacing = 0.0;
   Float_t hStep  = (1.- lMargin - rMargin - (Nx-1) * hSpacing) / Nx;

   Float_t vposd,vposu,vmard,vmaru,vfactor;
   Float_t hposl,hposr,hmarl,hmarr,hfactor;

   for (Int_t i=0;i<Nx;i++) {

      returnPadArray[i] = new TPad[Ny];

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

	 returnPadArray[i][j] = pad;
      }
   }
  return returnPadArray;
}


//________________________________________________________________________________________________________________
float** MultGraph::GetScaleFactors(AxisType aAxisType, TPad **fPadArray, int Nx, int Ny)
{
  float** returnScaleFactors = 0;
  returnScaleFactors = new float*[Nx];

  for(int i=0; i<Nx; i++)
  {
    returnScaleFactors[i] = new float[Ny];
    for(int j=0; j<Ny; j++)
    {
      if(aAxisType == kXaxis) returnScaleFactors[i][j] = fPadArray[0][0].GetAbsWNDC()/fPadArray[i][j].GetAbsWNDC();
      else if(aAxisType == kYaxis) returnScaleFactors[i][j] = fPadArray[0][0].GetAbsHNDC()/fPadArray[i][j].GetAbsHNDC();
      else
      {
        cout << "ERROR: MultGraph::GetScaleFactors: Invalid aAxisType = " << aAxisType << endl;
        assert(0);
      }
    }
  }

  return returnScaleFactors;
}




//________________________________________________________________________________________________________________
TPaveText* MultGraph::SetupTPaveText(TString fText, double fXminOffset, double fYminOffset, float fXscale, float fYscale)
{
  double TextXmin = 0.75;
  double TextYmin = 0.75;
  double TextWidth = 0.15;
  double TextHeight = 0.10;

  double xmin,ymin,xmax,ymax;

  Size_t TextSize = 15;
  float MarkerSize = 0.75;
  float TextFont = 63;  //when TextFont prevision >=3, TextSize in pixels, not %pad

  xmin = TextXmin + fXminOffset;
  ymin = TextYmin + fYminOffset;
  xmax = xmin + fXscale*TextWidth;
  ymax = ymin + fYscale*TextHeight;

  TPaveText* returnText = new TPaveText(xmin,ymin,xmax,ymax,"NDC");
    returnText->SetFillColor(0);
    returnText->SetBorderSize(0);
    returnText->SetTextAlign(22);
    returnText->SetTextFont(TextFont);
    returnText->SetTextSize(TextSize);
    returnText->AddText(fText);

  return returnText;
}



//________________________________________________________________________________________________________________
void DrawInPad(TPad** fPadArray, int Nx, int Ny, TGraphAsymmErrors* ALICEstat, TGraphAsymmErrors* ALICEsys, TPaveText* text)
{
  fPadArray[Nx][Ny].cd();
  ALICEstat->Draw("aep");
  ALICEsys->Draw("e2psame");
  text->Draw();

}





