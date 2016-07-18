///////////////////////////////////////////////////////////////////////////
// Interpolate:                                                        //
///////////////////////////////////////////////////////////////////////////


#include "Interpolate.h"

#ifdef __ROOT__
ClassImp(Interpolate)
#endif


Interpolate::Interpolate():
  fGTildeReal(0),
  fGTildeImag(0),
  fPairKStar3dVec(0),
  fPairs2dVec(0),
  fInterpolateGPU(0)

{
  //TODO figure out why TNtuple libraries were not loading (at run time)
  //but for some reason this makes them load
  TNtuple* aTNtuple = new TNtuple();
  delete aTNtuple;

  fInterpolateGPU = new InterpolateGPU();
}


Interpolate::~Interpolate()
{
}


//________________________________________________________________________________________________________________
int Interpolate::GetBinNumber(double aBinSize, int aNbins, double aValue)
{
//TODO check the accuracy of this
  double tBinKStarMin, tBinKStarMax;

  for(int i=0; i<aNbins; i++)
  {
    tBinKStarMin = i*aBinSize;
    tBinKStarMax = (i+1)*aBinSize;

    if(aValue>=tBinKStarMin && aValue<tBinKStarMax) return i;
  }

  return -1;  //i.e. failure
}

//________________________________________________________________________________________________________________
int Interpolate::GetBinNumber(int aNbins, double aMin, double aMax, double aValue)
{
//TODO check the accuracy of this
  double tBinSize = (aMax-aMin)/aNbins;
  double tBinKStarMin, tBinKStarMax;

  for(int i=0; i<aNbins; i++)
  {
    tBinKStarMin = i*tBinSize + aMin;
    tBinKStarMax = (i+1)*tBinSize + aMin;

    if(aValue>=tBinKStarMin && aValue<tBinKStarMax) return i;
  }

  return -1;  //i.e. failure
}

//________________________________________________________________________________________________________________
int Interpolate::GetBinNumber(double aBinWidth, double aMin, double aMax, double aValue)
{
//TODO check the accuracy of this
  int tNbins = (aMax-aMin)/aBinWidth;
  double tBinKStarMin, tBinKStarMax;

  for(int i=0; i<tNbins; i++)
  {
    tBinKStarMin = i*aBinWidth + aMin;
    tBinKStarMax = (i+1)*aBinWidth + aMin;

    if(aValue>=tBinKStarMin && aValue<tBinKStarMax) return i;
  }

  return -1;  //i.e. failure
}


//________________________________________________________________________________________________________________
void Interpolate::BuildPairKStar3dVec(TString aPairKStarNtupleLocationBase, AnalysisType aAnalysisType, CentralityType aCentralityType, BFieldType aBFieldType, int aNbinsKStar, double aKStarMin, double aKStarMax)
{
  cout << "Beginning conversion of TNtuple to 3dVec" << endl;
  std::clock_t start = std::clock();

  //----****----****----****----****----****----****
  //TODO make this automated
  //TODO add centrality selection to this
  //Also, this is basically FitPartialAnalysis::ConnectAnalysisDirectory

  TString tFileLocation = aPairKStarNtupleLocationBase+TString(cBFieldTags[aBFieldType])+".root";
  TFile *tFile = TFile::Open(tFileLocation);

  TList *tFemtoList = (TList*)tFile->Get("femtolist");

  TString tArrayName = TString(cAnalysisBaseTags[aAnalysisType])+TString(cCentralityTags[aCentralityType]);
  TObjArray *tArray = (TObjArray*)tFemtoList->FindObject(tArrayName)->Clone();
    tArray->SetOwner();

  TString tNtupleName = "PairKStarKStarCf_"+TString(cAnalysisBaseTags[aAnalysisType]);
  TNtuple *tPairKStarNtuple = (TNtuple*)tArray->FindObject(tNtupleName);

  //----****----****----****----****----****----****

  float tTupleKStarMag, tTupleKStarOut, tTupleKStarSide, tTupleKStarLong;

  tPairKStarNtuple->SetBranchAddress("KStarMag", &tTupleKStarMag);
  tPairKStarNtuple->SetBranchAddress("KStarOut", &tTupleKStarOut);
  tPairKStarNtuple->SetBranchAddress("KStarSide", &tTupleKStarSide);
  tPairKStarNtuple->SetBranchAddress("KStarLong", &tTupleKStarLong);

  //--------------------------------------

  double tBinSize = (aKStarMax-aKStarMin)/aNbinsKStar;
cout << "tBinSize = " << tBinSize << endl;

  fPairKStar3dVec.clear();
//TODO check the following
//  fPairKStar3dVec.resize(aNbinsKStar, vector<vector<double> >());
  fPairKStar3dVec.resize(aNbinsKStar, vector<vector<double> >(0,vector<double>(0)));

cout << "Pre: fPairKStar3dVec.size() = " << fPairKStar3dVec.size() << endl;

  vector<double> tTempEntry;
  int tKStarBinNumber;

  for(int i=0; i<tPairKStarNtuple->GetEntries(); i++)
  {
    tPairKStarNtuple->GetEntry(i);

    tKStarBinNumber = GetBinNumber(tBinSize, aNbinsKStar, tTupleKStarMag);
    if(tKStarBinNumber>=0)  //i.e, the KStarMag value is within my bins of interest
    {
      tTempEntry.clear();
        tTempEntry.push_back(tTupleKStarMag);
        tTempEntry.push_back(tTupleKStarOut);
        tTempEntry.push_back(tTupleKStarSide);
        tTempEntry.push_back(tTupleKStarLong);

      fPairKStar3dVec[tKStarBinNumber].push_back(tTempEntry);
    }
  }

  //Clean up----------------------------------------------------
  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  //This needs to be done, otherwise the other TObjArrays in TList are
  //thrown onto the stack (even after calling delete on the tFemtoList object)
  //which causes the RAM to be used up rapidly!
  //In short, TLists are stupid
  TIter next(tFemtoList);
  TObject *obj = nullptr;
  while((obj = next()))
  {
    TObjArray *arr = dynamic_cast<TObjArray*>(obj);
    if(arr) arr->Delete();
  }
  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  tFemtoList->Delete();
  delete tFemtoList;

  tArray->Delete();
  delete tArray;

  //NOTE:  Apparently, tPairKStarNtuple is deleted when file is closed
  //       In fact, trying to delete it manually causes errors!

  tFile->Close();
  delete tFile;
  //------------------------------------------------------------

  double duration = (std::clock() - start)/(double) CLOCKS_PER_SEC;
  cout << "Conversion complete in " << duration << " seconds" << endl;


cout << "Final: fPairKStar3dVec.size() = " << fPairKStar3dVec.size() << endl;
for(int i=0; i<aNbinsKStar; i++) cout << "i = " << i << "and fPairKStar3dVec[i].size() = " << fPairKStar3dVec[i].size() << endl;
}

//________________________________________________________________________________________________________________
void Interpolate::MakeOtherArrays(TString aFileBaseName)
{
  TString aFileName = aFileBaseName+".root";
  TFile* tInterpHistFile = TFile::Open(aFileName);

//--------------------------------------------------------------
  TH2D* tGTildeRealHist = (TH2D*)tInterpHistFile->Get("GTildeReal");
  TH2D* tGTildeImagHist = (TH2D*)tInterpHistFile->Get("GTildeImag");

//--------------------------------------------------------------
  fGTildeInfo.nBinsK = tGTildeRealHist->GetXaxis()->GetNbins();
  fGTildeInfo.binWidthK = tGTildeRealHist->GetXaxis()->GetBinWidth(1);
  fGTildeInfo.minK = tGTildeRealHist->GetXaxis()->GetBinLowEdge(1);
  fGTildeInfo.minInterpK = tGTildeRealHist->GetXaxis()->GetBinCenter(1);
  fGTildeInfo.maxK = tGTildeRealHist->GetXaxis()->GetBinUpEdge(tGTildeRealHist->GetNbinsX());
  fGTildeInfo.maxInterpK = tGTildeRealHist->GetXaxis()->GetBinCenter(tGTildeRealHist->GetNbinsX());

  fGTildeInfo.nBinsR = tGTildeRealHist->GetYaxis()->GetNbins();
  fGTildeInfo.binWidthR = tGTildeRealHist->GetYaxis()->GetBinWidth(1);
  fGTildeInfo.minR = tGTildeRealHist->GetYaxis()->GetBinLowEdge(1);
  fGTildeInfo.minInterpR = tGTildeRealHist->GetYaxis()->GetBinCenter(1);
  fGTildeInfo.maxR = tGTildeRealHist->GetYaxis()->GetBinUpEdge(tGTildeRealHist->GetNbinsY());
  fGTildeInfo.maxInterpR = tGTildeRealHist->GetYaxis()->GetBinCenter(tGTildeRealHist->GetNbinsY());



//--------------------------------------------------------------

  int tNbinsK = tGTildeRealHist->GetXaxis()->GetNbins();
  int tNbinsR = tGTildeRealHist->GetYaxis()->GetNbins();

  fGTildeReal.resize(tNbinsK, vector<double>(tNbinsR,0));
  fGTildeImag.resize(tNbinsK, vector<double>(tNbinsR,0));

  for(int iK=1; iK<=tNbinsK; iK++)
  {
    for(int iR=1; iR<=tNbinsR; iR++)
    {
      fGTildeReal[iK-1][iR-1] = tGTildeRealHist->GetBinContent(iK,iR);
      fGTildeImag[iK-1][iR-1] = tGTildeImagHist->GetBinContent(iK,iR);
    }
  }

//--------------------------------------------------------------

  delete tGTildeRealHist;
  delete tGTildeImagHist;

  tInterpHistFile->Close();
  delete tInterpHistFile;

//--------------------------------------------------------------
  fInterpolateGPU->LoadGTildeReal(fGTildeReal);
  fInterpolateGPU->LoadGTildeImag(fGTildeImag);
  fInterpolateGPU->LoadGTildeInfo(fGTildeInfo);

}

//________________________________________________________________________________________________________________
vector<vector<double> > Interpolate::BuildPairs()
{
  fPairs2dVec.clear();

  int tMaxKStarCalls = 10000;
  int tBin = 10;

  std::default_random_engine generator (std::clock());  //std::clock() is seed
  std::normal_distribution<double> tROutSource(0.,1.);
  std::normal_distribution<double> tRSideSource(0.,1.);
  std::normal_distribution<double> tRLongSource(0.,1.);

  std::uniform_int_distribution<int> tRandomKStarElement(0.0, fPairKStar3dVec[tBin].size()-1);

  TVector3* tKStar3Vec = new TVector3(0.,0.,0.);
  TVector3* tSource3Vec = new TVector3(0.,0.,0.);

  int tI;
  double tTheta, tKStarMag, tRStarMag;


  vector<double> tTempPair(2);

  for(int i=0; i<tMaxKStarCalls; i++)
  {
    tI = tRandomKStarElement(generator);
    tKStar3Vec->SetXYZ(fPairKStar3dVec[tBin][tI][1],fPairKStar3dVec[tBin][tI][2],fPairKStar3dVec[tBin][tI][3]);
      tKStarMag = tKStar3Vec->Mag();
    tSource3Vec->SetXYZ(tROutSource(generator),tRSideSource(generator),tRLongSource(generator)); //TODO: for now, spherically symmetric
      tRStarMag = tSource3Vec->Mag();
    tTheta = tKStar3Vec->Angle(*tSource3Vec);

//    tTempPair.clear();  //TODO, for some reason, this is causing an error
      tTempPair[0] = tKStarMag;
      tTempPair[1] = tRStarMag;


    fPairs2dVec.push_back(tTempPair);
  }
  delete tKStar3Vec;
  delete tSource3Vec;


  return fPairs2dVec;
}

//________________________________________________________________________________________________________________
vector<double> Interpolate::RunBilinearInterpolateSerial(vector<vector<double> > &d_PairsIn, vector<vector<double> > &d_2dVecIn)
{
  vector<double> ReturnVector(10000);

  for(int i=0; i<(int)d_PairsIn.size(); i++)
  {

    int tIdx = i;

    double aX = d_PairsIn[tIdx][0];
    double aY = d_PairsIn[tIdx][1];

    double tF = 0.;
    double tX1=0., tX2=0., tY1=0., tY2=0.;
    double tdX, tdY;

    int aNbinsX = 160;
    double aMinX = 0.0;
    double aMaxX = 0.4;

    int aNbinsY = 100;
    double aMinY = 0.0;
    double aMaxY = 10.0;

    int tXbin = GetBinNumber(aNbinsX,aMinX,aMaxX,aX);
    int tYbin = GetBinNumber(aNbinsY,aMinY,aMaxY,aY);

    double tBinWidthX = (aMaxX-aMinX)/aNbinsX;
    double tBinMinX = aMinX + tXbin*tBinWidthX;
    double tBinMaxX = aMinX + (tXbin+1)*tBinWidthX;

    double tBinWidthY = (aMaxY-aMinY)/aNbinsY;
    double tBinMinY = aMinY + tYbin*tBinWidthY;
    double tBinMaxY = aMinY + (tYbin+1)*tBinWidthY;

  //---------------------------------
/*
  if(tXbin<0 || tYbin<0) 
  {
    cout << "Error in CoulombFitter::BilinearInterpolateVector, cannot interpolate outside histogram domain" << endl;
  }
  assert(tXbin >= 0);
  assert(tYbin >= 0);
*/
  //---------------------------------

    int tQuadrant = 0; //CCW from UR 1,2,3,4
    // which quadrant of the bin (bin_P) are we in?
    tdX = tBinMaxX - aX;
    tdY = tBinMaxY - aY;

    int tBinX1, tBinX2, tBinY1, tBinY2;

    if(tdX<=tBinWidthX/2 && tdY<=tBinWidthY/2) tQuadrant = 1; //upper right
    else if(tdX>tBinWidthX/2 && tdY<=tBinWidthY/2) tQuadrant = 2; //upper left
    else if(tdX>tBinWidthX/2 && tdY>tBinWidthY/2) tQuadrant = 3; //lower left
    else if(tdX<=tBinWidthX/2 && tdY>tBinWidthY/2) tQuadrant = 4; //lower right
//  else cout << "ERROR IN BilinearInterpolateVector!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;


    switch(tQuadrant)
    {
      case 1:
        tX1 = tBinMinX + tBinWidthX/2;
        tY1 = tBinMinY + tBinWidthY/2;
        tX2 = tBinMaxX + tBinWidthX/2;
        tY2 = tBinMaxY + tBinWidthY/2;

        tBinX1 = tXbin;
        tBinX2 = tXbin+1;
        tBinY1 = tYbin;
        tBinY2 = tYbin+1;

        break;
      case 2:
        tX1 = tBinMinX - tBinWidthX/2;
        tY1 = tBinMinY + tBinWidthY/2;
        tX2 = tBinMinX + tBinWidthX/2;
        tY2 = tBinMaxY + tBinWidthY/2;

        tBinX1 = tXbin-1;
        tBinX2 = tXbin;
        tBinY1 = tYbin;
        tBinY2 = tYbin+1;

        break;
      case 3:
        tX1 = tBinMinX - tBinWidthX/2;
        tY1 = tBinMinY - tBinWidthY/2;
        tX2 = tBinMinX + tBinWidthX/2;
        tY2 = tBinMinY + tBinWidthY/2;

        tBinX1 = tXbin-1;
        tBinX2 = tXbin;
        tBinY1 = tYbin-1;
        tBinY2 = tYbin;

        break;
      case 4:
        tX1 = tBinMinX + tBinWidthX/2;
        tY1 = tBinMinY - tBinWidthY/2;
        tX2 = tBinMaxX + tBinWidthX/2;
        tY2 = tBinMinY + tBinWidthY/2;

        tBinX1 = tXbin;
        tBinX2 = tXbin+1;
        tBinY1 = tYbin-1;
        tBinY2 = tYbin;

        break;
    }

    if(tBinX1<1) tBinX1 = 1;
    if(tBinX2>aNbinsX) tBinX2=aNbinsX;
    if(tBinY1<1) tBinY1 = 1;
    if(tBinY2>aNbinsY) tBinY2=aNbinsY;

    double tQ11 = d_2dVecIn[tBinX1][tBinY1];
    double tQ12 = d_2dVecIn[tBinX1][tBinY2];
    double tQ21 = d_2dVecIn[tBinX2][tBinY1];
    double tQ22 = d_2dVecIn[tBinX2][tBinY2];

    double tD = 1.0*(tX2-tX1)*(tY2-tY1);

    tF = (1.0/tD)*(tQ11*(tX2-aX)*(tY2-aY) + tQ21*(aX-tX1)*(tY2-aY) + tQ12*(tX2-aX)*(aY-tY1) + tQ22*(aX-tX1)*(aY-tY1));

    ReturnVector[tIdx] = tF;
  }

  return ReturnVector;
}


//________________________________________________________________________________________________________________
vector<double> Interpolate::RunBilinearInterpolateParallel(td2dVec &aPairs)
{
  vector<double> tReturnVec = fInterpolateGPU->RunBilinearInterpolate(aPairs);
  return tReturnVec;
}

