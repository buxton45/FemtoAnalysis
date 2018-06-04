/* Interpolator.cxx */

#include "Interpolator.h"

#ifdef __ROOT__
ClassImp(Interpolator)
#endif

//________________________________________________________________________________________________________________
Interpolator::Interpolator()
{ /*no-op*/}

//________________________________________________________________________________________________________________
Interpolator::~Interpolator()
{ /*no-op*/}


//________________________________________________________________________________________________________________
int Interpolator::GetBinNumber(double aBinSize, int aNbins, double aValue)
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
int Interpolator::GetBinNumber(int aNbins, double aMin, double aMax, double aValue)
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
int Interpolator::GetBinNumber(double aBinWidth, double aMin, double aMax, double aValue)
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
double Interpolator::LinearInterpolate(TH1* a1dHisto, double aX)
{
  if(a1dHisto->GetBuffer()) a1dHisto->BufferEmpty();  //not sure what this is all about

  int tXbin = a1dHisto->FindBin(aX);
  double tX0, tX1, tY0, tY1;

  //TODO: These allows evaluation in underflow and overflow bins, not sure I like this
  if(aX <= a1dHisto->GetBinCenter(1)) return a1dHisto->GetBinContent(1);
  else if( aX >= a1dHisto->GetBinCenter(a1dHisto->GetNbinsX()) ) return a1dHisto->GetBinContent(a1dHisto->GetNbinsX()); 

  else
  {
    if(aX <= a1dHisto->GetBinCenter(tXbin))
    {
      tY0 = a1dHisto->GetBinContent(tXbin-1);
      tX0 = a1dHisto->GetBinCenter(tXbin-1);
      tY1 = a1dHisto->GetBinContent(tXbin);
      tX1 = a1dHisto->GetBinCenter(tXbin);
    }
    else
    {
      tY0 = a1dHisto->GetBinContent(tXbin);
      tX0 = a1dHisto->GetBinCenter(tXbin);
      tY1 = a1dHisto->GetBinContent(tXbin+1);
      tX1 = a1dHisto->GetBinCenter(tXbin+1);
    }
    return tY0 + (aX-tX0)*((tY1-tY0)/(tX1-tX0));
  }
}

//________________________________________________________________________________________________________________
double Interpolator::BilinearInterpolate(TH2* a2dHisto, double aX, double aY)
{
  double tF = 0.;
  double tX1=0., tX2=0., tY1=0., tY2=0.;
  double tdX, tdY;

  TAxis* tXaxis = a2dHisto->GetXaxis();
  TAxis* tYaxis = a2dHisto->GetYaxis();

  int tXbin = tXaxis->FindBin(aX);
  int tYbin = tYaxis->FindBin(aY);

  //---------------------------------
  if(tXbin<1 || tXbin>a2dHisto->GetNbinsX() || tYbin<1 || tYbin>a2dHisto->GetNbinsY()) 
  {
    cout << "Error in Interpolator::BilinearInterpolate, cannot interpolate outside histogram domain" << endl;
  }
  assert(tXbin>0);
  assert(tXbin<=a2dHisto->GetNbinsX());
  assert(tYbin>0);
  assert(tYbin<=a2dHisto->GetNbinsY());
  //---------------------------------

  int tQuadrant = 0; //CCW from UR 1,2,3,4
  // which quadrant of the bin (bin_P) are we in?
  tdX = tXaxis->GetBinUpEdge(tXbin) - aX;
  tdY = tYaxis->GetBinUpEdge(tYbin) - aY;

  if(tdX<=tXaxis->GetBinWidth(tXbin)/2 && tdY<=tYaxis->GetBinWidth(tYbin)/2) tQuadrant = 1; //upper right
  if(tdX>tXaxis->GetBinWidth(tXbin)/2 && tdY<=tYaxis->GetBinWidth(tYbin)/2) tQuadrant = 2; //upper left
  if(tdX>tXaxis->GetBinWidth(tXbin)/2 && tdY>tYaxis->GetBinWidth(tYbin)/2) tQuadrant = 3; //lower left
  if(tdX<=tXaxis->GetBinWidth(tXbin)/2 && tdY>tYaxis->GetBinWidth(tYbin)/2) tQuadrant = 4; //lower right

  switch(tQuadrant)
  {
    case 1:
      tX1 = tXaxis->GetBinCenter(tXbin);
      tY1 = tYaxis->GetBinCenter(tYbin);
      tX2 = tXaxis->GetBinCenter(tXbin+1);
      tY2 = tYaxis->GetBinCenter(tYbin+1);
      break;
    case 2:
      tX1 = tXaxis->GetBinCenter(tXbin-1);
      tY1 = tYaxis->GetBinCenter(tYbin);
      tX2 = tXaxis->GetBinCenter(tXbin);
      tY2 = tYaxis->GetBinCenter(tYbin+1);
      break;
    case 3:
      tX1 = tXaxis->GetBinCenter(tXbin-1);
      tY1 = tYaxis->GetBinCenter(tYbin-1);
      tX2 = tXaxis->GetBinCenter(tXbin);
      tY2 = tYaxis->GetBinCenter(tYbin);
      break;
    case 4:
      tX1 = tXaxis->GetBinCenter(tXbin);
      tY1 = tYaxis->GetBinCenter(tYbin-1);
      tX2 = tXaxis->GetBinCenter(tXbin+1);
      tY2 = tYaxis->GetBinCenter(tYbin);
      break;
  }

  int tBinX1 = tXaxis->FindBin(tX1);
  if(tBinX1<1) tBinX1 = 1;

  int tBinX2 = tXaxis->FindBin(tX2);
  if(tBinX2>a2dHisto->GetNbinsX()) tBinX2=a2dHisto->GetNbinsX();

  int tBinY1 = tYaxis->FindBin(tY1);
  if(tBinY1<1) tBinY1 = 1;

  int tBinY2 = tYaxis->FindBin(tY2);
  if(tBinY2>a2dHisto->GetNbinsY()) tBinY2=a2dHisto->GetNbinsY();

  int tBinQ22 = a2dHisto->GetBin(tBinX2,tBinY2);
  int tBinQ12 = a2dHisto->GetBin(tBinX1,tBinY2);
  int tBinQ11 = a2dHisto->GetBin(tBinX1,tBinY1);
  int tBinQ21 = a2dHisto->GetBin(tBinX2,tBinY1);

  double tQ11 = a2dHisto->GetBinContent(tBinQ11);
  double tQ12 = a2dHisto->GetBinContent(tBinQ12);
  double tQ21 = a2dHisto->GetBinContent(tBinQ21);
  double tQ22 = a2dHisto->GetBinContent(tBinQ22);

  double tD = 1.0*(tX2-tX1)*(tY2-tY1);

  tF = (1.0/tD)*(tQ11*(tX2-aX)*(tY2-aY) + tQ21*(aX-tX1)*(tY2-aY) + tQ12*(tX2-aX)*(aY-tY1) + tQ22*(aX-tX1)*(aY-tY1));

  return tF;
}


//________________________________________________________________________________________________________________
double Interpolator::BilinearInterpolateVector(vector<vector<double> > &a2dVec, double aX, int aNbinsX, double aMinX, double aMaxX, double aY, int aNbinsY, double aMinY, double aMaxY)
{
  //NOTE: THIS IS SLOWER THAN BilinearInterpolate, but a method like this may be necessary for parallelization

  double tF = 0.;
  double tX1=0., tX2=0., tY1=0., tY2=0.;
  double tdX, tdY;

  int tXbin = GetBinNumber(aNbinsX,aMinX,aMaxX,aX);
  int tYbin = GetBinNumber(aNbinsY,aMinY,aMaxY,aY);

  double tBinWidthX = (aMaxX-aMinX)/aNbinsX;
  double tBinMinX = aMinX + tXbin*tBinWidthX;
  double tBinMaxX = aMinX + (tXbin+1)*tBinWidthX;

  double tBinWidthY = (aMaxY-aMinY)/aNbinsY;
  double tBinMinY = aMinY + tYbin*tBinWidthY;
  double tBinMaxY = aMinY + (tYbin+1)*tBinWidthY;

  //---------------------------------
  if(tXbin<0 || tYbin<0) 
  {
    cout << "Error in Interpolator::BilinearInterpolateVector, cannot interpolate outside histogram domain" << endl;
  }
  assert(tXbin >= 0);
  assert(tYbin >= 0);

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
  else cout << "ERROR IN BilinearInterpolateVector!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;


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

  double tQ11 = a2dVec[tBinX1][tBinY1];
  double tQ12 = a2dVec[tBinX1][tBinY2];
  double tQ21 = a2dVec[tBinX2][tBinY1];
  double tQ22 = a2dVec[tBinX2][tBinY2];

  double tD = 1.0*(tX2-tX1)*(tY2-tY1);

  tF = (1.0/tD)*(tQ11*(tX2-aX)*(tY2-aY) + tQ21*(aX-tX1)*(tY2-aY) + tQ12*(tX2-aX)*(aY-tY1) + tQ22*(aX-tX1)*(aY-tY1));

  return tF;
}

//________________________________________________________________________________________________________________
double Interpolator::TrilinearInterpolate(TH3* a3dHisto, double aX, double aY, double aZ)
{
  TAxis* tXaxis = a3dHisto->GetXaxis();
  TAxis* tYaxis = a3dHisto->GetYaxis();
  TAxis* tZaxis = a3dHisto->GetZaxis();

  //--------------------------------
  int ubx = tXaxis->FindBin(aX);
  if( aX < tXaxis->GetBinCenter(ubx) ) ubx -= 1;
  int obx = ubx + 1;

  int uby = tYaxis->FindBin(aY);
  if( aY < tYaxis->GetBinCenter(uby) ) uby -= 1;
  int oby = uby + 1;

  int ubz = tZaxis->FindBin(aZ);
  if( aZ < tZaxis->GetBinCenter(ubz) ) ubz -= 1;
  int obz = ubz + 1;

  //--------------------------------
  //make sure (aX,aY,aZ) is within the limits, so I can interpolate
  if(ubx<=0 || uby<=0 || ubz<=0 || obx>tXaxis->GetNbins() || oby>tYaxis->GetNbins() || obz>tZaxis->GetNbins())
  {
    cout << "Error in Interpolator::TrilinearInterpolate, cannot interpolate outside histogram domain" << endl;

    cout << "aX = " << aX << "\taY = " << aY << "\taZ = " << aZ << endl;
    cout << "ubx = " << ubx << "\tuby = " << uby << "\tubz = " << ubz << endl;
    cout << "obx = " << obx << "\toby = " << oby << "\tobz = " << obz << endl;
  }
  assert(ubx>0);
  assert(uby>0);
  assert(ubz>0);
  assert(obx<=tXaxis->GetNbins());
  assert(oby<=tYaxis->GetNbins());
  assert(obz<=tZaxis->GetNbins());
  //--------------------------------

  double xw = tXaxis->GetBinCenter(obx) - tXaxis->GetBinCenter(ubx);
  double yw = tYaxis->GetBinCenter(oby) - tYaxis->GetBinCenter(uby);
  double zw = tZaxis->GetBinCenter(obz) - tZaxis->GetBinCenter(ubz);

  double xd = (aX - tXaxis->GetBinCenter(ubx))/xw;
  double yd = (aY - tYaxis->GetBinCenter(uby))/yw;
  double zd = (aZ - tZaxis->GetBinCenter(ubz))/zw;

  double v[] = { a3dHisto->GetBinContent(ubx, uby, ubz), a3dHisto->GetBinContent(ubx, uby, obz),
                 a3dHisto->GetBinContent(ubx, oby, ubz), a3dHisto->GetBinContent(ubx, oby, obz),
                 a3dHisto->GetBinContent(obx, uby, ubz), a3dHisto->GetBinContent(obx, uby, obz),
                 a3dHisto->GetBinContent(obx, oby, ubz), a3dHisto->GetBinContent(obx, oby, obz) };

  double i1 = v[0]*(1.-zd) + v[1]*zd;
  double i2 = v[2]*(1.-zd) + v[3]*zd;
  double j1 = v[4]*(1.-zd) + v[5]*zd;
  double j2 = v[6]*(1.-zd) + v[7]*zd;

  double w1 = i1*(1.-yd) + i2*yd;
  double w2 = j1*(1.-yd) + j2*yd;

  double tResult = w1*(1.-xd) + w2*xd;

  return tResult;
}



//________________________________________________________________________________________________________________
double Interpolator::QuadrilinearInterpolate(THn* a4dHisto, double aT, double aX, double aY, double aZ)
{
  TAxis* tTaxis = a4dHisto->GetAxis(0);
  TAxis* tXaxis = a4dHisto->GetAxis(1);
  TAxis* tYaxis = a4dHisto->GetAxis(2);
  TAxis* tZaxis = a4dHisto->GetAxis(3);

  //--------------------------------

  int ubt = tTaxis->FindBin(aT);
  if( aT < tTaxis->GetBinCenter(ubt) ) ubt -= 1;
  int obt = ubt + 1;

  int ubx = tXaxis->FindBin(aX);
  if( aX < tXaxis->GetBinCenter(ubx) ) ubx -= 1;
  int obx = ubx + 1;

  int uby = tYaxis->FindBin(aY);
  if( aY < tYaxis->GetBinCenter(uby) ) uby -= 1;
  int oby = uby + 1;

  int ubz = tZaxis->FindBin(aZ);
  if( aZ < tZaxis->GetBinCenter(ubz) ) ubz -= 1;
  int obz = ubz + 1;

  //--------------------------------
  //make sure (aT,aX,aY,aZ) is within the limits, so I can interpolate
  if(ubt<=0 || ubx<=0 || uby<=0 || ubz<=0 || obt>tTaxis->GetNbins() || obx>tXaxis->GetNbins() || oby>tYaxis->GetNbins() || obz>tZaxis->GetNbins())
  {
    cout << "Error in Interpolator::QuadrilinearInterpolate, cannot interpolate outside histogram domain" << endl;
  }
  assert(ubt>0);
  assert(ubx>0);
  assert(uby>0);
  assert(ubz>0);
  assert(obt<=tTaxis->GetNbins());
  assert(obx<=tXaxis->GetNbins());
  assert(oby<=tYaxis->GetNbins());
  assert(obz<=tZaxis->GetNbins());
  //--------------------------------

  double tw = tTaxis->GetBinCenter(obt) - tTaxis->GetBinCenter(ubt);
  double xw = tXaxis->GetBinCenter(obx) - tXaxis->GetBinCenter(ubx);
  double yw = tYaxis->GetBinCenter(oby) - tYaxis->GetBinCenter(uby);
  double zw = tZaxis->GetBinCenter(obz) - tZaxis->GetBinCenter(ubz);

  double td = (aT - tTaxis->GetBinCenter(ubt))/tw;
  double xd = (aX - tXaxis->GetBinCenter(ubx))/xw;
  double yd = (aY - tYaxis->GetBinCenter(uby))/yw;
  double zd = (aZ - tZaxis->GetBinCenter(ubz))/zw;

  //--------------------------------
  //TODO these probably don't need to all be made at the same time
  // i.e., I can have a general int tBin, and put the appropriate numbers in there we needed
  int tBin0000[4] = {ubt,ubx,uby,ubz};
  int tBin1000[4] = {obt,ubx,uby,ubz};

  int tBin0100[4] = {ubt,obx,uby,ubz};
  int tBin1100[4] = {obt,obx,uby,ubz};

  int tBin0010[4] = {ubt,ubx,oby,ubz};
  int tBin1010[4] = {obt,ubx,oby,ubz};

  int tBin0110[4] = {ubt,obx,oby,ubz};
  int tBin1110[4] = {obt,obx,oby,ubz};

  int tBin0001[4] = {ubt,ubx,uby,obz};
  int tBin1001[4] = {obt,ubx,uby,obz};

  int tBin0101[4] = {ubt,obx,uby,obz};
  int tBin1101[4] = {obt,obx,uby,obz};

  int tBin0011[4] = {ubt,ubx,oby,obz};
  int tBin1011[4] = {obt,ubx,oby,obz};

  int tBin0111[4] = {ubt,obx,oby,obz};
  int tBin1111[4] = {obt,obx,oby,obz};

  //--------------------------------

  //Interpolate along t
  double tC000 = a4dHisto->GetBinContent(tBin0000)*(1.-td) + a4dHisto->GetBinContent(tBin1000)*td;
  double tC100 = a4dHisto->GetBinContent(tBin0100)*(1.-td) + a4dHisto->GetBinContent(tBin1100)*td;

  double tC010 = a4dHisto->GetBinContent(tBin0010)*(1.-td) + a4dHisto->GetBinContent(tBin1010)*td;
  double tC110 = a4dHisto->GetBinContent(tBin0110)*(1.-td) + a4dHisto->GetBinContent(tBin1110)*td;

  double tC001 = a4dHisto->GetBinContent(tBin0001)*(1.-td) + a4dHisto->GetBinContent(tBin1001)*td;
  double tC101 = a4dHisto->GetBinContent(tBin0101)*(1.-td) + a4dHisto->GetBinContent(tBin1101)*td;

  double tC011 = a4dHisto->GetBinContent(tBin0011)*(1.-td) + a4dHisto->GetBinContent(tBin1011)*td;
  double tC111 = a4dHisto->GetBinContent(tBin0111)*(1.-td) + a4dHisto->GetBinContent(tBin1111)*td;

  //--------------------------------

  //Interpolate along x
  double tC00 = tC000*(1.-xd) + tC100*xd;
  double tC10 = tC010*(1.-xd) + tC110*xd;
  double tC01 = tC001*(1.-xd) + tC101*xd;
  double tC11 = tC011*(1.-xd) + tC111*xd;

  //--------------------------------

  //Interpolate along y
  double tC0 = tC00*(1.-yd) + tC10*yd;
  double tC1 = tC01*(1.-yd) + tC11*yd;

  //--------------------------------

  //Interpolate along z
  double tC = tC0*(1.-zd) + tC1*zd;

  return tC;
}

