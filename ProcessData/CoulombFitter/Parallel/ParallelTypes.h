///////////////////////////////////////////////////////////////////////////
// ParallelTypes:                                                        //
//                                                                       //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

#ifndef PARALLELTYPES_H
#define PARALLELTYPES_H

#include <vector>
using std::vector;

  //-------enum types-----------------------------------------
  enum InterpType {kGTilde=0, kHyperGeo1F1=1, kScattLen=2};
  enum InterpAxisType {kKaxis=0, kRaxis=1, kThetaaxis=2, kReF0axis=3, kImF0axis=4, kD0axis=5};
  //----------------------------------------------------------



  //-------struct definitions-----------------------------------------
  struct BinInfoKStar
  {
    int nBinsK;
    double minK, maxK, binWidthK;
    int nPairsPerBin[100] = {};  //TODO for now, make this bigger than I'll ever need
    int binOffset[100] = {};
  };

  //------------------
  struct BinInfoGTilde
  {
    int nBinsK, nBinsR;
    double binWidthK, binWidthR;
    double minK, maxK, minR, maxR;
    double minInterpK, maxInterpK, minInterpR, maxInterpR;
  };

//------------------
  struct BinInfoHyperGeo1F1
  {
    int nBinsK, nBinsR, nBinsTheta;
    double binWidthK, binWidthR, binWidthTheta;
    double minK, maxK, minR, maxR, minTheta, maxTheta;
    double minInterpK, maxInterpK, minInterpR, maxInterpR, minInterpTheta, maxInterpTheta;
  };

//------------------
  struct BinInfoScattLen
  {
    int nBinsReF0, nBinsImF0, nBinsD0, nBinsK;
    double binWidthReF0, binWidthImF0, binWidthD0, binWidthK;
    double minReF0, maxReF0, minImF0, maxImF0, minD0, maxD0, minK, maxK;
    double minInterpReF0, maxInterpReF0, minInterpImF0, maxInterpImF0, minInterpD0, maxInterpD0, minInterpK, maxInterpK;
  };
  //------------------------------------------------------------------


  //-------typedefs-----------------------------------------
  typedef vector<double> td1dVec;
  typedef vector<vector<double> > td2dVec;
  typedef vector<vector<vector<double> > > td3dVec;
  typedef vector<vector<vector<vector<double> > > > td4dVec;
  //--------------------------------------------------------

  extern const double parallel_hbarc;
  extern const double parallel_gBohrRadiusXiK;

#endif

