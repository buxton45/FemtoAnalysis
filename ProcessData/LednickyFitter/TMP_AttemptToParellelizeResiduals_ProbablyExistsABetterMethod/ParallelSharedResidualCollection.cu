/* ParallelSharedResidualCollection.cu */

#include "ParallelSharedResidualCollection.h"


//________________________________________________________________________________________________________________
__global__ void GetTransformedCfValues(double* aCf, int aMatrixOffset, double *g_odata, int aOutputOffset)
{
  extern __shared__ double sdata2[][2];

  unsigned int tid = threadIdx.x;
  unsigned int tDaughterBin = blockIdx.x;
  unsigned int tParentBin = aMatrixOffset + tid;

  double tValue = aCf[tDaughterBin]*d_fTransformMatrices[tParentBin];
  double tNorm = d_fTransformMatrices[tParentBin];

  sdata2[tid][0] = tValue;
  sdata2[tid][1] = tNorm;

  __syncthreads();

  for(unsigned int s=1; s<blockDim.x; s*=2)
  {
    int index = 2*s*tid;

    if(index < blockDim.x)
    {
      sdata2[index][0] += sdata2[index+s][0];
      sdata2[index][1] += sdata2[index+s][1];
    }
    __syncthreads();
  }


  if(tid == 0) 
  {
    g_odata[blockIdx.x + aOutputOffset] = sdata2[0][0]/sdata2[0][1];
  }
}









//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________


ParallelSharedResidualCollection::ParallelSharedResidualCollection(int aNThreadsPerBlock, int aNBlocks):
  fNThreadsPerBlock(aNThreadsPerBlock),
  fNBlocks(aNBlocks),
  fTransformMatrices()
{
  cudaSetDeviceFlags(cudaDeviceMapHost);
}

//________________________________________________________________________________________________________________
ParallelSharedResidualCollection::~ParallelSharedResidualCollection()
{
  checkCudaErrors(cudaFree(d_fBinInfoTransformMatrix));
  checkCudaErrors(cudaFree(d_fTransformMatrices));
}


//________________________________________________________________________________________________________________
void ParallelSharedResidualCollection::LoadTransformMatrix(td2dVec &aTransformMatrix, BinInfoTransformMatrix &aBinInfo)
{
  fTransformMatrices.push_back(aTransformMatrix);
  fBinInfoTransformMatrices.push_back(aBinInfo);
  fBinInfoTransformMatrices[fBinInfoTransformMatrices.size()-1].global2DMatrixPosition = fBinInfoTransformMatrices.size()-1;

  assert(fTransformMatrices.size() == fBinInfoTransformMatrices.size());
  if(fTransformMatrices.size() > 1)
  {
    int tIndex1 = fTransformMatrices.size()-1;
    int tIndex2 = tIndex1-1;

    assert(fBinInfoTransformMatrices[tIndex1].nBinsX == fBinInfoTransformMatrices[tIndex2].nBinsX);
    assert(fBinInfoTransformMatrices[tIndex1].nBinsY == fBinInfoTransformMatrices[tIndex2].nBinsY);

    assert(fBinInfoTransformMatrices[tIndex1].binWidthX == fBinInfoTransformMatrices[tIndex2].binWidthX);
    assert(fBinInfoTransformMatrices[tIndex1].binWidthY == fBinInfoTransformMatrices[tIndex2].binWidthY);

    assert(fBinInfoTransformMatrices[tIndex1].minX == fBinInfoTransformMatrices[tIndex2].minX);
    assert(fBinInfoTransformMatrices[tIndex1].maxX == fBinInfoTransformMatrices[tIndex2].maxX);

    assert(fBinInfoTransformMatrices[tIndex1].minY == fBinInfoTransformMatrices[tIndex2].minY);
    assert(fBinInfoTransformMatrices[tIndex1].maxY == fBinInfoTransformMatrices[tIndex2].maxY);
  }
}

//________________________________________________________________________________________________________________
void ParallelSharedResidualCollection::SetOffsets()
{
  int tNbinsX = fBinInfoTransformMatrices[0].nBinsX;
  int tNbinsY = fBinInfoTransformMatrices[0].nBinsY;

  for(unsigned int iMatrix=0; iMatrix<fTransformMatrices.size(); iMatrix++)
  {
    fBinInfoTransformMatrices[iMatrix].globalOffset = iMatrix*tNbinsX*tNbinsY;
  }
}


//________________________________________________________________________________________________________________
void ParallelSharedResidualCollection::BuildDeviceTransformMatrices()
{
  SetOffsets();

  int tNMatrices = fTransformMatrices.size();
  int tNbinsX = fBinInfoTransformMatrices[0].nBinsX;
  int tNbinsY = fBinInfoTransformMatrices[0].nBinsY;

  int tSize = tNMatrices*tNbinsX*tNbinsY*sizeof(double);
  checkCudaErrors(cudaMallocManaged(&d_fTransformMatrices, tSize));

  int tIndex;
  for(int iMatrix=0; iMatrix<tNMatrices; iMatrix++)
  {
    for(int iX=0; iX<tNbinsX; iX++)
    {
      for(int iY=0; iY<tNbinsY; iY++)
      {
        tIndex = iY + iX*tNbinsY + iMatrix*tNbinsY*tNbinsX;
        d_fTransformMatrices[tIndex] = fTransformMatrices[iMatrix][iX][iY];
      }
    }
  }

  //-------------------------
  checkCudaErrors(cudaMallocManaged(&d_fBinInfoTransformMatrix, sizeof(BinInfoTransformMatrix)));

  d_fBinInfoTransformMatrix->daughterAnalysisType = fBinInfoTransformMatrices[0].daughterAnalysisType;
  d_fBinInfoTransformMatrix->parentResidualType = fBinInfoTransformMatrices[0].parentResidualType;
  d_fBinInfoTransformMatrix->centralityType = fBinInfoTransformMatrices[0].centralityType;
  d_fBinInfoTransformMatrix->global2DMatrixPosition = -1;   //Meaningless here 
  d_fBinInfoTransformMatrix->globalOffset = -1;             //Meaningless here 

  d_fBinInfoTransformMatrix->nBinsX = fBinInfoTransformMatrices[0].nBinsX;
  d_fBinInfoTransformMatrix->nBinsY = fBinInfoTransformMatrices[0].nBinsY;

  d_fBinInfoTransformMatrix->binWidthX = fBinInfoTransformMatrices[0].binWidthX;
  d_fBinInfoTransformMatrix->binWidthY = fBinInfoTransformMatrices[0].binWidthY;

  d_fBinInfoTransformMatrix->minX = fBinInfoTransformMatrices[0].minX;
  d_fBinInfoTransformMatrix->maxX = fBinInfoTransformMatrices[0].maxX;
  d_fBinInfoTransformMatrix->minY = fBinInfoTransformMatrices[0].minY;
  d_fBinInfoTransformMatrix->maxY = fBinInfoTransformMatrices[0].maxY;
}


//________________________________________________________________________________________________________________
int ParallelSharedResidualCollection::GetOffset(AnalysisType aAnType, AnalysisType aResType, CentralityType aCentType)
{
  int tOffset = -1;
  for(unsigned int i=0; i<fBinInfoTransformMatrices.size(); i++)
  {
    if(aAnType==fBinInfoTransformMatrices[i].daughterAnalysisType &&
       aResType==fBinInfoTransformMatrices[i].parentResidualType &&
       aCentType==fBinInfoTransformMatrices[i].centralityType) return i;
  }
  assert(tOffset>-1);
  return tOffset;
}



//________________________________________________________________________________________________________________
td2dVec ParallelSharedResidualCollection::GetTransformedResidualCorrelation(AnalysisType aAnType, CentralityType aCentType, td2dVec &aCfs)
{
  const int tNStreams = fTransformMatrices.size();

  int tNbins = aCfs[0].size();
  int tSizeOutput = tNbins*tNbins*tNStreams*sizeof(double);
  int tSizeShared = tNbins*sizeof(double);
  tSizeShared *= 2;  //to account for value and normalization

  double * h_Cf;
  checkCudaErrors(cudaMallocManaged(&h_Cf, tSizeOutput));  

  cudaStream_t tStreams[tNStreams];

  for(int i=0; i<tNStreams; i++)
  {
    cudaStreamCreate(&tStreams[i]);
  }

  int tMatrixOffset = -1;
  int tOutputOffset = -1;
  for(int i=0; i<tNStreams; i++)
  {
    tMatrixOffset = GetOffset(aAnType, static_cast<AnalysisType>(i), aCentType);
    tOutputOffset = i*tNbins;
    GetTransformedCfValues<<<tNbins,tNbins,tSizeShared,tStreams[i]>>>(aCfs[i].data(), tMatrixOffset, h_Cf, tOutputOffset);
  }

  td2dVec tReturnVec(tNStreams, td1dVec(tNbins, 0.));
  for(int iBin=0; iBin<tNbins; iBin++)
  {
    for(int iStream=0; iStream<tNStreams; iStream++)
    {
      tReturnVec[iStream][iBin] = h_Cf[iStream*tNbins + iBin];
    }
  }

  return tReturnVec;
}





