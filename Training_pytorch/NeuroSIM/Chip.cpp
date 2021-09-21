/*******************************************************************************
* Copyright (c) 2015-2017
* School of Electrical, Computer and Energy Engineering, Arizona State University
* PI: Prof. Shimeng Yu
* All rights reserved.
* 
* This source code is part of NeuroSim - a device-circuit-algorithm framework to benchmark 
* neuro-inspired architectures with synaptic devices(e.g., SRAM and emerging non-volatile memory). 
* Copyright of the model is maintained by the developers, and the model is distributed under 
* the terms of the Creative Commons Attribution-NonCommercial 4.0 International Public License 
* http://creativecommons.org/licenses/by-nc/4.0/legalcode.
* The source code is free and you can redistribute and/or modify it
* by providing that the following conditions are met:
* 
*  1) Redistributions of source code must retain the above copyright notice,
*     this list of conditions and the following disclaimer.
* 
*  2) Redistributions in binary form must reproduce the above copyright notice,
*     this list of conditions and the following disclaimer in the documentation
*     and/or other materials provided with the distribution.
* 
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
* 
* Developer list: 
*   Pai-Yu Chen	    Email: pchen72 at asu dot edu 
*                    
*   Xiaochen Peng   Email: xpeng15 at asu dot edu
********************************************************************************/

#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <vector>
#include <sstream>
#include "MaxPooling.h"
#include "Sigmoid.h"
#include "BitShifter.h"
#include "AdderTree.h"
#include "Buffer.h"
#include "HTree.h"
#include "DRAM.h"
#include "ProcessingUnit.h"
#include "Tile.h"
#include "WeightGradientUnit.h"
#include "constant.h"
#include "formula.h"
#include "Param.h"
#include "Chip.h"
#include "Adder.h"

using namespace std;

extern Param *param;
double globalBusWidth = 0;
int numBufferCore = 0;

/*** Circuit Modules ***/
Buffer *globalBuffer;
HTree *GhTree;
AdderTree *Gaccumulation;
Sigmoid *Gsigmoid;
BitShifter *GreLu;
MaxPooling *maxPool;
DRAM *dRAM;
WeightGradientUnit *weightGradientUnit;
Adder *gradientAccum;

vector<int> ChipDesignInitialize(InputParameter& inputParameter, Technology& tech, MemCell& cell, bool pip, const vector<vector<double> > &netStructure,
					double *maxPESizeNM, double *maxTileSizeCM, double *numPENM){

	globalBuffer = new Buffer(inputParameter, tech, cell);
	GhTree = new HTree(inputParameter, tech, cell);
	Gaccumulation = new AdderTree(inputParameter, tech, cell);
	Gsigmoid = new Sigmoid(inputParameter, tech, cell);
	GreLu = new BitShifter(inputParameter, tech, cell);
	maxPool = new MaxPooling(inputParameter, tech, cell);
	dRAM = new DRAM(inputParameter, tech, cell);
	weightGradientUnit = new WeightGradientUnit(inputParameter, tech, cell);
	gradientAccum = new Adder(inputParameter, tech, cell);
	
	int numRowPerSynapse, numColPerSynapse;
	numRowPerSynapse = param->numRowPerSynapse;
	numColPerSynapse = param->numColPerSynapse;
	
	double numLayer, minCube;
	
	// get information of network structure
	numLayer = netStructure.size();
	
	*maxPESizeNM = 0;
	*maxTileSizeCM = 0;
	*numPENM = 0;

	vector<int> markNM;
	if (param->novelMapping) {
		// define number of PE in COV layers
		int most = 0;
		int numPE = 0;
		for (int i=0; i<numLayer; i++) {
			int temp = netStructure[i][3]*netStructure[i][4];
			int count = 1;
			for (int j=0; j<numLayer; j++) {
				if (temp == netStructure[j][3]*netStructure[j][4]) {
					count ++;
				}
				if (most < count) {
					most = count;
					numPE = temp;
				}
			}
		}
		*numPENM = numPE;
		// mark the layers that use novel mapping
		for (int i=0; i<numLayer; i++) {
			
			if ((netStructure[i][3]*netStructure[i][4]== (*numPENM))
				// large Cov layers use novel mapping
				&&(netStructure[i][2]*netStructure[i][3]*netStructure[i][4]*numRowPerSynapse >= param->numRowSubArray)) {
				markNM.push_back(1);
				minCube = pow(2, ceil((double) log2((double) netStructure[i][5]*(double) numColPerSynapse) ) );
				*maxPESizeNM = max(minCube, (*maxPESizeNM));
			} else {
				// small Cov layers and FC layers use conventional mapping
				markNM.push_back(0);
				minCube = pow(2, ceil((double) log2((double) netStructure[i][5]*(double) numColPerSynapse) ) );
				*maxTileSizeCM = max(minCube, (*maxTileSizeCM));
			}
		}
	} else {
		// all layers use conventional mapping
		for (int i=0; i<numLayer; i++) {
			markNM.push_back(0);
			minCube = pow(2, ceil((double) log2((double) netStructure[i][5]*(double) numColPerSynapse) ) );
			*maxTileSizeCM = max(minCube, (*maxTileSizeCM));
		}
	}
	
	// for pipeline system
	vector<int> pipelineSpeedUp;
	if (param->pipeline) {
		// find max and min IFM size --> define how much the system can be speed-up
		int maxIFMSize = netStructure[0][0];
		int minIFMSize = maxIFMSize;
		for (int i=0; i<numLayer; i++) {
			if (netStructure[i][3] != 1) {
				int thisIFMSize = netStructure[i][0];
				if (thisIFMSize < minIFMSize) {
					minIFMSize = thisIFMSize;
				}
			}
		}
		
		// justify the speed-up degree is necessary
		int maxSpeedUpDegree = int(maxIFMSize/minIFMSize);
		if (maxSpeedUpDegree < param->speedUpDegree) {
			cout << "User assigned speed-up degree is larger than the upper bound (where no idle period during the whole process) " << endl;
			param->speedUpDegree = maxSpeedUpDegree;
			cout << "The speed-up degree is auto-assigned as the upper bound (where no idle period during the whole process) " << endl;
		}
		// define the pipeline speed-up
		int boundIFMSize = ceil((double) maxIFMSize/(param->speedUpDegree));
		for (int i=0; i<numLayer; i++) {
			int speedUp = ceil((double) pow((netStructure[i][0]/boundIFMSize), 2));
			pipelineSpeedUp.push_back(speedUp);
		}
	}
	
	if (pip) {
		return pipelineSpeedUp;
	} else {
		return markNM;
	}
}


vector<vector<double> > ChipFloorPlan(bool findNumTile, bool findUtilization, bool findSpeedUp, const vector<vector<double> > &netStructure, const vector<int > &markNM, 
					double maxPESizeNM, double maxTileSizeCM, double numPENM, const vector<int> &pipelineSpeedUp,
					double *desiredNumTileNM, double *desiredPESizeNM, double *desiredNumTileCM, double *desiredTileSizeCM, double *desiredPESizeCM, int *numTileRow, int *numTileCol) {
	
	
	int numRowPerSynapse, numColPerSynapse;
	numRowPerSynapse = param->numRowPerSynapse;
	numColPerSynapse = param->numColPerSynapse;
	
	double maxUtilizationNM = 0;
	double maxUtilizationCM = 0;
	
	vector<vector<double> > peDup;
	vector<vector<double> > subArrayDup;
	vector<vector<double> > numTileEachLayer;
	vector<vector<double> > utilizationEachLayer;
	vector<vector<double> > speedUpEachLayer;
	
	*desiredNumTileNM = 0;
	*desiredPESizeNM = 0;
	*desiredNumTileCM = 0;
	*desiredTileSizeCM = 0;
	*desiredPESizeCM = 0;
	*numTileRow = 0;
	*numTileCol = 0;

	if (param->novelMapping) {		// Novel Mapping
		if (maxPESizeNM < 2*param->numRowSubArray) {
			cout << "ERROR: SubArray Size is too large, which break the chip hierarchey, please decrease the SubArray size! " << endl;
		}else{
		
			/*** Tile Design ***/
			*desiredPESizeNM = MAX(maxPESizeNM, 2*param->numRowSubArray);
			vector<double> initialDesignNM;
			initialDesignNM = TileDesignNM((*desiredPESizeNM), markNM, netStructure, numRowPerSynapse, numColPerSynapse, numPENM);
			*desiredNumTileNM = initialDesignNM[0];
			for (double thisPESize = MAX(maxPESizeNM, 2*param->numRowSubArray); thisPESize> 2*param->numRowSubArray; thisPESize/=2) {
				// for layers use novel mapping
				double thisUtilization = 0;
				vector<double> thisDesign;
				thisDesign = TileDesignNM(thisPESize, markNM, netStructure, numRowPerSynapse, numColPerSynapse, numPENM);
				thisUtilization = thisDesign[2];
				if (thisUtilization > maxUtilizationNM) {
					maxUtilizationNM = thisUtilization;
					*desiredPESizeNM = thisPESize;
					*desiredNumTileNM = thisDesign[0];
				}
			}
			*desiredTileSizeCM = MAX(maxTileSizeCM, 4*param->numRowSubArray);
			vector<double> initialDesignCM;
			initialDesignCM = TileDesignCM((*desiredTileSizeCM), markNM, netStructure, numRowPerSynapse, numColPerSynapse);
			*desiredNumTileCM = initialDesignCM[0];
			for (double thisTileSize = MAX(maxTileSizeCM, 4*param->numRowSubArray); thisTileSize > 4*param->numRowSubArray; thisTileSize/=2) {
				// for layers use conventional mapping
				double thisUtilization = 0;
				vector<double> thisDesign;
				thisDesign = TileDesignCM(thisTileSize, markNM, netStructure, numRowPerSynapse, numColPerSynapse);
				thisUtilization = thisDesign[2];
				if (thisUtilization > maxUtilizationCM) {
					maxUtilizationCM = thisUtilization;
					*desiredTileSizeCM = thisTileSize;
					*desiredNumTileCM = thisDesign[0];
				}
			}
			*desiredPESizeCM = (*desiredTileSizeCM)/2;
			/*** PE Design ***/
			for (double thisPESize = (*desiredTileSizeCM)/2; thisPESize > 2*param->numRowSubArray; thisPESize/=2) {
				// define PE Size for layers use conventional mapping
				double thisUtilization = 0;
				vector<vector<double> > thisDesign;
				thisDesign = PEDesign(true, thisPESize, (*desiredTileSizeCM), (*desiredNumTileCM), markNM, netStructure, numRowPerSynapse, numColPerSynapse);
				thisUtilization = thisDesign[1][0];
				if (thisUtilization > maxUtilizationCM) {
					maxUtilizationCM = thisUtilization;
					*desiredPESizeCM = thisPESize;
				}
			}
			peDup = PEDesign(false, (*desiredPESizeCM), (*desiredTileSizeCM), (*desiredNumTileCM), markNM, netStructure, numRowPerSynapse, numColPerSynapse);
			/*** SubArray Duplication ***/
			subArrayDup = SubArrayDup((*desiredPESizeCM), (*desiredPESizeNM), markNM, netStructure, numRowPerSynapse, numColPerSynapse);
			/*** Design SubArray ***/
			numTileEachLayer = OverallEachLayer(false, false, peDup, subArrayDup, pipelineSpeedUp, (*desiredTileSizeCM), (*desiredPESizeNM), markNM, netStructure, numRowPerSynapse, numColPerSynapse, numPENM);
			utilizationEachLayer = OverallEachLayer(true, false, peDup, subArrayDup, pipelineSpeedUp, (*desiredTileSizeCM), (*desiredPESizeNM), markNM, netStructure, numRowPerSynapse, numColPerSynapse, numPENM);
			speedUpEachLayer = OverallEachLayer(false, true, peDup, subArrayDup, pipelineSpeedUp, (*desiredTileSizeCM), (*desiredPESizeNM), markNM, netStructure, numRowPerSynapse, numColPerSynapse, numPENM);
		}
	} else {   // all Conventional Mapping
		if (maxTileSizeCM < 4*param->numRowSubArray) {
			cout << "ERROR: SubArray Size is too large, which break the chip hierarchey, please decrease the SubArray size! " << endl;
		} else {
			/*** Tile Design ***/
			*desiredTileSizeCM = MAX(maxTileSizeCM, 4*param->numRowSubArray);
			vector<double> initialDesign;
			initialDesign = TileDesignCM((*desiredTileSizeCM), markNM, netStructure, numRowPerSynapse, numColPerSynapse);
			*desiredNumTileCM = initialDesign[0];
			for (double thisTileSize = MAX(maxTileSizeCM, 4*param->numRowSubArray); thisTileSize > 4*param->numRowSubArray; thisTileSize/=2) {
				// for layers use conventional mapping
				double thisUtilization = 0;
				vector<double> thisDesign;
				thisDesign = TileDesignCM(thisTileSize, markNM, netStructure, numRowPerSynapse, numColPerSynapse);
				thisUtilization = thisDesign[2];
				if (thisUtilization > maxUtilizationCM) {
					maxUtilizationCM = thisUtilization;
					*desiredTileSizeCM = thisTileSize;
					*desiredNumTileCM = thisDesign[0];
				}
			}
			*desiredPESizeCM = (*desiredTileSizeCM)/2;
			/*** PE Design ***/
			for (double thisPESize = (*desiredTileSizeCM)/2; thisPESize > 2*param->numRowSubArray; thisPESize/=2) {
				// define PE Size for layers use conventional mapping
				double thisUtilization = 0;
				vector<vector<double> > thisDesign;
				thisDesign = PEDesign(true, thisPESize, (*desiredTileSizeCM), (*desiredNumTileCM), markNM, netStructure, numRowPerSynapse, numColPerSynapse);
				thisUtilization = thisDesign[1][0];
				if (thisUtilization > maxUtilizationCM) {
					maxUtilizationCM = thisUtilization;
					*desiredPESizeCM = thisPESize;
				}
			}
			peDup = PEDesign(false, (*desiredPESizeCM), (*desiredTileSizeCM), (*desiredNumTileCM), markNM, netStructure, numRowPerSynapse, numColPerSynapse);
			/*** SubArray Duplication ***/
			subArrayDup = SubArrayDup((*desiredPESizeCM), 0, markNM, netStructure, numRowPerSynapse, numColPerSynapse);
			/*** Design SubArray ***/
			numTileEachLayer = OverallEachLayer(false, false, peDup, subArrayDup, pipelineSpeedUp, (*desiredTileSizeCM), 0, markNM, netStructure, numRowPerSynapse, numColPerSynapse, numPENM);
			utilizationEachLayer = OverallEachLayer(true, false, peDup, subArrayDup, pipelineSpeedUp, (*desiredTileSizeCM), 0, markNM, netStructure, numRowPerSynapse, numColPerSynapse, numPENM);
			speedUpEachLayer = OverallEachLayer(false, true, peDup, subArrayDup, pipelineSpeedUp, (*desiredTileSizeCM), 0, markNM, netStructure, numRowPerSynapse, numColPerSynapse, numPENM);
		}
	}
	
	if (param->pipeline) {
		// update # of tile for pipeline system design
		*desiredNumTileCM = 0;
		*desiredNumTileNM = 0;
		for (int i=0; i<netStructure.size(); i++) {
			if (markNM[i] == 0) {
				*desiredNumTileCM = (*desiredNumTileCM) + numTileEachLayer[0][i]*numTileEachLayer[1][i];
			} else {
				*desiredNumTileNM = (*desiredNumTileNM) + numTileEachLayer[0][i]*numTileEachLayer[1][i];
			}
		}
	}
	
	*numTileRow = ceil((double)sqrt((double)(*desiredNumTileCM)+(double)(*desiredNumTileNM)));
	*numTileCol = ceil((double)((*desiredNumTileCM)+(*desiredNumTileNM))/(double)(*numTileRow));
	
	vector<vector<double> > tileLocaEachLayer;
	vector<double> tileLocaEachLayerRow;
	vector<double> tileLocaEachLayerCol;
	double thisTileTotal=0;
	for (int i=0; i<netStructure.size(); i++) {
		if (i==0) {
			tileLocaEachLayerRow.push_back(0);
			tileLocaEachLayerCol.push_back(0);
		} else {
			thisTileTotal += numTileEachLayer[0][i]*numTileEachLayer[1][i];
			tileLocaEachLayerRow.push_back((int)thisTileTotal/(*numTileRow));
			tileLocaEachLayerCol.push_back((int)thisTileTotal%(*numTileRow)-1);
		}
	}
	tileLocaEachLayer.push_back(tileLocaEachLayerRow);
	tileLocaEachLayer.push_back(tileLocaEachLayerCol);
	
	if (findNumTile) {
		return numTileEachLayer;
	} else if (findUtilization) {
		return utilizationEachLayer;
	} else if (findSpeedUp) {
		return speedUpEachLayer;
	} else {
		return tileLocaEachLayer;
	}
	peDup.clear();
	subArrayDup.clear();
	numTileEachLayer.clear();
	utilizationEachLayer.clear();
	speedUpEachLayer.clear();
}


void ChipInitialize(InputParameter& inputParameter, Technology& tech, MemCell& cell, const vector<vector<double> > &netStructure, const vector<int > &markNM, const vector<vector<double> > &numTileEachLayer,
					double numPENM, double desiredNumTileNM, double desiredPESizeNM, double desiredNumTileCM, double desiredTileSizeCM, double desiredPESizeCM, int numTileRow, int numTileCol, int *numArrayWriteParallel) { 

	/*** Initialize Tile ***/
	TileInitialize(inputParameter, tech, cell, numPENM, desiredPESizeNM, ceil((double)(desiredTileSizeCM)/(double)(desiredPESizeCM)), desiredPESizeCM);
	
	// find max layer and define the global buffer: enough to hold the max layer inputs
	double maxLayerInput = 0; 
	// find max # tiles needed to be added at the same time
	double maxTileAdded = 0;
	int maxIFMLayer = 0;
	
	for (int i=0; i<netStructure.size(); i++) {
		double input = netStructure[i][0]*netStructure[i][1]*netStructure[i][2];  // IFM_Row * IFM_Column * IFM_depth
		if (! param->pipeline) {
			if (input > maxLayerInput) {
				maxLayerInput = input;
				maxIFMLayer = i;
			}
			if (markNM[i] == 0) {
				globalBusWidth += (desiredTileSizeCM)+(desiredTileSizeCM)/param->numColMuxed;
			} else {
				globalBusWidth += (desiredPESizeNM)*ceil((double)sqrt(numPENM))+(desiredPESizeNM)*ceil((double)sqrt(numPENM))/param->numColMuxed;
			}
		} else {
			maxLayerInput += netStructure[i][0]*netStructure[i][1]*netStructure[i][2]/2;
			if (markNM[i] == 0) {
				globalBusWidth += ((desiredTileSizeCM)+(desiredTileSizeCM)/param->numColMuxed)*numTileEachLayer[0][i]*numTileEachLayer[1][i];
			} else {
				globalBusWidth += ((desiredPESizeNM)*ceil((double)sqrt(numPENM))+(desiredPESizeNM)*ceil((double)sqrt(numPENM))/param->numColMuxed)*numTileEachLayer[0][i]*numTileEachLayer[1][i];
			}
		}

		if (numTileEachLayer[0][i] > maxTileAdded) {
			maxTileAdded = numTileEachLayer[0][i];
		}
	}
	// have to limit the global bus width --> cannot grow dramatically with num of tile
	while (globalBusWidth > param->maxGlobalBusWidth) {
		globalBusWidth /= 2;
	}
	// define bufferSize for inference operation
	int bufferSize = param->numBitInput*maxLayerInput;
	
	// consider limited buffer to store gradient of weight: only part of the weight matrix is processed at a specific cycle
	// we could set a bufferOverheadConstraint to limit the overhead and speed of computation and weight-update
	// start: at least can support gradient of one weight matrix = subArray size * weightPrecision/cellPrecision
	int bufferOverHead = param->numRowSubArray*param->numColSubArray*param->numColPerSynapse*(weightGradientUnit->outPrecision+ceil(log2(param->batchSize)));
	*numArrayWriteParallel = floor(bufferOverHead/((param->numRowSubArray*param->numColSubArray)*param->synapseBit));
	
	dRAM->Initialize(param->dramType);
	if (param->trainingEstimation) {
		int numMemInRow = (netStructure[maxIFMLayer][0]-netStructure[maxIFMLayer][3]+1)*(netStructure[maxIFMLayer][1]-netStructure[maxIFMLayer][4]+1);
		int numMemInCol = netStructure[maxIFMLayer][2]*param->numBitInput;
		weightGradientUnit->Initialize(numMemInRow, numMemInCol);
		int maxWeight = 0;
		for (int i=0; i<netStructure.size(); i++) {
			double weight = netStructure[i][2]*netStructure[i][3]*netStructure[i][4]*netStructure[i][5];  // IFM_Row * IFM_Column * IFM_depth
			if (weight > maxWeight) {
				maxWeight = weight;
			}
		}
		// consider limited buffer to store gradient of weight: only part of the weight matrix is processed at a specific cycle
		// we could set a bufferOverheadConstraint to limit the overhead and speed of computation and weight-update
		// start: at least can support gradient of one weight matrix = subArray size * weightPrecision/cellPrecision
		while((bufferSize+bufferOverHead) < bufferSize*(param->bufferOverHeadConstraint+1)) {
			bufferOverHead *= 2;
			*numArrayWriteParallel *= 2;
		}
		// update the buffer size to save weight gradient
		bufferSize += bufferOverHead;
		gradientAccum->Initialize(weightGradientUnit->outPrecision+ceil(log2(param->batchSize)), (*numArrayWriteParallel)*param->numRowSubArray*param->numColSubArray);
	} 
	
	//globalBuffer->Initialize(param->numBitInput*maxLayerInput, globalBusWidth, 1, param->unitLengthWireResistance, param->clkFreq, param->globalBufferType);
	numBufferCore = ceil(bufferSize/(param->globalBufferCoreSizeRow*param->globalBufferCoreSizeCol));
	//numBufferCore = ceil(1.5*numBufferCore);
	globalBuffer->Initialize((param->globalBufferCoreSizeRow*param->globalBufferCoreSizeCol), param->globalBufferCoreSizeCol, 1, param->unitLengthWireResistance, param->clkFreq, param->globalBufferType);
	
	maxPool->Initialize(param->numBitInput, 2*2, (desiredTileSizeCM));
	GhTree->Initialize((numTileRow), (numTileCol), param->globalBusDelayTolerance, globalBusWidth);
	
	//activation inside Tile or outside?
	if (param->chipActivation) {
		int maxThroughputTile, maxAddFromSubArray;
		if (param->novelMapping) {
			maxThroughputTile = (int) max((desiredTileSizeCM), ceil((double)sqrt(numPENM))*(desiredPESizeNM));
			maxAddFromSubArray = (int) max(ceil((double)(desiredPESizeCM)/(double)param->numRowSubArray), ceil((double)(desiredPESizeNM)/(double)param->numRowSubArray));   // from subArray to ProcessingUnit
			maxAddFromSubArray *= (int) max(ceil((double)(desiredTileSizeCM)/(double)(desiredPESizeCM)), ceil((double)sqrt(numPENM)));    // from ProcessingUnit to Tile
			if (param->pipeline) {
				maxThroughputTile *= (netStructure.size()+1);
				maxAddFromSubArray *= (netStructure.size()+1);
			}
			if (param->parallelRead) {
				Gaccumulation->Initialize((int) maxTileAdded, ceil((double) log2((double) param->levelOutput))+param->numBitInput+param->numColPerSynapse+1+ceil((double) log2((double) maxAddFromSubArray)), 
										ceil((double) maxThroughputTile/(double) param->numColMuxed));
			} else {
				Gaccumulation->Initialize((int) maxTileAdded, ceil((double) log2((double) param->numRowSubArray)+(double) param->cellBit-1)+param->numBitInput+param->numColPerSynapse+1+ceil((double) log2((double) maxAddFromSubArray)), 
										ceil((double) maxThroughputTile/(double) param->numColMuxed));
			}
			if (param->reLu) {
				GreLu->Initialize(ceil((double) maxThroughputTile/(double) param->numColMuxed), param->numBitInput, param->clkFreq);
			} else {
				Gsigmoid->Initialize(false, param->numBitInput, ceil((double) log2((double) param->numRowSubArray)+(double) param->cellBit-1)+param->numBitInput+param->numColPerSynapse+1+log2((double) maxAddFromSubArray)+ceil((double) log2((double) maxTileAdded)), 
										ceil((double) maxThroughputTile/(double) param->numColMuxed), param->clkFreq);
			}
		} else {
			maxAddFromSubArray = (int) ceil((double)(desiredPESizeCM)/(double)param->numRowSubArray);   // from subArray to ProcessingUnit
			maxAddFromSubArray *= (int) ceil((double)(desiredTileSizeCM)/(double)(desiredPESizeCM));    // from ProcessingUnit to Tile
			if (param->pipeline) {
				maxAddFromSubArray *= (netStructure.size()+1);
			}
			if (param->parallelRead) {
				Gaccumulation->Initialize((int) maxTileAdded, ceil((double)log2((double)param->levelOutput))+param->numBitInput+param->numColPerSynapse+1+ceil((double)log2((double)maxAddFromSubArray)), 
										ceil((double)(desiredTileSizeCM)/(double)param->numColMuxed));
			} else {
				Gaccumulation->Initialize((int) maxTileAdded, ceil((double)log2((double)param->numRowSubArray)+(double)param->cellBit-1)+param->numBitInput+param->numColPerSynapse+1+ceil((double)log2((double)maxAddFromSubArray)), 
										ceil((double)(desiredTileSizeCM)/(double)param->numColMuxed));
			}
			if (param->reLu) {
				GreLu->Initialize(ceil((double)(desiredTileSizeCM)/(double)param->numColMuxed), param->numBitInput, param->clkFreq);
			} else {
				Gsigmoid->Initialize(false, param->numBitInput, ceil((double) log2((double) param->numRowSubArray)+(double) param->cellBit-1)+param->numBitInput+param->numColPerSynapse+1+log2((double) maxAddFromSubArray)+ceil((double) log2((double) maxTileAdded)), 
										ceil((double) (desiredTileSizeCM)/(double) param->numColMuxed), param->clkFreq);
			}
		}
	} else {   // activation inside tiles
		int maxThroughputTile;
		if (param->novelMapping) {
			maxThroughputTile = (int) max((desiredTileSizeCM), ceil((double) sqrt((double) numPENM))*(double) (desiredPESizeNM));
			if (param->pipeline) {
				maxThroughputTile *= (netStructure.size()+1);
			}
			if (param->parallelRead) {
				Gaccumulation->Initialize((int) maxTileAdded, param->numBitInput, ceil((double) maxThroughputTile/(double) param->numColMuxed));
			} else {
				Gaccumulation->Initialize((int) maxTileAdded, param->numBitInput, ceil((double) maxThroughputTile/(double) param->numColMuxed));
			}
		} else {
			if (param->parallelRead) {
				Gaccumulation->Initialize((int) maxTileAdded, param->numBitInput, ceil((double) (desiredTileSizeCM)/(double) param->numColMuxed));
			} else {
				Gaccumulation->Initialize((int) maxTileAdded, param->numBitInput, ceil((double) (desiredTileSizeCM)/(double) param->numColMuxed));
			}
		}
	}
}



vector<double> ChipCalculateArea(InputParameter& inputParameter, Technology& tech, MemCell& cell, double desiredNumTileNM, double numPENM, double desiredPESizeNM, double desiredNumTileCM, double desiredTileSizeCM, 
						double desiredPESizeCM, int numTileRow, double *height, double *width, double *CMTileheight, double *CMTilewidth, double *NMTileheight, double *NMTilewidth) {
	
	vector<double> areaResults;
	
	double area = 0;
	double areaIC = 0;
	double areaADC = 0;
	double areaAccum = 0;
	double areaOther = 0;
	double areaArray = 0;
	
	double NMheight = 0;
	double NMwidth = 0;
	double CMheight = 0;
	double CMwidth = 0;
	
	*NMTileheight = 0;
	*NMTilewidth = 0;
	*CMTileheight = 0;
	*CMTilewidth = 0;
	*height = 0;
	*width = 0;
	
	vector<double> areaCMTile;
	vector<double> areaNMTile;
	
	if (param->novelMapping) {
		areaNMTile = TileCalculateArea(numPENM, desiredPESizeNM, true, &NMheight, &NMwidth);
		double NMTileArea = areaNMTile[0];
		double NMTileAreaIC = areaNMTile[1];
		double NMTileAreaADC = areaNMTile[2];
		double NMTileAreaAccum = areaNMTile[3];
		double NMTileAreaOther = areaNMTile[4];
		double NMTileAreaArray = areaNMTile[5];
		area += NMTileArea*desiredNumTileNM;
		areaIC += NMTileAreaIC*desiredNumTileNM;
		areaADC += NMTileAreaADC*desiredNumTileNM;
		areaAccum += NMTileAreaAccum*desiredNumTileNM;
		areaOther += NMTileAreaOther*desiredNumTileNM;
		areaArray += NMTileAreaArray*desiredNumTileNM;
		*NMTileheight = NMheight;
		*NMTilewidth = NMwidth;
	}
	
	areaCMTile = TileCalculateArea(pow(ceil((double) desiredTileSizeCM/(double) desiredPESizeCM), 2), desiredPESizeCM, false, &CMheight, &CMwidth);
	
	double CMTileArea = areaCMTile[0];
	double CMTileAreaIC = areaCMTile[1];
	double CMTileAreaADC = areaCMTile[2];
	double CMTileAreaAccum = areaCMTile[3];
	double CMTileAreaOther = areaCMTile[4];
	double CMTileAreaArray = areaCMTile[5];
	
	area += CMTileArea*desiredNumTileCM;
	areaIC += CMTileAreaIC*desiredNumTileCM;
	areaADC += CMTileAreaADC*desiredNumTileCM;
	areaAccum += CMTileAreaAccum*desiredNumTileCM;
	areaOther += CMTileAreaOther*desiredNumTileCM;
	areaArray += CMTileAreaArray*desiredNumTileCM;
	*CMTileheight = CMheight;
	*CMTilewidth = CMwidth;
	
	// global buffer is made up by multiple cores
	globalBuffer->CalculateArea(numTileRow*max(NMheight, CMheight), NULL, NONE);
	double globalBufferArea = globalBuffer->area*numBufferCore;
	double globalBufferHeight = numTileRow*max(NMheight, CMheight);
	double globalBufferWidth = globalBufferArea/globalBufferHeight;
	GhTree->CalculateArea(max(NMheight, CMheight), max(NMwidth, CMwidth), param->treeFoldedRatio);
	maxPool->CalculateUnitArea(NONE);
	maxPool->CalculateArea(globalBufferWidth);
	Gaccumulation->CalculateArea(NULL, globalBufferHeight/3, NONE);
	
	double areaGreLu = 0;
	double areaGsigmoid = 0;
	double areaWG = 0;
	
	if (param->chipActivation) {
		if (param->reLu) {
			GreLu->CalculateArea(NULL, globalBufferWidth/3, NONE);
			area += GreLu->area;
			areaGreLu += GreLu->area;
		} else {
			Gsigmoid->CalculateUnitArea(NONE);
			Gsigmoid->CalculateArea(NULL, globalBufferWidth/3, NONE);
			area += Gsigmoid->area;
			areaGsigmoid += Gsigmoid->area;
		}
	}
	
	if (param->trainingEstimation) {
		weightGradientUnit->CalculateArea();
		gradientAccum->CalculateArea(globalBufferHeight, NULL, NONE);
		area += weightGradientUnit->area + gradientAccum->area;
		areaWG = weightGradientUnit->area + gradientAccum->area;
	}
	
	area += globalBufferArea + GhTree->area + maxPool->area + Gaccumulation->area;
	areaIC += GhTree->area;
	areaResults.push_back(area);
	areaResults.push_back(areaIC);
	areaResults.push_back(areaADC);
	areaResults.push_back(areaAccum + Gaccumulation->area);
	areaResults.push_back(areaOther + globalBufferArea + maxPool->area + areaGreLu + areaGsigmoid);
	areaResults.push_back(areaWG);
	areaResults.push_back(areaArray);
	
	*height = sqrt(area);
	*width = area/(*height);
	
	return areaResults;
}


double ChipCalculatePerformance(InputParameter& inputParameter, Technology& tech, MemCell& cell, int layerNumber, const string &newweightfile, const string &oldweightfile, const string &inputfile, bool followedByMaxPool, 
							const vector<vector<double> > &netStructure, const vector<int> &markNM, const vector<vector<double> > &numTileEachLayer, const vector<vector<double> > &utilizationEachLayer, 
							const vector<vector<double> > &speedUpEachLayer, const vector<vector<double> > &tileLocaEachLayer, double numPENM, double desiredPESizeNM, double desiredTileSizeCM, 
							double desiredPESizeCM, double CMTileheight, double CMTilewidth, double NMTileheight, double NMTilewidth, int numArrayWriteParallel,
							double *readLatency, double *readDynamicEnergy, double *leakage, double *readLatencyAG, double *readDynamicEnergyAG, double *readLatencyWG, double *readDynamicEnergyWG, 
							double *writeLatencyWU, double *writeDynamicEnergyWU, double *bufferLatency, double *bufferDynamicEnergy, double *icLatency, double *icDynamicEnergy, double *coreLatencyADC, 
							double *coreLatencyAccum, double *coreLatencyOther, double *coreEnergyADC, double *coreEnergyAccum, double *coreEnergyOther, double *dramLatency, double *dramDynamicEnergy,
							double *readLatencyPeakFW, double *readDynamicEnergyPeakFW, double *readLatencyPeakAG, double *readDynamicEnergyPeakAG, double *readLatencyPeakWG, double *readDynamicEnergyPeakWG,
							double *writeLatencyPeakWU, double *writeDynamicEnergyPeakWU) {
	
	
	int numRowPerSynapse, numColPerSynapse;
	numRowPerSynapse = param->numRowPerSynapse;
	numColPerSynapse = param->numColPerSynapse;
	
	// only get performance of single layer
	int l = layerNumber;
	// get weight matrix file Size
	int weightMatrixRow = netStructure[l][2]*netStructure[l][3]*netStructure[l][4]*numRowPerSynapse;
	int weightMatrixCol = netStructure[l][5]*numColPerSynapse;
	
	// load in whole file 
	vector<vector<double> > inputVector;
	inputVector = LoadInInputData(inputfile); 
	vector<vector<double> > newMemory;
	newMemory = LoadInWeightData(newweightfile, numRowPerSynapse, numColPerSynapse, param->maxConductance, param->minConductance);
	vector<vector<double> > oldMemory;
	oldMemory = LoadInWeightData(oldweightfile, numRowPerSynapse, numColPerSynapse, param->maxConductance, param->minConductance);
	
	*readLatency = 0;
	*readDynamicEnergy = 0;
	*readLatencyAG = 0;
	*readDynamicEnergyAG = 0;
	*readLatencyPeakFW = 0;
	*readDynamicEnergyPeakFW = 0;
	*readLatencyPeakAG = 0;
	*readDynamicEnergyPeakAG = 0;
	
	*bufferLatency = 0;
	*bufferDynamicEnergy = 0;
	*icLatency = 0;
	*icDynamicEnergy = 0;
	*coreEnergyADC = 0;
	*coreEnergyAccum = 0;
	*coreEnergyOther = 0;
	*coreLatencyADC = 0;
	*coreLatencyAccum = 0;
	*coreLatencyOther = 0;
	
	*readLatencyWG = 0;
	*readDynamicEnergyWG = 0;
	*writeLatencyWU = 0;
	*writeDynamicEnergyWU = 0;

	*readLatencyPeakWG = 0;
	*readDynamicEnergyPeakWG = 0;
	*writeLatencyPeakWU = 0;
	*writeDynamicEnergyPeakWU = 0;
	
	*leakage = 0;
	*dramLatency = 0;
	*dramDynamicEnergy = 0;
	
	double tileLeakage = 0;
	double tileReadLatency,tileReadDynamicEnergy,tileReadLatencyAG,tileReadDynamicEnergyAG,tileWriteLatencyWU,tileWriteDynamicEnergyWU;
	double tileReadLatencyPeakFW,tileReadDynamicEnergyPeakFW,tileReadLatencyPeakAG,tileReadDynamicEnergyPeakAG;
	double tileWriteLatencyPeakWU,tileWriteDynamicEnergyPeakWU,tilebufferLatency,tilebufferDynamicEnergy,tileicLatency,tileicDynamicEnergy;
	double tileLatencyADC,tileLatencyAccum,tileLatencyOther,tileEnergyADC,tileEnergyAccum,tileEnergyOther;
	
	int numInVector = (netStructure[l][0]-netStructure[l][3]+1)/netStructure[l][7]*(netStructure[l][1]-netStructure[l][4]+1)/netStructure[l][7];
	int totalNumTile = 0;
	for (int i=0; i<netStructure.size(); i++) {
		totalNumTile += numTileEachLayer[0][i] * numTileEachLayer[1][i];
	}
	
	if (markNM[l] == 0) {   // conventional mapping
		for (int i=0; i<ceil((double) netStructure[l][2]*(double) netStructure[l][3]*(double) netStructure[l][4]*(double) numRowPerSynapse/desiredTileSizeCM); i++) {       // # of tiles in row
			for (int j=0; j<ceil((double) netStructure[l][5]*(double) numColPerSynapse/(double) desiredTileSizeCM); j++) {   // # of tiles in Column
				int numRowMatrix = min(desiredTileSizeCM, weightMatrixRow-i*desiredTileSizeCM);
				int numColMatrix = min(desiredTileSizeCM, weightMatrixCol-j*desiredTileSizeCM);
				
				// assign weight and input to specific tile
				vector<vector<double> > tileMemoryOld;
				tileMemoryOld = CopyArray(oldMemory, i*desiredTileSizeCM, j*desiredTileSizeCM, numRowMatrix, numColMatrix);
				vector<vector<double> > tileMemory;
				tileMemory = CopyArray(newMemory, i*desiredTileSizeCM, j*desiredTileSizeCM, numRowMatrix, numColMatrix);
				
				vector<vector<double> > tileInput;
				tileInput = CopyInput(inputVector, i*desiredTileSizeCM, numInVector*param->numBitInput, numRowMatrix);
				
				TileCalculatePerformance(tileMemory, tileMemoryOld, tileInput, markNM[l], layerNumber, ceil((double)desiredTileSizeCM/(double)desiredPESizeCM), desiredPESizeCM, speedUpEachLayer[0][l], speedUpEachLayer[1][l],
									numRowMatrix, numColMatrix, numInVector*param->numBitInput, tech, cell, &tileReadLatency, &tileReadDynamicEnergy, &tileLeakage,
									&tileReadLatencyAG, &tileReadDynamicEnergyAG, &tileWriteLatencyWU, &tileWriteDynamicEnergyWU,
									&tilebufferLatency, &tilebufferDynamicEnergy, &tileicLatency, &tileicDynamicEnergy, 
									&tileLatencyADC, &tileLatencyAccum, &tileLatencyOther, &tileEnergyADC, &tileEnergyAccum, &tileEnergyOther, 
									&tileReadLatencyPeakFW, &tileReadDynamicEnergyPeakFW, &tileReadLatencyPeakAG, &tileReadDynamicEnergyPeakAG,
									&tileWriteLatencyPeakWU, &tileWriteDynamicEnergyPeakWU);
				
				*readLatency = MAX(tileReadLatency, (*readLatency));
				*readDynamicEnergy += tileReadDynamicEnergy;
				*readLatencyPeakFW = MAX(tileReadLatencyPeakFW, (*readLatencyPeakFW));
				*readDynamicEnergyPeakFW += tileReadDynamicEnergyPeakFW;
				if (param->trainingEstimation) {
					*readLatencyAG = MAX(tileReadLatencyAG, (*readLatencyAG));
					*readDynamicEnergyAG += tileReadDynamicEnergyAG;
					// accumulate write latency as array need to be write sequentially (worst case)
					// limitation by on-chip buffer, write latency will be divided by numArrayWriteParallel (real case)
					*writeLatencyWU += tileWriteLatencyWU;
					*writeDynamicEnergyWU += tileWriteDynamicEnergyWU;
					
					*readLatencyPeakAG = MAX(tileReadLatencyPeakAG, (*readLatencyPeakAG));
					*readDynamicEnergyPeakAG += tileReadDynamicEnergyPeakAG;
					// accumulate write latency as array need to be write sequentially (worst case)
					// limitation by on-chip buffer, write latency will be divided by numArrayWriteParallel (real case)
					*writeLatencyPeakWU += tileWriteLatencyPeakWU;
					*writeDynamicEnergyPeakWU += tileWriteDynamicEnergyPeakWU;
				}
				*bufferLatency = MAX(tilebufferLatency, (*bufferLatency));
				*bufferDynamicEnergy += tilebufferDynamicEnergy;
				*icLatency = MAX(tileicLatency, (*icLatency));
				*icDynamicEnergy += tileicDynamicEnergy;
				
				*coreLatencyADC = MAX(tileLatencyADC, (*coreLatencyADC));
				*coreLatencyAccum = MAX(tileLatencyAccum, (*coreLatencyAccum));
				*coreLatencyOther = MAX(tileLatencyOther, (*coreLatencyOther));
				
				*coreEnergyADC += tileEnergyADC;
				*coreEnergyAccum += tileEnergyAccum;
				*coreEnergyOther += tileEnergyOther;
			}
		}
		if (param->chipActivation) {
			if (param->reLu) {
				GreLu->CalculateLatency(ceil(numInVector*netStructure[l][5]/(double) GreLu->numUnit));
				GreLu->CalculatePower(ceil(numInVector*netStructure[l][5]/(double) GreLu->numUnit));
				*readLatency += GreLu->readLatency;
				*readDynamicEnergy += GreLu->readDynamicEnergy;
				*readLatencyPeakFW += GreLu->readLatency;
				*readDynamicEnergyPeakFW += GreLu->readDynamicEnergy;
				*coreLatencyOther += GreLu->readLatency;
				*coreEnergyOther += GreLu->readDynamicEnergy;
			} else {
				Gsigmoid->CalculateLatency(ceil(numInVector*netStructure[l][5]/Gsigmoid->numEntry));
				Gsigmoid->CalculatePower(ceil(numInVector*netStructure[l][5]/Gsigmoid->numEntry));
				*readLatency += Gsigmoid->readLatency;
				*readDynamicEnergy += Gsigmoid->readDynamicEnergy;
				*readLatencyPeakFW += Gsigmoid->readLatency;
				*readDynamicEnergyPeakFW += Gsigmoid->readDynamicEnergy;
				*coreLatencyOther += Gsigmoid->readLatency;
				*coreEnergyOther += Gsigmoid->readDynamicEnergy;
			}
		}
		
		if (numTileEachLayer[0][l] > 1) {   
			Gaccumulation->CalculateLatency(numTileEachLayer[1][l]*netStructure[l][5]*(ceil(numInVector/(double) Gaccumulation->numAdderTree)), numTileEachLayer[0][l], 0);
			Gaccumulation->CalculatePower(numTileEachLayer[1][l]*netStructure[l][5]*(ceil(numInVector/(double) Gaccumulation->numAdderTree)), numTileEachLayer[0][l]);
			*readLatency += Gaccumulation->readLatency;
			*readDynamicEnergy += Gaccumulation->readDynamicEnergy;
			*readLatencyPeakFW += Gaccumulation->readLatency;
			*readDynamicEnergyPeakFW += Gaccumulation->readDynamicEnergy;
			if ((param->trainingEstimation) && (layerNumber!=0)) {
				*readLatencyAG += Gaccumulation->readLatency;
				*readDynamicEnergyAG += Gaccumulation->readDynamicEnergy;
				*readLatencyPeakAG += Gaccumulation->readLatency;
				*readDynamicEnergyPeakAG += Gaccumulation->readDynamicEnergy;
			}
			*coreLatencyAccum += Gaccumulation->readLatency*((param->trainingEstimation)&&(layerNumber!=0)==true? 2:1);
			*coreEnergyAccum += Gaccumulation->readDynamicEnergy*((param->trainingEstimation)&&(layerNumber!=0)==true? 2:1);
		}
		
		// if this layer is followed by Max Pool
		if (followedByMaxPool) {
			maxPool->CalculateLatency(1e20, 0, ceil((double) (numInVector/(double) maxPool->window)/(double) desiredTileSizeCM));
			maxPool->CalculatePower(ceil((double) (numInVector/maxPool->window)/(double) desiredTileSizeCM));
			*readLatency += maxPool->readLatency;
			*readDynamicEnergy += maxPool->readDynamicEnergy;
			*readLatencyPeakFW += maxPool->readLatency;
			*readDynamicEnergyPeakFW += maxPool->readDynamicEnergy;
			*coreLatencyOther += maxPool->readLatency;
			*coreEnergyOther += maxPool->readDynamicEnergy;
		}							  
		double numBitToLoadOut = weightMatrixRow*param->numBitInput*numInVector;
		double numBitToLoadIn = ceil(weightMatrixCol/param->numColPerSynapse)*param->numBitInput*numInVector/(netStructure[l][6]? 4:1);
		
		GhTree->CalculateLatency(0, 0, tileLocaEachLayer[0][l], tileLocaEachLayer[1][l], CMTileheight, CMTilewidth, ceil((numBitToLoadOut+numBitToLoadIn)/ceil(GhTree->busWidth*(numTileEachLayer[0][l]*numTileEachLayer[1][l]/totalNumTile))));
		GhTree->CalculatePower(0, 0, tileLocaEachLayer[0][l], tileLocaEachLayer[1][l], CMTileheight, CMTilewidth, ceil(GhTree->busWidth*(numTileEachLayer[0][l]*numTileEachLayer[1][l]/totalNumTile)), 
							ceil((numBitToLoadOut+numBitToLoadIn)/ceil(GhTree->busWidth*(numTileEachLayer[0][l]*numTileEachLayer[1][l]/totalNumTile))));
					
		globalBuffer->CalculateLatency(globalBuffer->interface_width, numBitToLoadOut/globalBuffer->interface_width,
								globalBuffer->interface_width, numBitToLoadIn/globalBuffer->interface_width);
		globalBuffer->CalculatePower(globalBuffer->interface_width, numBitToLoadOut/globalBuffer->interface_width,
								globalBuffer->interface_width, numBitToLoadIn/globalBuffer->interface_width);
		// since multi-core buffer has improve the parallelism
		globalBuffer->readLatency /= MIN(numBufferCore, ceil(globalBusWidth/globalBuffer->interface_width));
		globalBuffer->writeLatency /= MIN(numBufferCore, ceil(globalBusWidth/globalBuffer->interface_width));
		// each time, only a part of the ic is used to transfer data to a part of the tiles
		globalBuffer->readLatency *= ceil(totalNumTile/(numTileEachLayer[0][l]*numTileEachLayer[1][l]));
		globalBuffer->writeLatency *= ceil(totalNumTile/(numTileEachLayer[0][l]*numTileEachLayer[1][l]));
	
	} else {   // novel Mapping
		for (int i=0; i<ceil((double) netStructure[l][2]*(double) numRowPerSynapse/(double) desiredPESizeNM); i++) {       // # of tiles in row
			for (int j=0; j<ceil((double) netStructure[l][5]*(double) numColPerSynapse/(double) desiredPESizeNM); j++) {   // # of tiles in Column
				// novel mapping
				int numtileEachLayerRow = ceil((double) netStructure[l][2]*(double) numRowPerSynapse/(double) desiredPESizeNM);
				int numtileEachLayerCol = ceil((double) netStructure[l][5]*(double) numColPerSynapse/(double) desiredPESizeNM);
				
				int numRowMatrix = min(desiredPESizeNM*numPENM, weightMatrixRow-i*desiredPESizeNM*numPENM);
				int numColMatrix = min(desiredPESizeNM, weightMatrixCol-j*desiredPESizeNM);
				
				// assign weight and input to specific tile
				vector<vector<double> > tileMemoryOld;
				tileMemoryOld = ReshapeArray(oldMemory, i*desiredPESizeNM, j*desiredPESizeNM, (int) netStructure[l][2]*numRowPerSynapse/numtileEachLayerRow, 
									(int) netStructure[l][5]*numColPerSynapse/numtileEachLayerCol, numPENM, (int) netStructure[l][2]*numRowPerSynapse);
				
				vector<vector<double> > tileMemory;
				tileMemory = ReshapeArray(newMemory, i*desiredPESizeNM, j*desiredPESizeNM, (int) netStructure[l][2]*numRowPerSynapse/numtileEachLayerRow, 
									(int) netStructure[l][5]*numColPerSynapse/numtileEachLayerCol, numPENM, (int) netStructure[l][2]*numRowPerSynapse);
				
				vector<vector<double> > tileInput;
				tileInput = ReshapeInput(inputVector, i*desiredPESizeNM, (int) (netStructure[l][0]-netStructure[l][3]+1)*(netStructure[l][1]-netStructure[l][4]+1)*param->numBitInput, 
									(int) netStructure[l][2]*numRowPerSynapse/numtileEachLayerRow, numPENM, (int) netStructure[l][2]*numRowPerSynapse);
	
				TileCalculatePerformance(tileMemory, tileMemoryOld, tileInput, markNM[l], layerNumber, numPENM, desiredPESizeNM, speedUpEachLayer[0][l], speedUpEachLayer[1][l],
									numRowMatrix, numColMatrix, numInVector*param->numBitInput, tech, cell, 
									&tileReadLatency, &tileReadDynamicEnergy, &tileLeakage, &tileReadLatencyAG, &tileReadDynamicEnergyAG, &tileWriteLatencyWU, &tileWriteDynamicEnergyWU,
									&tilebufferLatency, &tilebufferDynamicEnergy, &tileicLatency, &tileicDynamicEnergy,
									&tileLatencyADC, &tileLatencyAccum, &tileLatencyOther, &tileEnergyADC, &tileEnergyAccum, &tileEnergyOther, 
									&tileReadLatencyPeakFW, &tileReadDynamicEnergyPeakFW, &tileReadLatencyPeakAG, &tileReadDynamicEnergyPeakAG,
									&tileWriteLatencyPeakWU, &tileWriteDynamicEnergyPeakWU);
				
				*readLatency = MAX(tileReadLatency, (*readLatency));
				*readDynamicEnergy += tileReadDynamicEnergy;
				*readLatencyPeakFW = MAX(tileReadLatencyPeakFW, (*readLatencyPeakFW));
				*readDynamicEnergyPeakFW += tileReadDynamicEnergyPeakFW;
				if (param->trainingEstimation) {
					*readLatencyAG = MAX(tileReadLatencyAG, (*readLatencyAG));
					*readDynamicEnergyAG += tileReadDynamicEnergyAG;
					// accumulate write latency as array need to be write sequentially (worst case)
					// limitation by on-chip buffer, write latency will be divided by numArrayWriteParallel (real case)
					*writeLatencyWU += tileWriteLatencyWU;
					*writeDynamicEnergyWU += tileWriteDynamicEnergyWU;
					
					*readLatencyPeakAG = MAX(tileReadLatencyPeakAG, (*readLatencyPeakAG));
					*readDynamicEnergyPeakAG += tileReadDynamicEnergyPeakAG;
					// accumulate write latency as array need to be write sequentially (worst case)
					// limitation by on-chip buffer, write latency will be divided by numArrayWriteParallel (real case)
					*writeLatencyPeakWU += tileWriteLatencyPeakWU;
					*writeDynamicEnergyPeakWU += tileWriteDynamicEnergyPeakWU;
				}
				*bufferLatency = MAX(tilebufferLatency, (*bufferLatency));
				*bufferDynamicEnergy += tilebufferDynamicEnergy;
				*icLatency = MAX(tileicLatency, (*icLatency));
				*icDynamicEnergy += tileicDynamicEnergy;
				
				*coreLatencyADC = MAX(tileLatencyADC, (*coreLatencyADC));
				*coreLatencyAccum = MAX(tileLatencyAccum, (*coreLatencyAccum));
				*coreLatencyOther = MAX(tileLatencyOther, (*coreLatencyOther));
				
				*coreEnergyADC += tileEnergyADC;
				*coreEnergyAccum += tileEnergyAccum;
				*coreEnergyOther += tileEnergyOther;
			}
		}
		
		if (param->chipActivation) {
			if (param->reLu) {
				GreLu->CalculateLatency(ceil(numInVector*netStructure[l][5]/(double) GreLu->numUnit));
				GreLu->CalculatePower(ceil(numInVector*netStructure[l][5]/(double) GreLu->numUnit));
				*readLatency += GreLu->readLatency;
				*readDynamicEnergy += GreLu->readDynamicEnergy;
				*readLatencyPeakFW += GreLu->readLatency;
				*readDynamicEnergyPeakFW += GreLu->readDynamicEnergy;
				*coreLatencyOther += GreLu->readLatency;
				*coreEnergyOther += GreLu->readDynamicEnergy;
			} else {
				Gsigmoid->CalculateLatency(ceil(numInVector*netStructure[l][5]/Gsigmoid->numEntry));
				Gsigmoid->CalculatePower(ceil(numInVector*netStructure[l][5]/Gsigmoid->numEntry));
				*readLatency += Gsigmoid->readLatency;
				*readDynamicEnergy += Gsigmoid->readDynamicEnergy;
				*readLatencyPeakFW += Gsigmoid->readLatency;
				*readDynamicEnergyPeakFW += Gsigmoid->readDynamicEnergy;
				*coreLatencyOther += Gsigmoid->readLatency;
				*coreEnergyOther += Gsigmoid->readDynamicEnergy;
			}
		}
		
		if (numTileEachLayer[0][l] > 1) {   
			Gaccumulation->CalculateLatency(numTileEachLayer[1][l]*netStructure[l][5]*(ceil(numInVector/(double) Gaccumulation->numAdderTree)), numTileEachLayer[0][l], 0);
			Gaccumulation->CalculatePower(numTileEachLayer[1][l]*netStructure[l][5]*(ceil(numInVector/(double) Gaccumulation->numAdderTree)), numTileEachLayer[0][l]);
			*readLatency += Gaccumulation->readLatency;
			*readDynamicEnergy += Gaccumulation->readDynamicEnergy;
			*readLatencyPeakFW += Gaccumulation->readLatency;
			*readDynamicEnergyPeakFW += Gaccumulation->readDynamicEnergy;
			if ((param->trainingEstimation) && (layerNumber!=0)) {
				*readLatencyAG += Gaccumulation->readLatency;
				*readDynamicEnergyAG += Gaccumulation->readDynamicEnergy;
				*readLatencyPeakAG += Gaccumulation->readLatency;
				*readDynamicEnergyPeakAG += Gaccumulation->readDynamicEnergy;
			}
			*coreLatencyAccum += Gaccumulation->readLatency*((param->trainingEstimation)&&(layerNumber!=0)==true? 2:1);
			*coreEnergyAccum += Gaccumulation->readDynamicEnergy*((param->trainingEstimation)&&(layerNumber!=0)==true? 2:1);
		}
		
		// if this layer is followed by Max Pool
		if (followedByMaxPool) {
			maxPool->CalculateLatency(1e20, 0, ceil((double) (numInVector/(double) maxPool->window)/(double) desiredPESizeNM*sqrt((double) numPENM)));
			maxPool->CalculatePower(ceil((double) (numInVector/maxPool->window)/(double) desiredPESizeNM*sqrt((double) numPENM)));
			*readLatency += maxPool->readLatency;
			*readDynamicEnergy += maxPool->readDynamicEnergy;
			*readLatencyPeakFW += maxPool->readLatency;
			*readDynamicEnergyPeakFW += maxPool->readDynamicEnergy;
			*coreLatencyOther += maxPool->readLatency;
			*coreEnergyOther += maxPool->readDynamicEnergy;
		}
		
		double numBitToLoadOut = weightMatrixRow*param->numBitInput*numInVector/netStructure[l][3];
		double numBitToLoadIn = ceil(weightMatrixCol/param->numColPerSynapse)*param->numBitInput*numInVector/(netStructure[l][6]? 4:1);
		
		GhTree->CalculateLatency(0, 0, tileLocaEachLayer[0][l], tileLocaEachLayer[1][l], NMTileheight, NMTilewidth, ceil((numBitToLoadOut+numBitToLoadIn)/ceil(GhTree->busWidth*(numTileEachLayer[0][l]*numTileEachLayer[1][l]/totalNumTile))));
		GhTree->CalculatePower(0, 0, tileLocaEachLayer[0][l], tileLocaEachLayer[1][l], NMTileheight, NMTilewidth, ceil(GhTree->busWidth*(numTileEachLayer[0][l]*numTileEachLayer[1][l]/totalNumTile)), 
							ceil((numBitToLoadOut+numBitToLoadIn)/ceil(GhTree->busWidth*(numTileEachLayer[0][l]*numTileEachLayer[1][l]/totalNumTile))));
		
		globalBuffer->CalculateLatency(globalBuffer->interface_width, numBitToLoadOut/globalBuffer->interface_width,
								globalBuffer->interface_width, numBitToLoadIn/globalBuffer->interface_width);
		globalBuffer->CalculatePower(globalBuffer->interface_width, numBitToLoadOut/globalBuffer->interface_width,
								globalBuffer->interface_width, numBitToLoadIn/globalBuffer->interface_width);
		// since multi-core buffer has improve the parallelism
		globalBuffer->readLatency /= MIN(numBufferCore, ceil(globalBusWidth/globalBuffer->interface_width));
		globalBuffer->writeLatency /= MIN(numBufferCore, ceil(globalBusWidth/globalBuffer->interface_width));
		// each time, only a part of the ic is used to transfer data to a part of the tiles
		globalBuffer->readLatency *= ceil(totalNumTile/(numTileEachLayer[0][l]*numTileEachLayer[1][l]));
		globalBuffer->writeLatency *= ceil(totalNumTile/(numTileEachLayer[0][l]*numTileEachLayer[1][l]));
	}
	
	// training: FW(up and down)->2; AG(up and down)->2; WG(up and down)->2
	*bufferLatency += (globalBuffer->readLatency + globalBuffer->writeLatency)*((param->trainingEstimation)==true? ((layerNumber!=0)==true? 6:4):1);
	*bufferDynamicEnergy += (globalBuffer->readDynamicEnergy + globalBuffer->writeDynamicEnergy)*((param->trainingEstimation)==true? ((layerNumber!=0)==true? 6:4):1);
	*icLatency += GhTree->readLatency*((param->trainingEstimation)&&(layerNumber!=0)==true? 2:1);
	*icDynamicEnergy += GhTree->readDynamicEnergy*((param->trainingEstimation)&&(layerNumber!=0)==true? 2:1);
	
	*readLatency += (globalBuffer->readLatency + globalBuffer->writeLatency)*((param->trainingEstimation)==true? 2:1);
	*readDynamicEnergy += (globalBuffer->readDynamicEnergy + globalBuffer->writeDynamicEnergy)*((param->trainingEstimation)==true? 2:1);
	*readLatency += GhTree->readLatency;
	*readDynamicEnergy += GhTree->readDynamicEnergy;
	
	*readLatencyAG += (globalBuffer->readLatency + globalBuffer->writeLatency)*((param->trainingEstimation)&&(layerNumber!=0)==true? 2:0);
	*readDynamicEnergyAG += (globalBuffer->readDynamicEnergy + globalBuffer->writeDynamicEnergy)*((param->trainingEstimation)&&(layerNumber!=0)==true? 2:0);
	*readLatencyAG += GhTree->readLatency*((param->trainingEstimation)&&(layerNumber!=0)==true? 1:0);
	*readDynamicEnergyAG += GhTree->readDynamicEnergy*((param->trainingEstimation)&&(layerNumber!=0)==true? 1:0);
	
	*readLatencyWG = (globalBuffer->readLatency + globalBuffer->writeLatency)*((param->trainingEstimation)==true? 2:0);
	*readDynamicEnergyWG = (globalBuffer->readDynamicEnergy + globalBuffer->writeDynamicEnergy)*((param->trainingEstimation)==true? 2:0);
	
	int dataLoadIn = (netStructure[0][0])*(netStructure[0][1])*param->numBitInput; 
	// ONLY LOAD IMAGE
	dRAM->CalculateLatency(dataLoadIn);
	dRAM->CalculatePower(dataLoadIn);
	*readLatency += (dRAM->readLatency)*((param->trainingEstimation)==true? 0:1);
	*readDynamicEnergy += (dRAM->readDynamicEnergy)*((param->trainingEstimation)==true? 0:1);
	*dramLatency = (dRAM->readLatency*((param->trainingEstimation)==true? 0:1)); 
	*dramDynamicEnergy = (dRAM->readDynamicEnergy*((param->trainingEstimation)==true? 0:1));
	
	if (param->trainingEstimation) {
		// Dring on-chip training, the activation and gradient of activation of each layer will be sent to DRAM
		// After all the image in the batch is done with the gradient of activation, the activation and gradient of activation will be sent back to the chip, to get the gradient of weight
		// Limited by global buffer, the weight gradient will be sent to DRAM and then grab back to be accumulated across batch
		int dataLoadIn = (netStructure[l][0])*(netStructure[l][1])*param->numBitInput; 
		int dataLoadWeight = netStructure[l][2]*netStructure[l][3]*netStructure[l][4]*netStructure[l][5]*weightGradientUnit->outPrecision;
		
		// For activation and activation gradient transfer
		dRAM->CalculateLatency(dataLoadIn);
		dRAM->CalculatePower(dataLoadIn);
		// to load data size = IFM
		*readLatency += (dRAM->readLatency)*2;
		*readDynamicEnergy += (dRAM->readDynamicEnergy)*2;
		*readLatencyAG += (dRAM->readLatency)*2*((layerNumber!=0)==true? 1:0);
		*readDynamicEnergyAG += (dRAM->readDynamicEnergy)*2*((layerNumber!=0)==true? 1:0);
		*readLatencyWG += (dRAM->readLatency)*2;
		*readDynamicEnergyWG += (dRAM->readDynamicEnergy)*2;
		*dramLatency = (dRAM->readLatency)*6*((layerNumber!=0)==true? 6:4); // 2 for forward, 2 for AG, 2 for WG
		*dramDynamicEnergy = (dRAM->readDynamicEnergy)*6*((layerNumber!=0)==true? 6:4);
		
		// since for each iteration, need *batchSize computation
		*readLatency *= param->batchSize;
		*readDynamicEnergy *= param->batchSize;
		*readLatencyAG *= param->batchSize;
		*readDynamicEnergyAG *= param->batchSize;
		
		*readLatencyPeakFW *= param->batchSize;
		*readDynamicEnergyPeakFW *= param->batchSize;
		*readLatencyPeakAG *= param->batchSize;
		*readDynamicEnergyPeakAG *= param->batchSize;
		
		*icLatency *= param->batchSize;
		*icDynamicEnergy *= param->batchSize;
		
		*coreLatencyADC *= param->batchSize;
		*coreEnergyADC *= param->batchSize;
		*coreLatencyAccum *= param->batchSize;
		*coreEnergyAccum *= param->batchSize;
		
		
		// For weight gradient transfer
		dRAM->CalculateLatency(dataLoadWeight);
		dRAM->CalculatePower(dataLoadWeight);
		// Dring calculation of weight gradient, need to load in activation and gradient of activation
		globalBuffer->CalculateLatency(globalBuffer->interface_width, dataLoadWeight/globalBuffer->interface_width,
								globalBuffer->interface_width, dataLoadWeight/globalBuffer->interface_width);
		globalBuffer->CalculatePower(globalBuffer->interface_width, dataLoadWeight/globalBuffer->interface_width,
								globalBuffer->interface_width, dataLoadWeight/globalBuffer->interface_width);
		// since multi-core buffer has improve the parallelism
		globalBuffer->readLatency /= MIN(numBufferCore, ceil(globalBusWidth/globalBuffer->interface_width));
		globalBuffer->writeLatency /= MIN(numBufferCore, ceil(globalBusWidth/globalBuffer->interface_width));
		
		// calculation of weight gradient
		weightGradientUnit->CalculateLatency(netStructure[l][5], (netStructure[l][0]-netStructure[l][3]+1)*(netStructure[l][1]-netStructure[l][4]+1)*param->numBitInput);
		weightGradientUnit->CalculatePower(netStructure[l][5], (netStructure[l][0]-netStructure[l][3]+1)*(netStructure[l][1]-netStructure[l][4]+1)*param->numBitInput);
		// here consider speed up 
		double thisMatrixRow = (netStructure[l][0]-netStructure[l][3]+1)*(netStructure[l][1]-netStructure[l][4]+1);
		double thisMatrixCol = netStructure[l][2]*param->numBitInput;
		double arrayNeedRow = ceil(thisMatrixRow/param->numRowSubArrayWG)==0? 1:ceil(thisMatrixRow/param->numRowSubArrayWG);
		double arrayNeedCol = ceil(thisMatrixCol/param->numColSubArrayWG)==0? 1:ceil(thisMatrixCol/param->numColSubArrayWG);
		double speedUpRow, speedUpCol;
		if (thisMatrixRow != 1) {
			speedUpRow = floor(weightGradientUnit->numArrayInRow/arrayNeedRow)==0? 1:floor(weightGradientUnit->numArrayInRow/arrayNeedRow);
			speedUpCol = floor(weightGradientUnit->numArrayInCol/arrayNeedCol)==0? 1:floor(weightGradientUnit->numArrayInCol/arrayNeedCol);
		} else{
			speedUpRow = weightGradientUnit->numArrayInRow;
			speedUpCol = floor(thisMatrixCol/param->numColSubArrayWG)==0? 1:floor(thisMatrixCol/param->numColSubArrayWG);
		}
		double actualUsedArray = thisMatrixRow*thisMatrixCol/(weightGradientUnit->numArrayInRow*weightGradientUnit->numArrayInCol*param->numRowSubArrayWG*param->numColSubArrayWG);
		
		*readLatencyPeakWG = (weightGradientUnit->readLatencyPeak/(speedUpRow*speedUpCol) + weightGradientUnit->writeLatencyPeak)*(netStructure[l][3]*netStructure[l][4]);
		*readDynamicEnergyPeakWG = (weightGradientUnit->readDynamicEnergyPeak + weightGradientUnit->writeDynamicEnergyPeak)*actualUsedArray*(netStructure[l][3]*netStructure[l][4]);
		*readLatencyWG += (*readLatencyPeakWG);
		*readDynamicEnergyWG += (*readDynamicEnergyPeakWG);
		
		// weight gradient need to be send back to DRAM
		*readLatencyWG += dRAM->readLatency + (globalBuffer->readLatency + globalBuffer->writeLatency);
		*readDynamicEnergyWG += dRAM->readDynamicEnergy + (globalBuffer->readDynamicEnergy + globalBuffer->writeDynamicEnergy);
		// Data Transfer during Weight Gradient
		*bufferLatency += (globalBuffer->readLatency + globalBuffer->writeLatency);
		*bufferDynamicEnergy += (globalBuffer->readDynamicEnergy + globalBuffer->writeDynamicEnergy);
		*dramLatency += dRAM->readLatency; 
		*dramDynamicEnergy += dRAM->readDynamicEnergy;
		
		// Before weight update: accumulation of weight gradient
		// need to load weight gradient data from DRAM back to chip
		*readLatencyWG += dRAM->readLatency + (globalBuffer->readLatency + globalBuffer->writeLatency);
		*readDynamicEnergyWG += dRAM->readDynamicEnergy + (globalBuffer->readDynamicEnergy + globalBuffer->writeDynamicEnergy);
		// Data Transfer before weight update
		*bufferLatency += (globalBuffer->readLatency + globalBuffer->writeLatency);
		*bufferDynamicEnergy += (globalBuffer->readDynamicEnergy + globalBuffer->writeDynamicEnergy);
		*dramLatency += dRAM->readLatency; 
		*dramDynamicEnergy += dRAM->readDynamicEnergy;
		
		gradientAccum->CalculateLatency(1e20, 0, ceil(netStructure[l][2]*netStructure[l][3]*netStructure[l][4]*netStructure[l][5]/gradientAccum->numAdder));
		gradientAccum->CalculatePower(ceil(netStructure[l][2]*netStructure[l][3]*netStructure[l][4]*netStructure[l][5]/gradientAccum->numAdder), 
						MIN(netStructure[l][2]*netStructure[l][3]*netStructure[l][4]*netStructure[l][5], gradientAccum->numAdder));
		*readLatencyWG += gradientAccum->readLatency;
		*readDynamicEnergyWG += gradientAccum->readDynamicEnergy;
		*readLatencyPeakWG += gradientAccum->readLatency;
		*readDynamicEnergyPeakWG += gradientAccum->readDynamicEnergy;
		
		// weight gradient also need *batchSize computation
		*readLatencyWG *= param->batchSize;
		*readDynamicEnergyWG *= param->batchSize;
		*readLatencyPeakWG *= param->batchSize;
		*readDynamicEnergyPeakWG *= param->batchSize;
		*bufferLatency *= param->batchSize;
		*bufferDynamicEnergy *= param->batchSize;
		*coreLatencyOther *= param->batchSize;
		*coreEnergyOther *= param->batchSize;
		*dramLatency *= param->batchSize;
		*dramDynamicEnergy *= param->batchSize;
		
		
		// since the latency of weightUpdate is accumulated as seperate subArrays
		// one update every batch
		*writeLatencyPeakWU /= numArrayWriteParallel;
		*writeLatencyWU /= numArrayWriteParallel;
		
		
		// since for each epoch, need *numIteration computation
		*readLatency *= param->numIteration;
		*readDynamicEnergy *= param->numIteration;
		*readLatencyAG *= param->numIteration;
		*readDynamicEnergyAG *= param->numIteration;
		*readLatencyWG *= param->numIteration;
		*readDynamicEnergyWG *= param->numIteration;
		*writeLatencyWU *= param->numIteration;
		*writeDynamicEnergyWU *= param->numIteration;
		
		*readLatencyPeakFW *= param->numIteration;
		*readDynamicEnergyPeakFW *= param->numIteration;
		*readLatencyPeakAG *= param->numIteration;
		*readDynamicEnergyPeakAG *= param->numIteration;
		*readLatencyPeakWG *= param->numIteration;
		*readDynamicEnergyPeakWG *= param->numIteration;
		*writeLatencyPeakWU *= param->numIteration;
		*writeDynamicEnergyPeakWU *= param->numIteration;
		
		*bufferLatency *= param->numIteration;
		*bufferDynamicEnergy *= param->numIteration;
		*icLatency *= param->numIteration;
		*icDynamicEnergy *= param->numIteration;
		
		*coreLatencyADC *= param->numIteration;
		*coreEnergyADC *= param->numIteration;
		*coreLatencyAccum *= param->numIteration;
		*coreEnergyAccum *= param->numIteration;
		*coreLatencyOther *= param->numIteration;
		*coreEnergyOther *= param->numIteration;
		
		*dramLatency *= param->numIteration;
		*dramDynamicEnergy *= param->numIteration;
	} 
	
	*leakage = tileLeakage;
	return 0;
}



vector<double> TileDesignCM(double tileSize, const vector<int > &markNM, const vector<vector<double> > &netStructure, int numRowPerSynapse, int numColPerSynapse) {
	double numTileTotal = 0;
	double matrixTotalCM = 0;
	double utilization = 0;
	for (int i=0; i<netStructure.size(); i++) {
		if (markNM[i] == 0) {
			numTileTotal += ceil((double) netStructure[i][2]*(double) netStructure[i][3]*(double) netStructure[i][4]*(double) numRowPerSynapse/(double) tileSize) * ceil(netStructure[i][5]*numColPerSynapse/tileSize);
			matrixTotalCM += netStructure[i][2]*netStructure[i][3]*netStructure[i][4]*numRowPerSynapse*netStructure[i][5]*numColPerSynapse;
		}
	}
	utilization = matrixTotalCM/(numTileTotal*tileSize*tileSize);
	
	vector<double> tileDesignCM;
	tileDesignCM.push_back(numTileTotal);
	tileDesignCM.push_back(matrixTotalCM);
	tileDesignCM.push_back(utilization);
	return tileDesignCM;
	tileDesignCM.clear();
}

vector<double> TileDesignNM(double peSize, const vector<int > &markNM, const vector<vector<double> > &netStructure, int numRowPerSynapse, int numColPerSynapse, double numPENM){
	double numTileTotal = 0;
	double matrixTotalNM = 0;
	double utilization = 0;
	for (int i=0; i<netStructure.size(); i++) {
		if (markNM[i] == 1) {
			numTileTotal += ceil((double) netStructure[i][2]*(double) numRowPerSynapse/(double) peSize) * ceil((double) netStructure[i][5]*(double) numColPerSynapse/(double) peSize);
			matrixTotalNM += netStructure[i][2]*netStructure[i][3]*netStructure[i][4]*numRowPerSynapse*netStructure[i][5]*numColPerSynapse;
		}
	}
	utilization = matrixTotalNM/(numTileTotal*peSize*peSize*numPENM);
	vector<double> tileDesignNM;
	tileDesignNM.push_back(numTileTotal);
	tileDesignNM.push_back(matrixTotalNM);
	tileDesignNM.push_back(utilization);
	return tileDesignNM;
	tileDesignNM.clear();
}

vector<vector<double> > PEDesign(bool Design, double peSize, double desiredTileSize, double numTileTotal, const vector<int > &markNM, const vector<vector<double> > &netStructure, int numRowPerSynapse, int numColPerSynapse) {
	double matrixTotalCM = 0;
	double utilization = 0;
	vector<double> peDupRow;
	vector<double> peDupCol;
	for (int i=0; i<netStructure.size(); i++) {
		int actualDupRow = 0;
		int actualDupCol = 0;
		if (markNM[i] ==0) {
			if ( (netStructure[i][2]*netStructure[i][3]*netStructure[i][4]*numRowPerSynapse <= desiredTileSize)||(netStructure[i][5]*numColPerSynapse <= desiredTileSize) ) {
				int peForOneMatrixRow = ceil((double) netStructure[i][2]*(double) netStructure[i][3]*(double) netStructure[i][4]*(double) numRowPerSynapse/(double) peSize);
				int peForOneMatrixCol = ceil((double) netStructure[i][5]*(double) numColPerSynapse/(double) peSize);
				int numPERow = ceil((double) desiredTileSize/(double) peSize);
				int numPECol = ceil((double) desiredTileSize/(double) peSize);
				actualDupRow = floor(numPERow/peForOneMatrixRow)==0? 1:floor(numPERow/peForOneMatrixRow);
				actualDupCol = floor(numPECol/peForOneMatrixCol)==0? 1:floor(numPECol/peForOneMatrixCol);
				matrixTotalCM += actualDupRow*actualDupCol*netStructure[i][2]*netStructure[i][3]*netStructure[i][4]*numRowPerSynapse*netStructure[i][5]*numColPerSynapse;
			} else {
				actualDupRow = 1;
				actualDupCol = 1;
				matrixTotalCM += actualDupRow*actualDupCol*netStructure[i][2]*netStructure[i][3]*netStructure[i][4]*numRowPerSynapse*netStructure[i][5]*numColPerSynapse;
			}
		} else {
			actualDupRow = 1;
			actualDupCol = 1;
		}
		peDupRow.push_back(actualDupRow);
		peDupCol.push_back(actualDupCol);
	}
	utilization = matrixTotalCM/(numTileTotal*desiredTileSize*desiredTileSize);
	
	vector<double> matrixTotal;
	matrixTotal.push_back(matrixTotalCM);
	vector<double> utiliz;
	utiliz.push_back(utilization);
	vector<vector<double> > peDesignCM;
	peDesignCM.push_back(matrixTotal);
	peDesignCM.push_back(utiliz);
	matrixTotal.clear();
	utiliz.clear();
	
	vector<vector<double> > peDup;
	peDup.push_back(peDupRow);
	peDup.push_back(peDupCol);
	peDupRow.clear();
	peDupCol.clear();
	if (Design) {
		return peDesignCM;
	} else {
		return peDup;
	}
	peDesignCM.clear();
	peDup.clear();
}

vector<vector<double> > SubArrayDup(double desiredPESizeCM, double desiredPESizeNM, const vector<int > &markNM, const vector<vector<double> > &netStructure, int numRowPerSynapse, int numColPerSynapse) {
	vector<double> subArrayDupRow;
	vector<double> subArrayDupCol;
	
	for (int i=0; i<netStructure.size(); i++) {
		int actualDupRow = 0;
		int actualDupCol = 0;
		if (markNM[i] == 0){
			if ( (netStructure[i][2]*netStructure[i][3]*netStructure[i][4]*numRowPerSynapse <= desiredPESizeCM)||(netStructure[i][5]*numColPerSynapse <= desiredPESizeCM) ) {
				int arrayForOneMatrixRow = ceil((double) netStructure[i][2]*(double) netStructure[i][3]*(double) netStructure[i][4]*(double) numRowPerSynapse/(double) param->numRowSubArray);
				int arrayForOneMatrixCol = ceil((double) netStructure[i][5]*(double) numColPerSynapse/(double) param->numColSubArray);
				int numSubArrayRow = ceil((double) desiredPESizeCM/(double) param->numRowSubArray);
				int numSubArrayCol = ceil((double) desiredPESizeCM/(double) param->numColSubArray);
				actualDupRow = floor(numSubArrayRow/arrayForOneMatrixRow)==0? 1:floor(numSubArrayRow/arrayForOneMatrixRow);
				actualDupCol = floor(numSubArrayCol/arrayForOneMatrixCol)==0? 1:floor(numSubArrayCol/arrayForOneMatrixCol);
			} else {
				actualDupRow = 1;
				actualDupCol = 1;
			}
		} else {
			if ( (netStructure[i][2]*numRowPerSynapse <= desiredPESizeNM)||(netStructure[i][5]*numColPerSynapse <= desiredPESizeNM) ) {
				int arrayForOneMatrixRow = ceil((double) netStructure[i][2]*(double) numRowPerSynapse/(double) param->numRowSubArray);
				int arrayForOneMatrixCol = ceil((double) netStructure[i][5]*(double) numColPerSynapse/(double) param->numColSubArray);
				int numSubArrayRow = ceil((double) desiredPESizeNM/(double) param->numRowSubArray);
				int numSubArrayCol = ceil((double) desiredPESizeNM/(double) param->numColSubArray);
				actualDupRow = floor(numSubArrayRow/arrayForOneMatrixRow)==0? 1:floor(numSubArrayRow/arrayForOneMatrixRow);
				actualDupCol = floor(numSubArrayCol/arrayForOneMatrixCol)==0? 1:floor(numSubArrayCol/arrayForOneMatrixCol);
			} else {
				actualDupRow = 1;
				actualDupCol = 1;
			}
		}
		subArrayDupRow.push_back(actualDupRow);
		subArrayDupCol.push_back(actualDupCol);
	}
	vector<vector<double> > subArrayDup;
	subArrayDup.push_back(subArrayDupRow);
	subArrayDup.push_back(subArrayDupCol);
	subArrayDupRow.clear();
	subArrayDupCol.clear();
	return subArrayDup;
	subArrayDup.clear();
}

vector<vector<double> > OverallEachLayer(bool utilization, bool speedUp, const vector<vector<double> > &peDup, const vector<vector<double> > &subArrayDup, const vector<int> &pipelineSpeedUp, double desiredTileSizeCM, 
										double desiredPESizeNM, const vector<int > &markNM, const vector<vector<double> > &netStructure, int numRowPerSynapse, int numColPerSynapse, double numPENM) {
	vector<double> numTileEachLayerRow;
	vector<double> numTileEachLayerCol;
	vector<vector<double> > utilizationEachLayer;	
	vector<double> speedUpEachLayerRow;
	vector<double> speedUpEachLayerCol;
	
	for (int i=0; i<netStructure.size(); i++) {
		vector<double> utilization;
		double numtileEachLayerRow, numtileEachLayerCol, utilizationEach;
		if (markNM[i] == 0) {
			// conventional mapping
			if (!param->pipeline) {
				// layer-by-layer process
				numtileEachLayerRow = ceil((double) netStructure[i][2]*(double) netStructure[i][3]*(double) netStructure[i][4]*(double) numRowPerSynapse/desiredTileSizeCM);
				numtileEachLayerCol = ceil((double) netStructure[i][5]*(double) numColPerSynapse/(double) desiredTileSizeCM);
				utilizationEach = (peDup[0][i]*peDup[1][i]*subArrayDup[0][i]*subArrayDup[1][i]*netStructure[i][2]*netStructure[i][3]*netStructure[i][4]
											*numRowPerSynapse*netStructure[i][5]*numColPerSynapse)/(numtileEachLayerRow*numtileEachLayerCol*desiredTileSizeCM*desiredTileSizeCM);

				utilization.push_back(utilizationEach);
			} else {
				// pipeline system
				// original design
				numtileEachLayerRow = ceil((double) netStructure[i][2]*(double) netStructure[i][3]*(double) netStructure[i][4]*(double) numRowPerSynapse/desiredTileSizeCM)
										*ceil(pipelineSpeedUp[i]/(peDup[0][i]*peDup[1][i]*subArrayDup[0][i]*subArrayDup[1][i]));
				numtileEachLayerCol = ceil((double) netStructure[i][5]*(double) numColPerSynapse/(double) desiredTileSizeCM);

				utilizationEach = (MAX(pipelineSpeedUp[i], peDup[0][i]*peDup[1][i]*subArrayDup[0][i]*subArrayDup[1][i])*netStructure[i][2]*netStructure[i][3]*netStructure[i][4]
									*numRowPerSynapse*netStructure[i][5]*numColPerSynapse)/(numtileEachLayerRow*numtileEachLayerCol*desiredTileSizeCM*desiredTileSizeCM);

				utilization.push_back(utilizationEach);
			}
		} else {
			if (!param->pipeline) {
				// novel mapping
				numtileEachLayerRow = ceil((double) netStructure[i][2]*(double) numRowPerSynapse/(double) desiredPESizeNM);
				numtileEachLayerCol = ceil((double) netStructure[i][5]*(double) numColPerSynapse/(double) desiredPESizeNM);
				utilizationEach = (peDup[0][i]*peDup[1][i]*subArrayDup[0][i]*subArrayDup[1][i]*netStructure[i][2]*numPENM*numRowPerSynapse*netStructure[i][5]
											*numColPerSynapse)/(numtileEachLayerRow*numtileEachLayerCol*desiredPESizeNM*desiredPESizeNM*numPENM);
				
				utilization.push_back(utilizationEach);
			} else {
				// novel mapping
				numtileEachLayerRow = ceil((double) netStructure[i][2]*(double) numRowPerSynapse/(double) desiredPESizeNM)
										*ceil(pipelineSpeedUp[i]/(peDup[0][i]*peDup[1][i]*subArrayDup[0][i]*subArrayDup[1][i]));
				numtileEachLayerCol = ceil((double) netStructure[i][5]*(double) numColPerSynapse/(double) desiredPESizeNM);
				utilizationEach = (MAX(pipelineSpeedUp[i], peDup[0][i]*peDup[1][i]*subArrayDup[0][i]*subArrayDup[1][i])*netStructure[i][2]*numPENM*numRowPerSynapse*netStructure[i][5]
											*numColPerSynapse)/(numtileEachLayerRow*numtileEachLayerCol*desiredPESizeNM*desiredPESizeNM*numPENM);
				
				utilization.push_back(utilizationEach);
			}
		}
		numTileEachLayerRow.push_back(numtileEachLayerRow);
		numTileEachLayerCol.push_back(numtileEachLayerCol);
		utilizationEachLayer.push_back(utilization);
		if (!param->pipeline) {
			speedUpEachLayerRow.push_back(peDup[0][i]*subArrayDup[0][i]);
			speedUpEachLayerCol.push_back(peDup[1][i]*subArrayDup[1][i]);
		} else {
			speedUpEachLayerRow.push_back(peDup[0][i]*subArrayDup[0][i]*ceil(pipelineSpeedUp[i]/(peDup[0][i]*peDup[1][i]*subArrayDup[0][i]*subArrayDup[1][i])));
			speedUpEachLayerCol.push_back(peDup[1][i]*subArrayDup[1][i]);
		}
		utilization.clear();
	}

	vector<vector<double> > numTileEachLayer;
	numTileEachLayer.push_back(numTileEachLayerRow);
	numTileEachLayer.push_back(numTileEachLayerCol);
	numTileEachLayerRow.clear();
	numTileEachLayerCol.clear();
	
	vector<vector<double> > speedUpEachLayer;
	speedUpEachLayer.push_back(speedUpEachLayerRow);
	speedUpEachLayer.push_back(speedUpEachLayerCol);
	speedUpEachLayerRow.clear();
	speedUpEachLayerCol.clear();

	if (utilization) {
		return utilizationEachLayer;
	} else if (speedUp) {
		return speedUpEachLayer;
	} else {
		return numTileEachLayer;
	}
	utilizationEachLayer.clear();
	speedUpEachLayer.clear();
	numTileEachLayer.clear();
}



vector<vector<double> > LoadInWeightData(const string &weightfile, int numRowPerSynapse, int numColPerSynapse, double maxConductance, double minConductance) {
	
	ifstream fileone(weightfile.c_str());                           
	string lineone;
	string valone;
	
	int ROW = 0;
	int COL = 0;
	
	if (!fileone.good()) {                                       
		cerr << "Error: the fileone cannot be opened!" << endl;
		exit(1);
	}else{
		while (getline(fileone, lineone, '\n')) {                   
			ROW++;                                             
		}
		fileone.clear();
		fileone.seekg(0, ios::beg);                               
		if (getline(fileone, lineone, '\n')) {                      
			istringstream iss (lineone);                         
			while (getline(iss, valone, ',')) {                   
				COL++;
			}
		}	
	}
	fileone.clear();
	fileone.seekg(0, ios::beg);                   
	
	double NormalizedMin = 0;
	double NormalizedMax = pow(2, param->synapseBit);
	
	double RealMax = param->algoWeightMax;
	double RealMin = param->algoWeightMin;
	
	vector<vector<double> > weight;            
	// load the data into a weight matrix ...
	for (int row=0; row<ROW; row++) {	
		vector<double> weightrow;
		vector<double> weightrowb;
		getline(fileone, lineone, '\n');              
		istringstream iss;
		iss.str(lineone);
		for (int col=0; col<COL; col++) {       
			while(getline(iss, valone, ',')){	
				istringstream fs;
				fs.str(valone);
				double f=0;
				fs >> f;	
				if ((param->memcelltype != 1)&&(param->synapseBit == param->cellBit)) { // training version: linear mapping
					weightrow.push_back((f+1)/2*(maxConductance-minConductance)+minConductance);
				} else {
					//normalize weight to integer
					double newdata = ((NormalizedMax-NormalizedMin)/(RealMax-RealMin)*(f-RealMax)+NormalizedMax);
					if (newdata >= 0) {
						newdata += 0.5;
					}else {
						newdata -= 0.5;
					}
					// map and expend the weight in memory array
					int cellrange = pow(2, param->cellBit);
					vector<int> synapsevector(numColPerSynapse);       
					int value = newdata; 
					if (param->BNNparallelMode) {
						if (value == 1) {
							weightrow.push_back(maxConductance);
							weightrow.push_back(minConductance);
						} else {
							weightrow.push_back(minConductance);
							weightrow.push_back(maxConductance);
						}
					} else if (param->XNORparallelMode || param->XNORsequentialMode) {
						if (value == 1) {
							weightrow.push_back(maxConductance);
							weightrowb.push_back(minConductance);
						} else {
							weightrow.push_back(minConductance);
							weightrowb.push_back(maxConductance);
						}
					} else {
						int remainder;   
						for (int z=0; z<numColPerSynapse; z++) {   
							remainder = (int) value%cellrange;
							value = (int) value/cellrange;
							synapsevector.insert(synapsevector.begin(), value/*remainder*/);
						}
						for (int u=0; u<numColPerSynapse; u++) {
							int cellvalue = synapsevector[u];
							double conductance = cellvalue/(cellrange-1) * (maxConductance-minConductance) + minConductance;
							weightrow.push_back(conductance);
						}
					}
				}
			}
		}
		if (param->XNORparallelMode || param->XNORsequentialMode) {
			weight.push_back(weightrow);
			weightrow.clear();
			weight.push_back(weightrowb);
			weightrowb.clear();
		} else {
			weight.push_back(weightrow);
			weightrow.clear();
		}
	}
	fileone.close();
	
	return weight;
	weight.clear();
}



vector<vector<double> > CopyArray(const vector<vector<double> > &orginal, int positionRow, int positionCol, int numRow, int numCol) {
	
	vector<vector<double> > copy;
	for (int i=0; i<numRow; i++) {
		vector<double> copyRow;
		for (int j=0; j<numCol; j++) {
			copyRow.push_back(orginal[positionRow+i][positionCol+j]);
		}
		copy.push_back(copyRow);
		copyRow.clear();
	}
	
	return copy;
	copy.clear();
} 



vector<vector<double> > ReshapeArray(const vector<vector<double> > &orginal, int positionRow, int positionCol, int numRow, int numCol, int numPE, int weightMatrixRow) {
	
	vector<vector<double> > copy;

	for (int k=0; k<numPE; k++) {
		for (int i=0; i<numRow; i++) {
			vector<double> copyRow;
			for (int j=0; j<numCol; j++) {
				copyRow.push_back(orginal[positionRow+k*weightMatrixRow+i][positionCol+j]);
			}
			copy.push_back(copyRow);
			copyRow.clear();
		}
	}
	
	return copy;
	copy.clear();
} 



vector<vector<double> > LoadInInputData(const string &inputfile) {
	
	ifstream infile(inputfile.c_str());     
	string inputline;
	string inputval;
	
	int ROWin=0, COLin=0;      
	if (!infile.good()) {       
		cerr << "Error: the input file cannot be opened!" << endl;
		exit(1);
	}else{
		while (getline(infile, inputline, '\n')) {      
			ROWin++;                               
		}
		infile.clear();
		infile.seekg(0, ios::beg);    
		if (getline(infile, inputline, '\n')) {        
			istringstream iss (inputline);      
			while (getline(iss, inputval, ',')) {       
				COLin++;
			}
		}	
	}
	infile.clear();
	infile.seekg(0, ios::beg);          
	
	vector<vector<double> > inputvector;              
	// load the data into inputvector ...
	for (int row=0; row<ROWin; row++) {	
		vector<double> inputvectorrow;
		vector<double> inputvectorrowb;
		getline(infile, inputline, '\n');             
		istringstream iss;
		iss.str(inputline);
		for (int col=0; col<COLin; col++) {
			while(getline(iss, inputval, ',')){	
				istringstream fs;
				fs.str(inputval);
				double f=0;
				fs >> f;
				
				if (param->BNNparallelMode) {
					if (f == 1) {
						inputvectorrow.push_back(1);
					} else {
						inputvectorrow.push_back(0);
					}
				} else if (param->XNORparallelMode || param->XNORsequentialMode) {
					if (f == 1) {
						inputvectorrow.push_back(1);
						inputvectorrowb.push_back(0);
					} else {
						inputvectorrow.push_back(0);
						inputvectorrowb.push_back(1);
					}
				} else {
					inputvectorrow.push_back(f);
				}
			}
		}
		if (param->XNORparallelMode || param->XNORsequentialMode) {
			inputvector.push_back(inputvectorrow);
			inputvectorrow.clear();
			inputvector.push_back(inputvectorrowb);
			inputvectorrowb.clear();
		} else {
			inputvector.push_back(inputvectorrow);
			inputvectorrow.clear();
		}
	}
	// close the input file ...
	infile.close();
	
	return inputvector;
	inputvector.clear();
}



vector<vector<double> > CopyInput(const vector<vector<double> > &orginal, int positionRow, int numInputVector, int numRow) {
	
	vector<vector<double> > copy;
	for (int i=0; i<numRow; i++) {
		vector<double> copyRow;
		for (int j=0; j<numInputVector; j++) {
			copyRow.push_back(orginal[positionRow+i][j]);
		}
		copy.push_back(copyRow);
		copyRow.clear();
	}
	
	return copy;
	copy.clear();
	
} 



vector<vector<double> > ReshapeInput(const vector<vector<double> > &orginal, int positionRow, int numInputVector, int numRow, int numPE, int weightMatrixRow) {
	
	vector<vector<double> > copy;

	for (int k=0; k<numPE; k++) {
		for (int i=0; i<numRow; i++) {
			vector<double> copyRow;
			for (int j=0; j<numInputVector; j++) {
				copyRow.push_back(orginal[positionRow+k*weightMatrixRow+i][j]);
			}
			copy.push_back(copyRow);
			copyRow.clear();
		}
	}
	
	return copy;
	copy.clear();
} 
