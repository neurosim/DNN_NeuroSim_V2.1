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
#include "Sigmoid.h"
#include "BitShifter.h"
#include "AdderTree.h"
#include "Buffer.h"
#include "HTree.h"
#include "ProcessingUnit.h"
#include "SubArray.h"
#include "constant.h"
#include "formula.h"
#include "Param.h"
#include "Tile.h"

using namespace std;

extern Param *param;
int numInBufferCore = 0;
int numOutBufferCore = 0;

SubArray *subArrayInPE;
Buffer *inputBuffer;
Buffer *outputBuffer;
HTree *hTree;
AdderTree *accumulation;
Sigmoid *sigmoid;
BitShifter *reLu;


void TileInitialize(InputParameter& inputParameter, Technology& tech, MemCell& cell, double _numPE, double _peSize){
	
	subArrayInPE = new SubArray(inputParameter, tech, cell);
	inputBuffer = new Buffer(inputParameter, tech, cell);
	outputBuffer = new Buffer(inputParameter, tech, cell);
	hTree = new HTree(inputParameter, tech, cell);
	accumulation = new AdderTree(inputParameter, tech, cell);
	
	if (!param->chipActivation) {
		if (param->reLu) {
			reLu = new BitShifter(inputParameter, tech, cell);
		} else {
			sigmoid = new Sigmoid(inputParameter, tech, cell);
		}
	}
	
	/*** Parameters ***/
	double numPE, peSize, numSubArray;
	int numRowPerSynapse, numColPerSynapse;
	
	numPE = _numPE;
	peSize = _peSize;
	numRowPerSynapse = param->numRowPerSynapse;
	numColPerSynapse = param->numColPerSynapse;
	
	/*** Initialize ProcessingUnit ***/
	numSubArray = ceil((double)peSize/(double)param->numRowSubArray)*ceil((double)peSize/(double)param->numColSubArray);
	ProcessingUnitInitialize(subArrayInPE, inputParameter, tech, cell, ceil(sqrt(numSubArray)), ceil(sqrt(numSubArray)));
	
	if (param->parallelRead) {
		accumulation->Initialize(numPE, ceil((double)log2((double)param->levelOutput))+param->numBitInput+1+ceil((double)log2((double)peSize/(double)param->numRowSubArray)), 
								ceil((double)numPE*(double)param->numColSubArray/(double)param->numColMuxed));
		if (!param->chipActivation) {
			if (param->reLu) {
				reLu->Initialize(ceil((double)peSize*(double)param->numColSubArray/(double)param->numColMuxed), param->numBitInput, param->clkFreq);
			} else {
				sigmoid->Initialize(false, param->numBitInput, ceil((double)log2((double)param->levelOutput))+param->numBitInput+1+ceil((double)log2((double)peSize/(double)param->numRowSubArray))+ceil((double)log2((double)numPE)), 
								ceil((double)numPE*(double)param->numColSubArray/(double)param->numColMuxed), param->clkFreq);
			}
			//outputBuffer->Initialize(param->numBitInput*numPE*param->numColSubArray/param->numColMuxed, param->numBitInput*numPE, 1, param->unitLengthWireResistance, param->clkFreq, param->peBufferType);
			numOutBufferCore = ceil((param->numBitInput*numPE*param->numColSubArray/param->numColMuxed)/(param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol));
			
			if ((param->numBitInput*numPE*param->numColSubArray/param->numColMuxed) < (param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol)) {
				outputBuffer->Initialize(param->numBitInput*numPE*param->numColSubArray/param->numColMuxed, param->numBitInput*numPE, 1, param->unitLengthWireResistance, param->clkFreq, param->peBufferType);
			} else {
				outputBuffer->Initialize((param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol), param->tileBufferCoreSizeCol, 1, param->unitLengthWireResistance, param->clkFreq, param->peBufferType);
			}																																											 
		} else {
			//outputBuffer->Initialize((ceil((double)log2((double)param->levelOutput))+param->numBitInput+1+ceil((double)log2((double)peSize/(double)param->numRowSubArray)))*numPE*param->numColSubArray/param->numColMuxed, 
								//(ceil((double)log2((double)param->levelOutput))+param->numBitInput+1+ceil((double)log2((double)peSize/(double)param->numRowSubArray)))*numPE*2, 
								//1, param->unitLengthWireResistance, param->clkFreq, param->peBufferType);
			numOutBufferCore = ceil(((ceil((double)log2((double)param->levelOutput))+param->numBitInput+1+ceil((double)log2((double)peSize/(double)param->numRowSubArray)))*numPE*param->numColSubArray/param->numColMuxed)/(param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol));
			if (((ceil((double)log2((double)param->levelOutput))+param->numBitInput+1+ceil((double)log2((double)peSize/(double)param->numRowSubArray)))*numPE*param->numColSubArray/param->numColMuxed) < (param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol)) {
				outputBuffer->Initialize((ceil((double)log2((double)param->levelOutput))+param->numBitInput+1+ceil((double)log2((double)peSize/(double)param->numRowSubArray)))*numPE*param->numColSubArray/param->numColMuxed, 
								(ceil((double)log2((double)param->levelOutput))+param->numBitInput+1+ceil((double)log2((double)peSize/(double)param->numRowSubArray)))*numPE, 
								1, param->unitLengthWireResistance, param->clkFreq, param->peBufferType);
			} else {
				outputBuffer->Initialize((param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol), param->tileBufferCoreSizeCol, 1, param->unitLengthWireResistance, param->clkFreq, param->peBufferType);
			}
		}
	} else {
		accumulation->Initialize(numPE, ceil((double)log2((double)param->numRowSubArray)+(double)param->cellBit-1)+param->numBitInput+1+ceil((double)log2((double)peSize/(double)param->numRowSubArray)), 
								ceil(numPE*(double)param->numColSubArray/(double)param->numColMuxed));
		if (!param->chipActivation) {
			if (param->reLu) {
				reLu->Initialize(ceil((double)peSize*(double)param->numColSubArray/(double)param->numColMuxed), param->numBitInput, param->clkFreq);
			} else {
				sigmoid->Initialize(false, param->numBitInput, ceil((double)log2((double)param->numRowSubArray)+(double)param->cellBit-1)+param->numBitInput+1+ceil((double)log2((double)peSize/(double)param->numRowSubArray))+ceil((double)log2((double)numPE)), 
								ceil(numPE*(double)param->numColSubArray/(double)param->numColMuxed), param->clkFreq);
			}
			//outputBuffer->Initialize(param->numBitInput*numPE*param->numColSubArray/param->numColMuxed, param->numBitInput*numPE, 1, param->unitLengthWireResistance, param->clkFreq, param->peBufferType);
			numOutBufferCore = ceil((param->numBitInput*numPE*param->numColSubArray/param->numColMuxed)/(param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol));
			if ((param->numBitInput*numPE*param->numColSubArray/param->numColMuxed) < (param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol)) {
				outputBuffer->Initialize(param->numBitInput*numPE*param->numColSubArray/param->numColMuxed, param->numBitInput*numPE, 1, param->unitLengthWireResistance, param->clkFreq, param->peBufferType);
			} else {
				outputBuffer->Initialize((param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol), param->tileBufferCoreSizeCol, 1, param->unitLengthWireResistance, param->clkFreq, param->peBufferType);
			}
		} else {
				//outputBuffer->Initialize((ceil((double)log2((double)param->numRowSubArray)+(double)param->cellBit-1)+param->numBitInput+1+ceil((double)log2((double)peSize/(double)param->numRowSubArray)))*numPE*param->numColSubArray/param->numColMuxed, 
								//(ceil((double)log2((double)param->numRowSubArray)+(double)param->cellBit-1)+param->numBitInput+1+ceil((double)log2((double)peSize/(double)param->numRowSubArray)))*numPE, 
								//1, param->unitLengthWireResistance, param->clkFreq, param->peBufferType);
			numOutBufferCore = ceil(((ceil((double)log2((double)param->numRowSubArray)+(double)param->cellBit-1)+param->numBitInput+1+ceil((double)log2((double)peSize/(double)param->numRowSubArray)))*numPE*param->numColSubArray/param->numColMuxed)/(param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol));
			if (((ceil((double)log2((double)param->numRowSubArray)+(double)param->cellBit-1)+param->numBitInput+1+ceil((double)log2((double)peSize/(double)param->numRowSubArray)))*numPE*param->numColSubArray/param->numColMuxed) < (param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol)) {
				outputBuffer->Initialize((ceil((double)log2((double)param->numRowSubArray)+(double)param->cellBit-1)+param->numBitInput+1+ceil((double)log2((double)peSize/(double)param->numRowSubArray)))*numPE*param->numColSubArray/param->numColMuxed, 
								(ceil((double)log2((double)param->numRowSubArray)+(double)param->cellBit-1)+param->numBitInput+1+ceil((double)log2((double)peSize/(double)param->numRowSubArray)))*numPE, 
								1, param->unitLengthWireResistance, param->clkFreq, param->peBufferType);
			} else {
				outputBuffer->Initialize((param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol), param->tileBufferCoreSizeCol, 1, param->unitLengthWireResistance, param->clkFreq, param->peBufferType);
			}
		}
	}
	
	//inputBuffer->Initialize(numPE*param->numBitInput*param->numRowSubArray, numPE*param->numRowSubArray, 1, param->unitLengthWireResistance, param->clkFreq, param->peBufferType);
	numInBufferCore = ceil((numPE*param->numBitInput*param->numRowSubArray)/(param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol));
	
	if ((numPE*param->numBitInput*param->numRowSubArray) < (param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol)) {
		inputBuffer->Initialize(numPE*param->numBitInput*param->numRowSubArray, numPE*param->numRowSubArray, 1, param->unitLengthWireResistance, param->clkFreq, param->peBufferType);
	} else {
		inputBuffer->Initialize((param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol), param->tileBufferCoreSizeCol, 1, param->unitLengthWireResistance, param->clkFreq, param->peBufferType);
	}
	hTree->Initialize(numPE, numPE, param->localBusDelayTolerance, numPE*param->numRowSubArray);
}


vector<double> TileCalculateArea(double numPE, double peSize, double *height, double *width) {
	double area = 0;
	double PEheight, PEwidth, PEbufferArea;
	*height = 0;
	*width = 0;
	vector<double> areaResults;
	vector<double> peAreaResults;

	int numSubArray = ceil((double) peSize/(double) param->numRowSubArray)*ceil((double) peSize/(double) param->numColSubArray);
	peAreaResults = ProcessingUnitCalculateArea(subArrayInPE, ceil((double)sqrt((double)numSubArray)), ceil((double)sqrt((double)numSubArray)), &PEheight, &PEwidth, &PEbufferArea);
	double PEarea = peAreaResults[0];
	double PEareaADC = peAreaResults[1];
	double PEareaAccum = peAreaResults[2];
	double PEareaOther = peAreaResults[3];
	double PEareaArray = peAreaResults[4];
	
	double areareLu = 0;
	double areasigmoid = 0;
	
	accumulation->CalculateArea(NULL, ceil(sqrt((double)numPE))*PEwidth, NONE);
	if (!param->chipActivation) {
		if (param->reLu) {
			reLu->CalculateArea(NULL, ceil(sqrt((double)numPE))*PEwidth, NONE);
			area += reLu->area;
			areareLu = reLu->area;
		} else {
			sigmoid->CalculateUnitArea(NONE);
			sigmoid->CalculateArea(NULL, ceil(sqrt((double)numPE))*PEwidth, NONE);
			area += sigmoid->area;
			areasigmoid = sigmoid->area;
		}
	}
	inputBuffer->CalculateArea(ceil(sqrt((double)numPE))*PEheight, NULL, NONE);
	outputBuffer->CalculateArea(NULL, ceil(sqrt((double)numPE))*PEwidth, NONE);
	inputBuffer->area *= numInBufferCore;
	outputBuffer->area *= numOutBufferCore;								  
	hTree->CalculateArea(PEheight, PEwidth, 16);
	
	area += PEarea*numPE + accumulation->area + inputBuffer->area + outputBuffer->area + hTree->area;
	
	*height = sqrt(area);
	*width = area/(*height);
	
	areaResults.push_back(area);
	areaResults.push_back(hTree->area);
	areaResults.push_back(PEareaADC*numPE);
	areaResults.push_back(PEareaAccum*numPE + accumulation->area);
	areaResults.push_back(PEareaOther*numPE + inputBuffer->area + outputBuffer->area + areareLu + areasigmoid);
	areaResults.push_back(PEareaArray*numPE);
	
	return areaResults;
}


void TileCalculatePerformance(const vector<vector<double> > &newMemory, const vector<vector<double> > &oldMemory, const vector<vector<double> > &inputVector, int novelMap, double numPE, 
							double peSize, int speedUpRow, int speedUpCol, int weightMatrixRow, int weightMatrixCol, int numInVector, Technology& tech, MemCell& cell, 
							double *readLatency, double *readDynamicEnergy, double *leakage, double *readLatencyAG, double *readDynamicEnergyAG, double *writeLatencyWU, double *writeDynamicEnergyWU,
							double *bufferLatency, double *bufferDynamicEnergy, double *icLatency, double *icDynamicEnergy,
							double *coreLatencyADC, double *coreLatencyAccum, double *coreLatencyOther, double *coreEnergyADC, 
							double *coreEnergyAccum, double *coreEnergyOther, double *readLatencyPeakFW, double *readDynamicEnergyPeakFW,
							double *readLatencyPeakAG, double *readDynamicEnergyPeakAG, double *writeLatencyPeakWU, double *writeDynamicEnergyPeakWU) {

	/*** sweep PE ***/
	int numRowPerSynapse, numColPerSynapse;
	numRowPerSynapse = param->numRowPerSynapse;
	numColPerSynapse = param->numColPerSynapse;
	double PEreadLatency, PEreadDynamicEnergy, PEleakage, PEbufferLatency, PEbufferDynamicEnergy, PEicLatency, PEicDynamicEnergy;
	double PEreadLatencyAG, PEreadDynamicEnergyAG, PEwriteLatencyWU, PEwriteDynamicEnergyWU;
	double peLatencyADC, peLatencyAccum, peLatencyOther, peEnergyADC, peEnergyAccum, peEnergyOther;
	double peReadLatencyPeakFW, peReadDynamicEnergyPeakFW, peReadLatencyPeakAG, peReadDynamicEnergyPeakAG, peWriteLatencyPeakWU, peWriteDynamicEnergyPeakWU;
	int numSubArrayRow = ceil((double)peSize/(double)param->numRowSubArray);
	int numSubArrayCol = ceil((double)peSize/(double)param->numColSubArray);
	
	*readLatency = 0;
	*readDynamicEnergy = 0;
	*readLatencyAG = 0;
	*readDynamicEnergyAG = 0;
	*writeLatencyWU = 0;
	*writeDynamicEnergyWU = 0;
	
	*readLatencyPeakFW = 0;
	*readDynamicEnergyPeakFW = 0;
	*readLatencyPeakAG = 0;
	*readDynamicEnergyPeakAG = 0;
	*writeLatencyPeakWU = 0;
	*writeDynamicEnergyPeakWU = 0;
	
	*leakage = 0;
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
	
	if (!novelMap) {   // conventional Mapping
		if (speedUpRow*speedUpCol > 1) {
			if ((speedUpRow >= numPE) && (speedUpCol >= numPE)) {
				// duplication in PE or subArray --> tell each PE to take the whole assigned weight  --> "fully" duplication
				// assign weight and input to specific tile
				vector<vector<double> > pEMemoryOld;
				pEMemoryOld = CopyPEArray(oldMemory, 0, 0, weightMatrixRow, weightMatrixCol);
				vector<vector<double> > pEMemory;
				pEMemory = CopyPEArray(newMemory, 0, 0, weightMatrixRow, weightMatrixCol);
				vector<vector<double> > pEInput;
				pEInput = CopyPEInput(inputVector, 0, numInVector, weightMatrixRow);
				
				ProcessingUnitCalculatePerformance(subArrayInPE, tech, cell, pEMemory, pEMemoryOld, pEInput, ceil((double)speedUpRow/(double)numPE), ceil((double)speedUpCol/(double)numPE), 
											numSubArrayRow, numSubArrayCol, weightMatrixRow, weightMatrixCol, numInVector, &PEreadLatency, &PEreadDynamicEnergy, &PEleakage,
											&PEreadLatencyAG, &PEreadDynamicEnergyAG, &PEwriteLatencyWU, &PEwriteDynamicEnergyWU,
											&PEbufferLatency, &PEbufferDynamicEnergy, &PEicLatency, &PEicDynamicEnergy,
											&peLatencyADC, &peLatencyAccum, &peLatencyOther, &peEnergyADC, &peEnergyAccum, &peEnergyOther, 
											&peReadLatencyPeakFW, &peReadDynamicEnergyPeakFW, &peReadLatencyPeakAG, &peReadDynamicEnergyPeakAG,
											&peWriteLatencyPeakWU, &peWriteDynamicEnergyPeakWU);
				
				*readLatency = PEreadLatency/(numPE*numPE);  // further speed up in PE level
				*readDynamicEnergy = PEreadDynamicEnergy;   // since subArray.cpp takes all input vectors, no need to *numPE here
				*readLatencyAG = PEreadLatencyAG/(numPE*numPE);
				*readDynamicEnergyAG = PEreadDynamicEnergyAG;
				*writeLatencyWU = PEwriteLatencyWU*(numPE*numPE);
				*writeDynamicEnergyWU = PEwriteDynamicEnergyWU*(numPE*numPE);
				
				*readLatencyPeakFW = peReadLatencyPeakFW/(numPE*numPE);
				*readDynamicEnergyPeakFW = peReadDynamicEnergyPeakFW;
				*readLatencyPeakAG = peReadLatencyPeakAG/(numPE*numPE);
				*readDynamicEnergyPeakAG = peReadDynamicEnergyPeakAG;
				*writeLatencyPeakWU = peWriteLatencyPeakWU*(numPE*numPE);
				*writeDynamicEnergyPeakWU = peWriteDynamicEnergyPeakWU*(numPE*numPE);
				
				*bufferLatency = PEbufferLatency/(numPE*numPE);
				*bufferDynamicEnergy = PEbufferDynamicEnergy;
				*icLatency = PEicLatency/(numPE*numPE);
				*icDynamicEnergy = PEicDynamicEnergy;
				
				*coreLatencyADC = peLatencyADC/(numPE*numPE);
				*coreLatencyAccum = peLatencyAccum/(numPE*numPE);
				*coreLatencyOther = peLatencyOther/(numPE*numPE);
				
				*coreEnergyADC = peEnergyADC;
				*coreEnergyAccum = peEnergyAccum;
				*coreEnergyOther = peEnergyOther;
				// no accumulation access
			} else {
				// # duplication is smaller then # PE, means only a group of PE take the assigned weight  --> not "fully" duplication
				// also need to redefine a few data-grab start-point
				for (int i=0; i<ceil((double)weightMatrixRow/(double)peSize); i++) {
					for (int j=0; j<ceil((double)weightMatrixCol/(double)peSize); j++) {
						if ( (i*peSize < weightMatrixRow) && (j*peSize < weightMatrixCol) ) {
							int numRowMatrix = min(peSize, (double) weightMatrixRow-i*peSize);
							int numColMatrix = min(peSize, (double) weightMatrixCol-j*peSize);
					
							// assign weight and input to specific tile
							vector<vector<double> > pEMemoryOld;
							pEMemoryOld = CopyPEArray(oldMemory, i*peSize, j*peSize, numRowMatrix, numColMatrix);
							vector<vector<double> > pEMemory;
							pEMemory = CopyPEArray(newMemory, i*peSize, j*peSize, numRowMatrix, numColMatrix);
							vector<vector<double> > pEInput;
							pEInput = CopyPEInput(inputVector, i*peSize, numInVector, numRowMatrix);
							
							ProcessingUnitCalculatePerformance(subArrayInPE, tech, cell, pEMemory, pEMemoryOld, pEInput, 1, 1, 
												numSubArrayRow, numSubArrayCol, numRowMatrix, numColMatrix, numInVector, &PEreadLatency, &PEreadDynamicEnergy, &PEleakage,
												&PEreadLatencyAG, &PEreadDynamicEnergyAG, &PEwriteLatencyWU, &PEwriteDynamicEnergyWU,
												&PEbufferLatency, &PEbufferDynamicEnergy, &PEicLatency, &PEicDynamicEnergy,
												&peLatencyADC, &peLatencyAccum, &peLatencyOther, &peEnergyADC, &peEnergyAccum, &peEnergyOther, 
												&peReadLatencyPeakFW, &peReadDynamicEnergyPeakFW, &peReadLatencyPeakAG, &peReadDynamicEnergyPeakAG,
												&peWriteLatencyPeakWU, &peWriteDynamicEnergyPeakWU);
					
							*readLatency = MAX(PEreadLatency, (*readLatency));
							*readDynamicEnergy += PEreadDynamicEnergy;
							*readLatencyAG = MAX(PEreadLatencyAG, (*readLatencyAG));
							*readDynamicEnergyAG += PEreadDynamicEnergyAG;
							// accumulate write latency as array need to be write sequentially (worst case)
							// limitation by on-chip buffer, write latency will be divided by numArrayWriteParallel (real case)
							*writeLatencyWU += PEwriteLatencyWU;
							*writeDynamicEnergyWU += PEwriteDynamicEnergyWU;
							
							*readLatencyPeakFW = MAX(peReadLatencyPeakFW, (*readLatencyPeakFW));
							*readDynamicEnergyPeakFW += peReadDynamicEnergyPeakFW;
							*readLatencyPeakAG = MAX(peReadLatencyPeakAG, (*readLatencyPeakAG));
							*readDynamicEnergyPeakAG += peReadDynamicEnergyPeakAG;
							// accumulate write latency as array need to be write sequentially (worst case)
							// limitation by on-chip buffer, write latency will be divided by numArrayWriteParallel (real case)
							*writeLatencyPeakWU += peWriteLatencyPeakWU;
							*writeDynamicEnergyPeakWU += peWriteDynamicEnergyPeakWU;
							// cout << "*writeLatencyPeakWU: " << (*writeLatencyPeakWU) << endl;
							// cout << "*writeDynamicEnergyPeakWU: " << (*writeDynamicEnergyPeakWU) << endl;
							*bufferLatency = MAX(PEbufferLatency, (*bufferLatency));
							*bufferDynamicEnergy += PEbufferDynamicEnergy;
							*icLatency = MAX(PEicLatency,(*icLatency));
							*icDynamicEnergy += PEicDynamicEnergy;
							
							*coreLatencyADC = MAX(peLatencyADC, (*coreLatencyADC));
							*coreLatencyAccum = MAX(peLatencyAccum, (*coreLatencyAccum));
							*coreLatencyOther = MAX(peLatencyOther, (*coreLatencyOther));
							
							*coreEnergyADC += peEnergyADC;
							*coreEnergyAccum += peEnergyAccum;
							*coreEnergyOther += peEnergyOther;
						}
					}
				}
				*readLatency /= (speedUpRow*speedUpCol);   // further speedup in PE level
				*readLatencyAG /= (speedUpRow*speedUpCol);
				*readLatencyPeakFW /= (speedUpRow*speedUpCol);
				*readLatencyPeakAG /= (speedUpRow*speedUpCol);
				*coreLatencyADC /= (speedUpRow*speedUpCol);
				*coreLatencyAccum /= (speedUpRow*speedUpCol);
				*coreLatencyOther /= (speedUpRow*speedUpCol);
				*bufferLatency /= (speedUpRow*speedUpCol);
				*icLatency /= (speedUpRow*speedUpCol);
				
				// whether go through accumulation?
				if (ceil((double)weightMatrixRow/(double)peSize) > 1) {
					accumulation->CalculateLatency(param->numColMuxed, ceil((double)weightMatrixRow/(double)peSize), 0);
					accumulation->CalculatePower(param->numColMuxed, ceil((double)weightMatrixRow/(double)peSize));
					*readLatency += accumulation->readLatency; 
					*readLatencyAG += accumulation->readLatency*((param->trainingEstimation)==true? 1:0); 
					*readLatencyPeakFW += accumulation->readLatency; 
					*readLatencyPeakAG += accumulation->readLatency*((param->trainingEstimation)==true? 1:0); 
					*readDynamicEnergy += accumulation->readDynamicEnergy;
					*readDynamicEnergyAG += accumulation->readDynamicEnergy*((param->trainingEstimation)==true? 1:0);
					*readDynamicEnergyPeakFW += accumulation->readDynamicEnergy;
					*readDynamicEnergyPeakAG += accumulation->readDynamicEnergy*((param->trainingEstimation)==true? 1:0);
					*coreLatencyAccum += accumulation->readLatency*((param->trainingEstimation)==true? 2:1); 
					*coreEnergyAccum += accumulation->readDynamicEnergy*((param->trainingEstimation)==true? 2:1);
				}
			}
			
		} else {
			// no duplication --> tell PE to further partition the weight and grab data (redefine a few data-grab start-point)
			for (int i=0; i<numPE; i++) {
				for (int j=0; j<numPE; j++) {
					// each cycle assign to different PE
					if ( (i*peSize < weightMatrixRow) && (j*peSize < weightMatrixCol) ) {
						// assign weight and input to specific tile
						int numRowMatrix = min(peSize, (double) weightMatrixRow-i*peSize);
						int numColMatrix = min(peSize, (double) weightMatrixCol-j*peSize);
						
						vector<vector<double> > pEMemoryOld;
						pEMemoryOld = CopyPEArray(oldMemory, i*peSize, j*peSize, numRowMatrix, numColMatrix);
						vector<vector<double> > pEMemory;
						pEMemory = CopyPEArray(newMemory, i*peSize, j*peSize, numRowMatrix, numColMatrix);
						vector<vector<double> > pEInput;
						pEInput = CopyPEInput(inputVector, i*peSize, numInVector, numRowMatrix);
							
						ProcessingUnitCalculatePerformance(subArrayInPE, tech, cell, pEMemory, pEMemoryOld, pEInput, 1, 1, numSubArrayRow, numSubArrayCol, numRowMatrix,
												numColMatrix, numInVector, &PEreadLatency, &PEreadDynamicEnergy, &PEleakage,
												&PEreadLatencyAG, &PEreadDynamicEnergyAG, &PEwriteLatencyWU, &PEwriteDynamicEnergyWU,
												&PEbufferLatency, &PEbufferDynamicEnergy, &PEicLatency, &PEicDynamicEnergy,
												&peLatencyADC, &peLatencyAccum, &peLatencyOther, &peEnergyADC, &peEnergyAccum, &peEnergyOther,
												&peReadLatencyPeakFW, &peReadDynamicEnergyPeakFW, &peReadLatencyPeakAG, &peReadDynamicEnergyPeakAG,
												&peWriteLatencyPeakWU, &peWriteDynamicEnergyPeakWU);
					}
					*readLatency = MAX(PEreadLatency, (*readLatency));
					*readDynamicEnergy += PEreadDynamicEnergy;
					*readLatencyAG = MAX(PEreadLatencyAG, (*readLatencyAG));
					*readDynamicEnergyAG += PEreadDynamicEnergyAG;
					// accumulate write latency as array need to be write sequentially (worst case)
					// limitation by on-chip buffer, write latency will be divided by numArrayWriteParallel (real case)
					*writeLatencyWU += PEwriteLatencyWU;
					*writeDynamicEnergyWU += PEwriteDynamicEnergyWU;
					
					*readLatencyPeakFW = MAX(peReadLatencyPeakFW, (*readLatencyPeakFW));
					*readDynamicEnergyPeakFW += peReadDynamicEnergyPeakFW;
					*readLatencyPeakAG = MAX(peReadLatencyPeakAG, (*readLatencyPeakAG));
					*readDynamicEnergyPeakAG += peReadDynamicEnergyPeakAG;
					// accumulate write latency as array need to be write sequentially (worst case)
					// limitation by on-chip buffer, write latency will be divided by numArrayWriteParallel (real case)
					*writeLatencyPeakWU += peWriteLatencyPeakWU;
					*writeDynamicEnergyPeakWU += peWriteDynamicEnergyPeakWU;
					// cout << "*writeLatencyPeakWU: " << (*writeLatencyPeakWU) << endl;
					// cout << "*writeDynamicEnergyPeakWU: " << (*writeDynamicEnergyPeakWU) << endl;
					*bufferLatency = MAX(PEbufferLatency, (*bufferLatency));
					*bufferDynamicEnergy += PEbufferDynamicEnergy;
					*icLatency = MAX(PEicLatency,(*icLatency));
					*icDynamicEnergy += PEicDynamicEnergy;
					
					*coreLatencyADC = MAX(peLatencyADC, (*coreLatencyADC));
					*coreLatencyAccum = MAX(peLatencyAccum, (*coreLatencyAccum));
					*coreLatencyOther = MAX(peLatencyOther, (*coreLatencyOther));
					
					*coreEnergyADC += peEnergyADC;
					*coreEnergyAccum += peEnergyAccum;
					*coreEnergyOther += peEnergyOther;
				}
			}
			accumulation->CalculateLatency(param->numColMuxed, numPE, 0);
			accumulation->CalculatePower(param->numColMuxed, numPE);
			*readLatency += accumulation->readLatency;
			*readLatencyAG += accumulation->readLatency*((param->trainingEstimation)==true? 1:0);
			*readLatencyPeakFW += accumulation->readLatency;
			*readLatencyPeakAG += accumulation->readLatency*((param->trainingEstimation)==true? 1:0);
			*readDynamicEnergy += accumulation->readDynamicEnergy;
			*readDynamicEnergyAG += accumulation->readDynamicEnergy*((param->trainingEstimation)==true? 1:0);
			*readDynamicEnergyPeakFW += accumulation->readDynamicEnergy;
			*readDynamicEnergyPeakAG += accumulation->readDynamicEnergy*((param->trainingEstimation)==true? 1:0);
			*coreLatencyAccum += accumulation->readLatency*((param->trainingEstimation)==true? 2:1);
			*coreEnergyAccum += accumulation->readDynamicEnergy*((param->trainingEstimation)==true? 2:1);
		}
		double numBitToLoadOut, numBitToLoadIn;								 
		if (!param->chipActivation) {
			if (param->reLu) {
				reLu->CalculateLatency(param->numColMuxed);
				reLu->CalculatePower(param->numColMuxed);
				*readLatency += reLu->readLatency;
				*readDynamicEnergy += reLu->readDynamicEnergy;
				*readLatencyPeakFW += reLu->readLatency;
				*readDynamicEnergyPeakFW += reLu->readDynamicEnergy;
				
				*coreLatencyOther += reLu->readLatency;
				*coreEnergyOther += reLu->readDynamicEnergy;
				numBitToLoadIn = MAX(ceil(weightMatrixCol/param->numColPerSynapse)*(1+reLu->numBit)*numInVector/param->numBitInput, 0);
				outputBuffer->CalculateLatency(outputBuffer->interface_width, numBitToLoadIn/outputBuffer->interface_width, outputBuffer->interface_width, numBitToLoadIn/outputBuffer->interface_width);
				outputBuffer->CalculatePower(outputBuffer->interface_width, numBitToLoadIn/outputBuffer->interface_width, outputBuffer->interface_width, numBitToLoadIn/outputBuffer->interface_width);
			} else {
				sigmoid->CalculateLatency(param->numColMuxed);
				sigmoid->CalculatePower(param->numColMuxed);
				*readLatency += sigmoid->readLatency;
				*readDynamicEnergy += sigmoid->readDynamicEnergy;
				*readLatencyPeakFW += sigmoid->readLatency;
				*readDynamicEnergyPeakFW += sigmoid->readDynamicEnergy;
				
				*coreLatencyOther += sigmoid->readLatency;
				*coreEnergyOther += sigmoid->readDynamicEnergy;
				numBitToLoadIn = MAX(ceil(weightMatrixCol/param->numColPerSynapse)*(1+sigmoid->numYbit)*numInVector/param->numBitInput, 0);
				outputBuffer->CalculateLatency(outputBuffer->interface_width, numBitToLoadIn/outputBuffer->interface_width, outputBuffer->interface_width, numBitToLoadIn/outputBuffer->interface_width);
				outputBuffer->CalculatePower(outputBuffer->interface_width, numBitToLoadIn/outputBuffer->interface_width, outputBuffer->interface_width, numBitToLoadIn/outputBuffer->interface_width);
			}
		} else {
			numBitToLoadIn = MAX(ceil(weightMatrixCol/param->numColPerSynapse)*(1+accumulation->numAdderBit)*numInVector/param->numBitInput, 0);
			outputBuffer->CalculateLatency(outputBuffer->interface_width, numBitToLoadIn/outputBuffer->interface_width, outputBuffer->interface_width, numBitToLoadIn/outputBuffer->interface_width);
			outputBuffer->CalculatePower(outputBuffer->interface_width, numBitToLoadIn/outputBuffer->interface_width, outputBuffer->interface_width, numBitToLoadIn/outputBuffer->interface_width);
		}
		
		//considering buffer activation: no matter speedup or not, the total number of data transferred is fixed
		numBitToLoadOut = MAX(weightMatrixRow*numInVector, 0);
		inputBuffer->CalculateLatency(inputBuffer->interface_width, numBitToLoadOut/inputBuffer->interface_width, inputBuffer->interface_width, numBitToLoadOut/inputBuffer->interface_width);
		inputBuffer->CalculatePower(inputBuffer->interface_width, numBitToLoadOut/inputBuffer->interface_width, inputBuffer->interface_width, numBitToLoadOut/inputBuffer->interface_width);
		// since multi-core buffer has improve the parallelism
		inputBuffer->readLatency /= MIN(numInBufferCore, ceil(hTree->busWidth/inputBuffer->interface_width));
		inputBuffer->writeLatency /= MIN(numInBufferCore, ceil(hTree->busWidth/inputBuffer->interface_width));
		outputBuffer->readLatency /= MIN(numOutBufferCore, ceil(hTree->busWidth/outputBuffer->interface_width));
		outputBuffer->writeLatency /= MIN(numOutBufferCore, ceil(hTree->busWidth/outputBuffer->interface_width));
		
		// used to define travel distance
		double PEheight, PEwidth, PEbufferArea;
		int numSubArray = ceil((double) peSize/(double) param->numRowSubArray)*ceil((double) peSize/(double) param->numColSubArray);
		vector<double> PEarea;
		PEarea = ProcessingUnitCalculateArea(subArrayInPE, ceil((double)sqrt((double)numSubArray)), ceil((double)sqrt((double)numSubArray)), &PEheight, &PEwidth, &PEbufferArea);
		hTree->CalculateLatency(NULL, NULL, NULL, NULL, PEheight, PEwidth, (numBitToLoadOut+numBitToLoadIn)/hTree->busWidth);
		hTree->CalculatePower(NULL, NULL, NULL, NULL, PEheight, PEwidth, hTree->busWidth, (numBitToLoadOut+numBitToLoadIn)/hTree->busWidth);
		
		*readLatency += (inputBuffer->readLatency + inputBuffer->writeLatency);
		*readDynamicEnergy += inputBuffer->readDynamicEnergy + inputBuffer->writeDynamicEnergy;
		*readLatency += (outputBuffer->readLatency + outputBuffer->writeLatency);
		*readDynamicEnergy += outputBuffer->readDynamicEnergy + outputBuffer->writeDynamicEnergy;
		*readLatency += hTree->readLatency;
		*readDynamicEnergy += hTree->readDynamicEnergy;
		
		*bufferLatency += (inputBuffer->readLatency + outputBuffer->readLatency + inputBuffer->writeLatency + outputBuffer->writeLatency);
		*icLatency += hTree->readLatency;
		*bufferDynamicEnergy += inputBuffer->readDynamicEnergy + outputBuffer->readDynamicEnergy + inputBuffer->writeDynamicEnergy + outputBuffer->writeDynamicEnergy;
		*icDynamicEnergy += hTree->readDynamicEnergy;
		
		if (param->trainingEstimation) {
			*readLatencyAG += (inputBuffer->readLatency + inputBuffer->writeLatency);
			*readDynamicEnergyAG += (inputBuffer->readDynamicEnergy + inputBuffer->writeDynamicEnergy);
			*readLatencyAG += (outputBuffer->readLatency + outputBuffer->writeLatency);
			*readDynamicEnergyAG += (outputBuffer->readDynamicEnergy + outputBuffer->writeDynamicEnergy);
			*readLatencyAG += hTree->readLatency;
			*readDynamicEnergyAG += hTree->readDynamicEnergy;
			
			*bufferLatency += (inputBuffer->readLatency + outputBuffer->readLatency + inputBuffer->writeLatency + outputBuffer->writeLatency);
			*icLatency += hTree->readLatency;
			*bufferDynamicEnergy += (inputBuffer->readDynamicEnergy + outputBuffer->readDynamicEnergy + inputBuffer->writeDynamicEnergy + outputBuffer->writeDynamicEnergy);
			*icDynamicEnergy += hTree->readDynamicEnergy;
			
			// for delta weight transfer
			double numDeltaWeightBit = weightMatrixRow*weightMatrixCol;
			inputBuffer->CalculateLatency(inputBuffer->interface_width, numDeltaWeightBit/inputBuffer->interface_width, inputBuffer->interface_width, numDeltaWeightBit/inputBuffer->interface_width);
			inputBuffer->CalculatePower(inputBuffer->interface_width, numDeltaWeightBit/inputBuffer->interface_width, inputBuffer->interface_width, numDeltaWeightBit/inputBuffer->interface_width);
			hTree->CalculateLatency(NULL, NULL, NULL, NULL, PEheight, PEwidth, (numDeltaWeightBit)/hTree->busWidth);
			hTree->CalculatePower(NULL, NULL, NULL, NULL, PEheight, PEwidth, hTree->busWidth, (numDeltaWeightBit)/hTree->busWidth);
			*writeLatencyWU += (inputBuffer->readLatency + inputBuffer->writeLatency + hTree->readLatency);
			*writeDynamicEnergyWU += (inputBuffer->readDynamicEnergy + inputBuffer->writeDynamicEnergy + hTree->readDynamicEnergy);
		
			*bufferLatency += (inputBuffer->readLatency+ inputBuffer->writeLatency);
			*icLatency += hTree->readLatency;
			*bufferDynamicEnergy += (inputBuffer->readDynamicEnergy + inputBuffer->writeDynamicEnergy);
			*icDynamicEnergy += hTree->readDynamicEnergy;
		
		} 
		*leakage = PEleakage*numPE*numPE + accumulation->leakage + inputBuffer->leakage + outputBuffer->leakage;
	} else {  // novel Mapping
		for (int i=0; i<numPE; i++) {
			int location = i*MIN(peSize, (int) weightMatrixRow/numPE);
			vector<vector<double> > pEMemoryOld;
			pEMemoryOld = CopyPEArray(oldMemory, location, 0, (int)(weightMatrixRow/numPE), weightMatrixCol);
			
			vector<vector<double> > pEMemory;
			pEMemory = CopyPEArray(newMemory, location, 0, (int)(weightMatrixRow/numPE), weightMatrixCol);
			vector<vector<double> > pEInput;
			pEInput = CopyPEInput(inputVector, location, numInVector, weightMatrixRow/numPE);
			
			ProcessingUnitCalculatePerformance(subArrayInPE, tech, cell, pEMemory, pEMemoryOld, pEInput, 1, 1, numSubArrayRow, numSubArrayCol, weightMatrixRow/numPE,
									weightMatrixCol, numInVector, &PEreadLatency, &PEreadDynamicEnergy, &PEleakage,
									&PEreadLatencyAG, &PEreadDynamicEnergyAG, &PEwriteLatencyWU, &PEwriteDynamicEnergyWU,
									&PEbufferLatency, &PEbufferDynamicEnergy, &PEicLatency, &PEicDynamicEnergy, 
									&peLatencyADC, &peLatencyAccum, &peLatencyOther, &peEnergyADC, &peEnergyAccum, &peEnergyOther,
									&peReadLatencyPeakFW, &peReadDynamicEnergyPeakFW, &peReadLatencyPeakAG, &peReadDynamicEnergyPeakAG,
									&peWriteLatencyPeakWU, &peWriteDynamicEnergyPeakWU);

			*readLatency = MAX(PEreadLatency, (*readLatency));
			*readDynamicEnergy += PEreadDynamicEnergy;
			*readLatencyAG = MAX(PEreadLatencyAG, (*readLatencyAG));
			*readDynamicEnergyAG += PEreadDynamicEnergyAG;
			// accumulate write latency as array need to be write sequentially (worst case)
			// limitation by on-chip buffer, write latency will be divided by numArrayWriteParallel (real case)
			*writeLatencyWU += PEwriteLatencyWU; 
			*writeDynamicEnergyWU += PEwriteDynamicEnergyWU;
			
			*readLatencyPeakFW = MAX(peReadLatencyPeakFW, (*readLatencyPeakFW));
			*readDynamicEnergyPeakFW += peReadDynamicEnergyPeakFW;
			*readLatencyPeakAG = MAX(peReadLatencyPeakAG, (*readLatencyPeakAG));
			*readDynamicEnergyPeakAG += peReadDynamicEnergyPeakAG;
			// accumulate write latency as array need to be write sequentially (worst case)
			// limitation by on-chip buffer, write latency will be divided by numArrayWriteParallel (real case)
			*writeLatencyPeakWU += peWriteLatencyPeakWU;
			*writeDynamicEnergyPeakWU += peWriteDynamicEnergyPeakWU;

			*bufferLatency = MAX(PEbufferLatency, (*bufferLatency));
			*bufferDynamicEnergy += PEbufferDynamicEnergy;
			*icLatency = MAX(PEicLatency,(*icLatency));
			*icDynamicEnergy += PEicDynamicEnergy;
			
			*coreLatencyADC = MAX(peLatencyADC, (*coreLatencyADC));
			*coreLatencyAccum = MAX(peLatencyAccum, (*coreLatencyAccum));
			*coreLatencyOther = MAX(peLatencyOther, (*coreLatencyOther));
			
			*coreEnergyADC += peEnergyADC;
			*coreEnergyAccum += peEnergyAccum;
			*coreEnergyOther += peEnergyOther;
		}
		*readLatency /= (speedUpRow*speedUpCol);
		*readLatencyAG /= (speedUpRow*speedUpCol);
		*readLatencyPeakFW /= (speedUpRow*speedUpCol);
		*readLatencyPeakAG /= (speedUpRow*speedUpCol);
		*coreLatencyADC /= (speedUpRow*speedUpCol);
		*coreLatencyAccum /= (speedUpRow*speedUpCol);
		*coreLatencyOther /= (speedUpRow*speedUpCol);
		*bufferLatency /= (speedUpRow*speedUpCol);
		*icLatency /= (speedUpRow*speedUpCol);
		
		*writeDynamicEnergyWU *= (speedUpRow*speedUpCol);
		
		accumulation->CalculateLatency(param->numColMuxed, numPE, 0);
		accumulation->CalculatePower(param->numColMuxed, numPE);
		*readLatency += accumulation->readLatency;
		*readLatencyAG += accumulation->readLatency*((param->trainingEstimation)==true? 1:0);
		*readDynamicEnergy += accumulation->readDynamicEnergy;
		*readDynamicEnergyAG += accumulation->readDynamicEnergy*((param->trainingEstimation)==true? 1:0);
		*readLatencyPeakFW += accumulation->readLatency;
		*readDynamicEnergyPeakFW += accumulation->readDynamicEnergy;
		*readLatencyPeakAG += accumulation->readLatency*((param->trainingEstimation)==true? 1:0);
		*readDynamicEnergyPeakAG += accumulation->readDynamicEnergy*((param->trainingEstimation)==true? 1:0);
		
		*coreLatencyAccum += accumulation->readLatency*((param->trainingEstimation)==true? 2:1);
		*coreEnergyAccum += accumulation->readDynamicEnergy*((param->trainingEstimation)==true? 2:1);
		
		//considering buffer activation: no matter speedup or not, the total number of data transferred is fixed
		double numBitToLoadOut, numBitToLoadIn;
		numBitToLoadOut= MAX(weightMatrixRow*numInVector/sqrt(numPE), 0);
		inputBuffer->CalculateLatency(inputBuffer->interface_width, numBitToLoadOut/inputBuffer->interface_width, inputBuffer->interface_width, numBitToLoadOut/inputBuffer->interface_width);
		inputBuffer->CalculatePower(inputBuffer->interface_width, numBitToLoadOut/inputBuffer->interface_width, inputBuffer->interface_width, numBitToLoadOut/inputBuffer->interface_width);
		
		if (!param->chipActivation) {
			if (param->reLu) {
				reLu->CalculateLatency(param->numColMuxed);
				reLu->CalculatePower(param->numColMuxed);
				*readLatency += reLu->readLatency;
				*readDynamicEnergy += reLu->readDynamicEnergy;
				*readLatencyPeakFW += reLu->readLatency;
				*readDynamicEnergyPeakFW += reLu->readDynamicEnergy;
				*coreLatencyOther += reLu->readLatency;
				*coreEnergyOther += reLu->readDynamicEnergy;
				
				numBitToLoadIn = MAX(ceil(weightMatrixCol/param->numColPerSynapse)*(1+reLu->numBit)*numInVector/param->numBitInput/numPE, 0);
				outputBuffer->CalculateLatency(outputBuffer->interface_width, numBitToLoadIn/outputBuffer->interface_width, outputBuffer->interface_width, numBitToLoadIn/outputBuffer->interface_width);
				outputBuffer->CalculatePower(outputBuffer->interface_width, numBitToLoadIn/outputBuffer->interface_width, outputBuffer->interface_width, numBitToLoadIn/outputBuffer->interface_width);
			} else {
				sigmoid->CalculateLatency(param->numColMuxed);
				sigmoid->CalculatePower(param->numColMuxed);
				*readLatency += sigmoid->readLatency;
				*readDynamicEnergy += sigmoid->readDynamicEnergy;
				*readLatencyPeakFW += sigmoid->readLatency;
				*readDynamicEnergyPeakFW += sigmoid->readDynamicEnergy;
				*coreLatencyOther += sigmoid->readLatency;
				*coreEnergyOther += sigmoid->readDynamicEnergy;
				
				numBitToLoadIn = MAX(ceil(weightMatrixCol/param->numColPerSynapse)*(1+sigmoid->numYbit)*numInVector/param->numBitInput/numPE, 0);
				outputBuffer->CalculateLatency(outputBuffer->interface_width, numBitToLoadIn/outputBuffer->interface_width, outputBuffer->interface_width, numBitToLoadIn/outputBuffer->interface_width);
				outputBuffer->CalculatePower(outputBuffer->interface_width, numBitToLoadIn/outputBuffer->interface_width, outputBuffer->interface_width, numBitToLoadIn/outputBuffer->interface_width);
			}
		} else {
			numBitToLoadIn = MAX(ceil(weightMatrixCol/param->numColPerSynapse)*(1+accumulation->numAdderBit)*numInVector/param->numBitInput/numPE, 0);
			outputBuffer->CalculateLatency(outputBuffer->interface_width, numBitToLoadIn/outputBuffer->interface_width, outputBuffer->interface_width, numBitToLoadIn/outputBuffer->interface_width);
			outputBuffer->CalculatePower(outputBuffer->interface_width, numBitToLoadIn/outputBuffer->interface_width, outputBuffer->interface_width, numBitToLoadIn/outputBuffer->interface_width);
		}
		// since multi-core buffer has improve the parallelism
		inputBuffer->readLatency /= MIN(numInBufferCore, ceil(hTree->busWidth/inputBuffer->interface_width));
		inputBuffer->writeLatency /= MIN(numInBufferCore, ceil(hTree->busWidth/inputBuffer->interface_width));
		outputBuffer->readLatency /= MIN(numOutBufferCore, ceil(hTree->busWidth/inputBuffer->interface_width));
		outputBuffer->writeLatency /= MIN(numOutBufferCore, ceil(hTree->busWidth/inputBuffer->interface_width));
		
		// used to define travel distance
		double PEheight, PEwidth, PEbufferArea;
		int numSubArray = ceil((double) peSize/(double) param->numRowSubArray)*ceil((double) peSize/(double) param->numColSubArray);
		vector<double> PEarea;
		PEarea = ProcessingUnitCalculateArea(subArrayInPE, ceil((double)sqrt((double)numSubArray)), ceil((double)sqrt((double)numSubArray)), &PEheight, &PEwidth, &PEbufferArea);
		hTree->CalculateLatency(0, 0, 1, 1, PEheight, PEwidth, (numBitToLoadOut+numBitToLoadIn)/hTree->busWidth);
		hTree->CalculatePower(0, 0, 1, 1, PEheight, PEwidth, hTree->busWidth, (numBitToLoadOut+numBitToLoadIn)/hTree->busWidth);
		
		*readLatency += inputBuffer->readLatency + inputBuffer->writeLatency;
		*readDynamicEnergy += inputBuffer->readDynamicEnergy + inputBuffer->writeDynamicEnergy;
		*readLatency += (outputBuffer->readLatency + outputBuffer->writeLatency);
		*readDynamicEnergy += outputBuffer->readDynamicEnergy + outputBuffer->writeDynamicEnergy;
		*readLatency += hTree->readLatency;
		*readDynamicEnergy += hTree->readDynamicEnergy;
		
		*bufferLatency += (inputBuffer->readLatency + outputBuffer->readLatency + inputBuffer->writeLatency + outputBuffer->writeLatency);
		*icLatency += hTree->readLatency;
		*bufferDynamicEnergy += inputBuffer->readDynamicEnergy + outputBuffer->readDynamicEnergy + inputBuffer->writeDynamicEnergy + outputBuffer->writeDynamicEnergy;
		*icDynamicEnergy += hTree->readDynamicEnergy;

		if (param->trainingEstimation) {
			*readLatencyAG += (inputBuffer->readLatency + inputBuffer->writeLatency);
			*readDynamicEnergyAG += (inputBuffer->readDynamicEnergy + inputBuffer->writeDynamicEnergy);
			*readLatencyAG += (outputBuffer->readLatency + outputBuffer->writeLatency);
			*readDynamicEnergyAG += (outputBuffer->readDynamicEnergy + outputBuffer->writeDynamicEnergy);
			
			*readLatencyAG += hTree->readLatency;
			*readDynamicEnergyAG += hTree->readDynamicEnergy;
			
			*bufferLatency += (inputBuffer->readLatency + outputBuffer->readLatency + inputBuffer->writeLatency + outputBuffer->writeLatency);
			*icLatency += hTree->readLatency;
			*bufferDynamicEnergy += (inputBuffer->readDynamicEnergy + outputBuffer->readDynamicEnergy + inputBuffer->writeDynamicEnergy + outputBuffer->writeDynamicEnergy);
			*icDynamicEnergy += hTree->readDynamicEnergy;
			
			// for delta weight transfer
			double numDeltaWeightBit = weightMatrixRow*weightMatrixCol;
			inputBuffer->CalculateLatency(inputBuffer->interface_width, numDeltaWeightBit/inputBuffer->interface_width, inputBuffer->interface_width, numDeltaWeightBit/inputBuffer->interface_width);
			inputBuffer->CalculatePower(inputBuffer->interface_width, numDeltaWeightBit/inputBuffer->interface_width, inputBuffer->interface_width, numDeltaWeightBit/inputBuffer->interface_width);
			hTree->CalculateLatency(0, 0, 1, 1, PEheight, PEwidth, (numDeltaWeightBit)/hTree->busWidth);
			hTree->CalculatePower(0, 0, 1, 1, PEheight, PEwidth, hTree->busWidth, (numDeltaWeightBit)/hTree->busWidth);
			*writeLatencyWU += (inputBuffer->readLatency + inputBuffer->writeLatency + hTree->readLatency);
			*writeDynamicEnergyWU += (inputBuffer->readDynamicEnergy + inputBuffer->writeDynamicEnergy + hTree->readDynamicEnergy);
			
			*bufferLatency += (inputBuffer->readLatency+ inputBuffer->writeLatency);
			*icLatency += hTree->readLatency;
			*bufferDynamicEnergy += (inputBuffer->readDynamicEnergy + inputBuffer->writeDynamicEnergy);
			*icDynamicEnergy += hTree->readDynamicEnergy;
		}
		*leakage = PEleakage*numPE + accumulation->leakage + inputBuffer->leakage + outputBuffer->leakage;
	}
	
}


vector<vector<double> > CopyPEArray(const vector<vector<double> > &orginal, int positionRow, int positionCol, int numRow, int numCol) {
	
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


vector<vector<double> > CopyPEInput(const vector<vector<double> > &orginal, int positionRow, int numInputVector, int numRow) {
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

