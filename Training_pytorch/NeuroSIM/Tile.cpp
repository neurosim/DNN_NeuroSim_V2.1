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
Buffer *inputBufferCM;
Buffer *outputBufferCM;
HTree *hTreeCM;
AdderTree *accumulationCM;
Sigmoid *sigmoidCM;
BitShifter *reLuCM;
Buffer *inputBufferNM;
Buffer *outputBufferNM;
HTree *hTreeNM;
AdderTree *accumulationNM;
Sigmoid *sigmoidNM;
BitShifter *reLuNM;				   


void TileInitialize(InputParameter& inputParameter, Technology& tech, MemCell& cell, double _numPENM, double _peSizeNM, double _numPECM, double _peSizeCM){
	
	subArrayInPE = new SubArray(inputParameter, tech, cell);
	inputBufferNM = new Buffer(inputParameter, tech, cell);
	outputBufferNM = new Buffer(inputParameter, tech, cell);
	hTreeNM = new HTree(inputParameter, tech, cell);
	accumulationNM = new AdderTree(inputParameter, tech, cell);
	inputBufferCM = new Buffer(inputParameter, tech, cell);
	outputBufferCM = new Buffer(inputParameter, tech, cell);
	hTreeCM = new HTree(inputParameter, tech, cell);
	accumulationCM = new AdderTree(inputParameter, tech, cell);
	
	if (!param->chipActivation) {
		if (param->reLu) {
			reLuNM = new BitShifter(inputParameter, tech, cell);
			reLuCM = new BitShifter(inputParameter, tech, cell);
		} else {
			sigmoidNM = new Sigmoid(inputParameter, tech, cell);
			sigmoidCM = new Sigmoid(inputParameter, tech, cell);
		}
	}
	
	/*** Parameters ***/
	double numPENM, peSizeNM, numPECM, peSizeCM, numSubArrayNM, numSubArrayCM;
	int numRowPerSynapse, numColPerSynapse;
	
	numPECM = _numPECM;
	peSizeCM = _peSizeCM;
	numPENM = _numPENM;
	peSizeNM = _peSizeNM;
	numRowPerSynapse = param->numRowPerSynapse;
	numColPerSynapse = param->numColPerSynapse;
	
	/*** Initialize ProcessingUnit ***/
	numSubArrayNM = ceil((double)peSizeNM/(double)param->numRowSubArray)*ceil((double)peSizeNM/(double)param->numColSubArray);
	numSubArrayCM = ceil((double)peSizeCM/(double)param->numRowSubArray)*ceil((double)peSizeCM/(double)param->numColSubArray);
	ProcessingUnitInitialize(subArrayInPE, inputParameter, tech, cell, ceil(sqrt(numSubArrayNM)), ceil(sqrt(numSubArrayNM)), ceil(sqrt(numSubArrayCM)), ceil(sqrt(numSubArrayCM)));
    
	if (param->novelMapping) {
		if (param->parallelRead) {
			accumulationNM->Initialize(numPENM, ceil((double)log2((double)param->levelOutput))+param->numBitInput+param->numColPerSynapse+1+ceil((double)log2((double)peSizeNM/(double)param->numRowSubArray)), 
									ceil((double)numPENM*(double)param->numColSubArray/(double)param->numColMuxed));
			if (!param->chipActivation) {
				if (param->reLu) {
					reLuNM->Initialize(ceil((double)peSizeNM*(double)param->numColSubArray/(double)param->numColMuxed), param->numBitInput, param->clkFreq);
				} else {
					sigmoidNM->Initialize(false, param->numBitInput, ceil((double)log2((double)param->levelOutput))+param->numBitInput+param->numColPerSynapse+1+ceil((double)log2((double)peSizeNM/(double)param->numRowSubArray))+ceil((double)log2((double)numPENM)), 
									ceil((double)numPENM*(double)param->numColSubArray/(double)param->numColMuxed), param->clkFreq);
				}
				numOutBufferCore = ceil((param->numBitInput*numPENM*param->numColSubArray/param->numColMuxed)/(param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol));
				
				if ((param->numBitInput*numPENM*param->numColSubArray/param->numColMuxed) < (param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol)) {
					outputBufferNM->Initialize(param->numBitInput*numPENM*param->numColSubArray/param->numColMuxed, param->numBitInput*numPENM, 1, param->unitLengthWireResistance, param->clkFreq, param->peBufferType);
				} else {
					outputBufferNM->Initialize((param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol), param->tileBufferCoreSizeCol, 1, param->unitLengthWireResistance, param->clkFreq, param->peBufferType);
				}									
			} else {
				numOutBufferCore = ceil(((ceil((double)log2((double)param->levelOutput))+param->numBitInput+param->numColPerSynapse+1+ceil((double)log2((double)peSizeNM/(double)param->numRowSubArray)))*numPENM*param->numColSubArray/param->numColMuxed)/(param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol));
				if (((ceil((double)log2((double)param->levelOutput))+param->numBitInput+param->numColPerSynapse+1+ceil((double)log2((double)peSizeNM/(double)param->numRowSubArray)))*numPENM*param->numColSubArray/param->numColMuxed) < (param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol)) {
					outputBufferNM->Initialize((ceil((double)log2((double)param->levelOutput))+param->numBitInput+param->numColPerSynapse+1+ceil((double)log2((double)peSizeNM/(double)param->numRowSubArray)))*numPENM*param->numColSubArray/param->numColMuxed, 
									(ceil((double)log2((double)param->levelOutput))+param->numBitInput+param->numColPerSynapse+1+ceil((double)log2((double)peSizeNM/(double)param->numRowSubArray)))*numPENM, 
									1, param->unitLengthWireResistance, param->clkFreq, param->peBufferType);
				} else {
					outputBufferNM->Initialize((param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol), param->tileBufferCoreSizeCol, 1, param->unitLengthWireResistance, param->clkFreq, param->peBufferType);
				}
			}
		} else {
			accumulationNM->Initialize(numPENM, ceil((double)log2((double)param->numRowSubArray)+(double)param->cellBit-1)+param->numBitInput+param->numColPerSynapse+1+ceil((double)log2((double)peSizeNM/(double)param->numRowSubArray)), 
									ceil(numPENM*(double)param->numColSubArray/(double)param->numColMuxed));
			if (!param->chipActivation) {
				if (param->reLu) {
					reLuNM->Initialize(ceil((double)peSizeNM*(double)param->numColSubArray/(double)param->numColMuxed), param->numBitInput, param->clkFreq);
				} else {
					sigmoidNM->Initialize(false, param->numBitInput, ceil((double)log2((double)param->numRowSubArray)+(double)param->cellBit-1)+param->numBitInput+param->numColPerSynapse+1+ceil((double)log2((double)peSizeNM/(double)param->numRowSubArray))+ceil((double)log2((double)numPENM)), 
									ceil(numPENM*(double)param->numColSubArray/(double)param->numColMuxed), param->clkFreq);
				}
				numOutBufferCore = ceil((param->numBitInput*numPENM*param->numColSubArray/param->numColMuxed)/(param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol));
				if ((param->numBitInput*numPENM*param->numColSubArray/param->numColMuxed) < (param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol)) {
					outputBufferNM->Initialize(param->numBitInput*numPENM*param->numColSubArray/param->numColMuxed, param->numBitInput*numPENM, 1, param->unitLengthWireResistance, param->clkFreq, param->peBufferType);
				} else {
					outputBufferNM->Initialize((param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol), param->tileBufferCoreSizeCol, 1, param->unitLengthWireResistance, param->clkFreq, param->peBufferType);
				}
			} else {
				numOutBufferCore = ceil(((ceil((double)log2((double)param->numRowSubArray)+(double)param->cellBit-1)+param->numBitInput+param->numColPerSynapse+1+ceil((double)log2((double)peSizeNM/(double)param->numRowSubArray)))*numPENM*param->numColSubArray/param->numColMuxed)/(param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol));
				if (((ceil((double)log2((double)param->numRowSubArray)+(double)param->cellBit-1)+param->numBitInput+param->numColPerSynapse+1+ceil((double)log2((double)peSizeNM/(double)param->numRowSubArray)))*numPENM*param->numColSubArray/param->numColMuxed) < (param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol)) {
					outputBufferNM->Initialize((ceil((double)log2((double)param->numRowSubArray)+(double)param->cellBit-1)+param->numBitInput+param->numColPerSynapse+1+ceil((double)log2((double)peSizeNM/(double)param->numRowSubArray)))*numPENM*param->numColSubArray/param->numColMuxed, 
									(ceil((double)log2((double)param->numRowSubArray)+(double)param->cellBit-1)+param->numBitInput+param->numColPerSynapse+1+ceil((double)log2((double)peSizeNM/(double)param->numRowSubArray)))*numPENM, 
									1, param->unitLengthWireResistance, param->clkFreq, param->peBufferType);
				} else {
					outputBufferNM->Initialize((param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol), param->tileBufferCoreSizeCol, 1, param->unitLengthWireResistance, param->clkFreq, param->peBufferType);
				}
			}
		}
		numInBufferCore = ceil((numPENM*param->numBitInput*param->numRowSubArray)/(param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol));
		
		if ((numPENM*param->numBitInput*param->numRowSubArray) < (param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol)) {
			inputBufferNM->Initialize(numPENM*param->numBitInput*param->numRowSubArray, numPENM*param->numRowSubArray, 1, param->unitLengthWireResistance, param->clkFreq, param->peBufferType);
		} else {
			inputBufferNM->Initialize((param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol), param->tileBufferCoreSizeCol, 1, param->unitLengthWireResistance, param->clkFreq, param->peBufferType);
		}
		hTreeNM->Initialize(ceil(sqrt((double)numPENM)), ceil(sqrt((double)numPENM)), param->localBusDelayTolerance, ceil(sqrt((double)numPENM))*param->numRowSubArray);
	} 
	if (param->parallelRead) {
		accumulationCM->Initialize(numPECM, ceil((double)log2((double)param->levelOutput))+param->numBitInput+param->numColPerSynapse+1+ceil((double)log2((double)peSizeCM/(double)param->numRowSubArray)), 
								ceil((double)numPECM*(double)param->numColSubArray/(double)param->numColMuxed));
		if (!param->chipActivation) {
			if (param->reLu) {
				reLuCM->Initialize(ceil((double)peSizeCM*(double)param->numColSubArray/(double)param->numColMuxed), param->numBitInput, param->clkFreq);
			} else {
				sigmoidCM->Initialize(false, param->numBitInput, ceil((double)log2((double)param->levelOutput))+param->numBitInput+param->numColPerSynapse+1+ceil((double)log2((double)peSizeCM/(double)param->numRowSubArray))+ceil((double)log2((double)numPECM)), 
								ceil((double)numPECM*(double)param->numColSubArray/(double)param->numColMuxed), param->clkFreq);
			}
			numOutBufferCore = ceil((param->numBitInput*numPECM*param->numColSubArray/param->numColMuxed)/(param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol));
			
			if ((param->numBitInput*numPECM*param->numColSubArray/param->numColMuxed) < (param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol)) {
				outputBufferCM->Initialize(param->numBitInput*numPECM*param->numColSubArray/param->numColMuxed, param->numBitInput*numPECM, 1, param->unitLengthWireResistance, param->clkFreq, param->peBufferType);
			} else {
				outputBufferCM->Initialize((param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol), param->tileBufferCoreSizeCol, 1, param->unitLengthWireResistance, param->clkFreq, param->peBufferType);
			}									
		} else {
			numOutBufferCore = ceil(((ceil((double)log2((double)param->levelOutput))+param->numBitInput+param->numColPerSynapse+1+ceil((double)log2((double)peSizeCM/(double)param->numRowSubArray)))*numPECM*param->numColSubArray/param->numColMuxed)/(param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol));
			if (((ceil((double)log2((double)param->levelOutput))+param->numBitInput+param->numColPerSynapse+1+ceil((double)log2((double)peSizeCM/(double)param->numRowSubArray)))*numPECM*param->numColSubArray/param->numColMuxed) < (param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol)) {
				outputBufferCM->Initialize((ceil((double)log2((double)param->levelOutput))+param->numBitInput+param->numColPerSynapse+1+ceil((double)log2((double)peSizeCM/(double)param->numRowSubArray)))*numPECM*param->numColSubArray/param->numColMuxed, 
								(ceil((double)log2((double)param->levelOutput))+param->numBitInput+param->numColPerSynapse+1+ceil((double)log2((double)peSizeCM/(double)param->numRowSubArray)))*numPECM, 
								1, param->unitLengthWireResistance, param->clkFreq, param->peBufferType);
			} else {
				outputBufferCM->Initialize((param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol), param->tileBufferCoreSizeCol, 1, param->unitLengthWireResistance, param->clkFreq, param->peBufferType);
			}
		}
	} else {
		accumulationCM->Initialize(numPECM, ceil((double)log2((double)param->numRowSubArray)+(double)param->cellBit-1)+param->numBitInput+param->numColPerSynapse+1+ceil((double)log2((double)peSizeCM/(double)param->numRowSubArray)), 
								ceil(numPECM*(double)param->numColSubArray/(double)param->numColMuxed));
		if (!param->chipActivation) {
			if (param->reLu) {
				reLuCM->Initialize(ceil((double)peSizeCM*(double)param->numColSubArray/(double)param->numColMuxed), param->numBitInput, param->clkFreq);
			} else {
				sigmoidCM->Initialize(false, param->numBitInput, ceil((double)log2((double)param->numRowSubArray)+(double)param->cellBit-1)+param->numBitInput+param->numColPerSynapse+1+ceil((double)log2((double)peSizeCM/(double)param->numRowSubArray))+ceil((double)log2((double)numPECM)), 
								ceil(numPECM*(double)param->numColSubArray/(double)param->numColMuxed), param->clkFreq);
			}
			numOutBufferCore = ceil((param->numBitInput*numPECM*param->numColSubArray/param->numColMuxed)/(param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol));
			if ((param->numBitInput*numPECM*param->numColSubArray/param->numColMuxed) < (param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol)) {
				outputBufferCM->Initialize(param->numBitInput*numPECM*param->numColSubArray/param->numColMuxed, param->numBitInput*numPECM, 1, param->unitLengthWireResistance, param->clkFreq, param->peBufferType);
			} else {
				outputBufferCM->Initialize((param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol), param->tileBufferCoreSizeCol, 1, param->unitLengthWireResistance, param->clkFreq, param->peBufferType);
			}
		} else {
			numOutBufferCore = ceil(((ceil((double)log2((double)param->numRowSubArray)+(double)param->cellBit-1)+param->numBitInput+param->numColPerSynapse+1+ceil((double)log2((double)peSizeCM/(double)param->numRowSubArray)))*numPECM*param->numColSubArray/param->numColMuxed)/(param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol));
			if (((ceil((double)log2((double)param->numRowSubArray)+(double)param->cellBit-1)+param->numBitInput+param->numColPerSynapse+1+ceil((double)log2((double)peSizeCM/(double)param->numRowSubArray)))*numPECM*param->numColSubArray/param->numColMuxed) < (param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol)) {
				outputBufferCM->Initialize((ceil((double)log2((double)param->numRowSubArray)+(double)param->cellBit-1)+param->numBitInput+param->numColPerSynapse+1+ceil((double)log2((double)peSizeCM/(double)param->numRowSubArray)))*numPECM*param->numColSubArray/param->numColMuxed, 
								(ceil((double)log2((double)param->numRowSubArray)+(double)param->cellBit-1)+param->numBitInput+param->numColPerSynapse+1+ceil((double)log2((double)peSizeCM/(double)param->numRowSubArray)))*numPECM, 
								1, param->unitLengthWireResistance, param->clkFreq, param->peBufferType);
			} else {
				outputBufferCM->Initialize((param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol), param->tileBufferCoreSizeCol, 1, param->unitLengthWireResistance, param->clkFreq, param->peBufferType);
			}
		}
	}
	numInBufferCore = ceil((numPECM*param->numBitInput*param->numRowSubArray)/(param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol));
	
	if ((numPECM*param->numBitInput*param->numRowSubArray) < (param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol)) {
		inputBufferCM->Initialize(numPECM*param->numBitInput*param->numRowSubArray, numPECM*param->numRowSubArray, 1, param->unitLengthWireResistance, param->clkFreq, param->peBufferType);
	} else {
		inputBufferCM->Initialize((param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol), param->tileBufferCoreSizeCol, 1, param->unitLengthWireResistance, param->clkFreq, param->peBufferType);
	}
	hTreeCM->Initialize(numPECM, numPECM, param->localBusDelayTolerance, numPECM*param->numRowSubArray);
}

vector<double> TileCalculateArea(double numPE, double peSize, bool NMTile, double *height, double *width) {
	double area = 0;
	double PEheight, PEwidth, PEbufferArea;
	*height = 0;
	*width = 0;
	vector<double> areaResults;
	vector<double> peAreaResults;
	double areareLu = 0;
	double areasigmoid = 0;
	
	if (NMTile) {
		int numSubArray = ceil((double) peSize/(double) param->numRowSubArray)*ceil((double) peSize/(double) param->numColSubArray);
		peAreaResults = ProcessingUnitCalculateArea(subArrayInPE, ceil((double)sqrt((double)numSubArray)), ceil((double)sqrt((double)numSubArray)), true, &PEheight, &PEwidth, &PEbufferArea);
		double PEarea = peAreaResults[0];
		double PEareaADC = peAreaResults[1];
		double PEareaAccum = peAreaResults[2];
		double PEareaOther = peAreaResults[3];
		double PEareaArray = peAreaResults[4];
		accumulationNM->CalculateArea(NULL, ceil(sqrt((double)numPE))*PEwidth, NONE);
		if (!param->chipActivation) {
			if (param->reLu) {
				reLuNM->CalculateArea(NULL, ceil(sqrt((double)numPE))*PEwidth, NONE);
				area += reLuNM->area;
				areareLu += reLuNM->area;
			} else {
				sigmoidNM->CalculateUnitArea(NONE);
				sigmoidNM->CalculateArea(NULL, ceil(sqrt((double)numPE))*PEwidth, NONE);
				area += sigmoidNM->area;
				areasigmoid += sigmoidNM->area;
			}
		}
		inputBufferNM->CalculateArea(ceil(sqrt((double)numPE))*PEheight, NULL, NONE);
		outputBufferNM->CalculateArea(NULL, ceil(sqrt((double)numPE))*PEwidth, NONE);
		inputBufferNM->area *= numInBufferCore;
		outputBufferNM->area *= numOutBufferCore;												  
		hTreeNM->CalculateArea(PEheight, PEwidth, 16);
		
		area += PEarea*numPE + accumulationNM->area + inputBufferNM->area + outputBufferNM->area + hTreeNM->area;
		
		*height = sqrt(area);
		*width = area/(*height);
		
		areaResults.push_back(area);
		areaResults.push_back(hTreeNM->area);
		areaResults.push_back(PEareaADC*numPE);
		areaResults.push_back(PEareaAccum*numPE + accumulationNM->area);
		areaResults.push_back(PEareaOther*numPE + inputBufferNM->area + outputBufferNM->area + areareLu + areasigmoid);
		areaResults.push_back(PEareaArray*numPE);
	} else {
		int numSubArray = ceil((double) peSize/(double) param->numRowSubArray)*ceil((double) peSize/(double) param->numColSubArray);
		peAreaResults = ProcessingUnitCalculateArea(subArrayInPE, ceil((double)sqrt((double)numSubArray)), ceil((double)sqrt((double)numSubArray)), false, &PEheight, &PEwidth, &PEbufferArea);
		double PEarea = peAreaResults[0];
		double PEareaADC = peAreaResults[1];
		double PEareaAccum = peAreaResults[2];
		double PEareaOther = peAreaResults[3];
		double PEareaArray = peAreaResults[4];
		accumulationCM->CalculateArea(NULL, ceil(sqrt((double)numPE))*PEwidth, NONE);
		if (!param->chipActivation) {
			if (param->reLu) {
				reLuCM->CalculateArea(NULL, ceil(sqrt((double)numPE))*PEwidth, NONE);
				area += reLuCM->area;
				areareLu += reLuCM->area;
			} else {
				sigmoidCM->CalculateUnitArea(NONE);
				sigmoidCM->CalculateArea(NULL, ceil(sqrt((double)numPE))*PEwidth, NONE);
				area += sigmoidCM->area;
				areasigmoid += sigmoidCM->area;
			}
		}
		inputBufferCM->CalculateArea(ceil(sqrt((double)numPE))*PEheight, NULL, NONE);
		outputBufferCM->CalculateArea(NULL, ceil(sqrt((double)numPE))*PEwidth, NONE);
		inputBufferCM->area *= numInBufferCore;
		outputBufferCM->area *= numOutBufferCore;												  
		hTreeCM->CalculateArea(PEheight, PEwidth, 16);
		
		area += PEarea*numPE + accumulationCM->area + inputBufferCM->area + outputBufferCM->area + hTreeCM->area;
		
		*height = sqrt(area);
		*width = area/(*height);
		
		areaResults.push_back(area);
		areaResults.push_back(hTreeCM->area);
		areaResults.push_back(PEareaADC*numPE);
		areaResults.push_back(PEareaAccum*numPE + accumulationCM->area);
		areaResults.push_back(PEareaOther*numPE + inputBufferCM->area + outputBufferCM->area + areareLu + areasigmoid);
		areaResults.push_back(PEareaArray*numPE);
	}
	return areaResults;
}


void TileCalculatePerformance(const vector<vector<double> > &newMemory, const vector<vector<double> > &oldMemory, const vector<vector<double> > &inputVector, int novelMap, int layerNumber, double numPE, 
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
				
				ProcessingUnitCalculatePerformance(subArrayInPE, tech, cell, layerNumber, false, pEMemory, pEMemoryOld, pEInput, ceil((double)speedUpRow/(double)numPE), ceil((double)speedUpCol/(double)numPE), 
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
							
							ProcessingUnitCalculatePerformance(subArrayInPE, tech, cell, layerNumber, false, pEMemory, pEMemoryOld, pEInput, 1, 1, 
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
					accumulationCM->CalculateLatency((int)(numInVector/param->numBitInput)*ceil(param->numColMuxed/param->numColPerSynapse), ceil((double)weightMatrixRow/(double)peSize), 0);
					accumulationCM->CalculatePower((int)(numInVector/param->numBitInput)*ceil(param->numColMuxed/param->numColPerSynapse), ceil((double)weightMatrixRow/(double)peSize));
					*readLatency += accumulationCM->readLatency; 
					*readLatencyAG += accumulationCM->readLatency*((param->trainingEstimation)&&(layerNumber!=0)==true? 1:0); 
					*readLatencyPeakFW += accumulationCM->readLatency; 
					*readLatencyPeakAG += accumulationCM->readLatency*((param->trainingEstimation)&&(layerNumber!=0)==true? 1:0); 
					*readDynamicEnergy += accumulationCM->readDynamicEnergy;
					*readDynamicEnergyAG += accumulationCM->readDynamicEnergy*((param->trainingEstimation)&&(layerNumber!=0)==true? 1:0);
					*readDynamicEnergyPeakFW += accumulationCM->readDynamicEnergy;
					*readDynamicEnergyPeakAG += accumulationCM->readDynamicEnergy*((param->trainingEstimation)&&(layerNumber!=0)==true? 1:0);
					*coreLatencyAccum += accumulationCM->readLatency*((param->trainingEstimation)&&(layerNumber!=0)==true? 2:1); 
					*coreEnergyAccum += accumulationCM->readDynamicEnergy*((param->trainingEstimation)&&(layerNumber!=0)==true? 2:1);
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
							
						ProcessingUnitCalculatePerformance(subArrayInPE, tech, cell, layerNumber, false, pEMemory, pEMemoryOld, pEInput, 1, 1, numSubArrayRow, numSubArrayCol, numRowMatrix,
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
			accumulationCM->CalculateLatency((int)(numInVector/param->numBitInput)*ceil(param->numColMuxed/param->numColPerSynapse), numPE, 0);
			accumulationCM->CalculatePower((int)(numInVector/param->numBitInput)*ceil(param->numColMuxed/param->numColPerSynapse), numPE);
			*readLatency += accumulationCM->readLatency;
			*readLatencyAG += accumulationCM->readLatency*((param->trainingEstimation)&&(layerNumber!=0)==true? 1:0);
			*readLatencyPeakFW += accumulationCM->readLatency;
			*readLatencyPeakAG += accumulationCM->readLatency*((param->trainingEstimation)&&(layerNumber!=0)==true? 1:0);
			*readDynamicEnergy += accumulationCM->readDynamicEnergy;
			*readDynamicEnergyAG += accumulationCM->readDynamicEnergy*((param->trainingEstimation)&&(layerNumber!=0)==true? 1:0);
			*readDynamicEnergyPeakFW += accumulationCM->readDynamicEnergy;
			*readDynamicEnergyPeakAG += accumulationCM->readDynamicEnergy*((param->trainingEstimation)&&(layerNumber!=0)==true? 1:0);
			*coreLatencyAccum += accumulationCM->readLatency*((param->trainingEstimation)&&(layerNumber!=0)==true? 2:1);
			*coreEnergyAccum += accumulationCM->readDynamicEnergy*((param->trainingEstimation)&&(layerNumber!=0)==true? 2:1);
		}
		double numBitToLoadOut, numBitToLoadIn;								 
		if (!param->chipActivation) {
			if (param->reLu) {
				reLuCM->CalculateLatency((int)(numInVector/param->numBitInput)*ceil(param->numColMuxed/param->numColPerSynapse)/reLuCM->numUnit);
				reLuCM->CalculatePower((int)(numInVector/param->numBitInput)*ceil(param->numColMuxed/param->numColPerSynapse)/reLuCM->numUnit);
				*readLatency += reLuCM->readLatency;
				*readDynamicEnergy += reLuCM->readDynamicEnergy;
				*readLatencyPeakFW += reLuCM->readLatency;
				*readDynamicEnergyPeakFW += reLuCM->readDynamicEnergy;
				
				*coreLatencyOther += reLuCM->readLatency;
				*coreEnergyOther += reLuCM->readDynamicEnergy;
				numBitToLoadIn = MAX(ceil(weightMatrixCol/param->numColPerSynapse)*(1+reLuCM->numBit)*numInVector/param->numBitInput, 0);
				outputBufferCM->CalculateLatency(outputBufferCM->interface_width, numBitToLoadIn/outputBufferCM->interface_width, outputBufferCM->interface_width, numBitToLoadIn/outputBufferCM->interface_width);
				outputBufferCM->CalculatePower(outputBufferCM->interface_width, numBitToLoadIn/outputBufferCM->interface_width, outputBufferCM->interface_width, numBitToLoadIn/outputBufferCM->interface_width);
			} else {
				sigmoidCM->CalculateLatency((int)(numInVector/param->numBitInput)*ceil(param->numColMuxed/param->numColPerSynapse)/sigmoidCM->numEntry);
				sigmoidCM->CalculatePower((int)(numInVector/param->numBitInput)*ceil(param->numColMuxed/param->numColPerSynapse)/sigmoidCM->numEntry);
				*readLatency += sigmoidCM->readLatency;
				*readDynamicEnergy += sigmoidCM->readDynamicEnergy;
				*readLatencyPeakFW += sigmoidCM->readLatency;
				*readDynamicEnergyPeakFW += sigmoidCM->readDynamicEnergy;
				
				*coreLatencyOther += sigmoidCM->readLatency;
				*coreEnergyOther += sigmoidCM->readDynamicEnergy;
				numBitToLoadIn = MAX(ceil(weightMatrixCol/param->numColPerSynapse)*(1+sigmoidCM->numYbit)*numInVector/param->numBitInput, 0);
				outputBufferCM->CalculateLatency(outputBufferCM->interface_width, numBitToLoadIn/outputBufferCM->interface_width, outputBufferCM->interface_width, numBitToLoadIn/outputBufferCM->interface_width);
				outputBufferCM->CalculatePower(outputBufferCM->interface_width, numBitToLoadIn/outputBufferCM->interface_width, outputBufferCM->interface_width, numBitToLoadIn/outputBufferCM->interface_width);
			}
		} else {
			numBitToLoadIn = MAX(ceil(weightMatrixCol/param->numColPerSynapse)*(1+accumulationCM->numAdderBit)*numInVector/param->numBitInput, 0);
			outputBufferCM->CalculateLatency(outputBufferCM->interface_width, numBitToLoadIn/outputBufferCM->interface_width, outputBufferCM->interface_width, numBitToLoadIn/outputBufferCM->interface_width);
			outputBufferCM->CalculatePower(outputBufferCM->interface_width, numBitToLoadIn/outputBufferCM->interface_width, outputBufferCM->interface_width, numBitToLoadIn/outputBufferCM->interface_width);
		}
		
		//considering buffer activation: no matter speedup or not, the total number of data transferred is fixed
		numBitToLoadOut = MAX(weightMatrixRow*numInVector, 0);
		inputBufferCM->CalculateLatency(inputBufferCM->interface_width, numBitToLoadOut/inputBufferCM->interface_width, inputBufferCM->interface_width, numBitToLoadOut/inputBufferCM->interface_width);
		inputBufferCM->CalculatePower(inputBufferCM->interface_width, numBitToLoadOut/inputBufferCM->interface_width, inputBufferCM->interface_width, numBitToLoadOut/inputBufferCM->interface_width);
		// since multi-core buffer has improve the parallelism
		inputBufferCM->readLatency /= MIN(numInBufferCore, ceil(hTreeCM->busWidth/inputBufferCM->interface_width));
		inputBufferCM->writeLatency /= MIN(numInBufferCore, ceil(hTreeCM->busWidth/inputBufferCM->interface_width));
		outputBufferCM->readLatency /= MIN(numOutBufferCore, ceil(hTreeCM->busWidth/outputBufferCM->interface_width));
		outputBufferCM->writeLatency /= MIN(numOutBufferCore, ceil(hTreeCM->busWidth/outputBufferCM->interface_width));
		
		// used to define travel distance
		double PEheight, PEwidth, PEbufferArea;
		int numSubArray = ceil((double) peSize/(double) param->numRowSubArray)*ceil((double) peSize/(double) param->numColSubArray);
		vector<double> PEarea;
		PEarea = ProcessingUnitCalculateArea(subArrayInPE, ceil((double)sqrt((double)numSubArray)), ceil((double)sqrt((double)numSubArray)), false, &PEheight, &PEwidth, &PEbufferArea);
		hTreeCM->CalculateLatency(NULL, NULL, NULL, NULL, PEheight, PEwidth, (numBitToLoadOut+numBitToLoadIn)/hTreeCM->busWidth);
		hTreeCM->CalculatePower(NULL, NULL, NULL, NULL, PEheight, PEwidth, hTreeCM->busWidth, (numBitToLoadOut+numBitToLoadIn)/hTreeCM->busWidth);
		
		*readLatency += (inputBufferCM->readLatency + inputBufferCM->writeLatency);
		*readDynamicEnergy += inputBufferCM->readDynamicEnergy + inputBufferCM->writeDynamicEnergy;
		*readLatency += (outputBufferCM->readLatency + outputBufferCM->writeLatency);
		*readDynamicEnergy += outputBufferCM->readDynamicEnergy + outputBufferCM->writeDynamicEnergy;
		*readLatency += hTreeCM->readLatency;
		*readDynamicEnergy += hTreeCM->readDynamicEnergy;
		
		*bufferLatency += (inputBufferCM->readLatency + outputBufferCM->readLatency + inputBufferCM->writeLatency + outputBufferCM->writeLatency);
		*icLatency += hTreeCM->readLatency;
		*bufferDynamicEnergy += inputBufferCM->readDynamicEnergy + outputBufferCM->readDynamicEnergy + inputBufferCM->writeDynamicEnergy + outputBufferCM->writeDynamicEnergy;
		*icDynamicEnergy += hTreeCM->readDynamicEnergy;
		
		if (param->trainingEstimation) {
			*readLatencyAG += (inputBufferCM->readLatency + inputBufferCM->writeLatency)*((layerNumber!=0)==true? 1:0);
			*readDynamicEnergyAG += (inputBufferCM->readDynamicEnergy + inputBufferCM->writeDynamicEnergy)*((layerNumber!=0)==true? 1:0);
			*readLatencyAG += (outputBufferCM->readLatency + outputBufferCM->writeLatency)*((layerNumber!=0)==true? 1:0);
			*readDynamicEnergyAG += (outputBufferCM->readDynamicEnergy + outputBufferCM->writeDynamicEnergy)*((layerNumber!=0)==true? 1:0);
			*readLatencyAG += hTreeCM->readLatency*((layerNumber!=0)==true? 1:0);
			*readDynamicEnergyAG += hTreeCM->readDynamicEnergy*((layerNumber!=0)==true? 1:0);
			
			*bufferLatency += (inputBufferCM->readLatency + outputBufferCM->readLatency + inputBufferCM->writeLatency + outputBufferCM->writeLatency)*((layerNumber!=0)==true? 1:0);
			*icLatency += hTreeCM->readLatency*((layerNumber!=0)==true? 1:0);
			*bufferDynamicEnergy += (inputBufferCM->readDynamicEnergy + outputBufferCM->readDynamicEnergy + inputBufferCM->writeDynamicEnergy + outputBufferCM->writeDynamicEnergy)*((layerNumber!=0)==true? 1:0);
			*icDynamicEnergy += hTreeCM->readDynamicEnergy*((layerNumber!=0)==true? 1:0);
			
			// for delta weight transfer
			double numDeltaWeightBit = weightMatrixRow*weightMatrixCol;
			inputBufferCM->CalculateLatency(inputBufferCM->interface_width, numDeltaWeightBit/inputBufferCM->interface_width, inputBufferCM->interface_width, numDeltaWeightBit/inputBufferCM->interface_width);
			inputBufferCM->CalculatePower(inputBufferCM->interface_width, numDeltaWeightBit/inputBufferCM->interface_width, inputBufferCM->interface_width, numDeltaWeightBit/inputBufferCM->interface_width);
			hTreeCM->CalculateLatency(NULL, NULL, NULL, NULL, PEheight, PEwidth, (numDeltaWeightBit)/hTreeCM->busWidth);
			hTreeCM->CalculatePower(NULL, NULL, NULL, NULL, PEheight, PEwidth, hTreeCM->busWidth, (numDeltaWeightBit)/hTreeCM->busWidth);
			*writeLatencyWU += (inputBufferCM->readLatency + inputBufferCM->writeLatency + hTreeCM->readLatency);
			*writeDynamicEnergyWU += (inputBufferCM->readDynamicEnergy + inputBufferCM->writeDynamicEnergy + hTreeCM->readDynamicEnergy);
		
			*bufferLatency += (inputBufferCM->readLatency+ inputBufferCM->writeLatency);
			*icLatency += hTreeCM->readLatency;
			*bufferDynamicEnergy += (inputBufferCM->readDynamicEnergy + inputBufferCM->writeDynamicEnergy);
			*icDynamicEnergy += hTreeCM->readDynamicEnergy;
		
		} 
		*leakage = PEleakage*numPE*numPE + accumulationCM->leakage + inputBufferCM->leakage + outputBufferCM->leakage;
	} else {  // novel Mapping
		for (int i=0; i<numPE; i++) {
			int location = i*MIN(peSize, (int) weightMatrixRow/numPE);
			vector<vector<double> > pEMemoryOld;
			pEMemoryOld = CopyPEArray(oldMemory, location, 0, (int)(weightMatrixRow/numPE), weightMatrixCol);
			
			vector<vector<double> > pEMemory;
			pEMemory = CopyPEArray(newMemory, location, 0, (int)(weightMatrixRow/numPE), weightMatrixCol);
			vector<vector<double> > pEInput;
			pEInput = CopyPEInput(inputVector, location, numInVector, weightMatrixRow/numPE);
			
			ProcessingUnitCalculatePerformance(subArrayInPE, tech, cell, layerNumber, true, pEMemory, pEMemoryOld, pEInput, 1, 1, numSubArrayRow, numSubArrayCol, weightMatrixRow/numPE,
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
		
		accumulationNM->CalculateLatency((int)(numInVector/param->numBitInput)*ceil(param->numColMuxed/param->numColPerSynapse), numPE, 0);
		accumulationNM->CalculatePower((int)(numInVector/param->numBitInput)*ceil(param->numColMuxed/param->numColPerSynapse), numPE);
		*readLatency += accumulationNM->readLatency;
		*readLatencyAG += accumulationNM->readLatency*((param->trainingEstimation)&&(layerNumber!=0)==true? 1:0);
		*readDynamicEnergy += accumulationNM->readDynamicEnergy;
		*readDynamicEnergyAG += accumulationNM->readDynamicEnergy*((param->trainingEstimation)&&(layerNumber!=0)==true? 1:0);
		*readLatencyPeakFW += accumulationNM->readLatency;
		*readDynamicEnergyPeakFW += accumulationNM->readDynamicEnergy;
		*readLatencyPeakAG += accumulationNM->readLatency*((param->trainingEstimation)&&(layerNumber!=0)==true? 1:0);
		*readDynamicEnergyPeakAG += accumulationNM->readDynamicEnergy*((param->trainingEstimation)&&(layerNumber!=0)==true? 1:0);
		
		*coreLatencyAccum += accumulationNM->readLatency*((param->trainingEstimation)&&(layerNumber!=0)==true? 2:1);
		*coreEnergyAccum += accumulationNM->readDynamicEnergy*((param->trainingEstimation)&&(layerNumber!=0)==true? 2:1);
		
		//considering buffer activation: no matter speedup or not, the total number of data transferred is fixed
		double numBitToLoadOut, numBitToLoadIn;
		numBitToLoadOut= MAX(weightMatrixRow*numInVector/sqrt(numPE), 0);
		inputBufferNM->CalculateLatency(inputBufferNM->interface_width, numBitToLoadOut/inputBufferNM->interface_width, inputBufferNM->interface_width, numBitToLoadOut/inputBufferNM->interface_width);
		inputBufferNM->CalculatePower(inputBufferNM->interface_width, numBitToLoadOut/inputBufferNM->interface_width, inputBufferNM->interface_width, numBitToLoadOut/inputBufferNM->interface_width);
		
		if (!param->chipActivation) {
			if (param->reLu) {
				reLuNM->CalculateLatency((int)(numInVector/param->numBitInput)*ceil(param->numColMuxed/param->numColPerSynapse)/reLuNM->numUnit);
				reLuNM->CalculatePower((int)(numInVector/param->numBitInput)*ceil(param->numColMuxed/param->numColPerSynapse)/reLuNM->numUnit);
				*readLatency += reLuNM->readLatency;
				*readDynamicEnergy += reLuNM->readDynamicEnergy;
				*readLatencyPeakFW += reLuNM->readLatency;
				*readDynamicEnergyPeakFW += reLuNM->readDynamicEnergy;
				*coreLatencyOther += reLuNM->readLatency;
				*coreEnergyOther += reLuNM->readDynamicEnergy;
				
				numBitToLoadIn = MAX(ceil(weightMatrixCol/param->numColPerSynapse)*(1+reLuNM->numBit)*numInVector/param->numBitInput/numPE, 0);
				outputBufferNM->CalculateLatency(outputBufferNM->interface_width, numBitToLoadIn/outputBufferNM->interface_width, outputBufferNM->interface_width, numBitToLoadIn/outputBufferNM->interface_width);
				outputBufferNM->CalculatePower(outputBufferNM->interface_width, numBitToLoadIn/outputBufferNM->interface_width, outputBufferNM->interface_width, numBitToLoadIn/outputBufferNM->interface_width);
			} else {
				sigmoidNM->CalculateLatency((int)(numInVector/param->numBitInput)*ceil(param->numColMuxed/param->numColPerSynapse)/sigmoidNM->numEntry);
				sigmoidNM->CalculatePower((int)(numInVector/param->numBitInput)*ceil(param->numColMuxed/param->numColPerSynapse)/sigmoidNM->numEntry);
				*readLatency += sigmoidNM->readLatency;
				*readDynamicEnergy += sigmoidNM->readDynamicEnergy;
				*readLatencyPeakFW += sigmoidNM->readLatency;
				*readDynamicEnergyPeakFW += sigmoidNM->readDynamicEnergy;
				*coreLatencyOther += sigmoidNM->readLatency;
				*coreEnergyOther += sigmoidNM->readDynamicEnergy;
				
				numBitToLoadIn = MAX(ceil(weightMatrixCol/param->numColPerSynapse)*(1+sigmoidNM->numYbit)*numInVector/param->numBitInput/numPE, 0);
				outputBufferNM->CalculateLatency(outputBufferNM->interface_width, numBitToLoadIn/outputBufferNM->interface_width, outputBufferNM->interface_width, numBitToLoadIn/outputBufferNM->interface_width);
				outputBufferNM->CalculatePower(outputBufferNM->interface_width, numBitToLoadIn/outputBufferNM->interface_width, outputBufferNM->interface_width, numBitToLoadIn/outputBufferNM->interface_width);
			}
		} else {
			numBitToLoadIn = MAX(ceil(weightMatrixCol/param->numColPerSynapse)*(1+accumulationNM->numAdderBit)*numInVector/param->numBitInput/numPE, 0);
			outputBufferNM->CalculateLatency(outputBufferNM->interface_width, numBitToLoadIn/outputBufferNM->interface_width, outputBufferNM->interface_width, numBitToLoadIn/outputBufferNM->interface_width);
			outputBufferNM->CalculatePower(outputBufferNM->interface_width, numBitToLoadIn/outputBufferNM->interface_width, outputBufferNM->interface_width, numBitToLoadIn/outputBufferNM->interface_width);
		}
		// since multi-core buffer has improve the parallelism
		inputBufferNM->readLatency /= MIN(numInBufferCore, ceil(hTreeNM->busWidth/inputBufferNM->interface_width));
		inputBufferNM->writeLatency /= MIN(numInBufferCore, ceil(hTreeNM->busWidth/inputBufferNM->interface_width));
		outputBufferNM->readLatency /= MIN(numOutBufferCore, ceil(hTreeNM->busWidth/inputBufferNM->interface_width));
		outputBufferNM->writeLatency /= MIN(numOutBufferCore, ceil(hTreeNM->busWidth/inputBufferNM->interface_width));
		
		// used to define travel distance
		double PEheight, PEwidth, PEbufferArea;
		int numSubArray = ceil((double) peSize/(double) param->numRowSubArray)*ceil((double) peSize/(double) param->numColSubArray);
		vector<double> PEarea;
		PEarea = ProcessingUnitCalculateArea(subArrayInPE, ceil((double)sqrt((double)numSubArray)), ceil((double)sqrt((double)numSubArray)), true, &PEheight, &PEwidth, &PEbufferArea);
		hTreeNM->CalculateLatency(0, 0, 1, 1, PEheight, PEwidth, (numBitToLoadOut+numBitToLoadIn)/hTreeNM->busWidth);
		hTreeNM->CalculatePower(0, 0, 1, 1, PEheight, PEwidth, hTreeNM->busWidth, (numBitToLoadOut+numBitToLoadIn)/hTreeNM->busWidth);
		
		*readLatency += inputBufferNM->readLatency + inputBufferNM->writeLatency;
		*readDynamicEnergy += inputBufferNM->readDynamicEnergy + inputBufferNM->writeDynamicEnergy;
		*readLatency += (outputBufferNM->readLatency + outputBufferNM->writeLatency);
		*readDynamicEnergy += outputBufferNM->readDynamicEnergy + outputBufferNM->writeDynamicEnergy;
		*readLatency += hTreeNM->readLatency;
		*readDynamicEnergy += hTreeNM->readDynamicEnergy;
		
		*bufferLatency += (inputBufferNM->readLatency + outputBufferNM->readLatency + inputBufferNM->writeLatency + outputBufferNM->writeLatency);
		*icLatency += hTreeNM->readLatency;
		*bufferDynamicEnergy += inputBufferNM->readDynamicEnergy + outputBufferNM->readDynamicEnergy + inputBufferNM->writeDynamicEnergy + outputBufferNM->writeDynamicEnergy;
		*icDynamicEnergy += hTreeNM->readDynamicEnergy;

		if (param->trainingEstimation) {
			*readLatencyAG += (inputBufferNM->readLatency + inputBufferNM->writeLatency)*((layerNumber!=0)==true? 1:0);
			*readDynamicEnergyAG += (inputBufferNM->readDynamicEnergy + inputBufferNM->writeDynamicEnergy)*((layerNumber!=0)==true? 1:0);
			*readLatencyAG += (outputBufferNM->readLatency + outputBufferNM->writeLatency)*((layerNumber!=0)==true? 1:0);
			*readDynamicEnergyAG += (outputBufferNM->readDynamicEnergy + outputBufferNM->writeDynamicEnergy)*((layerNumber!=0)==true? 1:0);
			*readLatencyAG += hTreeNM->readLatency*((layerNumber!=0)==true? 1:0);
			*readDynamicEnergyAG += hTreeNM->readDynamicEnergy*((layerNumber!=0)==true? 1:0);
			
			*bufferLatency += (inputBufferNM->readLatency + outputBufferNM->readLatency + inputBufferNM->writeLatency + outputBufferNM->writeLatency)*((layerNumber!=0)==true? 1:0);
			*icLatency += hTreeNM->readLatency*((layerNumber!=0)==true? 1:0);
			*bufferDynamicEnergy += (inputBufferNM->readDynamicEnergy + outputBufferNM->readDynamicEnergy + inputBufferNM->writeDynamicEnergy + outputBufferNM->writeDynamicEnergy)*((layerNumber!=0)==true? 1:0);
			*icDynamicEnergy += hTreeNM->readDynamicEnergy*((layerNumber!=0)==true? 1:0);
			
			// for delta weight transfer
			double numDeltaWeightBit = weightMatrixRow*weightMatrixCol;
			inputBufferNM->CalculateLatency(inputBufferNM->interface_width, numDeltaWeightBit/inputBufferNM->interface_width, inputBufferNM->interface_width, numDeltaWeightBit/inputBufferNM->interface_width);
			inputBufferNM->CalculatePower(inputBufferNM->interface_width, numDeltaWeightBit/inputBufferNM->interface_width, inputBufferNM->interface_width, numDeltaWeightBit/inputBufferNM->interface_width);
			hTreeNM->CalculateLatency(0, 0, 1, 1, PEheight, PEwidth, (numDeltaWeightBit)/hTreeNM->busWidth);
			hTreeNM->CalculatePower(0, 0, 1, 1, PEheight, PEwidth, hTreeNM->busWidth, (numDeltaWeightBit)/hTreeNM->busWidth);
			*writeLatencyWU += (inputBufferNM->readLatency + inputBufferNM->writeLatency + hTreeNM->readLatency);
			*writeDynamicEnergyWU += (inputBufferNM->readDynamicEnergy + inputBufferNM->writeDynamicEnergy + hTreeNM->readDynamicEnergy);
			
			*bufferLatency += (inputBufferNM->readLatency+ inputBufferNM->writeLatency);
			*icLatency += hTreeNM->readLatency;
			*bufferDynamicEnergy += (inputBufferNM->readDynamicEnergy + inputBufferNM->writeDynamicEnergy);
			*icDynamicEnergy += hTreeNM->readDynamicEnergy;
		}
		*leakage = PEleakage*numPE + accumulationNM->leakage + inputBufferNM->leakage + outputBufferNM->leakage;
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

