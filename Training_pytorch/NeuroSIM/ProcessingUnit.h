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
*   Pai-Yu Chen     Email: pchen72 at asu dot edu 
*                     
*   Xiaochen Peng   Email: xpeng15 at asu dot edu
********************************************************************************/

#ifndef PROCESSINGUNIT_H_
#define PROCESSINGUNIT_H_
#include "InputParameter.h"
#include "Technology.h"
#include "MemCell.h"
#include "SubArray.h"
 
/*** Functions ***/
void ProcessingUnitInitialize(SubArray *& subArray, InputParameter& inputParameter, Technology& tech, MemCell& cell, int subArrayRowSize, int _numSubArrayCol);
vector<double> ProcessingUnitCalculateArea(SubArray *subArray, int numSubArrayRow, int numSubArrayCol, double *height, double *width, double *bufferArea);
double ProcessingUnitCalculatePerformance(SubArray *subArray, Technology& tech, MemCell& cell, 
										const vector<vector<double> > &newMemory, const vector<vector<double> > &oldMemory, const vector<vector<double> > &inputVector, 
										int arrayDupRow, int arrayDupCol, int numSubArrayRow, int numSubArrayCol, int weightMatrixRow, int weightMatrixCol, 
										int numInVector, double *readLatency, double *readDynamicEnergy, double *leakage, 
										double *readLatencyAG, double *readDynamicEnergyAG, double *writeLatencyWU, double *writeDynamicEnergyWU,
										double *bufferLatency, double *bufferDynamicEnergy, double *icLatency, double *icDynamicEnergy,
										double *coreLatencyADC, double *coreLatencyAccum, double *coreLatencyOther, double *coreEnergyADC, 
										double *coreEnergyAccum, double *coreEnergyOther, double *readLatencyPeakFW, double *readDynamicEnergyPeakFW,
										double *readLatencyPeakAG, double *readDynamicEnergyPeakAG, double *readLatencyPeakWU, double *readDynamicEnergyPeakWU);

vector<vector<double> > CopySubArray(const vector<vector<double> > &orginal, int positionRow, int positionCol, int numRow, int numCol);
vector<vector<double> > CopySubInput(const vector<vector<double> > &orginal, int positionRow, int numInputVector, int numRow);
vector<double> GetInputVector(const vector<vector<double> > &input, int numInput, double *activityRowRead);
vector<double> GetColumnResistance(const vector<double> &input, const vector<vector<double> > &weight, MemCell& cell, bool parallelRead, double resCellAccess);
vector<double> GetRowResistance(const vector<double> &input, const vector<vector<double> > &weight, MemCell& cell, bool parallelRead, double resCellAccess);
double GetWriteUpdateEstimation(SubArray *subArray, Technology& tech, MemCell& cell, const vector<vector<double> > &newMemory, const vector<vector<double> > &oldMemory, double *activityColWrite, double *activityRowWrite,
								int *numWritePulseAVG, int *totalNumWritePulse, double *writeDynamicEnergyArray);

#endif /* PROCESSINGUNIT_H_ */