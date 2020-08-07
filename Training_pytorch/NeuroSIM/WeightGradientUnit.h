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

#ifndef WEIGHTGRADIENTUNIT_H_
#define WEIGHTGRADIENTUNIT_H_

#include <vector>
#include "typedef.h"
#include "InputParameter.h"
#include "Technology.h"
#include "MemCell.h"
#include "formula.h"
#include "FunctionUnit.h"
#include "Adder.h"
#include "AdderTree.h"
#include "RowDecoder.h"
#include "Mux.h"
#include "DFF.h"
#include "Precharger.h"
#include "SenseAmp.h"
#include "SRAMWriteDriver.h"
#include "SwitchMatrix.h"
#include "ShiftAdd.h"
#include "MultilevelSenseAmp.h"
#include "MultilevelSAEncoder.h"
#include "SarADC.h"

using namespace std;

class WeightGradientUnit: public FunctionUnit {
public:
	WeightGradientUnit(const InputParameter& _inputParameter, const Technology& _tech, const MemCell& _cell);
	virtual ~WeightGradientUnit() {}
	const InputParameter& inputParameter;
	const Technology& tech;
	const MemCell& cell;

	/* Functions */
	void PrintProperty(const char* str);
	void Initialize(int _numMemRow, int _numMemCol);
	void CalculateArea();
	void CalculateLatency(int numRead, int numBitDataLoad);
	void CalculatePower(int numRead, int numBitDataLoad);

	/* Properties */	
	bool initialized;	   // Initialization flag
	int numRow;			   // Number of rows
	int numCol;			   // Number of columns
	
	int numMemRow;         // total memory size needed to support calculation
	int numMemCol;         // total memory size needed to support calculation
	
	int numArrayInRow;           
	int numArrayInCol;
	int outPrecision;
	
	double unitWireRes, lengthRow, lengthCol, capRow1, capRow2, capCol, resRow, resCol, resCellAccess, capCellAccess, colDelay, capSRAMCell;
	double heightArray;
	double widthArray;
	double areaArray, areaSubArray;
	double readDynamicEnergyArray, writeDynamicEnergyArray;
	double writeLatencyArray;
	double readLatencyPeak, writeLatencyPeak, readDynamicEnergyPeak, writeDynamicEnergyPeak;
	
	SpikingMode spikingMode;	
	
	
	/* Circuit modules */
	RowDecoder                   wlDecoder;
	SwitchMatrix                 wlSwitchMatrix;
	Mux                          mux;
	RowDecoder                   muxDecoder;
	Precharger                   precharger;
	SenseAmp                     senseAmp;
	SRAMWriteDriver              sramWriteDriver;
	DFF                          dff;
	Adder                        adder;
	MultilevelSenseAmp           multilevelSenseAmp;
	MultilevelSAEncoder          multilevelSAEncoder;
	ShiftAdd                     shiftAdd;
	AdderTree					 accumulation;
	DFF							 bufferInput;
	DFF							 bufferOutput;
	SarADC                       sarADC;
};

#endif /* WEIGHTGRADIENTUNIT_H_ */
