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
#include "constant.h"
#include "formula.h"
#include "Sigmoid.h"

using namespace std;

Sigmoid::Sigmoid(const InputParameter& _inputParameter, const Technology& _tech, const MemCell& _cell): inputParameter(_inputParameter), tech(_tech), cell(_cell), mux(_inputParameter, _tech, _cell), muxDecoder(_inputParameter, _tech, _cell), wlDecoder(_inputParameter, _tech, _cell), colDecoder(_inputParameter, _tech, _cell), senseAmp(_inputParameter, _tech, _cell), colDecoderDriver(_inputParameter, _tech, _cell), voltageSenseAmp(_inputParameter, _tech, _cell), FunctionUnit() {
	initialized = false;
}

void Sigmoid::Initialize(bool _SRAM, int _numYbit, int _numEntry, int _numFunction, double _clkFreq) {
	if (initialized)
		cout << "[Sigmoid] Warning: Already initialized!" << endl;
	
	SRAM = _SRAM;
	numYbit = _numYbit;               // # of y bit
	numEntry = _numEntry;             // # of (x,y) entry
	numFunction = _numFunction;       // # of sigmoid functions that can be processed in parallel
	clkFreq = _clkFreq;
	numCell = numYbit * numEntry;     // # of memory cell in each single sigmoid function
	
	// INV
	widthInvN = MIN_NMOS_SIZE * tech.featureSize;
	widthInvP = tech.pnSizeRatio * MIN_NMOS_SIZE * tech.featureSize;

	double hInv, wInv;
	// INV
	CalculateGateArea(INV, 1, widthInvN, widthInvP, tech.featureSize*MAX_TRANSISTOR_HEIGHT, tech, &hInv, &wInv);

	if (SRAM) { // calculate each memory cell size
		hUnit = hInv + cell.heightInFeatureSize * tech.featureSize;
		wUnit = MAX(wInv * 3, cell.widthInFeatureSize * tech.featureSize) * numYbit;
	} else {	// RRAM
		hUnit = cell.heightInFeatureSize * tech.featureSize;
		wUnit = cell.widthInFeatureSize * tech.featureSize * numYbit;
	}

	if (SRAM) { // initialize peripheral ckt for sigmoid function
		wlDecoder.Initialize(REGULAR_ROW, (int)ceil((double)log2((double)numEntry)), false, false);      // wlDecoder to give x values to sigmoid function
		senseAmp.Initialize(numYbit, false, cell.minSenseVoltage, wUnit/numYbit, clkFreq, 1);    // just assign one S/A
	} else {
		wlDecoder.Initialize(REGULAR_ROW, (int)ceil((double)log2((double)numEntry)), false, false);      // wlDecoder to give x values to sigmoid function
		voltageSenseAmp.Initialize(numYbit, clkFreq);
	}

	initialized = true;
}


void Sigmoid::CalculateUnitArea(AreaModify _option) {      // firstly calculate single sigmoid unit area
	if (!initialized) {
		cout << "[Sigmoid] Error: Require initialization first!" << endl;
	} else {

		areaUnit = 0;
		
		wlDecoder.CalculateArea(NULL, NULL, NONE);
		
		if (SRAM) { // initialize peripheral ckt for sigmoid function
			senseAmp.CalculateArea(NULL, NULL, NONE);
		} else {
			voltageSenseAmp.CalculateUnitArea();
			voltageSenseAmp.CalculateArea(NULL);
		}
		
		areaUnit += (hUnit * wUnit) * numEntry ;    
		areaUnit += wlDecoder.area + senseAmp.area + voltageSenseAmp.area;
		
		switch (_option) {
			case MAGIC:
				MagicLayout();
				break;
			case OVERRIDE:
				OverrideLayout();
				break;
			default:    // NONE
				break;
		}
		
	}
}


void Sigmoid::CalculateArea(double _newHeight, double _newWidth, AreaModify _option) {     // assign multiple sigmoid unit to operate in parallel
	if (!initialized) {
		cout << "[Sigmoid] Error: Require initialization first!" << endl;
	} else {
		area = 0;
		area = areaUnit * numFunction;
		if (_newWidth && _option==NONE) {
			width = _newWidth;
			height = area/width;
		} else {
			height = _newHeight;
            width = area/height;
		}
		// Modify layout
		newHeight = _newHeight;
		newWidth = _newWidth;
		switch (_option) {
			case MAGIC:
				MagicLayout();
				break;
			case OVERRIDE:
				OverrideLayout();
				break;  
			default:    // NONE
				break;
		}
	}
}

void Sigmoid::CalculateLatency(double numRead) {
	if (!initialized) {
		cout << "[Sigmoid] Error: Require initialization first!" << endl;
	} else {
		readLatency = 0;

		double resCellAccess = CalculateOnResistance(cell.widthAccessCMOS * tech.featureSize, NMOS, inputParameter.temperature, tech);
		double capCellAccess = CalculateDrainCap(cell.widthAccessCMOS * tech.featureSize, NMOS, cell.widthInFeatureSize * tech.featureSize, tech);
		capSRAMCell = capCellAccess + CalculateDrainCap(cell.widthSRAMCellNMOS * tech.featureSize, NMOS, cell.widthInFeatureSize * tech.featureSize, tech) + CalculateDrainCap(cell.widthSRAMCellPMOS * tech.featureSize, PMOS, cell.widthInFeatureSize * tech.featureSize, tech);
		
		if (SRAM) {
			wlDecoder.CalculateLatency(1e20, 0, capSRAMCell, 1, 1);
			senseAmp.CalculateLatency(1);
			readLatency = wlDecoder.readLatency + senseAmp.readLatency;
		} else {	// RRAM
			// Assuming no delay on RRAM wires
			wlDecoder.CalculateLatency(1e20, 0, capCellAccess, 1, 1);
			voltageSenseAmp.CalculateLatency(0, 1);
			readLatency = wlDecoder.readLatency + voltageSenseAmp.readLatency;
		}
		readLatency *= numRead;
	}
}

void Sigmoid::CalculatePower(double numRead) {
	if (!initialized) {
		cout << "[Sigmoid] Error: Require initialization first!" << endl;
	} else {
		leakage = 0;
		readDynamicEnergy = 0;

		if (SRAM) {
			wlDecoder.CalculatePower(1,1);
			senseAmp.CalculatePower(1);
			
			readDynamicEnergy += wlDecoder.readDynamicEnergy + senseAmp.readDynamicEnergy;

			// Array leakage (assume 2 INV)
			leakage += CalculateGateLeakage(INV, 1, cell.widthSRAMCellNMOS * tech.featureSize,
					cell.widthSRAMCellPMOS * tech.featureSize, inputParameter.temperature, tech) * tech.vdd * 2;
			leakage *= numCell;
			leakage += wlDecoder.leakage;
			leakage += senseAmp.leakage;
			
		} else {	// RRAM
			wlDecoder.CalculatePower(1,1);
			voltageSenseAmp.CalculatePower(1);
			readDynamicEnergy += voltageSenseAmp.readDynamicEnergy + wlDecoder.readDynamicEnergy;

			leakage += voltageSenseAmp.leakage;
			leakage += wlDecoder.leakage;
		}
		readDynamicEnergy *= numRead*numFunction;
	}
}

void Sigmoid::PrintProperty(const char* str) {
	FunctionUnit::PrintProperty(str);
}

void Sigmoid::SaveOutput(const char* str) {
	FunctionUnit::SaveOutput(str);
}

