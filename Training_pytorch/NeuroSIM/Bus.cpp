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
#include "typedef.h"
#include "formula.h"
#include "Bus.h"
#include "Param.h"

using namespace std;

extern Param *param;

Bus::Bus(const InputParameter& _inputParameter, const Technology& _tech, const MemCell& _cell): inputParameter(_inputParameter), tech(_tech), cell(_cell), FunctionUnit() {
	initialized = false;
}

void Bus::Initialize(BusMode _mode, int _numRow, int _numCol, double _delaytolerance, double _busWidth, double _unitHeight, double _unitWidth){
	if (initialized)
		cout << "[Bus] Warning: Already initialized!" << endl;
	
	mode = _mode;
	numRow = _numRow;
	numCol = _numCol;     // num of Row and Col in tile/pe level
	
	unitHeight = _unitHeight;
	unitWidth = _unitWidth;

	delaytolerance = _delaytolerance;
	busWidth = _busWidth;
	unitLengthWireResistance = param->unitLengthWireResistance;
	unitLengthWireCap = 0.2e-15/1e-6;;   // 0.2 fF/mm
	
	// define min INV resistance and capacitance to calculate repeater size
	widthMinInvN = MIN_NMOS_SIZE * tech.featureSize;
	widthMinInvP = tech.pnSizeRatio * MIN_NMOS_SIZE * tech.featureSize;
	CalculateGateArea(INV, 1, widthMinInvN, widthMinInvP, tech.featureSize * MAX_TRANSISTOR_HEIGHT, tech, &hMinInv, &wMinInv);
	CalculateGateCapacitance(INV, 1, widthMinInvN, widthMinInvP, hMinInv, tech, &capMinInvInput, &capMinInvOutput);
	double resOnRep = CalculateOnResistance(widthMinInvN, NMOS, inputParameter.temperature, tech) + CalculateOnResistance(widthMinInvP, PMOS, inputParameter.temperature, tech);
	
	// optimal repeater design to achieve highest speed
	repeaterSize = floor(sqrt(resOnRep*unitLengthWireCap/capMinInvInput/unitLengthWireResistance));
	minDist = sqrt(2*resOnRep*(capMinInvOutput+capMinInvInput)/(unitLengthWireResistance*unitLengthWireCap));
	CalculateGateArea(INV, 1, MIN_NMOS_SIZE * tech.featureSize * repeaterSize, tech.pnSizeRatio * MIN_NMOS_SIZE * tech.featureSize * repeaterSize, tech.featureSize * MAX_TRANSISTOR_HEIGHT, tech, &hRep, &wRep);
	CalculateGateCapacitance(INV, 1, MIN_NMOS_SIZE * tech.featureSize * repeaterSize, tech.pnSizeRatio * MIN_NMOS_SIZE * tech.featureSize * repeaterSize, hRep, tech, &capRepInput, &capRepOutput);
	resOnRep = CalculateOnResistance(MIN_NMOS_SIZE * tech.featureSize * repeaterSize, NMOS, inputParameter.temperature, tech) + CalculateOnResistance(tech.pnSizeRatio * MIN_NMOS_SIZE * tech.featureSize * repeaterSize, PMOS, inputParameter.temperature, tech);
	double minUnitLengthDelay = 0.7*(resOnRep*(capRepInput+capRepOutput+unitLengthWireCap*minDist)+0.5*unitLengthWireResistance*minDist*unitLengthWireCap*minDist+unitLengthWireResistance*minDist*capRepInput)/minDist;
	double maxUnitLengthEnergy = (capRepInput+capRepOutput+unitLengthWireCap*minDist)*tech.vdd*tech.vdd/minDist;
	
	if (delaytolerance) {   // tradeoff: increase delay to decrease energy
		double delay = 0;
		double energy = 100;
		while ((delay<minUnitLengthDelay*(1+delaytolerance)) && (repeaterSize >= 1)) {
			repeaterSize -= 1;
			minDist *= 0.9;
			CalculateGateArea(INV, 1, MIN_NMOS_SIZE * tech.featureSize * repeaterSize, tech.pnSizeRatio * MIN_NMOS_SIZE * tech.featureSize * repeaterSize, tech.featureSize * MAX_TRANSISTOR_HEIGHT, tech, &hRep, &wRep);
			CalculateGateCapacitance(INV, 1, MIN_NMOS_SIZE * tech.featureSize * repeaterSize, tech.pnSizeRatio * MIN_NMOS_SIZE * tech.featureSize * repeaterSize, hRep, tech, &capRepInput, &capRepOutput);
			resOnRep = CalculateOnResistance(MIN_NMOS_SIZE * tech.featureSize * repeaterSize, NMOS, inputParameter.temperature, tech) + CalculateOnResistance(tech.pnSizeRatio * MIN_NMOS_SIZE * tech.featureSize * repeaterSize, PMOS, inputParameter.temperature, tech);
			delay = 0.7*(resOnRep*(capRepInput+capRepOutput+unitLengthWireCap*minDist)+0.5*unitLengthWireResistance*minDist*unitLengthWireCap*minDist+unitLengthWireResistance*minDist*capRepInput)/minDist;
			energy = (capRepInput+capRepOutput+unitLengthWireCap*minDist)*tech.vdd*tech.vdd/minDist;
		}
	}
	
	widthInvN = repeaterSize * MIN_NMOS_SIZE * tech.featureSize;
	widthInvP = repeaterSize * tech.pnSizeRatio * MIN_NMOS_SIZE * tech.featureSize;
	
	if (mode == HORIZONTAL) {
		wireLength = unitWidth*(numCol-1);
	} else {
		wireLength = unitHeight*(numRow-1);
	}
	
	initialized = true;
}

void Bus::CalculateArea(double foldedratio, bool overLap) {
	if (!initialized) {
		cout << "[Bus] Error: Require initialization first!" << endl;
	} else {
		// INV
		CalculateGateArea(INV, 1, widthInvN, widthInvP, tech.featureSize * MAX_TRANSISTOR_HEIGHT, tech, &hInv, &wInv);

		numRepeater = ceil(wireLength/minDist);
	
		if (numRepeater > 0) {
			wireWidth += busWidth*wInv/foldedratio;  
		} else {
			wireWidth += busWidth*param->wireWidth/foldedratio;
		}
		
		if (!overLap) {
			area = numRow*wireLength*wireWidth;
		} else {
			area = 0;
		}

		// Capacitance
		// INV
		CalculateGateCapacitance(INV, 1, widthInvN, widthInvP, hInv, tech, &capInvInput, &capInvOutput);
	}
}

void Bus::CalculateLatency(double numRead){
	if (!initialized) {
		cout << "[Bus] Error: Require initialization first!" << endl;
	} else {
		readLatency = 0;
		
		double resOnRep = CalculateOnResistance(widthInvN, NMOS, inputParameter.temperature, tech) + CalculateOnResistance(widthInvP, PMOS, inputParameter.temperature, tech);
		unitLatencyRep = 0.7*(resOnRep*(capInvInput+capInvOutput+unitLengthWireCap*minDist)+0.5*unitLengthWireResistance*minDist*unitLengthWireCap*minDist+unitLengthWireResistance*minDist*capInvInput)/minDist;
		unitLatencyWire = 0.7*unitLengthWireResistance*minDist*unitLengthWireCap*minDist/minDist;
		
		if (numRepeater > 0) {
			readLatency += wireLength*unitLatencyRep;
		} else {
			readLatency += wireLength*unitLatencyWire;
		}
		
		readLatency *= numRead;	
	}
}

void Bus::CalculatePower(double numBitAccess, double numRead) {
	if (!initialized) {
		cout << "[Bus] Error: Require initialization first!" << endl;
	} else {
		leakage = 0;
		readDynamicEnergy = 0;
		
		unitLengthLeakage = CalculateGateLeakage(INV, 1, widthInvN, widthInvP, inputParameter.temperature, tech) * tech.vdd / minDist;
		leakage = unitLengthLeakage * wireLength * (numRow + numCol);

		unitLengthEnergyRep = (capInvInput+capInvOutput+unitLengthWireCap*minDist)*tech.vdd*tech.vdd/minDist * 0.25;
		unitLengthEnergyWire = (unitLengthWireCap*minDist)*tech.vdd*tech.vdd/minDist * 0.25;
		
		if (numRepeater > 0) {
			readDynamicEnergy += wireLength*unitLengthEnergyRep;
		} else {
			readDynamicEnergy += wireLength*unitLengthEnergyWire;
		}
		readDynamicEnergy *= numBitAccess*numRead;
	}
}

void Bus::PrintProperty(const char* str) {
	FunctionUnit::PrintProperty(str);
}

void Bus::SaveOutput(const char* str) {
	FunctionUnit::SaveOutput(str);
}


