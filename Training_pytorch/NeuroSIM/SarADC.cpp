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
#include <vector>
#include "constant.h"
#include "formula.h"
#include "Param.h"
#include "SarADC.h"

using namespace std;

extern Param *param;

SarADC::SarADC(const InputParameter& _inputParameter, const Technology& _tech, const MemCell& _cell): inputParameter(_inputParameter), tech(_tech), cell(_cell), FunctionUnit() {
	initialized = false;
}

void SarADC::Initialize(int _numCol, int _levelOutput, double _clkFreq, int _numReadCellPerOperationNeuro) {
	if (initialized) {
		cout << "[SarADC] Warning: Already initialized!" << endl;
    } else {
		numCol = _numCol;
		levelOutput = _levelOutput;                // # of bits for A/D output ... 
		clkFreq = _clkFreq;
		numReadCellPerOperationNeuro = _numReadCellPerOperationNeuro;
		
		widthNmos = MIN_NMOS_SIZE * tech.featureSize;
		widthPmos = tech.pnSizeRatio * MIN_NMOS_SIZE * tech.featureSize;
		
		initialized = true;
	}
}


void SarADC::CalculateUnitArea() {
	if (!initialized) {
		cout << "[SarADC] Error: Require initialization first!" << endl;
	} else {
		double hNmos, wNmos, hPmos, wPmos;
		
		CalculateGateArea(INV, 1, widthNmos, 0, tech.featureSize*MAX_TRANSISTOR_HEIGHT, tech, &hNmos, &wNmos);
		CalculateGateArea(INV, 1, 0, widthPmos, tech.featureSize*MAX_TRANSISTOR_HEIGHT, tech, &hPmos, &wPmos);
		
		areaUnit = (hNmos * wNmos) * (269+(log2(levelOutput)-1)*109) + (hPmos * wPmos) * (209+(log2(levelOutput)-1)*73);
	}
}


void SarADC::CalculateArea(double heightArray, double widthArray, AreaModify _option) {
	if (!initialized) {
		cout << "[SarADC] Error: Require initialization first!" << endl;
	} else {
		
		area = 0;
		height = 0;
		width = 0;
		
		if (widthArray && _option==NONE) {
			area = areaUnit * numCol;
			width = widthArray;
			height = area/widthArray;
		} else if (heightArray && _option==NONE) {
			area = areaUnit * numCol;
			height = heightArray;
			width = area/height;
		} else {
			cout << "[MultilevelSenseAmp] Error: No width or height assigned for the multiSenseAmp circuit" << endl;
			exit(-1);
		}
		// Assume the Current Mirrors are on the same row and the total width of them is smaller than the adder or DFF
		
		// Modify layout
		newHeight = heightArray;
		newWidth = widthArray;
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

void SarADC::CalculateLatency(double numRead) {
	if (!initialized) {
		cout << "[SarADC] Error: Require initialization first!" << endl;
	} else {
		readLatency = 0;
		readLatency += (log2(levelOutput)+1)*1e-9;
		readLatency *= numRead;
	}
}

void SarADC::CalculatePower(const vector<double> &columnResistance, double numRead) {
	if (!initialized) {
		cout << "[SarADC] Error: Require initialization first!" << endl;
	} else {
		leakage = 0;
		readDynamicEnergy = 0;
		for (double i=0; i<columnResistance.size(); i++) {
			double E_Col = 0;
			E_Col = GetColumnPower(columnResistance[i]);
			if (columnResistance[i] == columnResistance[i]) {
				readDynamicEnergy += E_Col;
			} else {
				readDynamicEnergy += 0;
			}
		}
		readDynamicEnergy *= numRead;
		
	}
} 

void SarADC::PrintProperty(const char* str) {
	FunctionUnit::PrintProperty(str);
}


double SarADC::GetColumnPower(double columnRes) {
	double Column_Power = 0;
	double Column_Energy = 0;
	// in Cadence simulation, we fix Vread to 0.5V, with user-defined Vread (different from 0.5V)
	// we should modify the equivalent columnRes
	columnRes *= 0.5/param->readVoltage;
	if ((double) 1/columnRes == 0) { 
		Column_Power = 1e-6;
	} else if (columnRes == 0) {
		Column_Power = 0;
	} else {
		if (param->deviceroadmap == 1) {  // HP
			if (param->technode == 130) {
				Column_Power = (6.4806*log2(levelOutput)+49.047)*1e-6;
				Column_Power += 0.207452*exp(-2.367*log10(columnRes));
			} else if (param->technode == 90) {
				Column_Power = (4.3474*log2(levelOutput)+31.782)*1e-6;
				Column_Power += 0.164900*exp(-2.345*log10(columnRes));
			} else if (param->technode == 65) {
				Column_Power = (2.9503*log2(levelOutput)+22.047)*1e-6;
				Column_Power += 0.128483*exp(-2.321*log10(columnRes));
			} else if (param->technode == 45) {
				Column_Power = (2.1843*log2(levelOutput)+11.931)*1e-6;
				Column_Power += 0.097754*exp(-2.296*log10(columnRes));
			} else if (param->technode == 32){  
				Column_Power = (1.0157*log2(levelOutput)+7.6286)*1e-6;
				Column_Power += 0.083709*exp(-2.313*log10(columnRes));
			} else if (param->technode == 22){   
				Column_Power = (0.7213*log2(levelOutput)+3.3041)*1e-6;
				Column_Power += 0.084273*exp(-2.311*log10(columnRes));
			} else if (param->technode == 14){   
				Column_Power = (0.4710*log2(levelOutput)+1.9529)*1e-6;
				Column_Power += 0.060584*exp(-2.311*log10(columnRes));
			} else if (param->technode == 10){   
				Column_Power = (0.3076*log2(levelOutput)+1.1543)*1e-6;
				Column_Power += 0.049418*exp(-2.311*log10(columnRes));
			} else {   // 7nm
				Column_Power = (0.2008*log2(levelOutput)+0.6823)*1e-6;
				Column_Power += 0.040310*exp(-2.311*log10(columnRes));
			}
		} else {                         // LP
			if (param->technode == 130) {
				Column_Power = (8.4483*log2(levelOutput)+65.243)*1e-6;
				Column_Power += 0.169380*exp(-2.303*log10(columnRes));
			} else if (param->technode == 90) {
				Column_Power = (5.9869*log2(levelOutput)+37.462)*1e-6;
				Column_Power += 0.144323*exp(-2.303*log10(columnRes));
			} else if (param->technode == 65) {
				Column_Power = (3.7506*log2(levelOutput)+25.844)*1e-6;
				Column_Power += 0.121272*exp(-2.303*log10(columnRes));
			} else if (param->technode == 45) {
				Column_Power = (2.1691*log2(levelOutput)+16.693)*1e-6;
				Column_Power += 0.100225*exp(-2.303*log10(columnRes));
			} else if (param->technode == 32){  
				Column_Power = (1.1294*log2(levelOutput)+8.8998)*1e-6;
				Column_Power += 0.079449*exp(-2.297*log10(columnRes));
			} else if (param->technode == 22){   
				Column_Power = (0.538*log2(levelOutput)+4.3753)*1e-6;
				Column_Power += 0.072341*exp(-2.303*log10(columnRes));
			} else if (param->technode == 14){   
				Column_Power = (0.3132*log2(levelOutput)+2.5681)*1e-6;
				Column_Power += 0.061085*exp(-2.303*log10(columnRes));
			} else if (param->technode == 10){   
				Column_Power = (0.1823*log2(levelOutput)+1.5073)*1e-6;
				Column_Power += 0.051580*exp(-2.303*log10(columnRes));
			} else {   // 7nm
				Column_Power = (0.1061*log2(levelOutput)+0.8847)*1e-6;
				Column_Power += 0.043555*exp(-2.303*log10(columnRes));
			}
		}
	}
	Column_Power *= (1+1.3e-3*(param->temp-300));
	Column_Energy = Column_Power * (log2(levelOutput)+1)*1e-9;
	return Column_Energy;
}
