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

///////////////////////////////////////////////////////////////////////////////////////////////////////
////// This NewMux is used for switch S/A and Vbl during reading and writing ...  Column connect //////
///////////////////////////////////////////////////////////////////////////////////////////////////////


#include <cmath>
#include <iostream>
#include "constant.h"
#include "formula.h"
#include "NewMux.h"

using namespace std;

NewMux::NewMux(const InputParameter& _inputParameter, const Technology& _tech, const MemCell& _cell): inputParameter(_inputParameter), tech(_tech), cell(_cell), FunctionUnit() {
	// TODO Auto-generated constructor stub
	initialized = false;
}

void NewMux::Initialize(int _numInput){
	if (initialized)
		cout << "[NewMux] Warning: Already initialized!" << endl;
	
	numInput = _numInput;
	
	widthTgN = MIN_NMOS_SIZE * tech.featureSize;
	widthTgP = tech.pnSizeRatio * MIN_NMOS_SIZE * tech.featureSize;
	
	initialized = true;
}

void NewMux::CalculateArea(double _newHeight, double _newWidth, AreaModify _option) {
	if (!initialized) {
		cout << "[NewMux] Error: Require initialization first!" << endl;
	} else {
		double hTg, wTg;

		if (_newWidth && _option==NONE) {
			numRowTgPair = 1;
			double minCellWidth = 2 * (POLY_WIDTH + MIN_GAP_BET_GATE_POLY) * tech.featureSize; // min standard cell width for 1 Tg
			if (minCellWidth > _newWidth) {
				cout << "[NewMux] Error: NewMux width is even larger than the assigned width !" << endl;
			}

			int numTgPairPerRow = (int)(_newWidth / (minCellWidth*2));    // Get max # Tg pair per row (this is not the final # Tg pair per row because the last row may have less # Tg)
			///////////////////// numInput*3 because there are 3 Tg in each single mux //////////////////////////
			numRowTgPair = (int)ceil((double)numInput*3 / numTgPairPerRow); // Get min # rows based on this max # Tg pair per row
			numTgPairPerRow = (int)ceil((double)numInput*3 / numRowTgPair);     // Get # Tg pair per row based on this min # rows
			TgWidth = _newWidth / numTgPairPerRow / 2;	// division of 2 because there are 2 Tg in one pair
			int numFold = (int)(TgWidth / (0.5*minCellWidth)) - 1;  // get the max number of folding

			// widthTgN, widthTgP and numFold can determine the height and width of each pass gate
			CalculatePassGateArea(widthTgN, widthTgP, tech, numFold, &hTg, &wTg);

			width = _newWidth;
			height = hTg * numRowTgPair;

		} else {
			// Default (pass gate with folding=1)
			CalculatePassGateArea(widthTgN, widthTgP, tech, 1, &hTg, &wTg);
			width = wTg * 2 * numInput * 3;
			height = hTg;
		}
		
	    area = height * width;

	    // Modify layout
	    newHeight = _newHeight;
	    newWidth = _newWidth;
	    switch (_option) {
		    case MAGIC:
			    MagicLayout();       // if MAGIC, call Magiclayout() in FunctionUnit.cpp
			    break;
		    case OVERRIDE:
			    OverrideLayout();    // if OVERRIDE, call Overridelayout() in FunctionUnit.cpp
			    break;
		    default:    // NONE
			    break;
	    }

	    // Capacitance
	    // TG
	    capTgGateN = CalculateGateCap(widthTgN, tech);
	    capTgGateP = CalculateGateCap(widthTgP, tech);
	    CalculateGateCapacitance(INV, 1, widthTgN, widthTgP, hTg, tech, NULL, &capTgDrain);
	}
}


void NewMux::CalculateLatency(double _rampInput, double _capLoad, double numRead, double numWrite) {	// For simplicity, assume shift register is ideal
	if (!initialized) {
		cout << "[NewMux] Error: Require initialization first!" << endl;
	} else {
		rampInput = _rampInput;
		capLoad = _capLoad;
		double tr;  /* time constant */
		readLatency = 0;

		// TG
		tr = resTg*2 * (capTgDrain + 0.5*capTgGateN + 0.5*capTgGateP + capLoad);	// Calibration: use resTg*2 (only one transistor is transmitting signal in the pass gate) may be more accurate, and include gate cap because the voltage at the source of NMOS and drain of PMOS is changing (assuming Cg = 0.5Cgs + 0.5Cgd)
		readLatency += 2.3 * tr;	// 2.3 means charging from 0% to 90%
		readLatency *= numRead;
		writeLatency = cell.writePulseWidth;     // write latency determined by write pulse width
		writeLatency *= numWrite;
	}
}

void NewMux::CalculatePower(double numRead, double numWrite, double numWritePulse, int mode_1T1R, double activityRowRead, double activityColWrite) {      
	if (!initialized) {
		cout << "[NewMux] Error: Require initialization first!" << endl;
	} else {
		
		leakage = 0;
		readDynamicEnergy = 0;
		writeDynamicEnergy = 0;
		
		// Read dynamic energy
		readDynamicEnergy += capTgDrain*3 * cell.readVoltage * cell.readVoltage * numInput;    // 2 TG pass Vread to BL, total loading is 3 Tg Drain capacitance
		readDynamicEnergy += (capTgGateN + capTgGateP) * 2 * tech.vdd * tech.vdd * numInput;    // open 2 TG when selected
		readDynamicEnergy *= numRead;
		readDynamicEnergy *= activityRowRead;
		
		// Write dynamic energy (2-step write and average case half SET and half RESET)
		if (mode_1T1R) {
			// LTP
			writeDynamicEnergy += (capTgDrain * 2) * cell.writeVoltage * cell.writeVoltage * numWritePulse * numInput*activityColWrite/2;   // Selected columns, '/2' means half of the writing cells are LTP
			writeDynamicEnergy += (capTgDrain * 2) * cell.writeVoltage * cell.writeVoltage * (numInput - numInput*activityColWrite/2);   // Unselected columns 
			// LTD
			writeDynamicEnergy += (capTgDrain * 2) * cell.writeVoltage * cell.writeVoltage * numWritePulse * numInput*activityColWrite/2;   // Selected columns	
			writeDynamicEnergy += (capTgGateN + capTgGateP) * tech.vdd * tech.vdd * numInput;
		}else {
			writeDynamicEnergy += (capTgDrain * 2) * cell.writeVoltage * cell.writeVoltage * numWritePulse * numInput*activityColWrite/2;   // Selected columns in LTP
			writeDynamicEnergy += (capTgDrain * 2) * cell.writeVoltage * cell.writeVoltage * numWritePulse * numInput*activityColWrite/2;   // Selected columns in LTD
			writeDynamicEnergy += (capTgDrain * 2) * cell.writeVoltage/2 * cell.writeVoltage/2 * numInput * (1-activityColWrite);   // Total unselected columns in LTP and LTD within the 2-step write
			writeDynamicEnergy += (capTgGateN + capTgGateP) * tech.vdd * tech.vdd * numInput;
		}
	}
}


void NewMux::PrintProperty(const char* str) {
	//cout << "NewMux Properties:" << endl;
	FunctionUnit::PrintProperty(str);
}

