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
#include "Buffer.h"
#include "Param.h"

using namespace std;

extern Param *param;


Buffer::Buffer(const InputParameter& _inputParameter, const Technology& _tech, const MemCell& _cell): inputParameter(_inputParameter), tech(_tech), cell(_cell), 
                      wlDecoder(_inputParameter, _tech, _cell), 
					  precharger(_inputParameter, _tech, _cell), 
					  sramWriteDriver(_inputParameter, _tech, _cell), 
					  senseAmp(_inputParameter, _tech, _cell), FunctionUnit() {
	initialized = false;
}

void Buffer::Initialize(int _numBit, int _interface_width, int _num_interface, double _unitWireRes, double _clkFreq, bool _SRAM){
	if (initialized)
		cout << "[Buffer] Warning: Already initialized!" << endl;
	
	numBit = _numBit;                             // # of bits that Buffer can store
	interface_width = _interface_width;           // # of bits in a "line", normally refered as # of column
	num_interface = _num_interface;               // # of interface
	unitWireRes = _unitWireRes;                   // Wire unit Resistance
	clkFreq = _clkFreq;                           // assigned clock frequency
	SRAM = _SRAM;                                 // SRAM based or DFF based?
	
	if (SRAM) {
		lengthRow = (double)interface_width * param->widthInFeatureSizeSRAM * tech.featureSize;
		lengthCol = (double)ceil((double)numBit/(double)interface_width) * param->heightInFeatureSizeSRAM * tech.featureSize;
		
		precharger.Initialize(interface_width, lengthCol * unitWireRes, 1, interface_width, interface_width);
		sramWriteDriver.Initialize(interface_width, 1, interface_width);
	} else {
		widthInvN = MIN_NMOS_SIZE * tech.featureSize;
		widthInvP = tech.pnSizeRatio * MIN_NMOS_SIZE * tech.featureSize;
	}
	
	wlDecoder.Initialize(REGULAR_ROW, (int)ceil((double)log2((double)ceil((double)numBit/(double)interface_width))), false, false);
	
	initialized = true;
}

void Buffer::CalculateArea(double _newHeight, double _newWidth, AreaModify _option) {
	if (!initialized) {
		cout << "[Buffer] Error: Require initialization first!" << endl;
	} else {
		area = 0;
		
		if (SRAM) {
			memoryArea = lengthRow * lengthCol;
			wlDecoder.CalculateArea(lengthCol, NULL, NONE);
			precharger.CalculateArea(NULL, lengthRow, NONE);
			sramWriteDriver.CalculateArea(NULL, lengthRow, NONE);
			area += memoryArea + wlDecoder.area + precharger.area + sramWriteDriver.area;
		} else {
			CalculateGateArea(INV, 1, widthInvN, widthInvP, tech.featureSize*MAX_TRANSISTOR_HEIGHT, tech, &hDffInv, &wDffInv);
			hDff = hDffInv;
			wDff = wDffInv * 12;
			memoryArea = hDff * wDff * numBit;
			wlDecoder.CalculateArea(hDff*ceil((double)numBit/(double)interface_width), NULL, NONE);
			area += memoryArea + wlDecoder.area;
			
			// Capacitance
			// INV
			CalculateGateCapacitance(INV, 1, widthInvN, widthInvP, hDffInv, tech, &capInvInput, &capInvOutput);
			// TG
			capTgGateN = CalculateGateCap(widthInvN, tech);
			capTgGateP = CalculateGateCap(widthInvP, tech);
			CalculateGateCapacitance(INV, 1, widthInvN, widthInvP, hDffInv, tech, NULL, &capTgDrain);
		}

		if (_newWidth && _option==NONE) {
			width = _newWidth;
			height = area/width;
		} else if (_newHeight && _option==NONE) {
			height = _newHeight;
			width = area/height;
		} else {
			cout << "[Buffer] Error: No width assigned for the buffer circuit" << endl;
			exit(-1);
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

void Buffer::CalculateLatency(double numAccessBitRead, double numRead, double numAccessBitWrite, double numWrite){
	if (!initialized) {
		cout << "[Buffer] Error: Require initialization first!" << endl;
	} else {
		readLatency = 0;
		writeLatency = 0;
		readWholeLatency = 0;
		writeWholeLatency = 0;
		
		if (SRAM) {
			wlDecoder.CalculateLatency(1e20, lengthRow * 0.2e-15/1e-6, NULL, (double) numBit/interface_width, (double) numBit/interface_width);
			precharger.CalculateLatency(1e20, lengthCol * 0.2e-15/1e-6, (double) numBit/interface_width, (double) numBit/interface_width);
			sramWriteDriver.CalculateLatency(1e20, lengthCol * 0.2e-15/1e-6, lengthCol * unitWireRes, (double) numBit/interface_width);
			
			double resCellAccess = CalculateOnResistance(param->widthAccessCMOS * tech.featureSize, NMOS, inputParameter.temperature, tech);
			double capCellAccess = CalculateDrainCap(param->widthAccessCMOS * tech.featureSize, NMOS, param->widthInFeatureSizeSRAM * tech.featureSize, tech);
			double resPullDown = CalculateOnResistance(param->widthSRAMCellNMOS * tech.featureSize, NMOS, inputParameter.temperature, tech);
			double tau = (resCellAccess + resPullDown) * (capCellAccess + lengthCol * 0.2e-15/1e-6) + lengthCol * unitWireRes * (lengthCol * 0.2e-15/1e-6) / 2;
			tau *= log(tech.vdd / (tech.vdd - param->minSenseVoltage / 2));   
			double gm = CalculateTransconductance(param->widthAccessCMOS * tech.featureSize, NMOS, tech);
			double beta = 1 / (resPullDown * gm);
			double colRamp = 0;
			colDelay = horowitz(tau, beta, wlDecoder.rampOutput, &colRamp)*((double) numBit/interface_width);
			readWholeLatency += wlDecoder.readLatency + precharger.readLatency + colDelay;
			writeWholeLatency += wlDecoder.writeLatency + precharger.writeLatency + sramWriteDriver.writeLatency;
		} else {
			wlDecoder.CalculateLatency(1e20, wDff * interface_width * 0.2e-15/1e-6, NULL, (double) numBit/interface_width, (double) numBit/interface_width);
			readWholeLatency += wlDecoder.readLatency;
			readWholeLatency += ((double) 1/clkFreq/2)*((double) numBit/interface_width);  // assume dff need half clock cycle to access
			writeWholeLatency += wlDecoder.writeLatency + ((double) 1/clkFreq/2)*((double) numBit/interface_width);
		}
		avgBitReadLatency = (double) readWholeLatency/(numBit/interface_width);     // average latency per line(sec/line)
		avgBitWriteLatency = (double) writeWholeLatency/(numBit/interface_width);
		readLatency = avgBitReadLatency*numRead;
		writeLatency = avgBitWriteLatency*numWrite;
	}
}

void Buffer::CalculatePower(double numAccessBitRead, double numRead, double numAccessBitWrite, double numWrite) {
	if (!initialized) {
		cout << "[Buffer] Error: Require initialization first!" << endl;
	} else {
		readDynamicEnergy = 0;
		writeDynamicEnergy = 0;
		leakage = 0;
		readWholeDynamicEnergy = 0;
		writeWholeDynamicEnergy = 0;
		
		if (SRAM) {
			wlDecoder.CalculatePower(numBit/interface_width, numBit/interface_width);
			precharger.CalculatePower(numBit/interface_width, numBit/interface_width);
			sramWriteDriver.CalculatePower(numBit/interface_width);
			readWholeDynamicEnergy += wlDecoder.readDynamicEnergy + precharger.readDynamicEnergy + sramWriteDriver.readDynamicEnergy;
			writeWholeDynamicEnergy += wlDecoder.writeDynamicEnergy + precharger.writeDynamicEnergy + sramWriteDriver.writeDynamicEnergy;
			leakage += wlDecoder.leakage + precharger.leakage + sramWriteDriver.leakage + senseAmp.leakage;
		} else {
			dffDynamicEnergy = 0;
			wlDecoder.CalculatePower(numBit/interface_width, numBit/interface_width);
			// Assume input D=1 and the energy of CLK INV and CLK TG are for 1 clock cycles
			// CLK INV (all DFFs have energy consumption)
			dffDynamicEnergy += (capInvInput + capInvOutput) * tech.vdd * tech.vdd * 4 * numBit;
			// CLK TG (all DFFs have energy consumption)
			dffDynamicEnergy += capTgGateN * tech.vdd * tech.vdd * 2 * numBit;
			dffDynamicEnergy += capTgGateP * tech.vdd * tech.vdd * 2 * numBit;
			// D to Q path (only selected DFFs have energy consumption)
			dffDynamicEnergy += (capTgDrain * 3 + capInvInput) * tech.vdd * tech.vdd * numBit;	    // D input side
			dffDynamicEnergy += (capTgDrain  + capInvOutput) * tech.vdd * tech.vdd * numBit;	    // D feedback side
			dffDynamicEnergy += (capInvInput + capInvOutput) * tech.vdd * tech.vdd * numBit;	    // Q output side
			
			readWholeDynamicEnergy += wlDecoder.readDynamicEnergy + dffDynamicEnergy;
			writeWholeDynamicEnergy += wlDecoder.writeDynamicEnergy + dffDynamicEnergy;
			
			leakage += CalculateGateLeakage(INV, 1, widthInvN, widthInvP, inputParameter.temperature, tech) * tech.vdd * 8 * numBit;
			leakage += wlDecoder.leakage;
		}
		avgBitReadDynamicEnergy = readWholeDynamicEnergy/numBit;
		avgBitWriteDynamicEnergy = writeWholeDynamicEnergy/numBit;
		
		readDynamicEnergy = avgBitReadDynamicEnergy*numAccessBitRead*numRead;
		writeDynamicEnergy = avgBitWriteDynamicEnergy*numAccessBitWrite*numWrite;
	}
}

void Buffer::PrintProperty(const char* str) {
	FunctionUnit::PrintProperty(str);
}











