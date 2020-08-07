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
#include "constant.h"
#include "formula.h"
#include "WeightGradientUnit.h"
#include "Param.h"

using namespace std;
extern Param *param;

WeightGradientUnit::WeightGradientUnit(const InputParameter& _inputParameter, const Technology& _tech, const MemCell& _cell): 
										inputParameter(_inputParameter), tech(_tech), cell(_cell), FunctionUnit(),
										wlDecoder(_inputParameter, _tech, _cell),
										wlSwitchMatrix(_inputParameter, _tech, _cell),
										mux(_inputParameter, _tech, _cell),
										muxDecoder(_inputParameter, _tech, _cell),
										precharger(_inputParameter, _tech, _cell),
										senseAmp(_inputParameter, _tech, _cell),
										sramWriteDriver(_inputParameter, _tech, _cell),
										adder(_inputParameter, _tech, _cell),
										dff(_inputParameter, _tech, _cell),
										multilevelSenseAmp(_inputParameter, _tech, _cell),
										multilevelSAEncoder(_inputParameter, _tech, _cell),
										sarADC(_inputParameter, _tech, _cell),
										shiftAdd(_inputParameter, _tech, _cell),
										accumulation(_inputParameter, _tech, _cell),
										bufferInput(_inputParameter, _tech, _cell),
										bufferOutput(_inputParameter, _tech, _cell){
	initialized = false;
}

void WeightGradientUnit::Initialize(int _numMemRow, int _numMemCol) {
	if (initialized)
		cout << "[WeightGradientUnit] Warning: Already initialized!" << endl;

	numMemRow = _numMemRow;                     	   // total memory size needed to support calculation
	numMemCol = _numMemCol;                      	   // total memory size needed to support calculation
	
	numRow = param->numRowSubArrayWG;                  // user defined sub-array size
	numCol = param->numColSubArrayWG;                  // user defined sub-array size
	unitWireRes = param->unitLengthWireResistance;
	
	numArrayInRow = floor(numMemRow/numRow);           
	numArrayInCol = floor(numMemCol/numCol);
	
	lengthRow = (double) numCol * param->widthInFeatureSizeSRAM * tech.featureSize;
	lengthCol = (double) numRow * param->heightInFeatureSizeSRAM * tech.featureSize;
	
	capRow1 = lengthRow * 0.2e-15/1e-6;
	capRow2 = lengthRow * 0.2e-15/1e-6;	
	capCol = lengthCol * 0.2e-15/1e-6;
	
	resRow = lengthRow * unitWireRes; 
	resCol = lengthCol * unitWireRes;
	
	// SRAM based 
	int numColPerSynapse = param->numBitInput;             // the activation will be written into the SRAM array, so use input precision to define numColPerSynapse
	//firstly calculate the CMOS resistance and capacitance
	resCellAccess = CalculateOnResistance(param->widthAccessCMOS * tech.featureSize, NMOS, inputParameter.temperature, tech);
	capCellAccess = CalculateDrainCap(param->widthAccessCMOS * tech.featureSize, NMOS, param->widthInFeatureSizeSRAM * tech.featureSize, tech);
	capSRAMCell = capCellAccess + CalculateDrainCap(param->widthSRAMCellNMOS * tech.featureSize, NMOS, param->widthInFeatureSizeSRAM * tech.featureSize, tech) 
						+ CalculateDrainCap(param->widthSRAMCellPMOS * tech.featureSize, PMOS, param->widthInFeatureSizeSRAM * tech.featureSize, tech) 
						+ CalculateGateCap(param->widthSRAMCellNMOS * tech.featureSize, tech) + CalculateGateCap(param->widthSRAMCellPMOS * tech.featureSize, tech);

	if (! param->parallelBP) {
		wlDecoder.Initialize(REGULAR_ROW, (int)ceil(log2(numRow)), false, false);
		senseAmp.Initialize(numCol, false, param->minSenseVoltage, lengthRow/numCol, param->clkFreq, numCol);
		int adderBit = (int)ceil(log2(numRow)) + 1;	
		int numAdder = numCol/numColPerSynapse;
		dff.Initialize((adderBit+1)*numAdder, param->clkFreq);	
		adder.Initialize(adderBit, numAdder);
		if (param->numBitInput > 1) {
			shiftAdd.Initialize(numAdder, adderBit+param->numBitInput+1, param->clkFreq, spikingMode, param->numBitInput);
		}
		accumulation.Initialize(numArrayInRow, (log2((double)numRow))+param->numBitInput+1, ceil((double)numArrayInCol*(double)numCol/(double)param->numRowMuxedWG));
		outPrecision = (log2((double)numRow))+param->numBitInput+accumulation.numStage;
		
	} else {
		wlSwitchMatrix.Initialize(ROW_MODE, numRow, resCellAccess, true, false, param->activityRowReadWG, param->activityColWriteWG, numCol, numCol, 1, param->clkFreq);
		mux.Initialize(ceil(numCol/param->numRowMuxedWG), param->numRowMuxedWG, resCellAccess, false);       
		muxDecoder.Initialize(REGULAR_ROW, (int)ceil(log2(param->numRowMuxedWG)), true, false);
		if (param->SARADC) {
			sarADC.Initialize(numCol/param->numRowMuxedWG, param->levelOutputWG, param->clkFreq, numCol);
		} else {
			multilevelSenseAmp.Initialize(numCol/param->numRowMuxedWG, param->levelOutputWG, param->clkFreq, numCol, true, param->currentMode);
			multilevelSAEncoder.Initialize(param->levelOutputWG, numCol/param->numRowMuxedWG);
		}
		if (param->numBitInput > 1) {
			shiftAdd.Initialize(ceil(numCol/param->numRowMuxedWG), log2(param->levelOutputWG)+param->numBitInput+1, param->clkFreq, spikingMode, param->numBitInput);
		}
		accumulation.Initialize(numArrayInRow, log2((double)param->levelOutputWG)+param->numBitInput+1, ceil((double)numArrayInCol*(double)numCol/(double)param->numRowMuxedWG));
		outPrecision = log2((double)param->levelOutputWG)+param->numBitInput+accumulation.numStage;
		
	}	
	precharger.Initialize(numCol, resCol, param->activityColWriteWG, numCol, numCol);
	sramWriteDriver.Initialize(numCol, param->activityColWriteWG, numCol);
	
	initialized = true;
}


void WeightGradientUnit::CalculateArea() {  //calculate layout area for total design
	if (!initialized) {
		cout << "[WeightGradientUnit] Error: Require initialization first!" << endl;  //ensure initialization first
	} else {  //if initialized, start to do calculation
		area = 0;
		areaSubArray = 0;
		
		// Array only
		heightArray = lengthCol;
		widthArray = lengthRow;
		areaArray = heightArray * widthArray;
			
		//precharger and writeDriver are always needed for all different designs
		precharger.CalculateArea(NULL, widthArray, NONE);
		sramWriteDriver.CalculateArea(NULL, widthArray, NONE);
			
		if (! param->parallelBP) {
			wlDecoder.CalculateArea(heightArray, NULL, NONE);  
			senseAmp.CalculateArea(NULL, widthArray, MAGIC);
			adder.CalculateArea(NULL, widthArray, NONE);
			dff.CalculateArea(NULL, widthArray, NONE);
			if (param->numBitInput > 1) {
				shiftAdd.CalculateArea(NULL, widthArray, NONE);
			}
			areaSubArray = areaArray + wlDecoder.area + precharger.area + sramWriteDriver.area + senseAmp.area + adder.area + dff.area + shiftAdd.area;
			accumulation.CalculateArea(NULL, (double) sqrt(areaSubArray), NONE);

			area = areaSubArray*(numArrayInRow*numArrayInCol) + accumulation.area; 
		} else { 
			wlSwitchMatrix.CalculateArea(heightArray, NULL, NONE);
			
			mux.CalculateArea(NULL, widthArray, NONE);
			muxDecoder.CalculateArea(NULL, NULL, NONE);
			double minMuxHeight = MAX(muxDecoder.height, mux.height);
			mux.CalculateArea(minMuxHeight, widthArray, OVERRIDE);
			if (param->SARADC) {
				sarADC.CalculateUnitArea();
				sarADC.CalculateArea(NULL, widthArray, NONE);
			} else {
				multilevelSenseAmp.CalculateArea(NULL, widthArray, NONE);
				multilevelSAEncoder.CalculateArea(NULL, widthArray, NONE);
			}
			if (param->numBitInput > 1) {
				shiftAdd.CalculateArea(NULL, widthArray, NONE);
			}
			areaSubArray = areaArray + wlSwitchMatrix.area + precharger.area + sramWriteDriver.area + multilevelSenseAmp.area + multilevelSAEncoder.area + shiftAdd.area + mux.area + muxDecoder.area + sarADC.area;
			accumulation.CalculateArea(NULL, (double) sqrt(areaSubArray), NONE);
			
			area = areaSubArray*(numArrayInRow*numArrayInCol) + accumulation.area; 
		} 
	}
}


void WeightGradientUnit::CalculateLatency(int numRead, int numBitDataLoad) {
	if (!initialized) {
		cout << "[WeightGradientUnit] Error: Require initialization first!" << endl;
	} else {
		readLatency = 0;
		writeLatency = 0;
		
		if (! param->parallelBP) {
			wlDecoder.CalculateLatency(1e20, capRow1, NULL, numRow*param->activityRowReadWG*param->numBitInput, numRow*param->activityRowWriteWG);
			precharger.CalculateLatency(1e20, capCol, numRow*param->activityRowReadWG*param->numBitInput, numRow*param->activityRowWriteWG);
			sramWriteDriver.CalculateLatency(1e20, capCol, resCol, numRow*param->activityRowWriteWG);
			senseAmp.CalculateLatency(numRow*param->activityRowReadWG*param->numBitInput);
			dff.CalculateLatency(1e20, numRow*param->activityRowReadWG*param->numBitInput);
			adder.CalculateLatency(1e20, dff.capTgDrain, numRow*param->activityRowReadWG*param->numBitInput);
			if (param->numBitInput > 1) {
				shiftAdd.CalculateLatency(param->numBitInput);	
			}
			accumulation.CalculateLatency(1, numArrayInRow, 0);
			
			// Read
			double resPullDown = CalculateOnResistance(param->widthSRAMCellNMOS * tech.featureSize, NMOS, inputParameter.temperature, tech);
			double tau = (resCellAccess + resPullDown) * (capCellAccess + capCol) + resCol * capCol / 2;
			tau *= log(tech.vdd / (tech.vdd - param->minSenseVoltage / 2));   
			double gm = CalculateTransconductance(param->widthAccessCMOS * tech.featureSize, NMOS, tech);
			double beta = 1 / (resPullDown * gm);
			double colRamp = 0;
			colDelay = horowitz(tau, beta, wlDecoder.rampOutput, &colRamp) * numRow * param->numBitInput * param->activityRowReadWG;

			readLatency += wlDecoder.readLatency;
			readLatency += precharger.readLatency;
			readLatency += colDelay;
			readLatency += senseAmp.readLatency;
			readLatency += adder.readLatency;
			readLatency += dff.readLatency;
			readLatency += shiftAdd.readLatency;
			readLatency += accumulation.readLatency;
			
			// Write (assume the average delay of pullup and pulldown inverter in SRAM cell)
			double resPull;
			resPull = (CalculateOnResistance(param->widthSRAMCellNMOS * tech.featureSize, NMOS, inputParameter.temperature, tech)
						+ CalculateOnResistance(param->widthSRAMCellPMOS * tech.featureSize, PMOS, inputParameter.temperature, tech)) / 2;    // take average
			tau = resPull * capSRAMCell;
			gm = (CalculateTransconductance(param->widthSRAMCellNMOS * tech.featureSize, NMOS, tech) + CalculateTransconductance(param->widthSRAMCellPMOS * tech.featureSize, PMOS, tech)) / 2;   // take average
			beta = 1 / (resPull * gm);

			writeLatency += horowitz(tau, beta, 1e20, NULL) * numRow * param->activityRowWriteWG;
			writeLatency += wlDecoder.writeLatency;
			writeLatency += precharger.writeLatency;
			writeLatency += sramWriteDriver.writeLatency;

		} else {
			// consider a average trace: estimate the column Resistance for S/A
			double totalWireResistance = resCellAccess+param->wireResistanceCol; 
			double columnG = (double) 1.0/totalWireResistance * ceil(param->numRowSubArrayWG * param->activityRowReadWG);
			vector<double> columnResistance;
			for (int i=0; i<numCol; i++) {
				columnResistance.push_back((double) 1/columnG);
			}
			
			wlSwitchMatrix.CalculateLatency(1e20, capRow1, resRow, param->numRowMuxedWG*param->numBitInput, 2*numRow*param->activityRowWriteWG);
			precharger.CalculateLatency(1e20, capCol, param->numRowMuxedWG*param->numBitInput, numRow*param->activityRowWriteWG);
			sramWriteDriver.CalculateLatency(1e20, capCol, resCol, numRow*param->activityRowWriteWG);
			
			mux.CalculateLatency(0, 0, param->numRowMuxedWG*param->numBitInput);
			muxDecoder.CalculateLatency(1e20, mux.capTgGateN*ceil(numCol/param->numRowMuxedWG), mux.capTgGateP*ceil(numCol/param->numRowMuxedWG), param->numRowMuxedWG*param->numBitInput, 0);
			if (param->SARADC) {
				sarADC.CalculateLatency(param->numRowMuxedWG*param->numBitInput);
			} else {
				multilevelSenseAmp.CalculateLatency(columnResistance, param->numRowMuxedWG*param->numBitInput, 1);
				multilevelSAEncoder.CalculateLatency(1e20, param->numRowMuxedWG*param->numBitInput);
			}
			if (param->numBitInput > 1) {
				shiftAdd.CalculateLatency(param->numRowMuxedWG*param->numBitInput);	
			}
			accumulation.CalculateLatency(param->numColMuxed, numArrayInRow, 0);
			
			// Read
			double resPullDown = CalculateOnResistance(param->widthSRAMCellNMOS * tech.featureSize, NMOS, inputParameter.temperature, tech);
			double tau = (resCellAccess + resPullDown) * (capCellAccess + capCol) + resCol * capCol / 2;
			tau *= log(tech.vdd / (tech.vdd - param->minSenseVoltage / 2));   
			double gm = CalculateTransconductance(param->widthAccessCMOS * tech.featureSize, NMOS, tech);
			double beta = 1 / (resPullDown * gm);
			double colRamp = 0;
			colDelay = horowitz(tau, beta, 1e20, &colRamp) * param->numBitInput;
			
			readLatency = 0;
			readLatency += MAX(wlSwitchMatrix.readLatency, (mux.readLatency+muxDecoder.readLatency));
			readLatency += precharger.readLatency;
			readLatency += colDelay;
			readLatency += multilevelSenseAmp.readLatency;
			readLatency += multilevelSAEncoder.readLatency;
			readLatency += shiftAdd.readLatency;
			readLatency += accumulation.readLatency;
			readLatency += sarADC.readLatency;
			
			// Write (assume the average delay of pullup and pulldown inverter in SRAM cell)
			double resPull;
			resPull = (CalculateOnResistance(param->widthSRAMCellNMOS * tech.featureSize, NMOS, inputParameter.temperature, tech) 
						+ CalculateOnResistance(param->widthSRAMCellPMOS * tech.featureSize, PMOS, inputParameter.temperature, tech)) / 2;    // take average
			tau = resPull * capSRAMCell;
			gm = (CalculateTransconductance(param->widthSRAMCellNMOS * tech.featureSize, NMOS, tech) + CalculateTransconductance(param->widthSRAMCellPMOS * tech.featureSize, PMOS, tech)) / 2;   // take average
			beta = 1 / (resPull * gm);
			
			writeLatency += horowitz(tau, beta, 1e20, NULL) * numRow * param->activityRowWriteWG;
			writeLatency += wlSwitchMatrix.writeLatency;
			writeLatency += precharger.writeLatency;
			writeLatency += sramWriteDriver.writeLatency;
		}
		
		readLatency *= numRead;
		readLatencyPeak = readLatency;
		writeLatencyPeak = writeLatency;
		
	}
}


void WeightGradientUnit::CalculatePower(int numRead, int numBitDataLoad) {
	if (!initialized) {
		cout << "[WeightGradientUnit] Error: Require initialization first!" << endl;
	} else {
		readDynamicEnergy = 0;
		writeDynamicEnergy = 0;
		readDynamicEnergyArray = 0;

		// Array leakage (assume 2 INV)
		leakage = 0;
		leakage += CalculateGateLeakage(INV, 1, param->widthSRAMCellNMOS * tech.featureSize,
				param->widthSRAMCellPMOS * tech.featureSize, inputParameter.temperature, tech) * tech.vdd * 2;
		leakage *= numRow * numCol;

		if (! param->parallelBP) {
			wlDecoder.CalculatePower(numRow*param->activityRowReadWG*param->numBitInput, numRow*param->activityRowWriteWG);
			precharger.CalculatePower(numRow*param->activityRowReadWG*param->numBitInput, numRow*param->activityRowWriteWG);
			sramWriteDriver.CalculatePower(numRow*param->activityRowWriteWG);
			adder.CalculatePower(numRow*param->activityRowReadWG*param->numBitInput, numCol);				
			dff.CalculatePower(numRow*param->activityRowReadWG*param->numBitInput, numCol*(adder.numBit+1));
			senseAmp.CalculatePower(numRow*param->activityRowReadWG*param->numBitInput);
			if (param->numBitInput > 1) {
				shiftAdd.CalculatePower(numRow*param->activityRowReadWG*param->numBitInput);
			}
			// Array
			readDynamicEnergyArray = 0; // Just BL discharging
			writeDynamicEnergyArray = capSRAMCell * tech.vdd * tech.vdd * 2 * numCol * param->activityColWriteWG * numRow * param->activityRowWriteWG;    // flip Q and Q_bar

			// Read
			readDynamicEnergy += wlDecoder.readDynamicEnergy;
			readDynamicEnergy += precharger.readDynamicEnergy;
			readDynamicEnergy += readDynamicEnergyArray;
			readDynamicEnergy += adder.readDynamicEnergy;
			readDynamicEnergy += dff.readDynamicEnergy;
			readDynamicEnergy += senseAmp.readDynamicEnergy;
			readDynamicEnergy += shiftAdd.readDynamicEnergy;
			
			// Write
			writeDynamicEnergy += wlDecoder.writeDynamicEnergy;
			writeDynamicEnergy += precharger.writeDynamicEnergy;
			writeDynamicEnergy += sramWriteDriver.writeDynamicEnergy;
			writeDynamicEnergy += writeDynamicEnergyArray;

			// Leakage
			leakage += wlDecoder.leakage;
			leakage += wlSwitchMatrix.leakage;
			leakage += precharger.leakage;
			leakage += sramWriteDriver.leakage;
			leakage += senseAmp.leakage;
			leakage += dff.leakage;
			leakage += adder.leakage;
			leakage += shiftAdd.leakage;

		} else {
			// consider a average trace: estimate the column Resistance for S/A
			double totalWireResistance = resCellAccess+param->wireResistanceCol; 
			double columnG = (double) 1.0/totalWireResistance * ceil(param->numRowSubArrayWG * param->activityRowReadWG);
			
			vector<double> columnResistance;
			for (int i=0; i<numCol; i++) {
				columnResistance.push_back((double) 1/columnG);
			}
			
			wlSwitchMatrix.CalculatePower(param->numRowMuxedWG*param->numBitInput, 2*numRow*param->activityRowWriteWG, param->activityRowReadWG, param->activityColWriteWG);
			precharger.CalculatePower(param->numRowMuxedWG*param->numBitInput, numRow*param->activityRowWriteWG);
			sramWriteDriver.CalculatePower(numRow*param->activityRowWriteWG);
			
			mux.CalculatePower(param->numRowMuxedWG*param->numBitInput);	// Mux still consumes energy during row-by-row read
			muxDecoder.CalculatePower(param->numRowMuxedWG*param->numBitInput, 1);
			if (param->SARADC) {
				sarADC.CalculatePower(columnResistance, param->numRowMuxedWG*param->numBitInput);
			} else {
				multilevelSenseAmp.CalculatePower(columnResistance, param->numRowMuxedWG*param->numBitInput);
				multilevelSAEncoder.CalculatePower(param->numRowMuxedWG*param->numBitInput);
			}
			if (param->numBitInput > 1) {
				shiftAdd.CalculatePower(param->numRowMuxedWG*param->numBitInput);
			}
			// Array
			readDynamicEnergyArray = 0; // Just BL discharging
			writeDynamicEnergyArray = capSRAMCell * tech.vdd * tech.vdd * 2 * numCol * param->activityColWriteWG * numRow * param->activityRowWriteWG;    // flip Q and Q_bar
			// Read
			readDynamicEnergy += wlSwitchMatrix.readDynamicEnergy;
			readDynamicEnergy += precharger.readDynamicEnergy;
			readDynamicEnergy += readDynamicEnergyArray;
			readDynamicEnergy += multilevelSenseAmp.readDynamicEnergy;
			readDynamicEnergy += multilevelSAEncoder.readDynamicEnergy;
			readDynamicEnergy += mux.readDynamicEnergy;
			readDynamicEnergy += muxDecoder.readDynamicEnergy;
			readDynamicEnergy += shiftAdd.readDynamicEnergy;
			readDynamicEnergy += sarADC.readDynamicEnergy;
			
			
			// cout << "wlSwitchMatrix.readDynamicEnergy: " << wlSwitchMatrix.readDynamicEnergy << endl;
			// cout << "precharger.readDynamicEnergy: " << precharger.readDynamicEnergy << endl;
			// cout << "readDynamicEnergyArray: " << readDynamicEnergyArray << endl;
			// cout << "multilevelSenseAmp.readDynamicEnergy: " << multilevelSenseAmp.readDynamicEnergy << endl;
			// cout << "multilevelSAEncoder.readDynamicEnergy: " << multilevelSAEncoder.readDynamicEnergy << endl;
			// cout << "mux.readDynamicEnergy: " << mux.readDynamicEnergy << endl;
			// cout << "muxDecoder.readDynamicEnergy: " << muxDecoder.readDynamicEnergy << endl;
			// cout << "shiftAdd.readDynamicEnergy: " << shiftAdd.readDynamicEnergy << endl;
			// cout << "sarADC.readDynamicEnergy: " << sarADC.readDynamicEnergy << endl;

			// Write
			writeDynamicEnergy += wlSwitchMatrix.writeDynamicEnergy;
			writeDynamicEnergy += precharger.writeDynamicEnergy;
			writeDynamicEnergy += sramWriteDriver.writeDynamicEnergy;
			writeDynamicEnergy += writeDynamicEnergyArray;
			
			// cout << "wlSwitchMatrix.writeDynamicEnergy: " << wlSwitchMatrix.writeDynamicEnergy << endl;
			// cout << "precharger.writeDynamicEnergy: " << precharger.writeDynamicEnergy << endl;
			// cout << "sramWriteDriver.writeDynamicEnergy: " << sramWriteDriver.writeDynamicEnergy << endl;
			// cout << "writeDynamicEnergyArray: " << writeDynamicEnergyArray << endl;
			
			// Leakage
			leakage += wlSwitchMatrix.leakage;
			leakage += precharger.leakage;
			leakage += sramWriteDriver.leakage;
			leakage += multilevelSenseAmp.leakage;
			leakage += multilevelSAEncoder.leakage;
			leakage += shiftAdd.leakage;
		}
		accumulation.CalculatePower(numRead, numArrayInRow);
		
		readDynamicEnergyPeak = readDynamicEnergy*(numArrayInRow*numArrayInCol)*numRead + accumulation.readDynamicEnergy;
		writeDynamicEnergyPeak = writeDynamicEnergy*(numArrayInRow*numArrayInCol);

		readDynamicEnergy *= (numArrayInRow*numArrayInCol)*numRead;
		readDynamicEnergy += accumulation.readDynamicEnergy; 
		writeDynamicEnergy *= (numArrayInRow*numArrayInCol);

		leakage *= (numArrayInRow*numArrayInCol);
	}
}

void WeightGradientUnit::PrintProperty(const char* str) {
	FunctionUnit::PrintProperty(str);
}


