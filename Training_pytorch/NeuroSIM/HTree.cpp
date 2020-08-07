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
#include "HTree.h"
#include "Param.h"

using namespace std;

extern Param *param;

HTree::HTree(const InputParameter& _inputParameter, const Technology& _tech, const MemCell& _cell): inputParameter(_inputParameter), tech(_tech), cell(_cell), FunctionUnit() {
	initialized = false;
}

void HTree::Initialize(int _numRow, int _numCol, double _delaytolerance, double _busWidth){
	if (initialized)
		cout << "[HTree] Warning: Already initialized!" << endl;
	
	numRow = _numRow;
	numCol = _numCol;     // num of Row and Col in tile/pe level

	delaytolerance = _delaytolerance;
	busWidth = _busWidth;

	numStage = 2*ceil(log2((double) max(numRow, numCol)))+1;   // vertical has N stage, horizontal has N+1 stage
	unitLengthWireResistance = param->unitLengthWireResistance;
	unitLengthWireCap = 0.2e-15/1e-6;;   // 0.2 fF/mm
	
	// define min INV resistance and capacitance to calculate repeater size
	widthMinInvN = MIN_NMOS_SIZE * tech.featureSize;
	widthMinInvP = tech.pnSizeRatio * MIN_NMOS_SIZE * tech.featureSize;
	CalculateGateArea(INV, 1, widthMinInvN, widthMinInvP, tech.featureSize * MAX_TRANSISTOR_HEIGHT, tech, &hMinInv, &wMinInv);
	CalculateGateCapacitance(INV, 1, widthMinInvN, widthMinInvP, hMinInv, tech, &capMinInvInput, &capMinInvOutput);
	double resOnRep = CalculateOnResistance(widthMinInvN, NMOS, inputParameter.temperature, tech) + CalculateOnResistance(widthMinInvP, PMOS, inputParameter.temperature, tech);
	// optimal repeater design to achieve highest speed
	repeaterSize = floor((double)sqrt( (double) resOnRep*unitLengthWireCap/capMinInvInput/unitLengthWireResistance));
	minDist = sqrt(2*resOnRep*(capMinInvOutput+capMinInvInput)/(unitLengthWireResistance*unitLengthWireCap));
	CalculateGateArea(INV, 1, MIN_NMOS_SIZE * tech.featureSize * repeaterSize, tech.pnSizeRatio * MIN_NMOS_SIZE * tech.featureSize * repeaterSize, tech.featureSize * MAX_TRANSISTOR_HEIGHT, tech, &hRep, &wRep);
	CalculateGateCapacitance(INV, 1, MIN_NMOS_SIZE * tech.featureSize * repeaterSize, tech.pnSizeRatio * MIN_NMOS_SIZE * tech.featureSize * repeaterSize, hRep, tech, &capRepInput, &capRepOutput);
	resOnRep = CalculateOnResistance(MIN_NMOS_SIZE * tech.featureSize * repeaterSize, NMOS, inputParameter.temperature, tech) + CalculateOnResistance(tech.pnSizeRatio * MIN_NMOS_SIZE * tech.featureSize * repeaterSize, PMOS, inputParameter.temperature, tech);
	double minUnitLengthDelay = 0.7*(resOnRep*(capRepInput+capRepOutput+unitLengthWireCap*minDist)+0.5*unitLengthWireResistance*minDist*unitLengthWireCap*minDist+unitLengthWireResistance*minDist*capRepInput)/minDist;
	double maxUnitLengthEnergy = (capRepInput+capRepOutput+unitLengthWireCap*minDist)*tech.vdd*tech.vdd/minDist;
	
	if (delaytolerance) {   // tradeoff: increase delay to decrease energy
		double delay = 0;
		double energy = 100;
		while(delay<minUnitLengthDelay*(1+delaytolerance)) {
			repeaterSize /=2;
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
	
	/*** define center point ***/
	x_center = floor(log2((double) min(numRow, numCol)));
	y_center = floor(log2((double) min(numRow, numCol)));
	int orc = 1;    // over-routing constraint: (important for unbalanced tree) avoid routing outside chip boundray
	
	if (numCol-x_center<orc) {
		x_center -= orc;
	}
	if (numRow-y_center<orc) {
		y_center -= orc;
	}  // redefine center point: try to slightly move to the actual chip center
	
	find_stage = 0;   // assume the top stage as find_stage = 0
	hit = 0;
	skipVer = 0;
	
	initialized = true;
}

void HTree::CalculateArea(double unitHeight, double unitWidth, double foldedratio) {
	if (!initialized) {
		cout << "[HTree] Error: Require initialization first!" << endl;
	} else {
		// INV
		CalculateGateArea(INV, 1, widthInvN, widthInvP, tech.featureSize * MAX_TRANSISTOR_HEIGHT, tech, &hInv, &wInv);
		
		area = 0;
		totalWireLength = 0;
		double wireLengthV = unitHeight*pow(2, (numStage-1)/2)/2;   // first vertical stage
		double wireLengthH = unitWidth*pow(2, (numStage-1)/2)/2;    // first horizontal stage (despite of main bus)
		double wireWidV = 0;
		double wireWidH = 0;
		double numRepeater = 0;
		
		for (int i=1; i<(numStage-1)/2; i++) {   // start from center point, consider both vertical and horizontal stage at each time, ignore last stage, assume it overlap with unit's layout
		
			double wireWidth, unitLengthWireResistance;
			wireWidth, unitLengthWireResistance = GetUnitLengthRes(i);

			/*** vertical stage ***/
			wireLengthV /= 2;   // wire length /2 
			numRepeater = ceil(wireLengthV/minDist);
			if (numRepeater > 0) {
				wireWidV += busWidth*wInv/foldedratio;   // which ever stage, the sum of wireWidth should always equal to busWidth (main bus width)
			} else {
				wireWidV += busWidth*wireWidth/foldedratio;
			}
			area += wireWidV*wireLengthV/2;
			
			/*** horizontal stage ***/
			wireLengthH /= 2;   // wire length /2 
			numRepeater = ceil(wireLengthH/minDist);
			if (numRepeater > 0) {
				wireWidH += busWidth*hInv/foldedratio;   // which ever stage, the sum of wireWidth should always equal to busWidth (main bus width)
			} else {
				wireWidH += busWidth*wireWidth/foldedratio;
			}
			area += wireWidH*wireLengthH/2;
			
			/*** count totalWireLength ***/
			totalWireLength += wireLengthV + wireLengthH;
		}
		totalWireLength += min(numCol-x_center, x_center)*unitWidth;
		area += (busWidth*hInv/foldedratio)*min(numCol-x_center, x_center)*unitWidth;   // main bus: find the way nearest to the boundray as source
		
		// Capacitance
		// INV
		CalculateGateCapacitance(INV, 1, widthInvN, widthInvP, hInv, tech, &capInvInput, &capInvOutput);
		
	}
}

void HTree::CalculateLatency(int x_init, int y_init, int x_end, int y_end, double unitHeight, double unitWidth, double numRead){
	if (!initialized) {
		cout << "[HTree] Error: Require initialization first!" << endl;
	} else {
		readLatency = 0;
		
		double wireLengthV = unitHeight*pow(2, (numStage-1)/2);   // first vertical stage
		double wireLengthH = unitWidth*pow(2, (numStage-1)/2);    // first horizontal stage (despite of main bus)
		double numRepeater = 0;
		double resOnRep = CalculateOnResistance(widthInvN, NMOS, inputParameter.temperature, tech) + CalculateOnResistance(widthInvP, PMOS, inputParameter.temperature, tech);
		
		if (((!x_init) && (!y_init)) || ((!x_end) && (!y_end))) {      // root-leaf communicate (fixed addr)
			for (int i=0; i<(numStage-1)/2; i++) {                     // ignore main bus here, but need to count until last stage (diff from area calculation)
				double wireWidth, unitLengthWireResistance;
				wireWidth, unitLengthWireResistance = GetUnitLengthRes(i);
				unitLatencyRep = 0.7*(resOnRep*(capInvInput+capInvOutput+unitLengthWireCap*minDist)+0.5*unitLengthWireResistance*minDist*unitLengthWireCap*minDist+unitLengthWireResistance*minDist*capInvInput)/minDist;
				unitLatencyWire = 0.7*unitLengthWireResistance*minDist*unitLengthWireCap*minDist/minDist;
			
				/*** vertical stage ***/
				wireLengthV /= 2;   // wire length /2 
				numRepeater = ceil(wireLengthV/minDist);
				if (numRepeater > 0) {
					readLatency += wireLengthV*unitLatencyRep;
				} else {
					readLatency += wireLengthV*unitLatencyWire;
				}
				
				/*** horizontal stage ***/
				wireLengthH /= 2;   // wire length /2 
				numRepeater = ceil(wireLengthH/minDist);
				if (numRepeater > 0) {
					readLatency += wireLengthH*unitLatencyRep;
				} else {
					readLatency += wireLengthH*unitLatencyWire;
				}
			}
			/*** main bus ***/
			readLatency += min(numCol-x_center, x_center)*unitWidth*unitLatencyRep;
		} else {       // leaf-leaf communicate
			/*** firstly need to find the zone of two units ***/
			/*** in each level, the units are defined as 4 zones, which used to decide the travel distance
			    ______________________
				|          |          |
				|          |          |
				|    0     |     1    |
				|          |          |
				|__________|__________|
				|          |          |
				|          |          |
				|    2     |     3    |      
				|          |          |
				|__________|__________|                       ***/

			while ((!hit) && (find_stage<(numStage-1)/2)) {
				double maxCoorDiff = pow(2, (numStage-1)/2-find_stage-1)-1;    // maximum difference of x- and y- coordinate at stage N: 2^(N-1)-1
				if ( abs(x_init-x_end)>maxCoorDiff || abs(y_init-y_end)>maxCoorDiff ) {
					// hit: means the belongs to different zone in this stage, stop searching
					hit = 1;
					if ( abs(x_init-x_end)<maxCoorDiff && abs(y_init-y_end)>maxCoorDiff ) {   // two zone belong to same row, do not pass the longest vertical bus at this stage
						skipVer = 1;
					}
				} else {  // keep searching in next stage
					find_stage += 1;
				}
			}
			/*** count the top find_stage, whether pass the vertical bus or not) ***/
			wireLengthV /= pow(2, find_stage);
			wireLengthH /= pow(2, find_stage);
			/*** horizontal stage ***/
			numRepeater = ceil(wireLengthH/minDist);
			if (numRepeater > 0) {
				readLatency += wireLengthH*unitLatencyRep;
			} else {
				readLatency += wireLengthH*unitLatencyWire;
			}
			if(!skipVer) {
				/*** vertical bus ***/
				numRepeater = ceil(wireLengthV/minDist);
				if (numRepeater > 0) {
					readLatency += wireLengthV*unitLatencyRep;
				} else {
					readLatency += wireLengthV*unitLatencyWire;
				}
			}
			/*** count the following stage ***/
			for (int i=find_stage+1; i<(numStage-1)/2; i++) {  
				double wireWidth, unitLengthWireResistance;
				wireWidth, unitLengthWireResistance = GetUnitLengthRes(i);
				unitLatencyRep = 0.7*(resOnRep*(capInvInput+capInvOutput+unitLengthWireCap*minDist)+0.5*unitLengthWireResistance*minDist*unitLengthWireCap*minDist+unitLengthWireResistance*minDist*capInvInput)/minDist;
				unitLatencyWire = 0.7*unitLengthWireResistance*minDist*unitLengthWireCap*minDist/minDist;
			
				/*** vertical stage ***/
				wireLengthV /= 2;   // wire length /2 
				numRepeater = ceil(wireLengthV/minDist);
				if (numRepeater > 0) {
					readLatency += wireLengthV*unitLatencyRep;
				} else {
					readLatency += wireLengthV*unitLatencyWire;
				}
				/*** horizontal stage ***/
				wireLengthH /= 2;   // wire length /2 
				numRepeater = ceil(wireLengthH/minDist);
				if (numRepeater > 0) {
					readLatency += wireLengthH*unitLatencyRep;
				} else {
					readLatency += wireLengthH*unitLatencyWire;
				}
			}
			// do not pass main bus
		}
		readLatency *= numRead;	
	}
}

void HTree::CalculatePower(int x_init, int y_init, int x_end, int y_end, double unitHeight, double unitWidth, double numBitAccess, double numRead) {
	if (!initialized) {
		cout << "[HTree] Error: Require initialization first!" << endl;
	} else {
		leakage = 0;
		readDynamicEnergy = 0;
		
		unitLengthLeakage = CalculateGateLeakage(INV, 1, widthInvN, widthInvP, inputParameter.temperature, tech) * tech.vdd / minDist;
		leakage = unitLengthLeakage * totalWireLength;
		unitLengthEnergyRep = (capInvInput+capInvOutput+unitLengthWireCap*minDist)*tech.vdd*tech.vdd/minDist*0.25;
		unitLengthEnergyWire = (unitLengthWireCap*minDist)*tech.vdd*tech.vdd/minDist*0.25;
		double wireLengthV = unitHeight*pow(2, (numStage-1)/2)/2;   // first vertical stage
		double wireLengthH = unitWidth*pow(2, (numStage-1)/2)/2;    // first horizontal stage (despite of main bus)
		
		if (((!x_init) && (!y_init)) || ((!x_end) && (!y_end))) {      // root-leaf communicate (fixed addr)
			for (int i=0; i<(numStage-1)/2; i++) {                     // ignore main bus here, but need to count until last stage (diff from area calculation)
				/*** vertical stage ***/
				wireLengthV /= 2;   // wire length /2 
				numRepeater = ceil(wireLengthV/minDist);
				if (numRepeater > 0) {
					readDynamicEnergy += wireLengthV*unitLengthEnergyRep;
				} else {
					readDynamicEnergy += wireLengthV*unitLengthEnergyWire;
				}
				/*** horizontal stage ***/
				wireLengthH /= 2;   // wire length /2 
				numRepeater = ceil(wireLengthH/minDist);
				if (numRepeater > 0) {
					readDynamicEnergy += wireLengthH*unitLengthEnergyRep;
				} else {
					readDynamicEnergy += wireLengthH*unitLengthEnergyWire;
				}
			}
			/*** main bus ***/
			readDynamicEnergy += min(numCol-x_center, x_center)*unitWidth*unitLengthEnergyRep;
			readDynamicEnergy *= numBitAccess;  
		} else {       // leaf-leaf communicate
			/*** count the top find_stage, whether pass the vertical bus or not) ***/
			wireLengthV /= pow(2, find_stage);
			wireLengthH /= pow(2, find_stage);
			/*** horizontal stage ***/
			numRepeater = ceil(wireLengthH/minDist);
			if (numRepeater > 0) {
				readDynamicEnergy += wireLengthH*unitLengthEnergyRep;
			} else {
				readDynamicEnergy += wireLengthH*unitLengthEnergyWire;
			}
			if(!skipVer) {
				/*** vertical bus ***/
				numRepeater = ceil(wireLengthV/minDist);
				if (numRepeater > 0) {
					readDynamicEnergy += wireLengthV*unitLengthEnergyRep;
				} else {
					readDynamicEnergy += wireLengthV*unitLengthEnergyWire;
				}
			}
			/*** count the following stage ***/
			for (int i=find_stage+1; i<(numStage-1)/2; i++) {  
				/*** vertical stage ***/
				wireLengthV /= 2;   // wire length /2 
				numRepeater = ceil(wireLengthV/minDist);
				if (numRepeater > 0) {
					readDynamicEnergy += wireLengthV*unitLengthEnergyRep;
				} else {
					readDynamicEnergy += wireLengthV*unitLengthEnergyWire;
				}
				/*** horizontal stage ***/
				wireLengthH /= 2;   // wire length /2 
				numRepeater = ceil(wireLengthH/minDist);
				if (numRepeater > 0) {
					readDynamicEnergy += wireLengthH*unitLengthEnergyRep;
				} else {
					readDynamicEnergy += wireLengthH*unitLengthEnergyWire;
				}
			}
			// do not pass main bus
			readDynamicEnergy *= numBitAccess;
		}
		readDynamicEnergy *= numRead;
	}
}

void HTree::PrintProperty(const char* str) {
	FunctionUnit::PrintProperty(str);
}

double HTree::GetUnitLengthRes(int numStage) {
	
	double wireWidth, AR, Rho, unitLengthWireResistance, wireResistance;
	
	if (numStage <= 2) {
		wireWidth = 4*param->wireWidth;
	} else if (2 < numStage <= 4) {
		wireWidth = 2*param->wireWidth;
	} else {
		wireWidth = 1*param->wireWidth;
	}
	
	if (wireWidth >= 200) {
		AR = 2.10; 
		Rho = 2.42e-8;
	} else if (100 <= wireWidth < 200) {
		AR = 2.30; 
		Rho = 2.73e-8;
	} else if (50 <= wireWidth < 100) {
		AR = 2.34; 
		Rho = 3.91e-8;
	} else if (40 <= wireWidth < 50) {
		AR = 1.90; 
		Rho = 4.03e-8;
	} else if (32 <= wireWidth < 40) {
		AR = 1.90; 
		Rho = 4.51e-8;
	} else if (22 <= wireWidth < 32) {
		AR = 2.00; 
		Rho = 5.41e-8;
	} else {
		AR = 2.10; 
		Rho = 7.43e-8;
	} 
	
	double wireLength = wireWidth * 1e-9 * 2;	
	if (wireWidth == -1) {
		unitLengthWireResistance = 1.0;	// Use a small number to prevent numerical error for NeuroSim
	} else {
		unitLengthWireResistance =  Rho / ( wireWidth*1e-9 * wireWidth*1e-9 * AR );
	}
	
	return wireWidth, unitLengthWireResistance;
}


