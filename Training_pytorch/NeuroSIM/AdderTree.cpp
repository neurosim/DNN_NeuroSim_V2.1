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
#include "AdderTree.h"

using namespace std;

AdderTree::AdderTree(const InputParameter& _inputParameter, const Technology& _tech, const MemCell& _cell): inputParameter(_inputParameter), tech(_tech), cell(_cell), adder(_inputParameter, _tech, _cell), FunctionUnit() {
	initialized = false;
}

void AdderTree::Initialize(int _numSubcoreRow, int _numAdderBit, int _numAdderTree) {
	if (initialized)
		cout << "[AdderTree] Warning: Already initialized!" << endl;
	
	numSubcoreRow = _numSubcoreRow;                  // # of row of subcore in the synaptic core
	numStage = ceil(log2(numSubcoreRow));            // # of stage of the adder tree, used for CalculateLatency ...
	numAdderBit = _numAdderBit;                      // # of input bits of the Adder
	numAdderTree = _numAdderTree;                    // # of Adder Tree
	
	initialized = true;
}

void AdderTree::CalculateArea(double _newHeight, double _newWidth, AreaModify _option) {
	if (!initialized) {
		cout << "[AdderTree] Error: Require initialization first!" << endl;
	} else {
		double hInv, wInv, hNand, wNand;
		
		// Adder
		int numAdderEachStage = 0;                          // define # of adder in each stage
		int numBitEachStage = numAdderBit;                  // define # of bits of the adder in each stage
		int numAdderEachTree = 0;                           // define # of Adder in each Adder Tree
		int i = ceil(log2(numSubcoreRow));
		int j = numSubcoreRow;
		
		while (i != 0) {  // calculate the total # of full adder in each Adder Tree
			numAdderEachStage = ceil(j/2);
			numAdderEachTree += numBitEachStage*numAdderEachStage;
			numBitEachStage += 1;
			j = ceil(j/2);
			i -= 1;
		}
		adder.Initialize(numAdderEachTree, numAdderTree);   
		
		if (_newWidth && _option==NONE) {
			adder.CalculateArea(NULL, _newWidth, NONE);
			width = _newWidth;
			height = adder.area/width;
		} else if (_newHeight && _option==NONE) {
			adder.CalculateArea(_newHeight, NULL, NONE);
			height = _newHeight;
			width = adder.area/height;
		} else {
			cout << "[AdderTree] Error: No width assigned for the adder tree circuit" << endl;
			exit(-1);
		}
		area = height*width;
		adder.initialized = false;
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

void AdderTree::CalculateLatency(double numRead, int numUnitAdd, double _capLoad) {
	if (!initialized) {
		cout << "[AdderTree] Error: Require initialization first!" << endl;
	} else {
		readLatency = 0;
		
		int numAdderEachStage = 0;                          // define # of adder in each stage
		int numBitEachStage = numAdderBit;                  // define # of bits of the adder in each stage
		int numAdderEachTree = 0;                           // define # of Adder in each Adder Tree
		int i = 0;
		int j = 0;
		
		if (!numUnitAdd) {
			i = ceil(log2(numSubcoreRow));
			j = numSubcoreRow;
		} else {
			i = ceil(log2(numUnitAdd));
			j = numUnitAdd;
		}

		while (i != 0) {   // calculate the total # of full adder in each Adder Tree
			numAdderEachStage = ceil(j/2);
			adder.Initialize(numBitEachStage, numAdderEachStage);   
			adder.CalculateLatency(1e20, _capLoad, 1);
			readLatency += adder.readLatency;
			numBitEachStage += 1;
			j = ceil(j/2);
			i -= 1;
			
			adder.initialized = false;
		}
        readLatency *= numRead;		
	}
}

void AdderTree::CalculatePower(double numRead, int numUnitAdd) {
	if (!initialized) {
		cout << "[AdderTree] Error: Require initialization first!" << endl;
	} else {
		leakage = 0;
		readDynamicEnergy = 0;
		
		int numAdderEachStage = 0;                          // define # of adder in each stage
		int numBitEachStage = numAdderBit;                  // define # of bits of the adder in each stage
		int numAdderEachTree = 0;                           // define # of Adder in each Adder Tree
		int i = 0;
		int j = 0;
		
		if (!numUnitAdd) {
			i = ceil(log2(numSubcoreRow));
			j = numSubcoreRow;
		} else {
			i = ceil(log2(numUnitAdd));
			j = numUnitAdd;
		}
		
		while (i != 0) {  // calculate the total # of full adder in each Adder Tree
			numAdderEachStage = ceil(j/2);
			adder.Initialize(numBitEachStage, numAdderEachStage);     
			adder.CalculatePower(1, numAdderEachStage);	
			readDynamicEnergy += adder.readDynamicEnergy;	
			leakage += adder.leakage;
			numBitEachStage += 1;
			j = ceil(j/2);
			i -= 1;
			
			adder.initialized = false;
		}
		readDynamicEnergy *= numAdderTree;	
		readDynamicEnergy *= numRead;
		leakage *= numAdderTree;
	}
}

void AdderTree::PrintProperty(const char* str) {
	FunctionUnit::PrintProperty(str);
}

