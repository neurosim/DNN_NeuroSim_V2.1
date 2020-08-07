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

#ifndef BUS_H_
#define BUS_H_

#include "typedef.h"
#include "InputParameter.h"
#include "Technology.h"
#include "MemCell.h"
#include "FunctionUnit.h"

class Bus: public FunctionUnit {
public:
	Bus(const InputParameter& _inputParameter, const Technology& _tech, const MemCell& _cell);
	virtual ~Bus() {}
	const InputParameter& inputParameter;
	const Technology& tech;
	const MemCell& cell;

	/* Functions */
	void PrintProperty(const char* str);
	void SaveOutput(const char* str);
	void Initialize(BusMode _mode, int _numRow, int _numCol, double _delaytolerance, double _busWidth, double _unitHeight, double _unitWidth);
	void CalculateArea(double foldedratio, bool overLap);
	void CalculateLatency(double numRead);
	void CalculatePower(double numBitAccess, double numRead);

	/* Properties */
	bool initialized;	/* Initialization flag */
	double widthInvN, widthInvP, wInv, hInv, capInvInput, capInvOutput;
	double widthMinInvN, widthMinInvP, wMinInv, hMinInv, capMinInvInput, capMinInvOutput, wRep, hRep, capRepInput, capRepOutput;
	double AR, Rho, unitLengthWireResistance, minDist, minDelay, resOnRep;
	int numRow, numCol, numRepeater, repeaterSize;
	double unitHeight, unitWidth, wireWidth;
	double busWidth, delaytolerance, unitLengthWireCap, wireLength;
	double unitLatencyRep, unitLatencyWire, unitLengthLeakage, unitLengthEnergyRep, unitLengthEnergyWire;
	BusMode mode;
};

#endif /* BUS_H_ */
