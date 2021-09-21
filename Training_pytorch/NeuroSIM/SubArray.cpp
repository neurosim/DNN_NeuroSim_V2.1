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
#include "SubArray.h"


using namespace std;

SubArray::SubArray(InputParameter& _inputParameter, Technology& _tech, MemCell& _cell):
						inputParameter(_inputParameter), tech(_tech), cell(_cell),
						wlDecoder(_inputParameter, _tech, _cell),
						wlDecoderDriver(_inputParameter, _tech, _cell),
						wlNewDecoderDriver(_inputParameter, _tech, _cell),
						wlSwitchMatrix(_inputParameter, _tech, _cell),
						wlNewSwitchMatrix(_inputParameter, _tech, _cell),
						slSwitchMatrix(_inputParameter, _tech, _cell),
						mux(_inputParameter, _tech, _cell),
						muxDecoder(_inputParameter, _tech, _cell),
						precharger(_inputParameter, _tech, _cell),
						senseAmp(_inputParameter, _tech, _cell),
						sramWriteDriver(_inputParameter, _tech, _cell),
						rowCurrentSenseAmp(_inputParameter, _tech, _cell),
						adder(_inputParameter, _tech, _cell),
						dff(_inputParameter, _tech, _cell),
						multilevelSenseAmp(_inputParameter, _tech, _cell),
						multilevelSAEncoder(_inputParameter, _tech, _cell),
						sarADC(_inputParameter, _tech, _cell),
						shiftAddInput(_inputParameter, _tech, _cell),
						shiftAddWeight(_inputParameter, _tech, _cell),
						/* for BP (Transpose SubArray) */
						wlDecoderBP(_inputParameter, _tech, _cell),
						wlSwitchMatrixBP(_inputParameter, _tech, _cell),
						prechargerBP(_inputParameter, _tech, _cell),
						senseAmpBP(_inputParameter, _tech, _cell),
						sramWriteDriverBP(_inputParameter, _tech, _cell),
						muxBP(_inputParameter, _tech, _cell),
						muxDecoderBP(_inputParameter, _tech, _cell),
						rowCurrentSenseAmpBP(_inputParameter, _tech, _cell),
						adderBP(_inputParameter, _tech, _cell),
						dffBP(_inputParameter, _tech, _cell),
						multilevelSenseAmpBP(_inputParameter, _tech, _cell),
						multilevelSAEncoderBP(_inputParameter, _tech, _cell),
						sarADCBP(_inputParameter, _tech, _cell),
						shiftAddBPInput(_inputParameter, _tech, _cell),
						shiftAddBPWeight(_inputParameter, _tech, _cell){
	initialized = false;
	readDynamicEnergyArray = writeDynamicEnergyArray = 0;
}

void SubArray::Initialize(int _numRow, int _numCol, double _unitWireRes){  //initialization module
	
	numRow = _numRow;    //import parameters
	numCol = _numCol;
	unitWireRes = _unitWireRes;
	
	double MIN_CELL_HEIGHT = MAX_TRANSISTOR_HEIGHT;  //set real layout cell height
	double MIN_CELL_WIDTH = (MIN_GAP_BET_GATE_POLY + POLY_WIDTH) * 2;  //set real layout cell width
	if (cell.memCellType == Type::SRAM) {  //if array is SRAM
		if (relaxArrayCellWidth) {  //if want to relax the cell width
			lengthRow = (double)numCol * MAX(cell.widthInFeatureSize, MIN_CELL_WIDTH) * tech.featureSize;
		} else { //if not relax the cell width
			lengthRow = (double)numCol * cell.widthInFeatureSize * tech.featureSize;
		}
		if (relaxArrayCellHeight) {  //if want to relax the cell height
			lengthCol = (double)numRow * MAX(cell.heightInFeatureSize, MIN_CELL_HEIGHT) * tech.featureSize;
		} else {  //if not relax the cell height
			lengthCol = (double)numRow * cell.heightInFeatureSize * tech.featureSize;
		}
	
	} else if (cell.memCellType == Type::RRAM ||  cell.memCellType == Type::FeFET) {  //if array is RRAM
		double cellHeight = cell.heightInFeatureSize; 
		double cellWidth = cell.widthInFeatureSize;  
		if (cell.accessType == CMOS_access) {  // 1T1R
			if (relaxArrayCellWidth) {
				lengthRow = (double)numCol * MAX(cellWidth, MIN_CELL_WIDTH*2) * tech.featureSize;	// Width*2 because generally switch matrix has 2 pass gates per column, even the SL/BL driver has 2 pass gates per column in traditional 1T1R memory
			} else {
				lengthRow = (double)numCol * cellWidth * tech.featureSize;
			}
			if (relaxArrayCellHeight) {
				lengthCol = (double)numRow * MAX(cellHeight, MIN_CELL_HEIGHT) * tech.featureSize;
			} else {
				lengthCol = (double)numRow * cellHeight * tech.featureSize;
			}
		} else {	// Cross-point, if enter anything else except 'CMOS_access'
			if (relaxArrayCellWidth) {
				lengthRow = (double)numCol * MAX(cellWidth*cell.featureSize, MIN_CELL_WIDTH*2*tech.featureSize);	// Width*2 because generally switch matrix has 2 pass gates per column, even the SL/BL driver has 2 pass gates per column in traditional 1T1R memory
			} else {
				lengthRow = (double)numCol * cellWidth * cell.featureSize;
			}
			if (relaxArrayCellHeight) {
				lengthCol = (double)numRow * MAX(cellHeight*cell.featureSize, MIN_CELL_HEIGHT*tech.featureSize);
			} else {  
				lengthCol = (double)numRow * cellHeight * cell.featureSize;
			}
		}
	}      //finish setting array size
	
	capRow1 = lengthRow * 0.2e-15/1e-6;	// BL for 1T1R, WL for Cross-point and SRAM
	capRow2 = lengthRow * 0.2e-15/1e-6;	// WL for 1T1R
	capCol = lengthCol * 0.2e-15/1e-6;
	
	resRow = lengthRow * unitWireRes; 
	resCol = lengthCol * unitWireRes;
	
	//start to initializing the subarray modules
	if (cell.memCellType == Type::SRAM) {  //if array is SRAM
		
		//firstly calculate the CMOS resistance and capacitance
		resCellAccess = CalculateOnResistance(cell.widthAccessCMOS * tech.featureSize, NMOS, inputParameter.temperature, tech);
		capCellAccess = CalculateDrainCap(cell.widthAccessCMOS * tech.featureSize, NMOS, cell.widthInFeatureSize * tech.featureSize, tech);
		cell.capSRAMCell = capCellAccess + CalculateDrainCap(cell.widthSRAMCellNMOS * tech.featureSize, NMOS, cell.widthInFeatureSize * tech.featureSize, tech) + CalculateDrainCap(cell.widthSRAMCellPMOS * tech.featureSize, PMOS, cell.widthInFeatureSize * tech.featureSize, tech) + CalculateGateCap(cell.widthSRAMCellNMOS * tech.featureSize, tech) + CalculateGateCap(cell.widthSRAMCellPMOS * tech.featureSize, tech);

		if (conventionalSequential) {
			wlDecoder.Initialize(REGULAR_ROW, (int)ceil(log2(numRow)), false, false);
			senseAmp.Initialize(numCol, false, cell.minSenseVoltage, lengthRow/numCol, clkFreq, numReadCellPerOperationNeuro);
			int adderBit = (int)ceil(log2(numRow)) + 1;	
			int numAdder = numCol/numCellPerSynapse;
			dff.Initialize((adderBit+1)*numAdder, clkFreq);	
			adder.Initialize(adderBit, numAdder);
			if (numCellPerSynapse > 1) {
				shiftAddWeight.Initialize(numAdder, adderBit, clkFreq, spikingMode, numCellPerSynapse);
			}
			if (numReadPulse > 1) {
				shiftAddInput.Initialize(numAdder, adderBit+numCellPerSynapse, clkFreq, spikingMode, numReadPulse);
			}
			
			/* Transpose Peripheral for BP */
			if (trainingEstimation) {
				wlDecoderBP.Initialize(REGULAR_ROW, (int)ceil(log2(numCol)), false, false);
				senseAmpBP.Initialize(numCol, false, cell.minSenseVoltage, lengthCol/numRow, clkFreq, numReadCellPerOperationNeuro);
				int adderBit = (int)ceil(log2(numCol)) + 1;	
				int numAdder = numRow/numCellPerSynapse;
				dffBP.Initialize((adderBit+1)*numAdder, clkFreq); 
				adderBP.Initialize(adderBit, numAdder);
				if (numCellPerSynapse > 1) {
					shiftAddBPWeight.Initialize(numAdder, adderBit, clkFreq, spikingMode, numCellPerSynapse);
				}
				if (numReadPulseBP > 1) {
					shiftAddBPInput.Initialize(numAdder, adderBit+numCellPerSynapse, clkFreq, spikingMode, numReadPulseBP);
				}
			}
			
		} else if (conventionalParallel) {
			wlSwitchMatrix.Initialize(ROW_MODE, numRow, resRow, true, false, activityRowRead, activityColWrite, numWriteCellPerOperationMemory, numWriteCellPerOperationNeuro, 1, clkFreq);
			if (numColMuxed>1) {
				mux.Initialize(ceil(numCol/numColMuxed), numColMuxed, resCellAccess/numRow/2, FPGA);       
				muxDecoder.Initialize(REGULAR_ROW, (int)ceil(log2(numColMuxed)), true, false);
			}
			if (SARADC) {
				sarADC.Initialize(numCol/numColMuxed, levelOutput, clkFreq, numReadCellPerOperationNeuro);
			} else {
				multilevelSenseAmp.Initialize(numCol/numColMuxed, levelOutput, clkFreq, numReadCellPerOperationNeuro, true, currentMode);
				multilevelSAEncoder.Initialize(levelOutput, numCol/numColMuxed);
			}
			if (numCellPerSynapse > 1) {
				shiftAddWeight.Initialize(ceil(numCol/numColMuxed), log2(levelOutput), clkFreq, spikingMode, numCellPerSynapse);
			}
			if (numReadPulse > 1) {
				shiftAddInput.Initialize(ceil(numCol/numColMuxed), log2(levelOutput)+numCellPerSynapse, clkFreq, spikingMode, numReadPulse);
			}
			
			if (trainingEstimation) {
				wlSwitchMatrixBP.Initialize(ROW_MODE, numCol, resCol, true, false, activityRowRead, activityColWrite, numWriteCellPerOperationMemory, numWriteCellPerOperationNeuro, 1, clkFreq);
				if (parallelBP) {
					if (numRowMuxedBP>1) {
						muxBP.Initialize(ceil(numRow/numRowMuxedBP), numRowMuxedBP, resCellAccess/numCol/2, FPGA);       
						muxDecoderBP.Initialize(REGULAR_ROW, (int)ceil(log2(numRowMuxedBP)), true, false);
					}
					if (SARADC) {
						sarADCBP.Initialize(numRow/numRowMuxedBP, levelOutputBP, clkFreq, numReadCellPerOperationNeuro);
					} else {
						multilevelSenseAmpBP.Initialize(numRow/numRowMuxedBP, levelOutputBP, clkFreq, numReadCellPerOperationNeuro, true, currentMode);
						multilevelSAEncoderBP.Initialize(levelOutputBP, numRow/numRowMuxedBP);
					}
					if (numCellPerSynapse > 1) {
						shiftAddBPWeight.Initialize(ceil(numRow/numRowMuxedBP), log2(levelOutputBP), clkFreq, spikingMode, numCellPerSynapse);
					}
					if (numReadPulseBP > 1) {
						shiftAddBPInput.Initialize(ceil(numRow/numRowMuxedBP), log2(levelOutputBP)+numCellPerSynapse, clkFreq, spikingMode, numReadPulseBP);
					}
				} else {
					senseAmpBP.Initialize(numCol, false, cell.minSenseVoltage, lengthCol/numRow, clkFreq, numReadCellPerOperationNeuro);
					int adderBit = (int)ceil(log2(numCol)) + 1;	
					int numAdder = numRow/numCellPerSynapse;
					dffBP.Initialize((adderBit+1)*numAdder, clkFreq); 
					adderBP.Initialize(adderBit, numAdder);
					if (numCellPerSynapse > 1) {
						shiftAddBPWeight.Initialize(numAdder, adderBit, clkFreq, spikingMode, numCellPerSynapse);
					}
					if (numReadPulseBP > 1) {
						shiftAddBPInput.Initialize(numAdder, adderBit+numCellPerSynapse, clkFreq, spikingMode, numReadPulseBP);
					}
				}
			}
			
		} else if (BNNsequentialMode || XNORsequentialMode) {
			wlDecoder.Initialize(REGULAR_ROW, (int)ceil(log2(numRow)), false, false);
			senseAmp.Initialize(numCol, false, cell.minSenseVoltage, lengthRow/numCol, clkFreq, numReadCellPerOperationNeuro);
			int adderBit = (int)ceil(log2(numRow)) + avgWeightBit;	
			int numAdder = numCol/numCellPerSynapse;
			dff.Initialize((adderBit+1)*numAdder, clkFreq);	
			adder.Initialize(adderBit, numAdder);
		} else if (BNNparallelMode || XNORparallelMode) {
			wlSwitchMatrix.Initialize(ROW_MODE, numRow, resRow, true, false, activityRowRead, activityColWrite, numWriteCellPerOperationMemory, numWriteCellPerOperationNeuro, 1, clkFreq);
			if (SARADC) {
				sarADC.Initialize(numCol/numColMuxed, levelOutput, clkFreq, numReadCellPerOperationNeuro);
			} else {
				multilevelSenseAmp.Initialize(numCol/numColMuxed, levelOutput, clkFreq, numReadCellPerOperationNeuro, true, currentMode);
				multilevelSAEncoder.Initialize(levelOutput, numCol/numColMuxed);
			}
		} else {
			wlSwitchMatrix.Initialize(ROW_MODE, numRow, resRow, true, false, activityRowRead, activityColWrite, numWriteCellPerOperationMemory, numWriteCellPerOperationNeuro, 1, clkFreq);
			if (SARADC) {
				sarADC.Initialize(numCol/numColMuxed, levelOutput, clkFreq, numReadCellPerOperationNeuro);
			} else {
				multilevelSenseAmp.Initialize(numCol/numColMuxed, levelOutput, clkFreq, numReadCellPerOperationNeuro, true, currentMode);
				multilevelSAEncoder.Initialize(levelOutput, numCol/numColMuxed);
			}
			if (numCellPerSynapse > 1) {
				shiftAddWeight.Initialize(ceil(numCol/numColMuxed), log2(levelOutput), clkFreq, spikingMode, numCellPerSynapse);
			}
			if (numReadPulse > 1) {
				shiftAddInput.Initialize(ceil(numCol/numColMuxed), log2(levelOutput)+numCellPerSynapse, clkFreq, spikingMode, numReadPulse);
			}
		}
		precharger.Initialize(numCol, resCol, activityColWrite, numReadCellPerOperationNeuro, numWriteCellPerOperationNeuro);
		sramWriteDriver.Initialize(numCol, activityColWrite, numWriteCellPerOperationNeuro);
		prechargerBP.Initialize(numRow, resRow, activityColWrite, numReadCellPerOperationNeuro, numWriteCellPerOperationNeuro);
		sramWriteDriverBP.Initialize(numRow, activityColWrite, numWriteCellPerOperationNeuro);
		
    } else if (cell.memCellType == Type::RRAM || cell.memCellType == Type::FeFET) {
		if (cell.accessType == CMOS_access) {	// 1T1R
			cell.resCellAccess = cell.resistanceOn * IR_DROP_TOLERANCE;    //calculate access CMOS resistance
			cell.widthAccessCMOS = CalculateOnResistance(tech.featureSize, NMOS, inputParameter.temperature, tech) * LINEAR_REGION_RATIO / cell.resCellAccess;   //get access CMOS width
			if (cell.widthAccessCMOS > cell.widthInFeatureSize) {	// Place transistor vertically
				printf("Transistor width of 1T1R=%.2fF is larger than the assigned cell width=%.2fF in layout\n", cell.widthAccessCMOS, cell.widthInFeatureSize);
				exit(-1);
			}

			cell.resMemCellOn = cell.resCellAccess + cell.resistanceOn;        //calculate single memory cell resistance_ON
			cell.resMemCellOff = cell.resCellAccess + cell.resistanceOff;      //calculate single memory cell resistance_OFF
			cell.resMemCellAvg = cell.resCellAccess + cell.resistanceAvg;      //calculate single memory cell resistance_AVG

			capRow2 += CalculateGateCap(cell.widthAccessCMOS * tech.featureSize, tech) * numCol;          //sum up all the gate cap of access CMOS, as the row cap
			capCol += CalculateDrainCap(cell.widthAccessCMOS * tech.featureSize, NMOS, cell.widthInFeatureSize * tech.featureSize, tech) * numRow;	// If capCol is found to be too large, increase cell.widthInFeatureSize to relax the limit
		} else {	// Cross-point
			// The nonlinearity is from the selector, assuming RRAM itself is linear
			if (cell.nonlinearIV) {   //introduce nonlinearity to the RRAM resistance
				cell.resMemCellOn = cell.resistanceOn;
				cell.resMemCellOff = cell.resistanceOff;
				cell.resMemCellOnAtHalfVw = NonlinearResistance(cell.resistanceOn, cell.nonlinearity, cell.writeVoltage, cell.readVoltage, cell.writeVoltage/2);
				cell.resMemCellOffAtHalfVw = NonlinearResistance(cell.resistanceOff, cell.nonlinearity, cell.writeVoltage, cell.readVoltage, cell.writeVoltage/2);
				cell.resMemCellOnAtVw = NonlinearResistance(cell.resistanceOn, cell.nonlinearity, cell.writeVoltage, cell.readVoltage, cell.writeVoltage);
				cell.resMemCellOffAtVw = NonlinearResistance(cell.resistanceOff, cell.nonlinearity, cell.writeVoltage, cell.readVoltage, cell.writeVoltage);
				cell.resMemCellAvg = cell.resistanceAvg;
				cell.resMemCellAvgAtHalfVw = (cell.resMemCellOnAtHalfVw + cell.resMemCellOffAtHalfVw) / 2;
				cell.resMemCellAvgAtVw = (cell.resMemCellOnAtVw + cell.resMemCellOffAtVw) / 2;
			} else {  //simply assume RRAM resistance is linear
				cell.resMemCellOn = cell.resistanceOn;
				cell.resMemCellOff = cell.resistanceOff;
				cell.resMemCellOnAtHalfVw = cell.resistanceOn;
				cell.resMemCellOffAtHalfVw = cell.resistanceOff;
				cell.resMemCellOnAtVw = cell.resistanceOn;
				cell.resMemCellOffAtVw = cell.resistanceOff;
				cell.resMemCellAvg = cell.resistanceAvg;
				cell.resMemCellAvgAtHalfVw = cell.resistanceAvg;
				cell.resMemCellAvgAtVw = cell.resistanceAvg;
			}
		}
		
		if (conventionalSequential) {  
			double capBL = lengthCol * 0.2e-15/1e-6;
			int numAdder = (int)ceil(numCol/numColMuxed);   // numCol is divisible by numCellPerSynapse
			int numInput = numAdder;        //XXX input number of MUX, 
			double resTg = cell.resMemCellOn;     //transmission gate resistance
			int adderBit = (int)ceil(log2(numRow)) + avgWeightBit;  
			
			wlDecoder.Initialize(REGULAR_ROW, (int)ceil(log2(numRow)), false, false);          
			if (cell.accessType == CMOS_access) {
				wlNewDecoderDriver.Initialize(numRow);          
			} else {
				wlDecoderDriver.Initialize(ROW_MODE, numRow, numCol);
			}
			slSwitchMatrix.Initialize(COL_MODE, numCol, resTg, true, false, activityRowRead, activityColWrite, numWriteCellPerOperationMemory, numWriteCellPerOperationNeuro, numWritePulseAVG, clkFreq);     //SL use switch matrix
			if (numColMuxed>1) {
				mux.Initialize(numInput, numColMuxed, resTg, FPGA);     
				muxDecoder.Initialize(REGULAR_ROW, (int)ceil(log2(numColMuxed)), true, false);
			}
			
			if (SARADC) {
				sarADC.Initialize(numCol/numColMuxed, pow(2, avgWeightBit), clkFreq, numReadCellPerOperationNeuro);
			} else {
				multilevelSenseAmp.Initialize(numCol/numColMuxed, pow(2, avgWeightBit), clkFreq, numReadCellPerOperationNeuro, false, currentMode);
				if (avgWeightBit > 1) {
					multilevelSAEncoder.Initialize(pow(2, avgWeightBit), numCol/numColMuxed);
				}
			}

			dff.Initialize((adderBit+1)*numAdder, clkFreq); 
			adder.Initialize(adderBit, numAdder);
			if (numCellPerSynapse > 1) {
				shiftAddWeight.Initialize(numAdder, adderBit, clkFreq, spikingMode, numCellPerSynapse);
			}
			if (numReadPulse > 1) {
				shiftAddInput.Initialize(numAdder, adderBit+numCellPerSynapse, clkFreq, spikingMode, numReadPulse);
			}
			
			/* Transpose Peripheral for BP */
			if (trainingEstimation) {
				if (numRowMuxedBP>1) {
					muxBP.Initialize(ceil(numRow/numRowMuxedBP), numRowMuxedBP, resTg, FPGA);       
					muxDecoderBP.Initialize(REGULAR_ROW, (int)ceil(log2(numRowMuxedBP)), true, false);
				}
				if (SARADC) {
					sarADCBP.Initialize(numRow/numRowMuxedBP, pow(2, avgWeightBit), clkFreq, numReadCellPerOperationNeuro);
				} else {
					multilevelSenseAmpBP.Initialize(numRow/numRowMuxedBP, pow(2, avgWeightBit), clkFreq, numReadCellPerOperationNeuro, false, currentMode);
					if (avgWeightBit > 1) {
						multilevelSAEncoderBP.Initialize(pow(2, avgWeightBit), numRow/numRowMuxedBP);
					}
				}

				dffBP.Initialize((ceil(log2(numCol)) + avgWeightBit+1)*ceil(numRow/numRowMuxedBP), clkFreq); 
				adderBP.Initialize(ceil(log2(numCol)) + avgWeightBit, ceil(numRow/numRowMuxedBP));
				if (numCellPerSynapse > 1) {
					shiftAddBPWeight.Initialize(ceil(numRow/numRowMuxedBP), ceil(log2(numCol))+avgWeightBit, clkFreq, spikingMode, numCellPerSynapse);
				}
				if (numReadPulseBP > 1) {
					shiftAddBPInput.Initialize(ceil(numRow/numRowMuxedBP), ceil(log2(numCol))+avgWeightBit+numCellPerSynapse, clkFreq, spikingMode, numReadPulseBP);
				}
			}
			
		} else if (conventionalParallel) { 
			double resTg = cell.resMemCellOn / numRow;
			
			if (cell.accessType == CMOS_access) {
				wlNewSwitchMatrix.Initialize(numRow, activityRowRead, clkFreq);         
			} else {
				wlSwitchMatrix.Initialize(ROW_MODE, numRow, resTg*numRow/numCol, true, false, activityRowRead, activityColWrite, numWriteCellPerOperationMemory, numWriteCellPerOperationNeuro, numWritePulseAVG, clkFreq);
			}
			slSwitchMatrix.Initialize(COL_MODE, numCol, resTg*numRow, true, false, activityRowRead, activityColWrite, numWriteCellPerOperationMemory, numWriteCellPerOperationNeuro, numWritePulseAVG, clkFreq);     
			if (numColMuxed>1) {
				mux.Initialize(ceil(numCol/numColMuxed), numColMuxed, resTg, FPGA);       
				muxDecoder.Initialize(REGULAR_ROW, (int)ceil(log2(numColMuxed)), true, false);
			}
			
			if (SARADC) {
				sarADC.Initialize(numCol/numColMuxed, levelOutput, clkFreq, numReadCellPerOperationNeuro);
			} else {
				multilevelSenseAmp.Initialize(numCol/numColMuxed, levelOutput, clkFreq, numReadCellPerOperationNeuro, true, currentMode);
				multilevelSAEncoder.Initialize(levelOutput, numCol/numColMuxed);
			}
			if (numCellPerSynapse > 1) {
				shiftAddWeight.Initialize(ceil(numCol/numColMuxed), log2(levelOutput), clkFreq, spikingMode, numCellPerSynapse);
			}
			if (numReadPulse > 1) {
				shiftAddInput.Initialize(ceil(numCol/numColMuxed), log2(levelOutput)+numCellPerSynapse, clkFreq, spikingMode, numReadPulse);
			}
			
			/* Transpose Peripheral for BP */
			if (trainingEstimation) {
				if (numRowMuxedBP>1) {
					muxBP.Initialize(ceil(numRow/numRowMuxedBP), numRowMuxedBP, cell.resMemCellOn/numCol, FPGA);       
					muxDecoderBP.Initialize(REGULAR_ROW, (int)ceil(log2(numRowMuxedBP)), true, false);
				}
				if (parallelBP) {
					if (SARADC) {
						sarADCBP.Initialize(numRow/numRowMuxedBP, levelOutputBP, clkFreq, numReadCellPerOperationNeuro);
					} else {
						multilevelSenseAmpBP.Initialize(numRow/numRowMuxedBP, levelOutputBP, clkFreq, numReadCellPerOperationNeuro, true, currentMode);
						multilevelSAEncoderBP.Initialize(levelOutputBP, numRow/numRowMuxedBP);
					}
					if (numCellPerSynapse > 1) {
						shiftAddBPWeight.Initialize(ceil(numRow/numRowMuxedBP), log2(levelOutputBP), clkFreq, spikingMode, numCellPerSynapse);
					}
					if (numReadPulseBP > 1) {
						shiftAddBPInput.Initialize(ceil(numRow/numRowMuxedBP), log2(levelOutputBP)+numCellPerSynapse, clkFreq, spikingMode, numReadPulseBP);
					}
				} else {
					
					if (SARADC) {
						sarADCBP.Initialize(numRow/numRowMuxedBP, pow(2, avgWeightBit), clkFreq, numReadCellPerOperationNeuro);
					} else {
						multilevelSenseAmpBP.Initialize(numRow/numRowMuxedBP, pow(2, avgWeightBit), clkFreq, numReadCellPerOperationNeuro, false, currentMode);
						if (avgWeightBit > 1) {
							multilevelSAEncoderBP.Initialize(pow(2, avgWeightBit), numRow/numRowMuxedBP);
						}
					}

					dffBP.Initialize((ceil(log2(numCol)) + avgWeightBit+1)*ceil(numRow/numRowMuxedBP), clkFreq); 
					adderBP.Initialize(ceil(log2(numCol)) + avgWeightBit, ceil(numRow/numRowMuxedBP));
					if (numCellPerSynapse > 1) {
						shiftAddBPWeight.Initialize(ceil(numRow/numRowMuxedBP), ceil(log2(numCol))+avgWeightBit, clkFreq, spikingMode, numCellPerSynapse);
					}
					if (numReadPulseBP > 1) {
						shiftAddBPInput.Initialize(ceil(numRow/numRowMuxedBP), ceil(log2(numCol))+avgWeightBit+numCellPerSynapse, clkFreq, spikingMode, numReadPulseBP);
					}
				}
			}
			
		} else if (BNNsequentialMode || XNORsequentialMode) {       
			double resTg = cell.resMemCellOn;
			int numAdder = (int)ceil(numCol/numColMuxed);  
			int numInput = numAdder;        
			int adderBit = (int)ceil(log2(numRow)) + 1; 
			
			wlDecoder.Initialize(REGULAR_ROW, (int)ceil(log2(numRow)), false, false);           
			if (cell.accessType == CMOS_access) {
				wlNewDecoderDriver.Initialize(numRow);          
			} else {
				wlDecoderDriver.Initialize(ROW_MODE, numRow, numCol);
			}
			slSwitchMatrix.Initialize(COL_MODE, numCol, resTg, true, false, activityRowRead, activityColWrite, numWriteCellPerOperationMemory, numWriteCellPerOperationNeuro, numWritePulseAVG, clkFreq);     //SL use switch matrix
			if (numColMuxed>1) {
				mux.Initialize(numInput, numColMuxed, resTg, FPGA);      
				muxDecoder.Initialize(REGULAR_ROW, (int)ceil(log2(numColMuxed)), true, false);
			}
			rowCurrentSenseAmp.Initialize(numCol/numColMuxed, true, false, clkFreq, numReadCellPerOperationNeuro);
			dff.Initialize((adderBit+1)*numAdder, clkFreq); 
			adder.Initialize(adderBit, numAdder);
		} else if (BNNparallelMode || XNORparallelMode) {      
			double resTg = cell.resMemCellOn / numRow;
			
			if (cell.accessType == CMOS_access) {
				wlNewSwitchMatrix.Initialize(numRow, activityRowRead, clkFreq);         
			} else {
				wlSwitchMatrix.Initialize(ROW_MODE, numRow, resTg*numRow/numCol, true, false, activityRowRead, activityColWrite, numWriteCellPerOperationMemory, numWriteCellPerOperationNeuro, numWritePulseAVG, clkFreq);
			}
			slSwitchMatrix.Initialize(COL_MODE, numCol, resTg * numRow, true, false, activityRowRead, activityColWrite, numWriteCellPerOperationMemory, numWriteCellPerOperationNeuro, numWritePulseAVG, clkFreq);     
			if (numColMuxed>1) {
				mux.Initialize(ceil(numCol/numColMuxed), numColMuxed, resTg, FPGA);       
				muxDecoder.Initialize(REGULAR_ROW, (int)ceil(log2(numColMuxed/2)), true, true);    
			}  
			if (SARADC) {
				sarADC.Initialize(numCol/numColMuxed, levelOutput, clkFreq, numReadCellPerOperationNeuro);
			} else {
				multilevelSenseAmp.Initialize(numCol/numColMuxed, levelOutput, clkFreq, numReadCellPerOperationNeuro, true, currentMode);
				multilevelSAEncoder.Initialize(levelOutput, numCol/numColMuxed);
			}
		} else {
			double resTg = cell.resMemCellOn / numRow;
			
			if (cell.accessType == CMOS_access) {
				wlNewSwitchMatrix.Initialize(numRow, activityRowRead, clkFreq);         
			} else {
				wlSwitchMatrix.Initialize(ROW_MODE, numRow, resTg*numRow/numCol, true, false, activityRowRead, activityColWrite, numWriteCellPerOperationMemory, numWriteCellPerOperationNeuro, numWritePulseAVG, clkFreq);
			}
			slSwitchMatrix.Initialize(COL_MODE, numCol, resTg*numRow, true, false, activityRowRead, activityColWrite, numWriteCellPerOperationMemory, numWriteCellPerOperationNeuro, numWritePulseAVG, clkFreq);     
			if (numColMuxed>1) {
				mux.Initialize(ceil(numCol/numColMuxed), numColMuxed, resTg, FPGA);      
				muxDecoder.Initialize(REGULAR_ROW, (int)ceil(log2(numColMuxed)), true, false);
			}
			if (SARADC) {
				sarADC.Initialize(numCol/numColMuxed, levelOutput, clkFreq, numReadCellPerOperationNeuro);
			} else {
				multilevelSenseAmp.Initialize(numCol/numColMuxed, levelOutput, clkFreq, numReadCellPerOperationNeuro, true, currentMode);
				multilevelSAEncoder.Initialize(levelOutput, numCol/numColMuxed);
			}
			if (numCellPerSynapse > 1) {
				shiftAddWeight.Initialize(ceil(numCol/numColMuxed), log2(levelOutput), clkFreq, spikingMode, numCellPerSynapse);
			}
			if (numReadPulse > 1) {
				shiftAddInput.Initialize(ceil(numCol/numColMuxed), log2(levelOutput)+numCellPerSynapse, clkFreq, spikingMode, numReadPulse);
			}
		}
	} 
	initialized = true;  //finish initialization
}



void SubArray::CalculateArea() {  //calculate layout area for total design
	if (!initialized) {
		cout << "[Subarray] Error: Require initialization first!" << endl;  //ensure initialization first
	} else {  //if initialized, start to do calculation
		area = 0;
		usedArea = 0;
		if (cell.memCellType == Type::SRAM) {       
			// Array only
			heightArray = lengthCol;
			widthArray = lengthRow;
			areaArray = heightArray * widthArray;
			
			//precharger and writeDriver are always needed for all different designs
			precharger.CalculateArea(NULL, widthArray, NONE);
			sramWriteDriver.CalculateArea(NULL, widthArray, NONE);
			prechargerBP.CalculateArea(heightArray, NULL, NONE);
			sramWriteDriverBP.CalculateArea(heightArray, NULL, NONE);
			
			if (conventionalSequential) {
				wlDecoder.CalculateArea(heightArray, NULL, NONE);  
				senseAmp.CalculateArea(NULL, widthArray, MAGIC);
				adder.CalculateArea(NULL, widthArray, NONE);
				dff.CalculateArea(NULL, widthArray, NONE);
				if (numReadPulse > 1) {
					shiftAddInput.CalculateArea(NULL, widthArray, NONE);
				}
				if (numCellPerSynapse > 1) {
					shiftAddWeight.CalculateArea(NULL, widthArray, NONE);
				}
				height = precharger.height + sramWriteDriver.height + heightArray + senseAmp.height + adder.height + dff.height + shiftAddInput.height + shiftAddWeight.height;
				width = wlDecoder.width + widthArray;
				usedArea = areaArray + wlDecoder.area + precharger.area + sramWriteDriver.area + senseAmp.area + adder.area + dff.area + shiftAddInput.area + shiftAddWeight.area;
				
				areaADC = senseAmp.area + precharger.area;
				areaAccum = adder.area + dff.area + shiftAddInput.area + shiftAddWeight.area;
				areaOther = wlDecoder.area + sramWriteDriver.area;
				
				/* Transpose Peripheral for BP */
				if (trainingEstimation) {
					wlDecoderBP.CalculateArea(NULL, widthArray, NONE);
					senseAmpBP.CalculateArea(heightArray, NULL, NONE);
					dffBP.CalculateArea(heightArray, NULL, NONE);
					adderBP.CalculateArea(heightArray, NULL, NONE);
					
					if (numReadPulseBP > 1) {
						shiftAddBPInput.CalculateArea(heightArray, NULL, NONE);
					}
					if (numCellPerSynapse > 1) {
						shiftAddBPWeight.CalculateArea(heightArray, NULL, NONE);
					}
					height += wlDecoderBP.height;
					width += senseAmpBP.width + dffBP.width + adderBP.width + shiftAddBPInput.width + shiftAddBPWeight.width + prechargerBP.width + sramWriteDriverBP.width;
					areaAG = senseAmpBP.area + dffBP.area + adderBP.area + shiftAddBPInput.area + shiftAddBPWeight.area + wlDecoderBP.area + prechargerBP.area + sramWriteDriverBP.area;
					usedArea += areaAG;
					areaADC += senseAmpBP.area + prechargerBP.area;
					areaAccum += dffBP.area + adderBP.area + shiftAddBPInput.area + shiftAddBPWeight.area;
					areaOther += muxBP.area + muxDecoderBP.area + wlDecoderBP.area+ sramWriteDriverBP.area;
					
				}
				area = height * width;
				emptyArea = area - usedArea;
				
			} else if (conventionalParallel) { 
				wlSwitchMatrix.CalculateArea(heightArray, NULL, NONE);
				if (numColMuxed>1) {
					mux.CalculateArea(NULL, widthArray, NONE);
					muxDecoder.CalculateArea(NULL, NULL, NONE);
					double minMuxHeight = MAX(muxDecoder.height, mux.height);
					mux.CalculateArea(minMuxHeight, widthArray, OVERRIDE);
				}
				if (SARADC) {
					sarADC.CalculateUnitArea();
					sarADC.CalculateArea(NULL, widthArray, NONE);
				} else {
					multilevelSenseAmp.CalculateArea(NULL, widthArray, NONE);
					multilevelSAEncoder.CalculateArea(NULL, widthArray, NONE);
				}
				
				if (numReadPulse > 1) {
					shiftAddInput.CalculateArea(NULL, widthArray, NONE);
				}
				if (numCellPerSynapse > 1) {
					shiftAddWeight.CalculateArea(NULL, widthArray, NONE);
				}
				height = precharger.height + sramWriteDriver.height + heightArray + multilevelSenseAmp.height + multilevelSAEncoder.height + shiftAddInput.height + shiftAddWeight.height + mux.height + sarADC.height;
				width = MAX(wlSwitchMatrix.width, muxDecoder.width) + widthArray;
				usedArea = areaArray + wlSwitchMatrix.area + precharger.area + sramWriteDriver.area + multilevelSenseAmp.area + multilevelSAEncoder.area + shiftAddInput.area + shiftAddWeight.area + mux.area + muxDecoder.area + sarADC.area;
				
				areaADC = multilevelSenseAmp.area + precharger.area + multilevelSAEncoder.area + sarADC.area;
				areaAccum = shiftAddInput.area + shiftAddWeight.area;
				areaOther = wlSwitchMatrix.area + sramWriteDriver.area + mux.area + muxDecoder.area;
				
				/* Transpose Peripheral for BP */
				if (trainingEstimation) {
					wlSwitchMatrixBP.CalculateArea(NULL, widthArray, NONE);
					
					if (parallelBP) {
						if (numRowMuxedBP>1) {
							muxBP.CalculateArea(heightArray, NULL, NONE);
							muxDecoderBP.CalculateArea(NULL, NULL, NONE);
							double minMuxWidth = MAX(muxDecoderBP.width, muxBP.width);
							muxBP.CalculateArea(heightArray, minMuxWidth, OVERRIDE);
						}
					
						if (SARADC) {
							sarADCBP.CalculateUnitArea();
							sarADCBP.CalculateArea(heightArray, NULL, NONE);
						} else {
							multilevelSenseAmpBP.CalculateArea(heightArray, NULL, NONE);
							multilevelSAEncoderBP.CalculateArea(heightArray, NULL, NONE);
						}
						
						if (numReadPulseBP > 1) {
							shiftAddBPInput.CalculateArea(heightArray, NULL, NONE);
						}
						if (numCellPerSynapse > 1) {
							shiftAddBPWeight.CalculateArea(heightArray, NULL, NONE);
						}
						height += wlSwitchMatrixBP.height;
						width += muxBP.width + multilevelSenseAmpBP.width + multilevelSAEncoderBP.width + shiftAddBPInput.width + shiftAddBPWeight.width + prechargerBP.width + sramWriteDriverBP.width + sarADCBP.width;
						areaAG = muxBP.area + muxDecoderBP.area + multilevelSenseAmpBP.area + multilevelSAEncoderBP.area + shiftAddBPInput.area + shiftAddBPWeight.area + wlSwitchMatrixBP.area + prechargerBP.area + sramWriteDriverBP.area + sarADCBP.area;
						usedArea += areaAG;
						areaADC += multilevelSenseAmpBP.area + multilevelSAEncoderBP.area + prechargerBP.area + sarADCBP.area;
						areaAccum += shiftAddBPInput.area + shiftAddBPWeight.area;
						areaOther += muxBP.area + muxDecoderBP.area + wlSwitchMatrixBP.area + sramWriteDriverBP.area;
						
					} else {
						senseAmpBP.CalculateArea(heightArray, NULL, NONE);
						dffBP.CalculateArea(heightArray, NULL, NONE);
						adderBP.CalculateArea(heightArray, NULL, NONE);
						
						if (numReadPulseBP > 1) {
							shiftAddBPInput.CalculateArea(heightArray, NULL, NONE);
						}
						if (numCellPerSynapse > 1) {
							shiftAddBPWeight.CalculateArea(heightArray, NULL, NONE);
						}
						height += wlSwitchMatrixBP.height;
						width += senseAmpBP.width + dffBP.width + adderBP.width + shiftAddBPInput.width + shiftAddBPWeight.width + prechargerBP.width + sramWriteDriverBP.width;
						areaAG = senseAmpBP.area + dffBP.area + adderBP.area + shiftAddBPInput.area + shiftAddBPWeight.area + wlSwitchMatrixBP.area + prechargerBP.area + sramWriteDriverBP.area;
						usedArea += areaAG;
						areaADC += senseAmpBP.area + prechargerBP.area;
						areaAccum += dffBP.area + adderBP.area + shiftAddBPInput.area + shiftAddBPWeight.area;
						areaOther += wlSwitchMatrixBP.area + sramWriteDriverBP.area;
					}
				}
				
				area = height * width;
				emptyArea = area - usedArea;
				
			} else if (BNNsequentialMode || XNORsequentialMode) {
				wlDecoder.CalculateArea(heightArray, NULL, NONE);  
				senseAmp.CalculateArea(NULL, widthArray, MAGIC);
				adder.CalculateArea(NULL, widthArray, NONE);
				dff.CalculateArea(NULL, widthArray, NONE);
				height = precharger.height + sramWriteDriver.height + heightArray + senseAmp.height + adder.height + dff.height;
				width = wlDecoder.width + widthArray;
				area = height * width;
				usedArea = areaArray + wlDecoder.area + precharger.area + sramWriteDriver.area + senseAmp.area + adder.area + dff.area;
				emptyArea = area - usedArea;
			} else if (BNNparallelMode || XNORparallelMode) {
				wlSwitchMatrix.CalculateArea(heightArray, NULL, NONE);
				if (SARADC) {
					sarADC.CalculateUnitArea();
					sarADC.CalculateArea(NULL, widthArray, NONE);
				} else {
					multilevelSenseAmp.CalculateArea(NULL, widthArray, NONE);
					multilevelSAEncoder.CalculateArea(NULL, widthArray, NONE);
				}
				height = precharger.height + sramWriteDriver.height + heightArray + multilevelSenseAmp.height + multilevelSAEncoder.height + sarADC.height;
				width = wlSwitchMatrix.width + widthArray;
				area = height * width;
				usedArea = areaArray + wlSwitchMatrix.area + precharger.area + sramWriteDriver.area + multilevelSenseAmp.area + multilevelSAEncoder.area + sarADC.area;
				emptyArea = area - usedArea;
			} else {
				wlSwitchMatrix.CalculateArea(heightArray, NULL, NONE);
				if (SARADC) {
					sarADC.CalculateUnitArea();
					sarADC.CalculateArea(NULL, widthArray, NONE);
				} else {
					multilevelSenseAmp.CalculateArea(NULL, widthArray, NONE);
					multilevelSAEncoder.CalculateArea(NULL, widthArray, NONE);
				}
				if (numReadPulse > 1) {
					shiftAddInput.CalculateArea(NULL, widthArray, NONE);
				}
				if (numCellPerSynapse > 1) {
					shiftAddWeight.CalculateArea(NULL, widthArray, NONE);
				}
				height = precharger.height + sramWriteDriver.height + heightArray + multilevelSenseAmp.height + multilevelSAEncoder.height + shiftAddInput.height + shiftAddWeight.height + sarADC.height;
				width = wlSwitchMatrix.width + widthArray;
				area = height * width;
				usedArea = areaArray + wlSwitchMatrix.area + precharger.area + sramWriteDriver.area + multilevelSenseAmp.area + multilevelSAEncoder.area + shiftAddInput.area + shiftAddWeight.area + sarADC.area;
				emptyArea = area - usedArea;
			}
	    } else if (cell.memCellType == Type::RRAM || cell.memCellType == Type::FeFET) {
			// Array only
			heightArray = lengthCol;
			widthArray = lengthRow;
			areaArray = heightArray * widthArray;
			
			if (conventionalSequential) {  
				wlDecoder.CalculateArea(heightArray, NULL, NONE);
				if (cell.accessType == CMOS_access) {
					wlNewDecoderDriver.CalculateArea(heightArray, NULL, NONE);
				} else {
					wlDecoderDriver.CalculateArea(heightArray, NULL, NONE);
				}
				slSwitchMatrix.CalculateArea(NULL, widthArray, NONE);
				
				if (numColMuxed > 1) {
					mux.CalculateArea(NULL, widthArray, NONE);
					muxDecoder.CalculateArea(NULL, NULL, NONE);
					double minMuxHeight = MAX(muxDecoder.height, mux.height);
					mux.CalculateArea(minMuxHeight, widthArray, OVERRIDE);
				}
				if (SARADC) {
					sarADC.CalculateUnitArea();
					sarADC.CalculateArea(NULL, widthArray, NONE);
				} else {
					multilevelSenseAmp.CalculateArea(NULL, widthArray, NONE);
					if (avgWeightBit > 1) {
						multilevelSAEncoder.CalculateArea(NULL, widthArray, NONE);
					}
				}

				dff.CalculateArea(NULL, widthArray, NONE);
				adder.CalculateArea(NULL, widthArray, NONE);
				if (numReadPulse > 1) {
					shiftAddInput.CalculateArea(NULL, widthArray, NONE);
				}
				if (numCellPerSynapse > 1) {
					shiftAddWeight.CalculateArea(NULL, widthArray, NONE);
				}
				height = slSwitchMatrix.height + heightArray + mux.height + multilevelSenseAmp.height + multilevelSAEncoder.height + adder.height + dff.height + shiftAddInput.height + shiftAddWeight.height + sarADC.height;
				width = MAX(wlDecoder.width + wlNewDecoderDriver.width + wlDecoderDriver.width, muxDecoder.width) + widthArray;
				usedArea = areaArray + wlDecoder.area + wlDecoderDriver.area + wlNewDecoderDriver.area + slSwitchMatrix.area + mux.area + multilevelSenseAmp.area + multilevelSAEncoder.area + muxDecoder.area + adder.area + dff.area + shiftAddInput.area + shiftAddWeight.area + sarADC.area;
				
				areaADC = multilevelSenseAmp.area + multilevelSAEncoder.area + sarADC.area;
				areaAccum = adder.area + dff.area + shiftAddInput.area + shiftAddWeight.area;
				areaOther = wlDecoder.area + wlNewDecoderDriver.area + wlDecoderDriver.area + slSwitchMatrix.area + mux.area + muxDecoder.area;
				
				/* Transpose Peripheral for BP */
				if (trainingEstimation) {
					if (numRowMuxedBP>1) {
						muxBP.CalculateArea(heightArray, NULL, NONE);
						muxDecoderBP.CalculateArea(NULL, NULL, NONE);
						double minMuxWidth = MAX(muxDecoderBP.width, muxBP.width);
						muxBP.CalculateArea(heightArray, minMuxWidth, OVERRIDE);
					}
					if (SARADC) {
						sarADCBP.CalculateUnitArea();
						sarADCBP.CalculateArea(heightArray, NULL, NONE);
					} else {
						multilevelSenseAmpBP.CalculateArea(heightArray, NULL, NONE);
						if (avgWeightBit > 1) {
							multilevelSAEncoderBP.CalculateArea(heightArray, NULL, NONE);
						}
					}

					dffBP.CalculateArea(heightArray, NULL, NONE);
					adderBP.CalculateArea(heightArray, NULL, NONE);
					if (numReadPulseBP > 1) {
						shiftAddBPInput.CalculateArea(heightArray, NULL, NONE);
					}
					if (numCellPerSynapse > 1) {
						shiftAddBPWeight.CalculateArea(heightArray, NULL, NONE);
					}
					width += muxBP.width + multilevelSenseAmpBP.width + multilevelSAEncoderBP.width + dffBP.width + adderBP.width + shiftAddBPInput.width + shiftAddBPWeight.width + sarADCBP.width;
					areaAG = muxBP.area + muxDecoderBP.area + multilevelSenseAmpBP.area + multilevelSAEncoderBP.area + dffBP.area + adderBP.area + shiftAddBPInput.area + shiftAddBPWeight.area + sarADCBP.area;
					usedArea += areaAG;
					areaADC += multilevelSenseAmpBP.area + multilevelSAEncoderBP.area + sarADCBP.area;
					areaAccum += dffBP.area + adderBP.area + shiftAddBPInput.area + shiftAddBPWeight.area;
					areaOther += muxBP.area + muxDecoderBP.area;
					
				}
				
				area = height * width;
				emptyArea = area - usedArea;
				
			} else if (conventionalParallel) { 
				if (cell.accessType == CMOS_access) {
					wlNewSwitchMatrix.CalculateArea(heightArray, NULL, NONE);
				} else {
					wlSwitchMatrix.CalculateArea(heightArray, NULL, NONE);
				}
				slSwitchMatrix.CalculateArea(NULL, widthArray, NONE);
				if (numColMuxed > 1) {
					mux.CalculateArea(NULL, widthArray, NONE);
					muxDecoder.CalculateArea(NULL, NULL, NONE);
					double minMuxHeight = MAX(muxDecoder.height, mux.height);
					mux.CalculateArea(minMuxHeight, widthArray, OVERRIDE);
				}
				if (SARADC) {
					sarADC.CalculateUnitArea();
					sarADC.CalculateArea(NULL, widthArray, NONE);
				} else {
					multilevelSenseAmp.CalculateArea(NULL, widthArray, NONE);
					multilevelSAEncoder.CalculateArea(NULL, widthArray, NONE);
				}
				
				if (numReadPulse > 1) {
					shiftAddInput.CalculateArea(NULL, widthArray, NONE);
				}
				if (numCellPerSynapse > 1) {
					shiftAddWeight.CalculateArea(NULL, widthArray, NONE);
				}
				
				height = slSwitchMatrix.height + heightArray + mux.height + multilevelSenseAmp.height + multilevelSAEncoder.height + shiftAddInput.height + shiftAddWeight.height + sarADC.height;
				width = MAX(wlNewSwitchMatrix.width + wlSwitchMatrix.width, muxDecoder.width) + widthArray;
				usedArea = areaArray + wlSwitchMatrix.area + wlNewSwitchMatrix.area + slSwitchMatrix.area + mux.area + multilevelSenseAmp.area + muxDecoder.area + multilevelSAEncoder.area + shiftAddInput.area + shiftAddWeight.area + sarADC.area;
				
				areaADC = multilevelSenseAmp.area + multilevelSAEncoder.area + sarADC.area;
				areaAccum = shiftAddInput.area + shiftAddWeight.area;
				areaOther = wlNewSwitchMatrix.area + wlSwitchMatrix.area + slSwitchMatrix.area + mux.area + muxDecoder.area;
				
				/* Transpose Peripheral for BP */
				if (trainingEstimation) {
					if (numRowMuxedBP>1) {
						muxBP.CalculateArea(heightArray, NULL, NONE);
						muxDecoderBP.CalculateArea(NULL, NULL, NONE);
						double minMuxWidth = MAX(muxDecoderBP.width, muxBP.width);
						muxBP.CalculateArea(heightArray, minMuxWidth, OVERRIDE);
					}
					
					if (parallelBP) {
						if (SARADC) {
							sarADCBP.CalculateUnitArea();
							sarADCBP.CalculateArea(heightArray, NULL, NONE);
						} else {
							multilevelSenseAmpBP.CalculateArea(heightArray, NULL, NONE);
							multilevelSAEncoderBP.CalculateArea(heightArray, NULL, NONE);
						}

						if (numReadPulseBP > 1) {
							shiftAddBPInput.CalculateArea(heightArray, NULL, NONE);
						}
						if (numCellPerSynapse > 1) {
							shiftAddBPWeight.CalculateArea(heightArray, NULL, NONE);
						}
						width += muxBP.width + multilevelSenseAmpBP.width + multilevelSAEncoderBP.width + shiftAddBPInput.width + shiftAddBPWeight.width + sarADCBP.width;
						areaAG = muxBP.area + muxDecoderBP.area + multilevelSenseAmpBP.area + multilevelSAEncoderBP.area + shiftAddBPInput.area + shiftAddBPWeight.area + sarADCBP.area;
						usedArea += areaAG;
						areaADC += multilevelSenseAmpBP.area + multilevelSAEncoderBP.area + sarADCBP.area;
						areaAccum += shiftAddBPInput.area + shiftAddBPWeight.area;
						areaOther += muxBP.area + muxDecoderBP.area;
					} else {
						if (SARADC) {
							sarADCBP.CalculateUnitArea();
							sarADCBP.CalculateArea(heightArray, NULL, NONE);
						} else {
							multilevelSenseAmpBP.CalculateArea(heightArray, NULL, NONE);
							if (avgWeightBit > 1) {
								multilevelSAEncoderBP.CalculateArea(heightArray, NULL, NONE);
							}
						}

						dffBP.CalculateArea(heightArray, NULL, NONE);
						adderBP.CalculateArea(heightArray, NULL, NONE);
						
						if (numReadPulseBP > 1) {
							shiftAddBPInput.CalculateArea(heightArray, NULL, NONE);
						}
						if (numCellPerSynapse > 1) {
							shiftAddBPWeight.CalculateArea(heightArray, NULL, NONE);
						}
						width += muxBP.width + multilevelSenseAmpBP.width + multilevelSAEncoderBP.width + dffBP.width + adderBP.width + shiftAddBPInput.width + shiftAddBPWeight.width + sarADCBP.width;
						areaAG = muxBP.area + muxDecoderBP.area + multilevelSenseAmpBP.area + multilevelSAEncoderBP.area + dffBP.area + adderBP.area + shiftAddBPInput.area + shiftAddBPWeight.area + sarADCBP.area;
						usedArea += areaAG;
						areaADC += multilevelSenseAmpBP.area + multilevelSAEncoderBP.area + sarADCBP.area;
						areaAccum += dffBP.area + adderBP.area + shiftAddBPInput.area + shiftAddBPWeight.area;
						areaOther += muxBP.area + muxDecoderBP.area;
					}
				}
				area = height * width;
				emptyArea = area - usedArea;
			} else if (BNNsequentialMode || XNORsequentialMode) {       
				wlDecoder.CalculateArea(heightArray, NULL, NONE);
				if (cell.accessType == CMOS_access) {
					wlNewDecoderDriver.CalculateArea(heightArray, NULL, NONE);
				} else {
					wlDecoderDriver.CalculateArea(heightArray, NULL, NONE);
				}
				slSwitchMatrix.CalculateArea(NULL, widthArray, NONE);
				if (numColMuxed > 1) {
					mux.CalculateArea(NULL, widthArray, NONE);
					muxDecoder.CalculateArea(NULL, NULL, NONE);
					double minMuxHeight = MAX(muxDecoder.height, mux.height);
					mux.CalculateArea(minMuxHeight, widthArray, OVERRIDE);
				}
				rowCurrentSenseAmp.CalculateUnitArea();
				rowCurrentSenseAmp.CalculateArea(widthArray);
				
				dff.CalculateArea(NULL, widthArray, NONE);
				adder.CalculateArea(NULL, widthArray, NONE);
				
				height = slSwitchMatrix.height + heightArray + mux.height + rowCurrentSenseAmp.height + adder.height + dff.height;
				width = MAX(wlDecoder.width + wlNewDecoderDriver.width + wlDecoderDriver.width, muxDecoder.width) + widthArray;
				area = height * width;
				usedArea = areaArray + wlDecoder.area + wlDecoderDriver.area + wlNewDecoderDriver.area + slSwitchMatrix.area + mux.area + rowCurrentSenseAmp.area + muxDecoder.area + adder.area + dff.area;
				emptyArea = area - usedArea;
			} else if (BNNparallelMode || XNORparallelMode) {      
				if (cell.accessType == CMOS_access) {
					wlNewSwitchMatrix.CalculateArea(heightArray, NULL, NONE);
				} else {
					wlSwitchMatrix.CalculateArea(heightArray, NULL, NONE);
				}
				slSwitchMatrix.CalculateArea(NULL, widthArray, NONE);
				if (numColMuxed > 1) {
					mux.CalculateArea(NULL, widthArray, NONE);
					muxDecoder.CalculateArea(NULL, NULL, NONE);
					double minMuxHeight = MAX(muxDecoder.height, mux.height);
					mux.CalculateArea(minMuxHeight, widthArray, OVERRIDE);
				}
				if (SARADC) {
					sarADC.CalculateUnitArea();
					sarADC.CalculateArea(NULL, widthArray, NONE);
				} else {
					multilevelSenseAmp.CalculateArea(NULL, widthArray, NONE);
					multilevelSAEncoder.CalculateArea(NULL, widthArray, NONE);
				}
				
				height = slSwitchMatrix.height + heightArray + mux.height + multilevelSenseAmp.height + multilevelSAEncoder.height + sarADC.height;
				width = MAX(wlNewSwitchMatrix.width + wlSwitchMatrix.width, muxDecoder.width) + widthArray;
				area = height * width;
				usedArea = areaArray + wlSwitchMatrix.area + wlNewSwitchMatrix.area + slSwitchMatrix.area + mux.area + multilevelSenseAmp.area + muxDecoder.area + multilevelSAEncoder.area + sarADC.area;
				emptyArea = area - usedArea;
			} else {   
				if (cell.accessType == CMOS_access) {
					wlNewSwitchMatrix.CalculateArea(heightArray, NULL, NONE);
				} else {
					wlSwitchMatrix.CalculateArea(heightArray, NULL, NONE);
				}
				slSwitchMatrix.CalculateArea(NULL, widthArray, NONE);
				if (numColMuxed > 1) {
					mux.CalculateArea(NULL, widthArray, NONE);
					muxDecoder.CalculateArea(NULL, NULL, NONE);
					double minMuxHeight = MAX(muxDecoder.height, mux.height);
					mux.CalculateArea(minMuxHeight, widthArray, OVERRIDE);
				}
				if (SARADC) {
					sarADC.CalculateUnitArea();
					sarADC.CalculateArea(NULL, widthArray, NONE);
				} else {
					multilevelSenseAmp.CalculateArea(NULL, widthArray, NONE);
					multilevelSAEncoder.CalculateArea(NULL, widthArray, NONE);
				}
				
				height = slSwitchMatrix.height + heightArray + mux.height + multilevelSenseAmp.height + multilevelSAEncoder.height + sarADC.height;
				width = MAX(wlNewSwitchMatrix.width + wlSwitchMatrix.width, muxDecoder.width) + widthArray;
				area = height * width;
				usedArea = areaArray + wlSwitchMatrix.area + wlNewSwitchMatrix.area + slSwitchMatrix.area + mux.area + multilevelSenseAmp.area + muxDecoder.area + multilevelSAEncoder.area + sarADC.area;
				emptyArea = area - usedArea;
			}
			
		} 
	}
}

void SubArray::CalculateLatency(double columnRes, const vector<double> &columnResistance, const vector<double> &rowResistance) {   //calculate latency for different mode 
	if (!initialized) {
		cout << "[Subarray] Error: Require initialization first!" << endl;
	} else {
		
		readLatency = 0;
		writeLatency = 0;

		if (cell.memCellType == Type::SRAM) {
			if (conventionalSequential) {
				int numReadOperationPerRow = (int)ceil((double)numCol/numReadCellPerOperationNeuro);
				int numWriteOperationPerRow = (int)ceil((double)numCol*activityColWrite/numWriteCellPerOperationNeuro);
				wlDecoder.CalculateLatency(1e20, capRow1, NULL, numRow*activityRowRead, numRow*activityRowWrite);
				
				precharger.CalculateLatency(1e20, capCol, numReadOperationPerRow*numRow*activityRowRead, numWriteOperationPerRow*numRow*activityRowWrite);
				sramWriteDriver.CalculateLatency(1e20, capCol, resCol, numWriteOperationPerRow*numRow*activityRowWrite);
				senseAmp.CalculateLatency(numReadOperationPerRow*numRow*activityRowRead);
				dff.CalculateLatency(1e20, numReadOperationPerRow*numRow*activityRowRead);
				adder.CalculateLatency(1e20, dff.capTgDrain, numReadOperationPerRow*numRow*activityRowRead);
				if (numCellPerSynapse > 1) {
					shiftAddWeight.CalculateLatency(numCellPerSynapse);	
				}																								  
				if (numReadPulse > 1) {
					shiftAddInput.CalculateLatency(1);					
				}
				// Read
				double resPullDown = CalculateOnResistance(cell.widthSRAMCellNMOS * tech.featureSize, NMOS, inputParameter.temperature, tech);
				double tau = (resCellAccess + resPullDown) * (capCellAccess + capCol) + resCol * capCol / 2;
				tau *= log(tech.vdd / (tech.vdd - cell.minSenseVoltage / 2));   
				double gm = CalculateTransconductance(cell.widthAccessCMOS * tech.featureSize, NMOS, tech);
				double beta = 1 / (resPullDown * gm);
				double colRamp = 0;
				colDelay = horowitz(tau, beta, wlDecoder.rampOutput, &colRamp) * numReadOperationPerRow * numRow * activityRowRead;

				readLatency += wlDecoder.readLatency;
				readLatency += precharger.readLatency;
				readLatency += colDelay;
				readLatency += senseAmp.readLatency;
				readLatency += adder.readLatency;
				readLatency += dff.readLatency;
				readLatency += shiftAddInput.readLatency + shiftAddWeight.readLatency;
				
				readLatencyADC = precharger.readLatency + colDelay + senseAmp.readLatency;
				readLatencyAccum = adder.readLatency + dff.readLatency + shiftAddInput.readLatency + shiftAddWeight.readLatency;
				readLatencyOther = wlDecoder.readLatency;
				
				// Write (assume the average delay of pullup and pulldown inverter in SRAM cell)
				writeLatency = 0;
				double resPull;
				resPull = (CalculateOnResistance(cell.widthSRAMCellNMOS * tech.featureSize, NMOS, inputParameter.temperature, tech) + CalculateOnResistance(cell.widthSRAMCellPMOS * tech.featureSize, PMOS, inputParameter.temperature, tech)) / 2;    // take average
				tau = resPull * cell.capSRAMCell;
				gm = (CalculateTransconductance(cell.widthSRAMCellNMOS * tech.featureSize, NMOS, tech) + CalculateTransconductance(cell.widthSRAMCellPMOS * tech.featureSize, PMOS, tech)) / 2;   // take average
				beta = 1 / (resPull * gm);
				writeLatency += horowitz(tau, beta, 1e20, NULL) * numWriteOperationPerRow * numRow * activityRowWrite;
				writeLatency += wlDecoder.writeLatency;
				writeLatency += precharger.writeLatency;
				writeLatency += sramWriteDriver.writeLatency;
				
				/* Transpose Peripheral for BP */
				if (trainingEstimation) {
					readLatencyAG = 0;
					if (layerNumber != 0) {
						int numReadOperationPerCol = (int)ceil((double)numCol/numReadCellPerOperationNeuro);
						int numWriteOperationPerRow = (int)ceil((double)numCol*activityColWrite/numWriteCellPerOperationNeuro);
						wlDecoderBP.CalculateLatency(1e20, capCol, NULL, numCol*activityBPColRead, numRow*activityRowWrite);
						prechargerBP.CalculateLatency(1e20, capRow1, numReadOperationPerCol*numCol*activityBPColRead, numWriteOperationPerRow*numRow*activityRowWrite);
						sramWriteDriverBP.CalculateLatency(1e20, capCol, resCol, numWriteOperationPerRow*numRow*activityRowWrite);
						senseAmpBP.CalculateLatency(numReadOperationPerCol*numCol*activityBPColRead);
						dffBP.CalculateLatency(1e20, numReadOperationPerCol*numCol*activityBPColRead);
						adderBP.CalculateLatency(1e20, dff.capTgDrain, numReadOperationPerCol*numCol*activityBPColRead);
						if (numCellPerSynapse > 1) {
							shiftAddBPWeight.CalculateLatency(numCellPerSynapse);	
						}																								  
						if (numReadPulseBP > 1) {
							shiftAddBPInput.CalculateLatency(1);					
						}
						
						double resPullDown = CalculateOnResistance(cell.widthSRAMCellNMOS * tech.featureSize, NMOS, inputParameter.temperature, tech);
						double tau = (resCellAccess + resPullDown) * (capCellAccess + capRow1) + resRow * capRow1 / 2;
						tau *= log(tech.vdd / (tech.vdd - cell.minSenseVoltage / 2));   
						double gm = CalculateTransconductance(cell.widthAccessCMOS * tech.featureSize, NMOS, tech);
						double beta = 1 / (resPullDown * gm);
						double colRamp = 0;
						colDelay = horowitz(tau, beta, wlDecoderBP.rampOutput, &colRamp) * numReadOperationPerCol * numCol * activityBPColRead;
						
						readLatencyAG += wlDecoderBP.readLatency;
						readLatencyAG += prechargerBP.readLatency;
						readLatencyAG += colDelay;
						readLatencyAG += senseAmpBP.readLatency;
						readLatencyAG += adderBP.readLatency;
						readLatencyAG += dffBP.readLatency;
						readLatencyAG += shiftAddBPInput.readLatency + shiftAddBPWeight.readLatency;
						
						readLatencyADC += prechargerBP.readLatency + colDelay + senseAmpBP.readLatency;
						readLatencyAccum += adderBP.readLatency + dffBP.readLatency + shiftAddBPInput.readLatency + shiftAddBPWeight.readLatency;
						readLatencyOther += wlDecoderBP.readLatency;
					}
				}
				
			} else if (conventionalParallel) {
				int numReadOperationPerRow = (int)ceil((double)numCol/numReadCellPerOperationNeuro);
				int numWriteOperationPerRow = (int)ceil((double)numCol*activityColWrite/numWriteCellPerOperationNeuro);
				
				wlSwitchMatrix.CalculateLatency(1e20, capRow1, resRow, numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite);
				precharger.CalculateLatency(1e20, capCol, numColMuxed, numWriteOperationPerRow*numRow*activityRowWrite);
				sramWriteDriver.CalculateLatency(1e20, capCol, resCol, numWriteOperationPerRow*numRow*activityRowWrite);
				if (numColMuxed > 1) {
					mux.CalculateLatency(0, 0, numColMuxed);
					muxDecoder.CalculateLatency(1e20, mux.capTgGateN*ceil(numCol/numColMuxed), mux.capTgGateP*ceil(numCol/numColMuxed), numColMuxed, 0);
				}
				if (SARADC) {
					sarADC.CalculateLatency(numColMuxed);
				} else {
					multilevelSenseAmp.CalculateLatency(columnResistance, numColMuxed, 1);
					multilevelSAEncoder.CalculateLatency(1e20, numColMuxed);
				}
				if (numCellPerSynapse > 1) {
					shiftAddWeight.CalculateLatency(numColMuxed);	
				}
				if (numReadPulse > 1) {
					shiftAddInput.CalculateLatency(ceil(numColMuxed/numCellPerSynapse));	
				}
				double resPullDown = CalculateOnResistance(cell.widthSRAMCellNMOS * tech.featureSize, NMOS, inputParameter.temperature, tech);
				double tau = (resCellAccess + resPullDown) * (capCellAccess + capCol) + resCol * capCol / 2;
				tau *= log(tech.vdd / (tech.vdd - cell.minSenseVoltage / 2));   
				double gm = CalculateTransconductance(cell.widthAccessCMOS * tech.featureSize, NMOS, tech);
				double beta = 1 / (resPullDown * gm);
				double colRamp = 0;
				colDelay = horowitz(tau, beta, wlSwitchMatrix.rampOutput, &colRamp);

				readLatency = 0;
				readLatency += MAX(wlSwitchMatrix.readLatency, ( ((numColMuxed > 1)==true? (mux.readLatency+muxDecoder.readLatency):0) )/numReadPulse);
				readLatency += precharger.readLatency;
				readLatency += colDelay;
				readLatency += multilevelSenseAmp.readLatency;
				readLatency += multilevelSAEncoder.readLatency;
				readLatency += shiftAddInput.readLatency + shiftAddWeight.readLatency;
				readLatency += sarADC.readLatency;
				
				readLatencyADC = precharger.readLatency + colDelay + multilevelSenseAmp.readLatency + multilevelSAEncoder.readLatency + sarADC.readLatency;
				readLatencyAccum = shiftAddInput.readLatency + shiftAddWeight.readLatency;
				readLatencyOther = MAX(wlSwitchMatrix.readLatency, ( ((numColMuxed > 1)==true? (mux.readLatency+muxDecoder.readLatency):0) )/numReadPulse);

				// Write (assume the average delay of pullup and pulldown inverter in SRAM cell)
				
				double resPull;
				resPull = (CalculateOnResistance(cell.widthSRAMCellNMOS * tech.featureSize, NMOS, inputParameter.temperature, tech) + CalculateOnResistance(cell.widthSRAMCellPMOS * tech.featureSize, PMOS, inputParameter.temperature, tech)) / 2;    // take average
				tau = resPull * cell.capSRAMCell;
				gm = (CalculateTransconductance(cell.widthSRAMCellNMOS * tech.featureSize, NMOS, tech) + CalculateTransconductance(cell.widthSRAMCellPMOS * tech.featureSize, PMOS, tech)) / 2;   // take average
				beta = 1 / (resPull * gm);
				
				writeLatency += horowitz(tau, beta, 1e20, NULL) * numWriteOperationPerRow * numRow * activityRowWrite;
				writeLatency += wlSwitchMatrix.writeLatency;
				writeLatency += precharger.writeLatency;
				writeLatency += sramWriteDriver.writeLatency;
				
				
				/* Transpose Peripheral for BP */
				if (trainingEstimation) {
					readLatencyAG = 0;
					if (layerNumber != 0) {
						int numReadOperationPerCol = (int)ceil((double)numCol/numReadCellPerOperationNeuro);
						int numWriteOperationPerRow = (int)ceil((double)numCol*activityColWrite/numWriteCellPerOperationNeuro);
						
						if (parallelBP) {
							wlSwitchMatrixBP.CalculateLatency(1e20, capCol, resCol, numRowMuxedBP, 2*numWriteOperationPerRow*numRow*activityRowWrite);
							prechargerBP.CalculateLatency(1e20, capRow1, numRowMuxedBP, numWriteOperationPerRow*numRow*activityRowWrite);
							sramWriteDriverBP.CalculateLatency(1e20, capRow1, resRow, numWriteOperationPerRow*numRow*activityRowWrite);
							if (numRowMuxedBP>1) {	
								muxBP.CalculateLatency(0, 0, numRowMuxedBP);
								muxDecoderBP.CalculateLatency(1e20, mux.capTgGateN*ceil(numCol/numRowMuxedBP), mux.capTgGateP*ceil(numCol/numRowMuxedBP), numRowMuxedBP, 0);
							}
							if (SARADC) {
								sarADCBP.CalculateLatency(numRowMuxedBP);
							} else {
								multilevelSenseAmpBP.CalculateLatency(columnResistance, numRowMuxedBP, 1);
								multilevelSAEncoderBP.CalculateLatency(1e20, numRowMuxedBP);
							}
							if (numCellPerSynapse > 1) {
								shiftAddBPWeight.CalculateLatency(numRowMuxedBP);	
							}
							if (numReadPulseBP > 1) {
								shiftAddBPInput.CalculateLatency(ceil(numRowMuxedBP/numCellPerSynapse));	
							}
							// Read
							double resPullDown = CalculateOnResistance(cell.widthSRAMCellNMOS * tech.featureSize, NMOS, inputParameter.temperature, tech);
							double tau = (resCellAccess + resPullDown) * (capCellAccess + capRow1) + resRow * capRow1 / 2;
							tau *= log(tech.vdd / (tech.vdd - cell.minSenseVoltage / 2));   
							double gm = CalculateTransconductance(cell.widthAccessCMOS * tech.featureSize, NMOS, tech);
							double beta = 1 / (resPullDown * gm);
							double colRamp = 0;
							colDelay = horowitz(tau, beta, wlSwitchMatrixBP.rampOutput, &colRamp);

							readLatencyAG = 0;
							readLatencyAG += MAX(wlSwitchMatrixBP.readLatency, ( ((numRowMuxedBP > 1)==true? (muxBP.readDynamicEnergy + muxDecoderBP.readDynamicEnergy):0) )/numReadPulseBP);
							readLatencyAG += prechargerBP.readLatency;
							readLatencyAG += colDelay;
							readLatencyAG += multilevelSenseAmpBP.readLatency;
							readLatencyAG += multilevelSAEncoderBP.readLatency;
							readLatencyAG += shiftAddBPInput.readLatency + shiftAddBPWeight.readLatency;
							readLatencyAG += sarADCBP.readLatency;
							
							readLatencyADC += prechargerBP.readLatency + colDelay + multilevelSenseAmpBP.readLatency + multilevelSAEncoderBP.readLatency + sarADCBP.readLatency;
							readLatencyAccum += shiftAddBPInput.readLatency + shiftAddBPWeight.readLatency;
							readLatencyOther += MAX(wlSwitchMatrixBP.readLatency, ( ((numRowMuxedBP > 1)==true? (muxBP.readDynamicEnergy + muxDecoderBP.readDynamicEnergy):0) )/numReadPulseBP);
							
						} else {
							wlSwitchMatrixBP.CalculateLatency(1e20, capCol, resCol, numReadOperationPerCol*numCol*activityBPColRead, 2*numWriteOperationPerRow*numRow*activityRowWrite);
							prechargerBP.CalculateLatency(1e20, capRow1, numReadOperationPerCol*numCol*activityBPColRead, numWriteOperationPerRow*numRow*activityRowWrite);
							sramWriteDriverBP.CalculateLatency(1e20, capRow1, resRow, numWriteOperationPerRow*numRow*activityRowWrite);
							senseAmpBP.CalculateLatency(numReadOperationPerCol*numCol*activityBPColRead);
							dffBP.CalculateLatency(1e20, numReadOperationPerCol*numCol*activityBPColRead);
							adderBP.CalculateLatency(1e20, dffBP.capTgDrain, numReadOperationPerCol*numCol*activityBPColRead);
							if (numCellPerSynapse > 1) {
								shiftAddBPWeight.CalculateLatency(numCellPerSynapse);	
							}																								  
							if (numReadPulseBP > 1) {
								shiftAddBPInput.CalculateLatency(1);					
							}
							
							double resPullDown = CalculateOnResistance(cell.widthSRAMCellNMOS * tech.featureSize, NMOS, inputParameter.temperature, tech);
							double tau = (resCellAccess + resPullDown) * (capCellAccess + capRow1) + resRow * capRow1 / 2;
							tau *= log(tech.vdd / (tech.vdd - cell.minSenseVoltage / 2));   
							double gm = CalculateTransconductance(cell.widthAccessCMOS * tech.featureSize, NMOS, tech);
							double beta = 1 / (resPullDown * gm);
							double colRamp = 0;
							colDelay = horowitz(tau, beta, wlSwitchMatrixBP.rampOutput, &colRamp) * numReadOperationPerCol * numCol * activityBPColRead;
							
							readLatencyAG += wlSwitchMatrixBP.readLatency;
							readLatencyAG += prechargerBP.readLatency;
							readLatencyAG += colDelay;
							readLatencyAG += senseAmpBP.readLatency;
							readLatencyAG += adderBP.readLatency;
							readLatencyAG += dffBP.readLatency;
							readLatencyAG += shiftAddBPInput.readLatency + shiftAddBPWeight.readLatency;
							
							readLatencyADC += prechargerBP.readLatency + colDelay + senseAmpBP.readLatency;
							readLatencyAccum += adderBP.readLatency + dffBP.readLatency + shiftAddBPInput.readLatency + shiftAddBPWeight.readLatency;
							readLatencyOther += wlSwitchMatrixBP.readLatency;
						}
					}
				}
				
			} else if (BNNsequentialMode || XNORsequentialMode) {
				int numReadOperationPerRow = (int)ceil((double)numCol/numReadCellPerOperationNeuro);
				int numWriteOperationPerRow = (int)ceil((double)numCol*activityColWrite/numWriteCellPerOperationNeuro);
				
				wlDecoder.CalculateLatency(1e20, capRow1, NULL, numRow*activityRowRead, numRow*activityRowWrite);
				precharger.CalculateLatency(1e20, capCol, numReadOperationPerRow*numRow*activityRowRead, numWriteOperationPerRow*numRow*activityRowWrite);
				sramWriteDriver.CalculateLatency(1e20, capCol, resCol, numWriteOperationPerRow*numRow*activityRowWrite);
				senseAmp.CalculateLatency(numReadOperationPerRow*numRow*activityRowRead);
				dff.CalculateLatency(1e20, numReadOperationPerRow*numRow*activityRowRead);
				adder.CalculateLatency(1e20, dff.capTgDrain, numReadOperationPerRow*numRow*activityRowRead);
				
				// Read
				double resPullDown = CalculateOnResistance(cell.widthSRAMCellNMOS * tech.featureSize, NMOS, inputParameter.temperature, tech);
				double tau = (resCellAccess + resPullDown) * (capCellAccess + capCol) + resCol * capCol / 2;
				tau *= log(tech.vdd / (tech.vdd - cell.minSenseVoltage / 2));   
				double gm = CalculateTransconductance(cell.widthAccessCMOS * tech.featureSize, NMOS, tech);
				double beta = 1 / (resPullDown * gm);
				double colRamp = 0;
				colDelay = horowitz(tau, beta, wlDecoder.rampOutput, &colRamp) * numReadOperationPerRow * numRow * activityRowRead;

				readLatency += wlDecoder.readLatency;
				readLatency += precharger.readLatency;
				readLatency += colDelay;
				readLatency += senseAmp.readLatency;
				readLatency += adder.readLatency;
				readLatency += dff.readLatency;
				
				// Write (assume the average delay of pullup and pulldown inverter in SRAM cell)
				
				double resPull;
				resPull = (CalculateOnResistance(cell.widthSRAMCellNMOS * tech.featureSize, NMOS, inputParameter.temperature, tech) + CalculateOnResistance(cell.widthSRAMCellPMOS * tech.featureSize, PMOS, inputParameter.temperature, tech)) / 2;    // take average
				tau = resPull * cell.capSRAMCell;
				gm = (CalculateTransconductance(cell.widthSRAMCellNMOS * tech.featureSize, NMOS, tech) + CalculateTransconductance(cell.widthSRAMCellPMOS * tech.featureSize, PMOS, tech)) / 2;   // take average
				beta = 1 / (resPull * gm);

				writeLatency += horowitz(tau, beta, 1e20, NULL) * numWriteOperationPerRow * numRow * activityRowWrite;
				writeLatency += wlDecoder.writeLatency;
				writeLatency += precharger.writeLatency;
				writeLatency += sramWriteDriver.writeLatency;
				
			} else if (BNNparallelMode || XNORparallelMode) {
				int numReadOperationPerRow = (int)ceil((double)numCol/numReadCellPerOperationNeuro);
				int numWriteOperationPerRow = (int)ceil((double)numCol*activityColWrite/numWriteCellPerOperationNeuro);
				
				wlSwitchMatrix.CalculateLatency(1e20, capRow1, resRow, numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite);
				precharger.CalculateLatency(1e20, capCol, numColMuxed, numWriteOperationPerRow*numRow*activityRowWrite);
				sramWriteDriver.CalculateLatency(1e20, capCol, resCol, numWriteOperationPerRow*numRow*activityRowWrite);
				
				if (SARADC) {
					sarADC.CalculateLatency(numColMuxed);
				} else {
					multilevelSenseAmp.CalculateLatency(columnResistance, numColMuxed, 1);
					multilevelSAEncoder.CalculateLatency(1e20, numColMuxed);
				}
				
				// Read
				double resPullDown = CalculateOnResistance(cell.widthSRAMCellNMOS * tech.featureSize, NMOS, inputParameter.temperature, tech);
				double tau = (resCellAccess + resPullDown) * (capCellAccess + capCol) + resCol * capCol / 2;
				tau *= log(tech.vdd / (tech.vdd - cell.minSenseVoltage / 2));   
				double gm = CalculateTransconductance(cell.widthAccessCMOS * tech.featureSize, NMOS, tech);
				double beta = 1 / (resPullDown * gm);
				double colRamp = 0;
				colDelay = horowitz(tau, beta, wlSwitchMatrix.rampOutput, &colRamp) * numReadOperationPerRow * numRow * activityRowRead;

				readLatency = 0;
				readLatency += wlSwitchMatrix.readLatency;
				readLatency += precharger.readLatency;
				readLatency += colDelay;
				readLatency += multilevelSenseAmp.readLatency;
				readLatency += multilevelSAEncoder.readLatency;
				readLatency += sarADC.readLatency;

				// Write (assume the average delay of pullup and pulldown inverter in SRAM cell)
				
				double resPull;
				resPull = (CalculateOnResistance(cell.widthSRAMCellNMOS * tech.featureSize, NMOS, inputParameter.temperature, tech) + CalculateOnResistance(cell.widthSRAMCellPMOS * tech.featureSize, PMOS, inputParameter.temperature, tech)) / 2;    // take average
				tau = resPull * cell.capSRAMCell;
				gm = (CalculateTransconductance(cell.widthSRAMCellNMOS * tech.featureSize, NMOS, tech) + CalculateTransconductance(cell.widthSRAMCellPMOS * tech.featureSize, PMOS, tech)) / 2;   // take average
				beta = 1 / (resPull * gm);
				
				writeLatency += horowitz(tau, beta, 1e20, NULL) * numWriteOperationPerRow * numRow * activityRowWrite;
				writeLatency += wlSwitchMatrix.writeLatency;
				writeLatency += precharger.writeLatency;
				writeLatency += sramWriteDriver.writeLatency;
				
			} else {
				int numReadOperationPerRow = (int)ceil((double)numCol/numReadCellPerOperationNeuro);
				int numWriteOperationPerRow = (int)ceil((double)numCol*activityColWrite/numWriteCellPerOperationNeuro);
				
				wlSwitchMatrix.CalculateLatency(1e20, capRow1, resRow, numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite);
				precharger.CalculateLatency(1e20, capCol, numColMuxed, numWriteOperationPerRow*numRow*activityRowWrite);
				sramWriteDriver.CalculateLatency(1e20, capCol, resCol, numWriteOperationPerRow*numRow*activityRowWrite);
				if (SARADC) {
					sarADC.CalculateLatency(numColMuxed);
				} else {
					multilevelSenseAmp.CalculateLatency(columnResistance, numColMuxed, 1);
					multilevelSAEncoder.CalculateLatency(1e20, numColMuxed);
				}
				if (numCellPerSynapse > 1) {
					shiftAddWeight.CalculateLatency(numCellPerSynapse);	
				}																								  
				if (numReadPulse > 1) {
					shiftAddInput.CalculateLatency(1);					
				}
				// Read
				double resPullDown = CalculateOnResistance(cell.widthSRAMCellNMOS * tech.featureSize, NMOS, inputParameter.temperature, tech);
				double tau = (resCellAccess + resPullDown) * (capCellAccess + capCol) + resCol * capCol / 2;
				tau *= log(tech.vdd / (tech.vdd - cell.minSenseVoltage / 2));   
				double gm = CalculateTransconductance(cell.widthAccessCMOS * tech.featureSize, NMOS, tech);
				double beta = 1 / (resPullDown * gm);
				double colRamp = 0;
				colDelay = horowitz(tau, beta, wlDecoder.rampOutput, &colRamp) * numReadOperationPerRow * numRow * activityRowRead;

				readLatency = 0;
				readLatency += wlSwitchMatrix.readLatency;
				readLatency += precharger.readLatency;
				readLatency += colDelay;
				readLatency += multilevelSenseAmp.readLatency;
				readLatency += multilevelSAEncoder.readLatency;
				readLatency += shiftAddInput.readLatency + shiftAddWeight.readLatency;
				readLatency += sarADC.readLatency;

				// Write (assume the average delay of pullup and pulldown inverter in SRAM cell)
				
				double resPull;
				resPull = (CalculateOnResistance(cell.widthSRAMCellNMOS * tech.featureSize, NMOS, inputParameter.temperature, tech) + CalculateOnResistance(cell.widthSRAMCellPMOS * tech.featureSize, PMOS, inputParameter.temperature, tech)) / 2;    // take average
				tau = resPull * cell.capSRAMCell;
				gm = (CalculateTransconductance(cell.widthSRAMCellNMOS * tech.featureSize, NMOS, tech) + CalculateTransconductance(cell.widthSRAMCellPMOS * tech.featureSize, PMOS, tech)) / 2;   // take average
				beta = 1 / (resPull * gm);
				
				writeLatency += horowitz(tau, beta, 1e20, NULL) * numWriteOperationPerRow * numRow * activityRowWrite;
				writeLatency += wlSwitchMatrix.writeLatency;
				writeLatency += precharger.writeLatency;
				writeLatency += sramWriteDriver.writeLatency;
				
			}
	    } else if (cell.memCellType == Type::RRAM || cell.memCellType == Type::FeFET) {
			if (conventionalSequential) {
				double capBL = lengthCol * 0.2e-15/1e-6;
				double colRamp = 0;
				double tau = (capCol)*(cell.resMemCellAvg);
				colDelay = horowitz(tau, 0, 1e20, &colRamp);	// Just to generate colRamp
				colDelay = tau * 0.2 * numColMuxed;  // assume the 15~20% voltage drop is enough for sensing
				int numWriteOperationPerRow = (int)ceil((double)numCol*activityColWrite/numWriteCellPerOperationNeuro);
				
				wlDecoder.CalculateLatency(1e20, capRow2, NULL, numRow*activityRowRead*numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite);
				if (cell.accessType == CMOS_access) {
					wlNewDecoderDriver.CalculateLatency(wlDecoder.rampOutput, capRow2, resRow, numRow*activityRowRead*numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite);	
				} else {
					wlDecoderDriver.CalculateLatency(wlDecoder.rampOutput, capRow1, capRow1, resRow, numRow*activityRowRead*numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite);
				}
				slSwitchMatrix.CalculateLatency(1e20, capCol, resCol, 0, 2*numWriteOperationPerRow*numRow*activityRowWrite);
				if (numColMuxed > 1) {
					mux.CalculateLatency(colRamp, 0, numColMuxed);
					muxDecoder.CalculateLatency(1e20, mux.capTgGateN*ceil(numCol/numColMuxed), mux.capTgGateP*ceil(numCol/numColMuxed), numColMuxed, 0);
				}
				if (SARADC) {
					sarADC.CalculateLatency(numColMuxed*numRow*activityRowRead);
				} else {
					multilevelSenseAmp.CalculateLatency(columnResistance, numColMuxed, numRow*activityRowRead);
					if (avgWeightBit > 1) {
						multilevelSAEncoder.CalculateLatency(1e20, numColMuxed*numRow*activityRowRead);
					}
				}
				
				adder.CalculateLatency(1e20, dff.capTgDrain, numColMuxed*numRow*activityRowRead);
				dff.CalculateLatency(1e20, numColMuxed*numRow*activityRowRead);
				if (numCellPerSynapse > 1) {				 
					shiftAddWeight.CalculateLatency(numColMuxed);							// There are numReadPulse times of shift-and-add
				}																								
				if (numReadPulse > 1) {
					shiftAddInput.CalculateLatency(ceil(numColMuxed/numCellPerSynapse));	// There are numReadPulse times of shift-and-add
				}
				
				// Read
				readLatency = 0;
				readLatency += MAX(wlDecoder.readLatency + wlNewDecoderDriver.readLatency + wlDecoderDriver.readLatency, ( ((numColMuxed > 1)==true? (mux.readLatency+muxDecoder.readLatency):0) )/numReadPulse);
				readLatency += multilevelSenseAmp.readLatency;
				readLatency += multilevelSAEncoder.readLatency;
				readLatency += adder.readLatency;
				readLatency += dff.readLatency;
				readLatency += shiftAddInput.readLatency + shiftAddWeight.readLatency;
				readLatency += colDelay/numReadPulse;
				readLatency += sarADC.readLatency;
				
				readLatencyADC = multilevelSenseAmp.readLatency + multilevelSAEncoder.readLatency + sarADC.readLatency;
				readLatencyAccum = adder.readLatency + dff.readLatency + shiftAddInput.readLatency + shiftAddWeight.readLatency;
				readLatencyOther = MAX(wlDecoder.readLatency + wlNewDecoderDriver.readLatency + wlDecoderDriver.readLatency, ( ((numColMuxed > 1)==true? (mux.readLatency+muxDecoder.readLatency):0) )/numReadPulse) + colDelay/numReadPulse;
				
				// Write
				writeLatency = 0;
				writeLatencyArray = 0;
				writeLatencyArray += totalNumWritePulse * cell.writePulseWidth;
				writeLatency += MAX(wlDecoder.writeLatency + wlNewDecoderDriver.writeLatency + wlDecoderDriver.writeLatency, slSwitchMatrix.writeLatency);
				writeLatency += writeLatencyArray;
				
				/* Transpose Peripheral for BP */
				if (trainingEstimation) {
					readLatencyAG = 0;
					if (layerNumber != 0) {
						double capRow = lengthRow * 0.2e-15/1e-6 + CalculateDrainCap(cell.widthAccessCMOS * tech.featureSize, NMOS, cell.widthInFeatureSize * tech.featureSize, tech) * numCol;
						tau = (capRow)*(cell.resMemCellAvg);
						double rowDelay = tau * 0.2 * numRowMuxedBP;  // assume the 15~20% voltage drop is enough for sensing
						
						slSwitchMatrix.CalculateLatency(1e20, capCol, resCol, numRowMuxedBP*numCol*activityBPColRead, 2*numWriteOperationPerRow*numRow*activityRowWrite);
						if (numRowMuxedBP>1) {
							muxBP.CalculateLatency(colRamp, 0, numRowMuxedBP);
							muxDecoderBP.CalculateLatency(1e20, muxBP.capTgGateN*ceil(numRow/numRowMuxedBP), muxBP.capTgGateP*ceil(numRow/numRowMuxedBP), numRowMuxedBP, 0);
						}
						if (SARADC) {
							sarADCBP.CalculateLatency(numRowMuxedBP*numCol*activityBPColRead);
						} else {
							multilevelSenseAmpBP.CalculateLatency(columnResistance, numRowMuxedBP, numCol*activityBPColRead);
							if (avgWeightBit > 1) {
								multilevelSAEncoderBP.CalculateLatency(1e20, numRowMuxedBP*numCol*activityBPColRead);
							}
						}
						
						dffBP.CalculateLatency(1e20, numRowMuxedBP*numCol*activityBPColRead);
						adderBP.CalculateLatency(1e20, dffBP.capTgDrain, numRowMuxedBP*numCol*activityBPColRead);
						
						if (numCellPerSynapse > 1) {				 
							shiftAddBPWeight.CalculateLatency(numRowMuxedBP);							// There are numReadPulse times of shift-and-add
						}																								
						if (numReadPulseBP > 1) {
							shiftAddBPInput.CalculateLatency(ceil(numRowMuxedBP/numCellPerSynapse));	// There are numReadPulse times of shift-and-add
						}

						readLatencyAG += MAX(slSwitchMatrix.readLatency, ( ((numRowMuxedBP > 1)==true? (muxBP.readDynamicEnergy + muxDecoderBP.readDynamicEnergy):0) )/numReadPulseBP);
						readLatencyAG += multilevelSenseAmpBP.readLatency;
						readLatencyAG += multilevelSAEncoderBP.readLatency;
						readLatencyAG += adderBP.readLatency;
						readLatencyAG += dffBP.readLatency;
						readLatencyAG += shiftAddBPInput.readLatency + shiftAddBPWeight.readLatency;
						readLatencyAG += rowDelay/numReadPulseBP;
						readLatencyAG += sarADCBP.readLatency;
						
						readLatencyADC += multilevelSenseAmpBP.readLatency + multilevelSAEncoderBP.readLatency + sarADCBP.readLatency;
						readLatencyAccum += adderBP.readLatency + dffBP.readLatency + shiftAddBPInput.readLatency + shiftAddBPWeight.readLatency;
						readLatencyOther += MAX(slSwitchMatrix.readLatency, ( ((numRowMuxedBP > 1)==true? (muxBP.readDynamicEnergy + muxDecoderBP.readDynamicEnergy):0) )/numReadPulseBP) + rowDelay/numReadPulseBP;
					}
				}
				
			} else if (conventionalParallel) {
				double capBL = lengthCol * 0.2e-15/1e-6;
				int numWriteOperationPerRow = (int)ceil((double)numCol*activityColWrite/numWriteCellPerOperationNeuro);
				double colRamp = 0;
				double tau = (capCol)*(cell.resMemCellAvg/(numRow/2));
				colDelay = horowitz(tau, 0, 1e20, &colRamp);
				colDelay = tau * 0.2 * numColMuxed;  // assume the 15~20% voltage drop is enough for sensing
				
				if (cell.accessType == CMOS_access) {
					wlNewSwitchMatrix.CalculateLatency(1e20, capRow2, resRow, numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite);
				} else {
					wlSwitchMatrix.CalculateLatency(1e20, capRow1, resRow, numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite);
				}
				slSwitchMatrix.CalculateLatency(1e20, capCol, resCol, 0, 2*numWriteOperationPerRow*numRow*activityRowWrite);
				if (numColMuxed>1) {
					mux.CalculateLatency(colRamp, 0, numColMuxed);
					muxDecoder.CalculateLatency(1e20, mux.capTgGateN*ceil(numCol/numColMuxed), mux.capTgGateP*ceil(numCol/numColMuxed), numColMuxed, 0);
				}
				if (SARADC) {
					sarADC.CalculateLatency(numColMuxed);
				} else {
					multilevelSenseAmp.CalculateLatency(columnResistance, numColMuxed, 1);
					multilevelSAEncoder.CalculateLatency(1e20, numColMuxed);
				}
				
				if (numCellPerSynapse > 1) {
					shiftAddWeight.CalculateLatency(numColMuxed);	
				}
				if (numReadPulse > 1) {
					shiftAddInput.CalculateLatency(ceil(numColMuxed/numCellPerSynapse));		
				}
				
				// Read
				readLatency = 0;
				readLatency += MAX(wlNewSwitchMatrix.readLatency + wlSwitchMatrix.readLatency, ( ((numColMuxed > 1)==true? (mux.readLatency+muxDecoder.readLatency):0) )/numReadPulse);
				readLatency += multilevelSenseAmp.readLatency;
				readLatency += multilevelSAEncoder.readLatency;
				readLatency += shiftAddInput.readLatency + shiftAddWeight.readLatency;
				readLatency += colDelay/numReadPulse;
				readLatency += sarADC.readLatency;
				
				readLatencyADC = multilevelSenseAmp.readLatency + multilevelSAEncoder.readLatency + sarADC.readLatency;
				readLatencyAccum = shiftAddInput.readLatency + shiftAddWeight.readLatency;
				readLatencyOther = MAX(wlNewSwitchMatrix.readLatency + wlSwitchMatrix.readLatency, ( ((numColMuxed > 1)==true? (mux.readLatency+muxDecoder.readLatency):0) )/numReadPulse) + colDelay/numReadPulse;

				// Write
				writeLatency = 0;
				writeLatencyArray = 0;
				writeLatencyArray += totalNumWritePulse * cell.writePulseWidth;
				writeLatency += MAX(wlNewSwitchMatrix.writeLatency + wlSwitchMatrix.writeLatency, slSwitchMatrix.writeLatency);
				writeLatency += writeLatencyArray;
				
				/* Transpose Peripheral for BP */
				if (trainingEstimation) {
					readLatencyAG = 0;
					if (layerNumber != 0) {
						double capRow = lengthRow * 0.2e-15/1e-6 + CalculateDrainCap(cell.widthAccessCMOS * tech.featureSize, NMOS, cell.widthInFeatureSize * tech.featureSize, tech) * numCol;
						tau = (capRow)*(cell.resMemCellAvg/(numCol/2));
						double rowDelay = tau * 0.2 * numRowMuxedBP;  // assume the 15~20% voltage drop is enough for sensing
						if (parallelBP) {
							slSwitchMatrix.CalculateLatency(1e20, capCol, resCol, numRowMuxedBP, 2*numWriteOperationPerRow*numRow*activityRowWrite);
							if (numRowMuxedBP>1) {
								muxBP.CalculateLatency(colRamp, 0, numRowMuxedBP);
								muxDecoderBP.CalculateLatency(1e20, muxBP.capTgGateN*ceil(numRow/numRowMuxedBP), muxBP.capTgGateP*ceil(numRow/numRowMuxedBP), numRowMuxedBP, 0);
							}
							if (SARADC) {
								sarADCBP.CalculateLatency(numRowMuxedBP);
							} else {
								multilevelSenseAmpBP.CalculateLatency(columnResistance, numRowMuxedBP, 1);
								multilevelSAEncoderBP.CalculateLatency(1e20, numRowMuxedBP);
							}
							
							if (numCellPerSynapse > 1) {
								shiftAddBPWeight.CalculateLatency(numRowMuxedBP);	
							}
							if (numReadPulseBP > 1) {
								shiftAddBPInput.CalculateLatency(ceil(numRowMuxedBP/numCellPerSynapse));		
							}
							
							readLatencyAG += MAX(slSwitchMatrix.readLatency, ( ((numRowMuxedBP > 1)==true? (muxBP.readDynamicEnergy + muxDecoderBP.readDynamicEnergy):0) )/numReadPulseBP);
							readLatencyAG += multilevelSenseAmpBP.readLatency;
							readLatencyAG += multilevelSAEncoderBP.readLatency;
							readLatencyAG += shiftAddBPInput.readLatency + shiftAddBPWeight.readLatency;
							readLatencyAG += rowDelay/numReadPulseBP;
							readLatencyAG += sarADCBP.readLatency;
							
							readLatencyADC += multilevelSenseAmpBP.readLatency + multilevelSAEncoderBP.readLatency + sarADCBP.readLatency;
							readLatencyAccum += shiftAddBPInput.readLatency + shiftAddBPWeight.readLatency;
							readLatencyOther += MAX(slSwitchMatrix.readLatency, ( ((numRowMuxedBP > 1)==true? (muxBP.readDynamicEnergy + muxDecoderBP.readDynamicEnergy):0) )/numReadPulseBP) + rowDelay/numReadPulseBP;
						} else {
							slSwitchMatrix.CalculateLatency(1e20, capCol, resCol, numRowMuxedBP*numCol*activityBPColRead, 2*numWriteOperationPerRow*numRow*activityRowWrite);
							if (numRowMuxedBP>1) {
								muxBP.CalculateLatency(colRamp, 0, numRowMuxedBP);
								muxDecoderBP.CalculateLatency(1e20, muxBP.capTgGateN*ceil(numRow/numRowMuxedBP), muxBP.capTgGateP*ceil(numRow/numRowMuxedBP), numRowMuxedBP, 0);
							}
							if (SARADC) {
								sarADCBP.CalculateLatency(numRowMuxedBP*numCol*activityBPColRead);
							} else {
								multilevelSenseAmpBP.CalculateLatency(columnResistance, numRowMuxedBP, numCol*activityBPColRead);
								if (avgWeightBit > 1) {
									multilevelSAEncoderBP.CalculateLatency(1e20, numRowMuxedBP*numCol*activityBPColRead);
								}
							}

							dffBP.CalculateLatency(1e20, numRowMuxedBP*numCol*activityBPColRead);
							adderBP.CalculateLatency(1e20, dffBP.capTgDrain, numRowMuxedBP*numCol*activityBPColRead);
							
							if (numCellPerSynapse > 1) {
								shiftAddBPWeight.CalculateLatency(numRowMuxedBP);	
							}
							if (numReadPulseBP > 1) {
								shiftAddBPInput.CalculateLatency(ceil(numRowMuxedBP/numCellPerSynapse));		
							}
							
							readLatencyAG += MAX(slSwitchMatrix.readLatency, ( ((numRowMuxedBP > 1)==true? (muxBP.readDynamicEnergy + muxDecoderBP.readDynamicEnergy):0) )/numReadPulseBP);
							readLatencyAG += multilevelSenseAmpBP.readLatency;
							readLatencyAG += multilevelSAEncoderBP.readLatency;
							readLatencyAG += adderBP.readLatency;
							readLatencyAG += dffBP.readLatency;
							readLatencyAG += shiftAddBPInput.readLatency + shiftAddBPWeight.readLatency;
							readLatencyAG += rowDelay/numReadPulseBP;
							readLatencyAG += sarADCBP.readLatency;
							
							readLatencyADC += multilevelSenseAmpBP.readLatency + multilevelSAEncoderBP.readLatency + sarADCBP.readLatency;
							readLatencyAccum += adderBP.readLatency + dffBP.readLatency + shiftAddBPInput.readLatency + shiftAddBPWeight.readLatency;
							readLatencyOther += MAX(slSwitchMatrix.readLatency, ( ((numRowMuxedBP > 1)==true? (muxBP.readDynamicEnergy + muxDecoderBP.readDynamicEnergy):0) )/numReadPulseBP) + rowDelay/numReadPulseBP;
						}
					}
				}
				
			} else if (BNNsequentialMode || XNORsequentialMode) {
				double capBL = lengthCol * 0.2e-15/1e-6;
				double colRamp = 0;
				double tau = (capCol)*(cell.resMemCellAvg);
				colDelay = horowitz(tau, 0, 1e20, &colRamp);	// Just to generate colRamp
				colDelay = tau * 0.2 * numColMuxed;  // assume the 15~20% voltage drop is enough for sensing
				int numWriteOperationPerRow = (int)ceil((double)numCol*activityColWrite/numWriteCellPerOperationNeuro);
				
				wlDecoder.CalculateLatency(1e20, capRow2, NULL, numRow*activityRowRead*numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite);
				if (cell.accessType == CMOS_access) {
					wlNewDecoderDriver.CalculateLatency(wlDecoder.rampOutput, capRow2, resRow, numRow*activityRowRead*numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite);	
				} else {
					wlDecoderDriver.CalculateLatency(wlDecoder.rampOutput, capRow1, capRow1, resRow, numRow*activityRowRead*numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite);
				}
				slSwitchMatrix.CalculateLatency(1e20, capCol, resCol, 0, 2*numWriteOperationPerRow*numRow*activityRowWrite);
				if (numColMuxed > 1) {
					mux.CalculateLatency(colRamp, 0, numColMuxed);
					muxDecoder.CalculateLatency(1e20, mux.capTgGateN*ceil(numCol/numColMuxed), mux.capTgGateP*ceil(numCol/numColMuxed), numColMuxed, 0);
				}
				rowCurrentSenseAmp.CalculateLatency(columnResistance, numColMuxed, numRow*activityRowRead);
				adder.CalculateLatency(1e20, dff.capTgDrain, numColMuxed*numRow*activityRowRead);
				dff.CalculateLatency(1e20, numColMuxed*numRow*activityRowRead);
				
				// Read
				readLatency = 0;
				readLatency += MAX(wlDecoder.readLatency + wlNewDecoderDriver.readLatency + wlDecoderDriver.readLatency, ( ((numColMuxed > 1)==true? (mux.readLatency+muxDecoder.readLatency):0) )/numReadPulse);
				readLatency += rowCurrentSenseAmp.readLatency;
				readLatency += adder.readLatency;
				readLatency += dff.readLatency;
				readLatency += colDelay/numReadPulse;
				
				// Write
				
				writeLatency = 0;
				writeLatencyArray = 0;
				writeLatencyArray += totalNumWritePulse * cell.writePulseWidth;
				writeLatency += MAX(wlDecoder.writeLatency + wlNewDecoderDriver.writeLatency + wlDecoderDriver.writeLatency, slSwitchMatrix.writeLatency);
				writeLatency += writeLatencyArray;
				
			} else if (BNNparallelMode || XNORparallelMode) {
				double capBL = lengthCol * 0.2e-15/1e-6;
				int numWriteOperationPerRow = (int)ceil((double)numCol*activityColWrite/numWriteCellPerOperationNeuro);
				double colRamp = 0;
				double tau = (capCol)*(cell.resMemCellAvg/(numRow/2));
				colDelay = horowitz(tau, 0, 1e20, &colRamp);
				colDelay = tau * 0.2 * numColMuxed;  // assume the 15~20% voltage drop is enough for sensing
				
				if (cell.accessType == CMOS_access) {
					wlNewSwitchMatrix.CalculateLatency(1e20, capRow2, resRow, numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite);
				} else {
					wlSwitchMatrix.CalculateLatency(1e20, capRow1, resRow, numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite);
				}
				slSwitchMatrix.CalculateLatency(1e20, capCol, resCol, 0, 2*numWriteOperationPerRow*numRow*activityRowWrite);
				if (numColMuxed > 1) {
					mux.CalculateLatency(colRamp, 0, numColMuxed);
					muxDecoder.CalculateLatency(1e20, mux.capTgGateN*ceil(numCol/numColMuxed), mux.capTgGateP*ceil(numCol/numColMuxed), numColMuxed, 0);
				}
				if (SARADC) {
					sarADC.CalculateLatency(numColMuxed);
				} else {
					multilevelSenseAmp.CalculateLatency(columnResistance, numColMuxed, 1);
					multilevelSAEncoder.CalculateLatency(1e20, numColMuxed);
				}

				// Read
				readLatency = 0;
				readLatency += MAX(wlNewSwitchMatrix.readLatency + wlSwitchMatrix.readLatency, ( ((numColMuxed > 1)==true? (mux.readLatency+muxDecoder.readLatency):0) )/numReadPulse);
				readLatency += multilevelSenseAmp.readLatency;
				readLatency += multilevelSAEncoder.readLatency;
				readLatency += colDelay/numReadPulse;
				readLatency += sarADC.readLatency;
				// Write
				
				writeLatency = 0;
				writeLatencyArray = 0;
				writeLatencyArray += totalNumWritePulse * cell.writePulseWidth;
				writeLatency += MAX(wlNewSwitchMatrix.writeLatency + wlSwitchMatrix.writeLatency, slSwitchMatrix.writeLatency);
				writeLatency += writeLatencyArray;
				
			} else {
				double capBL = lengthCol * 0.2e-15/1e-6;
				int numWriteOperationPerRow = (int)ceil((double)numCol*activityColWrite/numWriteCellPerOperationNeuro);
				double colRamp = 0;
				double tau = (capCol)*(cell.resMemCellAvg/(numRow/2));
				colDelay = horowitz(tau, 0, 1e20, &colRamp);
				colDelay = tau * 0.2 * numColMuxed;  // assume the 15~20% voltage drop is enough for sensing
				
				if (cell.accessType == CMOS_access) {
					wlNewSwitchMatrix.CalculateLatency(1e20, capRow2, resRow, numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite);
				} else {
					wlSwitchMatrix.CalculateLatency(1e20, capRow1, resRow, numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite);
				}
				slSwitchMatrix.CalculateLatency(1e20, capCol, resCol, 0, 2*numWriteOperationPerRow*numRow*activityRowWrite);
				if (numColMuxed > 1) {
					mux.CalculateLatency(colRamp, 0, numColMuxed);
					muxDecoder.CalculateLatency(1e20, mux.capTgGateN*ceil(numCol/numColMuxed), mux.capTgGateP*ceil(numCol/numColMuxed), numColMuxed, 0);
				}
				if (SARADC) {
					sarADC.CalculateLatency(numColMuxed);
				} else {
					multilevelSenseAmp.CalculateLatency(columnResistance, numColMuxed, 1);
					multilevelSAEncoder.CalculateLatency(1e20, numColMuxed);
				}
				if (numCellPerSynapse > 1) {
					shiftAddWeight.CalculateLatency(numColMuxed);	
				}
				if (numReadPulse > 1) {
					shiftAddInput.CalculateLatency(ceil(numColMuxed/numCellPerSynapse));		
				}

				// Read
				readLatency = 0;
				readLatency += MAX(wlNewSwitchMatrix.readLatency + wlSwitchMatrix.readLatency, ( ((numColMuxed > 1)==true? (mux.readLatency+muxDecoder.readLatency):0) )/numReadPulse);
				readLatency += multilevelSenseAmp.readLatency;
				readLatency += multilevelSAEncoder.readLatency;
				readLatency += shiftAddInput.readLatency + shiftAddWeight.readLatency;
				readLatency += colDelay/numReadPulse;
				readLatency += sarADC.readLatency;
				// Write
				
				writeLatency = 0;
				writeLatencyArray = 0;
				writeLatencyArray += totalNumWritePulse * cell.writePulseWidth;
				writeLatency += MAX(wlNewSwitchMatrix.writeLatency + wlSwitchMatrix.writeLatency, slSwitchMatrix.writeLatency);
				writeLatency += writeLatencyArray;
				
			}
		}
	}
}

void SubArray::CalculatePower(const vector<double> &columnResistance, const vector<double> &rowResistance) {
	if (!initialized) {
		cout << "[Subarray] Error: Require initialization first!" << endl;
	} else {
		readDynamicEnergy = 0;
		writeDynamicEnergy = 0;
		readDynamicEnergyArray = 0;
		
		double numReadOperationPerRow;   // average value (can be non-integer for energy calculation)
		if (numCol > numReadCellPerOperationNeuro)
			numReadOperationPerRow = numCol / numReadCellPerOperationNeuro;
		else
			numReadOperationPerRow = 1;

		double numWriteOperationPerRow;   // average value (can be non-integer for energy calculation)
		if (numCol * activityColWrite > numWriteCellPerOperationNeuro)
			numWriteOperationPerRow = numCol * activityColWrite / numWriteCellPerOperationNeuro;
		else
			numWriteOperationPerRow = 1;

		if (cell.memCellType == Type::SRAM) {
			
			// Array leakage (assume 2 INV)
			leakage = 0;
			leakage += CalculateGateLeakage(INV, 1, cell.widthSRAMCellNMOS * tech.featureSize,
					cell.widthSRAMCellPMOS * tech.featureSize, inputParameter.temperature, tech) * tech.vdd * 2;
			leakage *= numRow * numCol;

			if (conventionalSequential) {
				wlDecoder.CalculatePower(numRow*activityRowRead, numRow*activityRowWrite);
				precharger.CalculatePower(numReadOperationPerRow*numRow*activityRowRead, numWriteOperationPerRow*numRow*activityRowWrite);
				sramWriteDriver.CalculatePower(numWriteOperationPerRow*numRow*activityRowWrite);
				adder.CalculatePower(numReadOperationPerRow*numRow*activityRowRead, numReadCellPerOperationNeuro/numCellPerSynapse);				
				dff.CalculatePower(numReadOperationPerRow*numRow*activityRowRead, numReadCellPerOperationNeuro/numCellPerSynapse*(adder.numBit+1));
				senseAmp.CalculatePower(numReadOperationPerRow*numRow*activityRowRead);
				if (numCellPerSynapse > 1) {
					shiftAddWeight.CalculatePower(numCellPerSynapse);	
				}
				if (numReadPulse > 1) {
					shiftAddInput.CalculatePower(1);					
				}
				// Array
				readDynamicEnergyArray = 0; // Just BL discharging
				writeDynamicEnergyArray = cell.capSRAMCell * tech.vdd * tech.vdd * 2 * numCol * activityColWrite * numRow * activityRowWrite;    // flip Q and Q_bar

				// Read
				readDynamicEnergy += wlDecoder.readDynamicEnergy;
				readDynamicEnergy += precharger.readDynamicEnergy;
				readDynamicEnergy += readDynamicEnergyArray;
				readDynamicEnergy += adder.readDynamicEnergy;
				readDynamicEnergy += dff.readDynamicEnergy;
				readDynamicEnergy += senseAmp.readDynamicEnergy;
				readDynamicEnergy += shiftAddWeight.readDynamicEnergy + shiftAddInput.readDynamicEnergy;
				
				readDynamicEnergyADC = precharger.readDynamicEnergy + readDynamicEnergyArray + senseAmp.readDynamicEnergy;
				readDynamicEnergyAccum = adder.readDynamicEnergy + dff.readDynamicEnergy + shiftAddWeight.readDynamicEnergy + shiftAddInput.readDynamicEnergy;
				readDynamicEnergyOther = wlDecoder.readDynamicEnergy;
				
				
				/* Transpose Peripheral for BP */
				if (trainingEstimation) {
					readDynamicEnergyAG = 0;
					if (layerNumber != 0) {
						int numReadOperationPerCol = numRow / numReadCellPerOperationNeuro;
						
						wlDecoderBP.CalculatePower(numCol*activityBPColRead, numRow*activityRowWrite);
						prechargerBP.CalculatePower(numReadOperationPerCol*numCol*activityBPColRead, numWriteOperationPerRow*numRow*activityRowWrite);
						sramWriteDriverBP.CalculatePower(numWriteOperationPerRow*numCol*activityBPColRead);
						adderBP.CalculatePower(numReadOperationPerCol*numCol*activityBPColRead, numReadCellPerOperationNeuro/numCellPerSynapse);				
						dffBP.CalculatePower(numReadOperationPerCol*numCol*activityBPColRead, numReadCellPerOperationNeuro/numCellPerSynapse*(adder.numBit+1));
						senseAmpBP.CalculatePower(numReadOperationPerCol*numCol*activityBPColRead);
						if (numCellPerSynapse > 1) {
							shiftAddBPWeight.CalculatePower(numCellPerSynapse);	
						}
						if (numReadPulseBP > 1) {
							shiftAddBPInput.CalculatePower(1);					
						}

						// Array
						readDynamicEnergyArray = 0; // Just BL discharging
						
						// Read
						readDynamicEnergyAG += wlDecoderBP.readDynamicEnergy;
						readDynamicEnergyAG += prechargerBP.readDynamicEnergy;
						readDynamicEnergyAG += readDynamicEnergyArray;
						readDynamicEnergyAG += adderBP.readDynamicEnergy;
						readDynamicEnergyAG += dffBP.readDynamicEnergy;
						readDynamicEnergyAG += senseAmpBP.readDynamicEnergy;
						readDynamicEnergyAG += shiftAddBPInput.readDynamicEnergy + shiftAddBPWeight.readDynamicEnergy;
						
						readDynamicEnergyADC += prechargerBP.readDynamicEnergy + readDynamicEnergyArray + senseAmpBP.readDynamicEnergy;
						readDynamicEnergyAccum += adderBP.readDynamicEnergy + dffBP.readDynamicEnergy + shiftAddBPInput.readDynamicEnergy + shiftAddBPWeight.readDynamicEnergy;
						readDynamicEnergyOther += wlDecoderBP.readDynamicEnergy;
						
						leakage += wlDecoderBP.leakage;
						leakage += prechargerBP.leakage;
						leakage += adderBP.leakage;
						leakage += dffBP.leakage;
						leakage += senseAmpBP.leakage;
						leakage += shiftAddBPInput.leakage + shiftAddBPWeight.leakage;
					}
				}
				
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
				leakage += shiftAddInput.leakage + shiftAddWeight.leakage;
			} else if (conventionalParallel) {
				wlSwitchMatrix.CalculatePower(numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite, activityRowRead, activityColWrite);
				precharger.CalculatePower(numColMuxed, numWriteOperationPerRow*numRow*activityRowWrite);
				sramWriteDriver.CalculatePower(numWriteOperationPerRow*numRow*activityRowWrite);
				if (numColMuxed > 1) {
					mux.CalculatePower(numColMuxed);	// Mux still consumes energy during row-by-row read
					muxDecoder.CalculatePower(numColMuxed, 1);
				}
				if (SARADC) {
					sarADC.CalculatePower(columnResistance, 1);
				} else {
					multilevelSenseAmp.CalculatePower(columnResistance, 1);
					multilevelSAEncoder.CalculatePower(numColMuxed);
				}
				if (numCellPerSynapse > 1) {
					shiftAddWeight.CalculatePower(numColMuxed);	
				}	
				if (numReadPulse > 1) {
					shiftAddInput.CalculatePower(ceil(numColMuxed/numCellPerSynapse));		
				}
				// Array
				readDynamicEnergyArray = 0; // Just BL discharging
				writeDynamicEnergyArray = cell.capSRAMCell * tech.vdd * tech.vdd * 2 * numCol * activityColWrite * numRow * activityRowWrite;    // flip Q and Q_bar
				// Read
				readDynamicEnergy += wlSwitchMatrix.readDynamicEnergy;
				readDynamicEnergy += precharger.readDynamicEnergy;
				readDynamicEnergy += readDynamicEnergyArray;
				readDynamicEnergy += multilevelSenseAmp.readDynamicEnergy;
				readDynamicEnergy += multilevelSAEncoder.readDynamicEnergy;
				readDynamicEnergy += ((numColMuxed > 1)==true? (mux.readDynamicEnergy/numReadPulse):0);
				readDynamicEnergy += ((numColMuxed > 1)==true? (muxDecoder.readDynamicEnergy/numReadPulse):0);
				readDynamicEnergy += shiftAddWeight.readDynamicEnergy + shiftAddInput.readDynamicEnergy;
				readDynamicEnergy += sarADC.readDynamicEnergy;
 
				readDynamicEnergyADC = precharger.readDynamicEnergy + readDynamicEnergyArray + multilevelSenseAmp.readDynamicEnergy + multilevelSAEncoder.readDynamicEnergy + sarADC.readDynamicEnergy;
				readDynamicEnergyAccum = shiftAddWeight.readDynamicEnergy + shiftAddInput.readDynamicEnergy;
				readDynamicEnergyOther = wlSwitchMatrix.readDynamicEnergy + ( ((numColMuxed > 1)==true? (mux.readDynamicEnergy + muxDecoder.readDynamicEnergy):0) )/numReadPulse;
				
				/* Transpose Peripheral for BP */
				if (trainingEstimation) {
					readDynamicEnergyAG = 0;
					if (layerNumber != 0) {
						if (parallelBP) {
							wlSwitchMatrixBP.CalculatePower(numRowMuxedBP, 2*numWriteOperationPerRow*numRow*activityRowWrite, activityRowRead, activityColWrite);
							prechargerBP.CalculatePower(numRowMuxedBP, numWriteOperationPerRow*numRow*activityRowWrite);
							sramWriteDriverBP.CalculatePower(numWriteOperationPerRow*numRow*activityRowWrite);
							if (numRowMuxedBP>1) {
								muxBP.CalculatePower(numRowMuxedBP);	// Mux still consumes energy during row-by-row read
								muxDecoderBP.CalculatePower(numRowMuxedBP, 1);
							}
							if (SARADC) {
								sarADCBP.CalculatePower(columnResistance, 1);
							} else {
								multilevelSenseAmpBP.CalculatePower(columnResistance, 1);
								multilevelSAEncoderBP.CalculatePower(numRowMuxedBP);
							}
							if (numCellPerSynapse > 1) {
								shiftAddBPWeight.CalculatePower(numRowMuxedBP);	
							}	
							if (numReadPulseBP > 1) {
								shiftAddBPInput.CalculatePower(ceil(numRowMuxedBP/numCellPerSynapse));		
							}
							// Array
							readDynamicEnergyArray = 0; // Just BL discharging
							
							// Read
							readDynamicEnergyAG += wlSwitchMatrixBP.readDynamicEnergy;
							readDynamicEnergyAG += prechargerBP.readDynamicEnergy;
							readDynamicEnergyAG += readDynamicEnergyArray;
							readDynamicEnergyAG += multilevelSenseAmpBP.readDynamicEnergy;
							readDynamicEnergyAG += multilevelSAEncoderBP.readDynamicEnergy;
							readDynamicEnergyAG += ((numRowMuxedBP > 1)==true? (muxBP.readDynamicEnergy/numReadPulseBP):0);
							readDynamicEnergyAG += ((numRowMuxedBP > 1)==true? (muxDecoderBP.readDynamicEnergy/numReadPulseBP):0);
							readDynamicEnergyAG += shiftAddBPInput.readDynamicEnergy + shiftAddBPWeight.readDynamicEnergy;
							readDynamicEnergyAG += sarADCBP.readDynamicEnergy;

							readDynamicEnergyADC += prechargerBP.readDynamicEnergy + readDynamicEnergyArray + multilevelSenseAmpBP.readDynamicEnergy + multilevelSAEncoderBP.readDynamicEnergy + sarADCBP.readDynamicEnergy;
							readDynamicEnergyAccum += shiftAddBPInput.readDynamicEnergy + shiftAddBPWeight.readDynamicEnergy;
							readDynamicEnergyOther += wlSwitchMatrixBP.readDynamicEnergy + ( ((numRowMuxedBP > 1)==true? (muxBP.readDynamicEnergy + muxDecoderBP.readDynamicEnergy):0) )/numReadPulseBP;
							
							leakage += wlSwitchMatrixBP.leakage;
							leakage += prechargerBP.leakage;
							leakage += multilevelSenseAmpBP.leakage;
							leakage += multilevelSAEncoderBP.leakage;
							leakage += muxBP.leakage;
							leakage += muxDecoderBP.leakage;
							leakage += shiftAddBPInput.leakage + shiftAddBPWeight.leakage;
						} else {
							int numReadOperationPerCol = numRow / numReadCellPerOperationNeuro;
							
							wlSwitchMatrixBP.CalculatePower(numReadOperationPerCol*numCol*activityBPColRead, 2*numWriteOperationPerRow*numRow*activityRowWrite, activityRowRead, activityColWrite);
							prechargerBP.CalculatePower(numReadOperationPerCol*numCol*activityBPColRead, numWriteOperationPerRow*numRow*activityRowWrite);
							sramWriteDriverBP.CalculatePower(numWriteOperationPerRow*numCol*activityBPColRead);
							adderBP.CalculatePower(numReadOperationPerCol*numCol*activityBPColRead, numReadCellPerOperationNeuro/numCellPerSynapse);				
							dffBP.CalculatePower(numReadOperationPerCol*numCol*activityBPColRead, numReadCellPerOperationNeuro/numCellPerSynapse*(adder.numBit+1));
							senseAmpBP.CalculatePower(numReadOperationPerCol*numCol*activityBPColRead);
							if (numCellPerSynapse > 1) {
								shiftAddBPWeight.CalculatePower(numCellPerSynapse);	
							}
							if (numReadPulseBP > 1) {
								shiftAddBPInput.CalculatePower(1);					
							}
							// Array
							readDynamicEnergyArray = 0; // Just BL discharging
							
							// Read
							readDynamicEnergyAG += wlSwitchMatrixBP.readDynamicEnergy;
							readDynamicEnergyAG += prechargerBP.readDynamicEnergy;
							readDynamicEnergyAG += readDynamicEnergyArray;
							readDynamicEnergyAG += adderBP.readDynamicEnergy;
							readDynamicEnergyAG += dffBP.readDynamicEnergy;
							readDynamicEnergyAG += senseAmpBP.readDynamicEnergy;
							readDynamicEnergyAG += shiftAddBPInput.readDynamicEnergy + shiftAddBPWeight.readDynamicEnergy;
							
							readDynamicEnergyADC += prechargerBP.readDynamicEnergy + readDynamicEnergyArray + senseAmpBP.readDynamicEnergy;
							readDynamicEnergyAccum += adderBP.readDynamicEnergy + dffBP.readDynamicEnergy + shiftAddBPInput.readDynamicEnergy + shiftAddBPWeight.readDynamicEnergy;
							readDynamicEnergyOther += wlSwitchMatrixBP.readDynamicEnergy;
							
							leakage += wlSwitchMatrixBP.leakage;
							leakage += prechargerBP.leakage;
							leakage += adderBP.leakage;
							leakage += dffBP.leakage;
							leakage += senseAmpBP.leakage;
							leakage += shiftAddBPInput.leakage +shiftAddBPWeight.leakage;
						}
					}
				}
				
				// Write
				writeDynamicEnergy += wlSwitchMatrix.writeDynamicEnergy;
				writeDynamicEnergy += precharger.writeDynamicEnergy;
				writeDynamicEnergy += sramWriteDriver.writeDynamicEnergy;
				writeDynamicEnergy += writeDynamicEnergyArray;
				
				
				// Leakage
				leakage += wlSwitchMatrix.leakage;
				leakage += precharger.leakage;
				leakage += sramWriteDriver.leakage;
				leakage += multilevelSenseAmp.leakage;
				leakage += multilevelSAEncoder.leakage;
				leakage += shiftAddInput.leakage + shiftAddWeight.leakage;
			
			} else if (BNNsequentialMode || XNORsequentialMode) {
				wlDecoder.CalculatePower(numRow*activityRowRead, numRow*activityRowWrite);
				precharger.CalculatePower(numReadOperationPerRow*numRow*activityRowRead, numWriteOperationPerRow*numRow*activityRowWrite);
				sramWriteDriver.CalculatePower(numWriteOperationPerRow*numRow*activityRowWrite);
				adder.CalculatePower(numReadOperationPerRow*numRow*activityRowRead, numReadCellPerOperationNeuro/numCellPerSynapse);				
				dff.CalculatePower(numReadOperationPerRow*numRow*activityRowRead, numReadCellPerOperationNeuro/numCellPerSynapse*(adder.numBit+1));
				senseAmp.CalculatePower(numReadOperationPerRow*numRow*activityRowRead);
				
				// Array
				readDynamicEnergyArray = 0; // Just BL discharging
				writeDynamicEnergyArray = cell.capSRAMCell * tech.vdd * tech.vdd * 2 * numCol * activityColWrite * numRow * activityRowWrite;    // flip Q and Q_bar

				// Read
				readDynamicEnergy += wlDecoder.readDynamicEnergy;
				readDynamicEnergy += precharger.readDynamicEnergy;
				readDynamicEnergy += readDynamicEnergyArray;
				readDynamicEnergy += adder.readDynamicEnergy;
				readDynamicEnergy += dff.readDynamicEnergy;
				readDynamicEnergy += senseAmp.readDynamicEnergy;
				
				// Write
				
				writeDynamicEnergy += wlDecoder.writeDynamicEnergy;
				writeDynamicEnergy += precharger.writeDynamicEnergy;
				writeDynamicEnergy += sramWriteDriver.writeDynamicEnergy;
				writeDynamicEnergy += writeDynamicEnergyArray;
				
				
				// Leakage
				leakage += wlDecoder.leakage;
				leakage += precharger.leakage;
				leakage += sramWriteDriver.leakage;
				leakage += senseAmp.leakage;
				leakage += dff.leakage;
				leakage += adder.leakage;
				
			} else if (BNNparallelMode || XNORparallelMode) {
				wlSwitchMatrix.CalculatePower(numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite, activityRowRead, activityColWrite);
				precharger.CalculatePower(numColMuxed, numWriteOperationPerRow*numRow*activityRowWrite);
				sramWriteDriver.CalculatePower(numWriteOperationPerRow*numRow*activityRowWrite);
				if (SARADC) {
					sarADC.CalculatePower(columnResistance, 1);
				} else {
					multilevelSenseAmp.CalculatePower(columnResistance, 1);
					multilevelSAEncoder.CalculatePower(numColMuxed);
				}
				
				// Array
				readDynamicEnergyArray = 0; // Just BL discharging
				writeDynamicEnergyArray = cell.capSRAMCell * tech.vdd * tech.vdd * 2 * numCol * activityColWrite * numRow * activityRowWrite;    // flip Q and Q_bar
				// Read
				readDynamicEnergy += wlSwitchMatrix.readDynamicEnergy;
				readDynamicEnergy += precharger.readDynamicEnergy;
				readDynamicEnergy += readDynamicEnergyArray;
				readDynamicEnergy += multilevelSenseAmp.readDynamicEnergy;
				readDynamicEnergy += multilevelSAEncoder.readDynamicEnergy;
				readDynamicEnergy += sarADC.readDynamicEnergy;
				
				// Write				
				writeDynamicEnergy += wlSwitchMatrix.writeDynamicEnergy;
				writeDynamicEnergy += precharger.writeDynamicEnergy;
				writeDynamicEnergy += sramWriteDriver.writeDynamicEnergy;
				writeDynamicEnergy += writeDynamicEnergyArray;
				
				
				// Leakage
				leakage += wlSwitchMatrix.leakage;
				leakage += precharger.leakage;
				leakage += sramWriteDriver.leakage;
				leakage += multilevelSenseAmp.leakage;
				leakage += multilevelSAEncoder.leakage;
				
			} else {
				wlSwitchMatrix.CalculatePower(numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite, activityRowRead, activityColWrite);
				precharger.CalculatePower(numColMuxed, numWriteOperationPerRow*numRow*activityRowWrite);
				sramWriteDriver.CalculatePower(numWriteOperationPerRow*numRow*activityRowWrite);
				if (SARADC) {
					sarADC.CalculatePower(columnResistance, 1);
				} else {
					multilevelSenseAmp.CalculatePower(columnResistance, 1);
					multilevelSAEncoder.CalculatePower(numColMuxed);
				}
				if (numCellPerSynapse > 1) {
					shiftAddWeight.CalculatePower(numColMuxed);	
				}	
				if (numReadPulse > 1) {
					shiftAddInput.CalculatePower(ceil(numColMuxed/numCellPerSynapse));		
				}
				// Array
				readDynamicEnergyArray = 0; // Just BL discharging
				writeDynamicEnergyArray = cell.capSRAMCell * tech.vdd * tech.vdd * 2 * numCol * activityColWrite * numRow * activityRowWrite;    // flip Q and Q_bar
				// Read
				readDynamicEnergy += wlSwitchMatrix.readDynamicEnergy;
				readDynamicEnergy += precharger.readDynamicEnergy;
				readDynamicEnergy += readDynamicEnergyArray;
				readDynamicEnergy += multilevelSenseAmp.readDynamicEnergy;
				readDynamicEnergy += multilevelSAEncoder.readDynamicEnergy;
				readDynamicEnergy += shiftAddInput.readDynamicEnergy + shiftAddWeight.readDynamicEnergy;
				readDynamicEnergy += sarADC.readDynamicEnergy;

				// Write
				writeDynamicEnergy += wlSwitchMatrix.writeDynamicEnergy;
				writeDynamicEnergy += precharger.writeDynamicEnergy;
				writeDynamicEnergy += sramWriteDriver.writeDynamicEnergy;
				writeDynamicEnergy += writeDynamicEnergyArray;
				
				
				// Leakage
				leakage += wlSwitchMatrix.leakage;
				leakage += precharger.leakage;
				leakage += sramWriteDriver.leakage;
				leakage += multilevelSenseAmp.leakage;
				leakage += multilevelSAEncoder.leakage;
				leakage += shiftAddInput.leakage + shiftAddWeight.leakage;
			
			}
			
	    } else if (cell.memCellType == Type::RRAM || cell.memCellType == Type::FeFET) {
			leakage = 0;
			if (conventionalSequential) {
				double numReadCells = (int)ceil((double)numCol/numColMuxed);    // similar parameter as numReadCellPerOperationNeuro, which is for SRAM
				double numWriteCells = (int)ceil((double)numCol/*numWriteColMuxed*/); 
				int numWriteOperationPerRow = (int)ceil((double)numCol*activityColWrite/numWriteCellPerOperationNeuro);
				double capBL = lengthCol * 0.2e-15/1e-6;
				
				wlDecoder.CalculatePower(numRow*activityRowRead*numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite);
				if (cell.accessType == CMOS_access) {
					wlNewDecoderDriver.CalculatePower(numRow*activityRowRead*numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite);
				} else {
					wlDecoderDriver.CalculatePower(numReadCells, numWriteCells, numRow*activityRowRead*numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite);
				}
				slSwitchMatrix.CalculatePower(0, 2*numWriteOperationPerRow*numRow*activityRowWrite, activityRowRead, activityColWrite);
				if (numColMuxed > 1) {
					mux.CalculatePower(numColMuxed);	// Mux still consumes energy during row-by-row read
					muxDecoder.CalculatePower(numColMuxed, 1);
				}
				
				if (SARADC) {
					sarADC.CalculatePower(columnResistance, numRow*activityRowRead);
				} else {
					multilevelSenseAmp.CalculatePower(columnResistance, numRow*activityRowRead);
					if (avgWeightBit > 1) {
						multilevelSAEncoder.CalculatePower(numRow*activityRowRead*numColMuxed);
					}
				}

				adder.CalculatePower(numColMuxed*numRow*activityRowRead, numReadCells);
				dff.CalculatePower(numColMuxed*numRow*activityRowRead, numReadCells*(adder.numBit+1)); 
				if (numCellPerSynapse > 1) {
					shiftAddWeight.CalculatePower(numColMuxed);	
				}
				if (numReadPulse > 1) {
					shiftAddInput.CalculatePower(ceil(numColMuxed/numCellPerSynapse));		
				}
				// Read
				readDynamicEnergyArray = 0;
				readDynamicEnergyArray += capBL * cell.readVoltage * cell.readVoltage * numReadCells; // Selected BLs activityColWrite
				readDynamicEnergyArray += capRow2 * tech.vdd * tech.vdd; // Selected WL
				readDynamicEnergyArray *= numRow * activityRowRead * numColMuxed;

				readDynamicEnergy = 0;
				readDynamicEnergy += wlDecoder.readDynamicEnergy;
				readDynamicEnergy += wlNewDecoderDriver.readDynamicEnergy;
				readDynamicEnergy += wlDecoderDriver.readDynamicEnergy;
				readDynamicEnergy +=  ( ((numColMuxed > 1)==true? (mux.readDynamicEnergy + muxDecoder.readDynamicEnergy):0) )/numReadPulse;
				readDynamicEnergy += adder.readDynamicEnergy;
				readDynamicEnergy += dff.readDynamicEnergy;
				readDynamicEnergy += shiftAddWeight.readDynamicEnergy + shiftAddInput.readDynamicEnergy;
				readDynamicEnergy += readDynamicEnergyArray;
				readDynamicEnergy += sarADC.readDynamicEnergy;
				
				readDynamicEnergyADC = readDynamicEnergyArray + multilevelSenseAmp.readDynamicEnergy + multilevelSAEncoder.readDynamicEnergy + sarADC.readDynamicEnergy;
				readDynamicEnergyAccum = adder.readDynamicEnergy + dff.readDynamicEnergy + shiftAddWeight.readDynamicEnergy + shiftAddInput.readDynamicEnergy;
				readDynamicEnergyOther = wlDecoder.readDynamicEnergy + wlNewDecoderDriver.readDynamicEnergy + wlDecoderDriver.readDynamicEnergy + ( ((numColMuxed > 1)==true? (mux.readDynamicEnergy + muxDecoder.readDynamicEnergy):0) )/numReadPulse;

				/* Transpose Peripheral for BP */
				if (trainingEstimation) {
					readDynamicEnergyAG = 0;
					if (layerNumber != 0) {
						readDynamicEnergyArray = 0;
						readDynamicEnergyArray += capBL * cell.readVoltage * cell.readVoltage * numReadCells; // Selected BLs activityColWrite
						readDynamicEnergyArray += capRow2 * tech.vdd * tech.vdd; // Selected WL
						readDynamicEnergyArray *= numRow * activityBPColRead * numRowMuxedBP;
						
						slSwitchMatrix.CalculatePower(numRowMuxedBP*numCol*activityBPColRead, 2*numWriteOperationPerRow*numRow*activityRowWrite, activityRowRead, activityColWrite);
						if (numRowMuxedBP>1) {
							muxBP.CalculatePower(numRowMuxedBP);	// Mux still consumes energy during row-by-row read
							muxDecoderBP.CalculatePower(numRowMuxedBP, 1);
						}
						
						if (SARADC) {
							sarADCBP.CalculatePower(columnResistance, numCol*activityBPColRead);
						} else {
							multilevelSenseAmpBP.CalculatePower(columnResistance, numCol*activityBPColRead);
							if (avgWeightBit > 1) {
								multilevelSAEncoderBP.CalculatePower(numRowMuxedBP*numCol*activityBPColRead);
							}
						}
						
						dffBP.CalculatePower(numRowMuxedBP*numCol*activityBPColRead, ceil(numRow/numRowMuxedBP)*(adderBP.numBit+1)); 
						adderBP.CalculatePower(numRowMuxedBP*numCol*activityBPColRead, ceil(numRow/numRowMuxedBP));
						
						if (numCellPerSynapse > 1) {
							shiftAddBPWeight.CalculatePower(numRowMuxedBP);	
						}
						if (numReadPulseBP > 1) {
							shiftAddBPInput.CalculatePower(ceil(numRowMuxedBP/numCellPerSynapse));		
						}
						
						readDynamicEnergyAG += slSwitchMatrix.readDynamicEnergy;
						readDynamicEnergyAG += ( ((numRowMuxedBP > 1)==true? (muxBP.readDynamicEnergy + muxDecoderBP.readDynamicEnergy):0) )/numReadPulseBP;
						readDynamicEnergyAG += multilevelSenseAmpBP.readDynamicEnergy;
						readDynamicEnergyAG += multilevelSAEncoderBP.readDynamicEnergy;
						readDynamicEnergyAG += dffBP.readDynamicEnergy;
						readDynamicEnergyAG += adderBP.readDynamicEnergy;
						readDynamicEnergyAG += shiftAddBPInput.readDynamicEnergy + shiftAddBPWeight.readDynamicEnergy;
						readDynamicEnergyAG += readDynamicEnergyArray;
						readDynamicEnergyAG += sarADCBP.readDynamicEnergy;
						
						readDynamicEnergyADC += multilevelSenseAmpBP.readDynamicEnergy + multilevelSAEncoderBP.readDynamicEnergy + readDynamicEnergyArray + sarADCBP.readDynamicEnergy;
						readDynamicEnergyAccum += dffBP.readDynamicEnergy + adderBP.readDynamicEnergy + shiftAddBPInput.readDynamicEnergy + shiftAddBPWeight.readDynamicEnergy;
						readDynamicEnergyOther += slSwitchMatrix.readDynamicEnergy + ( ((numRowMuxedBP > 1)==true? (muxBP.readDynamicEnergy + muxDecoderBP.readDynamicEnergy):0) )/numReadPulseBP;
						
						leakage += slSwitchMatrix.leakage;
						leakage += (muxBP.leakage+muxDecoderBP.leakage);
						leakage += multilevelSenseAmpBP.leakage;
						leakage += multilevelSAEncoderBP.leakage;
						leakage += dffBP.leakage;
						leakage += adderBP.leakage;
						leakage += shiftAddBPInput.leakage + shiftAddBPWeight.leakage;
					}
				}
				
				// Write
				writeDynamicEnergyArray = writeDynamicEnergyArray;
				writeDynamicEnergy = 0;
				writeDynamicEnergy += wlDecoder.writeDynamicEnergy;
				writeDynamicEnergy += wlNewDecoderDriver.writeDynamicEnergy;
				writeDynamicEnergy += wlDecoderDriver.writeDynamicEnergy;
				writeDynamicEnergy += slSwitchMatrix.writeDynamicEnergy;
				writeDynamicEnergy += writeDynamicEnergyArray;
				
				// Leakage
				
				leakage += wlDecoder.leakage;
				leakage += wlDecoderDriver.leakage;
				leakage += wlNewDecoderDriver.leakage;
				leakage += slSwitchMatrix.leakage;
				leakage += mux.leakage;
				leakage += muxDecoder.leakage;
				leakage += multilevelSenseAmp.leakage;
				leakage += multilevelSAEncoder.leakage;
				leakage += dff.leakage;
				leakage += adder.leakage;
				leakage += shiftAddInput.leakage + shiftAddWeight.leakage;
					
			} else if (conventionalParallel) {
				double numReadCells = (int)ceil((double)numCol/numColMuxed);    // similar parameter as numReadCellPerOperationNeuro, which is for SRAM
				int numWriteOperationPerRow = (int)ceil((double)numCol*activityColWrite/numWriteCellPerOperationNeuro);
				double capBL = lengthCol * 0.2e-15/1e-6;
			
				if (cell.accessType == CMOS_access) {
					wlNewSwitchMatrix.CalculatePower(numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite, activityRowRead);
				} else {
					wlSwitchMatrix.CalculatePower(numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite, activityRowRead, activityColWrite);
				}
				slSwitchMatrix.CalculatePower(0, 2*numWriteOperationPerRow*numRow*activityRowWrite, activityRowRead, activityColWrite);
				if (numColMuxed > 1) {
					mux.CalculatePower(numColMuxed);	// Mux still consumes energy during row-by-row read
					muxDecoder.CalculatePower(numColMuxed, 1);
				}
				
				if (SARADC) {
					sarADC.CalculatePower(columnResistance, 1);
				} else {
					multilevelSenseAmp.CalculatePower(columnResistance, 1);
					multilevelSAEncoder.CalculatePower(numColMuxed);
				}
				if (numCellPerSynapse > 1) {
					shiftAddWeight.CalculatePower(numColMuxed);	
				}
				if (numReadPulse > 1) {
					shiftAddInput.CalculatePower(ceil(numColMuxed/numCellPerSynapse));		
				}

				// Read
				readDynamicEnergyArray = 0;
				readDynamicEnergyArray += capBL * cell.readVoltage * cell.readVoltage * numReadCells; // Selected BLs activityColWrite
				readDynamicEnergyArray += capRow2 * tech.vdd * tech.vdd * numRow * activityRowRead; // Selected WL
				readDynamicEnergyArray *= numColMuxed;
				
				readDynamicEnergy = 0;
				readDynamicEnergy += wlNewSwitchMatrix.readDynamicEnergy;
				readDynamicEnergy += wlSwitchMatrix.readDynamicEnergy;
				readDynamicEnergy += ( ((numColMuxed > 1)==true? (mux.readDynamicEnergy + muxDecoder.readDynamicEnergy):0) )/numReadPulse;
				readDynamicEnergy += multilevelSenseAmp.readDynamicEnergy;
				readDynamicEnergy += multilevelSAEncoder.readDynamicEnergy;
				readDynamicEnergy += shiftAddWeight.readDynamicEnergy + shiftAddInput.readDynamicEnergy;
				readDynamicEnergy += readDynamicEnergyArray;
				readDynamicEnergy += sarADC.readDynamicEnergy;
				
				readDynamicEnergyADC = readDynamicEnergyArray + multilevelSenseAmp.readDynamicEnergy + multilevelSAEncoder.readDynamicEnergy + sarADC.readDynamicEnergy;
				readDynamicEnergyAccum = shiftAddWeight.readDynamicEnergy + shiftAddInput.readDynamicEnergy;
				readDynamicEnergyOther = wlNewSwitchMatrix.readDynamicEnergy + wlSwitchMatrix.readDynamicEnergy + ( ((numColMuxed > 1)==true? (mux.readDynamicEnergy + muxDecoder.readDynamicEnergy):0) )/numReadPulse;
				
				/* Transpose Peripheral for BP */
				if (trainingEstimation) {
					readDynamicEnergyAG = 0;
					if (layerNumber != 0) {
						if (parallelBP) {
							readDynamicEnergyArray = 0;
							readDynamicEnergyArray += capBL * cell.readVoltage * cell.readVoltage * numReadCells; // Selected BLs activityColWrite
							readDynamicEnergyArray += capRow2 * tech.vdd * tech.vdd * numRow * activityBPColRead; // Selected WL
							readDynamicEnergyArray *= numRowMuxedBP;
							
							slSwitchMatrix.CalculatePower(numRowMuxedBP, 2*numWriteOperationPerRow*numRow*activityRowWrite, activityRowRead, activityColWrite);
							
							if (numRowMuxedBP>1) {
								muxBP.CalculatePower(numRowMuxedBP);	// Mux still consumes energy during row-by-row read
								muxDecoderBP.CalculatePower(numRowMuxedBP, 1);
							}
							
							if (SARADC) {
								sarADCBP.CalculatePower(columnResistance, 1);
							} else {
								multilevelSenseAmpBP.CalculatePower(columnResistance, 1);
								multilevelSAEncoderBP.CalculatePower(numRowMuxedBP);
							}
							
							if (numCellPerSynapse > 1) {
								shiftAddBPWeight.CalculatePower(numRowMuxedBP);	
							}
							if (numReadPulseBP > 1) {
								shiftAddBPInput.CalculatePower(ceil(numRowMuxedBP/numCellPerSynapse));		
							}
							
							readDynamicEnergyAG += slSwitchMatrix.readDynamicEnergy;
							readDynamicEnergyAG +=( ((numRowMuxedBP > 1)==true? (muxBP.readDynamicEnergy + muxDecoderBP.readDynamicEnergy):0) )/numReadPulseBP;
							readDynamicEnergyAG += multilevelSenseAmpBP.readDynamicEnergy;
							readDynamicEnergyAG += multilevelSAEncoderBP.readDynamicEnergy;
							readDynamicEnergyAG += shiftAddBPInput.readDynamicEnergy + shiftAddBPWeight.readDynamicEnergy;
							readDynamicEnergyAG += readDynamicEnergyArray;
							readDynamicEnergyAG += sarADCBP.readDynamicEnergy;
							
							readDynamicEnergyADC += multilevelSenseAmpBP.readDynamicEnergy + multilevelSAEncoderBP.readDynamicEnergy + readDynamicEnergyArray + sarADCBP.readDynamicEnergy;
							readDynamicEnergyAccum += shiftAddBPInput.readDynamicEnergy + shiftAddBPWeight.readDynamicEnergy;
							readDynamicEnergyOther += slSwitchMatrix.readDynamicEnergy + ( ((numRowMuxedBP > 1)==true? (muxBP.readDynamicEnergy + muxDecoderBP.readDynamicEnergy):0) )/numReadPulseBP;
							
							leakage += slSwitchMatrix.leakage;
							leakage += (muxBP.leakage+muxDecoderBP.leakage);
							leakage += multilevelSenseAmpBP.leakage;
							leakage += multilevelSAEncoderBP.leakage;
							leakage += shiftAddBPInput.leakage + shiftAddBPWeight.leakage;
						} else {
							readDynamicEnergyArray = 0;
							readDynamicEnergyArray += capBL * cell.readVoltage * cell.readVoltage * numReadCells; // Selected BLs activityColWrite
							readDynamicEnergyArray += capRow2 * tech.vdd * tech.vdd; // Selected WL
							readDynamicEnergyArray *= numRow * activityBPColRead * numRowMuxedBP;
							
							slSwitchMatrix.CalculatePower(numRowMuxedBP*numCol*activityBPColRead, 2*numWriteOperationPerRow*numRow*activityRowWrite, activityRowRead, activityColWrite);
							
							if (numRowMuxedBP>1) {
								muxBP.CalculatePower(numRowMuxedBP);	// Mux still consumes energy during row-by-row read
								muxDecoderBP.CalculatePower(numRowMuxedBP, 1);
							}
							
							if (SARADC) {
								sarADCBP.CalculatePower(columnResistance, numCol*activityBPColRead);
							} else {
								multilevelSenseAmpBP.CalculatePower(columnResistance, numCol*activityBPColRead);
								if (avgWeightBit > 1) {
									multilevelSAEncoderBP.CalculatePower(numRowMuxedBP*numCol*activityBPColRead);
								}
							}

							dffBP.CalculatePower(numRowMuxedBP*numCol*activityBPColRead, ceil(numRow/numRowMuxedBP)*(adderBP.numBit+1)); 
							adderBP.CalculatePower(numRowMuxedBP*numCol*activityBPColRead, ceil(numRow/numRowMuxedBP));
							
							if (numCellPerSynapse > 1) {
								shiftAddBPWeight.CalculatePower(numRowMuxedBP);	
							}
							if (numReadPulseBP > 1) {
								shiftAddBPInput.CalculatePower(ceil(numRowMuxedBP/numCellPerSynapse));		
							}
							
							readDynamicEnergyAG += slSwitchMatrix.readDynamicEnergy;
							readDynamicEnergyAG += ( ((numRowMuxedBP > 1)==true? (muxBP.readDynamicEnergy + muxDecoderBP.readDynamicEnergy):0) )/numReadPulseBP;
							readDynamicEnergyAG += multilevelSenseAmpBP.readDynamicEnergy;
							readDynamicEnergyAG += multilevelSAEncoderBP.readDynamicEnergy;
							readDynamicEnergyAG += dffBP.readDynamicEnergy;
							readDynamicEnergyAG += adderBP.readDynamicEnergy;
							readDynamicEnergyAG += shiftAddBPInput.readDynamicEnergy + shiftAddBPWeight.readDynamicEnergy;
							readDynamicEnergyAG += readDynamicEnergyArray;
							readDynamicEnergyAG += sarADCBP.readDynamicEnergy;
							
							readDynamicEnergyADC += multilevelSenseAmpBP.readDynamicEnergy + multilevelSAEncoderBP.readDynamicEnergy + readDynamicEnergyArray + sarADCBP.readDynamicEnergy;
							readDynamicEnergyAccum += dffBP.readDynamicEnergy + adderBP.readDynamicEnergy + shiftAddBPInput.readDynamicEnergy + shiftAddBPWeight.readDynamicEnergy;
							readDynamicEnergyOther += slSwitchMatrix.readDynamicEnergy + ( ((numRowMuxedBP > 1)==true? (muxBP.readDynamicEnergy + muxDecoderBP.readDynamicEnergy):0) )/numReadPulseBP;
							
							leakage += slSwitchMatrix.leakage;
							leakage += (muxBP.leakage+muxDecoderBP.leakage);
							leakage += multilevelSenseAmpBP.leakage;
							leakage += multilevelSAEncoderBP.leakage;
							leakage += dffBP.leakage;
							leakage += adderBP.leakage;
							leakage += shiftAddBPInput.leakage + shiftAddBPWeight.leakage;
						}
					}
				}
				
				// Write
				writeDynamicEnergyArray = writeDynamicEnergyArray;
				writeDynamicEnergy = 0;
				writeDynamicEnergy += wlNewSwitchMatrix.writeDynamicEnergy;
				writeDynamicEnergy += wlSwitchMatrix.writeDynamicEnergy;
				writeDynamicEnergy += slSwitchMatrix.writeDynamicEnergy;
				writeDynamicEnergy += writeDynamicEnergyArray;
				
				// Leakage
				leakage += wlSwitchMatrix.leakage;
				leakage += wlNewSwitchMatrix.leakage;
				leakage += slSwitchMatrix.leakage;
				leakage += mux.leakage;
				leakage += muxDecoder.leakage;
				leakage += multilevelSenseAmp.leakage;
				leakage += multilevelSAEncoder.leakage;
				leakage += shiftAddWeight.leakage + shiftAddInput.leakage;
			} else if (BNNsequentialMode || XNORsequentialMode) {
				double numReadCells = (int)ceil((double)numCol/numColMuxed);    // similar parameter as numReadCellPerOperationNeuro, which is for SRAM
				double numWriteCells = (int)ceil((double)numCol/*numWriteColMuxed*/); 
				int numWriteOperationPerRow = (int)ceil((double)numCol*activityColWrite/numWriteCellPerOperationNeuro);
				double capBL = lengthCol * 0.2e-15/1e-6;
			
				wlDecoder.CalculatePower(numRow*activityRowRead*numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite);
				if (cell.accessType == CMOS_access) {
					wlNewDecoderDriver.CalculatePower(numRow*activityRowRead*numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite);
				} else {
					wlDecoderDriver.CalculatePower(numReadCells, numWriteCells, numRow*activityRowRead*numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite);
				}
				slSwitchMatrix.CalculatePower(0, 2*numWriteOperationPerRow*numRow*activityRowWrite, activityRowRead, activityColWrite);
				if (numColMuxed > 1) {
					mux.CalculatePower(numColMuxed);	// Mux still consumes energy during row-by-row read
					muxDecoder.CalculatePower(numColMuxed, 1);
				}
				rowCurrentSenseAmp.CalculatePower(columnResistance, numRow*activityRowRead);
				adder.CalculatePower(numColMuxed*numRow*activityRowRead, numReadCells);
				dff.CalculatePower(numColMuxed*numRow*activityRowRead, numReadCells*(adder.numBit+1)); 
				
				// Read
				readDynamicEnergyArray = 0;
				readDynamicEnergyArray += capBL * cell.readVoltage * cell.readVoltage * numReadCells; // Selected BLs activityColWrite
				readDynamicEnergyArray += capRow2 * tech.vdd * tech.vdd; // Selected WL
				readDynamicEnergyArray *= numRow * activityRowRead * numColMuxed;

				readDynamicEnergy = 0;
				readDynamicEnergy += wlDecoder.readDynamicEnergy;
				readDynamicEnergy += wlNewDecoderDriver.readDynamicEnergy;
				readDynamicEnergy += wlDecoderDriver.readDynamicEnergy;
				readDynamicEnergy += ( ((numColMuxed > 1)==true? (mux.readDynamicEnergy + muxDecoder.readDynamicEnergy):0) )/numReadPulse;
				readDynamicEnergy += rowCurrentSenseAmp.readDynamicEnergy;
				readDynamicEnergy += adder.readDynamicEnergy;
				readDynamicEnergy += dff.readDynamicEnergy;
				readDynamicEnergy += readDynamicEnergyArray;

				// Write
				
				writeDynamicEnergyArray = writeDynamicEnergyArray;
				writeDynamicEnergy = 0;
				writeDynamicEnergy += wlDecoder.writeDynamicEnergy;
				writeDynamicEnergy += wlNewDecoderDriver.writeDynamicEnergy;
				writeDynamicEnergy += wlDecoderDriver.writeDynamicEnergy;
				writeDynamicEnergy += slSwitchMatrix.writeDynamicEnergy;
				writeDynamicEnergy += writeDynamicEnergyArray;
				
				
				// Leakage
				leakage = 0;
				leakage += wlDecoder.leakage;
				leakage += wlDecoderDriver.leakage;
				leakage += wlNewDecoderDriver.leakage;
				leakage += slSwitchMatrix.leakage;
				leakage += mux.leakage;
				leakage += muxDecoder.leakage;
				leakage += rowCurrentSenseAmp.leakage;
				leakage += dff.leakage;
				leakage += adder.leakage;
				
			} else if (BNNparallelMode || XNORparallelMode) {
				double numReadCells = (int)ceil((double)numCol/numColMuxed);    // similar parameter as numReadCellPerOperationNeuro, which is for SRAM
				int numWriteOperationPerRow = (int)ceil((double)numCol*activityColWrite/numWriteCellPerOperationNeuro);
				double capBL = lengthCol * 0.2e-15/1e-6;
			
				if (cell.accessType == CMOS_access) {
					wlNewSwitchMatrix.CalculatePower(numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite, activityRowRead);
				} else {
					wlSwitchMatrix.CalculatePower(numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite, activityRowRead, activityColWrite);
				}
				slSwitchMatrix.CalculatePower(0, 2*numWriteOperationPerRow*numRow*activityRowWrite, activityRowRead, activityColWrite);
				if (numColMuxed > 1) {
					mux.CalculatePower(numColMuxed);	// Mux still consumes energy during row-by-row read
					muxDecoder.CalculatePower(numColMuxed, 1);
				}
				if (SARADC) {
					sarADC.CalculatePower(columnResistance, 1);
				} else {
					multilevelSenseAmp.CalculatePower(columnResistance, 1);
					multilevelSAEncoder.CalculatePower(numColMuxed);
				}
				
				// Read
				readDynamicEnergyArray = 0;
				readDynamicEnergyArray += capBL * cell.readVoltage * cell.readVoltage * numReadCells; // Selected BLs activityColWrite
				readDynamicEnergyArray += capRow2 * tech.vdd * tech.vdd * numRow * activityRowRead; // Selected WL
				readDynamicEnergyArray *= numColMuxed;

				readDynamicEnergy = 0;
				readDynamicEnergy += wlNewSwitchMatrix.readDynamicEnergy;
				readDynamicEnergy += wlSwitchMatrix.readDynamicEnergy;
				readDynamicEnergy += ( ((numColMuxed > 1)==true? (mux.readDynamicEnergy + muxDecoder.readDynamicEnergy):0) )/numReadPulse;
				readDynamicEnergy += multilevelSenseAmp.readDynamicEnergy;
				readDynamicEnergy += multilevelSAEncoder.readDynamicEnergy;
				readDynamicEnergy += readDynamicEnergyArray;
				readDynamicEnergy += sarADC.readDynamicEnergy;

				// Write
				
				writeDynamicEnergyArray = writeDynamicEnergyArray;
				writeDynamicEnergy = 0;
				writeDynamicEnergy += wlNewSwitchMatrix.writeDynamicEnergy;
				writeDynamicEnergy += wlSwitchMatrix.writeDynamicEnergy;
				writeDynamicEnergy += slSwitchMatrix.writeDynamicEnergy;
				writeDynamicEnergy += writeDynamicEnergyArray;
				
				
				// Leakage
				leakage = 0;
				leakage += wlSwitchMatrix.leakage;
				leakage += wlNewSwitchMatrix.leakage;
				leakage += slSwitchMatrix.leakage;
				leakage += mux.leakage;
				leakage += muxDecoder.leakage;
				leakage += multilevelSenseAmp.leakage;
				leakage += multilevelSAEncoder.leakage;

			} else {
				double numReadCells = (int)ceil((double)numCol/numColMuxed);    // similar parameter as numReadCellPerOperationNeuro, which is for SRAM
				int numWriteOperationPerRow = (int)ceil((double)numCol*activityColWrite/numWriteCellPerOperationNeuro);
				double capBL = lengthCol * 0.2e-15/1e-6;
			
				if (cell.accessType == CMOS_access) {
					wlNewSwitchMatrix.CalculatePower(numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite, activityRowRead);
				} else {
					wlSwitchMatrix.CalculatePower(numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite, activityRowRead, activityColWrite);
				}
				slSwitchMatrix.CalculatePower(0, 2*numWriteOperationPerRow*numRow*activityRowWrite, activityRowRead, activityColWrite);
				if (numColMuxed > 1) {
					mux.CalculatePower(numColMuxed);	// Mux still consumes energy during row-by-row read
					muxDecoder.CalculatePower(numColMuxed, 1);
				}
				if (SARADC) {
					sarADC.CalculatePower(columnResistance, numColMuxed);
				} else {
					multilevelSenseAmp.CalculatePower(columnResistance, numColMuxed);
					multilevelSAEncoder.CalculatePower(numColMuxed);
				}
				if (numCellPerSynapse > 1) {
					shiftAddWeight.CalculatePower(numColMuxed);	
				}
				if (numReadPulse > 1) {
					shiftAddInput.CalculatePower(ceil(numColMuxed/numCellPerSynapse));		
				}
				// Read
				readDynamicEnergyArray = 0;
				readDynamicEnergyArray += capBL * cell.readVoltage * cell.readVoltage * numReadCells; // Selected BLs activityColWrite
				readDynamicEnergyArray += capRow2 * tech.vdd * tech.vdd * numRow * activityRowRead; // Selected WL
				readDynamicEnergyArray *= numReadPulse * numColMuxed;
				
				readDynamicEnergy = 0;
				readDynamicEnergy += wlNewSwitchMatrix.readDynamicEnergy;
				readDynamicEnergy += wlSwitchMatrix.readDynamicEnergy;
				readDynamicEnergy += ( ((numColMuxed > 1)==true? (mux.readDynamicEnergy + muxDecoder.readDynamicEnergy):0) )/numReadPulse;
				readDynamicEnergy += multilevelSenseAmp.readDynamicEnergy;
				readDynamicEnergy += multilevelSAEncoder.readDynamicEnergy;
				readDynamicEnergy += shiftAddInput.readDynamicEnergy + shiftAddWeight.readDynamicEnergy;
				readDynamicEnergy += readDynamicEnergyArray;
				readDynamicEnergy += sarADC.readDynamicEnergy;
				
				// Write
				
				writeDynamicEnergyArray = writeDynamicEnergyArray;
				writeDynamicEnergy = 0;
				writeDynamicEnergy += wlNewSwitchMatrix.writeDynamicEnergy;
				writeDynamicEnergy += wlSwitchMatrix.writeDynamicEnergy;
				writeDynamicEnergy += slSwitchMatrix.writeDynamicEnergy;
				writeDynamicEnergy += writeDynamicEnergyArray;
				
				
				// Leakage
				leakage = 0;
				leakage += wlSwitchMatrix.leakage;
				leakage += wlNewSwitchMatrix.leakage;
				leakage += slSwitchMatrix.leakage;
				leakage += mux.leakage;
				leakage += muxDecoder.leakage;
				leakage += multilevelSenseAmp.leakage;
				leakage += multilevelSAEncoder.leakage;
				leakage += shiftAddInput.leakage + shiftAddWeight.leakage;
			}
		} 
	}
}

void SubArray::PrintProperty() {

	if (cell.memCellType == Type::SRAM) {
		
		cout << endl << endl;
	    cout << "Array:" << endl;
	    cout << "Area = " << heightArray*1e6 << "um x " << widthArray*1e6 << "um = " << areaArray*1e12 << "um^2" << endl;
	    cout << "Read Dynamic Energy = " << readDynamicEnergyArray*1e12 << "pJ" << endl;
	    //cout << "Write Dynamic Energy = " << writeDynamicEnergyArray*1e12 << "pJ" << endl;
		
		precharger.PrintProperty("precharger");
		sramWriteDriver.PrintProperty("sramWriteDriver");
		
		if (conventionalSequential) {
			wlDecoder.PrintProperty("wlDecoder");			
			senseAmp.PrintProperty("senseAmp");
			dff.PrintProperty("dff"); 
			adder.PrintProperty("adder");
			if (numReadPulse > 1) {
				shiftAddWeight.PrintProperty("shiftAddWeight");
				shiftAddInput.PrintProperty("shiftAddInput");		
			}
		} else if (conventionalParallel) {
			wlSwitchMatrix.PrintProperty("wlSwitchMatrix");
			multilevelSenseAmp.PrintProperty("multilevelSenseAmp");
			multilevelSAEncoder.PrintProperty("multilevelSAEncoder");
			if (numReadPulse > 1) {
				shiftAddWeight.PrintProperty("shiftAddWeight");
				shiftAddInput.PrintProperty("shiftAddInput");
			}
		} else if (BNNsequentialMode || XNORsequentialMode) {
			wlDecoder.PrintProperty("wlDecoder");			
			senseAmp.PrintProperty("senseAmp");
			dff.PrintProperty("dff"); 
			adder.PrintProperty("adder");
		} else if (BNNparallelMode || XNORparallelMode) {
			wlSwitchMatrix.PrintProperty("wlSwitchMatrix");
			multilevelSenseAmp.PrintProperty("multilevelSenseAmp");
			multilevelSAEncoder.PrintProperty("multilevelSAEncoder");
		} else {
			wlSwitchMatrix.PrintProperty("wlSwitchMatrix");
			multilevelSenseAmp.PrintProperty("multilevelSenseAmp");
			multilevelSAEncoder.PrintProperty("multilevelSAEncoder");
			if (numReadPulse > 1) {
				shiftAddWeight.PrintProperty("shiftAddWeight");
				shiftAddInput.PrintProperty("shiftAddInput");
			}
		}
		
	} else if (cell.memCellType == Type::RRAM || cell.memCellType == Type::FeFET) {
		
		cout << endl << endl;
	    cout << "Array:" << endl;
	    cout << "Area = " << heightArray*1e6 << "um x " << widthArray*1e6 << "um = " << areaArray*1e12 << "um^2" << endl;
	    cout << "Read Dynamic Energy = " << readDynamicEnergyArray*1e12 << "pJ" << endl;
	    //cout << "Write Dynamic Energy = " << writeDynamicEnergyArray*1e12 << "pJ" << endl;
		//cout << "Write Latency = " << writeLatencyArray*1e9 << "ns" << endl;

		if (conventionalSequential) {
			wlDecoder.PrintProperty("wlDecoder");
			if (cell.accessType == CMOS_access) {
				wlNewDecoderDriver.PrintProperty("wlNewDecoderDriver");
			} else {
				wlDecoderDriver.PrintProperty("wlDecoderDriver");
			} 
			slSwitchMatrix.PrintProperty("slSwitchMatrix");
			mux.PrintProperty("mux");
			muxDecoder.PrintProperty("muxDecoder");
			multilevelSenseAmp.PrintProperty("multilevelSenseAmp or single-bit SenseAmp");
			multilevelSAEncoder.PrintProperty("multilevelSAEncoder");
			adder.PrintProperty("adder");
			dff.PrintProperty("dff");
			if (numReadPulse > 1) {
				shiftAddWeight.PrintProperty("shiftAddWeight");
				shiftAddInput.PrintProperty("shiftAddInput"); 
			}
		} else if (conventionalParallel) {
			if (cell.accessType == CMOS_access) {
				wlNewSwitchMatrix.PrintProperty("wlNewSwitchMatrix");
			} else {
				wlSwitchMatrix.PrintProperty("wlSwitchMatrix");
			}
			slSwitchMatrix.PrintProperty("slSwitchMatrix");
			mux.PrintProperty("mux");
			muxDecoder.PrintProperty("muxDecoder");
			multilevelSenseAmp.PrintProperty("multilevelSenseAmp");
			multilevelSAEncoder.PrintProperty("multilevelSAEncoder");
			if (numReadPulse > 1) {
				shiftAddWeight.PrintProperty("shiftAddWeight");
				shiftAddInput.PrintProperty("shiftAddInput");
			}
		} else if (BNNsequentialMode || XNORsequentialMode) {
			wlDecoder.PrintProperty("wlDecoder");
			if (cell.accessType == CMOS_access) {
				wlNewDecoderDriver.PrintProperty("wlNewDecoderDriver");
			} else {
				wlDecoderDriver.PrintProperty("wlDecoderDriver");
			} 
			slSwitchMatrix.PrintProperty("slSwitchMatrix");
			mux.PrintProperty("mux");
			muxDecoder.PrintProperty("muxDecoder");
			rowCurrentSenseAmp.PrintProperty("currentSenseAmp");
			adder.PrintProperty("adder");
			dff.PrintProperty("dff");
		} else if (BNNparallelMode || XNORparallelMode) {
			if (cell.accessType == CMOS_access) {
				wlNewSwitchMatrix.PrintProperty("wlNewSwitchMatrix");
			} else {
				wlSwitchMatrix.PrintProperty("wlSwitchMatrix");
			}
			slSwitchMatrix.PrintProperty("slSwitchMatrix");
			mux.PrintProperty("mux");
			muxDecoder.PrintProperty("muxDecoder");
			multilevelSenseAmp.PrintProperty("multilevelSenseAmp");
			multilevelSAEncoder.PrintProperty("multilevelSAEncoder");
		} else {
			if (cell.accessType == CMOS_access) {
				wlNewSwitchMatrix.PrintProperty("wlNewSwitchMatrix");
			} else {
				wlSwitchMatrix.PrintProperty("wlSwitchMatrix");
			}
			slSwitchMatrix.PrintProperty("slSwitchMatrix");
			mux.PrintProperty("mux");
			muxDecoder.PrintProperty("muxDecoder");
			multilevelSenseAmp.PrintProperty("multilevelSenseAmp");
			multilevelSAEncoder.PrintProperty("multilevelSAEncoder");
			if (numReadPulse > 1) {
				shiftAddWeight.PrintProperty("shiftAddWeight");
				shiftAddInput.PrintProperty("shiftAddInput");																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																					   
			}
		}
	} 
	FunctionUnit::PrintProperty("SubArray");
	cout << "Used Area = " << usedArea*1e12 << "um^2" << endl;
	cout << "Empty Area = " << emptyArea*1e12 << "um^2" << endl;
}

