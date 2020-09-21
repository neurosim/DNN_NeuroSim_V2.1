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

#include <cstdio>
#include <random>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <vector>
#include <sstream>
#include <chrono>
#include <algorithm>
#include "constant.h"
#include "formula.h"
#include "Param.h"
#include "Tile.h"
#include "Chip.h"
#include "ProcessingUnit.h"
#include "SubArray.h"
#include "Definition.h"

using namespace std;

vector<vector<double> > getNetStructure(const string &inputfile);

int main(int argc, char * argv[]) {   

	auto start = chrono::high_resolution_clock::now();
	
	gen.seed(0);
	
	vector<vector<double> > netStructure;
	netStructure = getNetStructure(argv[2]);
	
	// define weight/input/memory precision from wrapper
	param->synapseBit = atoi(argv[3]);             		 // precision of synapse weight
	param->numBitInput = atoi(argv[4]);            		 // precision of input neural activation
	
	if (param->cellBit > param->synapseBit) {
		cout << "ERROR!: Memory precision is even higher than synapse precision, please modify 'cellBit' in Param.cpp!" << endl;
		param->cellBit = param->synapseBit;
	}
	
	/*** initialize operationMode as default ***/
	param->conventionalParallel = 0;
	param->conventionalSequential = 0;
	param->BNNparallelMode = 0;                // parallel BNN
	param->BNNsequentialMode = 0;              // sequential BNN
	param->XNORsequentialMode = 0;           // Use several multi-bit RRAM as one synapse
	param->XNORparallelMode = 0;         // Use several multi-bit RRAM as one synapse
	switch(param->operationmode) {
		case 6:	    param->XNORparallelMode = 1;               break;     
		case 5:	    param->XNORsequentialMode = 1;             break;     
		case 4:	    param->BNNparallelMode = 1;                break;     
		case 3:	    param->BNNsequentialMode = 1;              break;    
		case 2:	    param->conventionalParallel = 1;           break;     
		case 1:	    param->conventionalSequential = 1;         break;    
		case -1:	break;
		default:	exit(-1);
	}
	
	if (param->XNORparallelMode || param->XNORsequentialMode) {
		param->numRowPerSynapse = 2;
	} else {
		param->numRowPerSynapse = 1;
	}
	if (param->BNNparallelMode) {
		param->numColPerSynapse = 2;
	} else if (param->XNORparallelMode || param->XNORsequentialMode || param->BNNsequentialMode) {
		param->numColPerSynapse = 1;
	} else {
		param->numColPerSynapse = ceil((double)param->synapseBit/(double)param->cellBit); 
	}
	
	switch(param->transistortype) {
		case 3:	    inputParameter.transistorType = TFET;          break;
		case 2:	    inputParameter.transistorType = FET_2D;        break;
		case 1:	    inputParameter.transistorType = conventional;  break;
		case -1:	break;
		default:	exit(-1);
	}
	
	switch(param->deviceroadmap) {
		case 2:	    inputParameter.deviceRoadmap = LSTP;  break;
		case 1:	    inputParameter.deviceRoadmap = HP;    break;
		case -1:	break;
		default:	exit(-1);
	}
	
	/* Create SubArray object and link the required global objects (not initialization) */
	inputParameter.temperature = param->temp;   // Temperature (K)
	inputParameter.processNode = param->technode;    // Technology node
	tech.Initialize(inputParameter.processNode, inputParameter.deviceRoadmap, inputParameter.transistorType);

	double maxPESizeNM, maxTileSizeCM, numPENM;
	vector<int> markNM;
	vector<int> pipelineSpeedUp;
	markNM = ChipDesignInitialize(inputParameter, tech, cell, false, netStructure, &maxPESizeNM, &maxTileSizeCM, &numPENM);
	pipelineSpeedUp = ChipDesignInitialize(inputParameter, tech, cell, true, netStructure, &maxPESizeNM, &maxTileSizeCM, &numPENM);
	
	double desiredNumTileNM, desiredPESizeNM, desiredNumTileCM, desiredTileSizeCM, desiredPESizeCM;
	int numTileRow, numTileCol;
	int numArrayWriteParallel;
	
	vector<vector<double> > numTileEachLayer;
	vector<vector<double> > utilizationEachLayer;
	vector<vector<double> > speedUpEachLayer;
	vector<vector<double> > tileLocaEachLayer;
	
	numTileEachLayer = ChipFloorPlan(true, false, false, netStructure, markNM, 
					maxPESizeNM, maxTileSizeCM, numPENM, pipelineSpeedUp,
					&desiredNumTileNM, &desiredPESizeNM, &desiredNumTileCM, &desiredTileSizeCM, &desiredPESizeCM, &numTileRow, &numTileCol);	
	
	utilizationEachLayer = ChipFloorPlan(false, true, false, netStructure, markNM, 
					maxPESizeNM, maxTileSizeCM, numPENM, pipelineSpeedUp,
					&desiredNumTileNM, &desiredPESizeNM, &desiredNumTileCM, &desiredTileSizeCM, &desiredPESizeCM, &numTileRow, &numTileCol);
	
	speedUpEachLayer = ChipFloorPlan(false, false, true, netStructure, markNM,
					maxPESizeNM, maxTileSizeCM, numPENM, pipelineSpeedUp,
					&desiredNumTileNM, &desiredPESizeNM, &desiredNumTileCM, &desiredTileSizeCM, &desiredPESizeCM, &numTileRow, &numTileCol);
					
	tileLocaEachLayer = ChipFloorPlan(false, false, false, netStructure, markNM,
					maxPESizeNM, maxTileSizeCM, numPENM, pipelineSpeedUp,
					&desiredNumTileNM, &desiredPESizeNM, &desiredNumTileCM, &desiredTileSizeCM, &desiredPESizeCM, &numTileRow, &numTileCol);
	
	cout << "------------------------------ FloorPlan --------------------------------" <<  endl;
	cout << endl;
	cout << "Tile and PE size are optimized to maximize memory utilization ( = memory mapped by synapse / total memory on chip)" << endl;
	cout << endl;
	if (!param->novelMapping) {
		cout << "Desired Conventional Mapped Tile Storage Size: " << desiredTileSizeCM << "x" << desiredTileSizeCM << endl;
		cout << "Desired Conventional PE Storage Size: " << desiredPESizeCM << "x" << desiredPESizeCM << endl;
	} else {
		cout << "Desired Conventional Mapped Tile Storage Size: " << desiredTileSizeCM << "x" << desiredTileSizeCM << endl;
		cout << "Desired Conventional PE Storage Size: " << desiredPESizeCM << "x" << desiredPESizeCM << endl;
		cout << "Desired Novel Mapped Tile Storage Size: " << numPENM << "x" << desiredPESizeNM << "x" << desiredPESizeNM << endl;
	}
	cout << "User-defined SubArray Size: " << param->numRowSubArray << "x" << param->numColSubArray << endl;
	cout << endl;
	cout << "----------------- # of tile used for each layer -----------------" <<  endl;
	double totalNumTile = 0;
	for (int i=0; i<netStructure.size(); i++) {
		cout << "layer" << i+1 << ": " << numTileEachLayer[0][i] * numTileEachLayer[1][i] << endl;
		totalNumTile += numTileEachLayer[0][i] * numTileEachLayer[1][i];
	}
	cout << endl;

	cout << "----------------- Speed-up of each layer ------------------" <<  endl;
	for (int i=0; i<netStructure.size(); i++) {
		cout << "layer" << i+1 << ": " << speedUpEachLayer[0][i] * speedUpEachLayer[1][i] << endl;
	}
	cout << endl;
	
	cout << "----------------- Utilization of each layer ------------------" <<  endl;
	double realMappedMemory = 0;
	for (int i=0; i<netStructure.size(); i++) {
		cout << "layer" << i+1 << ": " << utilizationEachLayer[i][0] << endl;
		realMappedMemory += numTileEachLayer[0][i] * numTileEachLayer[1][i] * utilizationEachLayer[i][0];
	}
	cout << "Memory Utilization of Whole Chip: " << realMappedMemory/totalNumTile*100 << " % " << endl;
	cout << endl;
	cout << "---------------------------- FloorPlan Done ------------------------------" <<  endl;
	cout << endl;
	cout << endl;
	cout << endl;
	
	double numComputation = 0;
	for (int i=0; i<netStructure.size(); i++) {
		numComputation += 2*(netStructure[i][0] * netStructure[i][1] * netStructure[i][2] * netStructure[i][3] * netStructure[i][4] * netStructure[i][5]);
	}
	
	if (param->trainingEstimation) {
		numComputation *= 3;  // forward, computation of activation gradient, weight gradient
		numComputation -= 2*(netStructure[0][0] * netStructure[0][1] * netStructure[0][2] * netStructure[0][3] * netStructure[0][4] * netStructure[0][5]);  //L-1 does not need AG
		numComputation *= param->batchSize * param->numIteration;  // count for one epoch
	}
		
	ChipInitialize(inputParameter, tech, cell, netStructure, markNM, numTileEachLayer,
					numPENM, desiredNumTileNM, desiredPESizeNM, desiredNumTileCM, desiredTileSizeCM, desiredPESizeCM, numTileRow, numTileCol, &numArrayWriteParallel);
	
	double chipHeight, chipWidth, chipArea, chipAreaIC, chipAreaADC, chipAreaAccum, chipAreaOther, chipAreaWG, chipAreaArray;
	double CMTileheight = 0;
	double CMTilewidth = 0;
	double NMTileheight = 0;
	double NMTilewidth = 0;
	vector<double> chipAreaResults;
				
	chipAreaResults = ChipCalculateArea(inputParameter, tech, cell, desiredNumTileNM, numPENM, desiredPESizeNM, desiredNumTileCM, desiredTileSizeCM, desiredPESizeCM, numTileRow, 
					&chipHeight, &chipWidth, &CMTileheight, &CMTilewidth, &NMTileheight, &NMTilewidth);		
	chipArea = chipAreaResults[0];
	chipAreaIC = chipAreaResults[1];
	chipAreaADC = chipAreaResults[2];
	chipAreaAccum = chipAreaResults[3];
	chipAreaOther = chipAreaResults[4];
	chipAreaWG = chipAreaResults[5];
	chipAreaArray = chipAreaResults[6];

	double chipReadLatency = 0;
	double chipReadDynamicEnergy = 0;
	double chipReadLatencyAG = 0;
	double chipReadDynamicEnergyAG = 0;
	double chipReadLatencyWG = 0;
	double chipReadDynamicEnergyWG = 0;
	double chipWriteLatencyWU = 0;
	double chipWriteDynamicEnergyWU = 0;
	
	double chipReadLatencyPeakFW = 0;
	double chipReadDynamicEnergyPeakFW = 0;
	double chipReadLatencyPeakAG = 0;
	double chipReadDynamicEnergyPeakAG = 0;
	double chipReadLatencyPeakWG = 0;
	double chipReadDynamicEnergyPeakWG = 0;
	double chipWriteLatencyPeakWU = 0;
	double chipWriteDynamicEnergyPeakWU = 0;
	
	double chipLeakageEnergy = 0;
	double chipLeakage = 0;
	double chipbufferLatency = 0;
	double chipbufferReadDynamicEnergy = 0;
	double chipicLatency = 0;
	double chipicReadDynamicEnergy = 0;
	
	double chipLatencyADC = 0;
	double chipLatencyAccum = 0;
	double chipLatencyOther = 0;
	double chipEnergyADC = 0;
	double chipEnergyAccum = 0;
	double chipEnergyOther = 0;
	
	double chipDRAMLatency = 0;
	double chipDRAMDynamicEnergy = 0;
	
	double layerReadLatency = 0;
	double layerReadDynamicEnergy = 0;
	double layerReadLatencyAG = 0;
	double layerReadDynamicEnergyAG = 0;
	double layerReadLatencyWG = 0;
	double layerReadDynamicEnergyWG = 0;
	double layerWriteLatencyWU = 0;
	double layerWriteDynamicEnergyWU = 0;
	
	double layerReadLatencyPeakFW = 0;
	double layerReadDynamicEnergyPeakFW = 0;
	double layerReadLatencyPeakAG = 0;
	double layerReadDynamicEnergyPeakAG = 0;
	double layerReadLatencyPeakWG = 0;
	double layerReadDynamicEnergyPeakWG = 0;
	double layerWriteLatencyPeakWU = 0;
	double layerWriteDynamicEnergyPeakWU = 0;
	
	double layerDRAMLatency = 0;
	double layerDRAMDynamicEnergy = 0;
	
	double tileLeakage = 0;
	double layerbufferLatency = 0;
	double layerbufferDynamicEnergy = 0;
	double layericLatency = 0;
	double layericDynamicEnergy = 0;
	
	double coreLatencyADC = 0;
	double coreLatencyAccum = 0;
	double coreLatencyOther = 0;
	double coreEnergyADC = 0;
	double coreEnergyAccum = 0;
	double coreEnergyOther = 0;
	
	
	cout << "-------------------------------------- Hardware Performance --------------------------------------" <<  endl;
	
	// save breakdown results of each layer to csv files
	ofstream breakdownfile;
	string breakdownfile_name = "./NeuroSim_Results_Each_Epoch/NeuroSim_Breakdown_Epock_";
	breakdownfile_name.append(argv[1]);
	breakdownfile_name.append(".csv");
	breakdownfile.open (breakdownfile_name, ios::app);
	if (breakdownfile.is_open()) {
		// firstly save the area results to file
		breakdownfile << "Total Area(m^2), Total CIM (FW+AG) Area (m^2), Routing Area(m^2), ADC Area(m^2), Accumulation Area(m^2), Other Logic&Storage Area(m^2), Weight Gradient Area(m^2),"<< endl;
		breakdownfile << chipArea << "," << chipAreaArray << "," << chipAreaIC << "," << chipAreaADC << "," << chipAreaAccum << "," << chipAreaOther << "," << chipAreaWG << endl;
		breakdownfile << endl;
		breakdownfile << endl;
		breakdownfile << "layer_number, latency_FW(s), latency_AG(s), latency_WG(s), latency_WU(s), energy_FW(J), energy_AG(J), energy_WG(J), energy_WU(J),";
		breakdownfile << "Peak_latency_FW(s), Peak_latency_AG(s), Peak_latency_WG(s), Peak_latency_WU(s), Peak_energy_FW(J), Peak_energy_AG(J), Peak_energy_WG(J), Peak_energy_WU(J),";
		breakdownfile << ", , ADC_latency(s), Accumulation_latency(s), Synaptic Array w/o ADC_latency(s), Buffer_latency(s), IC_latency(s), Weight_gradient_latency(s), Weight_update(s), DRAM_latency(s), ";
		breakdownfile << "ADC_energy(J), Accumulation_energy(J), Synaptic Array w/o ADC_energy(J), Buffer_energy(J), IC_energy(J), Weight_gradient_energy(J), Weight_update_energy(J), DRAM_energy(J)" << endl;
	} else {
		cout << "Error: the breakdown file cannot be opened!" << endl;
	}
	
	if (! param->pipeline) {
		// layer-by-layer process
		// show the detailed hardware performance for each layer
		for (int i=0; i<netStructure.size(); i++) {
			cout << "-------------------- Estimation of Layer " << i+1 << " ----------------------" << endl;
			
			param->activityRowReadWG = atof(argv[4*i+8]);
                        param->activityRowWriteWG = atof(argv[4*i+8]);
                        param->activityColWriteWG = atof(argv[4*i+8]);
			
			ChipCalculatePerformance(inputParameter, tech, cell, i, argv[4*i+5], argv[4*i+6], argv[4*i+7], netStructure[i][6],
						netStructure, markNM, numTileEachLayer, utilizationEachLayer, speedUpEachLayer, tileLocaEachLayer,
						numPENM, desiredPESizeNM, desiredTileSizeCM, desiredPESizeCM, CMTileheight, CMTilewidth, NMTileheight, NMTilewidth, numArrayWriteParallel,
						&layerReadLatency, &layerReadDynamicEnergy, &tileLeakage, &layerReadLatencyAG, &layerReadDynamicEnergyAG, &layerReadLatencyWG, &layerReadDynamicEnergyWG, 
						&layerWriteLatencyWU, &layerWriteDynamicEnergyWU, &layerbufferLatency, &layerbufferDynamicEnergy, &layericLatency, &layericDynamicEnergy,
						&coreLatencyADC, &coreLatencyAccum, &coreLatencyOther, &coreEnergyADC, &coreEnergyAccum, &coreEnergyOther, &layerDRAMLatency, &layerDRAMDynamicEnergy,
						&layerReadLatencyPeakFW, &layerReadDynamicEnergyPeakFW, &layerReadLatencyPeakAG, &layerReadDynamicEnergyPeakAG,
						&layerReadLatencyPeakWG, &layerReadDynamicEnergyPeakWG, &layerWriteLatencyPeakWU, &layerWriteDynamicEnergyPeakWU);
			
			double numTileOtherLayer = 0;
			double layerLeakageEnergy = 0;		
			for (int j=0; j<netStructure.size(); j++) {
				if (j != i) {
					numTileOtherLayer += numTileEachLayer[0][j] * numTileEachLayer[1][j];
				}
			}
			layerLeakageEnergy = numTileOtherLayer*tileLeakage*(layerReadLatency+layerReadLatencyAG);
			
			cout << "layer" << i+1 << "'s readLatency of Forward is: " << layerReadLatency*1e9 << "ns" << endl;
			cout << "layer" << i+1 << "'s readDynamicEnergy of Forward is: " << layerReadDynamicEnergy*1e12 << "pJ" << endl;
			cout << "layer" << i+1 << "'s readLatency of Activation Gradient is: " << layerReadLatencyAG*1e9 << "ns" << endl;
			cout << "layer" << i+1 << "'s readDynamicEnergy of Activation Gradient is: " << layerReadDynamicEnergyAG*1e12 << "pJ" << endl;
			cout << "layer" << i+1 << "'s readLatency of Weight Gradient is: " << layerReadLatencyWG*1e9 << "ns" << endl;
			cout << "layer" << i+1 << "'s readDynamicEnergy of Weight Gradient is: " << layerReadDynamicEnergyWG*1e12 << "pJ" << endl;
			cout << "layer" << i+1 << "'s writeLatency of Weight Update is: " << layerWriteLatencyWU*1e9 << "ns" << endl;
			cout << "layer" << i+1 << "'s writeDynamicEnergy of Weight Update is: " << layerWriteDynamicEnergyWU*1e12 << "pJ" << endl;
			cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
			cout << "layer" << i+1 << "'s PEAK readLatency of Forward is: " << layerReadLatencyPeakFW*1e9 << "ns" << endl;
			cout << "layer" << i+1 << "'s PEAK readDynamicEnergy of Forward is: " << layerReadDynamicEnergyPeakFW*1e12 << "pJ" << endl;
			cout << "layer" << i+1 << "'s PEAK readLatency of Activation Gradient is: " << layerReadLatencyPeakAG*1e9 << "ns" << endl;
			cout << "layer" << i+1 << "'s PEAK readDynamicEnergy of Activation Gradient is: " << layerReadDynamicEnergyPeakAG*1e12 << "pJ" << endl;
			cout << "layer" << i+1 << "'s PEAK readLatency of Weight Gradient is: " << layerReadLatencyPeakWG*1e9 << "ns" << endl;
			cout << "layer" << i+1 << "'s PEAK readDynamicEnergy of Weight Gradient is: " << layerReadDynamicEnergyPeakWG*1e12 << "pJ" << endl;
			cout << "layer" << i+1 << "'s PEAK writeLatency of Weight Update is: " << layerWriteLatencyPeakWU*1e9 << "ns" << endl;
			cout << "layer" << i+1 << "'s PEAK writeDynamicEnergy of Weight Update is: " << layerWriteDynamicEnergyPeakWU*1e12 << "pJ" << endl;
			cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
			cout << "layer" << i+1 << "'s leakagePower is: " << numTileEachLayer[0][i] * numTileEachLayer[1][i] * tileLeakage*1e6 << "uW" << endl;
			cout << "layer" << i+1 << "'s leakageEnergy is: " << layerLeakageEnergy*1e12 << "pJ" << endl;

			cout << endl;
			cout << "************************ Breakdown of Latency and Dynamic Energy *************************" << endl;
			cout << endl;
			cout << "----------- ADC (or S/As and precharger for SRAM) readLatency is : " << coreLatencyADC*1e9 << "ns" << endl;
			cout << "----------- Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) readLatency is : " << coreLatencyAccum*1e9 << "ns" << endl;
			cout << "----------- Synaptic Array w/o ADC (Forward + Activate Gradient) readLatency is : " << coreLatencyOther*1e9 << "ns" << endl;
			cout << "----------- Buffer buffer latency is: " << layerbufferLatency*1e9 << "ns" << endl;
			cout << "----------- Interconnect latency is: " << layericLatency*1e9 << "ns" << endl;
			cout << "----------- Weight Gradient Calculation readLatency is : " << layerReadLatencyPeakWG*1e9 << "ns" << endl;
			cout << "----------- Weight Update writeLatency is : " << layerWriteLatencyPeakWU*1e9 << "ns" << endl;
			cout << "----------- DRAM data transfer Latency is : " << layerDRAMLatency*1e9 << "ns" << endl;
			cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
			cout << "----------- ADC (or S/As and precharger for SRAM) readDynamicEnergy is : " << coreEnergyADC*1e12 << "pJ" << endl;
			cout << "----------- Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) readDynamicEnergy is : " << coreEnergyAccum*1e12 << "pJ" << endl;
			cout << "----------- Synaptic Array w/o ADC (Forward + Activate Gradient) readDynamicEnergy is : " << coreEnergyOther*1e12 << "pJ" << endl;
			cout << "----------- Buffer readDynamicEnergy is: " << layerbufferDynamicEnergy*1e12 << "pJ" << endl;
			cout << "----------- Interconnect readDynamicEnergy is: " << layericDynamicEnergy*1e12 << "pJ" << endl;
			cout << "----------- Weight Gradient Calculation readDynamicEnergy is : " << layerReadDynamicEnergyPeakWG*1e12 << "pJ" << endl;
			cout << "----------- Weight Update writeDynamicEnergy is : " << layerWriteDynamicEnergyPeakWU*1e12 << "pJ" << endl;
			cout << "----------- DRAM data transfer Energy is : " << layerDRAMDynamicEnergy*1e12 << "pJ" << endl;
			cout << endl;
			
			cout << "************************ Breakdown of Latency and Dynamic Energy *************************" << endl;
			cout << endl;
			
			
			if (breakdownfile.is_open()) {
				breakdownfile << i+1 << "," << layerReadLatency << "," << layerReadLatencyAG << "," << layerReadLatencyWG << "," << layerWriteLatencyWU << ",";
				breakdownfile << layerReadDynamicEnergy << "," << layerReadDynamicEnergyAG << "," << layerReadDynamicEnergyWG << "," << layerWriteDynamicEnergyWU << ",";
				breakdownfile << layerReadLatencyPeakFW << "," << layerReadLatencyPeakAG << "," << layerReadLatencyPeakWG << "," << layerWriteLatencyPeakWU << ",";
				breakdownfile << layerReadDynamicEnergyPeakFW << "," << layerReadDynamicEnergyPeakAG << "," << layerReadDynamicEnergyPeakWG << "," << layerWriteDynamicEnergyPeakWU <<",";
				breakdownfile << ", , " << coreLatencyADC << "," << coreLatencyAccum << "," << coreLatencyOther << "," <<layerbufferLatency << "," << layericLatency << "," << layerReadLatencyPeakWG << "," << layerWriteLatencyPeakWU << "," << layerDRAMLatency << ",";
				breakdownfile << coreEnergyADC << "," << coreEnergyAccum << "," << coreEnergyOther << "," << layerbufferDynamicEnergy << "," << layericDynamicEnergy << "," << layerReadDynamicEnergyPeakWG << "," << layerWriteDynamicEnergyPeakWU << "," << layerDRAMDynamicEnergy << endl;
			} else {
				cout << "Error: the breakdown file cannot be opened!" << endl;
			}
			
			chipReadLatency += layerReadLatency;
			chipReadDynamicEnergy += layerReadDynamicEnergy;
			chipReadLatencyAG += layerReadLatencyAG;
			chipReadDynamicEnergyAG += layerReadDynamicEnergyAG;
			chipReadLatencyWG += layerReadLatencyWG;
			chipReadDynamicEnergyWG += layerReadDynamicEnergyWG;
			chipWriteLatencyWU += layerWriteLatencyWU;
			chipWriteDynamicEnergyWU += layerWriteDynamicEnergyWU;
			chipDRAMLatency += layerDRAMLatency;
			chipDRAMDynamicEnergy += layerDRAMDynamicEnergy;
			
			chipReadLatencyPeakFW += layerReadLatencyPeakFW;
			chipReadDynamicEnergyPeakFW += layerReadDynamicEnergyPeakFW;
			chipReadLatencyPeakAG += layerReadLatencyPeakAG;
			chipReadDynamicEnergyPeakAG += layerReadDynamicEnergyPeakAG;
			chipReadLatencyPeakWG += layerReadLatencyPeakWG;
			chipReadDynamicEnergyPeakWG += layerReadDynamicEnergyPeakWG;
			chipWriteLatencyPeakWU += layerWriteLatencyPeakWU;
			chipWriteDynamicEnergyPeakWU += layerWriteDynamicEnergyPeakWU;
			
			chipLeakageEnergy += layerLeakageEnergy;
			chipLeakage += tileLeakage*numTileEachLayer[0][i] * numTileEachLayer[1][i];
			chipbufferLatency += layerbufferLatency;
			chipbufferReadDynamicEnergy += layerbufferDynamicEnergy;
			chipicLatency += layericLatency;
			chipicReadDynamicEnergy += layericDynamicEnergy;
			
			chipLatencyADC += coreLatencyADC;
			chipLatencyAccum += coreLatencyAccum;
			chipLatencyOther += coreLatencyOther;
			chipEnergyADC += coreEnergyADC;
			chipEnergyAccum += coreEnergyAccum;
			chipEnergyOther += coreEnergyOther;
		}
	} else {
		// pipeline system
		// firstly define system clock
		double systemClock = 0;
		double systemClockAG = 0;
		double systemClockPeakFW = 0;
		double systemClockPeakAG = 0;
		
		vector<double> readLatencyPerLayer;
		vector<double> readDynamicEnergyPerLayer;
		vector<double> readLatencyPerLayerAG;
		vector<double> readDynamicEnergyPerLayerAG;
		vector<double> readLatencyPerLayerWG;
		vector<double> readDynamicEnergyPerLayerWG;
		vector<double> writeLatencyPerLayerWU;
		vector<double> writeDynamicEnergyPerLayerWU;
		
		vector<double> readLatencyPerLayerPeakFW;
		vector<double> readDynamicEnergyPerLayerPeakFW;
		vector<double> readLatencyPerLayerPeakAG;
		vector<double> readDynamicEnergyPerLayerPeakAG;
		vector<double> readLatencyPerLayerPeakWG;
		vector<double> readDynamicEnergyPerLayerPeakWG;
		vector<double> writeLatencyPerLayerPeakWU;
		vector<double> writeDynamicEnergyPerLayerPeakWU;
		
		vector<double> dramLatencyPerLayer;
		vector<double> dramDynamicEnergyPerLayer;
		
		vector<double> leakagePowerPerLayer;
		vector<double> bufferLatencyPerLayer;
		vector<double> bufferEnergyPerLayer;
		vector<double> icLatencyPerLayer;
		vector<double> icEnergyPerLayer;
		
		vector<double> coreLatencyADCPerLayer;
		vector<double> coreEnergyADCPerLayer;
		vector<double> coreLatencyAccumPerLayer;
		vector<double> coreEnergyAccumPerLayer;
		vector<double> coreLatencyOtherPerLayer;
		vector<double> coreEnergyOtherPerLayer;
		
		for (int i=0; i<netStructure.size(); i++) {
			
            param->activityRowReadWG = atof(argv[4*i+8]);
            param->activityRowWriteWG = atof(argv[4*i+8]);
            param->activityColWriteWG = atof(argv[4*i+8]);
			ChipCalculatePerformance(inputParameter, tech, cell, i, argv[4*i+5], argv[4*i+6], argv[4*i+7], netStructure[i][6],
						netStructure, markNM, numTileEachLayer, utilizationEachLayer, speedUpEachLayer, tileLocaEachLayer,
						numPENM, desiredPESizeNM, desiredTileSizeCM, desiredPESizeCM, CMTileheight, CMTilewidth, NMTileheight, NMTilewidth, numArrayWriteParallel,
						&layerReadLatency, &layerReadDynamicEnergy, &tileLeakage, &layerReadLatencyAG, &layerReadDynamicEnergyAG, &layerReadLatencyWG, &layerReadDynamicEnergyWG, &layerWriteLatencyWU, &layerWriteDynamicEnergyWU,
						&layerbufferLatency, &layerbufferDynamicEnergy, &layericLatency, &layericDynamicEnergy,
						&coreLatencyADC, &coreLatencyAccum, &coreLatencyOther, &coreEnergyADC, &coreEnergyAccum, &coreEnergyOther, &layerDRAMLatency, &layerDRAMDynamicEnergy,
						&layerReadLatencyPeakFW, &layerReadDynamicEnergyPeakFW, &layerReadLatencyPeakAG, &layerReadDynamicEnergyPeakAG,
						&layerReadLatencyPeakWG, &layerReadDynamicEnergyPeakWG, &layerWriteLatencyPeakWU, &layerWriteDynamicEnergyPeakWU);
						
			
			systemClock = MAX(systemClock, layerReadLatency);
			systemClockAG = MAX(systemClockAG, layerReadLatencyAG);
			systemClockPeakFW = MAX(systemClockPeakFW, layerReadLatencyPeakFW);
			systemClockPeakAG = MAX(systemClockPeakAG, layerReadLatencyPeakAG);
			chipLatencyADC = MAX(chipLatencyADC, coreLatencyADCPerLayer[i]);
			chipLatencyAccum = MAX(chipLatencyAccum, coreLatencyAccumPerLayer[i]);
			chipLatencyOther = MAX(chipLatencyOther, coreLatencyOtherPerLayer[i]);
			
			readLatencyPerLayer.push_back(layerReadLatency);
			readDynamicEnergyPerLayer.push_back(layerReadDynamicEnergy);
			readLatencyPerLayerAG.push_back(layerReadLatencyAG);
			readDynamicEnergyPerLayerAG.push_back(layerReadDynamicEnergyAG);
			readLatencyPerLayerWG.push_back(layerReadLatencyWG);
			readDynamicEnergyPerLayerWG.push_back(layerReadDynamicEnergyWG);
			writeLatencyPerLayerWU.push_back(layerWriteLatencyWU);
			writeDynamicEnergyPerLayerWU.push_back(layerWriteDynamicEnergyWU);
			dramLatencyPerLayer.push_back(layerDRAMLatency);
			dramDynamicEnergyPerLayer.push_back(layerDRAMDynamicEnergy);
			
			readLatencyPerLayerPeakFW.push_back(layerReadLatencyPeakFW);
			readDynamicEnergyPerLayerPeakFW.push_back(layerReadDynamicEnergyPeakFW);
			readLatencyPerLayerPeakAG.push_back(layerReadLatencyPeakAG);
			readDynamicEnergyPerLayerPeakAG.push_back(layerReadDynamicEnergyPeakAG);
			readLatencyPerLayerPeakWG.push_back(layerReadLatencyPeakWG);
			readDynamicEnergyPerLayerPeakWG.push_back(layerReadDynamicEnergyPeakWG);
			writeLatencyPerLayerPeakWU.push_back(layerWriteLatencyPeakWU);
			writeDynamicEnergyPerLayerPeakWU.push_back(layerWriteDynamicEnergyPeakWU);
			
			leakagePowerPerLayer.push_back(numTileEachLayer[0][i] * numTileEachLayer[1][i] * tileLeakage);
			bufferLatencyPerLayer.push_back(layerbufferLatency);
			bufferEnergyPerLayer.push_back(layerbufferDynamicEnergy);
			icLatencyPerLayer.push_back(layericLatency);
			icEnergyPerLayer.push_back(layericDynamicEnergy);
			
			coreLatencyADCPerLayer.push_back(coreLatencyADC);
			coreEnergyADCPerLayer.push_back(coreEnergyADC);
			coreLatencyAccumPerLayer.push_back(coreLatencyAccum);
			coreEnergyAccumPerLayer.push_back(coreEnergyAccum);
			coreLatencyOtherPerLayer.push_back(coreLatencyOther);
			coreEnergyOtherPerLayer.push_back(coreEnergyOther);
			
			chipReadDynamicEnergy += layerReadDynamicEnergy;
			chipReadDynamicEnergyAG += layerReadDynamicEnergyAG;
			chipReadDynamicEnergyWG += layerReadDynamicEnergyWG;
			chipWriteDynamicEnergyWU += layerWriteDynamicEnergyWU;
			// since Weight Gradient and Weight Update have limitation on hardware resource, do not implement pipeline
			chipReadLatencyWG += layerReadLatencyWG;
			chipWriteLatencyWU += layerWriteLatencyWU;
			
			chipReadDynamicEnergyPeakFW += layerReadDynamicEnergyPeakFW;
			chipReadDynamicEnergyPeakAG += layerReadDynamicEnergyPeakAG;
			chipReadDynamicEnergyPeakWG += layerReadDynamicEnergyPeakWG;
			chipWriteDynamicEnergyPeakWU += layerWriteDynamicEnergyPeakWU;
			
			chipDRAMLatency += layerDRAMLatency;
			chipDRAMDynamicEnergy += layerDRAMDynamicEnergy;
			
			chipLeakage += numTileEachLayer[0][i] * numTileEachLayer[1][i] * tileLeakage;
			chipbufferLatency = MAX(chipbufferLatency, layerbufferLatency);
			chipbufferReadDynamicEnergy += layerbufferDynamicEnergy;
			chipicLatency = MAX(chipicLatency, layericLatency);
			chipicReadDynamicEnergy += layericDynamicEnergy;
			chipEnergyADC += coreEnergyADC;
			chipEnergyAccum += coreEnergyAccum;
			chipEnergyOther += coreEnergyOther;
			
		}
		chipReadLatency = systemClock;
		chipReadLatencyAG = systemClockAG;
		chipReadLatencyPeakFW = systemClockPeakFW;
		chipReadLatencyPeakAG = systemClockPeakAG;
		
		for (int i=0; i<netStructure.size(); i++) {
			
			cout << "-------------------- Estimation of Layer " << i+1 << " ----------------------" << endl;

			cout << "layer" << i+1 << "'s readLatency is: " << readLatencyPerLayer[i]*1e9 << "ns" << endl;
			cout << "layer" << i+1 << "'s readDynamicEnergy is: " << readDynamicEnergyPerLayer[i]*1e12 << "pJ" << endl;
			cout << "layer" << i+1 << "'s readLatency of Activation Gradient is: " << readLatencyPerLayerAG[i]*1e9 << "ns" << endl;
			cout << "layer" << i+1 << "'s readDynamicEnergy of Activation Gradient is: " << readDynamicEnergyPerLayerAG[i]*1e12 << "pJ" << endl;
			cout << "layer" << i+1 << "'s readLatency of Weight Gradient is: " << readLatencyPerLayerWG[i]*1e9 << "ns" << endl;
			cout << "layer" << i+1 << "'s readDynamicEnergy of Weight Gradient is: " << readDynamicEnergyPerLayerWG[i]*1e12 << "pJ" << endl;
			cout << "layer" << i+1 << "'s writeLatency of Weight Update is: " << writeLatencyPerLayerWU[i]*1e9 << "ns" << endl;
			cout << "layer" << i+1 << "'s writeDynamicEnergy of Weight Update is: " << writeDynamicEnergyPerLayerWU[i]*1e12 << "pJ" << endl;
			cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
			cout << "layer" << i+1 << "'s PEAK readLatency is: " << readLatencyPerLayerPeakFW[i]*1e9 << "ns" << endl;
			cout << "layer" << i+1 << "'s PEAK readDynamicEnergy is: " << readDynamicEnergyPerLayerPeakFW[i]*1e12 << "pJ" << endl;
			cout << "layer" << i+1 << "'s PEAK readLatency of Activation Gradient is: " << readLatencyPerLayerPeakAG[i]*1e9 << "ns" << endl;
			cout << "layer" << i+1 << "'s PEAK readDynamicEnergy of Activation Gradient is: " << readDynamicEnergyPerLayerPeakAG[i]*1e12 << "pJ" << endl;
			cout << "layer" << i+1 << "'s PEAK readLatency of Weight Gradient is: " << readLatencyPerLayerPeakWG[i]*1e9 << "ns" << endl;
			cout << "layer" << i+1 << "'s PEAK readDynamicEnergy of Weight Gradient is: " << readDynamicEnergyPerLayerPeakWG[i]*1e12 << "pJ" << endl;
			cout << "layer" << i+1 << "'s PEAK writeLatency of Weight Update is: " << writeLatencyPerLayerPeakWU[i]*1e9 << "ns" << endl;
			cout << "layer" << i+1 << "'s PEAK writeDynamicEnergy of Weight Update is: " << writeDynamicEnergyPerLayerPeakWU[i]*1e12 << "pJ" << endl;
			cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
			cout << "layer" << i+1 << "'s leakagePower is: " << leakagePowerPerLayer[i]*1e6 << "uW" << endl;
			cout << "layer" << i+1 << "'s leakageEnergy is: " << leakagePowerPerLayer[i] * (systemClock-readLatencyPerLayer[i]) *1e12 << "pJ" << endl;
			cout << endl;
			cout << "************************ Breakdown of Latency and Dynamic Energy *************************" << endl;
			cout << endl;
			cout << "----------- ADC (or S/As and precharger for SRAM) readLatency is : " << coreLatencyADCPerLayer[i]*1e9 << "ns" << endl;
			cout << "----------- Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) readLatency is : " << coreLatencyAccumPerLayer[i]*1e9 << "ns" << endl;
			cout << "----------- Synaptic Array w/o ADC (Forward + Activate Gradient) readLatency is : " << coreLatencyOtherPerLayer[i]*1e9 << "ns" << endl;
			cout << "----------- Buffer latency is: " << bufferLatencyPerLayer[i]*1e9 << "ns" << endl;
			cout << "----------- Interconnect latency is: " << icLatencyPerLayer[i]*1e9 << "ns" << endl;
			cout << "----------- Weight Gradient Calculation readLatency is : " << readLatencyPerLayerPeakWG[i]*1e9 << "ns" << endl;
			cout << "----------- Weight Update writeLatency is : " << writeLatencyPerLayerPeakWU[i]*1e9 << "ns" << endl;
			cout << "----------- DRAM data transfer Latency is : " << dramLatencyPerLayer[i]*1e9 << "ns" << endl;
			cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
			cout << "----------- ADC (or S/As and precharger for SRAM) readDynamicEnergy is : " << coreEnergyADCPerLayer[i]*1e12 << "pJ" << endl;
			cout << "----------- Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) readDynamicEnergy is : " << coreEnergyAccumPerLayer[i]*1e12 << "pJ" << endl;
			cout << "----------- Synaptic Array w/o ADC (Forward + Activate Gradient) readDynamicEnergy is : " << coreEnergyOtherPerLayer[i]*1e12 << "pJ" << endl;
			cout << "----------- Buffer readDynamicEnergy is: " << bufferEnergyPerLayer[i]*1e12 << "pJ" << endl;
			cout << "----------- Interconnect readDynamicEnergy is: " << icEnergyPerLayer[i]*1e12 << "pJ" << endl;
			cout << "----------- Weight Gradient Calculation readDynamicEnergy is : " << readDynamicEnergyPerLayerPeakWG[i]*1e12 << "pJ" << endl;
			cout << "----------- Weight Update writeDynamicEnergy is : " << writeDynamicEnergyPerLayerPeakWU[i]*1e12 << "pJ" << endl;
			cout << "----------- DRAM data transfer DynamicEnergy is : " << dramDynamicEnergyPerLayer[i]*1e12 << "pJ" << endl;
			cout << endl;
			cout << "************************ Breakdown of Latency and Dynamic Energy *************************" << endl;
			cout << endl;
			
			chipLeakageEnergy += leakagePowerPerLayer[i] * ((systemClock-readLatencyPerLayer[i]) + (systemClockAG-readLatencyPerLayerAG[i]));
			
			if (breakdownfile.is_open()) {
				breakdownfile << i+1 << "," << readLatencyPerLayer[i] << "," << readLatencyPerLayerAG[i] << "," << readLatencyPerLayerWG[i] << "," << writeLatencyPerLayerWU[i] << ",";
				breakdownfile << readDynamicEnergyPerLayer[i] << "," << readDynamicEnergyPerLayerAG[i] << "," << readDynamicEnergyPerLayerWG[i] << "," << writeDynamicEnergyPerLayerWU[i] << ",";
				breakdownfile << readLatencyPerLayerPeakFW[i] << "," << readLatencyPerLayerPeakAG[i] << "," << readLatencyPerLayerPeakWG[i] << "," << writeLatencyPerLayerPeakWU[i] << ",";
				breakdownfile << readDynamicEnergyPerLayerPeakFW[i] << "," << readDynamicEnergyPerLayerPeakAG[i] << "," << readDynamicEnergyPerLayerPeakWG[i] << "," << writeDynamicEnergyPerLayerPeakWU[i] << ",";
				breakdownfile << ", , " << coreLatencyADCPerLayer[i] << "," << coreLatencyAccumPerLayer[i] << "," << coreLatencyOtherPerLayer[i] << "," << bufferLatencyPerLayer[i] << "," << icLatencyPerLayer[i] << "," << readLatencyPerLayerPeakWG[i] << "," << writeLatencyPerLayerPeakWU[i] << "," << dramLatencyPerLayer[i] <<",";
				breakdownfile << coreEnergyADCPerLayer[i] << "," << coreEnergyAccumPerLayer[i] << "," << coreEnergyOtherPerLayer[i] << "," << bufferEnergyPerLayer[i] << "," << icEnergyPerLayer[i] << "," << readDynamicEnergyPerLayerPeakWG[i] << "," << writeDynamicEnergyPerLayerPeakWU[i] << "," << dramDynamicEnergyPerLayer[i] << endl;
			} else {
				cout << "Error: the breakdown file cannot be opened!" << endl;
			}
		}
	}
	
	if (breakdownfile.is_open()) {
		breakdownfile << "Total" << "," << chipReadLatency << "," << chipReadLatencyAG << "," << chipReadLatencyWG << "," << chipWriteLatencyWU << ",";
		breakdownfile << chipReadDynamicEnergy << "," << chipReadDynamicEnergyAG << "," << chipReadDynamicEnergyWG << "," << chipWriteDynamicEnergyWU << ",";
		breakdownfile << chipReadLatencyPeakFW << "," << chipReadLatencyPeakAG << "," << chipReadLatencyPeakWG << "," << chipWriteLatencyPeakWU << ",";
		breakdownfile << chipReadDynamicEnergyPeakFW << "," << chipReadDynamicEnergyPeakAG << "," << chipReadDynamicEnergyPeakWG << "," << chipWriteDynamicEnergyPeakWU << ",";
		breakdownfile << ", , " << chipLatencyADC << "," << chipLatencyAccum << "," << chipLatencyOther << "," << chipbufferLatency << "," << chipicLatency << "," << chipReadLatencyPeakWG <<"," << chipWriteLatencyPeakWU <<"," << chipDRAMLatency <<",";
		breakdownfile << chipEnergyADC << "," << chipEnergyAccum << "," << chipEnergyOther << "," << chipbufferReadDynamicEnergy << "," << chipicReadDynamicEnergy << "," << chipReadDynamicEnergyPeakWG << "," << chipWriteDynamicEnergyPeakWU << "," << chipDRAMDynamicEnergy << endl;
		breakdownfile << endl;
		breakdownfile << endl;
		breakdownfile << "TOPS/W,FPS,TOPS,Peak TOPS/W,Peak FPS,Peak TOPS," << endl;
		breakdownfile << numComputation/((chipReadDynamicEnergy+chipLeakageEnergy+chipReadDynamicEnergyAG+chipReadDynamicEnergyWG+chipWriteDynamicEnergyWU)*1e12) << ",";
		breakdownfile <<  1/(chipReadLatency+chipReadLatencyAG+chipReadLatencyWG+chipWriteLatencyWU) << ",";
		breakdownfile <<  numComputation/(chipReadLatency+chipReadLatencyAG+chipReadLatencyWG+chipWriteLatencyWU)*1e-12 << ",";
		breakdownfile <<  numComputation/((chipReadDynamicEnergyPeakFW+chipReadDynamicEnergyPeakAG+chipReadDynamicEnergyPeakWG+chipWriteDynamicEnergyPeakWU)*1e12) << ",";
		breakdownfile <<  1/(chipReadLatencyPeakFW+chipReadLatencyPeakAG+chipReadLatencyPeakWG+chipWriteLatencyPeakWU) << ",";
		breakdownfile <<  numComputation/(chipReadLatencyPeakFW+chipReadLatencyPeakAG+chipReadLatencyPeakWG+chipWriteLatencyPeakWU)*1e-12 << endl;
	} else {
		cout << "Error: the breakdown file cannot be opened!" << endl;
	}
	
	breakdownfile.close();
	
	cout << "------------------------------ Summary --------------------------------" <<  endl;
	cout << endl;
	cout << "ChipArea : " << chipArea*1e12 << "um^2" << endl;
	cout << "Chip total CIM (Forward+Activation Gradient) array : " << chipAreaArray*1e12 << "um^2" << endl;
	cout << "Total IC Area on chip (Global and Tile/PE local): " << chipAreaIC*1e12 << "um^2" << endl;
	cout << "Total ADC (or S/As and precharger for SRAM) Area on chip : " << chipAreaADC*1e12 << "um^2" << endl;
	cout << "Total Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) on chip : " << chipAreaAccum*1e12 << "um^2" << endl;
	cout << "Other Peripheries (e.g. decoders, mux, switchmatrix, buffers, pooling and activation units) : " << chipAreaOther*1e12 << "um^2" << endl;
	cout << "Weight Gradient Calculation : " << chipAreaWG*1e12 << "um^2" << endl;
	cout << endl;
	if (! param->pipeline) {
		cout << "-----------------------------------Chip layer-by-layer Estimation---------------------------------" << endl;
	} else {
		cout << "--------------------------------------Chip pipeline Estimation---------------------------------" << endl;
	}
	cout << "Chip readLatency of Forward (per epoch) is: " << chipReadLatency*1e9 << "ns" << endl;
	cout << "Chip readDynamicEnergy of Forward (per epoch) is: " << chipReadDynamicEnergy*1e12 << "pJ" << endl;
	cout << "Chip readLatency of Activation Gradient (per epoch) is: " << chipReadLatencyAG*1e9 << "ns" << endl;
	cout << "Chip readDynamicEnergy of Activation Gradient (per epoch) is: " << chipReadDynamicEnergyAG*1e12 << "pJ" << endl;
	cout << "Chip readLatency of Weight Gradient (per epoch) is: " << chipReadLatencyWG*1e9 << "ns" << endl;
	cout << "Chip readDynamicEnergy of Weight Gradient (per epoch) is: " << chipReadDynamicEnergyWG*1e12 << "pJ" << endl;
	cout << "Chip writeLatency of Weight Update (per epoch) is: " << chipWriteLatencyWU*1e9 << "ns" << endl;
	cout << "Chip writeDynamicEnergy of Weight Update (per epoch) is: " << chipWriteDynamicEnergyWU*1e12 << "pJ" << endl;
	cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
	cout << "Chip total Latency (per epoch) is: " << (chipReadLatency+chipReadLatencyAG+chipReadLatencyWG+chipWriteLatencyWU)*1e9 << "ns" << endl;
	cout << "Chip total Energy (per epoch) is: " << (chipReadDynamicEnergy+chipReadDynamicEnergyAG+chipReadDynamicEnergyWG+chipWriteDynamicEnergyWU)*1e12 << "pJ" << endl;
	cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
	cout << "Chip PEAK readLatency of Forward (per epoch) is: " << chipReadLatencyPeakFW*1e9 << "ns" << endl;
	cout << "Chip PEAK readDynamicEnergy of Forward (per epoch) is: " << chipReadDynamicEnergyPeakFW*1e12 << "pJ" << endl;
	cout << "Chip PEAK readLatency of Activation Gradient (per epoch) is: " << chipReadLatencyPeakAG*1e9 << "ns" << endl;
	cout << "Chip PEAK readDynamicEnergy of Activation Gradient (per epoch) is: " << chipReadDynamicEnergyPeakAG*1e12 << "pJ" << endl;
	cout << "Chip PEAK readLatency of Weight Gradient (per epoch) is: " << chipReadLatencyPeakWG*1e9 << "ns" << endl;
	cout << "Chip PEAK readDynamicEnergy of Weight Gradient (per epoch) is: " << chipReadDynamicEnergyPeakWG*1e12 << "pJ" << endl;
	cout << "Chip PEAK writeLatency of Weight Update (per epoch) is: " << chipWriteLatencyPeakWU*1e9 << "ns" << endl;
	cout << "Chip PEAK writeDynamicEnergy of Weight Update (per epoch) is: " << chipWriteDynamicEnergyPeakWU*1e12 << "pJ" << endl;
	cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
	cout << "Chip PEAK total Latency (per epoch) is: " << (chipReadLatencyPeakFW+chipReadLatencyPeakAG+chipReadLatencyPeakWG+chipWriteLatencyPeakWU)*1e9 << "ns" << endl;
	cout << "Chip PEAK total Energy (per epoch) is: " << (chipReadDynamicEnergyPeakFW+chipReadDynamicEnergyPeakAG+chipReadDynamicEnergyPeakWG+chipWriteDynamicEnergyPeakWU)*1e12 << "pJ" << endl;
	cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
	cout << "Chip leakage Energy is: " << chipLeakageEnergy*1e12 << "pJ" << endl;
	cout << "Chip leakage Power is: " << chipLeakage*1e6 << "uW" << endl;
	cout << endl;
	cout << "************************ Breakdown of Latency and Dynamic Energy *************************" << endl;
	cout << endl;
	cout << "----------- ADC (or S/As and precharger for SRAM) readLatency is : " << chipLatencyADC*1e9 << "ns" << endl;
	cout << "----------- Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) readLatency is : " << chipLatencyAccum*1e9 << "ns" << endl;
	cout << "----------- Synaptic Array w/o ADC (Forward + Activate Gradient) readLatency is : " << chipLatencyOther*1e9 << "ns" << endl;
	cout << "----------- Buffer readLatency is: " << chipbufferLatency*1e9 << "ns" << endl;
	cout << "----------- Interconnect readLatency is: " << chipicLatency*1e9 << "ns" << endl;
	cout << "----------- Weight Gradient Calculation readLatency is : " << chipReadLatencyPeakWG*1e9 << "ns" << endl;
	cout << "----------- Weight Update writeLatency is : " << chipWriteLatencyPeakWU*1e9 << "ns" << endl;
	cout << "----------- DRAM data transfer Latency is : " << chipDRAMLatency*1e9 << "ns" << endl;
	cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
	cout << "----------- ADC (or S/As and precharger for SRAM) readDynamicEnergy is : " << chipEnergyADC*1e12 << "pJ" << endl;
	cout << "----------- Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) readDynamicEnergy is : " << chipEnergyAccum*1e12 << "pJ" << endl;
	cout << "----------- Synaptic Array w/o ADC (Forward + Activate Gradient) readDynamicEnergy is : " << chipEnergyOther*1e12 << "pJ" << endl;
	cout << "----------- Buffer readDynamicEnergy is: " << chipbufferReadDynamicEnergy*1e12 << "pJ" << endl;
	cout << "----------- Interconnect readDynamicEnergy is: " << chipicReadDynamicEnergy*1e12 << "pJ" << endl;
	cout << "----------- Weight Gradient Calculation readDynamicEnergy is : " << chipReadDynamicEnergyPeakWG*1e12 << "pJ" << endl;
	cout << "----------- Weight Update writeDynamicEnergy is : " << chipWriteDynamicEnergyPeakWU*1e12 << "pJ" << endl;
	cout << "----------- DRAM data transfer DynamicEnergy is : " << chipDRAMDynamicEnergy*1e12 << "pJ" << endl;
	cout << endl;
	cout << "************************ Breakdown of Latency and Dynamic Energy *************************" << endl;
	cout << endl;
	cout << endl;
	if (! param->pipeline) {
		cout << "-----------------------------------Chip layer-by-layer Performance---------------------------------" << endl;
	} else {
		cout << "--------------------------------------Chip pipeline Performance---------------------------------" << endl;
	}
	
	cout << "Energy Efficiency TOPS/W: " << numComputation/((chipReadDynamicEnergy+chipLeakageEnergy+chipReadDynamicEnergyAG+chipReadDynamicEnergyWG+chipWriteDynamicEnergyWU)*1e12) << endl;
	cout << "Throughput TOPS: " << numComputation/(chipReadLatency+chipReadLatencyAG+chipReadLatencyWG+chipWriteLatencyWU)*1e-12 << endl;
	cout << "Throughput FPS: " << 1/(chipReadLatency+chipReadLatencyAG+chipReadLatencyWG+chipWriteLatencyWU) << endl;
	cout << "--------------------------------------------------------------------------" << endl;
	cout << "Peak Energy Efficiency TOPS/W: " << numComputation/((chipReadDynamicEnergyPeakFW+chipReadDynamicEnergyPeakAG+chipReadDynamicEnergyPeakWG+chipWriteDynamicEnergyPeakWU)*1e12) << endl;
	cout << "Peak Throughput TOPS: " << numComputation/(chipReadLatencyPeakFW+chipReadLatencyPeakAG+chipReadLatencyPeakWG+chipWriteLatencyPeakWU)*1e-12 << endl;
	cout << "Peak Throughput FPS: " << 1/(chipReadLatencyPeakFW+chipReadLatencyPeakAG+chipReadLatencyPeakWG+chipWriteLatencyPeakWU) << endl;
	
	cout << "-------------------------------------- Hardware Performance Done --------------------------------------" <<  endl;
	cout << endl;
	auto stop = chrono::high_resolution_clock::now();
	auto duration = chrono::duration_cast<chrono::seconds>(stop-start);
    cout << "------------------------------ Simulation Performance --------------------------------" <<  endl;
	cout << "Total Run-time of NeuroSim: " << duration.count() << " seconds" << endl;
	cout << "------------------------------ Simulation Performance --------------------------------" <<  endl;
	
	// save results to top level csv file (only total results)
	ofstream outfile;
	outfile.open ("NeuroSim_Output.csv", ios::app);
	if (outfile.is_open()) {
		outfile << chipReadLatency << "," << chipReadLatencyAG << "," << chipReadLatencyWG << "," << chipWriteLatencyWU << ",";
		outfile << chipReadDynamicEnergy << "," << chipReadDynamicEnergyAG << "," << chipReadDynamicEnergyWG << "," << chipWriteDynamicEnergyWU << ",";
		outfile << chipReadLatencyPeakFW << "," << chipReadLatencyPeakAG << "," << chipReadLatencyPeakWG << "," << chipWriteLatencyPeakWU << ",";
		outfile << chipReadDynamicEnergyPeakFW << "," << chipReadDynamicEnergyPeakAG << "," << chipReadDynamicEnergyPeakWG << "," << chipWriteDynamicEnergyPeakWU << ",";
		outfile << numComputation/((chipReadDynamicEnergy+chipLeakageEnergy+chipReadDynamicEnergyAG+chipReadDynamicEnergyWG+chipWriteDynamicEnergyWU)*1e12) << ",";
		outfile << numComputation/(chipReadLatency+chipReadLatencyAG+chipReadLatencyWG+chipWriteLatencyWU)*1e-12 << ",";
		outfile << numComputation/((chipReadDynamicEnergyPeakFW+chipReadDynamicEnergyPeakAG+chipReadDynamicEnergyPeakWG+chipWriteDynamicEnergyPeakWU)*1e12) << ",";
		outfile << numComputation/(chipReadLatencyPeakFW+chipReadLatencyPeakAG+chipReadLatencyPeakWG+chipWriteLatencyPeakWU)*1e-12 << endl;
	} else {
		cout << "Error: the output file cannot be opened!" << endl;
	}
	outfile.close();
	
	
	return 0;
}

vector<vector<double> > getNetStructure(const string &inputfile) {
	ifstream infile(inputfile.c_str());      
	string inputline;
	string inputval;
	
	int ROWin=0, COLin=0;      
	if (!infile.good()) {        
		cerr << "Error: the input file cannot be opened!" << endl;
		exit(1);
	}else{
		while (getline(infile, inputline, '\n')) {       
			ROWin++;                                
		}
		infile.clear();
		infile.seekg(0, ios::beg);      
		if (getline(infile, inputline, '\n')) {        
			istringstream iss (inputline);      
			while (getline(iss, inputval, ',')) {       
				COLin++;
			}
		}	
	}
	infile.clear();
	infile.seekg(0, ios::beg);          

	vector<vector<double> > netStructure;               
	for (int row=0; row<ROWin; row++) {	
		vector<double> netStructurerow;
		getline(infile, inputline, '\n');             
		istringstream iss;
		iss.str(inputline);
		for (int col=0; col<COLin; col++) {       
			while(getline(iss, inputval, ',')){	
				istringstream fs;
				fs.str(inputval);
				double f=0;
				fs >> f;				
				netStructurerow.push_back(f);			
			}			
		}		
		netStructure.push_back(netStructurerow);
	}
	infile.close();
	
	return netStructure;
	netStructure.clear();
}	



