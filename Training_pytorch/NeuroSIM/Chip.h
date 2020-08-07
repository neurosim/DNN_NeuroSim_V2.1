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
*   Pai-Yu Chen     Email: pchen72 at asu dot edu 
*                     
*   Xiaochen Peng   Email: xpeng15 at asu dot edu
********************************************************************************/

#ifndef CHIP_H_
#define CHIP_H_

/*** Functions ***/
vector<int> ChipDesignInitialize(InputParameter& inputParameter, Technology& tech, MemCell& cell, bool pip, const vector<vector<double> > &netStructure,
					double *maxPESizeNM, double *maxTileSizeCM, double *numPENM);
					
vector<vector<double> > ChipFloorPlan(bool findNumTile, bool findUtilization, bool findSpeedUp, const vector<vector<double> > &netStructure, const vector<int > &markNM, 
					double maxPESizeNM, double maxTileSizeCM, double numPENM, const vector<int> &pipelineSpeedUp,
					double *desiredNumTileNM, double *desiredPESizeNM, double *desiredNumTileCM, double *desiredTileSizeCM, double *desiredPESizeCM, int *numTileRow, int *numTileCol);
					
void ChipInitialize(InputParameter& inputParameter, Technology& tech, MemCell& cell, const vector<vector<double> > &netStructure, const vector<int > &markNM, const vector<vector<double> > &numTileEachLayer,
					double numPENM, double desiredNumTileNM, double desiredPESizeNM, double desiredNumTileCM, double desiredTileSizeCM, double desiredPESizeCM, int numTileRow, int numTileCol, int *numArrayWriteParallel);
					
vector<double> ChipCalculateArea(InputParameter& inputParameter, Technology& tech, MemCell& cell, double desiredNumTileNM, double numPENM, double desiredPESizeNM, double desiredNumTileCM, double desiredTileSizeCM, double desiredPESizeCM, 
						int numTileRow, double *height, double *width, double *CMTileheight, double *CMTilewidth, double *NMTileheight, double *NMTilewidth);
						
double ChipCalculatePerformance(Technology& tech, MemCell& cell, int layerNumber, const string &newweightfile, const string &oldweightfile, const string &inputfile, bool followedByMaxPool, const vector<vector<double> > &netStructure, 
							const vector<int> &markNM, const vector<vector<double> > &numTileEachLayer, const vector<vector<double> > &utilizationEachLayer, const vector<vector<double> > &speedUpEachLayer, 
							const vector<vector<double> > &tileLocaEachLayer, double numPENM, double desiredPESizeNM, double desiredTileSizeCM, double desiredPESizeCM,
							double CMTileheight, double CMTilewidth, double NMTileheight, double NMTilewidth, int numArrayWriteParallel, double *readLatency, double *readDynamicEnergy, 
							double *leakage, double *readLatencyAG, double *readDynamicEnergyAG, double *readLatencyWG, double *readDynamicEnergyWG, double *writeLatencyWU, double *writeDynamicEnergyWU,
							double *bufferLatency, double *bufferDynamicEnergy, double *icLatency, double *icDynamicEnergy,double *coreLatencyADC, double *coreLatencyAccum, 
							double *coreLatencyOther, double *coreEnergyADC, double *coreEnergyAccum, double *coreEnergyOther, double *dramLatency, double *dramDynamicEnergy,
							double *readLatencyPeakFW, double *readDynamicEnergyPeakFW, double *readLatencyPeakAG, double *readDynamicEnergyPeakAG, double *readLatencyPeakWG, double *readDynamicEnergyPeakWG,
							double *writeLatencyPeakWU, double *writeDynamicEnergyPeakWU);
							
vector<double> TileDesignCM(double tileSize, const vector<int > &markNM, const vector<vector<double> > &netStructure, int numRowPerSynapse, int numColPerSynapse);
vector<double> TileDesignNM(double peSize, const vector<int > &markNM, const vector<vector<double> > &netStructure, int numRowPerSynapse, int numColPerSynapse, double numPENM);
vector<vector<double> > PEDesign(bool Design, double peSize, double desiredTileSize, double numTileTotal, const vector<int > &markNM, const vector<vector<double> > &netStructure, int numRowPerSynapse, int numColPerSynapse);
vector<vector<double> > SubArrayDup(double desiredPESizeCM, double desiredPESizeNM, const vector<int > &markNM, const vector<vector<double> > &netStructure, int numRowPerSynapse, int numColPerSynapse);
vector<vector<double> > OverallEachLayer(bool utilization, bool speedUp, const vector<vector<double> > &peDup, const vector<vector<double> > &subArrayDup, const vector<int> &pipelineSpeedUp, double desiredTileSizeCM, 
										double desiredPESizeNM, const vector<int > &markNM, const vector<vector<double> > &netStructure, int numRowPerSynapse, int numColPerSynapse, double numPENM);

vector<vector<double> > LoadInWeightData(const string &weightfile, int numRowPerSynapse, int numColPerSynapse, double maxConductance, double minConductance);
vector<vector<double> > CopyArray(const vector<vector<double> > &orginal, int positionRow, int positionCol, int numRow, int numCol);
vector<vector<double> > ReshapeArray(const vector<vector<double> > &orginal, int positionRow, int positionCol, int numRow, int numCol, int numPE, int weightMatrixRow);
vector<vector<double> > LoadInInputData(const string &inputfile);
vector<vector<double> > CopyInput(const vector<vector<double> > &orginal, int positionRow, int numInputVector, int numRow);
vector<vector<double> > ReshapeInput(const vector<vector<double> > &orginal, int positionRow, int numInputVector, int numRow, int numPE, int weightMatrixRow);

#endif /* CHIP_H_ */