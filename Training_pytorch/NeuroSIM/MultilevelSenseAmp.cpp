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
#include "MultilevelSenseAmp.h"

using namespace std;

extern Param *param;

MultilevelSenseAmp::MultilevelSenseAmp(const InputParameter& _inputParameter, const Technology& _tech, const MemCell& _cell): inputParameter(_inputParameter), tech(_tech), cell(_cell), currentSenseAmp(_inputParameter, _tech, _cell), FunctionUnit() {
	initialized = false;
}

void MultilevelSenseAmp::Initialize(int _numCol, int _levelOutput, double _clkFreq, int _numReadCellPerOperationNeuro, bool _parallel, bool _currentMode) {
	if (initialized) {
		cout << "[MultilevelSenseAmp] Warning: Already initialized!" << endl;
    } else {
		numCol = _numCol;
		levelOutput = _levelOutput;                // # of bits for A/D output ... 
		clkFreq = _clkFreq;
		numReadCellPerOperationNeuro = _numReadCellPerOperationNeuro;
		parallel = _parallel;
		currentMode = _currentMode;
		
		if (parallel) {
			for (int i=0; i<levelOutput-1; i++){
				double R_start = (double) param->resistanceOn / param->numRowSubArray;
				double R_index = (double) param->resistanceOff / param->numRowSubArray;
				double R_this = R_start + (double) (i+1)*R_index/levelOutput;
				Rref.push_back(R_this);
			} // TODO: Nonlinear Quantize
		} else {
			for (int i=0; i<levelOutput-1; i++){
				double R_start = (double) param->resistanceOn;
				double R_index = (double) param->resistanceOff;
				double R_this = R_start + (double) (i+1)*R_index/levelOutput;
				Rref.push_back(R_this);
			} // TODO: Nonlinear Quantize
		}
		widthNmos = MIN_NMOS_SIZE * tech.featureSize;
		widthPmos = tech.pnSizeRatio * MIN_NMOS_SIZE * tech.featureSize;
		initialized = true;
	}
}

void MultilevelSenseAmp::CalculateArea(double heightArray, double widthArray, AreaModify _option) {
	if (!initialized) {
		cout << "[MultilevelSenseAmp] Error: Require initialization first!" << endl;
	} else {
		
		area = 0;
		height = 0;
		width = 0;
		
		double hNmos, wNmos, hPmos, wPmos;
		CalculateGateArea(INV, 1, widthNmos, 0, tech.featureSize*MAX_TRANSISTOR_HEIGHT, tech, &hNmos, &wNmos);
		CalculateGateArea(INV, 1, 0, widthPmos, tech.featureSize*MAX_TRANSISTOR_HEIGHT, tech, &hPmos, &wPmos);
		
		if (widthArray && _option==NONE) {
			if (currentMode) {
				area = ((hNmos*wNmos)*48 + (hPmos*wPmos)*24)*(levelOutput-1)*numCol;
			} else {
				area = ((hNmos*wNmos)*52 + (hPmos*wPmos)*60)*(levelOutput-1)*numCol;
			}
			width = widthArray;
			height = area / width;
		} else if (heightArray && _option==NONE) {
			if (currentMode) {
				area = ((hNmos*wNmos)*48 + (hPmos*wPmos)*24)*(levelOutput-1)*numCol;
			} else {
				area = ((hNmos*wNmos)*52 + (hPmos*wPmos)*60)*(levelOutput-1)*numCol;
			}
			height = heightArray;
			width = area / height;
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

void MultilevelSenseAmp::CalculateLatency(const vector<double> &columnResistance, double numColMuxed, double numRead) {
	if (!initialized) {
		cout << "[MultilevelSenseAmp] Error: Require initialization first!" << endl;
	} else {
		readLatency = 0;
		double LatencyCol = 0;
		for (double j=0; j<columnResistance.size(); j++){
			double T_Col = 0;
			T_Col = GetColumnLatency(columnResistance[j]);
			if (columnResistance[j] == columnResistance[j]) {
				LatencyCol = max(LatencyCol, T_Col);
			} else {
				LatencyCol = LatencyCol;
			}
			if (LatencyCol < 1e-9) {
				LatencyCol = 1e-9;
			} else if (LatencyCol > 10e-9) {
				LatencyCol = 10e-9;
			}
		}
		if (currentMode) {
			readLatency = LatencyCol*numColMuxed;
			readLatency *= numRead;
		} else {
			readLatency = (1e-9)*numColMuxed;
			readLatency *= numRead;
		}
		
	}
}

void MultilevelSenseAmp::CalculatePower(const vector<double> &columnResistance, double numRead) {
	if (!initialized) {
		cout << "[MultilevelSenseAmp] Error: Require initialization first!" << endl;
	} else {
		leakage = 0;
		readDynamicEnergy = 0;
		
		double LatencyCol = 0;
		for (double j=0; j<columnResistance.size(); j++){
			double T_Col = 0;
			T_Col = GetColumnLatency(columnResistance[j]);
			if (columnResistance[j] == columnResistance[j]) {
				LatencyCol = max(LatencyCol, T_Col);
			} else {
				LatencyCol = LatencyCol;
			}
			if (LatencyCol < 1e-9) {
				LatencyCol = 1e-9;
			} else if (LatencyCol > 10e-9) {
				LatencyCol = 10e-9;
			}
		}

		for (double i=0; i<columnResistance.size(); i++) {
			double P_Col = 0;
			P_Col = GetColumnPower(columnResistance[i]);
			if (currentMode) {
				readDynamicEnergy += MAX(P_Col*LatencyCol, 0);
			} else {
				readDynamicEnergy += MAX(P_Col*1e-9, 0);
			}
		}
		readDynamicEnergy *= numRead;
	}
} 

void MultilevelSenseAmp::PrintProperty(const char* str) {
	FunctionUnit::PrintProperty(str);
}


double MultilevelSenseAmp::GetColumnLatency(double columnRes) {
	double Column_Latency = 0;
	double up_bound = 3, mid_bound = 1.1, low_bound = 0.9;
	double T_max = 0;
	// in Cadence simulation, we fix Vread to 0.5V, with user-defined Vread (different from 0.5V)
	// we should modify the equivalent columnRes
	columnRes *= 0.5/param->readVoltage;
	if (((double) 1/columnRes == 0) || (columnRes == 0)) {
		Column_Latency = 0;
	} else {
		if (param->deviceroadmap == 1) {  // HP
			Column_Latency = 1e-9;
		} else {                         // LP
			if (param->technode == 130) {
				T_max = (0.2679*log(columnRes/1000)+0.0478)*1e-9;   // T_max = (0.2679*log(R_BL/1000)+0.0478)*10^-9;

				for (int i=1; i<levelOutput-1; i++){
					double ratio = Rref[i]/columnRes;
					double T = 0;
					if (ratio >= 20 || ratio <= 0.05) {
						T = 1e-9;
					} else {
						if (ratio <= low_bound){
							T = T_max * (3.915*pow(ratio,3)-5.3996*pow(ratio,2)+2.4653*ratio+0.3856);  // y = 3.915*x^3-5.3996*x^2+2.4653*x+0.3856;
						} else if (mid_bound <= ratio <= up_bound){
							T = T_max * (0.0004*pow(ratio,4)-0.0087*pow(ratio,3)+0.0742*pow(ratio,2)-0.2725*ratio+1.2211);  // y = 0.0004*x^4-0.0087*x^3+0.0742*x^2-0.2725*x+1.2211;
						} else if (ratio>up_bound){
							T = T_max * (0.0004*pow(ratio,4)-0.0087*pow(ratio,3)+0.0742*pow(ratio,2)-0.2725*ratio+1.2211);
						} else {
							T = T_max;
						}
					}
					Column_Latency = max(Column_Latency, T);
				}
			} else if (param->technode == 90) {
				T_max = (0.0586*log(columnRes/1000)+1.41)*1e-9;   // T_max = (0.0586*log(R_BL/1000)+1.41)*10^-9;

				for (int i=1; i<levelOutput-1; i++){
					double ratio = Rref[i]/columnRes;
					double T = 0;
					if (ratio >= 20 || ratio <= 0.05) {
						T = 1e-9;
					} else {
						if (ratio <= low_bound){
							T = T_max * (3.726*pow(ratio,3)-5.651*pow(ratio,2)+2.8249*ratio+0.3574);    // y = 3.726*x^3-5.651*x^2+2.8249*x+0.3574;
						} else if (mid_bound <= ratio <= up_bound){
							T = T_max * (0.0000008*pow(ratio,4)-0.00007*pow(ratio,3)+0.0017*pow(ratio,2)-0.0188*ratio+0.9835);  // y = 0.0000008*x^4-0.00007*x^3+0.0017*x^2-0.0188*x+0.9835;
						} else if (ratio>up_bound){
							T = T_max * (0.0000008*pow(ratio,4)-0.00007*pow(ratio,3)+0.0017*pow(ratio,2)-0.0188*ratio+0.9835);
						} else {
							T = T_max;
						}
					}
					Column_Latency = max(Column_Latency, T);
				}
			} else if (param->technode == 65) {
				T_max = (0.1239*log(columnRes/1000)+0.6642)*1e-9;   // T_max = (0.1239*log(R_BL/1000)+0.6642)*10^-9;

				for (int i=1; i<levelOutput-1; i++){
					double ratio = Rref[i]/columnRes;
					double T = 0;
					if (ratio >= 20 || ratio <= 0.05) {
						T = 1e-9;
					} else {
						if (ratio <= low_bound){
							T = T_max * (1.3899*pow(ratio,3)-2.6913*pow(ratio,2)+2.0483*ratio+0.3202);    // y = 1.3899*x^3-2.6913*x^2+2.0483*x+0.3202;
						} else if (mid_bound <= ratio <= up_bound){
							T = T_max * (0.0036*pow(ratio,4)-0.0363*pow(ratio,3)+0.1043*pow(ratio,2)-0.0346*ratio+1.0512);   // y = 0.0036*x^4-0.0363*x^3+0.1043*x^2-0.0346*x+1.0512;
						} else if (ratio>up_bound){
							T = T_max * (0.0036*pow(ratio,4)-0.0363*pow(ratio,3)+0.1043*pow(ratio,2)-0.0346*ratio+1.0512);
						} else {
							T = T_max;
						}
					}
					Column_Latency = max(Column_Latency, T);
				}
			} else if (param->technode == 45 || param->technode == 32) {
				T_max = (0.0714*log(columnRes/1000)+0.7651)*1e-9;    // T_max = (0.0714*log(R_BL/1000)+0.7651)*10^-9;

				for (int i=1; i<levelOutput-1; i++){
					double ratio = Rref[i]/columnRes;
					double T = 0;
					if (ratio >= 20 || ratio <= 0.05) {
						T = 1e-9;
					} else {
						if (ratio <= low_bound){
							T = T_max * (3.7949*pow(ratio,3)-5.6685*pow(ratio,2)+2.6492*ratio+0.4807);    // y = 3.7949*x^3-5.6685*x^2+2.6492*x+0.4807
						} else if (mid_bound <= ratio <= up_bound){
							T = T_max * (0.000001*pow(ratio,4)-0.00006*pow(ratio,3)+0.0001*pow(ratio,2)-0.0171*ratio+1.0057);   // 0.000001*x^4-0.00006*x^3+0.0001*x^2-0.0171*x+1.0057;
						} else if (ratio>up_bound){
							T = T_max * (0.000001*pow(ratio,4)-0.00006*pow(ratio,3)+0.0001*pow(ratio,2)-0.0171*ratio+1.0057);
						} else {
							T = T_max;
						}
					}
					Column_Latency = max(Column_Latency, T);
				}
			} else {   // technode below and equal to 22nm
				Column_Latency = 1e-9;
			}
		}
	}
	return Column_Latency;
}



double MultilevelSenseAmp::GetColumnPower(double columnRes) {
	double Column_Power = 0;
	// in Cadence simulation, we fix Vread to 0.5V, with user-defined Vread (different from 0.5V)
	// we should modify the equivalent columnRes
	columnRes *= 0.5/param->readVoltage;
	if (currentMode) {
		if ((double) 1/columnRes == 0) { 
			Column_Power = 1e-6;
		} else if (columnRes == 0) {
			Column_Power = 0;
		} else {
			if (param->deviceroadmap == 1) {  // HP
				if (param->technode == 130) {
					Column_Power = 19.898*(levelOutput-1)*1e-6;
					Column_Power += 0.17452*exp(-2.367*log10(columnRes));
				} else if (param->technode == 90) {
					Column_Power = 13.09*(levelOutput-1)*1e-6;
					Column_Power += 0.14900*exp(-2.345*log10(columnRes));
				} else if (param->technode == 65) {
					Column_Power = 9.9579*(levelOutput-1)*1e-6;
					Column_Power += 0.1083*exp(-2.321*log10(columnRes));
				} else if (param->technode == 45) {
					Column_Power = 7.7017*(levelOutput-1)*1e-6;
					Column_Power += 0.0754*exp(-2.296*log10(columnRes));
				} else if (param->technode == 32){  
					Column_Power = 3.9648*(levelOutput-1)*1e-6;
					Column_Power += 0.079*exp(-2.313*log10(columnRes));
				} else if (param->technode == 22){   
					Column_Power = 1.8939*(levelOutput-1)*1e-6;
					Column_Power += 0.073*exp(-2.311*log10(columnRes));
				} else if (param->technode == 14){  
					Column_Power = 1.2*(levelOutput-1)*1e-6;
					Column_Power += 0.0584*exp(-2.311*log10(columnRes));
				} else if (param->technode == 10){  
					Column_Power = 0.8*(levelOutput-1)*1e-6;
					Column_Power += 0.0318*exp(-2.311*log10(columnRes));
				} else {   // 7nm
					Column_Power = 0.5*(levelOutput-1)*1e-6;
					Column_Power += 0.0210*exp(-2.311*log10(columnRes));
				}
			} else {                         // LP
				if (param->technode == 130) {
					Column_Power = 18.09*(levelOutput-1)*1e-6;
					Column_Power += 0.1380*exp(-2.303*log10(columnRes));
				} else if (param->technode == 90) {
					Column_Power = 12.612*(levelOutput-1)*1e-6;
					Column_Power += 0.1023*exp(-2.303*log10(columnRes));
				} else if (param->technode == 65) {
					Column_Power = 8.4147*(levelOutput-1)*1e-6;
					Column_Power += 0.0972*exp(-2.303*log10(columnRes));
				} else if (param->technode == 45) {
					Column_Power = 6.3162*(levelOutput-1)*1e-6;
					Column_Power += 0.075*exp(-2.303*log10(columnRes));
				} else if (param->technode == 32){  
					Column_Power = 3.0875*(levelOutput-1)*1e-6;
					Column_Power += 0.0649*exp(-2.297*log10(columnRes));
				} else if (param->technode == 22){   
					Column_Power = 1.7*(levelOutput-1)*1e-6;
					Column_Power += 0.0631*exp(-2.303*log10(columnRes));
				} else if (param->technode == 14){   
					Column_Power = 1.0*(levelOutput-1)*1e-6;
					Column_Power += 0.0508*exp(-2.303*log10(columnRes));
				} else if (param->technode == 10){   
					Column_Power = 0.55*(levelOutput-1)*1e-6;
					Column_Power += 0.0315*exp(-2.303*log10(columnRes));
				} else {   // 7nm
					Column_Power = 0.35*(levelOutput-1)*1e-6;
					Column_Power += 0.0235*exp(-2.303*log10(columnRes));
				}
			}
		}
		
	} else {
		if ((double) 1/columnRes == 0) { 
			Column_Power = 1e-6;
		} else if (columnRes == 0) {
			Column_Power = 0;
		} else {
			if (param->deviceroadmap == 1) {  // HP
				if (param->technode == 130) {
					Column_Power = 27.84*(levelOutput-1)*1e-6;
					Column_Power += 0.207452*exp(-2.367*log10(columnRes));
				} else if (param->technode == 90) {
					Column_Power = 22.2*(levelOutput-1)*1e-6;
					Column_Power += 0.164900*exp(-2.345*log10(columnRes));
				} else if (param->technode == 65) {
					Column_Power = 13.058*(levelOutput-1)*1e-6;
					Column_Power += 0.128483*exp(-2.321*log10(columnRes));
				} else if (param->technode == 45) {
					Column_Power = 8.162*(levelOutput-1)*1e-6;
					Column_Power += 0.097754*exp(-2.296*log10(columnRes));
				} else if (param->technode == 32){  
					Column_Power = 4.76*(levelOutput-1)*1e-6;
					Column_Power += 0.083709*exp(-2.313*log10(columnRes));
				} else if (param->technode == 22){   
					Column_Power = 2.373*(levelOutput-1)*1e-6;
					Column_Power += 0.084273*exp(-2.311*log10(columnRes));
				} else if (param->technode == 14){  
					Column_Power = 1.467*(levelOutput-1)*1e-6;
					Column_Power += 0.060584*exp(-2.311*log10(columnRes));
				} else if (param->technode == 10){  
					Column_Power = 0.9077*(levelOutput-1)*1e-6;
					Column_Power += 0.049418*exp(-2.311*log10(columnRes));
				} else {   // 7nm
					Column_Power = 0.5614*(levelOutput-1)*1e-6;
					Column_Power += 0.040310*exp(-2.311*log10(columnRes));
				}
			} else {                         // LP
				if (param->technode == 130) {
					Column_Power = 23.4*(levelOutput-1)*1e-6;
					Column_Power += 0.169380*exp(-2.303*log10(columnRes));
				} else if (param->technode == 90) {
					Column_Power = 14.42*(levelOutput-1)*1e-6;
					Column_Power += 0.144323*exp(-2.303*log10(columnRes));
				} else if (param->technode == 65) {
					Column_Power = 10.18*(levelOutput-1)*1e-6;
					Column_Power += 0.121272*exp(-2.303*log10(columnRes));
				} else if (param->technode == 45) {
					Column_Power = 7.062*(levelOutput-1)*1e-6;
					Column_Power += 0.100225*exp(-2.303*log10(columnRes));
				} else if (param->technode == 32){  
					Column_Power = 3.692*(levelOutput-1)*1e-6;
					Column_Power += 0.079449*exp(-2.297*log10(columnRes));
				} else if (param->technode == 22){   
					Column_Power = 1.866*(levelOutput-1)*1e-6;
					Column_Power += 0.072341*exp(-2.303*log10(columnRes));
				} else if (param->technode == 14){   
					Column_Power = 1.126*(levelOutput-1)*1e-6;
					Column_Power += 0.061085*exp(-2.303*log10(columnRes));
				} else if (param->technode == 10){   
					Column_Power = 0.6917*(levelOutput-1)*1e-6;
					Column_Power += 0.051580*exp(-2.303*log10(columnRes));
				} else {   // 7nm
					Column_Power = 0.4211*(levelOutput-1)*1e-6;
					Column_Power += 0.043555*exp(-2.303*log10(columnRes));
				}
			}
		}
	}
	Column_Power *= (1+1.3e-3*(param->temp-300));
	return Column_Power;
}
