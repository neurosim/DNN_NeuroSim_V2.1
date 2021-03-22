# DNN+NeuroSim V2.1

The DNN+NeuroSim framework was developed by [Prof. Shimeng Yu's group](https://shimeng.ece.gatech.edu/) (Georgia Institute of Technology). The model is made publicly available on a non-commercial basis. Copyright of the model is maintained by the developers, and the model is distributed under the terms of the [Creative Commons Attribution-NonCommercial 4.0 International Public License](http://creativecommons.org/licenses/by-nc/4.0/legalcode)

This is the released version 2.1 (Aug 8, 2020) for the tool.

:star2: This V2.1 has **_improved following estimation_**:
```
1. Calibrate FinFET technology library: temperature-related features.
2. Calibrate FinFET technology layout features.
3. Include FeFET polarization during weight-update.
```
:star2: This version has also added **_new features into inference accuracy estimation_**:
```
1. Introduce VSA-bsaed MLSA (in addition to the original CSA-based MLSA)
2. Introduce SAR ADC
```
:point_right: :point_right: :point_right: **In "Param.cpp", to switch ADC mode:**
```
SARADC = false;           // false: MLSA            // true: sar ADC
currentMode = true;       // false: MLSA use VSA    // true: MLSA use CSA
```

<br/>

**_For estimation of inference engine, please visit released V1.3 [DNN+NeuroSim V1.3](https://github.com/neurosim/DNN_NeuroSim_V1.3)_**

In V2.1, we currently only support Pytorch wrapper, where users are able to define **_network structures, parameter precisions and hardware non-ideal properties_**. With the integrated NeuroSim which takes real traces from wrapper, the framework can support hierarchical organization from device level to circuit level, to chip level and to algorithm level, enabling **_instruction-accurate evaluation on both accuracy and hardware performance of on-chip training accelerator_**.

The default example is VGG-8 for CIFAR-10 in this framework:
    1. Users can modify weight/error/gradient/activation precisions for training, while _some adjusts of "beta" value in function "scale_limit" from file "wage_initializer.py" could be necessary, corresponding values have been provided as example_.
    2. We temporally only provide the option where: weight/error/gradient precision equals to cell-precision, i.e. one-cell-per-synapse scheme for training evaluation.

Due to additional functions (of non-ideal properties) being implemented in the framework, please expect ~12 hours simulation time for whole training process (default network VGG-8 for CIFAR-10, with 256 epochs).  

Developers: [Xiaochen Peng](mailto:xpeng76@gatech.edu) :two_women_holding_hands: [Shanshi Huang](mailto:shuang406@gatech.edu).

This research is supported by NSF CAREER award, NSF/SRC E2CDA program, and ASCENT, one of the SRC/DARPA JUMP centers.

If you use the tool or adapt the tool in your work or publication, you are required to cite the following reference:

**_X. Peng, S. Huang, H. Jiang, A. Lu and S. Yu, ※[DNN+NeuroSim V2.0: An End-to-End Benchmarking Framework for Compute-in-Memory Accelerators for Training](https://ieeexplore-ieee-org.prx.library.gatech.edu/document/9292971), *§ IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems,* doi: 10.1109/TCAD.2020.3043731, 2020._**

**_X. Peng, S. Huang, Y. Luo, X. Sun and S. Yu, ※[DNN+NeuroSim: An End-to-End Benchmarking Framework for Compute-in-Memory Accelerators with Versatile Device Technologies](https://ieeexplore-ieee-org.prx.library.gatech.edu/document/8993491), *§ IEEE International Electron Devices Meeting (IEDM)*, 2019._**

If you have logistic questions or comments on the model, please contact :man: [Prof. Shimeng Yu](mailto:shimeng.yu@ece.gatech.edu), and if you have technical questions or comments, please contact :woman: [Xiaochen Peng](mailto:xpeng76@gatech.edu) or :woman: [Shanshi Huang](mailto:shuang406@gatech.edu).


## File lists
1. Manual: `Documents/DNN NeuroSim V2.1 Manual.pdf`
2. Nonlinearity-to-A table: `Documents/Nonlinearity-NormA.htm`
3. MATLAB fitting script: `Documents/nonlinear_fit.m`
4. DNN_NeuroSim wrapped by Pytorch: `Training_pytorch`
5. NeuroSim under Pytorch Inference: `Training_pytorch/NeuroSIM`


## Installation steps (Linux)
1. Get the tool from GitHub
```
git clone https://github.com/neurosim/DNN_NeuroSim_V2.1.git
```

2. Set up hardware parameters in NeuroSim Core and compile the Code
```
make
```

3. Set up hardware constraints in Python wrapper (train.py)

4. Run Pytorch wrapper (integrated with NeuroSim)

5. A list of simulation results are expected as below:
  - Input activity of every layer for each epoch: `input_activity.csv`
  - Weight distribution parameters (mean and std) of every layer for each epoch: `weight_dist.csv`
  - Delta weight distribution parameters (mean and std) of every layer for each epoch: `delta_dist.csv`
  - Estimation of average loss and accuracy for each epoch: `PythonWrapper_Output.csv`
  - Estimation of on-chip training system for each epoch: `NeuroSim_Output.csv`
  - Detailed breakdowns of estimation of on-chip training system for each epoch:   
    `NeuroSim_Results_Each_Epoch/NeuroSim_Breakdown_Epoch_0.csv` 
    `NeuroSim_Results_Each_Epoch/NeuroSim_Breakdown_Epoch_1.csv  ... ... `
    `NeuroSim_Results_Each_Epoch/NeuroSim_Breakdown_Epoch_256.csv`

6. For the usage of this tool, please refer to the user manual.

## References related to this tool 
1. X. Peng, S. Huang, H. Jiang, A. Lu and S. Yu, ※DNN+NeuroSim V2.0: An End-to-End Benchmarking Framework for Compute-in-Memory Accelerators for On-chip Training, *§ IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, doi: 10.1109/TCAD.2020.3043731*, 2020.
2. X. Peng, S. Huang, Y. Luo, X. Sun and S. Yu, ※DNN+NeuroSim: An End-to-End Benchmarking Framework for Compute-in-Memory Accelerators with Versatile Device Technologies, *§ IEEE International Electron Devices Meeting (IEDM)*, 2019.
3. X. Peng, R. Liu, S. Yu, ※Optimizing weight mapping and data flow for convolutional neural networks on RRAM based processing-in-memory architecture, *§ IEEE International Symposium on Circuits and Systems (ISCAS)*, 2019.
4. P.-Y. Chen, S. Yu, ※Technological benchmark of analog synaptic devices for neuro-inspired architectures, *§ IEEE Design & Test*, 2019.
5. P.-Y. Chen, X. Peng, S. Yu, ※NeuroSim: A circuit-level macro model for benchmarking neuro-inspired architectures in online learning, *§ IEEE Trans. CAD*, 2018.
6. X. Sun, S. Yin, X. Peng, R. Liu, J.-S. Seo, S. Yu, ※XNOR-RRAM: A scalable and parallel resistive synaptic architecture for binary neural networks,*§ ACM/IEEE Design, Automation & Test in Europe Conference (DATE)*, 2018.
7. P.-Y. Chen, X. Peng, S. Yu, ※NeuroSim+: An integrated device-to-algorithm framework for benchmarking synaptic devices and array architectures, *§ IEEE International Electron Devices Meeting (IEDM)*, 2017.
8. P.-Y. Chen, S. Yu, ※Partition SRAM and RRAM based synaptic arrays for neuro-inspired computing,*§ IEEE International Symposium on Circuits and Systems (ISCAS)*, 2016.
9. P.-Y. Chen, D. Kadetotad, Z. Xu, A. Mohanty, B. Lin, J. Ye, S. Vrudhula, J.-S. Seo, Y. Cao, S. Yu, ※Technology-design co-optimization of resistive cross-point array for accelerating learning algorithms on chip,*§ IEEE Design, Automation & Test in Europe (DATE)*, 2015.
10. S. Wu, et al., ※Training and inference with integers in deep neural networks,*§ arXiv: 1802.04680*, 2018.
11. github.com/boluoweifenda/WAGE
12. github.com/stevenygd/WAGE.pytorch
13. github.com/aaron-xichen/pytorch-playground
