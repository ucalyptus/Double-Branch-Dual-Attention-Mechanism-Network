


Requirements：
------- 
```
numpy >= 1.16.5
PyTorch >= 1.3.1
sklearn >= 0.20.4
```

Datasets:
------- 
You can download the hyperspectral datasets in mat format at: http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes, and move the files to `./datasets` folder.

Usage:
------- 
1. Set the percentage of training and validation samples by the `load_dataset` function in the file `./global_module/generate_pic.py`.
2. Taking the DBDA framework as an example, run `./DBDA/main.py` and type the name of dataset. 
3. The classfication maps are obtained in `./DBDA/classification_maps` folder, and accuracy result is generated in `./DBDA/records` folder.

Network:
------- 
* [DBDA](https://www.mdpi.com/2072-4292/12/3/582)
* [DBMA](https://www.mdpi.com/2072-4292/11/11/1307/xml)
* [FDSSC](https://www.mdpi.com/2072-4292/10/7/1068/htm)
* [SSRN](https://ieeexplore.ieee.org/document/8061020)
* [CDCNN](https://ieeexplore.ieee.org/document/7973178)
* [SVM](https://ieeexplore.ieee.org/document/1323134/)

![network](https://github.com/lironui/Double-Branch-Dual-Attention-Mechanism-Network/blob/master/figures/Figure%206.%20The%20structure%20of%20the%20DBDA%20network.png)  
Figure 1. The structure of the DBDA network. The upper Spectral Branch composed of the dense 
spectral block and channel attention block is designed to capture spectral features. The lower Spatial 
Branch constituted by dense spatial block, and spatial attention block is designed to exploit spatial 
features. 
