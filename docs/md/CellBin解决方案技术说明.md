# CellBin解决方案技术说明

**引言**

本文档分为两大部分：“Cellbin的介绍说明”，“Cellbin的使用方法”，旨在帮助用户更好的理解“Cellbin”，并提供获得细胞精度级别的基因矩阵的解决方案。使用此文档的用户需要提前获得测序数据以及生物影像图。

**CellBin**

**CellBin框架说明**

CellBin是一套将时空测序数据与生物影像图相结合，最后获得细胞级别精度的基因矩阵数据的流程，它嵌入在StereoMap和SAW流程当中，关系如下：

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/1X3lE63B7m04nJbv/img/335efa9d-3b54-4d7f-8948-c026929451e3.png)

CellBin流程包含QC，拼接，配准，组织分割，细胞分割，细胞修正，生成cellbin矩阵这几个模块。其中QC模块与图像手动处理部分（详见《CellBin解决方案技术说明》中的QC介绍）放在StereoMap v4软件中执行，剩余模块在Stereo-seq Analysis Workflow (SAW) 流程中执行。

**StereoMap v4：** StereoMap 是一款无需编程的桌面端软件，包括 Stereo-seq 数据交互式的可视化、图像的手动处理及小工具三个模块。图像可视化兼容了之前StereoMap3.0的功能，图像处理和小工具兼容 ImageStudio 的部分功能，修改为 Step by step 的方式，降低用户的学习成本，提升用户满意度。

**Stereo-seq Analysis Workflow (SAW) ：** SAW软件套件是一套捆绑式pipelines，用于将测序读数定位到其在组织切片上的空间位置，量化空间特征表达，并直观地呈现空间表达分布。

**QC**：对生物影像图中拍摄到的芯片 track 线进行质检，判断是否达到后续自动化配准模块要求

**拼接**：如果提供的影像图为拼接好的大图，则跳过这一步。若提供的影像图为显微镜小图，则会将小图拼接成一张大图。

**配准**：基于track线的高精度自动化配准分为两部分：第一部分：有 track 且 QC 通过的情况下，通过track线将计算出图像相对表达矩阵的缩放尺度，以及旋转角度；第二部分：拼接好的影像大图与测序下机后的基因生成的基因可视化图（gene）的做形态学上配准，得到90度倍数的旋转或是翻转，以及超过位移参数。

**组织分割**：对影像图中的生物切片的组织轮廓做分割。

**细胞分割** ：对影像图中的生物切片的细胞核/膜轮廓做分割。

**细胞修正** ：将分割后的细胞轮廓往外扩大10pixels（默认），作为新的细胞分割区域结果。

以上算法模块的详细介绍可以看：[https://www.biorxiv.org/content/10.1101/2023.02.28.530414v5](https://www.biorxiv.org/content/10.1101/2023.02.28.530414v5)

**CellBin使用**

CellBin的使用**需要分别执行**Stereomap v4中**QC**以及SAW中的**SAW count，**才能获得最终Cellbin结果。

以**ssDNA Demo：SS200000135TL\_D1** 为例：

**数据链接：**

[http://116.6.21.110:8090/share/21bb9df9-e6c5-47c5-9aa8-29f2d23a6df4](http://116.6.21.110:8090/share/21bb9df9-e6c5-47c5-9aa8-29f2d23a6df4)

**QC输入：** 显微镜软件输出的大图，文件格式为TIFF (.tif、.tiff)、BigTIFF图像格式。

若只使用Demo数据体验流程，可根据这份简易文档逐步操作：_[《stereomap4.0版本手动方案》](https://alidocs.dingtalk.com/i/nodes/QOG9lyrgJP3d2vbofrad092yVzN67Mw4?utm_scene=team_space)_；若想了解更详细的流程与信息，以下附上了执行QC和SAW的完整操作。
    

*   **如何执行QC：**
    

如何使用Stereomap 中的QC功能，可阅读操作文档：[https://stereotoolss-organization.gitbook.io/stereomap-user-manual/tutorials/navigation-for-tools/stereo-seq-image-qc](https://stereotoolss-organization.gitbook.io/stereomap-user-manual/tutorials/navigation-for-tools/stereo-seq-image-qc)

如何使用Stereomap v4中的其他功能，可阅读操作文档：[https://stereotoolss-organization.gitbook.io/stereomap-user-manual/download](https://stereotoolss-organization.gitbook.io/stereomap-user-manual/download)


_图像QC后，如果QC成功，在执行SAW count时需要在这个可选参数__**"--image-tar "**_ _输入QC后的tar.gz文件。若QC失败，则不使用该参数。QC成功和失败的情况可在_[《CellBin解决方案技术说明》](https://alidocs.dingtalk.com/i/nodes/2Amq4vjg89gGdom7FL6njLn7V3kdP0wQ?utm_scene=team_space&iframeQuery=anchorId%3Duu_lyxswww8i7ki45uluw)_查看。_

*   **如何执行SAW count：**
    

如何使用SAW count，可阅读操作文档：[https://stereotoolss-organization.gitbook.io/saw-user-manual-v8.0/analysis/pipelines/saw-commands](https://stereotoolss-organization.gitbook.io/saw-user-manual-v8.0/analysis/pipelines/saw-commands)

如何使用SAW中的其他功能，可阅读详细操作文档：[https://stereotoolss-organization.gitbook.io/saw-user-manual-v8.0](https://stereotoolss-organization.gitbook.io/saw-user-manual-v8.0)


**怎么查看CellBin结果**

**QC情况**

**QC失败：** 在QC时如果软件界面出现以下情况 “The results of this image analysis evaluation are: FAIL”，则表示QC失败。

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/1X3lE63B7m04nJbv/img/30b658a3-fca5-488e-b164-a692402f1eed.png)

若图像QC失败，则只能执行SAW count无图流程，即不使用_**"--image-tar "**_ 参数。流程结束后则，若需要图像结果，需要用Stereomap V4进行手动处理。

**QC成功：** 在QC时如果软件界面出现以下情况 “The results of this image analysis evaluation are: **PASS**”，则表示QC成功。

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/1X3lE63B7m04nJbv/img/51d6c550-ef6f-4f54-85b9-e23fc3bdb1ba.png)

若图像QC成功，需要跑SAW count有图流程，即增加_**"--image-tar "**_ 参数。

跑完SAW count后，流程结果如下：

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/1X3lE63B7m04nJbv/img/ad6b4e52-8ecd-4f72-918e-45c35eae7bfc.png)![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/1X3lE63B7m04nJbv/img/2581b561-2392-4518-8667-5de32716e13e.png)

**如何看结果**

下载outs文件夹中的“SN.report.tar.gz”到本地解压打开。 ![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/1X3lE63B7m04nJbv/img/5f6c957e-e5ac-439b-8669-909e379e73de.png)

文件夹中的“report.html”文件为本次流程结果的统计报告。打开报告后可以按照这份说明文档查看结果。[《 如何查看Cellbin指标》](https://alidocs.dingtalk.com/i/nodes/ZX6GRezwJl7doqRbHXdANOABVdqbropQ?utm_scene=team_space)

**手动处理图像结果。**

如果在上述查看结果的问题遇到了QC失败，配准异常，组织分割异常，细胞分割异常等图像问题，可使用StereoMap中的手动处理工具进行修改，再次接回SAW流程，操作请看SOP：[《Cellbin流程手动操作》](https://alidocs.dingtalk.com/i/nodes/QG53mjyd80RbdBYktXMZoD1lV6zbX04v?utm_scene=team_space)