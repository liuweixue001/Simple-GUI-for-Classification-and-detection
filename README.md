# Simple-GUI-for-Classification-and-detection
a simple gui demo used vs+qt+opencv+ncnn for Classification and detection

版本：vs2019(需要下载141工具集)，qt5.12(msvc2017_64)，opencv4.4，ncnn最新版

vs里面选用qt模板直接开搞

进入ui界面，拖拽出自己喜欢的界面

进入qfile.h和qfile.cpp文件，将ui修改为指针类型

直接使用ui->调用ui中生成的组件(注：ui拖拽完成后，先进行界面生成，才可以找到相应的组件)

配置键和槽，键为组件按钮，槽为调用的目标分类和检测函数

只需要调用操作qfile.h和qfile.cpp文件即可，.bin和.param文件为训练好的ncnn模型，其他文件为系统自动生成。

大部分文件为无效文件，操作不便，没有删除

