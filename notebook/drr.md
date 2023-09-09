
DRR计算
----------

> From <https://github.com/SlicerRt/SlicerRT/blob/907ef2e9e49db5216c4f3fb15a97379362992af3/Docs/user_guide/modules/drr.md>

数字重建X光片（DRR）是一种合成X光片，其 可通过计算机断层扫描 （CT） 生成。它被用作参考 用于在放射治疗前验证患者正确设置位置的图像。

DRR计算过程由两个模块组成：命令行界面模块和 图形用户界面可加载模块。

### 命令行界面模块 （CLI）
命令行界面模块（CLI）

[![image](https://user-images.githubusercontent.com/3785912/107616458-2c8c0980-6c5f-11eb-923e-fafb55a0e61f.png)](https://user-images.githubusercontent.com/3785912/107616458-2c8c0980-6c5f-11eb-923e-fafb55a0e61f.png)

CLI 模块"slicer\_plastimatch\_drr"将 CT 图像作为输入，并生成一个 根据探测器方向、光源距离、 等中心位置和图像分辨率和间距。这些参数中的大多数是 基于[`plastimatch`的 DRR](http://www.plastimatch.org/drr.html) 的。

#### 输入和输出卷参数
输入和输出体积参数

1.  CT 输入**volume：输入 CT** 数据2.  **DRR 输出体积：输出 DRR** 图像

#### 源参数 震源参数

1.  **SAD**：X 射线源与旋转轴（等中心）之间的距离，单位为 mm2.  **SID**：X 射线源和探测器之间的距离，单位为 mm （SID >= SAD）

#### 探测器方向参数
探测器方位参数

1.  **向上查看矢量**：在 LPS 坐标系中向上查看探测器第一行的矢量2.  法线向量：LPS坐标系中的探测器表面**法向量**

#### CT扫描参数 CT扫描参数

1.  **等中心位置**：LPS 坐标系中的等中心位置
    等中心位置：LPS坐标系中的等中心位置

#### 探测器和图像参数
探测器和图像参数

1.  分辨率：以像素为单位的检测器
    分辨率：检测器分辨率（以像素为单位）（列、行）**分辨率**（列、行）2.  **间距**：以毫米为单位的检测器间距（列、行）3.  使用图像窗口：是**使用图像子窗口**还是整个检测器4.  **图像窗口**：将 DRR 输出限制为子窗口（列 1、行 1、列 2、行 2）

#### 加工参数 处理参数

1.  自动重新调整强度：**自动重新调整强度**范围2.  自动缩放范围：用于表单**自动缩放**的范围（最小值、最大值）3.  指数映射：应用输出值的**指数映射**4.  **线程**类型：计算的并行性类型5.  **HU 转换类型**：计算期间霍恩斯菲尔德单位转换的类型6.  曝光**类型：曝光**算法的类型7.  输出格式：正式
    输出格式：输出文件格式的类型（3DSlicer为"raw"）**输出**文件的类型（3DSlicer为"原始"）8.  **反**转强度：在数据后处理期间应用强度反转。
    反转强度：在数据后处理期间应用强度反演。9.  低于
    低于阈值：低于阈值的HU值计为-1000（空气值）**阈值：低于**阈值的 HU 值计为 -1000（空气值）

### 图形用户界面可加载模块 （GUI）
图形用户界面可加载模块（GUI）

[![image](https://user-images.githubusercontent.com/3785912/118298680-726b9e80-b4e8-11eb-8a39-c5e607459e62.png)](https://user-images.githubusercontent.com/3785912/118298680-726b9e80-b4e8-11eb-8a39-c5e607459e62.png)

可加载的 GUI 模块"DRR 图像计算"使用 CLI 模块的逻辑和节点数据进行计算 并可视化 DRR 图像。它还显示了基本的检测器元素，例如：检测器边界， 探测器法线矢量， 探测器俯视矢量， 探测器图像原点（0,0）像素， 图像子窗口边界作为切片和 3D 视图上的标记数据。
可加载的GUI模块"DRR图像计算"使用CLI模块的逻辑和节点数据来计算和可视化DRR图像。它还显示了基本的探测器元件，例如：探测器边界、探测器法向量、探测器向上观察向量、探测器图像原点（0，0）像素、图像子窗口边界作为切片和3D视图上的标记数据。

标记数据仅用于所见即所得的目的。
标记数据仅用于WYSIWYG目的。

#### 参考输入节点 参考输入节点

[![image](https://user-images.githubusercontent.com/3785912/118298709-7e576080-b4e8-11eb-8c2b-b1a4f8222eba.png)](https://user-images.githubusercontent.com/3785912/118298709-7e576080-b4e8-11eb-8c2b-b1a4f8222eba.png)

1.  CT 体积：输入 **CT** 数据2.  RT 光束：输入 **RT 光束** （vtkMRMLRTBeamNode） 用于源和探测器方向参数3.  相机：**相机**节点 （vtkMRMLCameraNode），用于在需要时更新光束几何形状和变换4.  **更新**光束：使用相机节点数据更新光束几何形状和变换5.  **显示 DRR** 标记：显示或隐藏检测器标记

CT 数据由 vtkMRMLScalarVolumeNode 对象表示。RT光束数据是 由 vtkMRMLRTBeamNode 对象表示。相机数据由 vtkMRMLCameraNode 对象表示。

#### 几何基本参数

[![image](https://user-images.githubusercontent.com/3785912/107616499-40d00680-6c5f-11eb-9d45-c6cccc9e12cd.png)](https://user-images.githubusercontent.com/3785912/107616499-40d00680-6c5f-11eb-9d45-c6cccc9e12cd.png)

1.  等中心**到成像器的距离**：从等中心到检测器中心的距离，单位为mm2.  成像器分辨率**（列、行）：**以像素为单位的检测器分辨率3.  以毫米为单位的成像器间距**（列、行）：以毫米为单位的检测器间距**4.  图像窗口参数：使用和设置**图像子窗口**或整个检测器

#### 图像窗口参数

1.  列：子窗口中的图像**列**数2.  行：子窗口中的图像**行**数

#### `plastimatch` DRR 图像处理

[![image](https://user-images.githubusercontent.com/3785912/107617306-b38db180-6c60-11eb-9dd1-b2751b23a314.png)](https://user-images.githubusercontent.com/3785912/107617306-b38db180-6c60-11eb-9dd1-b2751b23a314.png)

1.  使用指数映射：应用输出值的**指数映射**2.  自动缩放：**自动重新缩放**强度3.  **反转**：反转图像强度4.  范围：形式形式的**范围**强度（最小值、最大值）5.  **重建**算法：重构算法的类型（CLI 模块中的暴露算法类型）6.  霍恩斯菲尔德单位转换：计算期间**霍恩斯菲尔德单位转换**的类型7.  **线程**：计算的并行性类型8.  **霍恩斯菲尔德单位**阈值：低于阈值的 HU 值计为 -1000（空气值）

#### `plastimatch` DRR 命令参数（只读）

[![image](https://user-images.githubusercontent.com/3785912/96576928-8c8b2880-12db-11eb-875e-a06df31fd792.png)](https://user-images.githubusercontent.com/3785912/96576928-8c8b2880-12db-11eb-875e-a06df31fd792.png)

[`plastimatch` drr](http://www.plastimatch.org/drr.html) 程序的参数是使用可加载模块参数生成的 用于测试和调试目的。

如何在 3D 切片器中使用 python 计算 DRR 图像？
-------------------------------

必须有 CT 容量（强制性）和 RTSTRUCT 或分割（可选）。

#### 示例1（带分割的CT体积，手动设置光束，等中心是ROI的中心）

```
# Create dummy RTPlan 创建虚拟RT计划
rtImagePlan = slicer.mrmlScene.AddNewNodeByClass( 'vtkMRMLRTPlanNode', 'rtImagePlan')
# Create RTImage dummy beam 创建虚拟RT图像光束
rtImageBeam = slicer.mrmlScene.AddNewNodeByClass( 'vtkMRMLRTBeamNode', 'rtImageBeam')
# Add beam to the plan 将光束添加到计划
rtImagePlan.AddBeam(rtImageBeam)
# Set required beam parameters 设置必需的光束参数
rtImageBeam.SetGantryAngle(90.)
rtImageBeam.SetCouchAngle(12.)
# Get CT volume 创建CT卷
ctVolume = slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLScalarVolumeNode')
# Get Segmentation (RTSTRUCT) 分割图(RTSTRUCT)
ctSegmentation = slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLSegmentationNode')
# Set and observe CT volume by the plan 按计划设置并监测CT卷
rtImagePlan.SetAndObserveReferenceVolumeNode(ctVolume)
# Set and observe Segmentation by the plan 按计划设置并监测分割
rtImagePlan.SetAndObserveSegmentationNode(ctSegmentation)
# Set isocenter position as a center of ROI 设置等中心位置为ROI中心
rtImagePlan.SetIsocenterSpecification(slicer.vtkMRMLRTPlanNode.CenterOfTarget)
# Set required segment ID (for example 'PTV') 设置必需的分割ID(例如'PTV')
rtImagePlan.SetTargetSegmentID('PTV')
rtImagePlan.SetIsocenterToTargetCenter()
# Create DRR image computation node for user imager parameters 由用户成像参数创建DRR图像计算节点
drrParameters = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLDrrImageComputationNode', 'rtImageBeamParams')
# Set and observe RTImage beam by the DRR node 按DRR节点设置并监测RT图像光束
drrParameters.SetAndObserveBeamNode(rtImageBeam)
# Get DRR computation logic 获取DRR计算逻辑
drrLogic = slicer.modules.drrimagecomputation.logic()
# Update imager markups for the 3D view and slice views (optional) 为3D视图和切片视图更新成像器标记(可选)
drrLogic.UpdateMarkupsNodes(drrParameters)
# Update imager normal and view-up vectors (mandatory) 更新成像器法向量和视图向上向量(强制)
drrLogic.UpdateNormalAndVupVectors(drrParameters) # REQUIRED
# Compute DRR image 计算DRR图像
drrLogic.ComputePlastimatchDRR( drrParameters, ctVolume)

```

#### 示例2（带分割的CT体积，光束根据3D相机方向更新，等中心是ROI的中心）

```
# Create dummy plan
rtImagePlan = slicer.mrmlScene.AddNewNodeByClass( 'vtkMRMLRTPlanNode', 'rtImagePlan')
# Create RTImage dummy beam
rtImageBeam = slicer.mrmlScene.AddNewNodeByClass( 'vtkMRMLRTBeamNode', 'rtImageBeam')
# Add beam to the plan
rtImagePlan.AddBeam(rtImageBeam)
# Get CT volume
ctVolume = slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLScalarVolumeNode')
# Set and observe CT volume by the plan
rtImagePlan.SetAndObserveReferenceVolumeNode(ctVolume)
# Get Segmentation (RTSTRUCT)
ctSegmentation = slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLSegmentationNode')
# Set and observe CT volume by the plan
rtImagePlan.SetAndObserveReferenceVolumeNode(ctVolume)
# Set and observe Segmentation by the plan
rtImagePlan.SetAndObserveSegmentationNode(ctSegmentation)
# Set isocenter position as a center of ROI
rtImagePlan.SetIsocenterSpecification(slicer.vtkMRMLRTPlanNode.CenterOfTarget)
# Set name of target segment from segmentation (for example 'PTV')
rtImagePlan.SetTargetSegmentID('PTV')
rtImagePlan.SetIsocenterToTargetCenter()
# Get 3D camera
threeDcamera = slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLCameraNode')
# Create DRR image computation node for user imager parameters
rtImageParameters = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLDrrImageComputationNode', 'rtImageBeamParams')
# Set and observe RTImage beam by the DRR node
rtImageParameters.SetAndObserveBeamNode(rtImageBeam)
# Set and observe camera node by the DRR node
rtImageParameters.SetAndObserveCameraNode(threeDcamera)
# Set required DRR parameters
rtImageParameters.SetHUThresholdBelow(120)
# Get DRR computation logic
drrLogic = slicer.modules.drrimagecomputation.logic()

# Update beam according to the 3D camera orientation (mandatory)
if (drrLogic.UpdateBeamFromCamera(rtImageParameters)): # REQUIRED
  print('Beam orientation updated according to the 3D camera orientation')

# Update imager markups for the 3D view and slice views (optional)
drrLogic.UpdateMarkupsNodes(rtImageParameters)
# Update imager normal and view-up vectors (mandatory)
drrLogic.UpdateNormalAndVupVectors(rtImageParameters) # REQUIRED
# Compute DRR image
drrLogic.ComputePlastimatchDRR( rtImageParameters, ctVolume)

```

#### 示例3（仅CT体积，手动设置光束，手动设置等心）

```
# Create dummy plan
rtImagePlan = slicer.mrmlScene.AddNewNodeByClass( 'vtkMRMLRTPlanNode', 'rtImagePlan')
# Create RTImage dummy beam
rtImageBeam = slicer.mrmlScene.AddNewNodeByClass( 'vtkMRMLRTBeamNode', 'rtImageBeam')
# Set required beam parameters
rtImageBeam.SetGantryAngle(90.)
rtImageBeam.SetCouchAngle(12.)
# Add beam to the plan
rtImagePlan.AddBeam(rtImageBeam)
# Get CT volume
ctVolume = slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLScalarVolumeNode')
# Set and observe CT volume by the plan
rtImagePlan.SetAndObserveReferenceVolumeNode(ctVolume)

# Set required isocenter position as a point
rtImagePlan.SetIsocenterSpecification(slicer.vtkMRMLRTPlanNode.ArbitraryPoint)
isocenterPosition = [ -1., -2., -3. ]
if (rtImagePlan.SetIsocenterPosition(isocenterPosition)):
  print('New isocenter position is set')

# Create DRR image computation node for user imager parameters
rtImageParameters = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLDrrImageComputationNode', 'rtImageBeamParams')
# Set and observe RTImage beam by the DRR node
rtImageParameters.SetAndObserveBeamNode(rtImageBeam)
# Set required DRR parameters
rtImageParameters.SetHUThresholdBelow(50)
# Get DRR computation logic
drrLogic = slicer.modules.drrimagecomputation.logic()
# Update imager markups for the 3D view and slice views (optional)
drrLogic.UpdateMarkupsNodes(rtImageParameters)
# Update imager normal and view-up vectors (mandatory)
drrLogic.UpdateNormalAndVupVectors(rtImageParameters) # REQUIRED
# Compute DRR image
drrLogic.ComputePlastimatchDRR( rtImageParameters, ctVolume)

```

#### 示例4（仅CT体积，光束根据3D相机方向更新，等中心手动设置）

```
# Create dummy plan
rtImagePlan = slicer.mrmlScene.AddNewNodeByClass( 'vtkMRMLRTPlanNode', 'rtImagePlan')
# Create RTImage dummy beam
rtImageBeam = slicer.mrmlScene.AddNewNodeByClass( 'vtkMRMLRTBeamNode', 'rtImageBeam')
# Add beam to the plan
rtImagePlan.AddBeam(rtImageBeam)
# Get CT volume
ctVolume = slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLScalarVolumeNode')
# Set and observe CT volume by the plan
rtImagePlan.SetAndObserveReferenceVolumeNode(ctVolume)

# Set required isocenter position as a point
rtImagePlan.SetIsocenterSpecification(slicer.vtkMRMLRTPlanNode.ArbitraryPoint)
isocenterPosition = [ -1., -2., -3. ]
if (rtImagePlan.SetIsocenterPosition(isocenterPosition)):
  print('New isocenter position is set')

# Get 3D camera
threeDcamera = slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLCameraNode')
# Create DRR image computation node for user imager parameters
rtImageParameters = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLDrrImageComputationNode', 'rtImageBeamParams')
# Set and observe RTImage beam by the DRR node
rtImageParameters.SetAndObserveBeamNode(rtImageBeam)
# Set and observe camera node by the DRR node
rtImageParameters.SetAndObserveCameraNode(threeDcamera)
# Set required DRR parameters
rtImageParameters.SetHUThresholdBelow(-500)
# Get DRR computation logic
drrLogic = slicer.modules.drrimagecomputation.logic()

# Update beam according to the 3D camera orientation (mandatory)
if (drrLogic.UpdateBeamFromCamera(rtImageParameters)): # REQUIRED
  print('Beam orientation updated according to the 3D camera orientation')

# Update imager markups for the 3D view and slice views (optional)
drrLogic.UpdateMarkupsNodes(rtImageParameters)
# Update imager normal and view-up vectors (mandatory)
drrLogic.UpdateNormalAndVupVectors(rtImageParameters) # REQUIRED
# Compute DRR image
drrLogic.ComputePlastimatchDRR( rtImageParameters, ctVolume)

```


参考: <https://discourse.slicer.org/t/optimizing-resolution-and-spacing-parameters-in-plastimatch-drr/25873/63?page=3>

#### 示例5

```
# Get isocenter, get control points
isocenterPoint = slicer.mrmlScene.GetFirstNodeByName('Isocenter')
controlPoints = slicer.mrmlScene.GetFirstNodeByName('ControlPoints')

# Get CT
ctVolume = slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLScalarVolumeNode')

# Create RTPlan
rtImagePlan = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLRTPlanNode', 'rtImagePlan')
rtImagePlan.SetAndObserveReferenceVolumeNode(ctVolume)
rtImagePlan.SetIsocenterSpecification(slicer.vtkMRMLRTPlanNode.ArbitraryPoint)
rtImagePlan.SetAndObservePoisMarkupsFiducialNode(isocenterPoint)

# Create RTBeam, add beam to the plan, setup beam parameters
rtImageBeam = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLRTBeamNode', 'rtImageBeam')
rtImagePlan.AddBeam(rtImageBeam)
rtImageBeam.SetGantryAngle(90.)
rtImageBeam.SetSAD(1000.)

# Create vtkMRMLDrrImageComputationNode node, setup and update Normal and View-Up vectors
# Compute DRR using defined rtImageParameters and ctVolume
rtImageParameters = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLDrrImageComputationNode', 'rtImageBeamParams')
rtImageParameters.SetAndObserveBeamNode(rtImageBeam)
rtImageParameters.SetHUThresholdBelow(120)

drrLogic = slicer.modules.drrimagecomputation.logic()
drrLogic.UpdateMarkupsNodes(rtImageParameters)
drrLogic.UpdateNormalAndVupVectors(rtImageParameters) # REQUIRED
drrLogic.ComputePlastimatchDRR( rtImageParameters, ctVolume)

# Get Plastimatch matrices (for testing and control)
plastimatchIntrinsic = vtk.vtkMatrix4x4()
drrLogic.GetPlastimatchIntrinsicMatrix(rtImageParameters, plastimatchIntrinsic)

plastimatchExtrinsic = vtk.vtkMatrix4x4()
drrLogic.GetPlastimatchExtrinsicMatrix(rtImageParameters, plastimatchExtrinsic)

plastimatchProjection = vtk.vtkMatrix4x4()
drrLogic.GetPlastimatchProjectionMatrix(rtImageParameters, plastimatchProjection)

# Create markups nodes for the results
projPoints = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode', 'ProjectedPoints')
whPoints = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode', 'WidthHeightPoints')
crPoints = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode', 'ColumnRowPoints')

# Get number of original control points
nofPoints = controlPoints.GetNumberOfControlPoints()

# Loop all original control points, calculate projection and offsets
for i in range(nofPoints):
    cpRAS = [0,0,0]
    projPAS = [0,0,0]
    offsetWidthHeight = [0,0]
    offsetColumnRow = [0,0]
    # Get control point coordinate
    controlPoints.GetNthControlPointPositionWorld(i, cpRAS)
    # Calculate projected point offsets
    if drrLogic.GetPointOffsetFromImagerOrigin(rtImageParameters, cpRAS, offsetWidthHeight, offsetColumnRow):
        whPoints.AddControlPoint(offsetWidthHeight[0], offsetWidthHeight[1], 0, controlPoints.GetNthControlPointLabel(i))
        crPoints.AddControlPoint(offsetColumnRow[0], offsetColumnRow[1], 0, controlPoints.GetNthControlPointLabel(i))
    # Calculate projected point world coordinates
    if drrLogic.GetRayIntersectWithImagerPlane(rtImageParameters, cpRAS, projPAS):
        projPoints.AddControlPoint(projPAS, controlPoints.GetNthControlPointLabel(i))

# Save the results into csv files
slicer.modules.markups.logic().ExportControlPointsToCSV(projPoints, "/tmp/ProjectedControlPoints.csv")
slicer.modules.markups.logic().ExportControlPointsToCSV(whPoints, "/tmp/WidthHeightPoints.csv")
slicer.modules.markups.logic().ExportControlPointsToCSV(crPoints, "/tmp/ColumnRowPoints.csv")

```

#### 示例6

```
drrLogic.ShowMarkupsNodes(False)
lines = slicer.mrmlScene.GetNodesByClass('vtkMRMLMarkupsLineNode')
projPoints = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode', 'ProjectedPoints')

for line in lines:
    cpRAS = [0,0,0]
    projPAS = [0,0,0]
    
    if line.GetMarkupsDisplayNode().GetVisibility() == 1:
      line.GetNthControlPointPositionWorld( 0, cpRAS)
      if drrLogic.GetRayIntersectWithImagerPlane(rtImageParameters, cpRAS, projPAS):
          projPoints.AddControlPoint(projPAS, line.GetNthControlPointLabel(0)  + '_Proj')
    
      line.GetNthControlPointPositionWorld( 1, cpRAS)
      if drrLogic.GetRayIntersectWithImagerPlane(rtImageParameters, cpRAS, projPAS):
          projPoints.AddControlPoint(projPAS, line.GetNthControlPointLabel(1)  + '_Proj')
    
    line.UnRegister(None)
```