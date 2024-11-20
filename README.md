# Tripo_Cap3D_ToolBox
Caption, Attribute, Tags, toolbox 


## Quality
Usage
```python   
cd quality
python prepare_union.py --input uuids.txt --input_features features.bin --output res.txt
```
Format of res.txt
```
uuid  geo_v1 tex_v1 geo_v2 geo_v3
6087849a-1c51-4df2-8cf3-29188c6eaaef 0.7382056150014855 1.130776306295785 1.8398140668869019 1.9899262189865112
ae776419-a703-46e9-aca5-98e8bc9cd68a 1.3860469875570975 1.3193187405622835 2.4530045986175537 2.2435150146484375
f9188d80-56dd-4d42-94d5-956fa6630acf 0.970039920099245 0.704023767054555 2.215405225753784 2.1199488639831543
4ce7e0d5-9c1b-4cd2-b6ff-518024e7c915 0.20988135131734964 -0.029122512622030916 1.725032925605774 1.4892148971557617
b64443a6-1b7b-4a86-877b-8abe6f98ec0b 0.6854109183417603 0.4970730957011553 2.016000747680664 1.7844688892364502
19f74ef4-8753-4a53-85bd-9353bcf66136 -0.5658391394291664 0.16840021876148192 -0.1347360610961914 -0.9907581806182861

```
description of columns:
- v1:  input rgb x 4, elevation angle: 45; azimuth angle: 10.  geometry and texture 
- v2: input normal x 4, elevation angle: 45; azimuth angle: 10. geometry score only
- v3: input normal x 4, elevation angle: 45; azimuth angle: 10. geometry score only (rectified by another annotations)
