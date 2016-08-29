#Introduction

## Download data
http://www.nlpr.ia.ac.cn/databases/handwriting/Download.html
* HWDB1.1trn_gnt (1873MB)
* HWDB1.1tst_gnt (471MB)
Note that one part of the first file is corrupted, so trn has 239 files, tst has 60 files.
```
ls /home/ubuntu/workspace/data/validation/
1241-c.gnt  1246-c.gnt  1251-c.gnt  1256-c.gnt  1261-c.gnt  1266-c.gnt  1271-c.gnt  1276-c.gnt  1281-c.gnt  1286-c.gnt  1291-c.gnt  1296-c.gnt
1242-c.gnt  1247-c.gnt  1252-c.gnt  1257-c.gnt  1262-c.gnt  1267-c.gnt  1272-c.gnt  1277-c.gnt  1282-c.gnt  1287-c.gnt  1292-c.gnt  1297-c.gnt
1243-c.gnt  1248-c.gnt  1253-c.gnt  1258-c.gnt  1263-c.gnt  1268-c.gnt  1273-c.gnt  1278-c.gnt  1283-c.gnt  1288-c.gnt  1293-c.gnt  1298-c.gnt
1244-c.gnt  1249-c.gnt  1254-c.gnt  1259-c.gnt  1264-c.gnt  1269-c.gnt  1274-c.gnt  1279-c.gnt  1284-c.gnt  1289-c.gnt  1294-c.gnt  1299-c.gnt
1245-c.gnt  1250-c.gnt  1255-c.gnt  1260-c.gnt  1265-c.gnt  1270-c.gnt  1275-c.gnt  1280-c.gnt  1285-c.gnt  1290-c.gnt  1295-c.gnt  1300-c.gnt
ubuntu@b418e3f89a08:~/workspace/tensorflow-models/inception/handwriting$ ls /home/ubuntu/workspace/data/train/
1001-c.gnt  1019-c.gnt  1037-c.gnt  1055-c.gnt  1073-c.gnt  1091-c.gnt  1109-c.gnt  1128-c.gnt  1146-c.gnt  1164-c.gnt  1182-c.gnt  1200-c.gnt  1218-c.gnt  1236-c.gnt
1002-c.gnt  1020-c.gnt  1038-c.gnt  1056-c.gnt  1074-c.gnt  1092-c.gnt  1110-c.gnt  1129-c.gnt  1147-c.gnt  1165-c.gnt  1183-c.gnt  1201-c.gnt  1219-c.gnt  1237-c.gnt
1003-c.gnt  1021-c.gnt  1039-c.gnt  1057-c.gnt  1075-c.gnt  1093-c.gnt  1111-c.gnt  1130-c.gnt  1148-c.gnt  1166-c.gnt  1184-c.gnt  1202-c.gnt  1220-c.gnt  1238-c.gnt
1004-c.gnt  1022-c.gnt  1040-c.gnt  1058-c.gnt  1076-c.gnt  1094-c.gnt  1112-c.gnt  1131-c.gnt  1149-c.gnt  1167-c.gnt  1185-c.gnt  1203-c.gnt  1221-c.gnt  1239-c.gnt
1005-c.gnt  1023-c.gnt  1041-c.gnt  1059-c.gnt  1077-c.gnt  1095-c.gnt  1113-c.gnt  1132-c.gnt  1150-c.gnt  1168-c.gnt  1186-c.gnt  1204-c.gnt  1222-c.gnt  1240-c.gnt
1006-c.gnt  1024-c.gnt  1042-c.gnt  1060-c.gnt  1078-c.gnt  1096-c.gnt  1114-c.gnt  1133-c.gnt  1151-c.gnt  1169-c.gnt  1187-c.gnt  1205-c.gnt  1223-c.gnt
1007-c.gnt  1025-c.gnt  1043-c.gnt  1061-c.gnt  1079-c.gnt  1097-c.gnt  1116-c.gnt  1134-c.gnt  1152-c.gnt  1170-c.gnt  1188-c.gnt  1206-c.gnt  1224-c.gnt
1008-c.gnt  1026-c.gnt  1044-c.gnt  1062-c.gnt  1080-c.gnt  1098-c.gnt  1117-c.gnt  1135-c.gnt  1153-c.gnt  1171-c.gnt  1189-c.gnt  1207-c.gnt  1225-c.gnt
1009-c.gnt  1027-c.gnt  1045-c.gnt  1063-c.gnt  1081-c.gnt  1099-c.gnt  1118-c.gnt  1136-c.gnt  1154-c.gnt  1172-c.gnt  1190-c.gnt  1208-c.gnt  1226-c.gnt
1010-c.gnt  1028-c.gnt  1046-c.gnt  1064-c.gnt  1082-c.gnt  1100-c.gnt  1119-c.gnt  1137-c.gnt  1155-c.gnt  1173-c.gnt  1191-c.gnt  1209-c.gnt  1227-c.gnt
1011-c.gnt  1029-c.gnt  1047-c.gnt  1065-c.gnt  1083-c.gnt  1101-c.gnt  1120-c.gnt  1138-c.gnt  1156-c.gnt  1174-c.gnt  1192-c.gnt  1210-c.gnt  1228-c.gnt
1012-c.gnt  1030-c.gnt  1048-c.gnt  1066-c.gnt  1084-c.gnt  1102-c.gnt  1121-c.gnt  1139-c.gnt  1157-c.gnt  1175-c.gnt  1193-c.gnt  1211-c.gnt  1229-c.gnt
1013-c.gnt  1031-c.gnt  1049-c.gnt  1067-c.gnt  1085-c.gnt  1103-c.gnt  1122-c.gnt  1140-c.gnt  1158-c.gnt  1176-c.gnt  1194-c.gnt  1212-c.gnt  1230-c.gnt
1014-c.gnt  1032-c.gnt  1050-c.gnt  1068-c.gnt  1086-c.gnt  1104-c.gnt  1123-c.gnt  1141-c.gnt  1159-c.gnt  1177-c.gnt  1195-c.gnt  1213-c.gnt  1231-c.gnt
1015-c.gnt  1033-c.gnt  1051-c.gnt  1069-c.gnt  1087-c.gnt  1105-c.gnt  1124-c.gnt  1142-c.gnt  1160-c.gnt  1178-c.gnt  1196-c.gnt  1214-c.gnt  1232-c.gnt
1016-c.gnt  1034-c.gnt  1052-c.gnt  1070-c.gnt  1088-c.gnt  1106-c.gnt  1125-c.gnt  1143-c.gnt  1161-c.gnt  1179-c.gnt  1197-c.gnt  1215-c.gnt  1233-c.gnt
1017-c.gnt  1035-c.gnt  1053-c.gnt  1071-c.gnt  1089-c.gnt  1107-c.gnt  1126-c.gnt  1144-c.gnt  1162-c.gnt  1180-c.gnt  1198-c.gnt  1216-c.gnt  1234-c.gnt
1018-c.gnt  1036-c.gnt  1054-c.gnt  1072-c.gnt  1090-c.gnt  1108-c.gnt  1127-c.gnt  1145-c.gnt  1163-c.gnt  1181-c.gnt  1199-c.gnt  1217-c.gnt  1235-c.gnt
```
## generate the records
```
cd /home/ubuntu/workspace/tensorflow-models/inception
python handwriting/build_gnt_data.py --validation_directory /home/ubuntu/workspace/data/validation --train_directory /home/ubuntu/workspace/data/train --output_directory /home/ubuntu/workspace/data/records --num_threads 4
```

It took more than a day
```
# validation
2016-08-28 05:30:18.342434 [thread 0]: Wrote 55855 images to 15 shards.
2016-08-28 05:21:44.484108 [thread 1]: Wrote 56050 images to 15 shards.
2016-08-28 05:13:04.433880 [thread 2]: Wrote 56000 images to 15 shards.
2016-08-28 05:29:25.813527 [thread 3]: Wrote 56086 images to 15 shards.
# train
2016-08-28 23:28:09.498198 [thread 1]: Wrote 224478 images to 60 shards.
2016-08-28 23:27:00.462883 [thread 2]: Wrote 224440 images to 60 shards.
2016-08-28 23:23:14.546340 [thread 3]: Wrote 224456 images to 60 shards.
2016-08-28 23:19:03.951833 [thread 0]: Wrote 220679 images to 59 shards.
```

### spot check and output the labels
python handwriting/test_read_proto.py /home/ubuntu/workspace/data/records/label_output.pkl `ls ~/workspace/data/records/validation*`

## Training a model
should use training data instead
```
python handwriting/inception_train.py --train_dir /home/ubuntu/workspace/data/workdir --data_dir /home/ubuntu/workspace/data/records --subset train

2016-08-28 05:43:32.381453: step 0, loss = 13.53 (1.6 examples/sec; 20.497 sec/batch)
2016-08-28 19:25:27.266072: step 18050, loss = 3.91 (12.3 examples/sec; 2.603 sec/batch)
2016-09-04 17:24:41.463257: step 185910, loss = 2.70 (12.4 examples/sec; 2.585 sec/batch)
```
### evaluation
```
cp records/validation-00000-of-00060 sample_test/test-00000-of-00060
python handwriting/inception_eval.py --eval_dir /home/ubuntu/workspace/data/evalworkdir --checkpoint_dir /home/ubuntu/workspace/data/workdir --num_examples 5000 --data_dir /home/ubuntu/workspace/data/sample_test --subset test

2016-08-28 19:51:33.128669: precision @ 1 = 0.9236 recall @ 5 = 0.9896 [5024 examples]
```
### continue training
```
python handwriting/inception_train.py --train_dir /home/ubuntu/workspace/data/trainworkdir --data_dir /home/ubuntu/workspace/data/records_fix --pretrained_model_checkpoint_path /home/ubuntu/workspace/data/workdir/model.ckpt-18000
```
## Export the model
export to serving bundle
```
python handwriting/inception_export.py --checkpoint_dir /home/ubuntu/workspace/data/workdir --export_dir /home/ubuntu/workspace/data/graph_output --label_file /home/ubuntu/workspace/data/records/label_output.pkl
```
Export to pb
```
python handwriting/inception_freeze.py --checkpoint_dir /home/ubuntu/workspace/data/workdir --export_dir /home/ubuntu/workspace/data/graph_output
```

## Hacks
Python thread lock seems to fail, so that two characters are assigned the same lable!. We use the following script to modify the record file to fix the issue
```
python handwriting/fix_ri.py  `ls ~/workspace/data/records/train*`
python handwriting/fix_ri.py  `ls ~/workspace/data/records/validation*`
```