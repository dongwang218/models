#Introduction

## generate the records
```
cd /home/ubuntu/workspace/tensorflow-models/inception
python handwriting/build_gnt_data.py --validation_directory /home/ubuntu/workspace/data/validation --train_directory /home/ubuntu/workspace/data/train --output_directory /home/ubuntu/workspace/data/records --num_threads 8

It took more than a day
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
2016-08-28 23:28:09.863358: Finished writing all 239 images in data set.
```

### spot check a file and output the labels
python handwriting/test_read_proto.py /home/ubuntu/workspace/data/records/label_output.pkl `ls ~/workspace/data/records/validation*`

## Training a model
should use training data instead
```
python handwriting/inception_train.py --train_dir /home/ubuntu/workspace/data/workdir --data_dir /home/ubuntu/workspace/data/records --subset validation

2016-08-28 05:43:32.381453: step 0, loss = 13.53 (1.6 examples/sec; 20.497 sec/batch)
2016-08-28 19:25:27.266072: step 18050, loss = 3.91 (12.3 examples/sec; 2.603 sec/batch)
```
### evaluation
```
cp records/train-00000-of-00239 sample_test/test-00000-of-00239
python handwriting/inception_eval.py --eval_dir /home/ubuntu/workspace/data/evalworkdir --checkpoint_dir /home/ubuntu/workspace/data/workdir --num_examples 5000 --data_dir /home/ubuntu/workspace/data/sample_test --subset test

2016-08-28 19:51:33.128669: precision @ 1 = 0.9236 recall @ 5 = 0.9896 [5024 examples]
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
