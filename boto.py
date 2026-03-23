import boto3
import os

s3 = boto3.client('s3', region_name='eu-west-2')
s3.put_object(
    Bucket='mlops-log-anomaly-artifacts',
    Key='test/hello.txt',
    Body=b'hello from python'
)
print('S3 write worked')

