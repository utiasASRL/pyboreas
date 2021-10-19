Boreas Examples
==============================================


These example use AWS CloudFormation to create an Amazon SageMaker Jupyter Notebook instance with the appropriate Python libraries installed for working with the Boreas dataset.

[![cloudformation-launch-stack](cloudformation/cloudformation-launch-stack.png)](https://console.aws.amazon.com/cloudformation/home?region=us-west-2#/stacks/new?stackName=BOREAS&templateURL=https://github.com/utiasASRL/pyboreas/tree/master/pyboreas/tutorials/aws/cloudformation/aws.yaml)

1. upload `aws.yaml` instead of using the URL version.
2. Open the jupyter notebook instance.
	a. Go to `CloudFormation` > `Stacks` > `BOREAS`
	b. Go to `Outputs` and click on the `Value`, it should look something like:

`https://us-west-2.console.aws.amazon.com/sagemaker/home?region=us-west-2#/notebook-instances/openNotebook/INTRO-0a449d0be91d?view=lab`

This will open a new tab with the sagemaker notebook instance running.

3. Open the terminal within the Sagemaker and run the following commands:

```
bash setup.sh
conda init bash
source ~/.bashrc
conda activate boreaspy3
pip install -e /home/ec2-user/SageMaker/pyboreas
```

Then a jupyter notebook with the conda_boreaspy3 environment


`intro.ipynb` visualizes the sensor types contained in this dataset
`lidar_camera_projection.ipynb` demonstrates how lidar and camera data can be fused together

