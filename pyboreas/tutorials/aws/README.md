Boreas Examples
==============================================

These examples use AWS CloudFormation to create an Amazon SageMaker Jupyter Notebook instance with the appropriate Python libraries installed for working with the Boreas dataset.

1. From AWS Management Console go to `CloudFormation` > `Stacks` > `Create Stack with New Resources`
2. upload `aws.yaml` for the template
3. Name the Stack `BOREAS`
4. Keep clicking through until the stack creation starts, this will take a few minutes to complete.
5. Open the jupyter notebook instance.
	a. Go to `CloudFormation` > `Stacks` > `BOREAS`
	b. Go to `Outputs` and click on the `Value`, it should look something like:

`https://us-west-2.console.aws.amazon.com/sagemaker/home?region=us-west-2#/notebook-instances/openNotebook/INTRO-0a449d0be91d?view=lab`

This will open a new tab with the sagemaker notebook instance running.

6. Open the terminal within the Sagemaker and run the following commands:

```
bash setup.sh
conda init bash
source ~/.bashrc
conda activate boreaspy3
pip install -e /home/ec2-user/SageMaker/pyboreas
```

Then a open jupyter notebook with the `conda_boreaspy3` environment.

Example notebooks:

`intro.ipynb` visualizes the sensor types contained in this dataset

`lidar_camera_projection.ipynb` demonstrates how lidar and camera data can be fused together
