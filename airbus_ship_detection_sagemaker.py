from sagemaker.tensorflow import TensorFlow
from sagemaker.inputs import FileSystemInput
import os

hvd_instance_type = 'ml.p3.16xlarge'
hvd_processes_per_host = 8
hvd_instance_count = 1

distributions = {'mpi': {
                    'enabled': True,
                    'processes_per_host': hvd_processes_per_host,
                    'custom_mpi_options': '-verbose --NCCL_DEBUG=INFO -x OMPI_MCA_btl_vader_single_copy_mechanism=none'
                        }
                }

metric_definitions=[
                   {'Name': 'train:loss', 'Regex': 'train_loss=(.*?),'},
                   {'Name': 'train:dice_coef', 'Regex': 'train_dice_coef=(.*?),'},
                   {'Name': 'train:binary_accuracy', 'Regex': 'train_binary_accuracy=(.*?),'},
                   {'Name': 'train:true_positive_rate', 'Regex': 'train_true_positive_rate=(.*?),'},
                   {'Name': 'val:loss', 'Regex': 'val_loss=(.*?),'},
                   {'Name': 'val:dice_coef', 'Regex': 'val_dice_coef=(.*?),'},
                   {'Name': 'val:binary_accuracy', 'Regex': 'val_bin_accuracy=(.*?),'},
                   {'Name': 'val:true_positive_rate', 'Regex': 'val_true_positive_rate=(.*?)'}
                ]

hyperparameters = {'epochs': 2, 'batch-size' : 8}
role = "arn:aws:iam::869530972998:role/SagemakerAdmin"
bucket = "weteh-data-repo-us-east-2"
output_path = os.path.join('s3://', bucket, "models")

estimator_hvd = TensorFlow(
                       source_dir='.',
                       entry_point='launcher.sh',
                       base_job_name='airbus-unet34-keras-3-distributed',
                       role=role,
                       framework_version='2.1.0',
                       py_version='py3',
                       hyperparameters=hyperparameters,
                       train_instance_count=hvd_instance_count,
                       train_instance_type=hvd_instance_type,
                       distributions=distributions,
                       subnets=["subnet-0951bd7928432fe42"],
                       security_group_ids=["sg-087f69d528dfc0196"],
                       metric_definitions=metric_definitions,
                       output_path=output_path
)

file_system_input = FileSystemInput(file_system_id='fs-0a4699a0747f9b546',
                                    file_system_type='FSxLustre',
                                    directory_path='/3rjrpbmv/airbus_data/deep_learning_v2',
                                    file_system_access_mode='ro')

# Start an Amazon SageMaker training job with Fsx for Lustre using the FileSystemInput class
estimator_hvd.fit(file_system_input)
print("done")