import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Fusion Model Training')

    #Project name:
    
    parser.add_argument('--Project_name', type=str, default='Fusion Model - Tactile finger', help='Name the project')
    
    # Model architecture arguments
    parser.add_argument('--input_size_fc', type=int, default=3, help='Fully connected input size')
    parser.add_argument('--hidden_size1_fc', type=int, default=64, help='first hidden layer size')
    parser.add_argument('--hidden_size2_fc', type=int, default=32, help='second hidden layer size')
    parser.add_argument('--hidden_size3_fc', type=int, default=16, help='third hidden layer size')
    parser.add_argument('--output_size_fc', type=int, default=8, help='output size of fully connected model')
    parser.add_argument('--output_size_image', type=int, default=8, help='output size of ResNet18 model')

    
    # Training hyperparameters
    parser.add_argument('--lr',dest='lr', type=float, default=0.0017, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--decay_epochs', type=int, default=75, help='Number of epoch to start linear lr decay')
    parser.add_argument('--model_type', type=str, default='ResNet18', help='EfficientNetV2 Large / EfficientNetV2 small')

    # Training Options and Paths 
    parser.add_argument('--train_real_JSON', type=str, default=r'C:\Users\yarde\Documents\GitHub\Hneg_SRC\Pre-Process_MM\Json\train_real_yarden.json', help='Name of real JSON file for training')
    parser.add_argument('--train_sim_JSON', type=str, default=r'C:\Users\yarde\Documents\GitHub\Hneg_SRC\Pre-Process_MM\Json\train_sim_yarden.json', help='Name of simullation JSON file for training')
    parser.add_argument('--train_SRC_JSON', type=str, default=r'C:\Users\yarde\Documents\GitHub\Hneg_SRC\Pre-Process_MM\Json\train_gen_paired_yarden.json', help='Name of generated JSON file for training')
    parser.add_argument('--Model_name', type=str, default='Resnet18', help='name of current model')
    parser.add_argument('--device', type=int, default=0, help='Device to use(GPU\CPU) GPU id: 0, 1,2,... -1 for CPU')
    parser.add_argument('--log_interval', type=int, default=1, help='Every epoch to print progress to terminal')
    parser.add_argument('--save_iter', type=int, default=10, help='Save the model every epoch')

    #Test options
    parser.add_argument('--test_real_JSON', type=str, default=r'C:\Users\yarde\Documents\GitHub\Hneg_SRC\Pre-Process_MM\Json\test_real_yarden.json', help='Name of real JSON file for testing')
    parser.add_argument('--test_sim_JSON', type=str, default=r'C:\Users\yarde\Documents\GitHub\Hneg_SRC\Pre-Process_MM\Json\test_sim_yarden.json', help='Name of sim JSON file for testing')
    parser.add_argument('--test_data', type=str, default='real', help='real for model evaluation, gen for testing with regresor')
    parser.add_argument('--test_gen_JSON', type=str, default=r'C:\Users\yarde\Documents\GitHub\Hneg_SRC\Compression\datasets\paired_finger\json_data\test_generated_950.json', help='Name of real JSON file for testing')



    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    print("Arguments:")
    print(args)
    
    # Now you can access the arguments using args.attribute_name

if __name__ == '__main__':
    main()
