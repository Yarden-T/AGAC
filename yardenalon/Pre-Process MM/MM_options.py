import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Fusion Model Training')
    
    # Model architecture arguments
    parser.add_argument('--num_numerical_features', type=int, default=7, help='Number of numerical features')
    parser.add_argument('--hidden_units_image', type=int, nargs='+', default=[4096, 1024], help='Hidden units for image processing')
    parser.add_argument('--hidden_units_numerical', type=int, nargs='+', default=[128, 64], help='Hidden units for numerical processing')
    parser.add_argument('--fusion_hidden_units', type=int, nargs='+', default=[512, 128], help='Hidden units for fusion network')
    
    # Training hyperparameters
    parser.add_argument('--lr',dest='lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--model_type', type=str, default='VGG16', help='EfficientNetV2 Large / EfficientNetV2 small')

    # Training Options and Paths 
    parser.add_argument('--base_path', type=str, default=r'C:\Users\yarde\Documents\GitHub\Hneg_SRC\Compression\datasets\Finger_new\json_data', help='Path to base directory containing JSON file')
    parser.add_argument('--real_JSON', type=str, default='real_train_4_transformed.json', help='Name of real JSON file')
    parser.add_argument('--sim_JSON', type=str, default='sim_train_4_transformed.json', help='Name of simullation JSON file')
    parser.add_argument('--Save_path', type=str, default='sim_train_4_transformed.json', help='Path to save the model to')
    parser.add_argument('--Model_name', type=str, default='Exp', help='name of current model')
    parser.add_argument('--device', type=int, default=0, help='Device to use(GPU\CPU) GPU id: 0, 1,2,... -1 for CPU')
    parser.add_argument('--log_interval', type=int, default=1, help='Every epoch to print progress to terminal')
    parser.add_argument('--save_iter', type=int, default=1, help='Save the model every epoch')


    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    print("Arguments:")
    print(args)
    
    # Now you can access the arguments using args.attribute_name

if __name__ == '__main__':
    main()
