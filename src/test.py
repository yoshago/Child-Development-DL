from src.CNN_Trainer import CNN_Trainer

BATCH_SIZE=100
NUM_CLASSES=4

def main():
    cnn1=CNN_Trainer('../Data/compressed_data/', NUM_CLASSES)
    cnn1.initial_model()
    cnn1.load_dictionary('data_key_new.txt')
    cnn1.train(BATCH_SIZE)
    cnn1.save_model('model1')

main()